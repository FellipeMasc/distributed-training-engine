import os
import glob
import re

import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors import safe_open
import huggingface_hub
from core.process_groups_config import CommGroupsConfig
from core.distributed.data.comm import sync_grads
from core.distributed.tensor.layers import ColumnParallelLinear
from core.distributed.tensor.loss import vocab_parallel_cross_entropy


def download_model(model_name, hf_token):
    dst = os.path.join("hf_model_safetensors")
    os.makedirs(dst, exist_ok=True)
    if os.path.exists(os.path.join(dst, "config.json")) and os.path.exists(
        os.path.join(dst, "model.safetensors")
    ):
        print(f"Model {model_name} already exists at {dst}")
        return
    print("Downloading SafeTensors files...")
    huggingface_hub.snapshot_download(
        model_name,
        repo_type="model",
        local_dir="hf_model_safetensors",
        token=hf_token,
        allow_patterns=["*.safetensors", "*.json"],
    )
    if not glob.glob("hf_model_safetensors/*.safetensors"):
        raise ValueError(f"Model {model_name} does not have SafeTensors files.")
    print("SafeTensors files downloaded successfully!")


def distribute_layers_pp_stages(num_layers, pp_rank, pp_world_size):
    layers = []
    chunk = num_layers // pp_world_size
    mod = num_layers % pp_world_size
    consumed_layers = 0
    for rank in range(pp_world_size):
        local_chunk = chunk + 1 if rank < mod else chunk
        layers.append(list(range(consumed_layers, consumed_layers + local_chunk)))
        consumed_layers += local_chunk
    return layers[pp_rank]


# How each HF Llama parameter is partitioned across tensor-parallel ranks.
#   column    : slice dim 0 (output features / vocabulary)
#   row       : slice dim 1 (input features)
#   replicate : full tensor on every rank
_TP_PLAN = {
    "embed_tokens.weight": "column",
    "self_attn.q_proj.weight": "column",
    "self_attn.k_proj.weight": "column",
    "self_attn.v_proj.weight": "column",
    "self_attn.q_proj.bias": "column",
    "self_attn.k_proj.bias": "column",
    "self_attn.v_proj.bias": "column",
    "self_attn.o_proj.weight": "row",
    "self_attn.o_proj.bias": "replicate",
    "mlp.gate_proj.weight": "column",
    "mlp.up_proj.weight": "column",
    "mlp.gate_proj.bias": "column",
    "mlp.up_proj.bias": "column",
    "mlp.down_proj.weight": "row",
    "mlp.down_proj.bias": "replicate",
    "lm_head.weight": "column",
    "input_layernorm.weight": "replicate",
    "post_attention_layernorm.weight": "replicate",
    "norm.weight": "replicate",
}

# Separate HF projections that the tensor-parallel blocks fuse into one
# parameter. Order matters: it is the concatenation order along dim 0.
_FUSED_PARAMS = (
    (re.compile(r"^(?P<prefix>.*\.self_attn\.)(?P<part>q|k|v)_proj\.(?P<kind>weight|bias)$"),
     "qkv_proj", ("q", "k", "v")),
    (re.compile(r"^(?P<prefix>.*\.mlp\.)(?P<part>gate|up)_proj\.(?P<kind>weight|bias)$"),
     "gate_up_proj", ("gate", "up")),
)


def _shard_for_tp(tensor_slice, tp_type: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Read this rank's shard of a safetensors slice without loading the rest."""
    if tp_size == 1 or tp_type == "replicate":
        return tensor_slice[:]
    shape = tensor_slice.get_shape()
    dim = 0 if tp_type == "column" else 1
    if shape[dim] % tp_size != 0:
        raise ValueError(
            f"Cannot shard dim {dim} of size {shape[dim]} across tp_size={tp_size}."
        )
    per_rank = shape[dim] // tp_size
    start, end = tp_rank * per_rank, (tp_rank + 1) * per_rank
    if dim == 0:
        return tensor_slice[start:end]
    return tensor_slice[:, start:end]


def _fuse_tensor_parallel_params(state_dict: dict[str, torch.Tensor]) -> None:
    """Concatenate q/k/v (gate/up) shards into the fused TP parameters, in place.

    Each part has already been sliced for this rank, so concatenating along
    dim 0 gives `[q_local; k_local; v_local]`, matching `TensorParallelAttention`.
    """
    pending: dict[str, tuple[tuple[str, ...], dict[str, torch.Tensor]]] = {}
    for key in list(state_dict):
        for pattern, fused_name, order in _FUSED_PARAMS:
            match = pattern.match(key)
            if match is None:
                continue
            fused_key = f"{match['prefix']}{fused_name}.{match['kind']}"
            parts = pending.setdefault(fused_key, (order, {}))[1]
            parts[match["part"]] = state_dict.pop(key)
            break
    for fused_key, (order, parts) in pending.items():
        missing = [p for p in order if p not in parts]
        if missing:
            raise KeyError(f"Missing {missing} while fusing {fused_key}")
        state_dict[fused_key] = torch.cat([parts[p] for p in order], dim=0)


def load_safetensor_weights(
    model: nn.Module,
    safetensors_dir: str,
    global_layer_indices: list[int],
    is_first_stage: bool,
    is_last_stage: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Load safetensor weights into a (possibly PP/TP-partitioned) Llama model.

    Maps safetensor keys (HF Llama naming) to the module's state dict:
      - embed_tokens        → first stage only
      - layers.{local}.*    → model.layers.{global}.*
      - norm                → last stage only
      - lm_head             → last stage only (tied to embed_tokens when absent)

    With tensor parallelism each tensor is sliced for this rank according to
    `_TP_PLAN`, and q/k/v (gate/up) are concatenated into the fused parameters
    used by `TensorParallelAttention` / `TensorParallelMLP`.
    """
    safetensors_files = glob.glob(os.path.join(safetensors_dir, "*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {safetensors_dir}")

    global_to_local = {g: l for l, g in enumerate(global_layer_indices)}
    commConfig = CommGroupsConfig()
    tp_rank, tp_size = commConfig.tp_rank, commConfig.tp_degree

    def _build_key_map() -> dict[str, tuple[str, str, str]]:
        """Build safetensor_key → (module_key, file, tp_type) mapping."""
        key_map: dict[str, tuple[str, str, str]] = {}
        for sf_path in safetensors_files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for sf_key in f.keys():
                    pp_key, tp_type = _safetensor_to_tensor_pipeline_key(
                        sf_key, global_to_local, is_first_stage, is_last_stage
                    )
                    if pp_key is not None and tp_type is not None:
                        key_map[sf_key] = (pp_key, sf_path, tp_type)
        return key_map

    key_map = _build_key_map()

    state_dict: dict[str, torch.Tensor] = {}
    for sf_key, (pp_key, sf_path, tp_type) in key_map.items():
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            shard = _shard_for_tp(f.get_slice(sf_key), tp_type, tp_rank, tp_size)
            state_dict[pp_key] = shard.to(dtype=dtype, device=device)

    if is_last_stage and "lm_head.weight" not in state_dict:
        # Tied embeddings: lm_head shares the embedding table. Both are
        # sharded along the vocabulary (dim 0) under tensor parallelism.
        for sf_path in safetensors_files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                if "model.embed_tokens.weight" in f.keys():
                    shard = _shard_for_tp(
                        f.get_slice("model.embed_tokens.weight"),
                        _TP_PLAN["lm_head.weight"],
                        tp_rank,
                        tp_size,
                    )
                    state_dict["lm_head.weight"] = shard.to(dtype=dtype, device=device)
                    break

    if tp_size > 1:
        _fuse_tensor_parallel_params(state_dict)

    model.load_state_dict(state_dict, strict=True, assign=True)


def _safetensor_to_tensor_pipeline_key(
    sf_key: str,
    global_to_local: dict[int, int],
    is_first_stage: bool,
    is_last_stage: bool,
) -> tuple[str | None, str | None]:
    """Convert a safetensor key to the corresponding module key and TP plan.

    Returns (None, None) if this key doesn't belong to the current stage.
    """
    if sf_key.startswith("model.embed_tokens."):
        if not is_first_stage:
            return None, None
        return (
            sf_key.replace("model.embed_tokens.", "embed_tokens.", 1),
            _TP_PLAN["embed_tokens.weight"],
        )

    if sf_key.startswith("model.layers."):
        rest = sf_key[len("model.layers.") :]
        dot_pos = rest.index(".")
        global_idx = int(rest[:dot_pos])
        if global_idx not in global_to_local:
            return None, None
        local_idx = global_to_local[global_idx]
        suffix = rest[dot_pos + 1 :]
        return f"layers.{local_idx}.{suffix}", _TP_PLAN[suffix]

    if sf_key.startswith("model.norm."):
        if not is_last_stage:
            return None, None
        return sf_key.replace("model.norm.", "norm.", 1), _TP_PLAN["norm.weight"]

    if sf_key.startswith("lm_head."):
        if not is_last_stage:
            return None, None
        return sf_key, _TP_PLAN[sf_key]

    return None, None


def causal_lm_loss(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Project hidden states through `lm_head` and compute the mean token loss.

    `labels` are already shifted (the dataset emits `labels[t] = tokens[t+1]`).

    When `lm_head` is a `ColumnParallelLinear` that keeps its output sharded,
    each tensor-parallel rank only has logits for its vocabulary slice, so the
    loss is computed with the vocab-parallel cross-entropy instead of gathering
    the full `[B, S, vocab_size]` logits on every rank.
    """
    from transformers.loss.loss_utils import ForCausalLMLoss

    logits = lm_head(hidden_states)
    if isinstance(lm_head, ColumnParallelLinear) and not lm_head.gather_output and lm_head.tp_size > 1:
        return vocab_parallel_cross_entropy(
            logits,
            labels,
            lm_head.partition_start,
            lm_head.partition_end,
            lm_head.tp_group,
        )
    return ForCausalLMLoss(logits, labels, vocab_size, shift_labels=labels)


def training_step(
    model: nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model_config,
    dp_group: dist.ProcessGroup | None = None,
) -> float:
    """One optimizer step for standard (non-pipeline) training.

    Runs forward, computes loss, backward, averages gradients across the
    data-parallel group, and steps the optimizer.

    Returns:
        Loss value as a float.
    """
    if dp_group is None:
        dp_group = CommGroupsConfig().dp_group

    optimizer.zero_grad(set_to_none=True)

    outputs = model(input_ids=tokens)
    loss = causal_lm_loss(
        model.lm_head, outputs.last_hidden_state, labels, model_config.vocab_size
    )

    loss.backward()
    sync_grads(model, dp_group)
    optimizer.step()

    return loss.item()
