"""Check that DP / TP / PP training steps match a single-process reference.

Builds a tiny random Llama checkpoint in HF safetensors layout, then runs one
training step through the same code path as `training/ddp_only.py`
(`load_safetensor_weights` + `training_step` / `training_step_1f1b`) and
compares against a saved single-process reference:

  * the batch-mean loss,
  * gradients of every parameter whose shape matches the reference (all of
    them for DP and PP; the replicated norm weights for TP),
  * the loss after one SGD step on the same batch (catches wrong gradients on
    sharded parameters too).

Usage (from the repo root, CPU/gloo):

    python scripts/tests/parallel_equivalence.py --mode make-ckpt
    torchrun --nproc_per_node 1 scripts/tests/parallel_equivalence.py --mode reference
    torchrun --nproc_per_node 2 scripts/tests/parallel_equivalence.py --mode check --tp 2
    torchrun --nproc_per_node 2 scripts/tests/parallel_equivalence.py --mode check --pp 2
    torchrun --nproc_per_node 2 scripts/tests/parallel_equivalence.py --mode check  # dp=2
    torchrun --nproc_per_node 4 scripts/tests/parallel_equivalence.py --mode check --tp 2 --pp 2
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import sys

import torch
import torch.distributed as dist

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accelerate import init_empty_weights
from safetensors.torch import save_file
from transformers.models.llama.configuration_llama import LlamaConfig

WORK_DIR = pathlib.Path(
    os.environ.get("PARALLEL_EQ_DIR", PROJECT_ROOT / "scripts" / "tests" / "_parallel_eq")
)
CKPT_DIR = WORK_DIR / "ckpt"
REF_PATH = WORK_DIR / "reference.pt"

BATCH = 4
SEQ = 16
LR = 0.1
ATOL = 1e-4
RTOL = 1e-4


def tiny_config() -> LlamaConfig:
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
    )
    cfg._attn_implementation = "eager"
    return cfg


def make_batch(device):
    g = torch.Generator().manual_seed(1234)
    tokens = torch.randint(0, 256, (BATCH, SEQ), generator=g)
    labels = torch.full_like(tokens, -100)
    labels[:, :-1] = tokens[:, 1:]
    return tokens.to(device), labels.to(device)


def make_ckpt() -> None:
    from core.models import LlamaModel

    torch.manual_seed(0)
    cfg = tiny_config()
    model = LlamaModel(cfg)
    sd = {}
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight":
            continue  # tied to embed_tokens, like the real checkpoint
        if name.startswith("rotary_emb."):
            continue
        sd[f"model.{name}"] = tensor.detach().contiguous().clone()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(CKPT_DIR / "model.safetensors"))
    cfg.save_pretrained(str(CKPT_DIR))
    print(f"wrote {len(sd)} tensors to {CKPT_DIR}")


def build_and_load(cfg, tp, pp, num_microbatches, device):
    from core.models import LlamaModel
    from core.distributed.pipeline.layers import PipelineParallelModule
    from core.distributed.tensor.module import TensorParallelModule
    from core.process_groups_config import CommGroupsConfig
    from core.utils import distribute_layers_pp_stages, load_safetensor_weights

    comm = CommGroupsConfig()
    with init_empty_weights(include_buffers=False):
        model = LlamaModel(cfg)
        if tp > 1:
            model = TensorParallelModule(model, cfg, comm.tp_group)
        if pp > 1:
            model = PipelineParallelModule(
                model, cfg, num_microbatches=num_microbatches, seq_len=SEQ
            )
    pp_rank = dist.get_rank(group=comm.pp_group)
    layer_indices = distribute_layers_pp_stages(cfg.num_hidden_layers, pp_rank, pp)
    load_safetensor_weights(
        model=model,
        safetensors_dir=str(CKPT_DIR),
        global_layer_indices=layer_indices,
        is_first_stage=comm.is_first_stage,
        is_last_stage=comm.is_last_stage,
        device=device,
        dtype=torch.float32,
    )
    return model, layer_indices


def run_step(model, tokens, labels, optimizer, cfg, pp):
    from core.distributed.pipeline.module import training_step_1f1b
    from core.utils import training_step

    if pp > 1:
        return training_step_1f1b(model, tokens, labels, optimizer, dtype=torch.float32)
    return training_step(model, tokens, labels, optimizer, cfg)


def global_param_name(local_name: str, layer_indices: list[int]) -> str:
    """Map `layers.{local}.x` back to the reference `layers.{global}.x`."""
    if local_name.startswith("layers."):
        rest = local_name[len("layers.") :]
        idx, suffix = rest.split(".", 1)
        return f"layers.{layer_indices[int(idx)]}.{suffix}"
    return local_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["make-ckpt", "reference", "check"], required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--num-microbatches", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "make-ckpt":
        make_ckpt()
        return 0

    from core.parallel_state import initialize_parallel_state
    from core.process_groups_config import CommGroupsConfig

    dist.init_process_group(
        backend="gloo", init_method="env://", timeout=datetime.timedelta(minutes=2)
    )
    world = dist.get_world_size()
    rank = dist.get_rank()
    tp, pp = args.tp, args.pp
    dp = world // (tp * pp)
    initialize_parallel_state(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    comm = CommGroupsConfig()
    device = torch.device("cpu")
    cfg = LlamaConfig.from_pretrained(str(CKPT_DIR))
    cfg._attn_implementation = "eager"

    num_microbatches = args.num_microbatches or (min(4 * pp, BATCH // dp) if pp > 1 else 1)

    model, layer_indices = build_and_load(cfg, tp, pp, num_microbatches, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    tokens, labels = make_batch(device)
    dp_rank = dist.get_rank(group=comm.dp_group)
    per_dp = BATCH // dp
    tokens = tokens[dp_rank * per_dp : (dp_rank + 1) * per_dp]
    labels = labels[dp_rank * per_dp : (dp_rank + 1) * per_dp]

    # Step 1: loss + grads (grads are read before the optimizer zeroes them
    # on the next step; step functions call zero_grad at their start).
    loss = run_step(model, tokens, labels, optimizer, cfg, pp)
    grads = {
        global_param_name(n, layer_indices): p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    # Step 2: loss after the update, with the same batch.
    post_loss = run_step(model, tokens, labels, optimizer, cfg, pp)

    def reduce_loss(value):
        # Only the last stage has a loss; average it across DP replicas.
        if value is None:
            return None
        t = torch.tensor(value, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=comm.dp_group)
        return (t / dp).item()

    loss = reduce_loss(loss)
    post_loss = reduce_loss(post_loss)

    if args.mode == "reference":
        assert world == 1, "reference must run with a single process"
        REF_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"loss": loss, "post_loss": post_loss, "grads": grads}, REF_PATH)
        print(f"reference loss={loss:.6f} post_step_loss={post_loss:.6f} ({len(grads)} grads)")
        dist.destroy_process_group()
        return 0

    ref = torch.load(REF_PATH)
    failures: list[str] = []
    if comm.is_last_stage:
        if abs(loss - ref["loss"]) > ATOL:
            failures.append(f"loss {loss:.6f} != ref {ref['loss']:.6f}")
        if abs(post_loss - ref["post_loss"]) > ATOL:
            failures.append(f"post_step_loss {post_loss:.6f} != ref {ref['post_loss']:.6f}")

    compared = 0
    for name, grad in grads.items():
        ref_grad = ref["grads"].get(name)
        if ref_grad is None or ref_grad.shape != grad.shape:
            continue  # sharded / fused under TP: covered by the post-step loss check
        compared += 1
        if not torch.allclose(grad, ref_grad, atol=ATOL, rtol=RTOL):
            failures.append(
                f"grad mismatch {name}: max abs diff {(grad - ref_grad).abs().max():.3e}"
            )

    status = "PASS" if not failures else "FAIL"
    print(
        f"[rank {rank}] {status} dp={dp} tp={tp} pp={pp} m={num_microbatches} "
        f"loss={loss} post={post_loss} grads_compared={compared}/{len(grads)}"
    )
    for f in failures:
        print(f"[rank {rank}]   {f}")

    any_fail = torch.tensor(1 if failures else 0)
    dist.all_reduce(any_fail, op=dist.ReduceOp.MAX)
    dist.barrier()
    dist.destroy_process_group()
    return int(any_fail.item())


if __name__ == "__main__":
    raise SystemExit(main())
