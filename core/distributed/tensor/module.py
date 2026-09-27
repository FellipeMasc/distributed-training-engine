"""Outermost tensor-parallel transform.

Unlike Megatron-core, where the model is *built* from parallel layers via a
spec system, here the HF-style `LlamaModel` already exists (constructed on the
`meta` device under `accelerate.init_empty_weights`). So we take the Megatron
*idea* but apply it from the outside: walk the module tree and swap the target
submodules for their tensor-parallel equivalents.

Llama → tensor-parallel mapping (per Megatron's attention/MLP partitioning):

    embed_tokens                 -> VocabParallelEmbedding      (vocab split)
    self_attn                    -> TensorParallelAttention
        q_proj/k_proj/v_proj         fused into one ColumnParallelLinear
                                     `qkv_proj` (Megatron's `linear_qkv`)
        o_proj                       RowParallelLinear (input already parallel)
    mlp                          -> TensorParallelMLP
        gate_proj/up_proj            fused into one ColumnParallelLinear
                                     `gate_up_proj` (Megatron's `linear_fc1`)
        down_proj                    RowParallelLinear (input already parallel)
    lm_head                      -> ColumnParallelLinear (gather_output=False)
                                    + vocab-parallel cross-entropy (see loss.py)

Fusing q/k/v (and gate/up) matters for communication, not just GEMM size:
each ColumnParallelLinear applies the `f` operator to its input, and `f`
all-reduces the input gradient in the backward pass. Three separate
projections sharing one input therefore all-reduce the same `[B, S, H]`
gradient three times; one fused projection does it once.

Because every replacement layer allocates only its local shard, the parameter
shapes after this transform are already the *sharded* shapes. The checkpoint
loader (`core/utils.load_safetensor_weights`) slices each full tensor for this
rank and concatenates the q/k/v (gate/up) slices into the fused parameters.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from core.models import (
    LlamaAttention,
    LlamaMLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from core.distributed.tensor.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from core.parallel_state import get_tensor_model_parallel_group


def TensorParallelModule(
    model: nn.Module,
    model_config,
    tp_group: dist.ProcessGroup | None = None,
) -> nn.Module:
    """Replace the model's linear/embedding submodules with TP layers in place.

    Args:
        model: a meta-device `LlamaModel` exposing `embed_tokens`, `layers`,
            `lm_head`.
        model_config: the HF config (needs hidden_size, intermediate_size,
            vocab_size, num_attention_heads, num_key_value_heads, head_dim).
        tp_group: tensor-parallel process group. Defaults to the global TP group.

    Returns:
        The same `model` object, mutated in place.
    """
    if tp_group is None:
        tp_group = get_tensor_model_parallel_group()

    tp_size = dist.get_world_size(group=tp_group)
    if tp_size == 1:
        return model

    assert model_config.num_attention_heads % tp_size == 0
    assert model_config.num_key_value_heads % tp_size == 0
    assert model_config.vocab_size % tp_size == 0

    if getattr(model, "embed_tokens", None) is not None:
        _replace_vocab_embedding(model, "embed_tokens", tp_group)

    for layer in _iter_decoder_layers(model):
        layer.self_attn = TensorParallelAttention(layer.self_attn, model_config)
        layer.mlp = TensorParallelMLP(layer.mlp, model_config)

    if getattr(model, "lm_head", None) is not None:
        _replace_column_linear(model, "lm_head", gather_output=False)

    return model


# ────────────────────────────────────────────────────────────────────────────
# Fused tensor-parallel blocks.
# ────────────────────────────────────────────────────────────────────────────


class TensorParallelAttention(nn.Module):
    """Llama attention with q/k/v fused into one column-parallel projection.

    Heads are split across TP ranks: this rank owns
    `num_attention_heads // tp_size` query heads and
    `num_key_value_heads // tp_size` key/value heads. The fused local weight is
    laid out as `[q_local; k_local; v_local]` along dim 0, which is exactly what
    the loader builds by slicing each of q/k/v for this rank and concatenating.
    """

    def __init__(self, attn: LlamaAttention, config) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = attn.layer_idx
        self.head_dim = attn.head_dim
        self.num_key_value_groups = attn.num_key_value_groups
        self.scaling = attn.scaling
        self.attention_dropout = attn.attention_dropout
        self.is_causal = attn.is_causal

        q_out = config.num_attention_heads * self.head_dim
        kv_out = config.num_key_value_heads * self.head_dim
        bias = attn.q_proj.bias is not None
        dtype = attn.q_proj.weight.dtype

        self.qkv_proj = ColumnParallelLinear(
            in_features=config.hidden_size,
            out_features=q_out + 2 * kv_out,
            bias=bias,
            gather_output=False,
            dtype=dtype,
            name="qkv_proj",
        )
        tp_size = self.qkv_proj.tp_size
        self.q_size_per_partition = q_out // tp_size
        self.kv_size_per_partition = kv_out // tp_size

        self.o_proj = RowParallelLinear(
            in_features=q_out,
            out_features=config.hidden_size,
            bias=attn.o_proj.bias is not None,
            input_is_parallel=True,
            dtype=attn.o_proj.weight.dtype,
            name="o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(
            [
                self.q_size_per_partition,
                self.kv_size_per_partition,
                self.kv_size_per_partition,
            ],
            dim=-1,
        )
        query_states = query_states.reshape(hidden_shape).transpose(1, 2)
        key_states = key_states.reshape(hidden_shape).transpose(1, 2)
        value_states = value_states.reshape(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TensorParallelMLP(nn.Module):
    """Llama MLP with gate/up fused into one column-parallel projection.

    The fused local weight is laid out as `[gate_local; up_local]` along dim 0.
    """

    def __init__(self, mlp: LlamaMLP, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=mlp.gate_proj.bias is not None,
            gather_output=False,
            dtype=mlp.gate_proj.weight.dtype,
            name="gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=mlp.down_proj.bias is not None,
            input_is_parallel=True,
            dtype=mlp.down_proj.weight.dtype,
            name="down_proj",
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


# ────────────────────────────────────────────────────────────────────────────
# Per-module swap helpers.
# ────────────────────────────────────────────────────────────────────────────


def _replace_column_linear(
    parent: nn.Module,
    attr: str,
    gather_output: bool = False,
) -> None:
    """Replace `parent.<attr>` (an nn.Linear) with a ColumnParallelLinear."""
    old: nn.Linear = getattr(parent, attr)
    new = ColumnParallelLinear(
        in_features=old.in_features,
        out_features=old.out_features,
        bias=old.bias is not None,
        gather_output=gather_output,
        dtype=old.weight.dtype,
        name=attr,
    )
    setattr(parent, attr, new)


def _replace_row_linear(
    parent: nn.Module,
    attr: str,
    input_is_parallel: bool = True,
) -> None:
    """Replace `parent.<attr>` (an nn.Linear) with a RowParallelLinear."""
    old: nn.Linear = getattr(parent, attr)
    new = RowParallelLinear(
        in_features=old.in_features,
        out_features=old.out_features,
        bias=old.bias is not None,
        input_is_parallel=input_is_parallel,
        dtype=old.weight.dtype,
        name=attr,
    )
    setattr(parent, attr, new)


def _replace_vocab_embedding(
    parent: nn.Module,
    attr: str,
    tp_group: dist.ProcessGroup,
) -> None:
    """Replace `parent.<attr>` (an nn.Embedding) with a VocabParallelEmbedding."""
    old: nn.Embedding = getattr(parent, attr)
    new = VocabParallelEmbedding(
        num_embeddings=old.num_embeddings,
        embedding_dim=old.embedding_dim,
        tp_group=tp_group,
        dtype=old.weight.dtype,
    )
    setattr(parent, attr, new)


def _iter_decoder_layers(model: nn.Module):
    """Yield the transformer decoder layers, whether the model is the raw
    `LlamaModel` or a pipeline-wrapped module (both expose `.layers`)."""
    return model.layers
