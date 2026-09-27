"""Tensor-parallel layers (Megatron-LM style).

Three layers, mirroring `megatron/core/tensor_parallel/layers.py`:

    ColumnParallelLinear : weight [out, in] split along dim 0 (output features).
                           Y = X A, with A = [A_1, ..., A_p] column-wise.
    RowParallelLinear    : weight [out, in] split along dim 1 (input features).
                           Y = X A, with A = [A_1; ...; A_p] row-wise, and X
                           already split along its last dim.
    VocabParallelEmbedding : embedding table [vocab, hidden] split along dim 0
                             (the vocabulary), with out-of-range ids masked and
                             the result all-reduced.

Each layer only ever allocates its *local shard* of the weight. Nothing is
resized at load time: the parameter shape here already equals the shard shape,
so the checkpoint loader just needs to copy the matching slice into it.

Placement of the `f`/`g` operators (see comm.py):
    - ColumnParallelLinear applies `f` (copy_to_tensor_parallel_region) to its
      input, then optionally gathers its output.
    - RowParallelLinear optionally scatters its input, then applies `g`
      (reduce_from_tensor_parallel_region) to its output.
    - VocabParallelEmbedding applies `g` to its output.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from core.distributed.tensor.comm import (
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
)

from core.process_groups_config import CommGroupsConfig


def _divide(numerator: int, denominator: int) -> int:
    if numerator % denominator != 0:
        raise ValueError(
            f"{numerator} is not divisible by tensor-parallel size {denominator}."
        )
    return numerator // denominator


class ColumnParallelLinear(nn.Module):
    """Linear layer with the weight partitioned along the output dimension.

    The global weight is `[out_features, in_features]`; this rank holds
    `[out_features // tp_size, in_features]`.

    Args:
        in_features: full input dimension (same on every rank).
        out_features: full output dimension (split across ranks).
        bias: whether to learn an additive bias (also split along dim 0).
        gather_output: if True, all-gather the output so every rank returns the
            full `[..., out_features]` tensor; if False, return the local shard
            (typical when the next layer is row-parallel).
        dtype: passed through to parameter allocation (becomes `meta` when
            building under `init_empty_weights`).
        name: label used in `extra_repr` only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        dtype=None,
        name: str = None,
    ) -> None:
        super().__init__()
        self.comm_groups_config = CommGroupsConfig()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_group = self.comm_groups_config.tp_group
        self.tp_size = self.comm_groups_config.tp_degree
        self.tp_rank = self.comm_groups_config.tp_rank
        self.name = name
        self.output_size_per_partition = _divide(out_features, self.tp_size)
        # Global output-feature range owned by this rank: [start, end).
        self.partition_start = self.tp_rank * self.output_size_per_partition
        self.partition_end = self.partition_start + self.output_size_per_partition

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, in_features, device=self.comm_groups_config.device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, device=self.comm_groups_config.device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_tensor_parallel_region(input_, self.tp_group)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = gather_from_tensor_parallel_region(output_parallel, self.tp_group)
        else:
            output = output_parallel
        return output

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"out_per_partition={self.output_size_per_partition}, "
            f"tp_size={self.tp_size}, gather_output={self.gather_output}"
        )

class RowParallelLinear(nn.Module):
    """Linear layer with the weight partitioned along the input dimension.

    The global weight is `[out_features, in_features]`; this rank holds
    `[out_features, in_features // tp_size]`.

    Args:
        in_features: full input dimension (split across ranks).
        out_features: full output dimension (same on every rank).
        bias: whether to learn an additive bias (NOT split; added once after
            the all-reduce).
        input_is_parallel: if True, the input is already sharded along its last
            dim (e.g. it came from a column-parallel layer); if False, scatter
            it first.
        dtype: passed through to parameter allocation.
        name: label used in `extra_repr` only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        dtype=None,
        name: str = None,
    ) -> None:
        super().__init__()
        self.comm_groups_config = CommGroupsConfig()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tp_group = self.comm_groups_config.tp_group
        self.tp_size = self.comm_groups_config.tp_degree
        self.tp_rank = self.comm_groups_config.tp_rank
        self.name = name
        self.input_size_per_partition = _divide(in_features, self.tp_size)

        self.weight = nn.Parameter(torch.empty(out_features, self.input_size_per_partition, device=self.comm_groups_config.device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=self.comm_groups_config.device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_parallel_region(input_, self.tp_group)
        # The bias is replicated, so add it once *after* the all-reduce.
        output_parallel = F.linear(input_parallel, self.weight)
        output = reduce_from_tensor_parallel_region(output_parallel, self.tp_group)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"in_per_partition={self.input_size_per_partition}, "
            f"tp_size={self.tp_size}"
        )

class VocabParallelEmbedding(nn.Module):
    """Embedding with the vocabulary dimension partitioned across ranks.

    The global table is `[num_embeddings, embedding_dim]`; this rank owns the
    contiguous vocab range `[vocab_start_index, vocab_end_index)`.

    Forward: mask ids outside this rank's range, look up the (shifted) local
    ids, zero the masked rows, then all-reduce (`g`) so every rank ends up with
    the full embedding for every token.

    Args:
        num_embeddings: full vocabulary size (must be divisible by tp_size).
        embedding_dim: hidden size (not partitioned).
        tp_group: the tensor-parallel process group. Defaults to the global one.
        dtype: passed through to parameter allocation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_group: dist.ProcessGroup | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.comm_groups_config = CommGroupsConfig()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_group = tp_group if tp_group is not None else self.comm_groups_config.tp_group
        self.tp_size = dist.get_world_size(group=self.tp_group)
        self.tp_rank = dist.get_rank(group=self.tp_group)

        self.num_embeddings_per_partition = _divide(num_embeddings, self.tp_size)
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                embedding_dim,
                device=device if device is not None else self.comm_groups_config.device,
                dtype=dtype,
            )
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(input_, self.weight)
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_ - self.vocab_start_index
        masked_input = masked_input.masked_fill(input_mask, 0)
        output_parallel = F.embedding(masked_input, self.weight)
        output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0.0)
        return reduce_from_tensor_parallel_region(output_parallel, self.tp_group)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"per_partition={self.num_embeddings_per_partition}, tp_size={self.tp_size}"
        )
