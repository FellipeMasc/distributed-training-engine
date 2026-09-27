"""Tensor-parallel communication primitives (the `f` and `g` operators).

These implement the two conjugate operators from the Megatron-LM paper
(Shoeybi et al., 2019, "Megatron-LM: Training Multi-Billion Parameter Language
Models Using Model Parallelism"):

    f : identity in the forward pass, all-reduce in the backward pass.
        Placed at the *input* of a column-parallel region.

    g : all-reduce in the forward pass, identity in the backward pass.
        Placed at the *output* of a row-parallel region.

Intuition:
    - The column-parallel weight splits the *output* features across ranks, so
      the forward needs no communication (`f` = identity fwd), but the input's
      gradient is summed across ranks in the backward (`f` = all-reduce bwd).
    - The row-parallel weight splits the *input* features across ranks, so each
      rank computes a partial sum that must be all-reduced in the forward
      (`g` = all-reduce fwd), while the backward is a no-op (`g` = identity bwd).

Additionally we provide scatter/gather along the last dimension, used when a
column-parallel layer wants to gather its output, or a row-parallel layer needs
to scatter a non-parallel input.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


# ────────────────────────────────────────────────────────────────────────────
# Low-level collective helpers (operate on plain tensors, no autograd).
# ────────────────────────────────────────────────────────────────────────────


def _all_reduce(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """All-reduce (sum) `input_` across the tensor-parallel `group`.

    Returns the reduced tensor. Should be a no-op when the group has size 1.
    """
    if dist.get_world_size(group) == 1:
        return input_
    dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=group)
    return input_


def _split_along_last_dim(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Split `input_` along its last dim and keep only this rank's shard.

    The last-dim size must be divisible by `world_size(group)`.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_
    rank = dist.get_rank(group)
    last_dim = input_.size(-1)
    if last_dim % world_size != 0:
        raise ValueError(
            f"Last dim {last_dim} is not divisible by tensor-parallel size {world_size}."
        )
    chunk = last_dim // world_size
    return input_[..., rank * chunk : (rank + 1) * chunk].contiguous()


def _gather_along_last_dim(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """All-gather shards along the last dim and concatenate them.

    Inverse of `_split_along_last_dim`.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_
    input_ = input_.contiguous()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    dist.all_gather(tensor_list, input_, group=group)
    return torch.cat(tensor_list, dim=-1)


# ────────────────────────────────────────────────────────────────────────────
# Autograd operators.
# ────────────────────────────────────────────────────────────────────────────


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """The `f` operator: identity forward, all-reduce backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _all_reduce(grad_output, ctx.group), None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """The `g` operator: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        return _all_reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Split along last dim in forward, all-gather in backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return _split_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _gather_along_last_dim(grad_output, ctx.group), None


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """All-gather along last dim in forward, split in backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _split_along_last_dim(grad_output, ctx.group), None


# ────────────────────────────────────────────────────────────────────────────
# Functional wrappers (use these from the layers).
# ────────────────────────────────────────────────────────────────────────────


def copy_to_tensor_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """`f`: apply at the input of a column-parallel layer."""
    return _CopyToTensorParallelRegion.apply(input_, group)


def reduce_from_tensor_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """`g`: apply at the output of a row-parallel layer (or vocab embedding)."""
    return _ReduceFromTensorParallelRegion.apply(input_, group)


def scatter_to_tensor_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Split a full tensor across ranks along the last dim (with autograd)."""
    return _ScatterToTensorParallelRegion.apply(input_, group)


def gather_from_tensor_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Gather sharded tensors into a full tensor along the last dim."""
    return _GatherFromTensorParallelRegion.apply(input_, group)
