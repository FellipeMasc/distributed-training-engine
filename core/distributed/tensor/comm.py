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
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _all_reduce(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """All-reduce (sum) `input_` across the tensor-parallel `group`.

    Returns the reduced tensor. Should be a no-op when the group has size 1.
    """
    if dist.get_world_size(group) == 1:
        return input_
    dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=group)
    return input_

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

