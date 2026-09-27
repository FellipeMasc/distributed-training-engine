"""Data-parallel gradient synchronization.

The training loop drives its own forward/backward schedule (1F1B for pipeline
parallelism), so instead of wrapping the model in ``DistributedDataParallel``
we reduce gradients manually right before ``optimizer.step()``.

This is the simplest correct option: one all-reduce per parameter per step,
with no bucketing and no overlap with backward. Add bucketing / ``no_sync``
once data-parallel communication shows up as the bottleneck in a profile.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def sync_grads(model: nn.Module, dp_group: dist.ProcessGroup | None) -> None:
    """Average ``p.grad`` across the data-parallel group for every parameter.

    A no-op when the group has a single member. Parameters whose ``grad`` is
    ``None`` (e.g. frozen, or not touched this step) are skipped on every rank,
    so all ranks must agree on which parameters received gradients.
    """
    if dp_group is None:
        return
    world_size = dist.get_world_size(group=dp_group)
    if world_size == 1:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.div_(world_size)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
