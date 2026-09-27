# file: cp_sdpa_example.py
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import context_parallel_unshard
from torch.nn.attention import sdpa_kernel, SDPBackend


def context_parallel_sdpa_example(world_size: int, rank: int):
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    device_mesh = init_device_mesh(
        device_type="cpu", mesh_shape=(world_size,), mesh_dim_names=("cp",)
    )

    batch = 8
    nheads = 8
    qkv_len = 64
    dim = 32
    backend = SDPBackend.MATH
    dtype = (
        torch.bfloat16
        if backend == SDPBackend.MATH
        else torch.float32
    )

    qkv = [
        torch.rand(
            (batch, nheads, qkv_len, dim),
            dtype=dtype,
            requires_grad=True,
            device='cpu',
        )
        for _ in range(3)
    ]
    # specify the SDPBackend to use
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(*qkv, is_causal=True)

    # make a clean copy of QKV for output comparison
    cp_qkv = [t.detach().clone() for t in qkv]

    with sdpa_kernel(backend):
        # This `context_parallel()` performs two actions:
        # 1. Shard the tensor objects in `buffers` in-place along the dimension
        #    specified in `buffer_seq_dims`, the tensors in `buffers` and their
        #    sharding dims in `buffer_seq_dims` are organized in the same order.
        # 2. Replace the execution of `F.scaled_dot_product_attention` with a
        #    context-paralleled-enabled Ring Attention.
        with context_parallel(
            device_mesh, buffers=tuple(cp_qkv), buffer_seq_dims=(2, 2, 2)
        ):
            cp_out = F.scaled_dot_product_attention(*cp_qkv, is_causal=True)

        # The output `cp_out` is still sharded in the same way as QKV
        # the `context_parallel_unshard` API allows users to easily
        # unshard to gain the full tensor.
        (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])

    assert torch.allclose(
        cp_out,
        out,
        atol=(1e-08 if dtype == torch.float32 else 1e-03 * world_size),
    )


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    try:
        context_parallel_sdpa_example(world_size, rank)
    finally:
        dist.barrier()
        dist.destroy_process_group()