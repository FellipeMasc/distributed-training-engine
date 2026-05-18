import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeContext:
    rank: int
    local_rank: int
    world_size: int
    device: str


def build_runtime_context() -> RuntimeContext:
    import torch

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return RuntimeContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device)
