from engine.config.schema import RunConfig
from engine.core.errors import DistributedSetupError
from engine.core.runtime_context import RuntimeContext


def initialize_distributed(context: RuntimeContext, config: RunConfig) -> None:
    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(context.local_rank)

    if context.world_size <= 1:
        return

    if dist.is_initialized():
        return

    backend = config.distributed.backend
    try:
        dist.init_process_group(backend=backend)
    except Exception as exc:
        raise DistributedSetupError(f"Failed to init process group with backend '{backend}'") from exc


def shutdown_distributed() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
