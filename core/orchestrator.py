from dataclasses import asdict

from engine.config.schema import RunConfig
from engine.core.logging import configure_logging
from engine.core.runtime_context import build_runtime_context


def build_execution_plan(config: RunConfig) -> dict[str, object]:
    return {
        "command": config.run.command,
        "model": config.model.name,
        "dtype": config.model.dtype,
        "dataset": config.data.dataset_name,
        "backend": config.distributed.backend,
        "data_parallel_size": config.distributed.data_parallel_size,
        "max_steps": config.training.max_steps,
        "batch_size": config.training.batch_size,
        "grad_accum": config.training.grad_accum,
        "resolved_config": asdict(config),
    }


def run_dryrun(config: RunConfig) -> None:
    from engine.distributed.setup import initialize_distributed, shutdown_distributed
    from engine.distributed.topology import validate_dp_topology

    configure_logging()
    context = build_runtime_context()
    initialize_distributed(context, config)
    try:
        validate_dp_topology(context.world_size, config.distributed.data_parallel_size)
        print("Dryrun passed.")
    finally:
        shutdown_distributed()


def run_train(config: RunConfig) -> None:
    from engine.distributed.setup import initialize_distributed, shutdown_distributed
    from engine.distributed.topology import validate_dp_topology
    from engine.trainers.dp_trainer import run_dp_training

    configure_logging()
    context = build_runtime_context()
    initialize_distributed(context, config)
    try:
        validate_dp_topology(context.world_size, config.distributed.data_parallel_size)
        run_dp_training(config, context)
    finally:
        shutdown_distributed()
