from engine.config.schema import RunConfig
from engine.core.runtime_context import RuntimeContext


def run_torch_training(config: RunConfig, context: RuntimeContext) -> None:
    # Optional fallback adapter placeholder.
    print(
        "Torch adapter placeholder | "
        f"rank={context.rank} world_size={context.world_size} "
        f"model={config.model.name} steps={config.training.max_steps}"
    )
