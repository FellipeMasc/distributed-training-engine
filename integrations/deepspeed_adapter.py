from engine.config.schema import RunConfig
from engine.core.runtime_context import RuntimeContext


def run_deepspeed_training(config: RunConfig, context: RuntimeContext) -> None:
    # Placeholder adapter boundary for integrating logic from deepspeed/main.py.
    # We keep this as the integration point so trainer logic stays framework-agnostic.
    print(
        "DeepSpeed adapter placeholder | "
        f"rank={context.rank} world_size={context.world_size} "
        f"model={config.model.name} steps={config.training.max_steps}"
    )
