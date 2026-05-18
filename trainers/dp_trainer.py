from engine.config.schema import RunConfig
from engine.core.runtime_context import RuntimeContext
from engine.integrations.deepspeed_adapter import run_deepspeed_training


def run_dp_training(config: RunConfig, context: RuntimeContext) -> None:
    run_deepspeed_training(config, context)
