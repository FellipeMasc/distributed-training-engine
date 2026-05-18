from engine.config.loader import load_run_config
from engine.core.orchestrator import run_train


def run_train_command(config_path: str) -> int:
    config = load_run_config(config_path)
    run_train(config)
    return 0
