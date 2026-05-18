from engine.config.loader import load_run_config
from engine.core.orchestrator import run_dryrun


def run_dryrun_command(config_path: str) -> int:
    config = load_run_config(config_path)
    run_dryrun(config)
    return 0
