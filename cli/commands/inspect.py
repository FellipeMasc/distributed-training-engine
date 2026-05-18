import json

from engine.config.loader import load_run_config
from engine.core.orchestrator import build_execution_plan


def run_inspect_command(config_path: str, as_json: bool) -> int:
    config = load_run_config(config_path)
    plan = build_execution_plan(config)

    if as_json:
        print(json.dumps(plan, indent=2))
    else:
        for key, value in plan.items():
            print(f"{key}: {value}")

    return 0
