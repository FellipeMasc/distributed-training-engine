from pathlib import Path
from typing import Any

import yaml

from engine.config.schema import (
    DataSection,
    DistributedSection,
    ModelSection,
    RunConfig,
    RunSection,
    TrainingSection,
)
from engine.core.errors import ConfigValidationError


def _required_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ConfigValidationError(f"Missing or invalid '{name}' section")
    return section


def load_run_config(config_path: str) -> RunConfig:
    path = Path(config_path)
    if not path.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")

    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ConfigValidationError("Top-level YAML content must be a map/object")

    run = _required_section(payload, "run")
    distributed = _required_section(payload, "distributed")
    model = _required_section(payload, "model")
    data = _required_section(payload, "data")
    training = _required_section(payload, "training")

    config = RunConfig(
        run=RunSection(
            command=str(run.get("command", "train")),
            seed=int(run.get("seed", 42)),
            output_dir=str(run.get("output_dir", "logs/run-001")),
        ),
        distributed=DistributedSection(
            backend=str(distributed.get("backend", "nccl")),
            data_parallel_size=int(distributed.get("data_parallel_size", 1)),
        ),
        model=ModelSection(
            name=str(model.get("name", "NousResearch/Llama-3.2-1B")),
            dtype=str(model.get("dtype", "bfloat16")),
        ),
        data=DataSection(
            dataset_name=str(data.get("dataset_name", "teknium/GPT4-LLM-Cleaned")),
            context_length=int(data.get("context_length", 512)),
            num_sequences=int(data.get("num_sequences", 1024)),
        ),
        training=TrainingSection(
            batch_size=int(training.get("batch_size", 1)),
            grad_accum=int(training.get("grad_accum", 1)),
            max_steps=int(training.get("max_steps", 10)),
            learning_rate=float(training.get("learning_rate", 2e-5)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: RunConfig) -> None:
    if config.distributed.data_parallel_size <= 0:
        raise ConfigValidationError("'distributed.data_parallel_size' must be > 0")
    if config.training.batch_size <= 0:
        raise ConfigValidationError("'training.batch_size' must be > 0")
    if config.training.grad_accum <= 0:
        raise ConfigValidationError("'training.grad_accum' must be > 0")
    if config.training.max_steps <= 0:
        raise ConfigValidationError("'training.max_steps' must be > 0")
