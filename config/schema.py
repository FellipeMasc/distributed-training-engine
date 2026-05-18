from dataclasses import dataclass


@dataclass(frozen=True)
class RunSection:
    command: str
    seed: int
    output_dir: str


@dataclass(frozen=True)
class DistributedSection:
    backend: str
    data_parallel_size: int


@dataclass(frozen=True)
class ModelSection:
    name: str
    dtype: str


@dataclass(frozen=True)
class DataSection:
    dataset_name: str
    context_length: int
    num_sequences: int


@dataclass(frozen=True)
class TrainingSection:
    batch_size: int
    grad_accum: int
    max_steps: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class RunConfig:
    run: RunSection
    distributed: DistributedSection
    model: ModelSection
    data: DataSection
    training: TrainingSection
