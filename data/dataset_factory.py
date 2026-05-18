from datasets import load_dataset

from engine.config.schema import RunConfig


def build_dataset(config: RunConfig):
    return load_dataset(config.data.dataset_name, split="train")
