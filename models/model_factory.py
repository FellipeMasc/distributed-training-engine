import torch
from transformers import AutoModelForCausalLM

from engine.config.schema import RunConfig
from engine.core.errors import ModelInitializationError

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_model(config: RunConfig):
    dtype = DTYPE_MAP.get(config.model.dtype)
    if dtype is None:
        raise ModelInitializationError(f"Unsupported model dtype: {config.model.dtype}")

    return AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
