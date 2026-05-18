from transformers import AutoTokenizer

from engine.config.schema import RunConfig


def build_tokenizer(config: RunConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
