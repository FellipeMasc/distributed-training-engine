import pathlib
import sys
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from core.models import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig


tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
tokenizer.pad_token_id = 128004
config = LlamaConfig.from_pretrained("NousResearch/Llama-3.2-1B")
config.is_causal = True 
model = LlamaForCausalLM(config)
input = tokenizer("Hello, world! <|finetune_right_pad_id|> hi <|finetune_right_pad_id|>", return_tensors="pt",add_special_tokens=True)
output = model(**input)
print(output)