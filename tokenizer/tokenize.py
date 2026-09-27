from torchrl.data.llm import TensorDictTokenizer
from torchrl.data import TokenizedDatasetLoader

split = "train"
max_length = 550
dataset_name = "CarperAI/openai_summarize_comparisons"
loader = TokenizedDatasetLoader(
    split,
    max_length,
    dataset_name,
    TensorDictTokenizer,
)
dataset = loader.load()
print(dataset)