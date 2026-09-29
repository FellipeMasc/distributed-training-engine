import os
from datasets import load_dataset
import json
ds = load_dataset("Dxniz/TinyStories-Multilingual", split='train')
ds.to_json('data/tinystories.jsonl', force_ascii=False)

if not os.path.exists('data/tinystories-portuguese.jsonl'):
    os.makedirs('data', exist_ok=True)

with open('data/tinystories.jsonl', 'r') as f:
    with open('data/tinystories-portuguese.jsonl', 'w') as f2:
        for line in f:
            row = json.loads(line)
            if row['language_code'] == 'pt':
                f2.write(line)

# idenpotent way to write the dataset to a jsonl file
# with open('data/tinystories-portuguese.jsonl', 'w') as f:
#     for row in ds:
#         if row['language_code'] == 'pt':
#             f.write(json.dumps({'text': row['output']}) + '\n')