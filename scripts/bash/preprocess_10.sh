python /Users/fellipe/TG/distributed_training_engine/tools/preprocessed_data.py \
    --input /Users/fellipe/TG/distributed_training_engine/dataset/data/tinystories-portuguese.jsonl \
    --output-prefix /Users/fellipe/TG/distributed_training_engine/dataset/data/tinystories-portuguese-10 \
    --tokenizer-name-or-path NousResearch/Llama-3.2-1B \
    --json-keys output \
    --append-eod \
    --workers 8 \
    --seq-length 10