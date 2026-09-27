"""Preprocess a JSONL dataset into Megatron-compatible .bin/.idx files.

Adapted from Megatron-LM tools/preprocess_data.py.
Uses HuggingFace tokenizers instead of the Megatron tokenizer framework,
and the self-contained IndexedDatasetBuilder from dataset/indexed_dataset.py.

Usage:
    python tools/preprocessed_data.py \
        --input data/smoltalk.jsonl \
        --output-prefix data/smoltalk \
        --tokenizer-name-or-path HuggingFaceTB/SmolLM2-135M \
        --json-keys text \
        --append-eod \
        --workers 8
"""

import argparse
import json
import math
import gzip
import glob
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from dataset.indexed_dataset import DType, IndexedDatasetBuilder


class Encoder:
    """Tokenizes JSON lines using a HuggingFace tokenizer.

    The tokenizer is initialized once per worker process via `initializer()`,
    then shared across all `encode()` calls in that process.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.padding_token_id = self.tokenizer.pad_token_id = 128004
        self.eos_token_id = self.tokenizer.eos_token_id
        self.begin_of_text_token_id = self.tokenizer.bos_token_id


    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            sentences = text if isinstance(text, list) else [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = self.tokenizer.encode(
                    sentence, add_special_tokens=True
                )
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


def get_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data for distributed training"
    )

    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="HuggingFace tokenizer name or local path (e.g. 'gpt2', 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading tokenizers with custom code from the Hub.",
    )
    parser.add_argument(
        "--eod-id",
        type=int,
        default=None,
        help="Override end-of-document token id (defaults to tokenizer.eos_token_id).",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--json-keys",
        nargs="+",
        default=["text"],
        help="Space-separated list of keys to extract from each JSON object.",
    )

    parser.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an end-of-document token to every document.",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path prefix for the output .bin/.idx files (without extension).",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel tokenization.",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=1,
        help="Number of file partitions for very large datasets.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Log progress every N documents.",
    )
    parser.add_argument(
        "--keep-sequential-samples",
        action="store_true",
        help="Preserve ordering when using partitions > 1.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length.",
    )
    return parser.parse_args()


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    return {
        "partition": input_file_name,
        "sentence_split": sentence_split_file,
        "output_prefix": output_prefix,
    }


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def process_json_file(input_file_name, output_prefix, args, num_workers):
    """Tokenize a single JSONL file and write .bin/.idx output."""
    print("Opening", input_file_name)
    fin = open(input_file_name, "r", encoding="utf-8")

    startup_start = time.time()
    SEQ_LENGTH = args.seq_length

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path, trust_remote_code=args.trust_remote_code
    )
    vocab_size = tokenizer.vocab_size
    encoder = Encoder(args, tokenizer)

    pool = multiprocessing.Pool(num_workers)
    encoded_docs = pool.imap(encoder.encode, fin, 32)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix, key, "document")
        output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix, key, "document")
        builders[key] = IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=DType.optimal_dtype(vocab_size),
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)
    cache = []
    KEY = args.json_keys[0]
    for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        chunks = []
        if len(cache) > SEQ_LENGTH:
            chunks.append(cache[:SEQ_LENGTH])
            cache = cache[SEQ_LENGTH:]
        document = doc[KEY]
        if len(cache) > 0:
            document.extend(cache)
            cache = []
        steps = len(document) // SEQ_LENGTH
        for step in range(steps):
            chunk = document[step*SEQ_LENGTH:step*SEQ_LENGTH + SEQ_LENGTH]
            chunks.append(chunk)
        cache.extend(document[steps*SEQ_LENGTH:steps*SEQ_LENGTH + len(document)])
        for chunk in chunks:
            builders[KEY].add_document(chunk, [len(chunk)])
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(
                f"Processed {i} documents ({i / elapsed:.1f} docs/s, {mbs:.2f} MB/s).",
                file=sys.stderr,
            )
    remaining = SEQ_LENGTH - len(cache)
    cache.extend([encoder.padding_token_id] * remaining)
    builders[KEY].add_document(cache, [len(cache)])
    fin.close()
    builders[KEY].finalize(output_idx_files[KEY])

    pool.close()
    pool.join()

    print(f"Done. Processed {i} documents total.")


def main():
    args = get_args()

    if args.partitions == 1:
        process_json_file(args.input, args.output_prefix, args, args.workers)
    else:
        in_ss_out_names = []

        if args.partitions > 1:
            in_file_names = glob.glob(args.input)

            if args.keep_sequential_samples:
                total_sample_count = 0
                for filename in in_file_names:
                    with open(filename, "r") as fin:
                        for fc, _ in enumerate(fin):
                            pass
                    total_sample_count += fc + 1
                partition_size = math.ceil(total_sample_count / args.partitions)

            for idx in range(args.partitions):
                in_ss_out_names.append(get_file_name(args, idx))

            partitions_present = check_files_exist(
                in_ss_out_names, "partition", args.partitions
            )

            if not partitions_present:
                partitioned_input_files = []
                for idx in range(args.partitions):
                    partitioned_input_files.append(
                        open(in_ss_out_names[idx]["partition"], "w")
                    )

                index = 0
                if args.keep_sequential_samples:
                    line_count = 0
                for in_file_name in in_file_names:
                    if in_file_name.endswith(".gz"):
                        fin = gzip.open(in_file_name, "rt")
                    else:
                        fin = open(in_file_name, "r", encoding="utf-8")

                    for line in fin:
                        partitioned_input_files[index].write(line)
                        if args.keep_sequential_samples:
                            line_count += 1
                            if line_count % partition_size == 0:
                                index += 1
                        else:
                            index = (index + 1) % args.partitions

                    fin.close()

                for idx in range(args.partitions):
                    partitioned_input_files[idx].close()

        workers_per_partition = max(1, args.workers // args.partitions)

        processes = []
        q = multiprocessing.Queue()

        def _process_partition(name, q):
            process_json_file(
                name["partition"], name["output_prefix"], args, workers_per_partition
            )
            q.put(True)

        for name in in_ss_out_names:
            p = multiprocessing.Process(target=_process_partition, args=(name, q))
            p.start()
            processes.append(p)

        for _ in processes:
            q.get()

        for p in processes:
            p.join()

        # Merge partitions
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, trust_remote_code=args.trust_remote_code
        )
        vocab_size = tokenizer.vocab_size

        for key in args.json_keys:
            output_bin = "{}_{}_{}.bin".format(args.output_prefix, key, "document")
            output_idx = "{}_{}_{}.idx".format(args.output_prefix, key, "document")
            builder = IndexedDatasetBuilder(
                output_bin, dtype=DType.optimal_dtype(vocab_size)
            )
            for name in in_ss_out_names:
                partition_prefix = "{}_{}_{}".format(
                    name["output_prefix"], key, "document"
                )
                builder.add_index(partition_prefix)
            builder.finalize(output_idx)

        print("Merged all partitions.")


if __name__ == "__main__":
    main()
