"""Preprocess a JSONL dataset into Megatron-compatible .bin/.idx files.

Adapted from Megatron-LM tools/preprocess_data.py.
Uses HuggingFace tokenizers instead of the Megatron tokenizer framework,
and the self-contained IndexedDatasetBuilder from dataset/indexed_dataset.py.

Documents are tokenized, concatenated into one token stream per JSON key,
and cut into fixed-length chunks of `--seq-length` tokens. Each chunk is
stored as one sequence/document in the output. The final partial chunk is
right-padded with the pad token.

Usage:
    python tools/preprocessed_data.py \
        --input data/tinystories.jsonl \
        --output-prefix data/tinystories \
        --json-keys text \
        --append-eod \
        --seq-length 512 \
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

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from dataset.indexed_dataset import DType, IndexedDatasetBuilder

TOKENIZER_PATH = "NousResearch/Llama-3.2-1B"

# Llama 3 `<|finetune_right_pad_id|>`. Used when the tokenizer defines no pad
# token. Must match the pad id used by PackingDataset in dataset/indexed_dataset.py.
DEFAULT_PAD_TOKEN_ID = 128004


def load_tokenizer(trust_remote_code):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, trust_remote_code=trust_remote_code
    )


class Encoder:
    """Tokenizes JSON lines using a HuggingFace tokenizer.

    The encoder (and its tokenizer) is pickled to each pool worker along with
    the bound `encode` method, so every worker holds its own tokenizer copy.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        if args.pad_id is not None:
            self.padding_token_id = args.pad_id
        elif tokenizer.pad_token_id is not None:
            self.padding_token_id = tokenizer.pad_token_id
        else:
            self.padding_token_id = DEFAULT_PAD_TOKEN_ID

        if args.eod_id is not None:
            self.eod_token_id = args.eod_id
        else:
            self.eod_token_id = tokenizer.eos_token_id
        if args.append_eod and self.eod_token_id is None:
            raise ValueError(
                "--append-eod requested but the tokenizer has no eos token; "
                "pass --eod-id explicitly."
            )

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
            if self.args.append_eod and len(doc_ids) > 0:
                doc_ids.append(self.eod_token_id)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


def get_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data for distributed training"
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
        "--pad-id",
        type=int,
        default=None,
        help=(
            "Override pad token id used to fill the last chunk "
            f"(defaults to tokenizer.pad_token_id, then {DEFAULT_PAD_TOKEN_ID})."
        ),
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
    seq_length = args.seq_length

    tokenizer = load_tokenizer(args.trust_remote_code)
    vocab_size = tokenizer.vocab_size
    encoder = Encoder(args, tokenizer)

    # A single worker tokenizes inline: no process pool, so this path is safe
    # to call from inside an already-running torchrun process.
    pool = None
    if num_workers > 1:
        pool = multiprocessing.Pool(num_workers)
        encoded_docs = pool.imap(encoder.encode, fin, 32)
    else:
        encoded_docs = map(encoder.encode, fin)

    output_idx_files = {}
    builders = {}
    # Leftover tokens (fewer than seq_length) carried over to the next document.
    caches = {}

    for key in args.json_keys:
        output_bin_file = "{}_{}_{}.bin".format(output_prefix, key, "document")
        output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix, key, "document")
        builders[key] = IndexedDatasetBuilder(
            output_bin_file,
            dtype=DType.optimal_dtype(vocab_size),
        )
        caches[key] = []

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    i = 0
    for i, (doc, _sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key in args.json_keys:
            # Leftover from the previous document goes first, so token order
            # across the packed stream matches the input order.
            stream = caches[key] + doc[key]
            num_full = len(stream) // seq_length
            for step in range(num_full):
                chunk = stream[step * seq_length : (step + 1) * seq_length]
                builders[key].add_document(chunk, [len(chunk)])
            caches[key] = stream[num_full * seq_length :]
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(
                f"Processed {i} documents ({i / elapsed:.1f} docs/s, {mbs:.2f} MB/s).",
                file=sys.stderr,
            )

    for key in args.json_keys:
        cache = caches[key]
        if len(cache) > 0:
            cache.extend([encoder.padding_token_id] * (seq_length - len(cache)))
            builders[key].add_document(cache, [len(cache)])
        builders[key].finalize(output_idx_files[key])

    fin.close()
    if pool is not None:
        pool.close()
        pool.join()

    print(f"Done. Processed {i} documents total.")


def document_prefix(output_prefix, json_key):
    """Path prefix (without .bin/.idx) that process_json_file writes for a key."""
    return "{}_{}_{}".format(output_prefix, json_key, "document")


def preprocess_jsonl(
    input_path,
    output_prefix,
    seq_length,
    json_keys=("text",),
    append_eod=True,
    workers=1,
    eod_id=None,
    pad_id=None,
    trust_remote_code=False,
    log_interval=1000,
):
    """Tokenize a JSONL file into .bin/.idx chunks of ``seq_length`` tokens.

    Programmatic equivalent of the CLI (single partition). Returns the list of
    document prefixes written, one per key in ``json_keys``, so callers can
    open them with ``IndexedDataset``/``PackingDataset`` directly.
    """
    args = argparse.Namespace(
        input=str(input_path),
        output_prefix=str(output_prefix),
        seq_length=int(seq_length),
        json_keys=list(json_keys),
        append_eod=append_eod,
        workers=workers,
        eod_id=eod_id,
        pad_id=pad_id,
        trust_remote_code=trust_remote_code,
        log_interval=log_interval,
        partitions=1,
        keep_sequential_samples=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
    process_json_file(args.input, args.output_prefix, args, workers)
    return [document_prefix(args.output_prefix, key) for key in args.json_keys]


def _process_partition(name, args, num_workers, q):
    """Entry point for one partition subprocess.

    Defined at module level so it can be pickled under the `spawn` start
    method (the default on macOS and Windows).
    """
    process_json_file(name["partition"], name["output_prefix"], args, num_workers)
    q.put(True)


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

        for name in in_ss_out_names:
            p = multiprocessing.Process(
                target=_process_partition,
                args=(name, args, workers_per_partition, q),
            )
            p.start()
            processes.append(p)

        for _ in processes:
            q.get()

        for p in processes:
            p.join()

        # Merge partitions
        vocab_size = load_tokenizer(args.trust_remote_code).vocab_size

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
