import argparse
import contextlib
import datetime
import os
import pathlib
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import LlamaForCausalLM, LlamaModel
from core.parallel_state import initialize_parallel_state, _get_group_global_ranks
from core.process_groups_config import CommGroupsConfig
from core.utils import download_model, distribute_layers_pp_stages, load_safetensor_weights
from dataset.data_sampler import MegatronPretrainingRandomSampler
from dataset.sft_dataset import build_sft_dataset
from dataset.utils import Split
from dataset.indexed_dataset import IndexedDataset, PackingDataset
from transformers.loss.loss_utils import ForCausalLMLoss
from core.distributed.pipeline.module import PipelineParallelModule
from core.distributed.pipeline.module import training_step_1f1b
from core.utils import training_step
from accelerate import init_empty_weights
from torch.profiler import profile, record_function, ProfilerActivity
from core.distributed.tensor.module import TensorParallelModule

from tools.preprocessed_data import document_prefix, preprocess_jsonl

IGNORE_INDEX = -100
TOKENIZER_NAME = "NousResearch/Llama-3.2-1B"
# Raw corpus. The packed .bin/.idx pair is derived from it per --seq-length
# (see resolve_data_prefix / ensure_preprocessed_data) so the training run is
# self-contained: preprocess if missing, then train.
DATA_DIR = PROJECT_ROOT / "dataset" / "data"
RAW_DATA_PATH = DATA_DIR / "tinystories-portuguese.jsonl"
DATA_JSON_KEY = "output"
MODEL_CACHE_DIR = "hf_model_safetensors"

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=TOKENIZER_NAME,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token. Defaults to HF_TOKEN env var.",
    )
    parser.add_argument("--dp", type=int, default=None, help="Data parallel degree.")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel degree.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree.")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel degree.")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Per-rank micro batch size.")
    parser.add_argument(
        "--num-microbatches",
        type=int,
        default=None,
        help=(
            "Number of pipeline microbatches each per-rank batch is split into. "
            "Bubble fraction is (pp-1)/m, so defaults to 4*pp (capped at the batch "
            "size). Must divide --micro-batch-size. Ignored when pp=1."
        ),
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help=(
            "Override the checkpoint's num_hidden_layers. Must be between 1 and "
            "the checkpoint's value; the first N layers are loaded (useful for "
            "smaller debug/profiling runs). Defaults to the checkpoint config."
        ),
    )
    parser.add_argument("--seq-length", type=int, default=512, help="Packed sequence length.")
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help=(
            "Rebuild the packed .bin/.idx for --seq-length even if it already "
            "exists. By default preprocessing only runs when the files are missing."
        ),
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help=(
            "Tokenizer worker processes used when preprocessing the raw JSONL. "
            "1 tokenizes inline (no process pool), which is the safe default "
            "inside torchrun."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max-steps", type=int, default=1, help="Number of optimizer steps.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Training precision.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=[None, "nccl", "gloo"],
        help="Distributed backend. Defaults to nccl on CUDA, otherwise gloo.",
    )
    parser.add_argument(
        "--conversation",
        type = bool,
        default= False,
        help= "Whether a sft training or a pre training"
    )
    return parser.parse_args()


def collate_fn_conversation(batch):
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "loss_mask": torch.stack([item["loss_mask"] for item in batch]),
        "position_ids": torch.stack([item["position_ids"] for item in batch]),
    }
def collate_fn_text(batch):
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "position_ids": torch.stack([item["position_ids"] for item in batch]),
    }


def resolve_data_prefix(seq_length: int) -> str:
    """Prefix (without .bin/.idx) of the packed dataset for a sequence length.

    Layout matches what tools/preprocessed_data.py writes, e.g.
    ``dataset/data/tinystories-portuguese-512_output_document``.
    """
    output_prefix = DATA_DIR / f"{RAW_DATA_PATH.stem}-{seq_length}"
    return document_prefix(str(output_prefix), DATA_JSON_KEY)


def ensure_preprocessed_data(seq_length, local_rank, force=False, workers=1) -> str:
    """Make sure the packed .bin/.idx for ``seq_length`` exists, then return its prefix.

    One process per node (local_rank 0) tokenizes the raw JSONL into
    ``seq_length``-token chunks; every other rank waits on the barrier so the
    files are complete before any rank memory-maps them. Mirrors how
    load_model_config downloads the checkpoint.
    """
    data_prefix = resolve_data_prefix(seq_length)
    if local_rank == 0 and (force or not IndexedDataset.exists(data_prefix)):
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Raw dataset not found at {RAW_DATA_PATH}; run "
                "dataset/jsonl_text_dataset.py first."
            )
        print(
            f"Rank {dist.get_rank()}: preprocessing {RAW_DATA_PATH.name} "
            f"-> {os.path.basename(data_prefix)}.{{bin,idx}} (seq_length={seq_length})"
        )
        written = preprocess_jsonl(
            input_path=RAW_DATA_PATH,
            output_prefix=DATA_DIR / f"{RAW_DATA_PATH.stem}-{seq_length}",
            seq_length=seq_length,
            json_keys=(DATA_JSON_KEY,),
            append_eod=True,
            workers=workers,
        )
        assert written == [data_prefix], (written, data_prefix)
    dist.barrier()
    if not IndexedDataset.exists(data_prefix):
        raise FileNotFoundError(
            f"Packed dataset missing at {data_prefix}.{{bin,idx}} after preprocessing."
        )
    return data_prefix


def build_dataloader(
    data_prefix, seq_length, micro_batch_size, dp_rank, dp_size, dataset_type="text_dataset"
):
    if dataset_type == "text_dataset":
        dataset = PackingDataset(
            path_prefix=str(data_prefix),
            sequence_length=seq_length,
        )
    elif dataset_type == "sft_dataset":
        dataset = build_sft_dataset(
            dataset_path=str(data_prefix),
            tokenizer_path=TOKENIZER_NAME,
            sequence_length=seq_length,
            index_split=Split.train,
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    sampler = MegatronPretrainingRandomSampler(
        dataset=dataset,
        total_samples=len(dataset),
        consumed_samples=0,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        data_sharding=False,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn_conversation if dataset_type == "sft_dataset" else collate_fn_text,
    )


def masked_cross_entropy(logits, labels, loss_mask):
    vocab_size = logits.size(-1)
    losses = F.cross_entropy(
        logits.float().view(-1, vocab_size),
        labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    flat_mask = loss_mask.view(-1)
    return (losses * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)


def resolve_num_microbatches(args) -> int:
    """Pick and validate the pipeline microbatch count.

    The pipeline bubble is (pp-1)/m of the step, so a small m idles most of
    the stages (pp=4, m=2 -> 60% idle). Default to 4*pp, capped at the batch
    size so the default always divides it.
    """
    if args.pp <= 1:
        return 1
    m = args.num_microbatches
    if m is None:
        m = min(4 * args.pp, args.micro_batch_size)
    if m < 1:
        raise ValueError(f"--num-microbatches must be >= 1, got {m}.")
    if args.micro_batch_size % m != 0:
        raise ValueError(
            f"--micro-batch-size={args.micro_batch_size} must be divisible by "
            f"--num-microbatches={m}."
        )
    return m


def load_model_config(args, local_rank):
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token and local_rank == 0:
        download_model(args.model_name, hf_token)
    dist.barrier()

    if os.path.exists(os.path.join(MODEL_CACHE_DIR, "config.json")):
        config = AutoConfig.from_pretrained(MODEL_CACHE_DIR, local_files_only=True)
    else:
        config = AutoConfig.from_pretrained(args.model_name)
    return apply_config_overrides(config, args)


def apply_config_overrides(config, args):
    """Apply user overrides to the checkpoint config.

    Only `num_hidden_layers` is overridable for now. Truncating keeps the
    first N transformer layers: the weight loader only maps checkpoint layers
    whose global index belongs to this stage, so the remaining layers are
    simply never read.
    """
    if args.num_hidden_layers is not None:
        n = args.num_hidden_layers
        if not 1 <= n <= config.num_hidden_layers:
            raise ValueError(
                f"--num-hidden-layers={n} must be between 1 and the checkpoint's "
                f"num_hidden_layers={config.num_hidden_layers}."
            )
        config.num_hidden_layers = n
    return config


def train():
    args = parser_args()

    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if not dist.is_initialized():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=3),
        )

    model_dtype = DTYPE_MAP[args.dtype]

    world_size = dist.get_world_size()
    model_parallel_size = args.tp * args.pp * args.cp
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tp*pp*cp={model_parallel_size}."
        )
    inferred_dp = world_size // model_parallel_size
    if args.dp is not None and args.dp != inferred_dp:
        raise ValueError(
            f"Requested dp={args.dp}, but inferred dp={inferred_dp} from "
            f"world_size={world_size} and tp/pp/cp=({args.tp}/{args.pp}/{args.cp})."
        )

    initialize_parallel_state(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
    )

    comm_groups_config = CommGroupsConfig()
    local_rank = comm_groups_config.local_rank
    dp_group = comm_groups_config.dp_group
    pp_group = comm_groups_config.pp_group
    cp_group = comm_groups_config.cp_group
    tp_group = comm_groups_config.tp_group
    dp_rank = dist.get_rank(group=dp_group)
    dp_size = dist.get_world_size(group=dp_group)

    #debug 
    # list_dp_group = _get_group_global_ranks(dp_group)
    # list_pp_group = _get_group_global_ranks(pp_group)
    # list_cp_group = _get_group_global_ranks(cp_group)
    # list_tp_group = _get_group_global_ranks(tp_group)
    # print(f"RANK {comm_groups_config.local_rank}: dp_group: {list_dp_group}")
    # print(f"RANK {comm_groups_config.local_rank}: pp_group: {list_pp_group}")
    # print(f"RANK {comm_groups_config.local_rank}: cp_group: {list_cp_group}")
    # print(f"RANK {comm_groups_config.local_rank}: tp_group: {list_tp_group}")
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    model_config = load_model_config(args, local_rank)

    num_microbatches = resolve_num_microbatches(args)

    with init_empty_weights(include_buffers=False):
        model = LlamaModel(model_config)
        if args.tp > 1:
            model = TensorParallelModule(model, model_config, tp_group)
        if args.pp > 1:
            model = PipelineParallelModule(
                model, model_config, num_microbatches=num_microbatches, seq_len=args.seq_length
            )

    pp_rank = dist.get_rank(group=pp_group)
    pp_world_size = dist.get_world_size(group=pp_group)
    global_layer_indices = distribute_layers_pp_stages(
        model_config.num_hidden_layers, pp_rank, pp_world_size
    )


    load_safetensor_weights(
        model=model,
        safetensors_dir=MODEL_CACHE_DIR,
        global_layer_indices=global_layer_indices,
        is_first_stage=comm_groups_config.is_first_stage,
        is_last_stage=comm_groups_config.is_last_stage,
        device=device,
        dtype=model_dtype,
    )

    # Data-parallel gradient averaging is done manually inside the step
    # functions (see core/distributed/data/comm.py) rather than via DDP, since
    # the custom 1F1B schedule drives backward itself.
    data_prefix = ensure_preprocessed_data(
        args.seq_length,
        local_rank,
        force=args.force_preprocess,
        workers=args.preprocess_workers,
    )
    dataloader = build_dataloader(
        data_prefix, args.seq_length, args.micro_batch_size, dp_rank, dp_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # amp_enabled = use_cuda and model_dtype in (torch.float16, torch.bfloat16)
    # scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and model_dtype == torch.float16)

    use_pipeline = args.pp > 1

    print(
        f"Rank {dist.get_rank()}: Training "
        f"(dp={inferred_dp}, pp={args.pp}, tp={args.tp}, cp={args.cp}, "
        f"num_microbatches={num_microbatches if use_pipeline else 1}, "
        f"dtype={model_dtype}, device={device})"
    )

    model.train()
    step = 0

    rank = dist.get_rank()
    print(f"Rank {rank} starting training")
    while step < args.max_steps:
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)

            if use_pipeline:
                loss_val = training_step_1f1b(
                    model, tokens, labels, optimizer, dtype=model_dtype, dp_group=dp_group
                )
            else:
                loss_val = training_step(
                    model, tokens, labels, optimizer, model_config, dp_group=dp_group
                )
            step += 1
            if rank == 0 and loss_val is not None:
                print(f"step {step}/{args.max_steps} | loss {loss_val:.4f}")
            if step >= args.max_steps:
                break

    dist.barrier()
    print(f"Rank {dist.get_rank()}: Training finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    train()

# model.train()
#     step = 0
#     while step < args.max_steps:
#         for batch in dataloader:
#             tokens = batch["tokens"].to(device)
#             labels = batch["labels"].to(device)
#             position_ids = batch["position_ids"].to(device)

#             outputs = model(input_ids=tokens, position_ids=position_ids)
#             if args.conversation:
#                 loss_mask = batch["loss_mask"].to(device)
#                 loss = masked_cross_entropy(outputs.logits, labels, loss_mask)
#             else:
#                 loss = ForCausalLMLoss(outputs.logits,labels,model_config.vocab_size,shift_labels=labels)

#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()

#             step += 1
#             if dist.get_rank() == 0:
#                 print(f"step {step}/{args.max_steps} | loss {loss.item():.4f}")
#             if step >= args.max_steps:
#                 break