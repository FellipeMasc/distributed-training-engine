import argparse
import datetime
import os
import pathlib
import sys

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.parallel_state import initialize_parallel_state, get_device_mesh
from core.process_groups_config import CommGroupsConfig
from core.utils import download_model


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None, help="Hugging Face model id.")
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
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
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
    return parser.parse_args()


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

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

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

    if args.model_name:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if local_rank == 0:
            if hf_token is None:
                raise ValueError(
                    "HF token is required to download model. Pass --hf-token or set HF_TOKEN."
                )
            download_model(args.model_name, hf_token)

            model_cache_dir = "hf_model_safetensors"
            model_config = AutoConfig.from_pretrained(
                model_cache_dir,
                local_files_only=True,
            )
            with torch.device("meta"):
                model_meta = AutoModelForCausalLM.from_config(model_config)
            
            modules_list = list(model_meta.named_modules())
            parameters_list = list(model_meta.named_parameters())
            state_dict_list = model_meta.state_dict()
            total_shape_params = sum(p.numel() for p in model_meta.parameters())
            print(
                f"Local rank 0: loaded {args.model_name} on meta "
                f"(shapes only, params={total_shape_params})."
            )
        dist.barrier()

    print(
        f"Rank {dist.get_rank()}: Training "
        f"(dp={inferred_dp}, pp={args.pp}, tp={args.tp}, cp={args.cp}, dtype={model_dtype})"
    )
    dist.barrier()
    print(f"Rank {dist.get_rank()}: Training finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    train()