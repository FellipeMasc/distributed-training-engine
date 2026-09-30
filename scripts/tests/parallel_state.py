import argparse
import datetime
import os
import pathlib
import sys

import torch.distributed as dist

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distributed_training_engine.core.parallel_state import (
    get_data_parallel_group,
    get_device_mesh,
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group,
    initialize_parallel_state,
    is_pipeline_last_stage,
    is_pipeline_first_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal torchrun test for distributed_training_engine.core.parallel_state"
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="torch.distributed backend",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Optional DeviceMesh type override",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=5),
    )

    try:
        initialize_parallel_state(
            tensor_model_parallel_size=args.tp,
            pipeline_model_parallel_size=args.pp,
            device_type=args.device_type,
        )

        mesh = get_device_mesh()
        tp_group = get_tensor_model_parallel_group()
        pp_group = get_pipeline_model_parallel_group()
        dp_group = get_data_parallel_group()

        dp_rank = mesh.get_local_rank("dp")
        pp_rank = mesh.get_local_rank("pp")
        tp_rank = mesh.get_local_rank("tp")

        tp_world_size = dist.get_world_size(group=tp_group)
        pp_world_size = dist.get_world_size(group=pp_group)
        dp_world_size = dist.get_world_size(group=dp_group)

        print(
            f"[rank={rank}] coords(dp={dp_rank}, pp={pp_rank}, tp={tp_rank}) "
            f"group_sizes(dp={dp_world_size}, pp={pp_world_size}, tp={tp_world_size})",
            flush=True,
        )

        print(f"is_pipeline_last_stage={is_pipeline_last_stage()}")
        print(f"is_pipeline_first_stage={is_pipeline_first_stage()}")
        dist.barrier()
        if rank == 0:
            print(f"mesh_shape={tuple(mesh.mesh.shape)} mesh_dim_names={mesh.mesh_dim_names}")
            print("parallel_state test passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()