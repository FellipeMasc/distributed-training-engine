import torch.distributed as dist
from dataclasses import dataclass
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

_DEVICE_MESH = None
_DATA_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GROUP = None
_TP_DEGREE = None
_PP_DEGREE = None
_CP_DEGREE = None
_DP_DEGREE = None


@dataclass(frozen=True)
class PipelineContextNeighborRanks:
    pipeline_prev_rank: int | None
    pipeline_next_rank: int | None
    context_prev_rank: int
    context_next_rank: int


def _infer_device_type() -> str:
    backend = str(dist.get_backend()).lower()
    return "cuda" if "nccl" in backend else "cpu"


def initialize_parallel_state(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    device_type: str | None = None,
) -> None:
    global _DEVICE_MESH
    global _DATA_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GROUP
    global _TP_DEGREE
    global _PP_DEGREE
    global _CP_DEGREE
    global _DP_DEGREE
    global _DEVICE

    world_size = dist.get_world_size()
    model_parallel_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * context_parallel_size
    )
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "World size must be divisible by "
            f"TP*PP*CP ({model_parallel_size}). Got world_size={world_size}."
        )
    data_parallel_size = world_size // model_parallel_size

    mesh_device_type = device_type or _infer_device_type()
    _DEVICE_MESH = init_device_mesh(
        device_type=mesh_device_type,
        mesh_shape=(
            data_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            tensor_model_parallel_size,
        ),
        mesh_dim_names=("dp", "pp", "cp", "tp"),
    )

    _DATA_PARALLEL_GROUP = _DEVICE_MESH.get_group("dp")
    _PIPELINE_MODEL_PARALLEL_GROUP = _DEVICE_MESH.get_group("pp")
    _CONTEXT_PARALLEL_GROUP = _DEVICE_MESH.get_group("cp")
    _TENSOR_MODEL_PARALLEL_GROUP = _DEVICE_MESH.get_group("tp")
    _TP_DEGREE = tensor_model_parallel_size
    _PP_DEGREE = pipeline_model_parallel_size
    _CP_DEGREE = context_parallel_size
    _DP_DEGREE = world_size // model_parallel_size
    _DEVICE = mesh_device_type
    
def get_device_mesh() -> DeviceMesh:
    if _DEVICE_MESH is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _DEVICE_MESH


def get_data_parallel_group():
    if _DATA_PARALLEL_GROUP is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _DATA_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    if _PIPELINE_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_context_parallel_group():
    if _CONTEXT_PARALLEL_GROUP is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _CONTEXT_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_degree():
    if _TP_DEGREE is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _TP_DEGREE

def get_pipeline_model_parallel_degree():
    if _PP_DEGREE is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _PP_DEGREE

def get_context_model_parallel_degree():
    if _CP_DEGREE is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _CP_DEGREE

def get_data_parallel_degree():
    if _DP_DEGREE is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _DP_DEGREE

def _get_group_global_ranks(group) -> list[int]:
    group_world_size = dist.get_world_size(group=group)
    group_ranks: list[int | None] = [None] * group_world_size
    dist.all_gather_object(group_ranks, dist.get_rank(), group=group)
    return [int(rank) for rank in group_ranks]


def get_pipeline_prev_next_ranks() -> tuple[int | None, int | None]:
    pp_group = get_pipeline_model_parallel_group()
    pp_local_rank = dist.get_rank(group=pp_group)
    pp_group_ranks = _get_group_global_ranks(pp_group)

    prev_rank = pp_group_ranks[pp_local_rank - 1] if pp_local_rank > 0 else None
    next_rank = (
        pp_group_ranks[pp_local_rank + 1]
        if pp_local_rank < len(pp_group_ranks) - 1
        else None
    )
    return prev_rank, next_rank


def get_context_prev_next_ranks() -> tuple[int, int]:
    cp_group = get_context_parallel_group()
    cp_local_rank = dist.get_rank(group=cp_group)
    cp_group_ranks = _get_group_global_ranks(cp_group)
    cp_group_size = len(cp_group_ranks)

    prev_rank = cp_group_ranks[(cp_local_rank - 1) % cp_group_size]
    next_rank = cp_group_ranks[(cp_local_rank + 1) % cp_group_size]
    return prev_rank, next_rank


def get_pipeline_context_neighbor_ranks() -> PipelineContextNeighborRanks:
    pp_prev_rank, pp_next_rank = get_pipeline_prev_next_ranks()
    cp_prev_rank, cp_next_rank = get_context_prev_next_ranks()
    return PipelineContextNeighborRanks(
        pipeline_prev_rank=pp_prev_rank,
        pipeline_next_rank=pp_next_rank,
        context_prev_rank=cp_prev_rank,
        context_next_rank=cp_next_rank,
    )

def get_device():
    if _DEVICE is None:
        raise RuntimeError("Parallel state is not initialized.")
    return _DEVICE

def get_stage_id():
    return _DEVICE_MESH.get_local_rank("pp")

def is_initialized():
    return _DEVICE_MESH is not None

def is_pipeline_last_stage():
    return _DEVICE_MESH.get_local_rank("pp") == _DEVICE_MESH.get_group("pp").size() - 1

def is_pipeline_first_stage():
    return _DEVICE_MESH.get_local_rank("pp") == 0

def get_tensor_parallel_local_rank():
    return _DEVICE_MESH.get_local_rank("tp")