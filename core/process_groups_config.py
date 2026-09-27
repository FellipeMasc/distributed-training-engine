from dataclasses import dataclass
import os
import torch.distributed as dist

from core.parallel_state import (
    get_context_parallel_group,
    get_data_parallel_group,
    get_pipeline_context_neighbor_ranks,
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group,
    get_data_parallel_degree,
    get_context_model_parallel_degree,
    get_tensor_model_parallel_degree,
    get_pipeline_model_parallel_degree,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_device,
    get_tensor_parallel_local_rank
)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class CommGroupsConfig(metaclass=Singleton):

    def __init__(self):
        self.device = get_device()
        neighbor_ranks = get_pipeline_context_neighbor_ranks()
        self.local_rank=int(os.environ.get("LOCAL_RANK", dist.get_rank() % dist.get_world_size()))
        self.world_size=dist.get_world_size()
        self.dp_group=get_data_parallel_group()
        self.pp_group=get_pipeline_model_parallel_group()
        self.cp_group=get_context_parallel_group()
        self.tp_group=get_tensor_model_parallel_group()
        self.tp_rank=get_tensor_parallel_local_rank()
        self.num_stages=get_pipeline_model_parallel_degree()
        self.dp_degree=get_data_parallel_degree()
        self.cp_degree=get_context_model_parallel_degree()
        self.tp_degree=get_tensor_model_parallel_degree()
        self.pipeline_prev_rank=neighbor_ranks.pipeline_prev_rank
        self.pipeline_next_rank=neighbor_ranks.pipeline_next_rank
        self.context_prev_rank=neighbor_ranks.context_prev_rank
        self.context_next_rank=neighbor_ranks.context_next_rank
        self.is_first_stage=is_pipeline_first_stage()
        self.is_last_stage=is_pipeline_last_stage()


