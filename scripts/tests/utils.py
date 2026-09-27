import pathlib
import sys
import os
import datetime
import torch.distributed as dist
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core.utils import init_weights


def distribute_layers_pp_stages(num_layers, pp_rank, pp_world_size):
    layers = []
    chunk = num_layers // pp_world_size
    mod = num_layers % pp_world_size 
    consumed_layers = 0
    for rank in range(pp_world_size):
        local_chunk = chunk + 1 if rank < mod else chunk
        layers.append(list(range(consumed_layers,consumed_layers+local_chunk)))
        consumed_layers += local_chunk
    return layers[pp_rank]

if __name__ == "__main__":
    # print(distribute_layers_pp_stages(17, 0, 4))
    init_weights()