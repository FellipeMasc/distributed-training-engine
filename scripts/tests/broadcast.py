import torch
import torch.distributed as dist
import os
import datetime

def get_some_item(
    item,
    rank: int,
    dist_group: dist.ProcessGroup,
):
    if rank == 0:
        item = torch.tensor([rank, 1])
        dist.broadcast(item, src=0, group=dist_group)
    else:
        dist.broadcast(item, src=0, group=dist_group)
    return item



if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "gloo"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=f"env://", timeout=datetime.timedelta(minutes=10))
    dist_group = dist.new_group(ranks=list(range(world_size)))
    item = get_some_item(torch.tensor([1, 2]), rank, dist_group)
    print(f"Rank {rank} received item: {item}")
    dist.destroy_process_group()