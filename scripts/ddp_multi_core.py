import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import socket 
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
def main():
    print(socket.getfqdn(socket.gethostname()))
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    model = DDP(nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    data = torch.randn(32, 10)
    targets = torch.randn(32, 2)

    per_rank = data.size(0) // world_size
    data = data[rank * per_rank:(rank + 1) * per_rank]
    targets = targets[rank * per_rank:(rank + 1) * per_rank]

    for epoch in range(5):
        print("initiating")
        optimizer.zero_grad()
        loss = loss_fn(model(data), targets)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print(f"Rank {rank} finished")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()