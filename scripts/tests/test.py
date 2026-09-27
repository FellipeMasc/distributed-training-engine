import torch
import torch.distributed as dist
import os
import time

def pprint(rank, msg):
    # We add sleep to avoid printing clutter
    time.sleep(1 * rank)
    print(rank, msg)

if __name__ == "__main__":

    # os.environ['WORLD_SIZE'] = "4"
    # os.environ['MASTER_ADDR'] = '10.57.23.164'          
    # os.environ['MASTER_PORT'] = '8888'  
    dist.init_process_group("gloo")

    rank = dist.get_rank()
    pprint(rank, f"world size = {dist.get_world_size()}")            
    pprint(rank, f"backend = {dist.get_backend()}")                   
    pprint(rank,f"rank = {dist.get_rank()}")                          
    a = 2
    pprint(rank,f"a = {a+dist.get_rank()}")
    dist.barrier()
    pprint(rank,f"first barrier broken by {rank}")
    b = 3
    pprint(rank,f"b = {b+dist.get_rank()}")
    dist.barrier()
    pprint(rank,f"second barrier broken by {rank}")
