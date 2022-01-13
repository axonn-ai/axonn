from . import config
from typing import Optional
from .communication import communication_handle
import torch

is_initialized = False
comm_handle = None


def init(G_inter: int, G_data: int, micro_batch_size: int, batch_size: int, gpus_per_node: Optional[int] = None) -> None:
    global comm_handle, is_initialized
    comm_handle = communication_handle(G_inter, G_data, gpus_per_node)
    config.G_inter = G_inter
    config.G_data = G_data
    config.micro_batch_size = micro_batch_size
    config.batch_size = batch_size
    is_initialized = True
    if comm_handle.world_rank == 0:
        print(f"Running with G_data =  {config.G_data} X G_inter = {config.G_inter} | microbatch_size = {config.micro_batch_size} | batch_size = {config.batch_size}")
    print(f"Hello from ilp rank = {comm_handle.inter_layer_parallel_rank}, dp rank = {comm_handle.data_parallel_rank}")


def create_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int = 0) -> Optional[torch.utils.data.DataLoader]:
    assert is_initialized
    if comm_handle.data_parallel_rank!=0:
        return None
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=config.G_data, rank=comm_handle.data_parallel_rank)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler)


