# Copyright 2021-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from . import config
from typing import Optional
from .communication import communication_handle
import torch

# True when init has been called
is_initialized = False
# Communication handle for point-to-point (MPI) and collective (NCCL) communication
comm_handle = None


def init(
    G_inter: int = 1,
    G_data: int = 1,
    G_intra_r: int = 1,
    G_intra_c: int = 1,
    G_intra_d: int = 1,
    gpus_per_node: Optional[int] = None,
) -> None:
    """
    Initialize AxoNN's 2D parallelism with G_inter-way inter-layer
    parallelism and G_data-way data parallelism

    Arguments:
        G_inter (int): number of GPUs used for inter-layer parallelism
        G_data (int): number of GPUs used for data parallelism
        G_intra (int): number of GPUs for intra-layer parallelism, note
        that the user has to implement intra-layer kernels themselves.
        AxoNN just creates the required process groups.
        gpus_per_node (int, optional):  number of GPUs per node, if not
            provided this is inferred using pytorch

    """
    global comm_handle, is_initialized
    comm_handle = communication_handle(
        G_inter, G_data, G_intra_r, G_intra_c, G_intra_d, gpus_per_node=gpus_per_node
    )
    config.G_inter = G_inter
    config.G_data = G_data
    config.G_intra = G_intra_r * G_intra_c * G_intra_d
    config.G_intra_r = G_intra_r
    config.G_intra_c = G_intra_c
    config.G_intra_d = G_intra_d
    config.inter_layer_parallel_rank = comm_handle.inter_layer_parallel_rank
    config.data_parallel_rank = comm_handle.data_parallel_rank
    config.intra_layer_parallel_rank = comm_handle.intra_layer_parallel_rank
    config.intra_layer_depth_parallel_rank = comm_handle.intra_layer_depth_parallel_rank
    config.intra_layer_row_parallel_rank = comm_handle.intra_layer_row_parallel_rank
    config.intra_layer_column_parallel_rank = (
        comm_handle.intra_layer_column_parallel_rank
    )
    is_initialized = True


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    global_batch_size: int,
    micro_batch_size: int = 1,
    num_workers: int = 0,
    *args,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create dataloaders for each GPU. For inter_layer_parallel_rank > 0,
    this creates a proxy dataloader which returns zero tensors.

    Arguments:
        dataset (torch.utils.data.Dataset): a PyTorch dataset object
        global_batch_size (int): global batch size over all GPUs
        micro_batch_size (int): microbatch size for inter-layer parallelism
        num_workers (int): number of worker processes in the dataloader

    Returns:
        data_loader (torch.utils.data.DataLoader): the dataloader object which
        is a true dataloader for inter_layer_parallel_rank = 0, else it is
        a proxy dataloader
    """
    assert is_initialized
    config.micro_batch_size = micro_batch_size
    config.global_batch_size = global_batch_size
    config.batch_size_per_gpu = global_batch_size // (config.G_data * config.G_intra_d)
    assert (
        global_batch_size % (config.G_data * micro_batch_size) == 0
    ), "Batch Size should be divisible by the G_data*micro_batch_size"

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=config.G_data * config.G_intra_d,
        rank=config.G_intra_d * config.data_parallel_rank
        + config.intra_layer_depth_parallel_rank,
    )
    if config.G_inter > 1:
        batch_size_for_dataloader = config.batch_size_per_gpu
    else:
        batch_size_for_dataloader = config.micro_batch_size

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size_for_dataloader,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True,
        *args,
        **kwargs,
    )  # not working with drop_last=False
