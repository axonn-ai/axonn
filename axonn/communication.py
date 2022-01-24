# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from mpi4py import MPI
import torch


class communication_handle:
    """
    Communnication handle for point-to-point(MPI) and collective
    communication(NCCL) of GPU tensors.
    """

    def __init__(self, G_inter: int, G_data: int, gpus_per_node: int = None):
        """Constructor for the communication handle

        Arguments:
            G_inter (int): number of GPUs used for inter-layer parallelism
            G_data (int): number of GPUs used for data parallelism
            gpus_per_node (int, optional): number of GPUs per node, if not
            provided this is inferred using pytorch
        """

        self.world_rank = MPI.COMM_WORLD.Get_rank()
        self.world_size = MPI.COMM_WORLD.Get_size()
        assert (
            G_inter * G_data == self.world_size
        ), "The product of G_inter and G_data should be equal to the number of GPUs"
        self.G_inter = G_inter
        self.G_data = G_data

        # infer gpus per node if not provided
        self.gpus_per_node = (
            gpus_per_node if gpus_per_node is not None else torch.cuda.device_count()
        )
        self.local_rank = self.world_rank % self.gpus_per_node
        torch.cuda.set_device(self.local_rank)
        self.data_parallel_rank = self.world_rank // G_inter
        self.inter_layer_parallel_rank = self.world_rank % G_inter

        # create communicator for point-to-point(MPI) communication
        self.p2p_mpi_comm = MPI.COMM_WORLD.Split(self.data_parallel_rank)

        # create communicator for collective (NCCL) communication
        if not torch.distributed.is_initialized():
            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method += master_ip + ":" + master_port
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=self.world_size,
                rank=self.world_rank,
                init_method=init_method,
            )

        for i in range(self.G_inter):
            # all ranks have to form all data parallel communicators and not
            # just their own
            ranks_in_ith_data_parallel_group = [
                j * self.G_inter + i for j in range(self.G_data)
            ]
            ith_data_parallel_group = torch.distributed.new_group(
                ranks=ranks_in_ith_data_parallel_group, backend="nccl"
            )
            if self.inter_layer_parallel_rank == i:
                self.coll_nccl_comm = ith_data_parallel_group

    def _torch_to_mpi(self, tensor: torch.Tensor):
        """Converts a PyTorch tensor into an mpi4py compatible array using its
        unified virtual address

        Arguments:
            tensor (torch.Tensor): the Pytorch tensor
        """
        return [
            MPI.memory.fromaddress(
                tensor.data_ptr(), tensor.element_size() * tensor.nelement()
            ),
            MPI.FLOAT,
        ]

    def send(
        self, tensor: torch.Tensor, recv_rank: int, tag: int, async_op: bool = True
    ):
        """Send a PyTorch tensor to a particular rank using MPI

        Arguments:
            tensor (torch.Tensor): the PyTorch tensor to be sent
            recv_rank (int): the rank of the receiver in the
                inter_layer_parallel communicator (self.p2p_mpi_comm)
            tag (int): the MPI tag for this message
            async_op (bool, optional): use asynchronous send

        Returns:
            mpi4py future object if async is true, else None - this object can
            be queried to check for completion of communication
        """
        mpi4py_compatible_array = self._torch_to_mpi(tensor)
        if async_op:
            mpi_future_object = self.p2p_mpi_comm.Isend(
                mpi4py_compatible_array, recv_rank, tag
            )
            return mpi_future_object
        else:
            self.p2p_mpi_comm.Send(mpi4py_compatible_array, recv_rank, tag)

    def recv(
        self,
        tensor: torch.Tensor,
        send_rank: int,
        tag: int = MPI.ANY_TAG,
        async_op: bool = True,
    ):
        """Receive a PyTorch tensor from a particular rank using MPI

        Arguments:
            tensor (torch.Tensor): the PyTorch tensor that will receive the data
            send_rank (int): the rank of the sender in the inter_layer_parallel
                communicator (self.p2p_mpi_comm)
            tag (int): the MPI tag for this message
            async_op (bool, optional): use asynchronous recv

        Returns:
            mpi4py future object if async is true, else None - this object
            can be queried to check for completion of communication
        """
        mpi4py_compatible_array = self._torch_to_mpi(tensor)
        if async_op:
            mpi_future_object = self.p2p_mpi_comm.Irecv(
                mpi4py_compatible_array, send_rank, tag
            )
            return mpi_future_object
        else:
            self.p2p_mpi_comm.Recv(mpi4py_compatible_array, send_rank, tag)

    def allreduce(self, tensor, async_op: bool = True):
        """Allreduce a PyTorch tensor using NCCL, GPUs in the
           self.coll_nccl_comm process group participate in the all-reduce

        Arguments:
            tensor (torch.Tensor): the PyTorch tensor to be all-reduced
            async_op (bool, optional): use asynchronous all-reduce
        """
        return torch.distributed.all_reduce(
            tensor, group=self.coll_nccl_comm, async_op=async_op
        )

    def broadcast_inter_layer(self, tensor, root):
        mpi4py_compatible_array = self._torch_to_mpi(tensor)
        self.p2p_mpi_comm.Bcast(mpi4py_compatible_array, root=root)
