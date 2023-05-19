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

    def __init__(
        self, G_inter: int, G_data: int, G_intra_r=1, G_intra_c=1, gpus_per_node=None
    ):
        """Constructor for the communication handle

        Arguments:
            G_inter (int): number of GPUs used for inter-layer parallelism
            G_data (int): number of GPUs used for data parallelism
            gpus_per_node (int, optional): number of GPUs per node, if not
            provided this is inferred using pytorch
            G_intra (int): degree of intra-layer parallelism. Note that
            the user is supposed to implement their intra-layer parallel
            kernels. AxoNN will just create communicationgroups for
            intra-layer parallelism
        """
        self.world_rank = MPI.COMM_WORLD.Get_rank()
        self.world_size = MPI.COMM_WORLD.Get_size()
        G_intra = G_intra_r * G_intra_c
        assert (
            G_inter * G_data * G_intra == self.world_size
        ), "The product of G_inter and G_data should be equal to the number of GPUs"
        self.G_intra = G_intra
        self.G_inter = G_inter
        self.G_data = G_data
        self.G_intra_r = G_intra_r
        self.G_intra_c = G_intra_c

        # infer gpus per node if not provided
        self.gpus_per_node = (
            gpus_per_node if gpus_per_node is not None else torch.cuda.device_count()
        )
        self.local_rank = self.world_rank % self.gpus_per_node
        torch.cuda.set_device(self.local_rank)
        self.intra_layer_parallel_rank = self.world_rank % G_intra
        self.intra_layer_column_parallel_rank = (
            self.intra_layer_parallel_rank % G_intra_c
        )
        self.intra_layer_row_parallel_rank = (
            self.intra_layer_parallel_rank // G_intra_c
        )
        self.inter_layer_parallel_rank = (self.world_rank // G_intra) % G_inter
        self.data_parallel_rank = self.world_rank // (G_inter * G_intra)

        # create communicator for point-to-point(MPI) communication
        colour = self.intra_layer_parallel_rank + G_intra * self.data_parallel_rank
        self.p2p_mpi_comm = MPI.COMM_WORLD.Split(colour)
        assert self.p2p_mpi_comm.Get_size() == G_inter
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

        if self.world_rank == 0:
            print("Creating data parallel groups")
        for i in range(G_inter):
            for j in range(G_intra):
                # all ranks have to form all data parallel communicators and not
                # just their own
                ranks_in_ith_jth_data_parallel_group = [
                    k * self.G_inter * self.G_intra + i * G_intra + j
                    for k in range(self.G_data)
                ]
                if self.world_rank == 0:
                    print(ranks_in_ith_jth_data_parallel_group)
                ith_jth_data_parallel_group = torch.distributed.new_group(
                    ranks=ranks_in_ith_jth_data_parallel_group, backend="nccl"
                )
                if self.world_rank in ranks_in_ith_jth_data_parallel_group:
                    self.coll_nccl_comm = ith_jth_data_parallel_group

        if self.world_rank == 0:
            print("Creating intra-layer parallel groups")
        # create communicators for intra-layer parallelism
        for i in range(G_data):
            for j in range(G_inter):
                ranks_in_ith_jth_intra_layer_group = [
                    i * G_inter * G_intra + j * G_intra + k for k in range(G_intra)
                ]

                ith_jth_intra_layer_group = torch.distributed.new_group(
                    ranks=ranks_in_ith_jth_intra_layer_group, backend="nccl"
                )
                if self.world_rank in ranks_in_ith_jth_intra_layer_group:
                    self.intra_layer_group = ith_jth_intra_layer_group
                # form row and column tensor parallel groups
                # G_intra_r x G_intra_c
                assert len(ranks_in_ith_jth_intra_layer_group) == G_intra_r * G_intra_c
                intra_layer_ranks = ranks_in_ith_jth_intra_layer_group
                for i in range(G_intra_r):
                    offset = i * G_intra_c
                    group_members = intra_layer_ranks[offset : offset + G_intra_c]
                    group = torch.distributed.new_group(
                        ranks=group_members, backend="nccl"
                    )
                    if self.world_rank == 0:
                        print(f"Inner TP group = {group_members}")
                    if self.world_rank in group_members:
                        self.inner_intra_layer_parallel_group = group

                for i in range(G_intra_c):
                    group_members = intra_layer_ranks[i::G_intra_c]
                    group = torch.distributed.new_group(
                        ranks=group_members, backend="nccl"
                    )
                    if self.world_rank == 0:
                        print(f"Outer TP group = {group_members}")
                    if self.world_rank in group_members:
                        self.outer_intra_layer_parallel_group = group

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

    def get_tensor_model_parallel_rank(self):
        return self.intra_layer_parallel_rank

    def get_tensor_model_parallel_world_size(self):
        return self.G_intra

    def get_tensor_model_parallel_group(self):
        return self.intra_layer_group
