# Copyright 2021-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

try:
    import mpi4py

    MPI4PY = True
    mpi4py.rc.initialize = False  # do not initialize MPI automatically
    from mpi4py import MPI
except ImportError:
    MPI4PY = False
import torch
import numpy as np


class communication_handle:
    """
    Communnication handle for point-to-point(MPI) and collective
    communication(NCCL) of GPU tensors.
    """

    def __init__(
        self,
        G_inter: int,
        G_data: int,
        G_intra_r=1,
        G_intra_c=1,
        G_intra_d=1,
        gpus_per_node=None,
    ):
        """Constructor for the communication handle

        Arguments:
            G_inter (int): number of GPUs used for inter-layer parallelism
            G_data (int): number of GPUs used for data parallelism
            gpus_per_node (int, optional): number of GPUs per node, if not
            provided this is inferred using pytorch
            G_intra_r (int): number of GPUs in the row intra-layer parallel dimension
            G_intra_c (int): number of GPUs in the column intra-layer parallel dimension
            G_intra_d (int): number of GPUs in the depth intra-layer parallel dimension
        """
        if not torch.distributed.is_initialized():
            assert MPI4PY, "either install mpi4py and launch via mpirun/srun"
            "or initialize torch.distributed outside axonn"
            if not MPI.Is_initialized():
                MPI.Init()
            self.world_rank = MPI.COMM_WORLD.Get_rank()
            self.world_size = MPI.COMM_WORLD.Get_size()
        else:
            self.world_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        G_intra = G_intra_r * G_intra_c * G_intra_d
        assert (
            G_inter * G_data * G_intra == self.world_size
        ), "The product of G_inter, G_intra_r, G_intra_c, G_intra_d,"
        f"G_data should be equal to the number of GPUs - {self.world_size}"
        self.G_intra = G_intra
        self.G_inter = G_inter
        self.G_data = G_data
        self.G_intra_r = G_intra_r
        self.G_intra_c = G_intra_c
        self.G_intra_d = G_intra_d

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
        ) % G_intra_r
        self.intra_layer_depth_parallel_rank = self.intra_layer_parallel_rank // (
            G_intra_c * G_intra_r
        )

        self.inter_layer_parallel_rank = (self.world_rank // G_intra) % G_inter
        self.data_parallel_rank = self.world_rank // (G_inter * G_intra)

        # create communicator for point-to-point(MPI) communication
        colour = self.intra_layer_parallel_rank + G_intra * self.data_parallel_rank

        if G_inter > 1:
            # this needs to be checked
            if MPI4PY:
                if not MPI.Is_initialized():
                    MPI.Init()
                self.p2p_mpi_comm = MPI.COMM_WORLD.Split(colour)
                assert self.p2p_mpi_comm.Get_size() == G_inter
            else:
                self.p2p_mpi_comm = None
                print(
                    "Warning: AxoNN's implementation of inter-layer"
                    "parallelism (pipelining) requires mpi4py, which wasn't found."
                    "You will have to use an external implementation"
                    "of pipeline parallelism."
                )
        else:
            self.p2p_mpi_comm = None

        # create communicator for collective (NCCL) communication
        if not torch.distributed.is_initialized():
            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "29500")
            init_method += master_ip + ":" + master_port
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=self.world_size,
                rank=self.world_rank,
                init_method=init_method,
            )

        for i in range(G_inter):
            for j in range(G_intra):
                # all ranks have to form all data parallel communicators and not
                # just their own
                ranks_in_ith_jth_data_parallel_group = [
                    k * self.G_inter * self.G_intra + i * G_intra + j
                    for k in range(self.G_data)
                ]
                ith_jth_data_parallel_group = torch.distributed.new_group(
                    ranks=ranks_in_ith_jth_data_parallel_group, backend="nccl"
                )
                if self.world_rank in ranks_in_ith_jth_data_parallel_group:
                    self.coll_nccl_comm = ith_jth_data_parallel_group
                    self.data_parallel_group = ith_jth_data_parallel_group

        # create communicators for intra-layer parallelism
        for i_ in range(G_data):
            for j_ in range(G_inter):
                ranks_in_ith_jth_intra_layer_group = [
                    i_ * G_inter * G_intra + j_ * G_intra + k for k in range(G_intra)
                ]
                ith_jth_intra_layer_group = torch.distributed.new_group(
                    ranks=ranks_in_ith_jth_intra_layer_group, backend="nccl"
                )
                if self.world_rank in ranks_in_ith_jth_intra_layer_group:
                    self.intra_layer_group = ith_jth_intra_layer_group

                assert (
                    len(ranks_in_ith_jth_intra_layer_group)
                    == G_intra_r * G_intra_c * G_intra_d
                )

                ranks_in_ith_jth_intra_layer_group = np.array(
                    ranks_in_ith_jth_intra_layer_group
                ).reshape(G_intra_d, G_intra_r, G_intra_c)
                # form row and column tensor parallel groups
                # G_intra_d x G_intra_r x G_intra_c

                # inner
                for i in range(G_intra_d):
                    for j in range(G_intra_r):
                        group_members = list(
                            ranks_in_ith_jth_intra_layer_group[i, j, :]
                        )
                        group = torch.distributed.new_group(
                            ranks=group_members, backend="nccl"
                        )
                        if self.world_rank in group_members:
                            self.inner_intra_layer_parallel_group = group

                # outer
                for i in range(G_intra_d):
                    for j in range(G_intra_c):
                        group_members = list(
                            ranks_in_ith_jth_intra_layer_group[i, :, j]
                        )
                        group = torch.distributed.new_group(
                            ranks=group_members, backend="nccl"
                        )
                        if self.world_rank in group_members:
                            self.outer_intra_layer_parallel_group = group

                # depth
                for i in range(G_intra_r):
                    for j in range(G_intra_c):
                        group_members = list(
                            ranks_in_ith_jth_intra_layer_group[:, i, j]
                        )
                        group = torch.distributed.new_group(
                            ranks=group_members, backend="nccl"
                        )
                        if self.world_rank in group_members:
                            self.depth_intra_layer_parallel_group = group

                # combined inner+outer
                for i in range(G_intra_d):
                    group_members = list(
                        ranks_in_ith_jth_intra_layer_group[i, :, :].flatten()
                    )
                    group = torch.distributed.new_group(
                        ranks=group_members, backend="nccl"
                    )
                    if self.world_rank in group_members:
                        self.outer_inner_intra_layer_parallel_group = group
                        self.outer_inner_intra_layer_parallel_group_root = (
                            group_members[0]
                        )

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
        tag: int = None,
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
        if tag is None:
            tag = MPI.ANY_TAG
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
