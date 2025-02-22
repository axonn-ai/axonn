import numpy as np
import torch.distributed as dist
from mpi4py import MPI
from typing import List, Union, Optional

class ProcessGroups:
    """
    A utility class to create and manage 2D Cartesian grids of process groups
    using MPI and NCCL backends.
    """
    def __init__(
        self,
        intra_group_size: int,
        inter_group_size: int,
        inner_group_backend: str = "nccl",
        outer_group_backend: str = "mpi"
    ):
        """
        Initialize the ProcessGroups by creating a 2D grid of process groups.

        Args:
            intra_group_size (int): Size of the inner (intra-group) dimension.
            inter_group_size (int): Size of the outer (inter-group) dimension.
            inner_group_backend (str, optional): Backend for the inner groups.
                Must be either "mpi" or "nccl". Defaults to "nccl".
            outer_group_backend (str, optional): Backend for the outer groups.
                Must be either "mpi" or "nccl". Defaults to "mpi".

        Raises:
            ValueError: If unsupported backends are provided or world size is incompatible.
            RuntimeError: If torch.distributed is not initialized when using NCCL backend.
        """
        # Validate backends
        # Create process groups
        assert dist.is_initialized(), "pytorch distributed should be initialized by the user"
        self.inner_group, self.outer_group = self.create_2D_grid(
            intra_group_size=intra_group_size,
            inter_group_size=inter_group_size,
            inner_group_backend=inner_group_backend,
            outer_group_backend=outer_group_backend
        )

    @staticmethod
    def create_2D_grid(
        intra_group_size: int,
        inter_group_size: int,
        inner_group_backend: str = "nccl",
        outer_group_backend: str = "mpi"
    ) -> List[Union[dist.ProcessGroup, MPI.Comm]]:
        """
        Create a 2D Cartesian grid of process groups.

        Each dimension of the grid can independently use either MPI or NCCL backends.

        Args:
            intra_group_size (int): Size of the inner (intra-group) dimension.
            inter_group_size (int): Size of the outer (inter-group) dimension.
            inner_group_backend (str, optional): Backend for the inner groups. 
                Must be either "mpi" or "nccl". Defaults to "nccl".
            outer_group_backend (str, optional): Backend for the outer groups. 
                Must be either "mpi" or "nccl". Defaults to "mpi".

        Returns:
            List[Union[dist.ProcessGroup, MPI.Comm]]: A list containing the 
                inner and outer process groups for the current process.
                - If the backend is "nccl", the group is a torch.distributed.ProcessGroup.
                - If the backend is "mpi", the group is an mpi4py.MPI.Comm object.
                The list has two elements: [inner_group, outer_group].

        Raises:
            ValueError: If unsupported backends are provided or world size is incompatible.
            RuntimeError: If torch.distributed is not initialized when using NCCL backend.
        """
        # Validate backends
        valid_backends = ["mpi", "nccl"]
        if inner_group_backend not in valid_backends:
            raise ValueError(f"Unsupported inner_group_backend '{inner_group_backend}'. Choose 'mpi' or 'nccl'.")
        if outer_group_backend not in valid_backends:
            raise ValueError(f"Unsupported outer_group_backend '{outer_group_backend}'. Choose 'mpi' or 'nccl'.")

        # Ensure torch.distributed is initialized if NCCL is used
        if inner_group_backend == "nccl" or outer_group_backend == "nccl":
            if not dist.is_initialized():
                raise RuntimeError("torch.distributed is not initialized. Please initialize it before creating groups.")

        # Get rank and world size from torch.distributed
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Validate world size
        if world_size % (intra_group_size * inter_group_size) != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by (intra_group_size * inter_group_size) "
                f"({intra_group_size * inter_group_size} = {intra_group_size * inter_group_size})."
            )

        num_2d_grids = world_size // (intra_group_size * inter_group_size)
        process_group_grid = np.arange(world_size).reshape(num_2d_grids, inter_group_size, intra_group_size)

        inner_group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None
        outer_group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None

        # Create Inner Group
        if inner_group_backend == "nccl":
            # Inner group using NCCL
            # Assuming inner_group_size corresponds to the last dimension (intra_group_size)
            # Iterate through all 2D grids and inter_group_size to create inner groups
            for i in range(num_2d_grids):
                for j in range(inter_group_size):
                    ranks = list(process_group_grid[i, j, :])
                    # Create NCCL group
                    this_inner_group = dist.new_group(ranks=ranks, backend="nccl")
                    if rank in ranks:
                        inner_group = this_inner_group
                        

        elif inner_group_backend == "mpi":
            color = rank // intra_group_size # unique color for each node 
            inner_group_comm = MPI.COMM_WORLD.Split(color)
            inner_group = inner_group_comm

        # Create Outer Group
        if outer_group_backend == "nccl":
            # Outer group using NCCL
            # Assuming outer_group_size corresponds to the first dimension (inter_group_size)
            for i in range(num_2d_grids):
                for j in range(intra_group_size):
                    ranks = list(process_group_grid[i, :, j])
                    # Create NCCL group
                    this_outer_group = dist.new_group(ranks=ranks, backend="nccl")
                    if rank in ranks:
                        outer_group = this_outer_group
                        

        elif outer_group_backend == "mpi":
            # second term advances the color by the number of process per node in each 2d group
            color = rank % intra_group_size + (rank // (intra_group_size * inter_group_size)) * (intra_group_size) 
            outer_group_comm = MPI.COMM_WORLD.Split(color)
            outer_group = outer_group_comm

        # Final Checks
        if inner_group is None:
            raise RuntimeError("Failed to create the inner group for the current process.")
        if outer_group is None:
            raise RuntimeError("Failed to create the outer group for the current process.")

        groups = [inner_group, outer_group]
        return groups
    
    def get_rank(self, group_index: Optional[int] = None) -> Union[int, List[int]]:
        """
        Get the rank of the current process within the specified process groups.

        Args:
            group_index (int, optional): 
                If None, returns a list [inner_rank, outer_rank].
                If 0, returns the rank in the inner group.
                If 1, returns the rank in the outer group.
                Defaults to None.

        Returns:
            Union[int, List[int]]: The rank(s) in the specified group(s).

        Raises:
            ValueError: If group_index is not 0 or 1.
        """
        if group_index is None:
            inner_rank = self._get_rank_internal(self.inner_group)
            outer_rank = self._get_rank_internal(self.outer_group)
            return [inner_rank, outer_rank]
        elif group_index == 0:
            return self._get_rank_internal(self.inner_group)
        elif group_index == 1:
            return self._get_rank_internal(self.outer_group)
        else:
            raise ValueError("group_index must be 0 (inner) or 1 (outer).")

    def get_world_size(self, group_index: Optional[int] = None) -> Union[int, List[int]]:
        """
        Get the world size of the specified process groups.

        Args:
            group_index (int, optional): 
                If None, returns a list [inner_world_size, outer_world_size].
                If 0, returns the world size of the inner group.
                If 1, returns the world size of the outer group.
                Defaults to None.

        Returns:
            Union[int, List[int]]: The world size(s) in the specified group(s).

        Raises:
            ValueError: If group_index is not 0 or 1.
        """
        if group_index is None:
            inner_size = self._get_world_size_internal(self.inner_group)
            outer_size = self._get_world_size_internal(self.outer_group)
            return [inner_size, outer_size]
        elif group_index == 0:
            return self._get_world_size_internal(self.inner_group)
        elif group_index == 1:
            return self._get_world_size_internal(self.outer_group)
        else:
            raise ValueError("group_index must be 0 (inner) or 1 (outer) or None (for both).")

    def _get_rank_internal(self, group: Union[dist.ProcessGroup, MPI.Comm]) -> int:
        """
        Internal method to get the rank within a process group.

        Args:
            group (Union[dist.ProcessGroup, MPI.Comm]): The process group.

        Returns:
            int: The rank within the group.
        """
        if isinstance(group, MPI.Comm):
            return group.Get_rank()
        elif isinstance(group, dist.ProcessGroup):
            return dist.get_rank(group=group)
        else:
            raise TypeError("Unsupported group type.")

    def _get_world_size_internal(self, group: Union[dist.ProcessGroup, MPI.Comm]) -> int:
        """
        Internal method to get the world size within a process group.

        Args:
            group (Union[dist.ProcessGroup, MPI.Comm]): The process group.

        Returns:
            int: The world size within the group.
        """
        if isinstance(group, MPI.Comm):
            return group.Get_size()
        elif isinstance(group, dist.ProcessGroup):
            return dist.get_world_size(group=group)
        else:
            raise TypeError("Unsupported group type.")

    def get_inner_group(self) -> Union[dist.ProcessGroup, MPI.Comm]:
        """
        Get the inner process group.

        Returns:
            Union[dist.ProcessGroup, MPI.Comm]: The inner process group.
        """
        return self.inner_group

    def get_outer_group(self) -> Union[dist.ProcessGroup, MPI.Comm]:
        """
        Get the outer process group.

        Returns:
            Union[dist.ProcessGroup, MPI.Comm]: The outer process group.
        """
        return self.outer_group

    def __repr__(self) -> str:
        inner_backend = "MPI" if isinstance(self.inner_group, MPI.Comm) else "NCCL"
        outer_backend = "MPI" if isinstance(self.outer_group, MPI.Comm) else "NCCL"
        return (
            f"<ProcessGroups inner_group_backend={inner_backend}, outer_group_backend={outer_backend}>"
        )
