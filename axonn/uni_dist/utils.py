import torch 
from mpi4py import MPI 

def _torch_to_mpi(tensor: torch.Tensor):
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