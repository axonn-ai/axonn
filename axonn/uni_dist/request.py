import torch.distributed as dist
from mpi4py import MPI
from typing import Callable, Optional, Any, List, Union


class Request:
    """
    A unified Request class for managing non-blocking communication requests
    from torch.distributed and mpi4py backends.
    """

    def __init__(self, request: Any):
        """
        Initialize the Request object by determining the backend based on the request's class type.

        Args:
            request: The underlying request object from the backend.
                     It must be either a torch.distributed.Work or an mpi4py.MPI.Request.

        Raises:
            ValueError: If the request type is unsupported.
        """
        if isinstance(request, dist.Work):
            self.backend = 'torch'
            self.request = request
        elif isinstance(request, MPI.Request):
            self.backend = 'mpi'
            self.request = request
        else:
            raise ValueError(
                f"Unsupported request type: {type(request)}. "
                "Expected torch.distributed.Work or mpi4py.MPI.Request."
            )

    def wait(self, epilogue: Optional[Callable[[Any], None]] = None) -> Optional[Any]:
        """
        Wait for the request to complete and optionally execute an epilogue function.

        Args:
            epilogue (Callable[[Any], None], optional): 
                A function to be executed after the wait completes.
                It receives the result of the wait as its argument.
                Defaults to None.

        Returns:
            Depends on the backend:
                - torch.distributed: Returns None.
                - mpi4py: Returns an MPI.Status object.

        Raises:
            ValueError: If the backend is unsupported.
        """
        if self.backend == 'torch':
            # Wait for the torch.distributed request to complete
            self.request.wait()
            result = None
        elif self.backend == 'mpi':
            # Wait for the mpi4py request to complete and get the status
            status = MPI.Status()
            self.request.Wait(status)
            result = status
        else:
            raise ValueError("Unsupported backend.")

        # Execute the epilogue function if provided
        if epilogue is not None:
            if not callable(epilogue):
                raise TypeError("Epilogue must be a callable.")
            try:
                epilogue()
            except Exception as e:
                # Handle or propagate the exception as needed
                print(f"An error occurred in the epilogue: {e}")
                raise

        return result

   
    def __repr__(self) -> str:
        return f"<Request backend={self.backend} request={self.request}>"
