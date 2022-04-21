# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from . import config
from typing import Optional, List, Tuple
from .communication import communication_handle
from .optim import CPUAdam
import torch
from mpi4py import MPI
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from enum import Enum
import numpy as np
import types

# True when init has been called
is_initialized = False
# Communication handle for point-to-point (MPI) and collective (NCCL) communication
comm_handle = None
# store references to input activation
input_tensors_cache = {}
# store references to output activation
output_tensors_cache = {}
# store (future object, tensor reference) for pending isends
transit_tensors = []
# store (future object, tensor reference) for pending irecvs
requests = {
    "fw": None,
    "bw": None,
}
# store reference to model shard
model = None
# loss function
criterion = None
# reference to flattened model params
model_params_fp32, model_params_fp16 = None, None
# reference to flattened model gradients
model_grads_fp32, model_grads_fp16 = None, None
fp32_optimizer = None
# the computation dtype (one of fp16/fp32)
computation_dtype = None
# fp16 all reduce, only applicable with mixed precision
_fp16_all_reduce = None
# loss_scale
loss_scale = 2.0**16
max_scale = 2.0**24
min_scale = 2.0**10
scaling_window = 200
no_overflow_iters = 0

_cpu_offload = False


class Operation(Enum):
    """
    AxoNNs enum class for the 2 microbatch operations - forward and backward pass
    """

    FW = 0
    BW = 1


class empty_dataset(torch.utils.data.Dataset):
    """
    Proxy dataset object for GPUs with inter_layer_parallel_rank > 0
    """

    def __init__(self, length: int, num_tensors: int):
        """Constructor for the proxy dataset class

        Arguments:
            length (int): number of datapoints in the dataset
            num_tensors (int): number of tensors per datapoint

        Returns:
            A PyTorch dataset object

        """
        self.length = length
        self.num_tensors = num_tensors

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = [0 for _ in range(self.num_tensors)]
        if self.num_tensors == 1:
            return torch.Tensor(data)
        else:
            return data


def init(
    G_inter: int,
    G_data: int,
    gpus_per_node: Optional[int] = None,
    mixed_precision=False,
    fp16_allreduce=True,
    cpu_offload=False,
) -> None:
    """
    Initialize AxoNN's 2D parallelism with G_inter-way inter-layer
    parallelism and G_data-way data parallelism

    Arguments:
        G_inter (int): number of GPUs used for inter-layer parallelism
        G_data (int): number of GPUs used for data parallelism
        gpus_per_node (int, optional):  number of GPUs per node, if not
            provided this is inferred using pytorch
        mixed_precision (bool): whether to use mixed precision
        fp16_allreduce (bool): invoke all reduce on fp16 parameters,
        only applicable when mixed precision is True
        cpu_offload (bool): offload optimizer states and fp32 parameters to
        the cpu to save gpu memory. Currently only works with
        mixed_precision, fp16_allreduce and axonn.optim.CPUAdam optimizer.

    """
    global comm_handle, is_initialized, computation_dtype, _fp16_all_reduce
    global _cpu_offload
    comm_handle = communication_handle(G_inter, G_data, gpus_per_node)
    config.G_inter = G_inter
    config.G_data = G_data
    config.inter_layer_parallel_rank = comm_handle.inter_layer_parallel_rank
    config.data_parallel_rank = comm_handle.data_parallel_rank
    is_initialized = True
    # assert mixed_precision, "Only supports mixed precision at apex O2 level"
    # assert fp16_allreduce, "Only supports fp-16 allreduce"
    if mixed_precision:
        computation_dtype = torch.float16
    else:
        computation_dtype = torch.float32
    _fp16_all_reduce = fp16_allreduce
    _cpu_offload = cpu_offload
    if comm_handle.world_rank == 0:
        print(f"Running with G_data={config.G_data} X G_inter={config.G_inter}")


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    micro_batch_size: int,
    num_workers: int = 0,
    *args,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create dataloaders for each GPU. For inter_layer_parallel_rank > 0,
    this creates a proxy dataloader which returns zero tensors.

    Arguments:
        dataset (torch.utils.data.Dataset): a PyTorch dataset object
        batch_size (int): batch size for dataloading
        micro_batch_size (int): microbatch size for inter-layer parallelism
        num_workers (int): number of worker processes in the dataloader

    Returns:
        data_loader (torch.utils.data.DataLoader): the dataloader object which
        is a true dataloader for inter_layer_parallel_rank = 0, else it is
        a proxy dataloader
    """
    assert is_initialized
    config.micro_batch_size = micro_batch_size
    config.batch_size = batch_size
    config.batch_size_per_network = batch_size // config.G_data
    assert (
        batch_size % (config.G_data * micro_batch_size) == 0
    ), "Batch Size should be divisible by the G_data*micro_batch_size"

    if config.inter_layer_parallel_rank == 0:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=config.G_data, rank=config.data_parallel_rank
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size_per_network,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            *args,
            **kwargs,
        )  # not working with drop_last=False

    else:
        num_tensors = 1 if torch.is_tensor(dataset[0]) else len(dataset[0])
        dataset = empty_dataset(len(dataset), num_tensors)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=config.G_data, rank=config.data_parallel_rank
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size_per_network,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
            drop_last=True,
        )
    return data_loader


def _coalesce_and_reassign(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Coalesce tensors into a flattened 1D tensor and reassign them to
    subtensors in this 1D tensor.

    TODO:- By creating a flat tensor first this doubles the gpu memory.
          Make this less memory consuming

    Arguments:
        tensors (List[torch.Tensor]): list of tensors to be coalesced

    Returns:
        flatenned_tensors (torch.tensor): the flattened tensor.

    """
    flattened_tensor = _flatten_dense_tensors(tensors)
    for old_tensor, new_tensor in zip(
        tensors, _unflatten_dense_tensors(flattened_tensor, tensors)
    ):
        old_tensor.data = new_tensor
    return flattened_tensor


def _initialize_mixed_precision(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Initialize mixed precision. Makes model parameters and gradients fp-16 and
    optimizer parameters as an fp-32 copy. Similar to Apex's O2 mode.
    Also flattens fp-32/fp-16 parameters and gradients for a bulk
    descaling and all-reduce.

    Arguments:
        model: model object on the GPU
        optimizer: the optimizer for the model

    Returns
        model: modified model object with fp-16 parameters and gradients
        optimizer : modified optimizer object with fp-32 parameters and gradients
    """
    global model_params_fp32, model_params_fp16, model_grads_fp32, model_grads_fp16
    assert (
        computation_dtype == torch.float16
    ), "call this method only for mixed precision"
    model = model.half()
    # now model and optimizer both point to fp16 weights
    # change optimizer to point to fp32 weights
    fp32_params = []
    fp16_params = []
    fp32_grads = []
    fp16_grads = []
    for group in optimizer.param_groups:
        for param_no, param in enumerate(group["params"]):
            assert (
                param.dtype == torch.float16
            ), "currently does not handle a mix of fp-16/fp-32"
            if param.requires_grad:
                fp16_params.append(param)
                param.grad = torch.zeros_like(param)
                fp16_grads.append(param.grad)
                fp32_param = param.detach().float()
                fp32_params.append(fp32_param)
                fp32_param.grad = torch.empty_like(fp32_param)
                fp32_grads.append(fp32_param.grad)
                group["params"][param_no] = fp32_param

    optimizer.load_state_dict(
        optimizer.state_dict()
    )  # trick to recast optimizer states

    model_params_fp32 = _coalesce_and_reassign(fp32_params)
    model_params_fp16 = _coalesce_and_reassign(fp16_params)
    model_grads_fp32 = _coalesce_and_reassign(fp32_grads)
    model_grads_fp16 = _coalesce_and_reassign(fp16_grads)

    return model, optimizer


def _initialize_full_precision(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Initialize full precision training - leaves model and optimizer untouched.
    Flattens fp-32 parameters and gradients.
    """
    global model_params_fp32, model_params_fp16, model_grads_fp32, model_grads_fp16
    assert (
        computation_dtype == torch.float32
    ), "call this method only for mixed precision"

    fp32_params = []
    fp32_grads = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            assert (
                param.dtype == torch.float32
            ), "currently does not handle a mix of fp-16/fp-32"
            if param.requires_grad:
                fp32_params.append(param)
                param.grad = torch.empty_like(param)
                fp32_grads.append(param.grad)

    model_params_fp32 = _coalesce_and_reassign(fp32_params)
    model_grads_fp32 = _coalesce_and_reassign(fp32_grads)
    model_grads_fp16 = None
    model_params_fp16 = None

    return model, optimizer


def _initialize_mixed_precision_with_cpu_offload(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Initialize mixed precision. Makes model parameters and gradients fp-16 and
    optimizer parameters as an fp-32 copy. Similar to Apex's O2 mode.
    Also flattens fp-32/fp-16 parameters and gradients for a bulk
    descaling and all-reduce.

    Arguments:
        model: model object on the GPU
        optimizer: the optimizer for the model

    Returns
        model: modified model object with fp-16 parameters and gradients
        optimizer : modified optimizer object with fp-32 parameters and gradients
    """
    global model_params_fp32, model_params_fp16, model_grads_fp32, model_grads_fp16
    assert (
        computation_dtype == torch.float16
    ), "CPU offload only supports mixed precision"
    assert _fp16_all_reduce, "CPU offload only supports fp-16 allreduce"
    assert isinstance(
        optimizer, CPUAdam
    ), "only AxoNN's implementation of Adam is supported"

    model = model.half()
    # now model and optimizer both point to fp16 weights
    # change optimizer to point to fp32 weights
    fp32_params = []
    fp16_params = []
    fp16_grads = []
    for group in optimizer.param_groups:
        for param_no, param in enumerate(group["params"]):
            assert (
                param.dtype == torch.float16
            ), "currently does not handle a mix of fp-16/fp-32"
            if param.requires_grad:
                fp16_params.append(param)
                param.grad = torch.zeros_like(param)
                fp16_grads.append(param.grad)
                # create fp32 parameters and move them to cpu
                fp32_param = param.detach().float().cpu()
                fp32_params.append(fp32_param)
                group["params"][param_no] = fp32_param

    optimizer.load_state_dict(
        optimizer.state_dict()
    )  # trick to recast optimizer states

    model_params_fp32 = _coalesce_and_reassign(fp32_params)
    model_params_fp16 = _coalesce_and_reassign(fp16_params)
    model_grads_fp16 = _coalesce_and_reassign(fp16_grads)

    return model, optimizer


@torch.no_grad()
def register_model_and_optimizer(model_shard, optimizer):
    """AxoNN's user facing function to register a model shard and
    the corresponding optimizer.

    Arguments:
        model_shard (torch.nn.Module): the model shard created by the
        user to be registered
        optimizer  (torch.nn.Optim): optimizer object for the model
    """
    global model, model_params_fp32, model_grads_fp32, model_params_fp16
    global model_grads_fp16, fp32_optimizer

    assert is_initialized

    model = model_shard
    if _cpu_offload:
        model, optimizer = _initialize_mixed_precision_with_cpu_offload(
            model, optimizer
        )
        model_params = model_params_fp16
    elif computation_dtype == torch.float16:
        model, optimizer = _initialize_mixed_precision(model, optimizer)
        model_params = model_params_fp16
    else:
        model, optimizer = _initialize_full_precision(model, optimizer)
        model_params = model_params_fp32

    comm_handle.allreduce(
        model_params.div_(config.G_data), async_op=False
    )  # sync all parameters across data parallel ranks

    if computation_dtype == torch.float16:
        model_params_fp32.copy_(model_params_fp16)

    fp32_optimizer = optimizer
    fp32_optimizer.skip_next_step = False

    unmodified_step = fp32_optimizer.step

    def modified_step(self):
        if not self.skip_next_step:
            unmodified_step()
            model_params_fp16.copy_(model_params_fp32)

    if computation_dtype == torch.float16 and not _cpu_offload:
        fp32_optimizer.step = types.MethodType(modified_step, fp32_optimizer)

    return model, optimizer


def register_loss_fn(loss_fn):
    """AxoNN's user facing function to register a loss function.

    Arguments:
        loss_fn: a PyTorch loss function (eg: torch.nn.CrossEntropy)
    """
    global criterion
    assert is_initialized
    criterion = loss_fn


def _get_subtensor(tensor, microbatch_no):
    """divide the tensor into equal tensors of micro_batch_size and
    retrieve the microbatch_no tensor. Useful when fetching data
    corresponding to a microbatch from a batch/labels.

    Arguments:
        tensor (torch.Tensor): tensor to be divided
    """
    start = microbatch_no * config.micro_batch_size
    end = (microbatch_no + 1) * config.micro_batch_size
    return tensor[start:end]


def print_status(*msg):
    """print msg

    Arguments:
        msg (str): message to be printed
    """

    print(
        f"DP Rank : {config.data_parallel_rank} |\
ILP Rank : {config.inter_layer_parallel_rank} - ",
        *msg,
    )


def _forward_pass(input_activation: torch.Tensor, microbatch_no: int, eval_mode: bool):
    """do the forward pass on an input activation and send the data to a forward GPU

    Arguments:
        input_activation (torch.Tensor): input activation from the previous GPU
        microbatch_no (int): the microbatch number of the input activation
        eval_mode (bool): true if evaluating the model for validation/testing

    """
    if eval_mode:
        with torch.no_grad():
            output_activation = model(input_activation)
        if config.inter_layer_parallel_rank == config.G_inter - 1:
            output_tensors_cache[microbatch_no] = output_activation
    else:
        output_activation = model(input_activation)
        input_tensors_cache[microbatch_no] = input_activation
        output_tensors_cache[microbatch_no] = output_activation
    if config.inter_layer_parallel_rank + 1 < config.G_inter:
        _send(output_activation, config.inter_layer_parallel_rank + 1, microbatch_no)


def _clear_transit_tensors(clear_all=False):
    """test pending isends for completion and delete tensors that have been sent

    Arguments:
        clear_all (bool): if true, return only after all isends have finished
    """
    global transit_tensors
    remaining_tensors = []
    for f, tensor in transit_tensors:
        if clear_all:
            f.Wait()
        elif not f.Test():
            remaining_tensors.append([f, tensor])
    transit_tensors = remaining_tensors


def _send(tensor: torch.Tensor, destination: int, tag: int):
    """send a tensor to a particular rank with a particular tag using MPI

    Arguments:
        tensor (torch.Tensor): tensor to be sent
        destination (int): inter-layer-parallel rank of the destination
        tag (int): tag of the message
    """
    if (destination < 0) or (destination >= config.G_inter):
        return
    _clear_transit_tensors()
    tensor = tensor.contiguous()
    torch.cuda.synchronize()
    transit_tensors.append([comm_handle.send(tensor, destination, tag), tensor])


def _fill_shape(shape):
    return [config.micro_batch_size if x == -1 else x for x in shape]


def _post_fw_recv_requests():
    """
    Post a receive request for a forward pass
    """
    if (requests["fw"] is None) and config.inter_layer_parallel_rank > 0:
        tensor = torch.empty(
            size=_fill_shape(model.get_input_shape()),
            device="cuda",
            dtype=computation_dtype,
        )
        tensor.requires_grad = True
        requests["fw"] = [
            tensor,
            comm_handle.recv(tensor, config.inter_layer_parallel_rank - 1),
        ]


def _post_bw_recv_requests():
    """
    Post a receive request for a backward pass
    """
    if (requests["bw"] is None) and (
        config.inter_layer_parallel_rank < config.G_inter - 1
    ):
        tensor = torch.empty(
            size=_fill_shape(model.get_output_shape()),
            device="cuda",
            dtype=computation_dtype,
        )
        requests["bw"] = [
            tensor,
            comm_handle.recv(tensor, config.inter_layer_parallel_rank + 1),
        ]


def _post_recv_requests(post_fw_recv=True, post_bw_recv=True):
    """
    post mpi irecv requests if they haven't been posted.
    """
    if post_fw_recv:
        _post_fw_recv_requests()
    if post_bw_recv:
        _post_bw_recv_requests()


def _recv(post_fw_recv=True, post_bw_recv=True, eval_mode=False) -> int:
    """
    Message driven scheduling of forward and backward passes for pipelining.

    Arguments:
        post_fw_recv(bool): Post a new receive request for a forward pass if needed
        post_bw_recv(bool): post a new receive request for a backward pass if needed
        eval_mode(bool): True if evaluating
    Returns:
        tag(int): the tag of the received message which is the microbatch number
    """
    status = MPI.Status()
    if (requests["bw"] is None) and (requests["fw"] is not None):
        requests["fw"][1].Wait(status)
        tag = status.Get_tag()
        input_activation = requests["fw"][0]
        requests["fw"] = None
        if post_fw_recv:
            _post_fw_recv_requests()
        _forward_pass(input_activation, tag, eval_mode)
        op = Operation.FW
    elif (requests["fw"] is None) and (requests["bw"] is not None):
        requests["bw"][1].Wait(status)
        tag = status.Get_tag()
        output_gradients = requests["bw"][0]
        requests["bw"] = None
        if post_bw_recv:
            _post_bw_recv_requests()
        _backward_pass(output_gradients, tag)
        op = Operation.BW
    else:
        index = MPI.Request.Waitany([requests["fw"][1], requests["bw"][1]], status)
        tag = status.Get_tag()
        if index == 0:  # forward pass
            input_activation = requests["fw"][0]
            requests["fw"] = None
            if post_fw_recv:
                _post_fw_recv_requests()
            _forward_pass(input_activation, tag, eval_mode)
            op = Operation.FW
        else:
            output_gradients = requests["bw"][0]
            requests["bw"] = None
            if post_bw_recv:
                _post_bw_recv_requests()
            _backward_pass(output_gradients, tag)
            op = Operation.BW
    return tag, op


def _calc_loss(microbatch_no, microbatch_labels, mul_factor=1.0, eval_mode=False):
    """Calculate the loss for a given microbatch number and its corresponding labels

    Arguments:
        microbatch_no (int): the microbatch number
        microbatch_labels (torch.Tensor): the true labels for the microbatch
        mul_factor (float): premultiply loss by this number
    """
    # for cross entropy calculation use float
    loss = criterion(output_tensors_cache[microbatch_no].float(), microbatch_labels)
    if computation_dtype == torch.float16:
        output_tensors_cache[microbatch_no] = (
            mul_factor * loss * loss_scale
        )  # scale up for mixed precision to
        # prevent underflow
    else:
        output_tensors_cache[microbatch_no] = mul_factor * loss
    if eval_mode:
        del output_tensors_cache[microbatch_no]
    return loss


def _backward_pass(output_gradients, microbatch_no):
    """do the backward pass of a microbatch and send the input activation gradients
    to the previous GPU.

    Arguments:
        output gradients (torch.Tensor): the gradient of the loss wrt the output tensor
        microbatch_no (int): the microbatch number
    """

    output_tensors_cache[microbatch_no].backward(output_gradients)
    input_tensor = input_tensors_cache[microbatch_no]
    del output_tensors_cache[microbatch_no]
    del input_tensors_cache[microbatch_no]
    if config.inter_layer_parallel_rank - 1 >= 0:
        _send(input_tensor.grad, config.inter_layer_parallel_rank - 1, microbatch_no)


def _sync_scale(local_overflow):
    global loss_scale, no_overflow_iters, max_scale
    assert computation_dtype == torch.float16
    overflow_np = np.array(int(local_overflow), "i")
    overflow_np_recv = np.array(int(local_overflow), "i")
    MPI.COMM_WORLD.Allreduce(
        [overflow_np, MPI.INT], [overflow_np_recv, MPI.INT], op=MPI.SUM
    )
    if overflow_np_recv > 0:
        loss_scale = max(loss_scale / 2.0, min_scale)
        if comm_handle.world_rank == 0:
            print_status(f"overflow detected - reducing loss scale to {loss_scale}")
        no_overflow_iters = 0
        global_overflow = True
    else:
        no_overflow_iters += 1
        if no_overflow_iters == scaling_window:
            loss_scale = min(loss_scale * 2.0, max_scale)
            if comm_handle.world_rank == 0:
                print_status(f"increasing loss scale to {loss_scale}")
            no_overflow_iters = 0
        global_overflow = False
    return global_overflow


def run_batch(batch: torch.Tensor, labels: torch.Tensor, eval_mode=False) -> int:
    """Perform forward and backward pass on a batch. This function invokes
    inter-layer-parallelism followed by an all-reduce.

    Arguments:
        batch (torch.Tensor): the input batch, for inter-layer-parallel-rank > 0
        this is a proxy tensor with the first dimension equal to the batch size
        labels (torch.Tensor): the true labels, for inter-layer-parallel-rank
        < G_inter-1, this can be None
        eval_mode (bool): set to true if you are doing validation/testing

    Returns:
        loss (float): the loss on the batch for inter-layer-parallel-rank
        == G_inter - 1, else 0
    """
    batch_loss = 0
    ilp_rank, G_inter, G_data = (
        config.inter_layer_parallel_rank,
        config.G_inter,
        config.G_data,
    )
    num_microbatches_per_network = batch.shape[0] // config.micro_batch_size

    if computation_dtype == torch.float16 and batch.dtype == torch.float32:
        batch = batch.half()

    if eval_mode:
        model.eval()
    else:
        model.train()

    if G_inter == 1:
        for microbatch_no in range(num_microbatches_per_network):
            _forward_pass(
                _get_subtensor(batch, microbatch_no), microbatch_no, eval_mode
            )
            microbatch_loss = _calc_loss(
                microbatch_no,
                _get_subtensor(labels, microbatch_no),
                1 / G_data / num_microbatches_per_network,
                eval_mode,
            )
            batch_loss += microbatch_loss.item()
            if not eval_mode:
                _backward_pass(None, microbatch_no)
    else:
        remaining_microbatches = num_microbatches_per_network
        num_msgs = remaining_microbatches
        if (ilp_rank != 0) and (ilp_rank != G_inter - 1):
            num_msgs += remaining_microbatches
            forward_msgs = backward_msgs = num_msgs // 2
        elif ilp_rank == 0:
            backward_msgs = num_msgs
            forward_msgs = 0
        else:
            forward_msgs = num_msgs
            backward_msgs = 0
        if eval_mode:
            num_msgs -= backward_msgs
            backward_msgs = 0
        next_microbatch = 0
        if ilp_rank == 0:
            for _ in range(G_inter):
                if remaining_microbatches == 0:
                    break
                _forward_pass(
                    _get_subtensor(batch, next_microbatch), next_microbatch, eval_mode
                )
                next_microbatch += 1
                remaining_microbatches -= 1

        _post_recv_requests(
            post_fw_recv=(forward_msgs > 1), post_bw_recv=(backward_msgs > 1)
        )
        while num_msgs:
            microbatch_no, op = _recv(
                post_fw_recv=(forward_msgs > 1),
                post_bw_recv=(backward_msgs > 1),
                eval_mode=eval_mode,
            )
            num_msgs -= 1
            if op == Operation.FW:
                forward_msgs -= 1
            elif op == Operation.BW:
                backward_msgs -= 1
            if ilp_rank == 0 and remaining_microbatches:  # inject next microbatch
                _forward_pass(
                    _get_subtensor(batch, next_microbatch), next_microbatch, eval_mode
                )
                next_microbatch += 1
                remaining_microbatches -= 1
            elif ilp_rank == G_inter - 1:
                microbatch_loss = _calc_loss(
                    microbatch_no,
                    _get_subtensor(labels, microbatch_no),
                    1 / G_data / num_microbatches_per_network,
                    eval_mode,
                )
                batch_loss += microbatch_loss.item()
                if not eval_mode:
                    _backward_pass(None, microbatch_no)

        _clear_transit_tensors(clear_all=True)
    if not _cpu_offload:
        _allreduce_and_descale()
    return batch_loss / num_microbatches_per_network


def _check_nan(tensor):
    """
    check a tensor for overflow

    Arguments:
        tensor (torch.Tensor): the tensor to be checked
    Return
        overflow (bool): true if there is overflow
    """
    sum_ = tensor.sum()
    return (torch.isinf(sum_) + torch.isnan(sum_)) > 0


def _allreduce_and_descale():
    """
    allreduce and descale the gradients in accoradance with mixed precision
    semantics. For fp-16_all_reduce mode, we first all-reduce and then descale
    to prevent underflow. Note that it is not possible to check for underflow
    so it is absolutely essential to maintain this order. For fp-32 all reduce
    mode, we first descale and then all-reduce. After descaling there cannot
    be underflow so this order is safe and prevents overflow.
    """
    # at this point for mixed precision we will have unscaled fp-16 gradients
    # for full precision we will have normal gradients
    with torch.no_grad():
        if computation_dtype == torch.float32:
            comm_handle.all_reduce(model_grads_fp32, async_op=False)
        else:
            if _fp16_all_reduce:
                # first all reduce then descale to prevent underflow
                comm_handle.allreduce(model_grads_fp16, async_op=False)
                model_grads_fp32.copy_(model_grads_fp16)
                model_grads_fp32.div_(loss_scale)
            else:
                # first descale then allreduce to precent overflow
                model_grads_fp32.copy_(model_grads_fp16)
                model_grads_fp32.div_(loss_scale)
                comm_handle.allreduce(model_grads_fp32, async_op=False)

            model_grads_fp16.zero_()
            local_overflow = _check_nan(model_grads_fp32)
            global_overflow = _sync_scale(local_overflow)
            fp32_optimizer.skip_next_step = global_overflow
