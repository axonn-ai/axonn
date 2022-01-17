from . import config
from typing import Optional
from .communication import communication_handle
import torch
from mpi4py import MPI

is_initialized = False
comm_handle = None
input_tensors_cache = {}
output_tensors_cache = {}
transit_tensors = []
requests = {"fw" : None, "bw" : None}
model = None
criterion = None

def init(G_inter: int, G_data: int, micro_batch_size: int, batch_size: int, gpus_per_node: Optional[int] = None) -> None:
    global comm_handle, is_initialized
    comm_handle = communication_handle(G_inter, G_data, gpus_per_node)
    config.G_inter = G_inter
    config.G_data = G_data
    config.inter_layer_parallel_rank = comm_handle.inter_layer_parallel_rank
    config.data_parallel_rank = comm_handle.data_parallel_rank
    config.micro_batch_size = micro_batch_size
    config.batch_size = batch_size
    is_initialized = True
    if comm_handle.world_rank == 0:
        print(f"Running with G_data =  {config.G_data} X G_inter = {config.G_inter} | microbatch_size = {config.micro_batch_size} | batch_size = {config.batch_size}")
    print(f"Hello from ilp rank = {comm_handle.inter_layer_parallel_rank}, dp rank = {comm_handle.data_parallel_rank}")

def create_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int = 0) -> Optional[torch.utils.data.DataLoader]:
    assert is_initialized
    if config.data_parallel_rank!=0:
        return None
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=config.G_data, rank=config.data_parallel_rank)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler)

def register_model(model_shard):
    global model
    model = model_shard

def register_loss_fn(loss_fn):
    global criterion
    criterion = loss_fn

def _get_subtensor(tensor, microbatch_no):
    start = microbatch_no * config.micro_batch_size
    end = (microbatch_no+1) * config.micro_batch_size
    return tensor[start:end]

def print_status(msg):
    print(f"DP Rank : {config.data_parallel_rank} | ILP Rank : {config.inter_layer_parallel_rank} - {msg}")

def _forward_pass(input_activation: torch.Tensor, microbatch_no: int):
    #print_status(f"FW : {microbatch_no}")
    output_activation = model(input_activation)
    input_tensors_cache[microbatch_no] = input_activation
    output_tensors_cache[microbatch_no] = output_activation
    if config.inter_layer_parallel_rank + 1 < config.G_inter:
        _send(output_activation, config.inter_layer_parallel_rank + 1, microbatch_no)

def _clear_transit_tensors(clear_all=False):
    global transit_tensors
    remaining_tensors = []
    for f,tensor in transit_tensors:
        if clear_all:
            f.Wait()
        elif not f.Test():
            remaining_tensors.append([f, tensor])
    transit_tensors = remaining_tensors

def _send(tensor: torch.Tensor, destination: int, tag: int):
    if (destination < 0) or (destination >= config.G_inter):
        return
    _clear_transit_tensors()
    tensor = tensor.contiguous()
    torch.cuda.synchronize()
    transit_tensors.append([comm_handle.send(tensor, destination, tag), tensor])

def _post_recv_requests():
    if (requests["fw"] is None) and config.inter_layer_parallel_rank > 0:
        tensor = torch.cuda.FloatTensor(size=[config.micro_batch_size]+model.get_input_shape())
        tensor.requires_grad = True
        requests["fw"] = [tensor, comm_handle.recv(tensor, config.inter_layer_parallel_rank-1)]
    if (requests["bw"] is None) and (config.inter_layer_parallel_rank < config.G_inter - 1):
        tensor = torch.cuda.FloatTensor(size=[config.micro_batch_size]+model.get_output_shape())
        requests["bw"] = [tensor, comm_handle.recv(tensor, config.inter_layer_parallel_rank+1)]

def _recv() -> int:
    _post_recv_requests()
    status = MPI.Status()
    if config.inter_layer_parallel_rank == config.G_inter-1:
        requests["fw"][1].Wait(status)
        tag = status.Get_tag()
        input_activation = requests["fw"][0]
        requests["fw"] = None
        _forward_pass(input_activation, tag)
        return tag
    elif config.inter_layer_parallel_rank == 0:
        requests["bw"][1].Wait(status)
        tag = status.Get_tag()
        output_gradients = requests["bw"][0]
        requests["bw"] = None
        _backward_pass(output_gradients, tag)
        return tag
    else:
        index = MPI.Request.Waitany([requests["fw"][1], requests["bw"][1]], status)
        tag = status.Get_tag()
        if index == 0: #forward pass
            input_activation = requests["fw"][0]
            requests["fw"] = None
            _forward_pass(input_activation, tag)
        else:
            output_gradients = requests["bw"][0]
            requests["bw"] = None
            _backward_pass(output_gradients, tag)
        return tag

def _calc_loss(microbatch_no, microbatch_labels):
    return criterion(output_tensors_cache[microbatch_no], microbatch_labels)

def _backward_pass(output_gradients, microbatch_no):
    #print_status(f"BW : {microbatch_no}")
    output_tensors_cache[microbatch_no].backward(output_gradients)
    input_tensor = input_tensors_cache[microbatch_no]
    del output_tensors_cache[microbatch_no]
    del input_tensors_cache[microbatch_no]
    if config.inter_layer_parallel_rank - 1 >= 0:
        _send(input_tensor.grad, config.inter_layer_parallel_rank - 1, microbatch_no)

def run_batch(batch: torch.Tensor, labels: torch.Tensor):
    batch_loss = 0
    ilp_rank, dp_rank, G_inter, G_data = config.inter_layer_parallel_rank, config.data_parallel_rank, config.G_inter, config.G_data
    num_microbatches_in_batch = config.batch_size // config.micro_batch_size
    num_microbatches_per_network = num_microbatches_in_batch // G_data
    if dp_rank < num_microbatches_in_batch % G_data:
        num_microbatches_per_network += 1
    remaining_microbatches = num_microbatches_per_network
    num_msgs = remaining_microbatches
    if ((ilp_rank != 0) and (ilp_rank != G_inter-1)):
        num_msgs += remaining_microbatches

    next_microbatch = 0
    if ilp_rank == 0:
        for _ in range(G_inter):
            out = _forward_pass(_get_subtensor(batch, next_microbatch), next_microbatch)
            next_microbatch += 1
            remaining_microbatches -=1 
            if remaining_microbatches == 0:
                break

    while num_msgs:
        microbatch_no = _recv()
        num_msgs -= 1
        if ilp_rank == 0 and remaining_microbatches: #inject next microbatch
            _forward_pass(_get_subtensor(batch, next_microbatch), next_microbatch)
            next_microbatch += 1
            remaining_microbatches -= 1
        elif ilp_rank == G_inter -1:
            microbatch_loss = _calc_loss(microbatch_no, _get_subtensor(labels, microbatch_no))
            batch_loss += microbatch_loss.item()
            output_tensors_cache[microbatch_no] = microbatch_loss
            _backward_pass(None, microbatch_no)

    return batch_loss/num_microbatches_per_network

