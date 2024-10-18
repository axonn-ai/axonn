# Copyright 2021-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# from . import models  # noqa: F401


from enum import Enum
from dataclasses import dataclass
from axonn import axonn as ax
from mpi4py import MPI
from axonn.intra_layer import sync_gradients

import torch
import numpy as np


@dataclass
class LossScaler:
    """
    Dataclass for scaling the loss for fp-16 training
    """

    loss_scale: float = 2.0**16
    max_scale: float = 2.0**24
    min_scale: float = 2.0**10
    scaling_window: float = 200
    no_overflow_iters: float = 0


class Operation(Enum):
    """
    AxoNNs enum class for the 2 microbatch operations - forward and backward pass
    """

    FW = 0
    BW = 1


class AxoNN_Inter_Layer_Engine:
    def __init__(self, model, loss_fn, computation_dtype=torch.float16):
        assert (
            ax.is_initialized
        ), "Please call ax.init(....) before calling AxoNNPipelineEngine"
        self.model = model
        self.criterion = loss_fn

        # store references to input activation
        self.input_tensors_cache = {}
        # store references to output activation
        self.output_tensors_cache = {}
        # store (future object, tensor reference) for pending isends
        self.transit_tensors = []
        # store (future object, tensor reference) for pending irecvs
        self.requests = {
            "fw": None,
            "bw": None,
        }

        self.computation_dtype = computation_dtype
        self.scaler = LossScaler()

    def _get_subtensor(self, tensor, microbatch_no):
        """divide the tensor into equal tensors of micro_batch_size and
        retrieve the microbatch_no tensor. Useful when fetching data
        corresponding to a microbatch from a batch/labels.

        Arguments:
            tensor (torch.Tensor): tensor to be divided
        """
        start = microbatch_no * ax.config.micro_batch_size
        end = (microbatch_no + 1) * ax.config.micro_batch_size
        return tensor[start:end]

    def _forward_pass(
        self, input_activation: torch.Tensor, microbatch_no: int, eval_mode: bool
    ):
        """do the forward pass on an input activation and send the data to a forward GPU

        Arguments:
            input_activation (torch.Tensor): input activation from the previous GPU
            microbatch_no (int): the microbatch number of the input activation
            eval_mode (bool): true if evaluating the model for validation/testing

        """
        with torch.autocast(device_type="cuda", dtype=self.computation_dtype):
            if eval_mode:
                with torch.no_grad():
                    output_activation = self.model(input_activation)
                if ax.config.inter_layer_parallel_rank == ax.config.G_inter - 1:
                    self.output_tensors_cache[microbatch_no] = output_activation
            else:
                output_activation = self.model(input_activation)
                self.input_tensors_cache[microbatch_no] = input_activation
                self.output_tensors_cache[microbatch_no] = output_activation
            if ax.config.inter_layer_parallel_rank + 1 < ax.config.G_inter:
                self._send(
                    output_activation,
                    ax.config.inter_layer_parallel_rank + 1,
                    microbatch_no,
                )

    def _send(self, tensor: torch.Tensor, destination: int, tag: int):
        """send a tensor to a particular rank with a particular tag using MPI

        Arguments:
            tensor (torch.Tensor): tensor to be sent
            destination (int): inter-layer-parallel rank of the destination
            tag (int): tag of the message
        """
        if (destination < 0) or (destination >= ax.config.G_inter):
            return
        self._clear_transit_tensors()
        tensor = tensor.contiguous().to(self.computation_dtype)
        torch.cuda.synchronize()  # TODO - replace with stream synchronize.
        self.transit_tensors.append(
            [ax.comm_handle.send(tensor, destination, tag), tensor]
        )

    def _clear_transit_tensors(self, clear_all=False):
        """test pending isends for completion and delete tensors that have been sent
        Arguments:
            clear_all (bool): if true, return only after all isends have finished
        """
        remaining_tensors = []
        for f, tensor in self.transit_tensors:
            if clear_all:
                f.Wait()
            elif not f.Test():
                remaining_tensors.append([f, tensor])
        self.transit_tensors = remaining_tensors

    def _fill_shape(self, shape):
        return [ax.config.micro_batch_size if x == -1 else x for x in shape]

    def _post_fw_recv_requests(self):
        """
        Post a receive request for a forward pass
        """
        if (self.requests["fw"] is None) and ax.config.inter_layer_parallel_rank > 0:
            tensor = torch.empty(
                size=self._fill_shape(self.model.get_input_shape()),
                device="cuda",
                dtype=self.computation_dtype,
            )
            tensor.requires_grad = True
            self.requests["fw"] = [
                tensor,
                ax.comm_handle.recv(tensor, ax.config.inter_layer_parallel_rank - 1),
            ]

    def _post_bw_recv_requests(self):
        """
        Post a receive request for a backward pass
        """
        if (self.requests["bw"] is None) and (
            ax.config.inter_layer_parallel_rank < ax.config.G_inter - 1
        ):
            tensor = torch.empty(
                size=self._fill_shape(self.model.get_output_shape()),
                device="cuda",
                dtype=self.computation_dtype,
            )
            self.requests["bw"] = [
                tensor,
                ax.comm_handle.recv(tensor, ax.config.inter_layer_parallel_rank + 1),
            ]

    def _post_recv_requests(self, post_fw_recv=True, post_bw_recv=True):
        """
        post mpi irecv requests if they haven't been posted.
        """
        if post_fw_recv:
            self._post_fw_recv_requests()
        if post_bw_recv:
            self._post_bw_recv_requests()

    def _recv(self, post_fw_recv=True, post_bw_recv=True, eval_mode=False) -> int:
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
        if (self.requests["bw"] is None) and (self.requests["fw"] is not None):
            self.requests["fw"][1].Wait(status)
            tag = status.Get_tag()
            input_activation = self.requests["fw"][0]
            self.requests["fw"] = None
            if post_fw_recv:
                self._post_fw_recv_requests()
            self._forward_pass(input_activation, tag, eval_mode)
            op = Operation.FW
        elif (self.requests["fw"] is None) and (self.requests["bw"] is not None):
            self.requests["bw"][1].Wait(status)
            tag = status.Get_tag()
            output_gradients = self.requests["bw"][0]
            self.requests["bw"] = None
            if post_bw_recv:
                self._post_bw_recv_requests()
            self._backward_pass(output_gradients, tag)
            op = Operation.BW
        else:
            index = MPI.Request.Waitany(
                [self.requests["fw"][1], self.requests["bw"][1]], status
            )
            tag = status.Get_tag()
            if index == 0:  # forward pass
                input_activation = self.requests["fw"][0]
                self.requests["fw"] = None
                if post_fw_recv:
                    self._post_fw_recv_requests()
                self._forward_pass(input_activation, tag, eval_mode)
                op = Operation.FW
            else:
                output_gradients = self.requests["bw"][0]
                self.requests["bw"] = None
                if post_bw_recv:
                    self._post_bw_recv_requests()
                self._backward_pass(output_gradients, tag)
                op = Operation.BW
        return tag, op

    def _calc_loss(
        self, microbatch_no, microbatch_labels, mul_factor=1.0, eval_mode=False
    ):
        """Calculate the loss for a given microbatch number and its corresponding labels

        Arguments:
            microbatch_no (int): the microbatch number
            microbatch_labels (torch.Tensor): the true labels for the microbatch
            mul_factor (float): premultiply loss by this number
        """
        # for cross entropy calculation use float
        loss = self.criterion(
            self.output_tensors_cache[microbatch_no].float(), microbatch_labels
        )
        if self.computation_dtype == torch.float16:
            self.output_tensors_cache[microbatch_no] = (
                mul_factor * loss * self.scaler.loss_scale
            )  # scale up for mixed precision to
            # prevent underflow
        else:
            self.output_tensors_cache[microbatch_no] = mul_factor * loss
        if eval_mode:
            del self.output_tensors_cache[microbatch_no]
        return loss

    def _backward_pass(self, output_gradients, microbatch_no):
        """do the backward pass of a microbatch and send the input activation gradients
        to the previous GPU.

        Arguments:
            output gradients (torch.Tensor): the gradient of the loss wrt the
            output tensor
            microbatch_no (int): the microbatch number
        """
        self.output_tensors_cache[microbatch_no].backward(output_gradients)
        input_tensor = self.input_tensors_cache[microbatch_no]
        del self.output_tensors_cache[microbatch_no]
        del self.input_tensors_cache[microbatch_no]
        if ax.config.inter_layer_parallel_rank - 1 >= 0:
            self._send(
                input_tensor.grad,
                ax.config.inter_layer_parallel_rank - 1,
                microbatch_no,
            )

    def _sync_scale(self, local_overflow):
        assert self.computation_dtype == torch.float16
        overflow_np = np.array(int(local_overflow), "i")
        overflow_np_recv = np.array(int(local_overflow), "i")
        MPI.COMM_WORLD.Allreduce(
            [overflow_np, MPI.INT], [overflow_np_recv, MPI.INT], op=MPI.SUM
        )
        if overflow_np_recv > 0:
            self.scaler.loss_scale = max(
                self.scaler.loss_scale / 2.0, self.scaler.min_scale
            )
            if ax.comm_handle.world_rank == 0:
                print(
                    f"overflow detected - reducing loss scale"
                    f"to {self.scaler.loss_scale}"
                )
            self.scaler.no_overflow_iters = 0
            global_overflow = True
        else:
            self.scaler.no_overflow_iters += 1
            if self.scaler.no_overflow_iters == self.scaler.scaling_window:
                self.scaler.loss_scale = min(
                    self.scaler.loss_scale * 2.0, self.scaler.max_scale
                )
                if ax.comm_handle.world_rank == 0:
                    print(f"increasing loss scale to {self.scaler.loss_scale}")
                self.scaler.no_overflow_iters = 0
            global_overflow = False
        return global_overflow

    def forward_backward_optimizer(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        eval_mode=False,
        post_bw_hook=None,
    ) -> int:
        """Perform forward pass, backward pass and optimizer step on a batch.

        Arguments:
            batch (torch.Tensor): the input batch
            labels (torch.Tensor): the true labels
            eval_mode (bool): set to true if you are doing validation/testing

        Returns:
            loss (float): the loss on the batch for inter-layer-parallel-rank
            == G_inter - 1, else 0
        """
        batch_loss = 0
        ilp_rank, G_inter = (
            ax.config.inter_layer_parallel_rank,
            ax.config.G_inter,
        )
        num_microbatches_per_gpu = batch.shape[0] // ax.config.micro_batch_size

        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

        if G_inter == 1:
            for microbatch_no in range(num_microbatches_per_gpu):
                self._forward_pass(
                    self._get_subtensor(batch, microbatch_no), microbatch_no, eval_mode
                )
                microbatch_loss = self._calc_loss(
                    microbatch_no,
                    self._get_subtensor(labels, microbatch_no),
                    1 / num_microbatches_per_gpu,
                    eval_mode,
                )
                batch_loss += microbatch_loss.item()
                if not eval_mode:
                    self._backward_pass(None, microbatch_no)
        else:
            remaining_microbatches = num_microbatches_per_gpu
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
                    self._forward_pass(
                        self._get_subtensor(batch, next_microbatch),
                        next_microbatch,
                        eval_mode,
                    )
                    next_microbatch += 1
                    remaining_microbatches -= 1
            self._post_recv_requests(
                post_fw_recv=(forward_msgs > 1), post_bw_recv=(backward_msgs > 1)
            )
            while num_msgs:
                microbatch_no, op = self._recv(
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
                    self._forward_pass(
                        self._get_subtensor(batch, next_microbatch),
                        next_microbatch,
                        eval_mode,
                    )
                    next_microbatch += 1
                    remaining_microbatches -= 1
                elif ilp_rank == G_inter - 1:
                    microbatch_loss = self._calc_loss(
                        microbatch_no,
                        self._get_subtensor(labels, microbatch_no),
                        1 / num_microbatches_per_gpu,
                        eval_mode,
                    )
                    batch_loss += microbatch_loss.item()
                    if not eval_mode:
                        self._backward_pass(None, microbatch_no)

            if eval_mode and ilp_rank == 0:
                while remaining_microbatches:
                    while len(self.transit_tensors) == G_inter:
                        self._clear_transit_tensors()
                    self._forward_pass(
                        self._get_subtensor(batch, next_microbatch),
                        next_microbatch,
                        eval_mode,
                    )
                    next_microbatch += 1
                    remaining_microbatches -= 1

            self._clear_transit_tensors(clear_all=True)
        if post_bw_hook is not None:
            assert not eval_mode
            post_bw_hook(self.model)

        sync_gradients(self.model, mean=True, expert_mode=True)
        if self.computation_dtype == torch.float16:
            global_overflow = self._unscale_gradients()
            if not global_overflow:
                optimizer.step()
        else:
            optimizer.step()
        return batch_loss / num_microbatches_per_gpu

    def _check_nan(self, tensor):
        """
        check a tensor for overflow

        Arguments:
            tensor (torch.Tensor): the tensor to be checked
        Return
            overflow (bool): true if there is overflow
        """
        sum_ = tensor.sum()
        return (torch.isinf(sum_) + torch.isnan(sum_)) > 0

    def _unscale_gradients(self):
        """
        unscale the gradients and check for overflow across all GPUs
        """
        # at this point for mixed precision we will have unscaled fp-16 gradients
        # for full precision we will have normal gradients
        local_overflow = False
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    local_overflow = local_overflow or self._check_nan(p.grad)
                    p.grad.div_(self.scaler.loss_scale)
            global_overflow = self._sync_scale(local_overflow)
            return global_overflow
