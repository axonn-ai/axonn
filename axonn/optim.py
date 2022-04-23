# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch.optim.optimizer import Optimizer
from . import axonn as ax
import torch.optim._functional as F


class CPUAdam(Optimizer):
    """
    AxoNN's memory optimized implementation of Adam
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        bucket_size=16000000,
        coalescing_factor=4,
    ):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        assert ax.computation_dtype == torch.float16
        assert ax._fp16_all_reduce

        self.bucket_size = bucket_size
        self.coalescing_factor = coalescing_factor

        super(CPUAdam, self).__init__(params, defaults)

        self.calculate_group_offsets()
        self.state_pipeline = [[] for _ in range(len(self.param_groups))]

        self.stream = torch.cuda.Stream()

        self.buffers = [torch.cuda.FloatTensor(self.bucket_size) for _ in range(4)]

    def __setstate__(self, state):
        super(CPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def calculate_group_offsets(self):
        self.group_offsets = [0] * (len(self.param_groups) + 1)
        for i in range(1, len(self.group_offsets)):
            group_num_params = 0
            for param in self.param_groups[i - 1]["params"]:
                group_num_params += param.nelement()
            self.group_offsets[i] = self.group_offsets[i - 1] + group_num_params

    def empty_param_state(self, param):
        state = {
            "step": 0,
            "exp_avg": torch.zeros_like(
                param, memory_format=torch.preserve_format
            ).pin_memory(),
            "exp_avg_sq": torch.zeros_like(
                param, memory_format=torch.preserve_format
            ).pin_memory(),
        }

        return state

    def zero_grad(self):
        ax.model_grads_fp16.zero_()

    def step(self, closure=None):
        assert closure is None, "AxoNN CPUAdam does not support closure yet"
        stream = self.stream
        bucket_size = self.bucket_size
        flat_master_params = ax.model_params_fp32
        flat_fp16_grad = ax.model_grads_fp16
        flat_fp16_params = ax.model_params_fp16
        exp_avg_buffer, exp_avg_sq_buffer, param_buffer, grad_buffer = self.buffers

        skip_update = False

        for group_no in range(len(self.group_offsets) - 1):
            # Step 1 - Lazy init adam states on CPU for each chunk
            needs_init = len(self.state_pipeline[group_no]) == 0
            start_index = self.group_offsets[group_no]
            chunk_no = 0
            nccl_events = []
            while start_index != self.group_offsets[group_no + 1]:
                end_index = min(
                    start_index + bucket_size, self.group_offsets[group_no + 1]
                )
                if needs_init:
                    self.state_pipeline[group_no].append(
                        self.empty_param_state(
                            flat_master_params[start_index:end_index]
                        )
                    )
                chunk_no += 1
                start_index = end_index

            nccl_events = []
            start_index = self.group_offsets[group_no]
            nbf = self.coalescing_factor

            # Step 2 - Issue all-reduces in chunks
            while start_index != self.group_offsets[group_no + 1]:
                end_index = min(
                    start_index + bucket_size * nbf, self.group_offsets[group_no + 1]
                )
                event = ax.comm_handle.allreduce(
                    flat_fp16_grad[start_index:end_index], async_op=True
                )
                nccl_events.append(event)
                start_index = end_index

            # Step 3 - Run Adam and copy back to GPU
            group = self.param_groups[group_no]
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            start_index = self.group_offsets[group_no]
            chunk_no = 0
            loss_scale = ax.loss_scale

            while start_index != self.group_offsets[group_no + 1]:
                end_index = min(
                    start_index + bucket_size, self.group_offsets[group_no + 1]
                )
                state = self.state_pipeline[group_no][chunk_no]
                state["step"] += 1

                with torch.cuda.stream(stream):
                    size = end_index - start_index
                    param_buffer[:size].copy_(
                        flat_master_params[start_index:end_index], non_blocking=True
                    )
                    exp_avg_buffer[:size].copy_(state["exp_avg"], non_blocking=True)
                    exp_avg_sq_buffer[:size].copy_(
                        state["exp_avg_sq"], non_blocking=True
                    )
                    if (chunk_no % nbf) == 0:
                        nccl_events[chunk_no // nbf].wait()

                    grad_buffer[:size].copy_(
                        flat_fp16_grad[start_index:end_index], non_blocking=True
                    )
                    fp32_grad = grad_buffer[:size].div_(loss_scale)
                    isnan = ax._check_nan(fp32_grad)
                    if isnan:
                        skip_update = True
                        start_index = end_index
                        chunk_no += 1
                        continue

                    F.adam(
                        [param_buffer[:size]],
                        [fp32_grad],
                        [exp_avg_buffer[:size]],
                        [exp_avg_sq_buffer[:size]],
                        [],
                        [state["step"]],
                        amsgrad=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr,
                        weight_decay=weight_decay,
                        eps=eps,
                    )
                    flat_fp16_params[start_index:end_index].copy_(
                        param_buffer[:size], non_blocking=True
                    )
                    state["exp_avg"].copy_(exp_avg_buffer[:size], non_blocking=True)
                    state["exp_avg_sq"].copy_(
                        exp_avg_sq_buffer[:size], non_blocking=True
                    )
                    flat_master_params[start_index:end_index].copy_(param_buffer[:size])

                start_index = end_index
                chunk_no += 1

        torch.cuda.synchronize()
        ax._sync_scale(skip_update)
