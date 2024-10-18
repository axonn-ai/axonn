# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import timedelta
from typing import Any, Dict, List, Optional, Union, ContextManager, Callable, Type
from contextlib import nullcontext

import torch
import torch.distributed
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.launchers.subprocess_script import (
    _SubprocessScriptLauncher,
)
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import TBroadcast, _BackwardSyncControl
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH
from lightning.fabric.strategies.fsdp import (
    _activation_checkpointing_kwargs,
    _setup_activation_checkpointing,
)

from axonn import axonn as ax
from axonn.intra_layer import (
    sync_gradients,
    clip_grad_norm_,
    no_grad_sync,
    auto_parallelize,
    optimize_communication,
    overlap_all_gathers_for_checkpointed_forward,
)

from axonn.checkpoint import get_prefix_for_checkpoint
import os
import types


class AxonnStrategy(ParallelStrategy):
    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        G_intra_r: int = 1,
        G_intra_c: int = 1,
        G_intra_d: int = 1,
        overlap_communication=False,
        activation_checkpointing: Optional[
            Union[Type[Module], List[Type[Module]]]
        ] = None,
        activation_checkpointing_policy: Optional["_POLICY"] = None,  # noqa: F821
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
        )

        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self.G_intra_r = G_intra_r
        self.G_intra_c = G_intra_c
        self.G_intra_d = G_intra_d
        self._backward_sync_control = _AxoNNBackwardSyncControl()
        self.overlap_communication = overlap_communication

        self._activation_checkpointing_kwargs = _activation_checkpointing_kwargs(
            activation_checkpointing, activation_checkpointing_policy
        )

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        return {
            "num_replicas": torch.distributed.get_world_size(),
            "rank": torch.distributed.get_rank(),
        }
        # return {
        #    "num_replicas": ax.config.G_intra_d * ax.config.G_data,
        #    "rank": ax.config.G_intra_d * ax.config.data_parallel_rank
        #    + ax.config.intra_layer_depth_parallel_rank,
        # }

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        self._launcher = _SubprocessScriptLauncher(
            self.cluster_environment, self.num_processes, self.num_nodes
        )

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()

    @override
    def setup_module(self, module: Module):
        if self.overlap_communication:
            module.old_forward = module.forward

            def get_new_forward_with_overlap():
                def forward(self_, *args, **kwargs):
                    with optimize_communication(True, True, True, self_):
                        return self_.old_forward(*args, **kwargs)

                return forward

            module.forward = types.MethodType(get_new_forward_with_overlap(), module)

        # activation checkpointing needs to be set up after wrapping the model
        _setup_activation_checkpointing(module, self._activation_checkpointing_kwargs)
        return module

    @override
    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    @override
    def all_reduce(
        self,
        tensor: Tensor,
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_device_ids())
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        pass

    def _setup_distributed(self) -> None:
        self._set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(
            self.cluster_environment, self._process_group_backend, timeout=self._timeout
        )
        tensor_parallel_world_size = self.G_intra_c * self.G_intra_r * self.G_intra_d
        assert torch.distributed.get_world_size() % tensor_parallel_world_size == 0
        self.G_data = torch.distributed.get_world_size() // tensor_parallel_world_size

        ax.init(
            G_data=self.G_data,
            G_inter=1,
            G_intra_r=self.G_intra_r,
            G_intra_c=self.G_intra_c,
            G_intra_d=self.G_intra_d,
        )

    def _get_process_group_backend(self) -> str:
        return (
            self._process_group_backend
            or _get_default_process_group_backend_for_device(self.root_device)
        )

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(
                self.node_rank * self.num_processes + self.local_rank
            )
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

    def _determine_device_ids(self) -> Optional[List[int]]:
        return None if self.root_device.type == "cpu" else [self.root_device.index]

    @override
    def backward(
        self, tensor: Tensor, module: Module, *args: Any, **kwargs: Any
    ) -> None:
        assert module is not None, (
            "When using AxoNN,"
            " you need to pass the model in fabric.backward(model=..)"
        )
        if self.overlap_communication:
            with optimize_communication(True, True, True, module):
                modules_in_reverse = torch.nn.ModuleList(list(module.modules())[::-1])
                with overlap_all_gathers_for_checkpointed_forward(modules_in_reverse):
                    super().backward(tensor, module, *args, **kwargs)
        else:
            super().backward(tensor, module, *args, **kwargs)
        sync_gradients(module, mean=True)

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[
            Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]
        ] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        # different prefix for different tensor parallel ranks
        checkpoint_prefix = get_prefix_for_checkpoint()
        directory, filename = os.path.split(path)
        directory = os.path.join(directory, checkpoint_prefix)
        path = os.path.join(directory, filename)
        return super().load_checkpoint(path, state, strict)

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        if torch.distributed.get_rank(ax.comm_handle.data_parallel_group) == 0:
            # different prefix for different tensor parallel ranks
            checkpoint_prefix = get_prefix_for_checkpoint()
            directory, filename = os.path.split(path)
            directory = os.path.join(directory, checkpoint_prefix)
            state = self._convert_stateful_objects_in_state(
                state, filter=(filter or {})
            )
            path = os.path.join(directory, filename)
            self.checkpoint_io.save_checkpoint(
                checkpoint=state, path=path, storage_options=storage_options
            )

    @override
    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        self.precision.unscale_gradients(optimizer)
        parameters = self.precision.main_params(optimizer)
        grad_norm = clip_grad_norm_(
            parameters=parameters,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
        )
        return grad_norm

    @override
    def module_init_context(self, empty_init: Optional[bool] = None):
        return self.module_sharded_context()

    @override
    def module_sharded_context(self) -> ContextManager:
        return auto_parallelize()


class _AxoNNBackwardSyncControl(_BackwardSyncControl):
    @override
    def no_backward_sync(self, module: Module, enabled: bool) -> ContextManager:
        """Blocks gradient synchronization inside AxoNN"""
        if not enabled:
            return nullcontext()
        return no_grad_sync()
