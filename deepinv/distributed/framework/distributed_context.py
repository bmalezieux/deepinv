from __future__ import annotations

import os
from typing import Any, Callable
import warnings

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class DistributedContext:
    r"""Context manager for distributed computing.

    By default, all processes cooperate on one inverse problem. Setting
    ``inner_world_size`` creates a two-dimensional topology: contiguous groups
    of inner processes cooperate on one sample, while processes with the same
    inner rank form data-parallel groups.

    DeepInv collectives and :meth:`local_indices` operate on ``inner_group`` by
    default. ``rank`` and ``world_size`` remain global aliases for backward
    compatibility.

    :param str | None backend: distributed backend, selected automatically by default.
    :param bool cleanup: destroy a global process group created here on exit.
    :param int | None seed: random seed.
    :param bool seed_offset: offset the seed by rank. With an explicit inner
        topology, the data-parallel rank is used so inner ranks share a seed.
    :param bool deterministic: use deterministic cuDNN operations.
    :param str | None device_mode: ``'cpu'``, ``'gpu'``, or automatic if ``None``.
    :param int | None inner_world_size: processes cooperating on one sample. It
        must divide the global world size. ``None`` preserves legacy behavior.
    """

    def __init__(
        self,
        backend: str | None = None,
        cleanup: bool = True,
        seed: int | None = None,
        seed_offset: bool = True,
        deterministic: bool = False,
        device_mode: str | None = None,
        inner_world_size: int | None = None,
    ):
        if inner_world_size is not None and (
            isinstance(inner_world_size, bool)
            or not isinstance(inner_world_size, int)
            or inner_world_size < 1
        ):
            raise ValueError("inner_world_size must be a positive integer or None")

        self.backend = backend
        self.cleanup = cleanup
        self.seed = seed
        self.seed_offset = seed_offset
        self.deterministic = deterministic
        self.device_mode = device_mode
        self._requested_inner_world_size = inner_world_size

        self.created_dist = False
        self.use_dist = False
        self.global_world_size = 1
        self.global_rank = 0
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.local_world_size = 1

        self.inner_group = None
        self.inner_rank = 0
        self.inner_world_size = 1
        self.inner_group_ranks = [0]
        self.dp_group = None
        self.dp_rank = 0
        self.dp_world_size = 1
        self.dp_group_ranks = [0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dist_wrapper_cache: dict[str, Callable[..., Any]] = {}
        self._param_sync_scheduled_tasks: set[int] = set()
        self._param_sync_pending: dict[int, dict[int, torch.nn.Parameter]] = {}

    def __enter__(self):
        env_has_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
        should_init_pg = (not dist.is_initialized()) and env_has_dist

        visible_gpus = torch.cuda.device_count()
        cuda_ok = torch.cuda.is_available() and visible_gpus > 0
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.global_world_size = int(os.environ.get("WORLD_SIZE", 1))

        if should_init_pg:
            backend = self.backend
            if backend is None:
                if self.device_mode == "cpu":
                    backend = "gloo"
                elif self.device_mode == "gpu":
                    if not dist.is_nccl_available():
                        raise RuntimeError(
                            "GPU mode requested but NCCL backend not available"
                        )
                    backend = "nccl"
                elif (
                    cuda_ok
                    and dist.is_nccl_available()
                    and self.local_world_size <= visible_gpus
                ):
                    backend = "nccl"
                else:
                    backend = "gloo"
            dist.init_process_group(backend=backend)
            self.created_dist = True

        self.use_dist = dist.is_initialized()
        if self.use_dist:
            self.global_world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        self.rank = self.global_rank
        self.world_size = self.global_world_size

        if self.device_mode == "cpu":
            self.device = torch.device("cpu")
        elif self.device_mode == "gpu":
            if not cuda_ok:
                raise RuntimeError(
                    "GPU mode requested but CUDA not available or no visible GPUs"
                )
            dev_index = 0 if visible_gpus == 1 else self.local_rank % visible_gpus
            self.device = torch.device(f"cuda:{dev_index}")
            torch.cuda.set_device(self.device)
        elif cuda_ok:
            dev_index = 0 if visible_gpus == 1 else self.local_rank % visible_gpus
            self.device = torch.device(f"cuda:{dev_index}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        try:
            self._create_topology()
        except Exception:
            # __exit__ is not called when __enter__ raises.
            if self.created_dist and dist.is_initialized():
                dist.destroy_process_group()
                self.created_dist = False
            raise
        self._post_init_setup()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cleanup and self.created_dist and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    def _create_topology(self):
        """Create row (inner) and column (data-parallel) process groups."""
        requested = self._requested_inner_world_size
        inner_world_size = self.global_world_size if requested is None else requested
        if self.global_world_size % inner_world_size != 0:
            raise ValueError(
                "inner_world_size must divide the global world size, got "
                f"inner_world_size={inner_world_size} and "
                f"global_world_size={self.global_world_size}"
            )

        self.inner_world_size = inner_world_size
        self.dp_world_size = self.global_world_size // inner_world_size
        self.inner_rank = self.global_rank % inner_world_size
        self.dp_rank = self.global_rank // inner_world_size
        inner_start = self.dp_rank * inner_world_size
        self.inner_group_ranks = list(
            range(inner_start, inner_start + inner_world_size)
        )
        self.dp_group_ranks = list(
            range(self.inner_rank, self.global_world_size, inner_world_size)
        )

        if not self.use_dist:
            return

        if self.inner_world_size == self.global_world_size:
            self.inner_group = dist.group.WORLD
        else:
            for replica_rank in range(self.dp_world_size):
                ranks = list(
                    range(
                        replica_rank * self.inner_world_size,
                        (replica_rank + 1) * self.inner_world_size,
                    )
                )
                group = dist.new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self.inner_group = group

        if self.dp_world_size == self.global_world_size:
            self.dp_group = dist.group.WORLD
        elif self.dp_world_size > 1:
            for inner_rank in range(self.inner_world_size):
                ranks = list(
                    range(inner_rank, self.global_world_size, self.inner_world_size)
                )
                group = dist.new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self.dp_group = group

    def _post_init_setup(self):
        if self.seed is not None:
            if self.seed_offset:
                offset = (
                    self.dp_rank
                    if self._requested_inner_world_size is not None
                    else self.global_rank
                )
            else:
                offset = 0
            s = self.seed + offset
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        if self.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @property
    def is_global_main(self) -> bool:
        """Whether this process is global rank zero."""
        return self.global_rank == 0

    @property
    def is_inner_main(self) -> bool:
        """Whether this process is rank zero in its inner group."""
        return self.inner_rank == 0

    @property
    def is_dp_main(self) -> bool:
        """Whether this process is rank zero in its data-parallel group."""
        return self.dp_rank == 0

    def local_indices(self, num_items: int) -> list[int]:
        r"""Get indices assigned to this rank within its inner group."""
        indices = [
            i for i in range(num_items) if i % self.inner_world_size == self.inner_rank
        ]
        if self.inner_world_size > 1 and len(indices) == 0:
            warnings.warn(
                f"Inner rank {self.inner_rank} has no work items to process "
                f"(num_items={num_items}, inner_world_size={self.inner_world_size}). "
                "Consider reducing inner_world_size or increasing the workload.",
                UserWarning,
            )
        return indices

    def distributed_data_sampler(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> DistributedSampler:
        r"""Create a sampler that shards data across replicas, not inner ranks.

        Every process in an inner group receives the same sample indices. Call
        ``sampler.set_epoch(epoch)`` before each shuffled epoch.
        """
        sampler_seed = self.seed if seed is None and self.seed is not None else seed
        return DistributedSampler(
            dataset,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=shuffle,
            seed=0 if sampler_seed is None else sampler_seed,
            drop_last=drop_last,
        )

    def distributed_data_parallel(
        self, module: torch.nn.Module, **kwargs
    ) -> torch.nn.Module:
        r"""Wrap a module with PyTorch DDP over the data-parallel group.

        The module is returned unchanged for a single data-parallel replica.
        DeepInv objects should first be passed to
        :func:`deepinv.distributed.distribute`.
        """
        if "process_group" in kwargs:
            raise TypeError(
                "distributed_data_parallel() selects ctx.dp_group; "
                "do not pass process_group"
            )
        if self.dp_world_size == 1:
            return module
        if not self.use_dist or self.dp_group is None:
            raise RuntimeError("The data-parallel process group is not initialized")
        if self.device.type == "cuda" and "device_ids" not in kwargs:
            kwargs["device_ids"] = [self.device.index]
            kwargs.setdefault("output_device", self.device.index)
        return DistributedDataParallel(module, process_group=self.dp_group, **kwargs)

    def _default_group(self, group):
        return self.inner_group if group is None else group

    def _inner_src_to_global(self, src: int, group) -> int:
        if group is not None:
            return src
        if not 0 <= src < self.inner_world_size:
            raise ValueError(
                f"inner source rank must be in [0, {self.inner_world_size}), got {src}"
            )
        return self.inner_group_ranks[src]

    def _collective(
        self, fn: Callable, fn_functional: Callable, x: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if not self.use_dist:
            return x
        if x.requires_grad:
            if fn_functional is None:
                raise AttributeError(
                    f"No functional autograd path available for '{fn.__name__}'"
                )
            return fn_functional(x, **kwargs)
        fn(x, **kwargs)
        return x

    def all_reduce(
        self,
        x: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group=None,
    ) -> torch.Tensor:
        group = self._default_group(group)
        return self._collective(
            dist.all_reduce, dist_nn.all_reduce, x, op=op, group=group
        )

    def broadcast(self, x: torch.Tensor, src: int = 0, group=None) -> torch.Tensor:
        global_src = self._inner_src_to_global(src, group)
        group = self._default_group(group)
        return self._collective(
            dist.broadcast, dist_nn.broadcast, x, src=global_src, group=group
        )

    def all_gather(self, x: torch.Tensor, group=None) -> torch.Tensor:
        r"""Gather one tensor per rank in the selected group and stack them."""
        if not self.use_dist:
            return x.unsqueeze(0)
        group = self._default_group(group)
        if x.requires_grad:
            try:
                gathered = dist_nn.all_gather(x, group=group)
            except AttributeError as e:
                raise AttributeError(
                    "No functional autograd path available for 'all_gather'"
                ) from e
            if isinstance(gathered, torch.Tensor):
                return gathered
            return torch.stack(list(gathered), dim=0)
        group_world_size = dist.get_world_size(group=group)
        out_list = [torch.empty_like(x) for _ in range(group_world_size)]
        dist.all_gather(out_list, x, group=group)
        return torch.stack(out_list, dim=0)

    def all_gather_object(self, obj_list: list, obj: Any, group=None):
        if not self.use_dist:
            if obj_list:
                obj_list[0] = obj
            return None
        return dist.all_gather_object(obj_list, obj, group=self._default_group(group))

    def broadcast_object_list(
        self, object_list: list, src: int = 0, group=None, device=None
    ):
        if not self.use_dist:
            return object_list
        global_src = self._inner_src_to_global(src, group)
        kwargs = {"src": global_src, "group": self._default_group(group)}
        if device is not None:
            kwargs["device"] = device
        return dist.broadcast_object_list(object_list, **kwargs)

    def barrier(self, group=None):
        """Synchronize the inner group by default."""
        if self.use_dist:
            return dist.barrier(group=self._default_group(group))
        return None

    def __getattr__(self, name):
        if name in self._dist_wrapper_cache:
            return self._dist_wrapper_cache[name]
        if hasattr(dist, name):

            def wrapper(*args, **kwargs):
                if self.use_dist:
                    return getattr(dist, name)(*args, **kwargs)
                return None

            self._dist_wrapper_cache[name] = wrapper
            return wrapper
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
