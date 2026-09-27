# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.utils import Split


def _get_data_parallel_rank():
    from distributed_training_engine.core.parallel_state import get_data_parallel_group

    return get_data_parallel_group().rank()


def _get_data_parallel_world_size():
    from distributed_training_engine.core.parallel_state import get_data_parallel_group

    return get_data_parallel_group().size()


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""
    from distributed_training_engine.training import get_args

    if dataset is None:
        return None
    args = get_args()

    if hasattr(dataset, "split"):
        split = dataset.split
    elif hasattr(dataset, "index_split"):
        split = dataset.index_split
    else:
        split = None

    if getattr(args, "dataloader_type", "single") == "external":
        return dataset

    is_eval = split in (Split.valid, Split.test)
    micro_batch_size = (
        getattr(args, "eval_micro_batch_size", args.micro_batch_size)
        if is_eval
        else args.micro_batch_size
    )
    global_batch_size = (
        getattr(args, "eval_global_batch_size", args.global_batch_size)
        if is_eval
        else args.global_batch_size
    )

    data_parallel_rank = _get_data_parallel_rank()
    data_parallel_size = _get_data_parallel_world_size()

    if split == Split.valid and getattr(args, "full_validation", False):
        batch_sampler = MegatronFullValidationSampler(
            total_samples=len(dataset),
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
        )
    elif getattr(args, "dataloader_type", "single") == "single":
        if getattr(args, "hybrid_context_parallel", False):
            batch_sampler = HybridCPMegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                data_parallel_rank=data_parallel_rank,
                data_parallel_size=data_parallel_size,
            )
        else:
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=micro_batch_size,
                data_parallel_rank=data_parallel_rank,
                data_parallel_size=data_parallel_size,
            )
    elif getattr(args, "dataloader_type", "single") == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            data_sharding=getattr(args, "data_sharding", False),
        )
    else:
        raise Exception(
            f"{getattr(args, 'dataloader_type', 'single')} dataloader type is not supported."
        )

    def worker_init_fn(_):
        import os

        def close_nvidia_fds():
            for fd in os.listdir("/proc/self/fd"):
                try:
                    path = os.readlink(f"/proc/self/fd/{fd}")
                    if path.startswith("/dev/nvidia"):
                        os.close(int(fd))
                except OSError:
                    pass

        close_nvidia_fds()

    maybe_worker_init_fn = worker_init_fn if args.num_workers > 0 else None
    if getattr(args, "hybrid_context_parallel", False):
        extra_kwargs = {"collate_fn": lambda x: x}
    else:
        extra_kwargs = {}
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=maybe_worker_init_fn,
        **extra_kwargs,
    )


class MegatronPretrainingSampler:
    """
    Sampler for Megatron pretraining dataloaders that divides data samples across
    data parallel workers. Each worker receives a contiguous chunk of data determined by
    its rank and the micro batch size. Supports dropping the last incomplete batch if
    specified, and keeps track of total and consumed samples. Designed to work with
    distributed training using Megatron's data parallelism.
    """

    def __init__(
        self,
        total_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
    ):
        self.total_samples = total_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last
        self.epoch = 0

        assert self.total_samples > 0, "no sample to consume: {}".format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), "data_parallel_rank should be smaller than data size: {}, {}".format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        for idx in range(self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []
            
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

        self.epoch+=1



class HybridCPMegatronPretrainingSampler(MegatronPretrainingSampler):
    """Data sampler for hybrid context parallel (Hybrid CP) format."""

    def __init__(
        self,
        total_samples,
        consumed_samples,
        micro_batch_size,
        global_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
    ):
        super().__init__(
            total_samples,
            consumed_samples,
            micro_batch_size,
            data_parallel_rank,
            data_parallel_size,
            drop_last,
        )
        self.global_batch_size = global_batch_size
        self.data_parallel_size = data_parallel_size
        self.num_micro_batches = self.global_batch_size // self.micro_batch_times_data_parallel_size

    def get_start_end_idx_global_batch(self):
        start_idx = [
            self.data_parallel_rank * self.micro_batch_size
            + i * self.micro_batch_size * self.data_parallel_size
            for i in range(self.num_micro_batches)
        ]
        end_idx = [start_idx[i] + self.micro_batch_size for i in range(self.num_micro_batches)]
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size * self.num_micro_batches:
                start_idx, end_idx = self.get_start_end_idx_global_batch()
                global_batch_idx = []
                for i in range(self.num_micro_batches):
                    global_batch_idx.extend(batch[start_idx[i] : end_idx[i]])
                yield global_batch_idx
                batch = []

        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx_global_batch()
            global_batch_idx = []
            for i in range(self.num_micro_batches):
                global_batch_idx.extend(batch[start_idx[i] : end_idx[i]])
            yield global_batch_idx


class MegatronFullValidationSampler:
    """Sampler for full validation that handles small datasets gracefully."""

    def __init__(self, total_samples, data_parallel_rank, data_parallel_size):
        self.total_samples = total_samples
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_size = 1

        assert self.total_samples > 0, f"no sample to consume: {self.total_samples}"
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), f"data_parallel_rank should be smaller than data size: {self.data_parallel_rank}, {data_parallel_size}"

    def __len__(self):
        num_batches = 0
        for batch_idx in range(0, self.total_samples, self.data_parallel_size):
            sample_idx = batch_idx + self.data_parallel_rank
            if sample_idx < self.total_samples:
                num_batches += 1
        return num_batches

    def __iter__(self):
        for batch_idx in range(0, self.total_samples, self.data_parallel_size):
            sample_idx = batch_idx + self.data_parallel_rank
            if sample_idx < self.total_samples:
                yield [sample_idx]


class RandomSeedDataset(Dataset):
    """A dataset wrapper that resets the random seed before each sample."""

    def __init__(self, dataset, seed):
        self.base_seed = seed
        self.curr_seed = seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:
    """
    Sampler for Megatron pretraining dataloaders that performs random sampling
    across data parallel workers.
    """

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
    ):
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        assert self.total_samples > 0, "no sample to consume: {}".format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), "data_parallel_rank should be smaller than data size: {}, {}".format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        if self.data_sharding:
            bucket_size = (
                self.total_samples // self.micro_batch_times_data_parallel_size
            ) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

        batch = []
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
