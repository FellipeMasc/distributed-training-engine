"""Test the SFT dataset + MegatronPretrainingRandomSampler + DataLoader pipeline.

Runs standalone (no distributed init required). Simulates a single DP rank.

Usage:
    python -m scripts.tests.dataloader
"""

import os
import sys
import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import numpy as np
import torch.distributed as dist

from dataset.data_sampler import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
from dataset.sft_dataset import build_sft_dataset
from dataset.utils import Split
from dataset.indexed_dataset import IndexedDataset, PackingDataset
JSONL_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "dataset",
    "data",
    "glaive-code-assistant-conversation.jsonl",
)
TOKENIZER_NAME = "NousResearch/Llama-3.2-1B"
MICRO_BATCH_SIZE = 4
SEQ_LENGTH = 512

DATA_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "dataset",
    "data",
    "tinystories-portuguese_output_document",
)


def test_sft_dataset():
    """Verify SFTDataset tokenizes, packs, and pads to sequence_length."""
    print("=" * 60)
    print("1. Testing SFTDataset")
    print("=" * 60)
    rank = torch.distributed.get_rank()
    sft_ds = build_sft_dataset(
            dataset_path=JSONL_PATH,
            tokenizer_path=TOKENIZER_NAME,
            sequence_length=SEQ_LENGTH,
            index_split=Split.train,
        )
    dist.barrier()
    print(f"   Dataset length : {len(sft_ds)}")
    print(f"   Sequence length: {SEQ_LENGTH}")

    sample = sft_ds[0]
    print(f"   tokens shape     : {sample['tokens'].shape}")
    print(f"   labels shape     : {sample['labels'].shape}")
    print(f"   loss_mask shape  : {sample['loss_mask'].shape}")
    print(f"   position_ids shape: {sample['position_ids'].shape}")
    print(f"   cu_seqlens       : {sample['cu_seqlens'].tolist()}")
    print(f"   max_seqlen       : {sample['max_seqlen'].item()}")
    print(f"   loss tokens      : {sample['loss_mask'].sum().item():.0f} / {SEQ_LENGTH}")

    assert sample["tokens"].shape == (SEQ_LENGTH,)
    assert sample["labels"].shape == (SEQ_LENGTH,)
    assert sample["loss_mask"].shape == (SEQ_LENGTH,)

    # Validate multiple sampled batches directly from the sequential sampler.
    sampler = MegatronPretrainingSampler(
        total_samples=len(sft_ds),
        micro_batch_size=MICRO_BATCH_SIZE,
        data_parallel_rank=rank,
        data_parallel_size=dist.get_world_size(),
        drop_last=True,
    )
    sampler_iter = iter(sampler)
    sampled_batches = list(sampler_iter)
    expected_batches = len(sft_ds) // MICRO_BATCH_SIZE // dist.get_world_size()
    assert len(sampled_batches) >= 2, (
        "Need at least 2 batches to validate multi-batch sampling. "
        f"Got {len(sampled_batches)}; increase num_samples or reduce MICRO_BATCH_SIZE."
    )
    assert len(sampled_batches) == expected_batches, (
        f"Expected {expected_batches} full batches, got {len(sampled_batches)}"
    )

    for batch_idx, batch_indices in enumerate(sampled_batches):
        assert len(batch_indices) == MICRO_BATCH_SIZE, (
            f"Expected batch size {MICRO_BATCH_SIZE}, got {len(batch_indices)} in batch {batch_idx}"
        )
        for sample_idx in batch_indices:
            sampled = sft_ds[sample_idx]
            assert sampled["tokens"].shape == (SEQ_LENGTH,)
            assert sampled["labels"].shape == (SEQ_LENGTH,)
            assert sampled["loss_mask"].shape == (SEQ_LENGTH,)

    print(
        f"   Sampler-in-test OK: validated {len(sampled_batches)} batches "
        f"({len(sampled_batches) * MICRO_BATCH_SIZE} samples)"
    )
    print("   SFTDataset OK")


def test_text_dataset():
    """Verify SFTDataset tokenizes, packs, and pads to sequence_length."""
    print("=" * 60)
    print("1. Testing SFTDataset")
    print("=" * 60)
    rank = torch.distributed.get_rank()
    text_ds = PackingDataset(
        path_prefix=str(DATA_PATH),
        sequence_length=SEQ_LENGTH
    )
    dist.barrier()
    print(f"   Dataset length : {len(text_ds)}")
    print(f"   Sequence length: {SEQ_LENGTH}")

    sample = text_ds[0]
    print(f"   tokens shape     : {sample.shape}")

    # Validate multiple sampled batches directly from the sequential sampler.
    sampler = MegatronPretrainingSampler(
        total_samples=len(text_ds),
        micro_batch_size=MICRO_BATCH_SIZE,
        data_parallel_rank=rank,
        data_parallel_size=dist.get_world_size(),
        drop_last=True,
    )
    sampler_iter = iter(sampler)
    sampled_batches = list(sampler_iter)
    expected_batches = len(text_ds) // MICRO_BATCH_SIZE // dist.get_world_size()
    assert len(sampled_batches) >= 2, (
        "Need at least 2 batches to validate multi-batch sampling. "
        f"Got {len(sampled_batches)}; increase num_samples or reduce MICRO_BATCH_SIZE."
    )
    assert len(sampled_batches) == expected_batches, (
        f"Expected {expected_batches} full batches, got {len(sampled_batches)}"
    )

    for batch_idx, batch_indices in enumerate(sampled_batches):
        assert len(batch_indices) == MICRO_BATCH_SIZE, (
            f"Expected batch size {MICRO_BATCH_SIZE}, got {len(batch_indices)} in batch {batch_idx}"
        )
        for sample_idx in batch_indices:
            sampled = text_ds[sample_idx]

    print(
        f"   Sampler-in-test OK: validated {len(sampled_batches)} batches "
        f"({len(sampled_batches) * MICRO_BATCH_SIZE} samples)"
    )
    print("   SFTDataset OK")


def test_sampler(sft_ds):
    """Test the MegatronPretrainingRandomSampler yields correct batch indices."""
    print("\n" + "=" * 60)
    print("2. Testing MegatronPretrainingRandomSampler")
    print("=" * 60)

    sampler = MegatronPretrainingRandomSampler(
        dataset=sft_ds,
        total_samples=len(sft_ds),
        consumed_samples=0,
        micro_batch_size=MICRO_BATCH_SIZE,
        data_parallel_rank=0,
        data_parallel_size=1,
        data_sharding=False,
    )

    batches_to_show = 3
    batch_indices = None
    for i, batch_indices in enumerate(sampler):
        if i >= batches_to_show:
            break
        print(f"   Batch {i}: indices={batch_indices}, len={len(batch_indices)}")

    assert len(batch_indices) == MICRO_BATCH_SIZE, (
        f"Expected batch size {MICRO_BATCH_SIZE}, got {len(batch_indices)}"
    )
    print(f"   Sampler OK — yields batches of {MICRO_BATCH_SIZE} indices")
    return sampler


def test_dataloader(sft_ds):
    """Build a full DataLoader and iterate a few batches."""
    print("\n" + "=" * 60)
    print("3. Testing full DataLoader pipeline")
    print("=" * 60)

    sampler = MegatronPretrainingRandomSampler(
        dataset=sft_ds,
        total_samples=len(sft_ds),
        consumed_samples=0,
        micro_batch_size=MICRO_BATCH_SIZE,
        data_parallel_rank=0,
        data_parallel_size=1,
        data_sharding=False,
    )

    dataloader = torch.utils.data.DataLoader(
        sft_ds,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )

    for i, batch in enumerate(dataloader):
        if i >= 3:
            break

        tokens = torch.stack([item["tokens"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        loss_mask = torch.stack([item["loss_mask"] for item in batch])

        print(f"   Batch {i}: tokens={tokens.shape}, labels={labels.shape}, loss_mask={loss_mask.shape}")
        print(f"     sample 0 first 10 tokens : {tokens[0, :10].tolist()}")
        print(f"     sample 0 loss token count: {loss_mask[0].sum().item():.0f}")
        print()

    assert tokens.shape == (MICRO_BATCH_SIZE, SEQ_LENGTH)
    print("   DataLoader OK")


def main():
    print(f"JSONL path  : {JSONL_PATH}")
    print(f"Tokenizer   : {TOKENIZER_NAME}")
    print(f"Micro batch : {MICRO_BATCH_SIZE}")
    print(f"Seq length  : {SEQ_LENGTH}")
    print()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "gloo"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=f"env://", timeout=datetime.timedelta(minutes=10))
    # test_sft_dataset()
    test_text_dataset()
    # test_sampler(sft_ds)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
