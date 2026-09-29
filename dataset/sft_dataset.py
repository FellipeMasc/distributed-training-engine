# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from dataset.utils import Split

IGNORE_INDEX = -100


@dataclass
class SFTDatasetConfig:
    sequence_length: int
    tokenizer: Any
    random_seed: int = 42
    reset_position_ids: bool = False
    reset_attention_mask: bool = False
    create_attention_mask: bool = False
    context_parallel_size: int = 1


class SFTLowLevelDataset:
    """Load jsonl SFT data where each line has a ``messages`` field."""

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "SFTDataset requires the datasets library to be installed"
            ) from exc
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        return self.dataset[idx]["messages"]


class SFTTokenizer:
    """Minimal HuggingFace-backed tokenizer wrapper matching Megatron SFT expectations."""

    def __init__(self, tokenizer_path: str, trust_remote_code: bool = False) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=trust_remote_code
        )
        self.pad = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self._tokenizer.eos_token_id
        )
        self.eod = self._tokenizer.eos_token_id

    def apply_chat_template(self, conversation: List[Dict]):
        USER_TOKEN_START= "<|reserved_special_token_0|>"
        USER_TOKEN_END = "<|reserved_special_token_1|>"
        ASSISTANT_TOKEN_START = "<|start_header_id|>"
        ASSISTANT_TOKEN_END = "<|end_header_id|>"
        
        sentence_str = f"{conversation[0]['content']}{USER_TOKEN_START}{conversation[1]['content']}{USER_TOKEN_END}{ASSISTANT_TOKEN_START}{conversation[2]['content']}{ASSISTANT_TOKEN_END}"  
        
        input_ids = self._tokenizer.encode(sentence_str, add_special_tokens=False)
        user_token_end = self._tokenizer.encode(USER_TOKEN_END, add_special_tokens=False)[0]
        assistant_masks = np.zeros(len(input_ids), dtype=np.int64)
        user_token_end_index = input_ids.index(user_token_end)
        assistant_masks[user_token_end_index+1  :] = 1
        return {
            "input_ids": input_ids,
            "assistant_masks": assistant_masks,
        }


    def tokenize_conversation(
        self,
        conversation: List[Dict],
        return_target: bool,
        add_generation_prompt: bool,
    ):
        result = self.apply_chat_template(
            conversation
        )
        tokens = np.array(result["input_ids"], dtype=np.int64)
        if not return_target:
            return tokens

        targets = tokens.copy()
        assistant_mask = np.array(result["assistant_masks"], dtype=bool)
        targets[~assistant_mask] = IGNORE_INDEX
        return tokens, targets


class SFTDataset(torch.utils.data.Dataset):
    """Megatron-compatible SFT dataset: tokenize, pack, pad/truncate to sequence_length."""

    def __init__(
        self,
        dataset: SFTLowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: SFTDatasetConfig,
    ) -> None:
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.indices = indices
        self.num_samples = num_samples if num_samples is not None else len(indices)
        self.index_split = index_split
        self.config = config

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: SFTLowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: SFTDatasetConfig) -> SFTLowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def _split_conversations(self, merged_conversations):
        split_conversations = []
        current = []
        for msg in merged_conversations:
            if msg["role"] == "system":
                if current:
                    split_conversations.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            split_conversations.append(current)
        return split_conversations

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        merged_conversations = self.dataset[int(self.indices[idx % len(self.indices)])]
        split_conversations = self._split_conversations(merged_conversations)

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1] + 1, positions[-1] + 1 + pad_len))

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        pad = tokenizer.pad

        for conversation in split_conversations:
            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()

            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            if self.config.context_parallel_size > 1:
                pad_granularity = self.config.context_parallel_size * 2
                mod_token_count = len(pack_tokens) % pad_granularity
                if mod_token_count != 0:
                    pad_len = pad_granularity - mod_token_count
                    extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

            cu_seqlens.append(len(pack_tokens))

            if len(pack_tokens) >= pack_length + 1:
                max_body = pack_length
                pack_tokens = pack_tokens[:max_body]
                pack_targets = pack_targets[:max_body]
                pack_tokens.append(pad)
                pack_targets.append(pad)
                pack_positions = pack_positions[: pack_length + 1]
                cu_seqlens[-1] = len(pack_tokens) - 1
                break

        if len(pack_tokens) < pack_length + 1:
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        input_ids = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0
        loss_mask[labels == IGNORE_INDEX] = 0.0

        assert not self.config.create_attention_mask and not self.config.reset_attention_mask

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()

        return {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }


def build_sft_dataset(
    dataset_path: str,
    tokenizer_path: str,
    sequence_length: int,
    num_samples: Optional[int] = None,
    index_split: Split = Split.train,
    trust_remote_code: bool = False,
) -> SFTDataset:
    """Build an SFT dataset from a jsonl conversation file."""
    tokenizer = SFTTokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    config = SFTDatasetConfig(sequence_length=sequence_length, tokenizer=tokenizer)
    low_level = SFTLowLevelDataset(dataset_path)
    indices = np.arange(len(low_level))
    return SFTDataset(
        dataset=low_level,
        dataset_path=dataset_path,
        indices=indices,
        num_samples=num_samples if num_samples is not None else len(indices),
        index_split=index_split,
        config=config,
    )
