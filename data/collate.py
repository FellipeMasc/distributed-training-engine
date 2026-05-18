import torch


def collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples], dim=0),
        "labels": torch.stack([sample["labels"] for sample in samples], dim=0),
    }
