import torch


def get_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    bytes_used = torch.cuda.max_memory_allocated()
    return bytes_used / 1024 / 1024
