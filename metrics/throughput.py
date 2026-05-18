def compute_tokens_per_second(tokens_processed: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return tokens_processed / elapsed_seconds
