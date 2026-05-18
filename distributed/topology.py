from engine.core.errors import DistributedSetupError


def validate_dp_topology(world_size: int, data_parallel_size: int) -> None:
    if world_size != data_parallel_size:
        raise DistributedSetupError(
            f"DP topology mismatch: world_size ({world_size}) != data_parallel_size ({data_parallel_size})"
        )
