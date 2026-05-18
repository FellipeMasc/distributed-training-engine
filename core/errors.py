class EngineError(Exception):
    """Base engine exception."""


class ConfigValidationError(EngineError):
    """Raised when configuration fails validation."""


class DistributedSetupError(EngineError):
    """Raised when distributed runtime setup fails."""


class ModelInitializationError(EngineError):
    """Raised when model/tokenizer creation fails."""


class TrainingRuntimeError(EngineError):
    """Raised when training loop fails."""
