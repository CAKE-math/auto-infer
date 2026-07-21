"""Public errors shared across configuration, engine, and topology layers."""


class AutoInferError(Exception):
    """Base class for errors exposed by the inference runtime."""


class ConfigurationError(AutoInferError, ValueError):
    """The runtime configuration is internally inconsistent."""


class RequestRejectedError(AutoInferError, ValueError):
    """A request cannot be admitted by the configured engine."""


class EngineStalledError(AutoInferError, RuntimeError):
    """Unfinished requests exist but the engine cannot make progress."""
