class OpenConfError(Exception):
    """Base class for all OpenConf errors."""


class OpenConfValueError(OpenConfError, ValueError):
    """Invalid configuration or parameter value."""


class OpenConfRuntimeError(OpenConfError, RuntimeError):
    """Chemistry-level failure during conformer generation."""
