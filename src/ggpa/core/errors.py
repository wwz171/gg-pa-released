"""Custom error types for GG-PA."""


class GGPAError(Exception):
    """Base class for GG-PA errors."""


class ConfigurationError(GGPAError):
    """Raised when configuration or wiring is invalid."""


class NotSupportedError(GGPAError):
    """Raised when a requested feature is not supported by a component."""


class ShapeError(GGPAError):
    """Raised when array shapes are incompatible."""
