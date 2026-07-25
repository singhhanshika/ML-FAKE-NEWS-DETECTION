"""Project-specific exceptions."""


class FakeNewsDetectorError(Exception):
    """Base exception for expected project failures."""


class DataValidationError(FakeNewsDetectorError):
    """Raised when input data does not satisfy the data contract."""


class ModelNotReadyError(FakeNewsDetectorError):
    """Raised when a trained model artifact is unavailable or invalid."""


class InputValidationError(FakeNewsDetectorError):
    """Raised when prediction input cannot be analyzed safely."""
