"""Custom exceptions for forensic pipeline error handling."""


class ForensicsError(Exception):
    """Base exception for the detection engine."""

    def __init__(self, message: str, details: str | None = None) -> None:
        self.details = details
        full = f"{message}" + (f" | {details}" if details else "")
        super().__init__(full)


class MediaLoadError(ForensicsError):
    """Raised when image, video, or audio cannot be loaded."""


class ProcessingError(ForensicsError):
    """Raised when feature extraction or analysis fails."""


class ModelError(ForensicsError):
    """Raised when model load or inference fails."""
