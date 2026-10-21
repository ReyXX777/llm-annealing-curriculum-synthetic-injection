"""Custom exceptions with graceful handling."""

class AnnealingError(Exception):
    """Base class for curriculum errors."""

class EmptySourceError(AnnealingError):
    """Raised when a data source is exhausted or empty."""

class BudgetExceededError(AnnealingError):
    """Raised when token budget B is exceeded."""

class WeightNormalizationError(AnnealingError):
    """Raised when weight normalization fails."""

class QualityFilterError(AnnealingError):
    """Raised when quality filtering yields no candidates."""
