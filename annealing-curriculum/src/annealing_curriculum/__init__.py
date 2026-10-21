"""
Annealing Curriculum — top-level exports.
"""
from .config import AnnealingConfig, WeightScheduleConfig, QualityConfig
from .scheduling.scheduler import AnnealingScheduler
from .curriculum.annealing_curriculum import AnnealingCurriculum
from .sources.base import StreamingSource, MockStreamingSource
from .types import Example, Batch, SourceType
from .math import cosine_decay, cosine_ramp, normalize_weights, compute_progress

__all__ = [
    "AnnealingConfig",
    "WeightScheduleConfig",
    "QualityConfig",
    "AnnealingScheduler",
    "AnnealingCurriculum",
    "StreamingSource",
    "MockStreamingSource",
    "Example",
    "Batch",
    "SourceType",
    "cosine_decay",
    "cosine_ramp",
    "normalize_weights",
    "compute_progress",
]

__version__ = "0.1.0"
