# Annealing Curriculum Package - Public API
from .config import AnnealingConfig, WeightSchedule, QualityConfig
from .scheduler import AnnealingScheduler
from .curriculum import AnnealingCurriculum
from .data_sources import Example, MockStreamingSource, DataSourceRegistry
from .schedules import cosine_decay, cosine_ramp, linear_decay, linear_ramp

__all__ = [
    "AnnealingConfig",
    "WeightSchedule",
    "QualityConfig",
    "AnnealingScheduler",
    "AnnealingCurriculum",
    "Example",
    "MockStreamingSource",
    "DataSourceRegistry",
    "cosine_decay",
    "cosine_ramp",
    "linear_decay",
    "linear_ramp",
]
__version__ = "0.1.0"
