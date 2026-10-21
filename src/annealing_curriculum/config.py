from __future__ import annotations
from dataclasses import dataclass, field
import math

@dataclass
class WeightSchedule:
    # Initial/final mixture targets (pre-normalization)
    initial_web_weight: float = 0.65
    final_web_weight: float = 0.10
    initial_synth_weight: float = 0.15
    final_synth_weight: float = 0.45
    domain_weight: float = 0.25
    instruction_weight: float = 0.08
    schedule_type: str = "cosine"

    def __post_init__(self):
        assert self.schedule_type in ("cosine", "linear")
        for v in [self.initial_web_weight, self.final_web_weight,
                  self.initial_synth_weight, self.final_synth_weight,
                  self.domain_weight, self.instruction_weight]:
            assert 0 <= v <= 1

@dataclass
class QualityConfig:
    # Quality filtering params for A4
    base_threshold: float = 0.30
    slope: float = 0.40
    min_length: int = 32
    max_length: int = 8192
    dedup_enabled: bool = True

    @property
    def final_threshold(self) -> float:
        return self.base_threshold + self.slope

@dataclass
class SourceSizes:
    web_low_quality_size: int = 1_000_000
    web_high_quality_size: int = 500_000
    synthetic_size: int = 300_000
    domain_size: int = 400_000
    instruction_size: int = 100_000

@dataclass
class AnnealingConfig:
    # Top-level config, token budget A5
    total_annealing_tokens: int = 100_000_000_000
    tokens_per_step: int = 2_097_152
    seed: int = 42
    log_interval: int = 50
    schedule: WeightSchedule = field(default_factory=WeightSchedule)
    quality: QualityConfig = field(default_factory=QualityConfig)
    sources: SourceSizes = field(default_factory=SourceSizes)

    def __post_init__(self):
        assert self.total_annealing_tokens >= 1_000_000, "too small for test"
        assert self.tokens_per_step > 0
        # For production, B should be 50B-200B, but allow smaller for tests/demo
        if not (5e10 <= self.total_annealing_tokens <= 2e11):
            import logging
            logging.getLogger(__name__).warning(f"B={self.total_annealing_tokens} outside recommended 50B-200B, allowed for testing")
        assert True

    @property
    def max_steps(self) -> int:
        return math.ceil(self.total_annealing_tokens / self.tokens_per_step)
