from __future__ import annotations
from typing import Set
from .data_sources import Example
from .config import QualityConfig

class QualityFilter:
    # Implements QualityFilter(x,q) <=> Q(x) >= q and length checks
    def __init__(self, config: QualityConfig):
        self.config = config
        self.seen: Set[str] = set()

    def threshold_at(self, progress: float) -> float:
        # q(p) = q0 + alpha * p per A4
        return self.config.base_threshold + self.config.slope * progress

    def passes(self, example: Example, progress: float) -> bool:
        if not (self.config.min_length <= example.token_count <= self.config.max_length):
            return False
        thr = self.threshold_at(progress)
        if example.quality < thr:
            return False
        if self.config.dedup_enabled:
            if example.id in self.seen:
                return False
            self.seen.add(example.id)
            if len(self.seen) > 1_000_000:
                self.seen.clear()
        return True

def passes_quality_checks(example: Example, progress: float, qconfig: QualityConfig, seen: Set[str]) -> bool:
    if not (qconfig.min_length <= example.token_count <= qconfig.max_length):
        return False
    thr = qconfig.base_threshold + qconfig.slope * progress
    if example.quality < thr:
        return False
    if qconfig.dedup_enabled and example.id in seen:
        return False
    return True
