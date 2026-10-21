"""
Quality filtering implementing A4.
"""
from __future__ import annotations
from typing import Iterable, Iterator, List
import logging
from ..types import Example, SourceType
from ..math import quality_threshold
from ..config import QualityConfig

logger = logging.getLogger(__name__)

class QualityFilter:
    """
    Filters examples based on dynamic threshold q(t)=q0+alpha*progress.
    """
    def __init__(self, config: QualityConfig):
        self.config = config
        self.seen_text_hashes = set()  # simple dedup

    def threshold_for(self, progress: float) -> float:
        return quality_threshold(self.config.base_threshold, self.config.alpha, progress)

    def effective_web_mix_filter(self, examples: Iterable[Example], progress: float) -> Iterator[Example]:
        """
        Implements step 4: quality_filter(web_low, web_high, threshold=progress)
        Yields only examples with Q(x) >= threshold.
        """
        thr = self.threshold_for(progress)
        for ex in examples:
            if ex.source in (SourceType.WEB_LOW, SourceType.WEB_HIGH, SourceType.WEB):
                if ex.quality_score >= thr:
                    yield ex
            else:
                yield ex

    def passes(self, example: Example, progress: float = 0.0) -> bool:
        """Single example quality check + length + dedup."""
        # length constraints
        if not (self.config.min_tokens <= example.tokens <= self.config.max_tokens):
            return False
        # threshold for web sources only; synthetic/domain always pass quality unless very low
        if example.source in (SourceType.WEB_LOW, SourceType.WEB_HIGH, SourceType.WEB):
            thr = self.threshold_for(progress)
            if example.quality_score < thr:
                return False
        # simple dedup by text hash if enabled
        if self.config.dedup_enabled and example.text:
            h = hash(example.text)
            if h in self.seen_text_hashes:
                return False
            self.seen_text_hashes.add(h)
        return True

def passes_quality_checks(example: Example, config: QualityConfig | None = None, progress: float = 0.0) -> bool:
    """
    Standalone function for use in curriculum loop.
    """
    if config is None:
        # default permissive
        return example.tokens >= 8
    f = QualityFilter(config)
    return f.passes(example, progress=progress)
