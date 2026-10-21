"""
Quality scoring — Q: Omega -> [0,1]
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from ..types import Example

class QualityScorer(ABC):
    @abstractmethod
    def score(self, example: Example) -> float:
        ...

class HeuristicScorer(QualityScorer):
    """
    Simple heuristic scorer: returns example.quality_score already.
    In production, this would be a classifier (e.g., fastText, LLM judge).
    """
    def score(self, example: Example) -> float:
        return float(example.quality_score)
