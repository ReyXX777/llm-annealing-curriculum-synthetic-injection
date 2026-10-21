"""
Core type definitions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import time

class SourceType(str, Enum):
    """Enumeration of data sources D = {W_L, W_H, T, S, D, I}"""
    WEB_LOW = "web_low"
    WEB_HIGH = "web_high"
    TEXTBOOK = "textbook"
    SYNTHETIC = "synthetic"
    DOMAIN = "domain"
    INSTRUCTION = "instruction"
    # grouped aliases for scheduler
    WEB = "web"  # combined W_L ∪ W_H after quality filtering
    SYNTHETIC_GROUP = "synthetic_group"  # S ∪ T

@dataclass
class Example:
    """
    Single tokenized example.
    Attributes:
        tokens: token ids or token count proxy
        source: SourceType
        quality_score: Q(x) in [0,1]
        text: optional raw text for debugging
        metadata: arbitrary shard info
    """
    tokens: int
    source: SourceType
    quality_score: float = 0.5
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _id: str = field(default_factory=lambda: str(time.time_ns()))

    def __post_init__(self) -> None:
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(f"quality_score must be in [0,1], got {self.quality_score}")
        if self.tokens <= 0:
            raise ValueError(f"tokens must be >0, got {self.tokens}")

@dataclass
class Batch:
    """Batch yielded to trainer."""
    examples: List[Example] = field(default_factory=list)
    step: int = 0
    mixture_weights: Dict[str, float] = field(default_factory=dict)
    progress: float = 0.0

    @property
    def tokens(self) -> int:
        return sum(e.tokens for e in self.examples)

    @property
    def count(self) -> int:
        return len(self.examples)

    def source_counts(self) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for e in self.examples:
            d[e.source.value] = d.get(e.source.value, 0) + 1
        return d
