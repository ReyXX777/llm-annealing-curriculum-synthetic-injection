"""
Streaming data sources — generator based, constant memory.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Optional, List
import random
import itertools
import logging
from ..types import Example, SourceType
from ..exceptions import EmptySourceError

logger = logging.getLogger(__name__)

class StreamingSource(ABC):
    """Abstract streaming source, yields Example via __iter__."""

    def __init__(self, source_type: SourceType, name: Optional[str] = None):
        self.source_type = source_type
        self.name = name or source_type.value

    @abstractmethod
    def __iter__(self) -> Iterator[Example]:
        ...

    def sample(self) -> Example:
        """Sample single example; default uses iterator."""
        try:
            it = iter(self)
            return next(it)
        except StopIteration:
            raise EmptySourceError(f"Source {self.name} empty")

class MockStreamingSource(StreamingSource):
    """
    Mock source for testing / PhD assignment without real data.
    Generates random Examples with controllable quality distribution.
    Memory-safe: generates on-the-fly via yield (Fix #3).
    """
    def __init__(
        self,
        name: str,
        source_type: SourceType | str = SourceType.WEB_LOW,
        size: int = 10_000,
        avg_quality: float = 0.5,
        avg_tokens: int = 512,
        seed: int = 42,
        quality_std: float = 0.2,
    ):
        if isinstance(source_type, str):
            # allow string alias
            try:
                source_type = SourceType(source_type)
            except ValueError:
                # map group names
                mapping = {
                    "web": SourceType.WEB_LOW,
                    "synthetic": SourceType.SYNTHETIC,
                    "domain": SourceType.DOMAIN,
                    "web_low": SourceType.WEB_LOW,
                    "web_high": SourceType.WEB_HIGH,
                    "textbooks": SourceType.TEXTBOOK,
                    "textbook": SourceType.TEXTBOOK,
                    "instruction": SourceType.INSTRUCTION,
                }
                source_type = mapping.get(source_type.lower(), SourceType.WEB_LOW)
        super().__init__(source_type=source_type, name=name)
        self.size = size
        self.avg_quality = avg_quality
        self.avg_tokens = avg_tokens
        self.seed = seed
        self.quality_std = quality_std
        self._rng = random.Random(seed)
        self._counter = 0

    def __iter__(self) -> Iterator[Example]:
        self._counter = 0
        rng = random.Random(self.seed + self._counter)
        # Use local rng to avoid global state pollution
        import numpy as np
        np_rng = np.random.default_rng(self.seed)
        while self._counter < self.size:
            # quality sampled from clipped normal in [0,1]
            q = float(np_rng.normal(self.avg_quality, self.quality_std))
            q = max(0.0, min(1.0, q))
            tokens = max(8, int(np_rng.normal(self.avg_tokens, self.avg_tokens*0.2)))
            ex = Example(
                tokens=tokens,
                source=self.source_type,
                quality_score=q,
                text=f"[{self.name}] synthetic example {self._counter} q={q:.2f}",
                metadata={"mock_id": self._counter, "source_name": self.name}
            )
            self._counter += 1
            yield ex

    def sample(self) -> Example:
        import numpy as np
        np_rng = np.random.default_rng(self.seed + self._counter)
        q = float(np_rng.normal(self.avg_quality, self.quality_std))
        q = max(0.0, min(1.0, q))
        tokens = max(8, int(np_rng.normal(self.avg_tokens, self.avg_tokens*0.2)))
        self._counter += 1
        return Example(
            tokens=tokens,
            source=self.source_type,
            quality_score=q,
            text=f"[{self.name}] sampled q={q:.2f}",
            metadata={"mock_id": self._counter}
        )

class ListSource(StreamingSource):
    """Wraps an in-memory list but yields via generator to keep API uniform."""
    def __init__(self, examples: List[Example], source_type: SourceType, name: str = "list_source"):
        super().__init__(source_type=source_type, name=name)
        self.examples = examples
        self._idx = 0

    def __iter__(self) -> Iterator[Example]:
        for ex in self.examples:
            yield ex

    def sample(self) -> Example:
        if not self.examples:
            raise EmptySourceError(f"ListSource {self.name} empty")
        self._idx = (self._idx + 1) % len(self.examples)
        return self.examples[self._idx]
