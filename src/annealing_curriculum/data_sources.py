from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Dict, List
import random
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class Example:
    # Single tokenized example in Omega, maps to P_source and P_qual
    text: str
    source: str
    quality: float
    token_count: int
    id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.text.encode()[:200]).hexdigest()[:12]
        self.quality = max(0.0, min(1.0, self.quality))

    @property
    def length_ok(self) -> bool:
        return self.token_count > 0

class MockStreamingSource:
    # Mock streaming source yielding infinite examples, memory O(1)
    def __init__(self, name: str, size: int, quality_mean: float, quality_std: float,
                 avg_tokens: int = 512, seed: int = 0):
        self.name = name
        self.size = size
        self.quality_mean = quality_mean
        self.quality_std = quality_std
        self.avg_tokens = avg_tokens
        self.rng = random.Random(seed)
        self._counter = 0

    def __iter__(self) -> Iterator[Example]:
        while True:
            yield self.sample()

    def sample(self) -> Example:
        q = self.rng.gauss(self.quality_mean, self.quality_std)
        q = max(0.0, min(1.0, q))
        tokens = max(10, int(self.rng.gauss(self.avg_tokens, self.avg_tokens*0.2)))
        text = f"[{self.name} ex {self._counter} q={q:.2f}] synthetic placeholder for {self.name}"
        self._counter += 1
        return Example(text=text, source=self.name, quality=q, token_count=tokens)

    def sample_batch(self, n: int) -> List[Example]:
        return [self.sample() for _ in range(n)]

class DataSourceRegistry:
    # Holds all six data sources: W_L, W_H, T, S, D, I
    def __init__(self, sizes, seed: int = 42):
        self.sources: Dict[str, MockStreamingSource] = {
            "web_low": MockStreamingSource("web_low", sizes.web_low_quality_size, 0.35, 0.20, 600, seed+1),
            "web_high": MockStreamingSource("web_high", sizes.web_high_quality_size, 0.70, 0.15, 550, seed+2),
            "textbooks": MockStreamingSource("textbooks", 200000, 0.85, 0.10, 800, seed+3),
            "synthetic": MockStreamingSource("synthetic", sizes.synthetic_size, 0.90, 0.08, 700, seed+4),
            "domain": MockStreamingSource("domain", sizes.domain_size, 0.80, 0.12, 650, seed+5),
            "instruction": MockStreamingSource("instruction", sizes.instruction_size, 0.75, 0.15, 400, seed+6),
        }

    def get(self, key: str) -> MockStreamingSource:
        if key not in self.sources:
            raise KeyError(f"Unknown source {key}")
        return self.sources[key]

    def sample_from(self, keys: List[str]) -> Example:
        import random as rnd
        k = rnd.choice(keys)
        return self.get(k).sample()

    def effective_web_mix(self, progress: float, base_q: float, slope: float) -> List[str]:
        # Quality-aware web mix: threshold = q0 + alpha*progress, A4
        threshold = base_q + slope * progress
        active = []
        for name in ["web_low", "web_high"]:
            src = self.sources[name]
            if src.quality_mean + 0.1 >= threshold:
                active.append(name)
        if not active:
            active = ["web_high"]
            logger.warning(f"Web mix empty at p={progress:.2f} thr={threshold:.2f}, fallback web_high")
        return active
