from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import logging
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .data_sources import Example

logger = logging.getLogger(__name__)

@dataclass
class Batch:
    # Container for training batch, generator friendly
    examples: List[Example] = field(default_factory=list)
    mixture_weights: Dict[str, float] = field(default_factory=dict)
    step: int = 0
    progress: float = 0.0

    @property
    def tokens(self) -> int:
        return sum(e.token_count for e in self.examples)

    @property
    def num_examples(self) -> int:
        return len(self.examples)

    @property
    def avg_quality(self) -> float:
        if not self.examples:
            return 0.0
        return sum(e.quality for e in self.examples) / len(self.examples)

    def add(self, ex: Example):
        self.examples.append(ex)

class MixtureSampler:
    # Samples source via multinomial, uses torch/numpy if available
    # Implements A6 sampling correctness
    def __init__(self, seed: int = 42, backend: str = "auto"):
        self.seed = seed
        random.seed(seed)
        if HAS_NUMPY:
            np.random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)
        self.backend = backend
        if backend == "auto":
            if HAS_TORCH:
                self.backend = "torch"
            elif HAS_NUMPY:
                self.backend = "numpy"
            else:
                self.backend = "python"

    def sample_source(self, weights: Dict[str, float]) -> str:
        keys = list(weights.keys())
        probs = [weights[k] for k in keys]
        total = sum(probs)
        if total <= 1e-12:
            logger.warning("Zero probs, uniform fallback")
            probs = [1.0/len(keys)]*len(keys)
        else:
            probs = [p/total for p in probs]

        if self.backend == "torch" and HAS_TORCH:
            p_tensor = torch.tensor(probs, dtype=torch.float32)
            idx = torch.multinomial(p_tensor, 1).item()
            return keys[idx]
        elif self.backend == "numpy" and HAS_NUMPY:
            idx = np.random.choice(len(keys), p=probs)
            return keys[int(idx)]
        else:
            r = random.random()
            cum = 0.0
            for k, p in zip(keys, probs):
                cum += p
                if r <= cum:
                    return k
            return keys[-1]

    def sample_sources_batch(self, weights: Dict[str, float], n: int) -> List[str]:
        return [self.sample_source(weights) for _ in range(n)]
