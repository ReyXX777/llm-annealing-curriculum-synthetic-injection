"""
Batch sampler — vectorized multinomial sampling (Fix #4).
"""
from __future__ import annotations
from typing import Dict, List, Sequence
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class WeightedSampler:
    """
    Samples source keys according to mixture weights w_s(t).
    Uses numpy (fast, vectorized) or torch.multinomial if configured.
    """

    def __init__(self, seed: int = 42, use_torch: bool = False):
        self.seed = seed
        self.use_torch = use_torch and HAS_TORCH
        self._np_rng = np.random.default_rng(seed)
        if self.use_torch:
            self._torch_gen = torch.Generator().manual_seed(seed)
            logger.info("WeightedSampler using torch.multinomial")
        else:
            logger.info("WeightedSampler using numpy")

    def sample_sources(self, weights: Dict[str, float], n: int) -> List[str]:
        """
        Vectorized multinomial sample n times from weights dict.
        Args:
            weights: normalized simplex
            n: number of samples
        Returns:
            list of source keys length n
        """
        if not weights:
            raise ValueError("weights empty")
        keys = list(weights.keys())
        probs = np.array([weights[k] for k in keys], dtype=np.float64)
        # guard: ensure sum ~1, normalize if needed
        total = probs.sum()
        if total < 1e-12:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / total

        if self.use_torch:
            t_probs = torch.tensor(probs, dtype=torch.float32)
            # torch.multinomial expects 2D?
            samples = torch.multinomial(t_probs, num_samples=n, replacement=True, generator=self._torch_gen)
            return [keys[i] for i in samples.tolist()]
        else:
            # numpy vectorized choice
            sampled_indices = self._np_rng.choice(len(keys), size=n, p=probs)
            return [keys[i] for i in sampled_indices]

    def sample_one(self, weights: Dict[str, float]) -> str:
        return self.sample_sources(weights, 1)[0]
