"""
Pure mathematical functions — no side effects, easy to unit test.
Implements A2, A3, normalization, progress.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple

def compute_progress(step: int, max_steps: int) -> float:
    """
    Progress(t) = t / N in [0,1]
    Args:
        step: current step in [1..N]
        max_steps: N >0
    Returns:
        clipped progress in [0,1]
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be >0")
    p = step / max_steps
    return float(np.clip(p, 0.0, 1.0))

def cosine_decay(initial: float, final: float, progress: float) -> float:
    """
    Cosine decay: b + 0.5*(a-b)*(1+cos(pi*p))
    Monotonic decreasing if a > b, as required by A2.
    Args:
        initial: a in [0,1]
        final: b in [0,1], a > b for decay
        progress: p in [0,1]
    Returns:
        decayed weight
    """
    if not (0.0 <= progress <= 1.0):
        raise ValueError(f"progress must be in [0,1], got {progress}")
    return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * progress))

def cosine_ramp(initial: float, final: float, progress: float) -> float:
    """
    Cosine ramp: a + 0.5*(b-a)*(1-cos(pi*p))
    Monotonic increasing if b > a, as required by A3.
    """
    if not (0.0 <= progress <= 1.0):
        raise ValueError(f"progress must be in [0,1], got {progress}")
    return initial + 0.5 * (final - initial) * (1.0 - math.cos(math.pi * progress))

def normalize_weights(weights: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    """
    Normalize dict of weights to simplex Δ(D): sum=1, w>=0.
    Implements defensive check against division by zero (Fix #5).
    Args:
        weights: raw weights
        eps: tolerance for zero total
    Returns:
        normalized weights, or uniform fallback if total < eps
    """
    if not weights:
        raise ValueError("weights dict empty")
    # clip negatives to 0
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total < eps:
        # fallback: uniform distribution to avoid division by zero
        n = len(clipped)
        return {k: 1.0 / n for k in clipped}
    return {k: v / total for k, v in clipped.items()}

def quality_threshold(base: float, alpha: float, progress: float) -> float:
    """
    Implements A4: q(t) = q0 + alpha * progress, clipped to [0,1]
    """
    return float(np.clip(base + alpha * progress, 0.0, 1.0))

def expected_quality_from_weights(
    weights: Dict[str, float],
    source_mean_quality: Dict[str, float]
) -> float:
    """
    E[Q(batch)] = sum_s w_s * E[Q|s]
    Used for monitoring T1 monotonic improvement.
    """
    return sum(weights.get(s, 0.0) * q for s, q in source_mean_quality.items())
