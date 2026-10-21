from __future__ import annotations
import math
from typing import Callable

# Pure math functions - no side effects
# Relations: Progress(t)=t/N, CosineDecay, CosineRamp

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def cosine_decay(a: float, b: float, p: float) -> float:
    # Cosine decay: b + 0.5*(a-b)*(1+cos(pi*p))
    # a initial higher, b final lower, p in [0,1]
    p = clamp01(p)
    return b + 0.5 * (a - b) * (1.0 + math.cos(math.pi * p))

def cosine_ramp(a: float, b: float, p: float) -> float:
    # Cosine ramp: a + 0.5*(b-a)*(1-cos(pi*p))
    p = clamp01(p)
    return a + 0.5 * (b - a) * (1.0 - math.cos(math.pi * p))

def linear_decay(a: float, b: float, p: float) -> float:
    p = clamp01(p)
    return a + (b - a) * p

def linear_ramp(a: float, b: float, p: float) -> float:
    return linear_decay(a, b, p)

def get_schedule_fn(kind: str):
    if kind == "cosine":
        return cosine_decay, cosine_ramp
    elif kind == "linear":
        return linear_decay, linear_ramp
    else:
        raise ValueError(f"Unknown schedule {kind}")
