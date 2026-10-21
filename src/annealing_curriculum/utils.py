from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass
class TokenCounter:
    cumulative: int = 0
    def add(self, n: int):
        self.cumulative += n

def compute_max_steps(total_tokens: int, tokens_per_step: int) -> int:
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be >0")
    return math.ceil(total_tokens / tokens_per_step)

def format_tokens(n: int) -> str:
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return str(n)
