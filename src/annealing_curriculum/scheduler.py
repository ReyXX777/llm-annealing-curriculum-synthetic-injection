from __future__ import annotations
from typing import Dict, Tuple
import logging
from .config import AnnealingConfig
from .schedules import get_schedule_fn

logger = logging.getLogger(__name__)

class AnnealingScheduler:
    # State manager for w_s(t), encapsulates logic, separates pure math from state
    # Implements A1, A2, A3, A6 with safe normalization
    def __init__(self, config: AnnealingConfig):
        self.config = config
        self.decay_fn, self.ramp_fn = get_schedule_fn(config.schedule.schedule_type)
        self.eps = 1e-12

    def progress(self, step: int) -> float:
        # Progress: Tau -> [0,1]
        if self.config.max_steps <= 0:
            return 1.0
        p = step / self.config.max_steps
        return max(0.0, min(1.0, p))

    def raw_weights(self, progress: float) -> Dict[str, float]:
        s = self.config.schedule
        web = self.decay_fn(s.initial_web_weight, s.final_web_weight, progress)
        synth = self.ramp_fn(s.initial_synth_weight, s.final_synth_weight, progress)
        domain = s.domain_weight
        instruction = s.instruction_weight
        return {"web": web, "synthetic": synth, "domain": domain, "instruction": instruction}

    def normalized_weights(self, progress: float) -> Dict[str, float]:
        # Normalize to simplex Delta(D), guard div-by-zero per guideline
        raw = self.raw_weights(progress)
        total = sum(raw.values())
        if total <= self.eps:
            logger.warning(f"Total weight near zero at p={progress}, uniform fallback")
            n = len(raw)
            return {k: 1.0/n for k in raw}
        norm = {k: v/total for k, v in raw.items()}
        s = sum(norm.values())
        if abs(s - 1.0) > 1e-6:
            norm = {k: v/s for k, v in norm.items()}
        return norm

    def get_weights(self, step: int) -> Tuple[float, Dict[str, float]]:
        p = self.progress(step)
        w = self.normalized_weights(p)
        return p, w

    def expected_quality_proxy(self, weights: Dict[str, float]) -> float:
        quality_means = {"web": 0.45, "synthetic": 0.88, "domain": 0.80, "instruction": 0.75}
        return sum(weights[k]*quality_means.get(k,0.5) for k in weights)
