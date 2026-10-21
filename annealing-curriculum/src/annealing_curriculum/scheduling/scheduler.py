"""
AnnealingScheduler — encapsulates state for w_s(t).
Fix #1: class-based, not global.
"""
from __future__ import annotations
from typing import Dict, Tuple
import logging
import numpy as np
from ..config import AnnealingConfig
from ..math import cosine_decay, cosine_ramp, compute_progress, normalize_weights

logger = logging.getLogger(__name__)

class AnnealingScheduler:
    """
    Implements weight schedule for annealing phase.

    Mathematical form:
        progress p = step / N
        web_weight_raw = CosineDecay(initial_web, final_web, p)
        synth_weight_raw = CosineRamp(initial_synth, final_synth, p)
        then normalize so sum=1

    Attributes:
        config: AnnealingConfig
        current_step: int
        max_steps: int
    """
    def __init__(self, config: AnnealingConfig):
        self.config = config
        self.current_step = 0
        self.max_steps = config.max_steps
        self._rng = np.random.default_rng(config.seed)
        logger.info(f"Scheduler init: N={self.max_steps}, B={config.total_annealing_tokens}, "
                    f"tokens_per_step={config.tokens_per_step}")

    def get_progress(self, step: int | None = None) -> float:
        s = step if step is not None else self.current_step
        return compute_progress(s, self.max_steps)

    def get_raw_weights(self, progress: float) -> Dict[str, float]:
        """
        Compute raw (unnormalized) weights before simplex projection.
        Returns dict with keys: web, synthetic_group, domain, instruction
        """
        ws = self.config.weight_schedule
        web_raw = cosine_decay(ws.initial_web_weight, ws.final_web_weight, progress)
        synth_raw = cosine_ramp(ws.initial_synth_weight, ws.final_synth_weight, progress)
        return {
            "web": web_raw,
            "synthetic": synth_raw,  # S∪T grouped
            "domain": ws.domain_weight,
            "instruction": ws.instruction_weight,
        }

    def get_weights(self, step: int | None = None) -> Dict[str, float]:
        """
        Get normalized mixture weights w_s(t) for step t, satisfying A1.
        Includes division-by-zero guard.
        """
        progress = self.get_progress(step)
        raw = self.get_raw_weights(progress)
        try:
            norm = normalize_weights(raw)
        except Exception as e:
            logger.error(f"Normalization failed at step {step}, raw={raw}: {e}")
            # fallback uniform
            n = len(raw)
            norm = {k: 1.0/n for k in raw}
        logger.debug(f"Step {step or self.current_step} p={progress:.4f} weights={norm}")
        return norm

    def step(self) -> Dict[str, float]:
        """Advance one step, return new weights."""
        self.current_step += 1
        return self.get_weights(self.current_step)

    def reset(self) -> None:
        self.current_step = 0
        self._rng = np.random.default_rng(self.config.seed)
        logger.info("Scheduler reset")

    def state_dict(self) -> Dict:
        return {"current_step": self.current_step, "max_steps": self.max_steps}

    def load_state_dict(self, state: Dict) -> None:
        self.current_step = state.get("current_step", 0)
