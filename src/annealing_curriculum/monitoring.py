from __future__ import annotations
import logging
from typing import Dict
from .sampler import Batch

logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO, log_file: str | None = None):
    # Replace print with logging
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", handlers=handlers, force=True)

def log_mixture_weights(step: int, progress: float, weights: Dict[str, float]):
    w_str = " | ".join(f"{k}:{v:.3f}" for k, v in weights.items())
    logger.info(f"[Mixture] step={step} prog={progress:.3f} -> {w_str}")

def log_quality_metrics(batch: Batch, cumulative_tokens: int):
    logger.info(f"[Batch] step={batch.step} tokens={batch.tokens} examples={batch.num_examples} avg_q={batch.avg_quality:.3f} cum={cumulative_tokens/1e9:.2f}B")

def log_theorem_checks(progress: float, weights: Dict[str, float], batch: Batch):
    web_w = weights.get("web", 0)
    if progress > 0.9 and web_w > 0.15:
        logger.warning(f"T2 risk: web {web_w:.3f} >0.15 near end")
    logger.debug(f"T1 check p={progress:.2f} avg_q={batch.avg_quality:.3f}")
