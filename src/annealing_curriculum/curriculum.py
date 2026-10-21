from __future__ import annotations
from typing import List, Optional, Generator
import logging
from .config import AnnealingConfig
from .scheduler import AnnealingScheduler
from .sampler import MixtureSampler, Batch
from .data_sources import DataSourceRegistry, Example
from .filters import QualityFilter
from .monitoring import log_mixture_weights, log_quality_metrics

logger = logging.getLogger(__name__)

class AnnealingCurriculum:
    # Main annealing curriculum generator
    # Implements PROCEDURE AnnealingCurriculum, yields via generator
    def __init__(self, config: AnnealingConfig, registry: Optional[DataSourceRegistry] = None):
        self.config = config
        self.scheduler = AnnealingScheduler(config)
        self.sampler = MixtureSampler(seed=config.seed)
        self.registry = registry or DataSourceRegistry(config.sources, seed=config.seed)
        self.quality_filter = QualityFilter(config.quality)

    def __len__(self):
        return self.config.max_steps

    def __iter__(self) -> Generator[Batch, None, None]:
        return self.stream_batches()

    def _sample_example_for_source(self, source_key: str, effective_web_sources: List[str]) -> Example:
        # Handles union: web = web_low U web_high, synthetic = synthetic U textbooks
        try:
            if source_key == "web":
                if not effective_web_sources:
                    effective_web_sources = ["web_high"]
                import random
                src_name = random.choice(effective_web_sources)
                return self.registry.get(src_name).sample()
            elif source_key == "synthetic":
                return self.registry.sample_from(["synthetic", "textbooks"])
            elif source_key == "domain":
                return self.registry.get("domain").sample()
            elif source_key == "instruction":
                return self.registry.get("instruction").sample()
            else:
                raise ValueError(f"Unknown bucket {source_key}")
        except Exception as e:
            logger.error(f"Failed sampling {source_key}: {e}, fallback web_high")
            return self.registry.get("web_high").sample()

    def stream_batches(self) -> Generator[Batch, None, None]:
        # Core generator: steps 2-7 of spec, graceful handling
        max_steps = self.config.max_steps
        tokens_per_step = self.config.tokens_per_step
        log_interval = self.config.log_interval
        logger.info(f"Starting annealing: {max_steps} steps, {self.config.total_annealing_tokens/1e9:.1f}B tokens")
        cumulative_tokens = 0
        for step in range(1, max_steps + 1):
            try:
                progress, weights = self.scheduler.get_weights(step)
                effective_web_sources = self.registry.effective_web_mix(
                    progress, self.config.quality.base_threshold, self.config.quality.slope)
                batch = Batch(mixture_weights=weights, step=step, progress=progress)
                attempts = 0
                max_attempts = tokens_per_step * 10
                while batch.tokens < tokens_per_step and attempts < max_attempts:
                    attempts += 1
                    source_bucket = self.sampler.sample_source(weights)
                    example = self._sample_example_for_source(source_bucket, effective_web_sources)
                    if self.quality_filter.passes(example, progress):
                        batch.add(example)
                if batch.num_examples == 0:
                    logger.warning(f"Step {step}: empty batch thr={self.quality_filter.threshold_at(progress):.3f}")
                    example = self.registry.get("web_high").sample()
                    batch.add(example)
                cumulative_tokens += batch.tokens
                yield batch
                if step % log_interval == 0:
                    log_mixture_weights(step, progress, weights)
                    log_quality_metrics(batch, cumulative_tokens)
                    exp_q = self.scheduler.expected_quality_proxy(weights)
                    logger.info(f"Step {step}/{max_steps} prog {progress:.3f} exp_q {exp_q:.3f} cum {cumulative_tokens/1e9:.2f}B")
                if cumulative_tokens >= self.config.total_annealing_tokens:
                    logger.info(f"Budget reached at step {step}")
                    break
            except Exception as exc:
                logger.exception(f"Critical error step {step}: {exc}")
                if 'batch' in locals() and batch.num_examples > 0:
                    yield batch
                continue
        logger.info(f"Annealing finished: {cumulative_tokens} tokens")
