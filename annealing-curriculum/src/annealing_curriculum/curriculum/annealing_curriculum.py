"""
Core curriculum generator — yields batches, constant memory via yield (Fix #3).
"""
from __future__ import annotations
from typing import Dict, Iterator, Optional, List
import logging
from collections import deque

from ..config import AnnealingConfig
from ..types import Example, Batch, SourceType
from ..sources.base import StreamingSource, MockStreamingSource
from ..sources.registry import SourceRegistry
from ..scheduling.scheduler import AnnealingScheduler
from ..curriculum.sampler import WeightedSampler
from ..quality.filters import QualityFilter
from ..exceptions import EmptySourceError, BudgetExceededError
from ..math import compute_progress

logger = logging.getLogger(__name__)

class AnnealingCurriculum:
    """
    Main annealing curriculum class.

    PROCEDURE AnnealingCurriculum (from spec):
    FOR step=1..max_steps:
        progress = step/max_steps
        web_weight = cosine_decay(...)
        synth_weight = cosine_ramp(...)
        normalize
        effective_web_mix = quality_filter(..., threshold=progress)
        sample batch via multinomial
        yield batch

    This is an iterator/generator: `for batch in curriculum: ...`
    """

    def __init__(
        self,
        config: AnnealingConfig,
        sources: Dict[str, StreamingSource] | SourceRegistry,
        quality_filter: Optional[QualityFilter] = None,
    ):
        self.config = config
        self.scheduler = AnnealingScheduler(config)
        self.sampler = WeightedSampler(seed=config.seed, use_torch=config.use_torch_sampler)

        if isinstance(sources, SourceRegistry):
            self.registry = sources
        else:
            self.registry = SourceRegistry(sources)

        self.qfilter = quality_filter or QualityFilter(config.quality)

        # streaming iterators per source, kept as deque for round-robin fallback
        self._source_iters: Dict[str, Iterator[Example]] = {}
        for key, src in self.registry.sources.items():
            self._source_iters[key] = iter(src)

        self.current_step = 0
        self.tokens_consumed = 0
        self.max_steps = config.max_steps

        # mapping from grouped sampler keys to actual registry keys
        self.group_to_real = {
            "web": ["web_low", "web_high", "web"],  # try in order
            "synthetic": ["synthetic", "textbooks", "textbook", "synthetic_textbook"],
            "domain": ["domain", "domain_specialized", "code", "math"],
            "instruction": ["instruction", "instruction_mix", "instruct"],
        }

        logger.info(f"Curriculum init: max_steps={self.max_steps}, sources={self.registry.available_keys()}")

    def _get_next_example_from_group(self, group: str) -> Example | None:
        """Try to fetch an example from a logical group, with fallbacks."""
        candidates = self.group_to_real.get(group, [group])
        for real_key in candidates:
            if real_key not in self.registry.sources:
                continue
            try:
                # try iterator first, then sample()
                it = self._source_iters.get(real_key)
                if it is None:
                    src = self.registry.get(real_key)
                    it = iter(src)
                    self._source_iters[real_key] = it
                try:
                    ex = next(it)
                    return ex
                except StopIteration:
                    # exhausted iterator, try sampling via registry (mock can still sample)
                    logger.debug(f"Iterator for {real_key} exhausted, switching to sample()")
                    try:
                        return self.registry.sample_from(real_key)
                    except EmptySourceError:
                        continue
            except Exception as e:
                logger.warning(f"Failed to get example from {real_key}: {e}")
                continue
        return None

    def _build_batch(self, weights: Dict[str, float], progress: float, tokens_per_step: int) -> Batch:
        """Build batch until tokens >= tokens_per_step using multinomial sampling."""
        batch_examples: List[Example] = []
        tokens_acc = 0
        attempts = 0
        max_attempts = 5000  # guard against infinite loop if all sources filtered

        # pre-sample a chunk of source choices vectorized
        chunk_size = 128
        source_choices = deque(self.sampler.sample_sources(weights, chunk_size))

        while tokens_acc < tokens_per_step and attempts < max_attempts:
            if not source_choices:
                source_choices = deque(self.sampler.sample_sources(weights, chunk_size))

            group = source_choices.popleft()
            ex = self._get_next_example_from_group(group)

            if ex is None:
                attempts += 1
                continue

            # apply quality + length + dedup checks
            try:
                if not self.qfilter.passes(ex, progress=progress):
                    attempts += 1
                    continue
            except Exception as e:
                logger.warning(f"Quality check failed for example {ex._id}: {e}")
                attempts += 1
                continue

            batch_examples.append(ex)
            tokens_acc += ex.tokens
            attempts = 0  # reset on success

        if not batch_examples:
            logger.error(f"Could not build batch at progress {progress}, weights {weights}")
            # return empty batch instead of crash (graceful degradation)
            return Batch(examples=[], step=self.current_step, mixture_weights=weights, progress=progress)

        return Batch(
            examples=batch_examples,
            step=self.current_step,
            mixture_weights=weights,
            progress=progress,
        )

    def __iter__(self) -> Iterator[Batch]:
        """
        Generator that yields batches — memory constant regardless of B (Fix #3).
        """
        return self.generator()

    def generator(self) -> Iterator[Batch]:
        """Explicit generator method."""
        while self.current_step < self.max_steps and self.tokens_consumed < self.config.total_annealing_tokens:
            self.current_step += 1
            self.scheduler.current_step = self.current_step
            progress = compute_progress(self.current_step, self.max_steps)
            weights = self.scheduler.get_weights(self.current_step)

            # optional quality-aware reweighting note: threshold rises with progress
            batch = self._build_batch(weights, progress, self.config.tokens_per_step)

            if batch.count == 0:
                logger.warning(f"Empty batch at step {self.current_step}, attempting to continue")
                # try next step, but if many empty batches, break
                if self.current_step % 10 == 0:
                    logger.error("Too many empty batches, stopping curriculum")
                    break
                continue

            self.tokens_consumed += batch.tokens

            if self.current_step % self.config.log_interval == 0:
                avg_q = sum(e.quality_score for e in batch.examples) / max(1, len(batch.examples))
                logger.info(
                    f"[Step {self.current_step}/{self.max_steps}] p={progress:.3f} "
                    f"weights={ {k: f'{v:.3f}' for k,v in weights.items()} } "
                    f"batch_tokens={batch.tokens} avg_Q={avg_q:.3f} "
                    f"source_counts={batch.source_counts()} "
                    f"total_consumed={self.tokens_consumed/1e9:.2f}B/{self.config.total_annealing_tokens/1e9:.2f}B"
                )

            yield batch

        logger.info(f"Annealing complete: steps={self.current_step}, tokens={self.tokens_consumed}")

    def reset(self) -> None:
        self.current_step = 0
        self.tokens_consumed = 0
        self.scheduler.reset()
        self._source_iters = {k: iter(src) for k, src in self.registry.sources.items()}
        logger.info("Curriculum reset")
