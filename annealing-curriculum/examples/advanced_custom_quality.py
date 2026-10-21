"""
Advanced example: custom quality scorer and adaptive adjustment.
"""
import logging
from annealing_curriculum import AnnealingConfig
from annealing_curriculum.curriculum import AnnealingCurriculum
from annealing_curriculum.sources.base import MockStreamingSource
from annealing_curriculum.types import SourceType, Example
from annealing_curriculum.quality import QualityFilter
from annealing_curriculum.config import QualityConfig
from annealing_curriculum.quality.scorer import QualityScorer

logging.basicConfig(level=logging.INFO)

class LengthAndQualityScorer(QualityScorer):
    """Custom scorer that penalizes too-short examples."""
    def score(self, example: Example) -> float:
        # boost for domain + synthetic
        boost = 0.1 if example.source in (SourceType.DOMAIN, SourceType.SYNTHETIC) else 0.0
        length_penalty = 0.0 if example.tokens > 100 else -0.2
        return max(0.0, min(1.0, example.quality_score + boost + length_penalty))

def main():
    config = AnnealingConfig(
        total_annealing_tokens=20_000_000,
        tokens_per_step=100_000,
        log_interval=2,
    )
    config.quality = QualityConfig(base_threshold=0.4, alpha=0.6, dedup_enabled=True)

    sources = {
        "web_low": MockStreamingSource("web_low", source_type=SourceType.WEB_LOW, size=200_000, avg_quality=0.25),
        "web_high": MockStreamingSource("web_high", source_type=SourceType.WEB_HIGH, size=100_000, avg_quality=0.65),
        "synthetic": MockStreamingSource("synthetic", source_type=SourceType.SYNTHETIC, size=200_000, avg_quality=0.9),
        "textbooks": MockStreamingSource("textbooks", source_type=SourceType.TEXTBOOK, size=50_000, avg_quality=0.92),
        "domain": MockStreamingSource("domain", source_type=SourceType.DOMAIN, size=100_000, avg_quality=0.85),
        "instruction": MockStreamingSource("instruction", source_type=SourceType.INSTRUCTION, size=20_000, avg_quality=0.88),
    }

    curriculum = AnnealingCurriculum(config=config, sources=sources)

    # Track quality over time for T1 verification
    qualities = []
    for batch in curriculum:
        avg_q = sum(e.quality_score for e in batch.examples)/len(batch.examples)
        qualities.append(avg_q)
        print(f"Step {batch.step} p={batch.progress:.3f} avgQ={avg_q:.3f} weights={batch.mixture_weights}")
        # Adaptive adjustment example (optional monitoring)
        if len(qualities) > 5 and qualities[-1] < qualities[-2]:
            print("Warning: quality dipped, might need slower decay")
        if batch.step >= 15:
            break

    print("Monotonic improvement check (T1):", qualities)

if __name__ == "__main__":
    main()
