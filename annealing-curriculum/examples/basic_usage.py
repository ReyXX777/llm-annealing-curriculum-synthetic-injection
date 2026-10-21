"""
Basic usage example for PhD assignment.
"""
import logging
from annealing_curriculum import AnnealingConfig, AnnealingCurriculum, MockStreamingSource
from annealing_curriculum.types import SourceType

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def main():
    config = AnnealingConfig(
        total_annealing_tokens=10_000_000,  # 10M for demo, normally 100B
        tokens_per_step=50_000,
        log_interval=5,
    )

    sources = {
        "web_low": MockStreamingSource("web_low", source_type=SourceType.WEB_LOW, size=100_000, avg_quality=0.3, avg_tokens=512),
        "web_high": MockStreamingSource("web_high", source_type=SourceType.WEB_HIGH, size=50_000, avg_quality=0.7, avg_tokens=512),
        "textbooks": MockStreamingSource("textbooks", source_type=SourceType.TEXTBOOK, size=20_000, avg_quality=0.9, avg_tokens=1024),
        "synthetic": MockStreamingSource("synthetic", source_type=SourceType.SYNTHETIC, size=100_000, avg_quality=0.85, avg_tokens=768),
        "domain": MockStreamingSource("domain", source_type=SourceType.DOMAIN, size=50_000, avg_quality=0.8, avg_tokens=600),
        "instruction": MockStreamingSource("instruction", source_type=SourceType.INSTRUCTION, size=10_000, avg_quality=0.85, avg_tokens=256),
    }

    curriculum = AnnealingCurriculum(config=config, sources=sources)

    for batch in curriculum:
        # placeholder train step
        print(f"Step {batch.step} progress={batch.progress:.2f} tokens={batch.tokens} weights={batch.mixture_weights} counts={batch.source_counts()}")
        if batch.step >= 20:  # demo limit
            break

if __name__ == "__main__":
    main()
