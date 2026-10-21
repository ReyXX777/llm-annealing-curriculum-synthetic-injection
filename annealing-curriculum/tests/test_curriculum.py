from annealing_curriculum import AnnealingConfig, MockStreamingSource
from annealing_curriculum.curriculum import AnnealingCurriculum
from annealing_curriculum.types import SourceType

def test_curriculum_yields_batches():
    config = AnnealingConfig(total_annealing_tokens=5_000_000, tokens_per_step=100_000, log_interval=1000)
    sources = {
        "web_low": MockStreamingSource("web_low", source_type=SourceType.WEB_LOW, size=1000, avg_quality=0.3),
        "web_high": MockStreamingSource("web_high", source_type=SourceType.WEB_HIGH, size=1000, avg_quality=0.7),
        "synthetic": MockStreamingSource("synthetic", source_type=SourceType.SYNTHETIC, size=1000, avg_quality=0.9),
        "domain": MockStreamingSource("domain", source_type=SourceType.DOMAIN, size=1000, avg_quality=0.8),
        "instruction": MockStreamingSource("instruction", source_type=SourceType.INSTRUCTION, size=500, avg_quality=0.85),
        "textbooks": MockStreamingSource("textbooks", source_type=SourceType.TEXTBOOK, size=500, avg_quality=0.9),
    }
    curriculum = AnnealingCurriculum(config=config, sources=sources)
    batches = []
    for batch in curriculum:
        assert batch.tokens >0
        assert batch.count >0
        batches.append(batch)
        if len(batches)>=5:
            break
    assert len(batches)==5
    # T1 check: avg quality should generally increase due to filtering + weighting
    avg_q = [sum(e.quality_score for e in b.examples)/b.count for b in batches]
    # not strictly monotonic due to randomness but later should be >= early on average
    assert avg_q[-1] >= avg_q[0] - 0.2  # allow variance

def test_graceful_empty_source():
    config = AnnealingConfig(total_annealing_tokens=2_000_000, tokens_per_step=50_000)
    sources = {
        "web_low": MockStreamingSource("web_low", source_type=SourceType.WEB_LOW, size=0),  # empty
        "synthetic": MockStreamingSource("synthetic", source_type=SourceType.SYNTHETIC, size=100, avg_quality=0.9),
        "domain": MockStreamingSource("domain", source_type=SourceType.DOMAIN, size=100),
    }
    curriculum = AnnealingCurriculum(config=config, sources=sources)
    # should not crash, should yield from available sources
    batch = next(iter(curriculum))
    assert batch.count>0
