from annealing_curriculum.config import AnnealingConfig
from annealing_curriculum.curriculum import AnnealingCurriculum

def test_curriculum_yields_batches():
    cfg = AnnealingConfig(total_annealing_tokens=10_000_000, tokens_per_step=4096, log_interval=1000)
    cur = AnnealingCurriculum(cfg)
    batches = []
    for i, batch in enumerate(cur):
        batches.append(batch)
        assert batch.tokens >= cfg.tokens_per_step or batch.num_examples>0
        assert batch.avg_quality >= 0
        if i>=5:
            break
    assert len(batches) > 0

def test_quality_threshold_rises():
    from annealing_curriculum.filters import QualityFilter
    from annealing_curriculum.config import QualityConfig
    qf = QualityFilter(QualityConfig(base_threshold=0.3, slope=0.4))
    assert qf.threshold_at(0.0) == 0.3
    assert qf.threshold_at(1.0) == 0.7
    assert qf.threshold_at(0.0) < qf.threshold_at(1.0)
