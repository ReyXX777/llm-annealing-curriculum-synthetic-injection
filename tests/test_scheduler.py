from annealing_curriculum.config import AnnealingConfig
from annealing_curriculum.scheduler import AnnealingScheduler

def test_scheduler_progress():
    cfg = AnnealingConfig(total_annealing_tokens=10_000_000, tokens_per_step=1000)
    sched = AnnealingScheduler(cfg)
    assert sched.progress(0) == 0
    assert sched.progress(cfg.max_steps) == 1.0
    assert sched.progress(cfg.max_steps*2) == 1.0

def test_weights_sum_one():
    cfg = AnnealingConfig()
    sched = AnnealingScheduler(cfg)
    for step in [1, 10, 100, 500]:
        p,w = sched.get_weights(step)
        assert abs(sum(w.values())-1.0) < 1e-6
        assert all(v>=0 for v in w.values())

def test_expected_quality_increases():
    cfg = AnnealingConfig()
    sched = AnnealingScheduler(cfg)
    _, w0 = sched.get_weights(1)
    _, w_end = sched.get_weights(cfg.max_steps)
    assert sched.expected_quality_proxy(w_end) >= sched.expected_quality_proxy(w0)
