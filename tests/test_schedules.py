import math
from annealing_curriculum.schedules import cosine_decay, cosine_ramp

def test_cosine_decay_monotonic():
    a, b = 0.65, 0.10
    prev = cosine_decay(a,b,0.0)
    for p in [0.1,0.2,0.5,0.9,1.0]:
        cur = cosine_decay(a,b,p)
        assert cur <= prev + 1e-9
        prev = cur
    assert math.isclose(cosine_decay(a,b,0.0), a)
    assert math.isclose(cosine_decay(a,b,1.0), b)

def test_cosine_ramp_monotonic():
    a,b = 0.15,0.45
    prev = cosine_ramp(a,b,0.0)
    for p in [0.2,0.5,0.8,1.0]:
        cur = cosine_ramp(a,b,p)
        assert cur >= prev - 1e-9
        prev = cur
    assert math.isclose(cosine_ramp(a,b,0.0), a)
    assert math.isclose(cosine_ramp(a,b,1.0), b)

def test_normalization_safety():
    from annealing_curriculum.config import AnnealingConfig, WeightSchedule
    from annealing_curriculum.scheduler import AnnealingScheduler
    cfg = AnnealingConfig()
    cfg.schedule = WeightSchedule(domain_weight=0, instruction_weight=0, initial_web_weight=0, final_web_weight=0, initial_synth_weight=0, final_synth_weight=0)
    sched = AnnealingScheduler(cfg)
    w = sched.normalized_weights(0.5)
    assert abs(sum(w.values())-1.0) < 1e-6
