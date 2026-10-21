from annealing_curriculum import AnnealingConfig
from annealing_curriculum.scheduling.scheduler import AnnealingScheduler

def test_scheduler_weights_sum_to_one():
    config = AnnealingConfig(total_annealing_tokens=10_000_000, tokens_per_step=1_000_000)
    sched = AnnealingScheduler(config)
    for step in [1,10,50,100]:
        w = sched.get_weights(step)
        assert abs(sum(w.values())-1.0) < 1e-6
        assert all(v>=0 for v in w.values())

def test_web_decay_synth_ramp():
    config = AnnealingConfig()
    sched = AnnealingScheduler(config)
    w0 = sched.get_weights(0)
    w_mid = sched.get_weights(sched.max_steps//2)
    w_end = sched.get_weights(sched.max_steps)
    # web should decay overall, synth should ramp
    assert w_end["web"] < w0["web"]
    assert w_end["synthetic"] > w0["synthetic"]
