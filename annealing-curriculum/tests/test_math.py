import pytest
from annealing_curriculum.math import cosine_decay, cosine_ramp, normalize_weights, compute_progress, quality_threshold

def test_cosine_decay_monotonic():
    a, b = 0.65, 0.10
    prev = cosine_decay(a,b,0.0)
    assert abs(prev - a) < 1e-6
    for p in [0.1,0.2,0.5,0.8,1.0]:
        cur = cosine_decay(a,b,p)
        assert cur <= prev + 1e-9
        prev = cur
    assert abs(cosine_decay(a,b,1.0) - b) < 1e-6

def test_cosine_ramp_monotonic():
    a,b = 0.15,0.45
    prev = cosine_ramp(a,b,0.0)
    assert abs(prev - a) < 1e-6
    for p in [0.1,0.3,0.6,1.0]:
        cur = cosine_ramp(a,b,p)
        assert cur >= prev - 1e-9
        prev = cur
    assert abs(cosine_ramp(a,b,1.0) - b) < 1e-6

def test_normalize_guard_zero():
    w = {"a":0.0, "b":0.0}
    norm = normalize_weights(w)
    assert abs(sum(norm.values())-1.0) < 1e-6
    assert all(v==0.5 for v in norm.values())

def test_progress_clipping():
    assert compute_progress(0,100) == 0.0
    assert compute_progress(100,100) == 1.0
    assert compute_progress(150,100) == 1.0

def test_quality_threshold():
    assert quality_threshold(0.3,0.5,0.0) == 0.3
    assert quality_threshold(0.3,0.5,1.0) == 0.8
    assert quality_threshold(0.9,0.5,1.0) == 1.0  # clipped
