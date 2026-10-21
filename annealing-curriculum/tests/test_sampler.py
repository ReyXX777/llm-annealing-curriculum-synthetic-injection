from annealing_curriculum.curriculum.sampler import WeightedSampler

def test_sampler_distribution():
    sampler = WeightedSampler(seed=42)
    weights = {"web":0.1, "synthetic":0.6, "domain":0.3}
    samples = sampler.sample_sources(weights, 1000)
    assert len(samples)==1000
    # synthetic should be most common
    from collections import Counter
    c = Counter(samples)
    assert c["synthetic"] > c["web"]
    assert c["synthetic"] > 400

def test_sampler_one():
    sampler = WeightedSampler(seed=1)
    w = {"a":0.0, "b":1.0}
    assert sampler.sample_one(w) == "b"
