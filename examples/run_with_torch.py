import sys
sys.path.insert(0, "src")
try:
    import torch
    print(f"Torch {torch.__version__}")
    from annealing_curriculum.sampler import MixtureSampler
    sampler = MixtureSampler(backend="torch")
    weights = {"web":0.4, "synthetic":0.3, "domain":0.2, "instruction":0.1}
    for _ in range(10):
        print(sampler.sample_source(weights))
except ImportError:
    print("Torch not installed")
