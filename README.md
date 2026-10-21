# Curriculum Design & Data Scheduling — Annealing Phase

Standard implementation of the annealing + synthetic injection curriculum.

**Goal:** Finish pre-training with a high-quality annealing mix (50B-200B tokens) that heavily weights textbook-quality synthetic data + specialized domain data while decaying lower-quality web text.

## Formal Objective
For annealing steps Tau = {1..N}, mixture weights w_s(t) over sources D = {W_L, W_H, T, S, D, I}:

- A1 Normalization: sum_s w_s(t) = 1
- A2 Web decay: w_{W_L}(t) = CosineDecay(w_init, w_final, t/N)
- A3 Synth ramp: w_{S∪T}(t) = CosineRamp(...)
- A4 Quality filter: threshold q(t) = q0 + alpha * progress
- A5 Budget: sum Tokens(k) = B in [50e9,200e9]
- A6 Sampling: P(source=s at t) = w_s(t)

Derived: E[Q(batch(t))] is monotone non-decreasing, lim_{t->N} w_{W_L}(t) <= 0.10

## File Tree
```
annealing-curriculum/
├── src/annealing_curriculum/
│   ├── config.py
│   ├── math.py
│   ├── types.py
│   ├── exceptions.py
│   ├── sources/
│   ├── quality/
│   ├── scheduling/
│   └── curriculum/
├── configs/default.yaml
├── examples/
├── tests/
```

## Install
```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Usage
```python
from annealing_curriculum import AnnealingConfig, AnnealingCurriculum, MockStreamingSource

config = AnnealingConfig(total_annealing_tokens=100e9, tokens_per_step=2_000_000)
curriculum = AnnealingCurriculum(config=config, sources={
    "web_low": MockStreamingSource("web_low", size=1_000_000),
    "web_high": MockStreamingSource("web_high", size=500_000),
    "textbooks": MockStreamingSource("textbooks", size=200_000),
    "synthetic": MockStreamingSource("synthetic", size=1_000_000),
    "domain": MockStreamingSource("domain", size=500_000),
    "instruction": MockStreamingSource("instruction", size=100_000),
})

for batch in curriculum:
    train_step(batch)
    if curriculum.current_step >= curriculum.max_steps:
        break
```

## Design Principles Applied
1. Classes over monolithic scripts
2. dataclasses for config
3. Generators (yield) for streaming
4. NumPy vectorized sampling
5. Division-by-zero guards
6. logging instead of print
7. Full type hints
8. Pure math separated from state
9. Docstrings with mathematical ranges
10. Edge-case handling
```
