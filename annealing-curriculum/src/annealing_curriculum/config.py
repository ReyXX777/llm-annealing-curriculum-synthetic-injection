"""
Configuration dataclasses — groups magic numbers, validates ranges.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import yaml
import pathlib

@dataclass
class WeightScheduleConfig:
    """Schedule hyper-parameters for w_s(t)."""
    initial_web_weight: float = 0.65  # w_L^init in [0,1], must > final
    final_web_weight: float = 0.10    # w_L^final
    initial_synth_weight: float = 0.15  # w_S^init
    final_synth_weight: float = 0.45    # w_S^final, must > init
    domain_weight: float = 0.25
    instruction_weight: float = 0.10

    def __post_init__(self) -> None:
        for name in ["initial_web_weight", "final_web_weight", "initial_synth_weight",
                      "final_synth_weight", "domain_weight", "instruction_weight"]:
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {v}")
        if not (self.initial_web_weight > self.final_web_weight):
            raise ValueError("initial_web_weight must be > final_web_weight for decay (A2)")
        if not (self.final_synth_weight > self.initial_synth_weight):
            raise ValueError("final_synth_weight must be > initial_synth_weight for ramp (A3)")

@dataclass
class QualityConfig:
    """Quality filtering config for A4."""
    base_threshold: float = 0.3  # q0 in [0,1]
    alpha: float = 0.5  # slope for q(t)=q0+alpha*progress, alpha>0
    enabled: bool = True
    dedup_enabled: bool = True
    min_tokens: int = 16
    max_tokens: int = 8192

    def __post_init__(self) -> None:
        if not (0.0 <= self.base_threshold <= 1.0):
            raise ValueError("base_threshold must be in [0,1]")
        if self.alpha < 0:
            raise ValueError("alpha must be >=0")
        if self.min_tokens <= 0 or self.max_tokens <=0:
            raise ValueError("token bounds must be >0")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens > max_tokens")

@dataclass
class AnnealingConfig:
    """
    Top-level annealing config encapsulating:
    - Budget B in [50e9,200e9] (A5)
    - Weight schedule (A2,A3)
    - Quality (A4)
    """
    total_annealing_tokens: int = 100_000_000_000  # B, 100B default
    tokens_per_step: int = 2_097_152  # ~2M tokens per step
    weight_schedule: WeightScheduleConfig = field(default_factory=WeightScheduleConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    log_interval: int = 100
    seed: int = 42
    use_torch_sampler: bool = False  # if True, use torch.multinomial else numpy

    def __post_init__(self) -> None:
        if not (50_000_000_000 <= self.total_annealing_tokens <= 200_000_000_000):
            # allow outside but warn via validation: for research we keep soft check
            if self.total_annealing_tokens <= 0:
                raise ValueError("total_annealing_tokens must be >0")
        if self.tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be >0")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be >0")

    @property
    def max_steps(self) -> int:
        return max(1, self.total_annealing_tokens // self.tokens_per_step)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "AnnealingConfig":
        p = pathlib.Path(path)
        data = yaml.safe_load(p.read_text())
        ws = data.get("weight_schedule", {})
        # support flat yaml as in default.yaml
        if not ws:
            ws = {
                "initial_web_weight": data.get("initial_web_weight", 0.65),
                "final_web_weight": data.get("final_web_weight", 0.10),
                "initial_synth_weight": data.get("initial_synth_weight", 0.15),
                "final_synth_weight": data.get("final_synth_weight", 0.45),
                "domain_weight": data.get("domain_weight", 0.25),
                "instruction_weight": data.get("instruction_weight", 0.10),
            }
        q = QualityConfig(
            base_threshold=data.get("quality_base_threshold", 0.3),
            alpha=data.get("quality_alpha", 0.5),
            enabled=data.get("quality_filter_enabled", True),
            dedup_enabled=data.get("dedup_enabled", True),
            min_tokens=data.get("min_example_tokens", 16),
            max_tokens=data.get("max_example_tokens", 8192),
        )
        return cls(
            total_annealing_tokens=data.get("total_annealing_tokens", 100_000_000_000),
            tokens_per_step=data.get("tokens_per_step", 2097152),
            weight_schedule=WeightScheduleConfig(**ws),
            quality=q,
            log_interval=data.get("log_interval", 100),
            seed=data.get("seed", 42),
        )
