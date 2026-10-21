import sys
sys.path.insert(0, "src")
from annealing_curriculum.config import AnnealingConfig, WeightSchedule, QualityConfig
from annealing_curriculum.curriculum import AnnealingCurriculum
from annealing_curriculum.monitoring import setup_logging
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    setup_logging()
    cfg = AnnealingConfig(
        total_annealing_tokens=50_000_000,
        tokens_per_step=8192,
        log_interval=5,
        schedule=WeightSchedule(
            initial_web_weight=0.65,
            final_web_weight=0.10,
            initial_synth_weight=0.15,
            final_synth_weight=0.45,
            domain_weight=0.25,
            instruction_weight=0.08,
            schedule_type="cosine"
        ),
        quality=QualityConfig(base_threshold=0.30, slope=0.40)
    )
    curriculum = AnnealingCurriculum(cfg)
    print(f"\n=== Annealing Demo: {cfg.max_steps} max steps ===")
    for batch in curriculum:
        if batch.step > args.steps:
            break
        print(f"Step {batch.step:3d} | prog {batch.progress:.2f} | weights {batch.mixture_weights} | tokens {batch.tokens} | avgQ {batch.avg_quality:.3f} | n {batch.num_examples}")
    print("\nDemo finished. Replace MockStreamingSource with HF datasets.")

if __name__ == "__main__":
    main()
