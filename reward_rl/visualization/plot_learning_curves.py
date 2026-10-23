"""Plot ground-truth success learning curves from TensorBoard event files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("sparse", "dense", "potential", "hybrid")
THRESHOLDS = (0.25, 0.50, 0.75, 0.90)


def latest_event_run(variant: str, seed: int) -> Path:
    """Return the newest TensorBoard run directory for one variant and seed."""
    log_root = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "tb_logs"
    candidates = [path for path in log_root.glob(f"seed_{seed}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard run found for {variant}, seed {seed}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_success_curve(variant: str, seed: int) -> list[tuple[int, float]]:
    """Load ground-truth success values, never reward-derived proxy metrics."""
    accumulator = EventAccumulator(str(latest_event_run(variant, seed)))
    accumulator.Reload()
    return [(event.step, float(event.value)) for event in accumulator.Scalars("ground_truth/success_rate")]


def first_threshold_steps(curve: list[tuple[int, float]]) -> dict[str, int | None]:
    """Return first logged step at or above each target success rate."""
    return {
        f"{int(threshold * 100)}%": next(
            (step for step, success_rate in curve if success_rate >= threshold), None
        )
        for threshold in THRESHOLDS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ground-truth success learning curves")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs" / "pilot_ground_truth_success_curve.png")
    parser.add_argument("--metrics-output", type=Path, default=REPO_ROOT / "runs" / "pilot_learning_metrics.json")
    args = parser.parse_args()

    curves = {variant: load_success_curve(variant, args.seed) for variant in VARIANTS}
    metrics = {
        variant: {
            "seed": args.seed,
            "final_logged_success_rate": curve[-1][1],
            "steps_to_success_rate": first_threshold_steps(curve),
        }
        for variant, curve in curves.items()
    }

    figure, axis = plt.subplots(figsize=(10, 6))
    for variant, curve in curves.items():
        steps, success_rates = zip(*curve)
        axis.plot(steps, success_rates, label=variant)
    axis.set_title(f"Ground-truth success rate during Stage 4 pilot (seed {args.seed})")
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Rolling ground-truth success rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2, ensure_ascii=True)
    print(f"Saved plot: {args.output}")
    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
