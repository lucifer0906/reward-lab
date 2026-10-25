"""Aggregate Stage 4 learning curves and seed-level metrics.

All success measurements come from the reward-blind ground-truth evaluator
logged during training. Reward values are deliberately not used as a proxy for
task success.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("sparse", "dense", "potential", "hybrid")
SUCCESS_TAG = "ground_truth/success_rate"
METRIC_TAGS = {
    "success_rate": SUCCESS_TAG,
    "invalid_action_rate": "ground_truth/mean_invalid_action_rate",
    "subgoal_completion_rate": "ground_truth/mean_subgoal_rate",
}
THRESHOLDS = (0.25, 0.50, 0.75, 0.90)


def latest_event_run(variant: str, seed: int) -> Path:
    """Find the newest TensorBoard run for a completed variant/seed pair."""
    root = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "tb_logs"
    candidates = [path for path in root.glob(f"seed_{seed}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard logs for {variant} seed {seed}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_scalar_curve(variant: str, seed: int, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one scalar TensorBoard curve as sorted step/value arrays."""
    accumulator = EventAccumulator(str(latest_event_run(variant, seed)))
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    if not events:
        raise ValueError(f"No {tag} values for {variant} seed {seed}")
    return (
        np.asarray([event.step for event in events], dtype=float),
        np.asarray([event.value for event in events], dtype=float),
    )


def threshold_steps(steps: np.ndarray, values: np.ndarray) -> dict[str, int | None]:
    """Return the first logged step that reaches each target success level."""
    return {
        f"{int(threshold * 100)}%": next(
            (int(step) for step, value in zip(steps, values) if value >= threshold), None
        )
        for threshold in THRESHOLDS
    }


def normalized_auc(steps: np.ndarray, values: np.ndarray, budget: int) -> float:
    """Integrate a curve over the common training budget on a 0-to-1 scale."""
    grid = np.concatenate(([0.0], steps[steps <= budget], [float(budget)]))
    grid = np.unique(grid)
    interpolated = np.interp(grid, steps, values, left=values[0], right=values[-1])
    return float(np.trapezoid(interpolated, grid) / budget)


def summarize_variant(variant: str, seeds: list[int], budget: int) -> tuple[dict, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """Compute per-seed Stage 4 metrics and retain curves for aggregate plots."""
    per_seed: list[dict] = []
    curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in METRIC_TAGS}

    for seed in seeds:
        success_steps, success_values = load_scalar_curve(variant, seed, SUCCESS_TAG)
        result_path = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "results.json"
        with open(result_path, "r", encoding="utf-8") as file_handle:
            training_result = json.load(file_handle)

        per_seed.append({
            "seed": seed,
            "whole_training_success_rate": training_result["ground_truth"]["success_rate"],
            "final_logged_success_rate": float(success_values[-1]),
            "success_auc": normalized_auc(success_steps, success_values, budget),
            "steps_to_success_rate": threshold_steps(success_steps, success_values),
            "training_time_seconds": training_result["training_time_seconds"],
        })
        curves["success_rate"].append((success_steps, success_values))
        for name, tag in METRIC_TAGS.items():
            if name == "success_rate":
                continue
            curves[name].append(load_scalar_curve(variant, seed, tag))

    def mean_std(field: str) -> dict[str, float]:
        values = np.asarray([item[field] for item in per_seed], dtype=float)
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    threshold_summary = {}
    for threshold in ("25%", "50%", "75%", "90%"):
        reached = [item["steps_to_success_rate"][threshold] for item in per_seed]
        valid = [step for step in reached if step is not None]
        threshold_summary[threshold] = {
            "reached_seeds": len(valid),
            "total_seeds": len(seeds),
            "mean_steps_if_reached": float(np.mean(valid)) if valid else None,
            "per_seed": reached,
        }

    return {
        "per_seed": per_seed,
        "whole_training_success_rate": mean_std("whole_training_success_rate"),
        "final_logged_success_rate": mean_std("final_logged_success_rate"),
        "success_auc": mean_std("success_auc"),
        "training_time_seconds": mean_std("training_time_seconds"),
        "thresholds": threshold_summary,
    }, curves


def plot_metric(metric_name: str, curves_by_variant: dict[str, list[tuple[np.ndarray, np.ndarray]]], budget: int, output_dir: Path) -> None:
    """Draw mean plus/minus one standard deviation on a common step grid."""
    figure, axis = plt.subplots(figsize=(10, 6))
    grid = np.linspace(0, budget, 401)
    for variant, seed_curves in curves_by_variant.items():
        stacked = np.vstack([
            np.interp(grid, steps, values, left=values[0], right=values[-1])
            for steps, values in seed_curves
        ])
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        axis.plot(grid, mean, label=variant)
        axis.fill_between(grid, np.maximum(0.0, mean - std), np.minimum(1.0, mean + std), alpha=0.18)
    labels = {
        "success_rate": "Ground-truth rolling success rate",
        "invalid_action_rate": "Invalid-action rate",
        "subgoal_completion_rate": "Subgoal completion rate",
    }
    axis.set_title(f"Stage 4: {labels[metric_name]} (mean +/- std, 5 seeds)")
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel(labels[metric_name])
    axis.set_xlim(0, budget)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"stage4_{metric_name}.png", dpi=160)
    plt.close(figure)


def write_summary_csv(summary: dict[str, dict], output_path: Path) -> None:
    """Write a compact table for the primary Stage 4 comparisons."""
    with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=[
            "variant", "whole_training_success_mean", "whole_training_success_std",
            "final_logged_success_mean", "final_logged_success_std", "success_auc_mean",
            "success_auc_std", "25pct_reached_seeds", "50pct_reached_seeds",
            "75pct_reached_seeds", "90pct_reached_seeds",
        ])
        writer.writeheader()
        for variant, metrics in summary.items():
            writer.writerow({
                "variant": variant,
                "whole_training_success_mean": metrics["whole_training_success_rate"]["mean"],
                "whole_training_success_std": metrics["whole_training_success_rate"]["std"],
                "final_logged_success_mean": metrics["final_logged_success_rate"]["mean"],
                "final_logged_success_std": metrics["final_logged_success_rate"]["std"],
                "success_auc_mean": metrics["success_auc"]["mean"],
                "success_auc_std": metrics["success_auc"]["std"],
                "25pct_reached_seeds": metrics["thresholds"]["25%"]["reached_seeds"],
                "50pct_reached_seeds": metrics["thresholds"]["50%"]["reached_seeds"],
                "75pct_reached_seeds": metrics["thresholds"]["75%"]["reached_seeds"],
                "90pct_reached_seeds": metrics["thresholds"]["90%"]["reached_seeds"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage 4 full-matrix results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "stage4_analysis")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    metric_curves: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        name: {} for name in METRIC_TAGS
    }
    for variant in VARIANTS:
        summary[variant], curves = summarize_variant(variant, args.seeds, args.timesteps)
        for metric_name, seed_curves in curves.items():
            metric_curves[metric_name][variant] = seed_curves

    for metric_name, curves_by_variant in metric_curves.items():
        plot_metric(metric_name, curves_by_variant, args.timesteps, args.output_dir)
    with open(args.output_dir / "stage4_metrics.json", "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, ensure_ascii=True)
    write_summary_csv(summary, args.output_dir / "stage4_summary.csv")
    print(f"Saved Stage 4 metrics and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
