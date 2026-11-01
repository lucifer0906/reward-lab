"""Stage 6 follow-up: high action noise and stochastic collapsed-policy checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from reward_rl.analysis.stage6_zero_shot import CORE, LAYOUTS, bootstrap_ci, evaluate_model, retained_seeds


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    output = REPO_ROOT / "runs" / "stage6_followup"
    output.mkdir(parents=True, exist_ok=True)
    retained = retained_seeds()
    rows: list[dict] = []
    for variant, seeds in retained.items():
        for seed in sorted(seeds):
            checkpoint = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            for noise in (0.3, 0.4, 0.5):
                episodes, _ = evaluate_model(checkpoint, CORE[variant], LAYOUTS["base"], noise, 100, seed)
                rows.append({"variant": variant, "seed": seed, "noise_probability": noise,
                             "success_rate": float(np.mean([episode["success"] for episode in episodes]))})
    summary = []
    for variant in CORE:
        for noise in (0.3, 0.4, 0.5):
            values = [row["success_rate"] for row in rows if row["variant"] == variant and row["noise_probability"] == noise]
            ci = bootstrap_ci(values)
            summary.append({"variant": variant, "noise_probability": noise, "n_seeds": len(values),
                            "success_rate": float(np.mean(values)), "success_ci_low": ci[0], "success_ci_high": ci[1]})
    collapsed = []
    for variant, seed in (("dense", 3), ("hybrid", 0), ("hybrid", 4), ("potential", 9)):
        checkpoint = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "checkpoints" / "final_model.zip"
        episodes, _ = evaluate_model(checkpoint, CORE[variant], LAYOUTS["base"], 0.0, 100, seed, deterministic=False)
        collapsed.append({"variant": variant, "seed": seed, "policy": "stochastic", "episodes": 100,
                          "success_rate": float(np.mean([episode["success"] for episode in episodes]))})
    for filename, data in (("noise_per_seed.csv", rows), ("noise_summary.csv", summary), ("collapsed_stochastic.csv", collapsed)):
        with open(output / filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump({"noise_summary": summary, "collapsed_stochastic": collapsed}, handle, indent=2, ensure_ascii=True)
    print(f"Saved Stage 6 follow-up report to {output}")


if __name__ == "__main__":
    main()
