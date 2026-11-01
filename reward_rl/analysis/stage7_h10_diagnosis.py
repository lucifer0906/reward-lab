"""Evaluate saved H10 dense calibration checkpoints without training."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from reward_rl.analysis.stage6_zero_shot import evaluate_model


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    root = REPO_ROOT / "runs" / "stage7" / "h10" / "dense" / "seed_0"
    config = root / "stage7_config.yaml"
    layout = "configs/env/stage7_h10_5keys.yaml"
    output = REPO_ROOT / "runs" / "stage7" / "h10_diagnosis.csv"
    rows = []
    for step in (20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000):
        checkpoint = root / "checkpoints" / f"checkpoint_{step}.zip"
        deterministic, _ = evaluate_model(checkpoint, str(config), layout, 0.0, 200, 0, deterministic=True)
        stochastic, _ = evaluate_model(checkpoint, str(config), layout, 0.0, 100, 0, deterministic=False)
        rows.append({"step": step, "deterministic_episodes": 200,
                     "deterministic_success": float(np.mean([x["success"] for x in deterministic])),
                     "stochastic_episodes": 100,
                     "stochastic_success": float(np.mean([x["success"] for x in stochastic]))})
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Saved H10 diagnosis to {output}")


if __name__ == "__main__":
    main()
