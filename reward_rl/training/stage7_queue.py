"""Resumable Stage 7 horizon-scaling queue using the locked core configs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python"
CORE = ("sparse", "dense", "potential", "hybrid")
HORIZONS = {
    "h10": "configs/env/stage7_h10_5keys.yaml",
    "h50": "configs/env/stage7_h50_5keys.yaml",
    "h100": "configs/env/stage7_h100_6keys.yaml",
}


def job_config(variant: str, horizon: str, steps: int, path: Path) -> None:
    with open(REPO_ROOT / "configs" / f"{variant}.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["meta"]["name"] = f"stage7_{horizon}_{variant}"
    config["env"]["config"] = HORIZONS[horizon]
    config["training"]["total_timesteps"] = steps
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)


def run_one(variant: str, horizon: str, seed: int, steps: int, resume: bool, output_root: Path) -> dict:
    run_dir = output_root / horizon / variant / f"seed_{seed}"
    marker = run_dir / "completion.json"
    if resume and marker.exists():
        with open(marker, "r", encoding="utf-8") as handle:
            if json.load(handle).get("timesteps") == steps:
                return {"status": "skipped", "variant": variant, "horizon": horizon, "seed": seed}
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "stage7_config.yaml"
    job_config(variant, horizon, steps, config_path)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TF_ENABLE_ONEDNN_OPTS": "0"})
    start = time.time()
    result = subprocess.run([str(PYTHON), "-m", "reward_rl.train", "--config", str(config_path),
                             "--seed", str(seed), "--output", str(run_dir.parent)], cwd=REPO_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.time() - start
    if result.returncode:
        return {"status": "error", "variant": variant, "horizon": horizon, "seed": seed,
                "elapsed_seconds": elapsed, "stderr": result.stderr.decode("utf-8", errors="replace")[-500:]}
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "timesteps": steps}, handle, indent=2, ensure_ascii=True)
    return {"status": "complete", "variant": variant, "horizon": horizon, "seed": seed,
            "elapsed_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--budgets-json", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.calibrate:
        jobs = [("dense", horizon, 0, 1_000_000) for horizon in HORIZONS]
        label = "calibration"
        output_root = args.output_root or (REPO_ROOT / "runs" / "stage7")
    else:
        if args.budgets_json is None:
            parser.error("--budgets-json is required for the matrix")
        with open(args.budgets_json, "r", encoding="utf-8") as handle:
            budgets = json.load(handle)
        jobs = [(variant, horizon, seed, int(budgets[horizon])) for horizon in budgets for variant in CORE for seed in range(3)]
        label = "matrix"
        output_root = args.output_root or (REPO_ROOT / "runs" / "stage7" / "final")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, *job, args.resume, output_root) for job in jobs]
        results = [future.result() for future in as_completed(futures)]
    out = output_root / f"queue_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)
    print(f"Saved {label} queue results to {out}")


if __name__ == "__main__":
    main()
