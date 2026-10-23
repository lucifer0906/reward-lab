"""Parallel experiment runner - job queue with N workers (Stage 4).

Usage:
    # Pilot: 4 jobs (1 per variant, seed=42), 3 workers
    python run_experiments.py --mode pilot --workers 3

    # Full matrix: 20 jobs (4 variants x 5 seeds), 3 workers
    python run_experiments.py --mode full --workers 3 --timesteps 100000

Each worker is pinned to 1 thread (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1)
and uses device=cpu. Output goes to runs/<variant>/seed_<N>/.

Uses ThreadPoolExecutor to manage subprocess workers - each training job
runs as a fully independent subprocess with its own memory. No GIL issues
because the threads just wait on subprocess completion.
"""

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


def configure_utf8_stdio() -> None:
    """Make Windows console output deterministic for queue status messages."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


configure_utf8_stdio()

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python")
CONFIGS = [
    "configs/sparse.yaml",
    "configs/dense.yaml",
    "configs/potential.yaml",
    "configs/hybrid.yaml",
]
PILOT_SEED = 42
FINAL_SEEDS = [0, 1, 2, 3, 4]


def run_single_job(config_path: str, seed: int, timesteps: int | None = None) -> dict:
    """Run a single training job as a subprocess with thread pinning."""

    # Thread pinning - critical for parallel CPU runs
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    # Force UTF-8 everywhere on Windows
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Suppress TF/oneDNN warnings
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    variant_name = Path(config_path).stem
    tmp_config = None

    try:
        # If timesteps override, create a temp config
        if timesteps is not None:
            with open(REPO_ROOT / config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["training"]["total_timesteps"] = timesteps
            tmp_config = REPO_ROOT / "configs" / f"_tmp_{variant_name}_s{seed}.yaml"
            with open(tmp_config, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            actual_config = str(tmp_config)
        else:
            actual_config = config_path

        # Build command
        cmd = [PYTHON, "-m", "reward_rl.train", "--config", actual_config, "--seed", str(seed)]

        start = time.time()
        print(f"  [START] {variant_name} seed={seed} (timesteps={timesteps or 'from config'})", flush=True)

        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        elapsed = time.time() - start

        # Decode output with UTF-8, replacing errors
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        if result.returncode == 0:
            # Read results.json
            results_path = REPO_ROOT / "runs" / variant_name / f"seed_{seed}" / "results.json"
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    job_result = json.load(f)
                gt_rate = job_result.get("ground_truth", {}).get("success_rate", 0)
                print(f"  [DONE]  {variant_name} seed={seed} -> {gt_rate:.1%} success ({elapsed:.0f}s)", flush=True)
                return {
                    "status": "ok", "variant": variant_name, "seed": seed,
                    "elapsed": elapsed, "success_rate": gt_rate,
                }
            else:
                print(f"  [WARN]  {variant_name} seed={seed} finished but no results.json ({elapsed:.0f}s)", flush=True)
                return {"status": "no_results", "variant": variant_name, "seed": seed, "elapsed": elapsed}
        else:
            err_snippet = stderr[-500:] if stderr else stdout[-500:]
            print(f"  [FAIL]  {variant_name} seed={seed} (exit={result.returncode}, {elapsed:.0f}s)", flush=True)
            print(f"          {err_snippet[:200]}", flush=True)
            return {
                "status": "error", "variant": variant_name, "seed": seed,
                "elapsed": elapsed, "stderr": err_snippet,
            }

    except Exception as e:
        print(f"  [ERROR] {variant_name} seed={seed}: {e}", flush=True)
        return {"status": "exception", "variant": variant_name, "seed": seed, "error": str(e)}

    finally:
        # Clean up temp config
        if tmp_config and tmp_config.exists():
            tmp_config.unlink(missing_ok=True)


def run_job_queue(jobs: list[tuple[str, int]], workers: int, timesteps: int | None = None) -> list[dict]:
    """Run a list of (config_path, seed) jobs with N parallel workers."""
    print(f"\n{'='*60}")
    print(f"Job Queue: {len(jobs)} jobs, {workers} workers")
    if timesteps:
        print(f"Timesteps override: {timesteps:,}")
    print(f"{'='*60}\n", flush=True)

    start = time.time()
    results = []

    # ThreadPoolExecutor - threads just wait on subprocess.run(), no GIL issue
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_single_job, config, seed, timesteps): (config, seed)
            for config, seed in jobs
        }
        for future in as_completed(futures):
            result = future.result()  # run_single_job handles its own exceptions
            results.append(result)

    elapsed = time.time() - start
    ok_count = sum(1 for r in results if r.get("status") == "ok")

    print(f"\n{'='*60}")
    print(f"Queue complete: {ok_count}/{len(jobs)} succeeded in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if ok_count > 0:
        rates = [r["success_rate"] for r in results if r.get("status") == "ok"]
        times = [r["elapsed"] for r in results if r.get("status") == "ok"]
        print(f"  Avg success rate: {sum(rates)/len(rates):.1%}")
        print(f"  Avg training time: {sum(times)/len(times):.0f}s")
    print(f"{'='*60}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel experiment runner")
    parser.add_argument("--mode", choices=["pilot", "full", "custom"], default="pilot",
                        help="pilot: 4 jobs (seed=42), full: 20 jobs (5 seeds), custom: specify")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--configs", nargs="+", default=None, help="Config files (custom mode)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds (custom mode)")
    args = parser.parse_args()

    if args.mode == "pilot":
        jobs = [(cfg, PILOT_SEED) for cfg in CONFIGS]
        timesteps = args.timesteps or 100_000
    elif args.mode == "full":
        jobs = [(cfg, seed) for cfg in CONFIGS for seed in FINAL_SEEDS]
        timesteps = args.timesteps
        if timesteps is None:
            print("ERROR: --timesteps required for full mode. Run pilot first.")
            sys.exit(1)
    elif args.mode == "custom":
        configs = args.configs or CONFIGS
        seeds = args.seeds or [0]
        jobs = [(cfg, seed) for cfg in configs for seed in seeds]
        timesteps = args.timesteps

    results = run_job_queue(jobs, workers=args.workers, timesteps=timesteps)

    # Save queue results
    out_path = REPO_ROOT / "runs" / f"queue_results_{args.mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode, "workers": args.workers,
            "timesteps": timesteps, "results": results,
        }, f, indent=2, ensure_ascii=True)
    print(f"\nQueue results saved to: {out_path}")


if __name__ == "__main__":
    main()
