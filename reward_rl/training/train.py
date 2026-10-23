"""Stable-Baselines3 PPO training script (Section 13, Section 25).

Usage (founder's exact CLI from Section 25):
    python -m reward_rl.train --config configs/sparse.yaml
    python -m reward_rl.train --config configs/sparse.yaml --seed 42
    python -m reward_rl.train --config configs/sparse.yaml --seed 0 1 2 3 4

The config YAML contains:
    env:      environment config (layout, max_steps)
    reward:   reward type + weights
    training: PPO hyperparameters, total timesteps, eval frequency
    meta:     experiment name, seeds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from reward_rl.envs.key_door_env import DEFAULT_CONFIG, KeyDoorChainEnv, load_env_config
from reward_rl.envs.wrappers import FlattenObsWrapper
from reward_rl.rewards import make_reward
from reward_rl.training.callbacks import (
    CheckpointAndMetricsCallback,
    RewardComponentLogger,
    SuccessRateLogger,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_utf8_stdio() -> None:
    """Make direct module execution safe on Windows codepage consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


configure_utf8_stdio()


def make_env(env_cfg: dict, reward_fn, seed: int) -> Monitor:
    """Create a wrapped, monitored environment."""
    env_config_path = env_cfg.get("config", str(DEFAULT_CONFIG))
    # Resolve relative paths against repo root
    if not Path(env_config_path).is_absolute():
        env_config_path = str(REPO_ROOT / env_config_path)

    env = KeyDoorChainEnv.from_config(
        env_config_path,
        reward_fn=reward_fn,
    )
    env = FlattenObsWrapper(env)
    env = Monitor(env, info_keywords=("is_success",))
    env.reset(seed=seed)
    return env


def make_eval_env(env_cfg: dict, reward_fn, seed: int) -> Monitor:
    """Create a separate evaluation env (deterministic, per Section 13)."""
    env = make_env(env_cfg, reward_fn, seed + 10000)
    return env


def train_single_seed(config: dict, seed: int, run_dir: Path) -> dict:
    """Train PPO with one seed and return summary metrics."""
    import os
    import torch
    # Pin to 1 thread when running parallel workers (set by run_experiments.py)
    num_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    set_random_seed(seed)

    # --- Build env + reward ---
    env_cfg = config.get("env", {})
    reward_cfg = config.get("reward", {"type": "sparse"})
    training_cfg = config.get("training", {})

    env_config_path = env_cfg.get("config", str(DEFAULT_CONFIG))
    if not Path(env_config_path).is_absolute():
        env_config_path = str(REPO_ROOT / env_config_path)
    loaded_env_cfg = load_env_config(env_config_path)
    max_steps = loaded_env_cfg.get("max_steps", 100)

    reward_fn = make_reward(reward_cfg, max_steps=max_steps)
    env = make_env(env_cfg, reward_fn, seed)
    eval_env = make_eval_env(env_cfg, reward_fn, seed)

    # --- PPO hyperparameters (Section 13: keep fixed across reward experiments) ---
    ppo_kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": env,
        "seed": seed,
        "device": "cpu",
        "verbose": 1,
        "tensorboard_log": str(run_dir / "tb_logs"),
        # Defaults tuned for CPU training on small envs
        "learning_rate": training_cfg.get("learning_rate", 3e-4),
        "n_steps": training_cfg.get("n_steps", 2048),
        "batch_size": training_cfg.get("batch_size", 64),
        "n_epochs": training_cfg.get("n_epochs", 10),
        "gamma": training_cfg.get("gamma", 0.99),
        "gae_lambda": training_cfg.get("gae_lambda", 0.95),
        "clip_range": training_cfg.get("clip_range", 0.2),
        "ent_coef": training_cfg.get("ent_coef", 0.01),
        "vf_coef": training_cfg.get("vf_coef", 0.5),
        "max_grad_norm": training_cfg.get("max_grad_norm", 0.5),
        "policy_kwargs": dict(
            net_arch=training_cfg.get("net_arch", [64, 64]),
        ),
    }

    total_timesteps = training_cfg.get("total_timesteps", 100_000)
    eval_freq = training_cfg.get("eval_freq", 5000)
    n_eval_episodes = training_cfg.get("n_eval_episodes", 20)
    checkpoint_freq = training_cfg.get("checkpoint_freq", 10000)
    log_freq = training_cfg.get("log_freq", 1000)

    model = PPO(**ppo_kwargs)

    # --- Callbacks ---
    reward_logger = RewardComponentLogger(log_freq=log_freq)
    success_logger = SuccessRateLogger(log_freq=log_freq)
    checkpoint_cb = CheckpointAndMetricsCallback(
        save_path=run_dir / "checkpoints",
        save_freq=checkpoint_freq,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,  # Section 13: deterministic evaluation
        verbose=1,
    )

    callbacks = CallbackList([reward_logger, success_logger, checkpoint_cb, eval_cb])

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"Training: reward={reward_cfg.get('type', 'sparse')} | seed={seed}")
    print(f"  total_timesteps={total_timesteps} | run_dir={run_dir}")
    print(f"{'='*60}\n")

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=f"seed_{seed}")
    elapsed = time.time() - start

    # --- Save final results ---
    results = {
        "seed": seed,
        "reward_type": reward_cfg.get("type", "sparse"),
        "total_timesteps": total_timesteps,
        "training_time_seconds": round(elapsed, 1),
        "ground_truth": success_logger.get_results(),
        "config": config,
    }

    results_path = run_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    env.close()
    eval_env.close()

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Ground-truth success rate: {results['ground_truth']['success_rate']:.1%}")
    print(f"  Results saved to: {results_path}")
    return results


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on Key-Door-Chain with configurable rewards (Section 25).",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config (e.g. configs/sparse.yaml)",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=None,
        help="Override seed(s) from config. E.g. --seed 0 1 2 3 4",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output directory (default: runs/<experiment_name>)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    meta = config.get("meta", {})
    experiment_name = meta.get("name", Path(args.config).stem)

    seeds = args.seed or meta.get("seeds", [0])
    output_base = Path(args.output) if args.output else REPO_ROOT / "runs" / experiment_name

    all_results = []
    for seed in seeds:
        run_dir = output_base / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save the full config used (reproducibility, Section 23)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

        result = train_single_seed(config, seed, run_dir)
        all_results.append(result)

    # --- Summary across seeds ---
    if len(all_results) > 1:
        rates = [r["ground_truth"]["success_rate"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"Multi-seed summary ({len(seeds)} seeds):")
        print(f"  Success rate: {np.mean(rates):.1%} +/- {np.std(rates):.1%}")
        print(f"{'='*60}")

        summary = {
            "experiment": experiment_name,
            "seeds": seeds,
            "mean_success_rate": float(np.mean(rates)),
            "std_success_rate": float(np.std(rates)),
            "per_seed": all_results,
        }
        with open(output_base / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
