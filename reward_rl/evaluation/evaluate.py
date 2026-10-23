"""Evaluate a trained checkpoint against the ground-truth evaluator (Section 25).

Usage:
    python -m reward_rl.evaluate --checkpoint runs/sparse/seed_0/best_model/best_model
    python -m reward_rl.evaluate --checkpoint runs/sparse/seed_0/best_model/best_model --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from reward_rl.envs.key_door_env import DEFAULT_CONFIG, KeyDoorChainEnv
from reward_rl.envs.wrappers import FlattenObsWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_utf8_stdio() -> None:
    """Make direct module execution safe on Windows codepage consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


configure_utf8_stdio()


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    env_config: str | Path = DEFAULT_CONFIG,
    n_episodes: int = 100,
    deterministic: bool = True,
) -> dict:
    """Load a trained model and evaluate it with the ground-truth evaluator."""
    env = KeyDoorChainEnv.from_config(env_config)
    wrapped_env = FlattenObsWrapper(env)
    model = PPO.load(str(checkpoint_path), env=wrapped_env)

    successes = []
    episode_lengths = []
    subgoal_rates = []
    invalid_rates = []
    planning_efficiencies = []

    for ep in range(n_episodes):
        obs, _ = wrapped_env.reset(seed=ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated

        if "evaluation" in info:
            ev = info["evaluation"]
            successes.append(ev["success"])
            episode_lengths.append(ev["episode_length"])
            subgoal_rates.append(ev["subgoal_completion_rate"])
            invalid_rates.append(ev["invalid_action_rate"])
            if ev["success"]:
                planning_efficiencies.append(ev["planning_efficiency"])

    results = {
        "checkpoint": str(checkpoint_path),
        "n_episodes": n_episodes,
        "deterministic": deterministic,
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_subgoal_completion_rate": float(np.mean(subgoal_rates)),
        "mean_invalid_action_rate": float(np.mean(invalid_rates)),
        "mean_planning_efficiency": float(np.mean(planning_efficiencies)) if planning_efficiencies else 0.0,
        "successes": int(sum(successes)),
    }

    wrapped_env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained agent (Section 25).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Env config YAML")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic instead of deterministic policy")
    parser.add_argument("--output", type=str, default=None, help="Save results to this JSON file")
    args = parser.parse_args()

    results = evaluate_checkpoint(
        args.checkpoint,
        env_config=args.config,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
    )

    print(f"\nEvaluation Results ({args.episodes} episodes, {'deterministic' if not args.stochastic else 'stochastic'}):")
    print(f"  Success rate:              {results['success_rate']:.1%}")
    print(f"  Mean episode length:       {results['mean_episode_length']:.1f}")
    print(f"  Mean subgoal completion:   {results['mean_subgoal_completion_rate']:.1%}")
    print(f"  Mean invalid-action rate:  {results['mean_invalid_action_rate']:.1%}")
    print(f"  Mean planning efficiency:  {results['mean_planning_efficiency']:.1%}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {out}")


if __name__ == "__main__":
    main()
