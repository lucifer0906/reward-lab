"""Custom SB3 callbacks for logging, evaluation, and checkpointing.

These callbacks satisfy the founder spec requirements:
    - Section 11: Log reward components independently
    - Section 13: Track policy loss, value loss, entropy, episode reward,
                  episode length, success rate, environment steps
    - Section 23: Save model checkpoints, store evaluation results separately
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class RewardComponentLogger(BaseCallback):
    """Log individual reward components to TensorBoard (Section 11).

    Reads ``info["reward_components"]`` from the training env at each step
    and logs running means per component. This is how we later diagnose
    whether a policy is learning intended behavior or exploiting one component.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self._component_sums: dict[str, float] = {}
        self._component_counts: dict[str, int] = {}
        self._total_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            components = info.get("reward_components", {})
            for name, value in components.items():
                self._component_sums[name] = self._component_sums.get(name, 0.0) + value
                self._component_counts[name] = self._component_counts.get(name, 0) + 1
        self._total_steps += 1

        if self._total_steps % self.log_freq == 0 and self._component_counts:
            for name in self._component_counts:
                mean = self._component_sums[name] / self._component_counts[name]
                self.logger.record(f"reward_components/{name}", mean)
            self._component_sums.clear()
            self._component_counts.clear()
        return True


class SuccessRateLogger(BaseCallback):
    """Log ground-truth success rate from the evaluator (not from reward).

    Reads ``info["is_success"]`` (set by the env's ground-truth evaluator)
    at episode ends and logs rolling success rate. Also logs evaluation
    metrics like subgoal completion, invalid-action rate, etc.
    """

    def __init__(self, window: int = 100, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self.log_freq = log_freq
        self._successes: list[bool] = []
        self._eval_metrics: list[dict] = []
        self._total_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done and "is_success" in info:
                self._successes.append(info["is_success"])
                if "evaluation" in info:
                    self._eval_metrics.append(info["evaluation"])

        self._total_steps += 1
        if self._total_steps % self.log_freq == 0 and self._successes:
            recent = self._successes[-self.window:]
            self.logger.record("ground_truth/success_rate", np.mean(recent))
            self.logger.record("ground_truth/total_episodes", len(self._successes))

            if self._eval_metrics:
                recent_eval = self._eval_metrics[-self.window:]
                self.logger.record(
                    "ground_truth/mean_subgoal_rate",
                    np.mean([e["subgoal_completion_rate"] for e in recent_eval]),
                )
                self.logger.record(
                    "ground_truth/mean_invalid_action_rate",
                    np.mean([e["invalid_action_rate"] for e in recent_eval]),
                )
                self.logger.record(
                    "ground_truth/mean_episode_length",
                    np.mean([e["episode_length"] for e in recent_eval]),
                )
        return True

    def get_results(self) -> dict:
        """Return accumulated results for post-training analysis."""
        return {
            "total_episodes": len(self._successes),
            "successes": sum(self._successes),
            "success_rate": float(np.mean(self._successes)) if self._successes else 0.0,
        }


class CheckpointAndMetricsCallback(BaseCallback):
    """Save model checkpoints and evaluation metrics at intervals (Section 23)."""

    def __init__(self, save_path: str | Path, save_freq: int = 10000,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self._eval_results: list[dict] = []

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_path / f"checkpoint_{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose:
                print(f"Saved checkpoint: {path}")
        return True

    def _on_training_end(self) -> None:
        # Save final model
        path = self.save_path / "final_model"
        self.model.save(str(path))
        if self.verbose:
            print(f"Saved final model: {path}")
