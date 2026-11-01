"""Evaluation-only action-noise wrapper.

The wrapped environment receives the executed action, so its EpisodeRecord and
the reward-blind evaluator remain replayable without changing environment physics.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ExecutedActionNoiseWrapper(gym.Wrapper):
    """Replace requested discrete actions at probability ``noise_probability``."""

    def __init__(self, env: gym.Env, noise_probability: float, rng_seed: int = 0) -> None:
        super().__init__(env)
        if not 0.0 <= noise_probability <= 1.0:
            raise ValueError("noise_probability must be in [0, 1]")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("ExecutedActionNoiseWrapper requires a Discrete action space")
        self.noise_probability = float(noise_probability)
        self.rng_seed = int(rng_seed)
        self._rng = np.random.default_rng(self.rng_seed)
        self.executed_actions: list[int] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(self.rng_seed + int(seed))
        self.executed_actions = []
        return self.env.reset(seed=seed, options=options)

    def step(self, action: int):
        requested = int(action)
        replaced = bool(self._rng.random() < self.noise_probability)
        executed = int(self._rng.integers(self.action_space.n)) if replaced else requested
        observation, reward, terminated, truncated, info = self.env.step(executed)
        self.executed_actions.append(executed)
        info = dict(info)
        info["requested_action"] = requested
        info["executed_action"] = executed
        info["action_was_replaced"] = replaced
        return observation, reward, terminated, truncated, info
