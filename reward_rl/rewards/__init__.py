"""Reward function factory — builds a reward callable from config.

All reward weights live in YAML configs (Section 23 reproducibility rule:
"do not hardcode reward weights in Python"). This module is the single
entry point for creating reward functions from those configs.
"""

from __future__ import annotations

from typing import Any

from reward_rl.envs.key_door_env import RewardFn
from reward_rl.rewards.dense import DenseReward
from reward_rl.rewards.hybrid import HybridReward
from reward_rl.rewards.potential import PotentialReward
from reward_rl.rewards.sparse import SparseReward

AVAILABLE_TYPES = ("sparse", "dense", "potential", "hybrid")


def make_reward(reward_cfg: dict[str, Any], max_steps: int = 100) -> RewardFn:
    """Create a reward function from config.

    Args:
        reward_cfg: Dict with at minimum a ``type`` key (one of
                    ``AVAILABLE_TYPES``). Additional keys are passed as
                    kwargs to the reward class constructor.
        max_steps:  The episode step budget (needed for timeout detection).

    Returns:
        A callable matching the ``RewardFn`` signature.
    """
    rtype = reward_cfg.get("type", "sparse")
    params = {k: v for k, v in reward_cfg.items() if k != "type"}

    if rtype == "sparse":
        return SparseReward(
            success_reward=params.get("success_reward", 1.0),
            timeout_penalty=params.get("timeout_penalty", 0.0),
            step_cost=params.get("step_cost", 0.0),
            max_steps=max_steps,
        )

    if rtype == "dense":
        return DenseReward(
            key_pickup_reward=params.get("key_pickup_reward", 0.2),
            door_opened_reward=params.get("door_opened_reward", 0.3),
            invalid_action_penalty=params.get("invalid_action_penalty", -0.05),
            success_reward=params.get("success_reward", 1.0),
            timeout_penalty=params.get("timeout_penalty", 0.0),
            step_cost=params.get("step_cost", 0.0),
            max_steps=max_steps,
        )

    if rtype == "potential":
        return PotentialReward(
            gamma=params.get("gamma", 0.99),
            success_reward=params.get("success_reward", 1.0),
            timeout_penalty=params.get("timeout_penalty", 0.0),
            step_cost=params.get("step_cost", 0.0),
            max_steps=max_steps,
        )

    if rtype == "hybrid":
        return HybridReward(
            key_pickup_reward=params.get("key_pickup_reward", 0.2),
            door_opened_reward=params.get("door_opened_reward", 0.3),
            invalid_action_penalty=params.get("invalid_action_penalty", -0.05),
            step_cost=params.get("step_cost", -0.005),
            gamma=params.get("gamma", 0.99),
            use_potential=params.get("use_potential", True),
            success_reward=params.get("success_reward", 1.0),
            timeout_penalty=params.get("timeout_penalty", -0.1),
            max_steps=max_steps,
        )

    raise ValueError(
        f"Unknown reward type {rtype!r}. Available: {', '.join(AVAILABLE_TYPES)}"
    )

