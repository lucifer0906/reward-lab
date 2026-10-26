"""Stage 5b incentive tests for the deliberately adversarial distance rewards."""

from __future__ import annotations

import pytest

from reward_rl.analysis.reward_hacking_audit import (
    distance_oscillation_actions,
    run_trajectory,
    solve_actions,
)
from reward_rl.rewards import make_reward


def test_factory_creates_distance_rewards():
    assert make_reward({"type": "naive_distance"}).__class__.__name__ == "NaiveDistanceReward"
    assert make_reward({"type": "potential_distance"}).__class__.__name__ == "PotentialDistanceReward"


def test_naive_distance_farming_exceeds_success(base_layout):
    cycle = distance_oscillation_actions(base_layout, 100)
    farmed = run_trajectory(
        base_layout, 100,
        {"type": "naive_distance", "distance_decrease_reward": 0.1, "success_reward": 1.0},
        cycle,
    )
    successful = run_trajectory(
        base_layout, 100,
        {"type": "naive_distance", "distance_decrease_reward": 0.1, "success_reward": 1.0},
        solve_actions(base_layout, 100),
    )
    assert not farmed["success"]
    assert farmed["proxy_return"] > successful["proxy_return"]


def test_potential_distance_oscillation_is_nonpositive(base_layout):
    result = run_trajectory(
        base_layout, 100,
        {"type": "potential_distance", "gamma": 1.0, "distance_scale": 0.1, "success_reward": 1.0},
        distance_oscillation_actions(base_layout, 100),
    )
    assert not result["success"]
    assert result["proxy_return"] <= 1e-12
