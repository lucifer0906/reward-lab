"""Tests for the reward factory and cross-reward integration."""

import pytest

from reward_rl.rewards import AVAILABLE_TYPES, make_reward
from reward_rl.rewards.dense import DenseReward
from reward_rl.rewards.hybrid import HybridReward
from reward_rl.rewards.potential import PotentialReward
from reward_rl.rewards.sparse import SparseReward


def test_factory_creates_sparse():
    r = make_reward({"type": "sparse"})
    assert isinstance(r, SparseReward)


def test_factory_creates_dense():
    r = make_reward({"type": "dense"})
    assert isinstance(r, DenseReward)


def test_factory_creates_potential():
    r = make_reward({"type": "potential"})
    assert isinstance(r, PotentialReward)


def test_factory_creates_hybrid():
    r = make_reward({"type": "hybrid"})
    assert isinstance(r, HybridReward)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown reward type"):
        make_reward({"type": "nonexistent"})


def test_factory_passes_params():
    r = make_reward({"type": "sparse", "success_reward": 5.0})
    assert r.success_reward == 5.0


def test_factory_passes_max_steps():
    r = make_reward({"type": "dense"}, max_steps=50)
    assert r.max_steps == 50


def test_all_types_listed():
    """AVAILABLE_TYPES matches the actual implementations."""
    assert set(AVAILABLE_TYPES) == {"sparse", "dense", "potential", "hybrid"}


def test_all_rewards_have_reset():
    """Every reward type supports reset() for interface uniformity."""
    for rtype in AVAILABLE_TYPES:
        r = make_reward({"type": rtype})
        r.reset()  # should not raise
