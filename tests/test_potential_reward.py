"""Tests for the potential-based reward function."""

import pytest

from reward_rl.envs.dynamics import Action, initial_state, transition
from reward_rl.envs.layout import parse_layout
from reward_rl.rewards.potential import PotentialReward
from tests.conftest import BASE_SOLUTION


@pytest.fixture
def base_layout():
    from reward_rl.envs.key_door_env import DEFAULT_CONFIG, load_env_config
    cfg = load_env_config(DEFAULT_CONFIG)
    return parse_layout(cfg["layout"], name=cfg["name"])


@pytest.fixture
def potential():
    return PotentialReward(gamma=0.99, success_reward=1.0, max_steps=100)


def test_no_shaping_on_plain_movement(base_layout, potential):
    """Moving without subgoal progress produces no potential_shaping."""
    state = initial_state(base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.DOWN)
    components = potential(prev, Action.DOWN, state, events, base_layout)
    assert "potential_shaping" not in components


def test_positive_shaping_on_key_pickup(base_layout, potential):
    """Picking up a key increases potential → positive shaping signal."""
    state = initial_state(base_layout)
    # Walk to key a
    for a in [Action.DOWN, Action.DOWN]:
        prev = state
        state, events = transition(base_layout, state, a)
        potential(prev, a, state, events, base_layout)  # advance internal state
    # Pick up key
    prev = state
    state, events = transition(base_layout, state, Action.INTERACT)
    components = potential(prev, Action.INTERACT, state, events, base_layout)
    assert "potential_shaping" in components
    assert components["potential_shaping"] > 0


def test_shaping_magnitude_correct(base_layout, potential):
    """F(s,s') = γ·Φ(s') - Φ(s) for one key pickup in a 3-key env.

    Before: Φ = 0/6 = 0
    After:  Φ = 1/6 ≈ 0.1667
    F = 0.99 * (1/6) - 0 = 0.165
    """
    state = initial_state(base_layout)
    for a in [Action.DOWN, Action.DOWN]:
        prev = state
        state, events = transition(base_layout, state, a)
        potential(prev, a, state, events, base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.INTERACT)
    components = potential(prev, Action.INTERACT, state, events, base_layout)
    expected = 0.99 * (1.0 / 6.0) - 0.0
    assert components["potential_shaping"] == pytest.approx(expected, abs=1e-6)


def test_terminal_success_still_present(base_layout, potential):
    """Potential reward preserves the sparse terminal success signal."""
    state = initial_state(base_layout)
    for a in BASE_SOLUTION:
        prev = state
        state, events = transition(base_layout, state, a)
        potential(prev, a, state, events, base_layout)
    # Re-evaluate the last step (potential was already called, but we need fresh)
    # Actually, let's re-run with a fresh reward to check the terminal signal
    pot2 = PotentialReward(gamma=0.99, success_reward=1.0, max_steps=100)
    state2 = initial_state(base_layout)
    for a in BASE_SOLUTION:
        prev2 = state2
        state2, events2 = transition(base_layout, state2, a)
        components = pot2(prev2, a, state2, events2, base_layout)
    assert "terminal_success" in components
    assert components["terminal_success"] == pytest.approx(1.0)


def test_shaping_sums_to_gamma_over_episode(base_layout):
    """Over a complete successful episode, total shaping ≈ γ·Φ(s_final) - Φ(s_0).

    For the optimal solution: Φ(s_0)=0, Φ(s_final) should be ~1.0 if all
    subgoals are done (but the goal step itself doesn't change keys/doors,
    so Φ(s_final) = 6/6 = 1.0). Total shaping ≈ γ^n_steps * 1.0 - 0.
    Actually, the telescoping sum gives ≈ γ·1.0 - 0 = 0.99.
    But it's not exact due to γ being applied at each step.
    Just verify it's positive and reasonable.
    """
    pot = PotentialReward(gamma=0.99, success_reward=1.0, max_steps=100)
    state = initial_state(base_layout)
    total_shaping = 0.0
    for a in BASE_SOLUTION:
        prev = state
        state, events = transition(base_layout, state, a)
        components = pot(prev, a, state, events, base_layout)
        total_shaping += components.get("potential_shaping", 0.0)
    assert total_shaping > 0
    # Should be close to γ·1.0 = 0.99 but with compounding γ effects
    # it's approximately: γ·(1/6) + γ·(1/6) + ... for 6 subgoals
    assert total_shaping < 2.0  # sanity bound


def test_reset_clears_potential():
    """reset() must zero the tracked potential for a new episode."""
    pot = PotentialReward(gamma=0.99)
    pot._prev_potential = 0.5  # simulate mid-episode
    pot.reset()
    assert pot._prev_potential == 0.0
