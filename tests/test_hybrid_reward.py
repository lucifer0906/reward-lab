"""Tests for the hybrid reward function."""

import pytest

from reward_rl.envs.dynamics import Action, initial_state, transition
from reward_rl.envs.layout import parse_layout
from reward_rl.rewards.hybrid import HybridReward
from tests.conftest import BASE_SOLUTION


@pytest.fixture
def base_layout():
    from reward_rl.envs.key_door_env import DEFAULT_CONFIG, load_env_config
    cfg = load_env_config(DEFAULT_CONFIG)
    return parse_layout(cfg["layout"], name=cfg["name"])


@pytest.fixture
def hybrid():
    return HybridReward(
        key_pickup_reward=0.2,
        door_opened_reward=0.3,
        invalid_action_penalty=-0.05,
        step_cost=-0.005,
        gamma=0.99,
        use_potential=True,
        success_reward=1.0,
        timeout_penalty=-0.1,
        max_steps=100,
    )


def test_step_cost_every_step(base_layout, hybrid):
    """Hybrid emits step_cost on every step (efficiency incentive)."""
    state = initial_state(base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.DOWN)
    components = hybrid(prev, Action.DOWN, state, events, base_layout)
    assert "step_cost" in components
    assert components["step_cost"] == pytest.approx(-0.005)


def test_has_dense_and_potential_on_key_pickup(base_layout, hybrid):
    """Key pickup triggers BOTH dense key_pickup AND potential_shaping."""
    state = initial_state(base_layout)
    for a in [Action.DOWN, Action.DOWN]:
        prev = state
        state, events = transition(base_layout, state, a)
        hybrid(prev, a, state, events, base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.INTERACT)
    components = hybrid(prev, Action.INTERACT, state, events, base_layout)
    assert "key_pickup" in components
    assert "potential_shaping" in components
    assert "step_cost" in components
    # All three present simultaneously — this is the "kitchen sink" behavior
    assert len(components) >= 3


def test_invalid_action_penalty(base_layout, hybrid):
    """Hybrid penalizes invalid actions."""
    state = initial_state(base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.UP)  # wall bump
    components = hybrid(prev, Action.UP, state, events, base_layout)
    assert "invalid_action_penalty" in components
    assert components["invalid_action_penalty"] == pytest.approx(-0.05)


def test_timeout_penalty(base_layout):
    """Hybrid emits terminal_failure on timeout."""
    hybrid = HybridReward(timeout_penalty=-0.1, max_steps=3)
    state = initial_state(base_layout)
    for _ in range(3):
        prev = state
        state, events = transition(base_layout, state, Action.DOWN)
        hybrid(prev, Action.DOWN, state, events, base_layout)
    # After 3 steps of walking down, we should hit max_steps
    # Re-check: step_count increments each time, so after 3 transitions step_count=3
    prev = state
    state, events = transition(base_layout, state, Action.DOWN)
    # step_count is now 4, but max_steps=3 so check >= 3
    # Actually: step_count is set inside transition, let's check
    # After 3 transitions, step_count=3. Let me use a hybrid with max_steps=3
    # and check: nxt.step_count >= max_steps → 3 >= 3 → True
    # But we need to get to step 3. Let me redo cleanly:
    hybrid2 = HybridReward(timeout_penalty=-0.1, max_steps=2)
    state2 = initial_state(base_layout)
    for _ in range(2):
        prev2 = state2
        state2, events2 = transition(base_layout, state2, Action.DOWN)
        hybrid2(prev2, Action.DOWN, state2, events2, base_layout)
    # step_count = 2 = max_steps, not at goal → timeout
    components = hybrid2(prev2, Action.DOWN, state2, events2, base_layout)
    # Note: we already consumed the events above. Let me just check
    # the last call's components from the loop
    # Actually, let me trace through properly:
    hybrid3 = HybridReward(timeout_penalty=-0.1, max_steps=2)
    s = initial_state(base_layout)
    p = s
    s, ev = transition(base_layout, s, Action.DOWN)
    hybrid3(p, Action.DOWN, s, ev, base_layout)  # step_count=1, no timeout
    p = s
    s, ev = transition(base_layout, s, Action.DOWN)
    comps = hybrid3(p, Action.DOWN, s, ev, base_layout)  # step_count=2 = max_steps
    assert "terminal_failure" in comps
    assert comps["terminal_failure"] == pytest.approx(-0.1)


def test_terminal_success(base_layout, hybrid):
    """Hybrid emits terminal_success on goal completion."""
    state = initial_state(base_layout)
    for a in BASE_SOLUTION:
        prev = state
        state, events = transition(base_layout, state, a)
        hybrid(prev, a, state, events, base_layout)
    # Re-run with fresh reward to get the terminal step's components
    h2 = HybridReward(success_reward=1.0, max_steps=100)
    state2 = initial_state(base_layout)
    for a in BASE_SOLUTION:
        prev2 = state2
        state2, events2 = transition(base_layout, state2, a)
        components = h2(prev2, a, state2, events2, base_layout)
    assert "terminal_success" in components
    assert components["terminal_success"] == pytest.approx(1.0)


def test_potential_can_be_disabled(base_layout):
    """When use_potential=False, no potential_shaping component appears."""
    hybrid = HybridReward(use_potential=False, max_steps=100)
    state = initial_state(base_layout)
    for a in [Action.DOWN, Action.DOWN]:
        prev = state
        state, events = transition(base_layout, state, a)
        hybrid(prev, a, state, events, base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.INTERACT)
    components = hybrid(prev, Action.INTERACT, state, events, base_layout)
    assert "potential_shaping" not in components
    assert "key_pickup" in components  # dense still active


def test_reset_clears_potential():
    """reset() must zero the tracked potential for a new episode."""
    h = HybridReward()
    h._prev_potential = 0.5
    h.reset()
    assert h._prev_potential == 0.0


def test_all_component_types_appear_in_full_episode(base_layout):
    """Run the full optimal solution and verify we see ALL component types."""
    hybrid = HybridReward(
        key_pickup_reward=0.2,
        door_opened_reward=0.3,
        invalid_action_penalty=-0.05,
        step_cost=-0.005,
        gamma=0.99,
        use_potential=True,
        success_reward=1.0,
        max_steps=100,
    )
    state = initial_state(base_layout)
    seen = set()
    for a in BASE_SOLUTION:
        prev = state
        state, events = transition(base_layout, state, a)
        components = hybrid(prev, a, state, events, base_layout)
        seen.update(components.keys())

    # The optimal solution never does invalid actions, so we won't see that
    expected = {"key_pickup", "door_opened", "step_cost", "potential_shaping", "terminal_success"}
    assert expected.issubset(seen), f"Missing: {expected - seen}"
