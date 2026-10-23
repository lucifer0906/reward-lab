"""Tests for the dense reward function."""

import pytest

from reward_rl.envs.dynamics import Action, initial_state, transition
from reward_rl.envs.layout import parse_layout
from reward_rl.rewards.dense import DenseReward
from tests.conftest import BASE_SOLUTION


@pytest.fixture
def base_layout():
    from reward_rl.envs.key_door_env import DEFAULT_CONFIG, load_env_config
    cfg = load_env_config(DEFAULT_CONFIG)
    return parse_layout(cfg["layout"], name=cfg["name"])


@pytest.fixture
def dense():
    return DenseReward(
        key_pickup_reward=0.2,
        door_opened_reward=0.3,
        invalid_action_penalty=-0.05,
        success_reward=1.0,
        max_steps=100,
    )


def test_key_pickup_gives_reward(base_layout, dense):
    """Dense emits key_pickup when a key is first collected."""
    state = initial_state(base_layout)
    # Walk to key a: DOWN, DOWN, INTERACT
    for a in [Action.DOWN, Action.DOWN]:
        state, _ = transition(base_layout, state, a)
    prev = state
    state, events = transition(base_layout, state, Action.INTERACT)
    assert events.key_picked is not None
    components = dense(prev, Action.INTERACT, state, events, base_layout)
    assert "key_pickup" in components
    assert components["key_pickup"] == pytest.approx(0.2)


def test_door_opened_gives_reward(base_layout, dense):
    """Dense emits door_opened when a door is unlocked."""
    state = initial_state(base_layout)
    # Walk to key a and pick it up
    for a in [Action.DOWN, Action.DOWN, Action.INTERACT]:
        prev = state
        state, events = transition(base_layout, state, a)
    # Walk to door A and open it: UP, RIGHT, RIGHT, INTERACT
    for a in [Action.UP, Action.RIGHT, Action.RIGHT]:
        state, _ = transition(base_layout, state, a)
    prev = state
    state, events = transition(base_layout, state, Action.INTERACT)
    assert events.door_opened is not None
    components = dense(prev, Action.INTERACT, state, events, base_layout)
    assert "door_opened" in components
    assert components["door_opened"] == pytest.approx(0.3)


def test_invalid_action_penalized(base_layout, dense):
    """Dense emits invalid_action_penalty for wall bumps."""
    state = initial_state(base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.UP)  # bump into wall
    assert not events.valid
    components = dense(prev, Action.UP, state, events, base_layout)
    assert "invalid_action_penalty" in components
    assert components["invalid_action_penalty"] == pytest.approx(-0.05)


def test_valid_move_no_penalty(base_layout, dense):
    """Valid movement without key/door interaction emits no components."""
    state = initial_state(base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.DOWN)
    assert events.valid and events.moved
    components = dense(prev, Action.DOWN, state, events, base_layout)
    assert len(components) == 0, f"Expected empty, got {components}"


def test_terminal_success(base_layout, dense):
    """Dense emits terminal_success on goal with all doors open."""
    state = initial_state(base_layout)
    for a in BASE_SOLUTION:
        prev = state
        state, events = transition(base_layout, state, a)
    components = dense(prev, BASE_SOLUTION[-1], state, events, base_layout)
    assert "terminal_success" in components
    assert components["terminal_success"] == pytest.approx(1.0)


def test_all_components_separate(base_layout, dense):
    """Walk through the full solution and verify each component appears separately."""
    state = initial_state(base_layout)
    seen_components = set()
    for a in BASE_SOLUTION:
        prev = state
        state, events = transition(base_layout, state, a)
        components = dense(prev, a, state, events, base_layout)
        seen_components.update(components.keys())

    # Should have seen: key_pickup, door_opened, terminal_success
    assert "key_pickup" in seen_components
    assert "door_opened" in seen_components
    assert "terminal_success" in seen_components


def test_reset_is_callable():
    """Dense reward supports reset() even though it's stateless."""
    dense = DenseReward()
    dense.reset()  # should not raise
