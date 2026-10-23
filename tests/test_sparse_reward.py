"""Tests for the sparse reward function."""

import pytest

from reward_rl.envs.dynamics import Action, initial_state, transition
from reward_rl.envs.layout import parse_layout
from reward_rl.rewards.sparse import SparseReward
from tests.conftest import BASE_SOLUTION


@pytest.fixture
def base_layout():
    from reward_rl.envs.key_door_env import DEFAULT_CONFIG, load_env_config
    cfg = load_env_config(DEFAULT_CONFIG)
    return parse_layout(cfg["layout"], name=cfg["name"])


@pytest.fixture
def sparse():
    return SparseReward(success_reward=1.0, timeout_penalty=-0.1, step_cost=0.0, max_steps=100)


def test_zero_reward_on_intermediate_steps(base_layout, sparse):
    """Sparse reward must return NOTHING on intermediate steps."""
    state = initial_state(base_layout)
    for a in BASE_SOLUTION[:-1]:  # all but the last step
        prev = state
        state, events = transition(base_layout, state, a)
        components = sparse(prev, a, state, events, base_layout)
        assert len(components) == 0, f"Sparse emitted {components} on intermediate step"


def test_success_reward_on_goal(base_layout, sparse):
    """Sparse reward emits terminal_success when the agent reaches the goal."""
    state = initial_state(base_layout)
    for a in BASE_SOLUTION:
        prev = state
        state, events = transition(base_layout, state, a)
    # Last step should trigger success
    components = sparse(prev, BASE_SOLUTION[-1], state, events, base_layout)
    assert "terminal_success" in components
    assert components["terminal_success"] == 1.0


def test_timeout_penalty(base_layout):
    """Sparse reward emits terminal_failure on timeout."""
    sparse = SparseReward(timeout_penalty=-0.1, max_steps=5)
    state = initial_state(base_layout)
    # Bump into walls 5 times to exhaust the budget
    for i in range(5):
        prev = state
        state, events = transition(base_layout, state, Action.UP)
    components = sparse(prev, Action.UP, state, events, base_layout)
    assert "terminal_failure" in components
    assert components["terminal_failure"] == pytest.approx(-0.1)


def test_no_timeout_penalty_when_zero(base_layout):
    """With timeout_penalty=0 (default), no terminal_failure component is emitted."""
    sparse = SparseReward(timeout_penalty=0.0, max_steps=5)
    state = initial_state(base_layout)
    for i in range(5):
        prev = state
        state, events = transition(base_layout, state, Action.UP)
    components = sparse(prev, Action.UP, state, events, base_layout)
    assert "terminal_failure" not in components


def test_step_cost_optional(base_layout):
    """When step_cost is set, it appears in every step's components."""
    sparse = SparseReward(step_cost=-0.01, max_steps=100)
    state = initial_state(base_layout)
    prev = state
    state, events = transition(base_layout, state, Action.DOWN)
    components = sparse(prev, Action.DOWN, state, events, base_layout)
    assert "step_cost" in components
    assert components["step_cost"] == pytest.approx(-0.01)


def test_reward_components_are_decomposed(base_layout):
    """Each component is named separately, never merged into a single scalar."""
    sparse = SparseReward(success_reward=1.0, step_cost=-0.01, max_steps=100)
    state = initial_state(base_layout)
    for a in BASE_SOLUTION:
        prev = state
        state, events = transition(base_layout, state, a)
    components = sparse(prev, BASE_SOLUTION[-1], state, events, base_layout)
    # Success step should have BOTH terminal_success AND step_cost
    assert "terminal_success" in components
    assert "step_cost" in components
    assert len(components) == 2


def test_reset_is_callable():
    """Sparse reward supports reset() even though it's stateless."""
    sparse = SparseReward()
    sparse.reset()  # should not raise
