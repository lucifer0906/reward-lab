"""Validates the ground-truth evaluator BEFORE any reward code exists (founder's critical rule)."""

import ast
import inspect
from dataclasses import replace

import pytest

from reward_rl.envs.dynamics import StepEvents
from reward_rl.envs.layout import parse_layout
from reward_rl.evaluation import success_evaluator
from reward_rl.evaluation.success_evaluator import FailureReason, evaluate_episode
from tests.conftest import BASE_SOLUTION, D, I, L, R, U, rollout


# ---------------------------------------------------------------- success
def test_correct_solution_is_success(base_layout):
    res = evaluate_episode(rollout(base_layout, BASE_SOLUTION), base_layout, optimal_length=26)
    assert res.success and res.failure_reason is None
    assert res.door_order == (0, 1, 2)
    assert res.subgoals_completed == res.subgoals_total == 7
    assert res.planning_efficiency == pytest.approx(1.0)
    assert res.invalid_actions == 0


def test_inefficient_but_correct_is_still_success(base_layout):
    """Wasted steps lower planning efficiency but do NOT change the true verdict."""
    wasteful = [U, L, I, R, L] + BASE_SOLUTION  # 5 junk actions (3 invalid) first
    res = evaluate_episode(rollout(base_layout, wasteful), base_layout, optimal_length=26)
    assert res.success
    assert res.episode_length == 31
    assert res.invalid_actions == 3
    assert res.planning_efficiency == pytest.approx(26 / 31)


# ---------------------------------------------------------------- failures
def test_partial_progress_then_timeout_is_failure(base_layout):
    """Picking up key a and opening door A, then idling: real progress, but NOT success."""
    actions = [D, D, I, U, R, R, I] + [I] * 200
    res = evaluate_episode(rollout(base_layout, actions, max_steps=100), base_layout)
    assert not res.success and res.failure_reason == FailureReason.TIMEOUT
    assert (res.keys_collected, res.doors_opened, res.reached_goal) == (1, 1, False)
    assert res.subgoals_completed == 2
    assert res.subgoal_completion_rate == pytest.approx(2 / 7)


def test_all_keys_and_doors_but_no_goal_is_failure(base_layout):
    """6/7 subgoals done is still a failure -- only the true objective counts."""
    almost = BASE_SOLUTION[:-1]
    res = evaluate_episode(rollout(base_layout, almost), base_layout)
    assert not res.success and res.failure_reason == FailureReason.GOAL_NOT_REACHED
    assert res.subgoals_completed == 6


def test_unfinished_record_is_incomplete(base_layout):
    res = evaluate_episode(rollout(base_layout, BASE_SOLUTION[:5], finalize=False), base_layout)
    assert not res.success and res.failure_reason == FailureReason.INCOMPLETE


def test_success_after_budget_is_timeout(base_layout):
    """Reaching the goal is not enough if it took longer than the step budget."""
    record = rollout(base_layout, BASE_SOLUTION, max_steps=1000)
    record.max_steps = 20
    res = evaluate_episode(record, base_layout)
    assert not res.success and res.failure_reason == FailureReason.TIMEOUT


def test_wrong_door_order_is_failure():
    # Corridor where door B can physically be opened before door A.
    layout = parse_layout(["#########", "#baSBAG.#", "#########"], name="wrong_order")
    actions = [L, I, L, I, R, R, I, R, I, R, R]  # pick a, pick b, open B, open A, reach G
    record = rollout(layout, actions)
    assert record.final_state.at_goal  # physically reached the goal...
    res = evaluate_episode(record, layout)
    assert not res.success and res.failure_reason == FailureReason.WRONG_DOOR_ORDER  # ...but wrongly
    assert res.door_order == (1, 0)


def test_goal_without_all_doors_is_failure():
    # Goal reachable while door B stays closed (a "shortcut" layout).
    layout = parse_layout(["#######", "#SaAGb#", "###B###"], name="shortcut")
    record = rollout(layout, [R, I, I, R, R])
    assert record.final_state.at_goal
    res = evaluate_episode(record, layout)
    assert not res.success and res.failure_reason == FailureReason.DOORS_NOT_ALL_OPEN


# ------------------------------------------------- tamper-resistance (rule 3)
def test_tampered_events_are_rejected(base_layout):
    """A record that *claims* the goal was reached without physically doing it fails."""
    record = rollout(base_layout, [D, D, I])
    fake = list(record.events)
    fake[-1] = StepEvents(action=4, valid=True, reached_goal=True)
    record.events = fake
    record.final_state = replace(record.final_state, at_goal=True, doors_open=(True, True, True))
    res = evaluate_episode(record, base_layout)
    assert not res.success and res.failure_reason == FailureReason.REPLAY_MISMATCH


def test_tampered_final_state_is_rejected(base_layout):
    record = rollout(base_layout, [D, D, I])
    record.final_state = replace(record.final_state, agent_pos=base_layout.goal, at_goal=True)
    res = evaluate_episode(record, base_layout)
    assert res.failure_reason == FailureReason.REPLAY_MISMATCH


# ------------------------------------------------- reward independence (rule 1/2)
def test_evaluator_is_reward_blind():
    tree = ast.parse(inspect.getsource(success_evaluator))
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(m and m.startswith("reward_rl.rewards") for m in imported), imported
    params = inspect.signature(evaluate_episode).parameters
    assert "reward" not in " ".join(params)
