from dataclasses import replace

from reward_rl.envs.dynamics import Action, InvalidReason, initial_state, transition, valid_action_mask
from reward_rl.envs.solver import solve
from tests.conftest import BASE_SOLUTION, D, I, L, R, U


def test_initial_state(base_layout):
    s = initial_state(base_layout)
    assert s.agent_pos == base_layout.start
    assert s.keys_held == s.keys_collected == s.doors_open == (False, False, False)
    assert s.step_count == 0 and not s.at_goal


def test_wall_bump_is_invalid_and_costs_a_step(base_layout):
    s0 = initial_state(base_layout)
    s1, ev = transition(base_layout, s0, U)  # (0,1) is a wall
    assert not ev.valid and ev.invalid_reason == InvalidReason.WALL
    assert s1.agent_pos == s0.agent_pos and s1.step_count == 1


def test_interact_on_empty_cell_is_invalid(base_layout):
    _, ev = transition(base_layout, initial_state(base_layout), I)
    assert not ev.valid and ev.invalid_reason == InvalidReason.NOTHING_TO_INTERACT


def test_locked_door_blocks_movement_and_needs_key(base_layout):
    s = replace(initial_state(base_layout), agent_pos=(2, 3))  # next to door A, no key
    _, ev = transition(base_layout, s, R)
    assert ev.invalid_reason == InvalidReason.LOCKED_DOOR
    _, ev = transition(base_layout, s, I)
    assert ev.invalid_reason == InvalidReason.NO_KEY_FOR_DOOR


def test_pick_key_then_open_door_consumes_key(base_layout):
    s = initial_state(base_layout)
    for a in [D, D]:
        s, _ = transition(base_layout, s, a)
    s, ev = transition(base_layout, s, I)
    assert ev.key_picked == 0 and s.keys_held[0] and s.keys_collected[0]

    # Key cannot be picked up twice (the classic reward-hacking loop is
    # physically impossible in the env itself).
    s_again, ev = transition(base_layout, s, I)
    assert not ev.valid and s_again.keys_collected == s.keys_collected

    for a in [U, R, R]:
        s, _ = transition(base_layout, s, a)
    s, ev = transition(base_layout, s, I)
    assert ev.door_opened == 0
    assert s.doors_open[0] and not s.keys_held[0] and s.keys_collected[0]

    s, ev = transition(base_layout, s, R)  # can now walk onto the open door cell
    assert ev.valid and s.agent_pos == (2, 4)


def test_action_mask_matches_transition(base_layout):
    s = initial_state(base_layout)
    for a in BASE_SOLUTION:
        mask = valid_action_mask(base_layout, s)
        for act in Action:
            _, ev = transition(base_layout, s, act)
            assert mask[act] == ev.valid
        s, _ = transition(base_layout, s, a)


def test_hand_solution_reaches_goal(base_layout):
    s = initial_state(base_layout)
    for a in BASE_SOLUTION:
        s, ev = transition(base_layout, s, a)
        assert ev.valid
    assert s.at_goal and all(s.doors_open) and s.step_count == 26


def test_solver_is_optimal_and_valid(base_layout):
    plan = solve(base_layout)
    assert plan is not None and len(plan) == len(BASE_SOLUTION) == 26
    s = initial_state(base_layout)
    for a in plan:
        s, _ = transition(base_layout, s, a)
    assert s.at_goal


def test_no_action_after_goal(base_layout):
    s = initial_state(base_layout)
    for a in BASE_SOLUTION:
        s, _ = transition(base_layout, s, a)
    s2, ev = transition(base_layout, s, L)
    assert not ev.valid and ev.invalid_reason == InvalidReason.EPISODE_OVER and s2 == s
