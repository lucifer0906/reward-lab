"""Shared test helpers."""

from __future__ import annotations

import pytest

from reward_rl.envs.dynamics import Action, EpisodeRecord, initial_state, transition
from reward_rl.envs.key_door_env import DEFAULT_CONFIG, load_env_config
from reward_rl.envs.layout import Layout, parse_layout

U, D, L, R, I = Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.INTERACT

# Hand-traced optimal solution for configs/env/base_3keys.yaml (26 steps).
# Written by hand, NOT produced by the solver, so it independently checks
# both the solver and the env.
BASE_SOLUTION = [
    D, D, I,          # walk to key a (dead-end corner), pick it up
    U, R, R, I,       # stand next to door A, open it
    R, R, R, I,       # walk through A onto key b, pick it up
    R, I,             # stand next to door B, open it
    R, R, R, D, I,    # walk through B to key c, pick it up
    U, R, I,          # stand next to door C, open it
    R, R, R, R, U,    # walk through C to the goal
]


@pytest.fixture(scope="session")
def base_layout() -> Layout:
    cfg = load_env_config(DEFAULT_CONFIG)
    return parse_layout(cfg["layout"], name=cfg["name"])


def rollout(layout: Layout, actions, max_steps: int = 100, finalize: bool = True) -> EpisodeRecord:
    """Run actions through the pure dynamics and build an EpisodeRecord (no env, no reward)."""
    state = initial_state(layout)
    record = EpisodeRecord(layout_name=layout.name, max_steps=max_steps, initial_state=state)
    for a in actions:
        state, ev = transition(layout, state, a)
        record.actions.append(int(a))
        record.events.append(ev)
        if state.at_goal or state.step_count >= max_steps:
            break
    if finalize:
        record.final_state = state
        record.terminated = state.at_goal
        record.truncated = not state.at_goal and state.step_count >= max_steps
    return record
