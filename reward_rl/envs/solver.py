"""Breadth-first optimal solver for Key-Door-Chain layouts.

Uses the exact same ``transition`` function as the environment, so its
answer is the true shortest successful action sequence. Used for:

    * validating that every layout (base + unseen generalization layouts)
      is solvable within the step budget,
    * computing ``optimal_length`` for the planning-efficiency metric
      (Section 18): planning_efficiency = optimal_length / episode_length,
    * scripted end-to-end tests of the env and the success evaluator.

The solver respects the required door order (door 0, then 1, ...), i.e. it
only returns plans the ground-truth evaluator would accept.
"""

from __future__ import annotations

from collections import deque

from reward_rl.envs.dynamics import Action, GridState, initial_state, transition
from reward_rl.envs.layout import Layout


def _order_ok(state: GridState) -> bool:
    """Doors must be opened as a prefix: open doors are exactly 0..k-1."""
    k = state.num_doors_open
    return all(state.doors_open[:k]) and not any(state.doors_open[k:])


def solve(layout: Layout, start: GridState | None = None) -> list[int] | None:
    """Return a shortest valid action list reaching the goal, or None if unsolvable."""
    root = start if start is not None else initial_state(layout)
    if root.at_goal:
        return []
    parent: dict[tuple, tuple[tuple, int] | None] = {root.search_key(): None}
    frontier: deque[GridState] = deque([root])

    while frontier:
        state = frontier.popleft()
        for action in Action:
            nxt, ev = transition(layout, state, action)
            if not ev.valid or not _order_ok(nxt):
                continue
            key = nxt.search_key()
            if key in parent:
                continue
            parent[key] = (state.search_key(), int(action))
            if nxt.at_goal:
                # Only accept the goal once every door is open.
                if all(nxt.doors_open):
                    return _reconstruct(parent, key)
                continue
            frontier.append(nxt)
    return None


def _reconstruct(parent: dict, key: tuple) -> list[int]:
    actions: list[int] = []
    while parent[key] is not None:
        key, action = parent[key]
        actions.append(action)
    return actions[::-1]


def optimal_length(layout: Layout) -> int | None:
    plan = solve(layout)
    return None if plan is None else len(plan)
