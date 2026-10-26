"""Shared, reward-only next-target distance helpers for Stage 5b."""

from __future__ import annotations

from reward_rl.envs.dynamics import GridState
from reward_rl.envs.layout import Layout


def next_required_target(state: GridState, layout: Layout) -> tuple[int, int]:
    """Return the key, door, or goal required by the fixed task ordering."""
    for index in range(layout.num_pairs):
        if not state.keys_collected[index]:
            return layout.key_positions[index]
        if not state.doors_open[index]:
            return layout.door_positions[index]
    return layout.goal


def distance_to_next_target(state: GridState, layout: Layout) -> int:
    """Return Manhattan distance to the next required task target."""
    target_row, target_col = next_required_target(state, layout)
    row, col = state.agent_pos
    return abs(row - target_row) + abs(col - target_col)
