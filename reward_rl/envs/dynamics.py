"""Pure transition dynamics for the Key-Door-Chain MDP.

Everything here is a *pure function* of (layout, state, action). There is no
randomness, no reward, and no hidden mutable state. This lets three different
consumers share one source of truth for the physics:

    * the Gymnasium env (``key_door_env.py``)
    * the BFS optimal solver (``solver.py``)
    * the ground-truth success evaluator, which *replays* trajectories to
      verify them independently (``evaluation/success_evaluator.py``)

MDP summary (see docs/environment.md for the full formalisation):
    S : (agent_pos, keys_held, keys_collected, doors_open, step_count, at_goal)
    A : {UP, DOWN, LEFT, RIGHT, INTERACT}   -- fixed across all experiments
    P : deterministic, defined by ``transition`` below
    R : NOT defined here -- rewards live in reward_rl/rewards/ (Stage 2+)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum

from reward_rl.envs.layout import Layout, Pos


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4


NUM_ACTIONS = len(Action)
MOVE_DELTAS: dict[Action, Pos] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


class InvalidReason:
    """String constants describing why an action had no effect."""

    WALL = "wall"                          # moved into a wall / out of bounds
    LOCKED_DOOR = "locked_door"            # moved into a door that is still closed
    NO_KEY_FOR_DOOR = "no_key_for_door"    # interacted next to a locked door without its key
    NOTHING_TO_INTERACT = "nothing_to_interact"
    EPISODE_OVER = "episode_over"          # acted after reaching the goal


@dataclass(frozen=True)
class GridState:
    """Full, immutable environment state (hashable -> usable as a BFS node)."""

    agent_pos: Pos
    keys_held: tuple[bool, ...]       # key in inventory and not yet used
    keys_collected: tuple[bool, ...]  # key has ever been picked up
    doors_open: tuple[bool, ...]
    step_count: int = 0
    at_goal: bool = False

    @property
    def num_keys_collected(self) -> int:
        return sum(self.keys_collected)

    @property
    def num_doors_open(self) -> int:
        return sum(self.doors_open)

    def search_key(self) -> tuple:
        """State identity ignoring the step counter (for planning/search)."""
        return (self.agent_pos, self.keys_held, self.keys_collected, self.doors_open, self.at_goal)


@dataclass(frozen=True)
class StepEvents:
    """What *physically* happened during one transition.

    Reward functions (Stage 2+) and the success evaluator consume these
    events; they never need to re-derive physics themselves.
    """

    action: int
    valid: bool
    invalid_reason: str | None = None
    moved: bool = False
    key_picked: int | None = None
    door_opened: int | None = None
    reached_goal: bool = False

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "moved": self.moved,
            "key_picked": self.key_picked,
            "door_opened": self.door_opened,
            "reached_goal": self.reached_goal,
        }


def initial_state(layout: Layout) -> GridState:
    n = layout.num_pairs
    return GridState(
        agent_pos=layout.start,
        keys_held=(False,) * n,
        keys_collected=(False,) * n,
        doors_open=(False,) * n,
    )


def _set(t: tuple[bool, ...], i: int, value: bool) -> tuple[bool, ...]:
    return t[:i] + (value,) + t[i + 1:]


def _neighbors(pos: Pos) -> list[Pos]:
    r, c = pos
    return [(r + dr, c + dc) for dr, dc in MOVE_DELTAS.values()]


def _blocked(layout: Layout, state: GridState, pos: Pos) -> str | None:
    """Return an InvalidReason if ``pos`` cannot be entered, else None."""
    if not layout.in_bounds(pos) or pos in layout.walls:
        return InvalidReason.WALL
    door = layout.door_index_at(pos)
    if door is not None and not state.doors_open[door]:
        return InvalidReason.LOCKED_DOOR
    return None


def _interact_target(layout: Layout, state: GridState) -> tuple[str, int] | str:
    """Decide what INTERACT would do.

    Returns ("pick", key_idx) / ("open", door_idx) on success, or an
    InvalidReason string on failure. Priority: picking up a key under the
    agent beats opening an adjacent door.
    """
    key = layout.key_index_at(state.agent_pos)
    if key is not None and not state.keys_collected[key]:
        return ("pick", key)

    saw_locked_door = False
    for nb in _neighbors(state.agent_pos):
        door = layout.door_index_at(nb)
        if door is None or state.doors_open[door]:
            continue
        saw_locked_door = True
        if state.keys_held[door]:
            return ("open", door)
    return InvalidReason.NO_KEY_FOR_DOOR if saw_locked_door else InvalidReason.NOTHING_TO_INTERACT


def transition(layout: Layout, state: GridState, action: int) -> tuple[GridState, StepEvents]:
    """Apply one action. Invalid actions leave the state unchanged (except step_count)."""
    action = Action(int(action))

    if state.at_goal:
        return state, StepEvents(action=int(action), valid=False, invalid_reason=InvalidReason.EPISODE_OVER)

    stepped = replace(state, step_count=state.step_count + 1)

    if action in MOVE_DELTAS:
        dr, dc = MOVE_DELTAS[action]
        target = (state.agent_pos[0] + dr, state.agent_pos[1] + dc)
        reason = _blocked(layout, state, target)
        if reason is not None:
            return stepped, StepEvents(action=int(action), valid=False, invalid_reason=reason)
        reached = target == layout.goal
        nxt = replace(stepped, agent_pos=target, at_goal=reached)
        return nxt, StepEvents(action=int(action), valid=True, moved=True, reached_goal=reached)

    # INTERACT
    target = _interact_target(layout, state)
    if isinstance(target, str):
        return stepped, StepEvents(action=int(action), valid=False, invalid_reason=target)
    kind, idx = target
    if kind == "pick":
        nxt = replace(
            stepped,
            keys_held=_set(state.keys_held, idx, True),
            keys_collected=_set(state.keys_collected, idx, True),
        )
        return nxt, StepEvents(action=int(action), valid=True, key_picked=idx)
    # kind == "open": the key is consumed by the door
    nxt = replace(
        stepped,
        keys_held=_set(state.keys_held, idx, False),
        doors_open=_set(state.doors_open, idx, True),
    )
    return nxt, StepEvents(action=int(action), valid=True, door_opened=idx)


def valid_action_mask(layout: Layout, state: GridState) -> tuple[bool, ...]:
    """Which of the fixed actions would actually change the state right now."""
    if state.at_goal:
        return (False,) * NUM_ACTIONS
    mask = []
    for action in Action:
        if action in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[action]
            target = (state.agent_pos[0] + dr, state.agent_pos[1] + dc)
            mask.append(_blocked(layout, state, target) is None)
        else:
            mask.append(not isinstance(_interact_target(layout, state), str))
    return tuple(mask)


@dataclass
class EpisodeRecord:
    """Everything needed to judge an episode after the fact.

    Deliberately contains NO reward information: the ground-truth evaluator
    must never be able to see (and therefore be fooled by) the reward signal.
    """

    layout_name: str
    max_steps: int
    initial_state: GridState
    actions: list[int] = field(default_factory=list)
    events: list[StepEvents] = field(default_factory=list)
    final_state: GridState | None = None
    terminated: bool = False
    truncated: bool = False
