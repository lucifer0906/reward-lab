"""ASCII grid layouts for the Key-Door-Chain environment.

A layout is written as a list of equal-length strings. Legend:

    #        wall (impassable)
    .        floor
    S        agent start position (exactly one)
    G        goal cell (exactly one)
    a..f     key i  (a = key 0, b = key 1, ...)
    A..F     door i (A = door 0, B = door 1, ...) -- opened only with key i

Keys and doors must come in matching pairs, numbered contiguously from 0
(i.e. a 3-pair layout uses exactly a/A, b/B, c/C).

Layouts are pure data: they describe *where* things are, never *how* the
agent is rewarded. That separation is deliberate (founder spec, Section 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

WALL = "#"
FLOOR = "."
START = "S"
GOAL = "G"
KEY_CHARS = "abcdef"
DOOR_CHARS = "ABCDEF"
MAX_PAIRS = len(KEY_CHARS)

Pos = tuple[int, int]  # (row, col)


class LayoutError(ValueError):
    """Raised when an ASCII layout is malformed."""


@dataclass(frozen=True)
class Layout:
    """Immutable, parsed description of a grid layout."""

    height: int
    width: int
    walls: frozenset[Pos]
    start: Pos
    goal: Pos
    key_positions: tuple[Pos, ...]   # index i -> position of key i
    door_positions: tuple[Pos, ...]  # index i -> position of door i
    name: str = "unnamed"

    @property
    def num_pairs(self) -> int:
        return len(self.key_positions)

    def in_bounds(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def door_index_at(self, pos: Pos) -> int | None:
        try:
            return self.door_positions.index(pos)
        except ValueError:
            return None

    def key_index_at(self, pos: Pos) -> int | None:
        try:
            return self.key_positions.index(pos)
        except ValueError:
            return None


def parse_layout(rows: Sequence[str], name: str = "unnamed") -> Layout:
    """Parse and validate an ASCII layout."""
    if not rows:
        raise LayoutError("Layout is empty.")
    rows = [r.rstrip("\n") for r in rows]
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise LayoutError(f"All rows must have equal length ({width}); got {[len(r) for r in rows]}.")

    walls: set[Pos] = set()
    starts: list[Pos] = []
    goals: list[Pos] = []
    keys: dict[int, Pos] = {}
    doors: dict[int, Pos] = {}

    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            pos = (r, c)
            if ch == WALL:
                walls.add(pos)
            elif ch == FLOOR:
                pass
            elif ch == START:
                starts.append(pos)
            elif ch == GOAL:
                goals.append(pos)
            elif ch in KEY_CHARS:
                idx = KEY_CHARS.index(ch)
                if idx in keys:
                    raise LayoutError(f"Duplicate key '{ch}'.")
                keys[idx] = pos
            elif ch in DOOR_CHARS:
                idx = DOOR_CHARS.index(ch)
                if idx in doors:
                    raise LayoutError(f"Duplicate door '{ch}'.")
                doors[idx] = pos
            else:
                raise LayoutError(f"Unknown character {ch!r} at {pos}.")

    if len(starts) != 1:
        raise LayoutError(f"Layout needs exactly one 'S', found {len(starts)}.")
    if len(goals) != 1:
        raise LayoutError(f"Layout needs exactly one 'G', found {len(goals)}.")
    if set(keys) != set(doors):
        raise LayoutError(f"Keys {sorted(keys)} and doors {sorted(doors)} must pair up exactly.")
    n = len(keys)
    if n == 0:
        raise LayoutError("Layout needs at least one key/door pair.")
    if set(keys) != set(range(n)):
        raise LayoutError(f"Key/door indices must be contiguous from 0; got {sorted(keys)}.")

    return Layout(
        height=len(rows),
        width=width,
        walls=frozenset(walls),
        start=starts[0],
        goal=goals[0],
        key_positions=tuple(keys[i] for i in range(n)),
        door_positions=tuple(doors[i] for i in range(n)),
        name=name,
    )
