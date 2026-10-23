"""Key-Door-Chain: a long-horizon, dependent-subgoal Gymnasium environment.

The agent must:  pick up key a -> open door A -> pick up key b -> open door B
                 -> ... -> reach the goal G.
Later keys are physically behind earlier doors, so early actions (fetching
key a from a dead-end corner) have no visible benefit until much later --
exactly the credit-assignment problem the founder spec (Section 4) asks for.

Separation of concerns (founder spec Section 3):
    * Physics       -> reward_rl/envs/dynamics.py   (pure functions)
    * True success  -> reward_rl/evaluation/success_evaluator.py
    * Reward        -> injected via ``reward_fn`` (Stage 2+). In Stage 1 the
                       env returns reward 0.0 -- no reward logic exists yet.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from reward_rl.envs.dynamics import (
    NUM_ACTIONS,
    EpisodeRecord,
    GridState,
    StepEvents,
    initial_state,
    transition,
    valid_action_mask,
)
from reward_rl.envs.layout import DOOR_CHARS, KEY_CHARS, Layout, parse_layout
from reward_rl.envs.solver import optimal_length as compute_optimal_length
from reward_rl.evaluation.success_evaluator import EvaluationResult, evaluate_episode

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "env" / "base_3keys.yaml"

# Signature every reward function (Stage 2+) must follow. It returns a dict of
# named components (Section 11); the env sums them into the scalar reward.
RewardFn = Callable[[GridState, int, GridState, StepEvents, Layout], Mapping[str, float]]

NO_PREV_ACTION = NUM_ACTIONS  # sentinel value for "no previous action"


class KeyDoorChainEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        layout: Layout | Sequence[str],
        max_steps: int = 100,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
        layout_name: str = "unnamed",
    ) -> None:
        super().__init__()
        self.layout = layout if isinstance(layout, Layout) else parse_layout(layout, name=layout_name)
        self.max_steps = int(max_steps)
        self.reward_fn = reward_fn
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        self.optimal_length = compute_optimal_length(self.layout)
        if self.optimal_length is None:
            raise ValueError(f"Layout '{self.layout.name}' is unsolvable.")
        if self.optimal_length > self.max_steps:
            warnings.warn(
                f"Layout '{self.layout.name}' needs {self.optimal_length} steps but max_steps={self.max_steps}; "
                "success is impossible.",
                stacklevel=2,
            )

        H, W, n = self.layout.height, self.layout.width, self.layout.num_pairs
        # Structured (non-pixel) observation, per Section 6 / Part 2 decisions.
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.MultiDiscrete([H, W]),
                "keys_held": spaces.MultiBinary(n),        # resource status
                "keys_collected": spaces.MultiBinary(n),   # completed subgoals
                "doors_open": spaces.MultiBinary(n),       # completed subgoals / task progress
                "key_positions": spaces.MultiDiscrete([H, W] * n),
                "door_positions": spaces.MultiDiscrete([H, W] * n),
                "goal_pos": spaces.MultiDiscrete([H, W]),
                "action_mask": spaces.MultiBinary(NUM_ACTIONS),  # available actions
                "prev_action": spaces.Discrete(NUM_ACTIONS + 1),
                "time_remaining": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),  # termination info
            }
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._key_pos_arr = np.array([c for p in self.layout.key_positions for c in p], dtype=np.int64)
        self._door_pos_arr = np.array([c for p in self.layout.door_positions for c in p], dtype=np.int64)
        self._goal_arr = np.array(self.layout.goal, dtype=np.int64)

        self.state: GridState = initial_state(self.layout)
        self.prev_action: int = NO_PREV_ACTION
        self.record: EpisodeRecord | None = None
        self.last_evaluation: EvaluationResult | None = None

    # ------------------------------------------------------------------ API
    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG, **overrides: Any) -> "KeyDoorChainEnv":
        cfg = load_env_config(config_path)
        kwargs = {
            "layout": cfg["layout"],
            "max_steps": cfg.get("max_steps", 100),
            "layout_name": cfg.get("name", Path(config_path).stem),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # env is deterministic; seeds np_random for future stochastic variants
        self.state = initial_state(self.layout)
        self.prev_action = NO_PREV_ACTION
        self.record = EpisodeRecord(
            layout_name=self.layout.name,
            max_steps=self.max_steps,
            initial_state=self.state,
        )
        self.last_evaluation = None
        if self.reward_fn is not None and hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        if self.render_mode == "human":
            print(self.render())
        return self._obs(), self._base_info()

    def step(self, action):
        if self.record is None:
            raise RuntimeError("Call reset() before step().")
        if self.record.final_state is not None:
            raise RuntimeError("Episode is over; call reset().")
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action!r}")

        prev = self.state
        self.state, events = transition(self.layout, prev, int(action))
        self.prev_action = int(action)
        self.record.actions.append(int(action))
        self.record.events.append(events)

        terminated = self.state.at_goal
        truncated = (not terminated) and self.state.step_count >= self.max_steps

        # Reward hook. Stage 1 has no reward logic: reward is always 0.0.
        components: dict[str, float] = {}
        if self.reward_fn is not None:
            components = {k: float(v) for k, v in self.reward_fn(prev, int(action), self.state, events, self.layout).items()}
        reward = float(sum(components.values()))

        info = self._base_info()
        info["events"] = events.to_dict()
        info["reward_components"] = components

        if terminated or truncated:
            self.record.final_state = self.state
            self.record.terminated = terminated
            self.record.truncated = truncated
            self.last_evaluation = evaluate_episode(self.record, self.layout, self.optimal_length)
            # ``is_success`` is read by SB3's Monitor/EvalCallback. It comes from the
            # ground-truth evaluator, never from the reward.
            info["is_success"] = self.last_evaluation.success
            info["evaluation"] = self.last_evaluation.to_dict()

        if self.render_mode == "human":
            print(self.render())
        return self._obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """For sb3-contrib MaskablePPO compatibility (optional, not used by default)."""
        return np.array(valid_action_mask(self.layout, self.state), dtype=bool)

    # -------------------------------------------------------------- helpers
    def _obs(self) -> dict[str, np.ndarray]:
        s = self.state
        return {
            "agent_pos": np.array(s.agent_pos, dtype=np.int64),
            "keys_held": np.array(s.keys_held, dtype=np.int8),
            "keys_collected": np.array(s.keys_collected, dtype=np.int8),
            "doors_open": np.array(s.doors_open, dtype=np.int8),
            "key_positions": self._key_pos_arr.copy(),
            "door_positions": self._door_pos_arr.copy(),
            "goal_pos": self._goal_arr.copy(),
            "action_mask": np.array(valid_action_mask(self.layout, s), dtype=np.int8),
            "prev_action": int(self.prev_action),
            "time_remaining": np.array(
                [max(0.0, 1.0 - s.step_count / self.max_steps)], dtype=np.float32
            ),
        }

    def _base_info(self) -> dict[str, Any]:
        return {
            "step_count": self.state.step_count,
            "keys_collected": self.state.num_keys_collected,
            "doors_open": self.state.num_doors_open,
        }

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        L, s = self.layout, self.state
        grid = [["." for _ in range(L.width)] for _ in range(L.height)]
        for r, c in L.walls:
            grid[r][c] = "#"
        for i, (r, c) in enumerate(L.door_positions):
            grid[r][c] = "/" if s.doors_open[i] else DOOR_CHARS[i]
        for i, (r, c) in enumerate(L.key_positions):
            if not s.keys_collected[i]:
                grid[r][c] = KEY_CHARS[i]
        grid[L.goal[0]][L.goal[1]] = "G"
        grid[s.agent_pos[0]][s.agent_pos[1]] = "@"
        held = "".join(KEY_CHARS[i] for i, h in enumerate(s.keys_held) if h) or "-"
        opened = "".join(DOOR_CHARS[i] for i, o in enumerate(s.doors_open) if o) or "-"
        status = f"step {s.step_count}/{self.max_steps} | holding: {held} | doors open: {opened}"
        return "\n".join("".join(row) for row in grid) + "\n" + status


def load_env_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "layout" not in cfg:
        raise ValueError(f"Env config {path} has no 'layout' key.")
    return cfg


def make_from_config(config_path: str | Path = DEFAULT_CONFIG, **overrides: Any) -> KeyDoorChainEnv:
    """Gymnasium entry point."""
    return KeyDoorChainEnv.from_config(config_path, **overrides)
