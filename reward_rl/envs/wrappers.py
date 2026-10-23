"""Gymnasium wrappers for SB3 compatibility.

SB3's MlpPolicy requires a flat Box observation space. Our env uses a
Dict space (Section 6). This wrapper flattens it into a single float32
vector while preserving all the information the spec requires.

Flattened observation vector layout (for base 3-key env, total = 28):
    [0:2]   agent_pos (row, col) — normalized to [0,1]
    [2:5]   keys_held (binary)
    [5:8]   keys_collected (binary)
    [8:11]  doors_open (binary)
    [11:17] key_positions (r0,c0,r1,c1,r2,c2) — normalized
    [17:23] door_positions — normalized
    [23:25] goal_pos — normalized
    [25:30] action_mask (binary)
    [30]    prev_action — one-hot would be cleaner but normalized scalar
            keeps dim low for CPU training
    [31]    time_remaining [0,1]

Normalization: positions divided by (H-1, W-1) so all values sit in [0,1].
This helps PPO's value network converge faster on CPU.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reward_rl.envs.dynamics import NUM_ACTIONS


class FlattenObsWrapper(gym.ObservationWrapper):
    """Flatten the Dict observation into a 1-D float32 Box for SB3."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Grab dims from the wrapped env
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Dict), f"Expected Dict, got {type(obs_space)}"

        self._H = obs_space["agent_pos"].nvec[0]
        self._W = obs_space["agent_pos"].nvec[1]
        n = obs_space["keys_held"].n  # number of key-door pairs

        # Compute flat dimension
        self._flat_dim = (
            2           # agent_pos
            + n         # keys_held
            + n         # keys_collected
            + n         # doors_open
            + 2 * n     # key_positions
            + 2 * n     # door_positions
            + 2         # goal_pos
            + NUM_ACTIONS  # action_mask
            + 1         # prev_action (normalized)
            + 1         # time_remaining
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._flat_dim,), dtype=np.float32
        )

    def observation(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        H, W = self._H, self._W
        # Avoid division by zero for 1-cell grids (defensive)
        h_norm = max(H - 1, 1)
        w_norm = max(W - 1, 1)

        parts = []

        # agent_pos — normalize
        ap = obs["agent_pos"].astype(np.float32)
        parts.append(np.array([ap[0] / h_norm, ap[1] / w_norm], dtype=np.float32))

        # binary flags
        parts.append(obs["keys_held"].astype(np.float32))
        parts.append(obs["keys_collected"].astype(np.float32))
        parts.append(obs["doors_open"].astype(np.float32))

        # key positions — normalize alternating (row, col, row, col, ...)
        kp = obs["key_positions"].astype(np.float32)
        kp_norm = kp.copy()
        kp_norm[0::2] /= h_norm
        kp_norm[1::2] /= w_norm
        parts.append(kp_norm)

        # door positions
        dp = obs["door_positions"].astype(np.float32)
        dp_norm = dp.copy()
        dp_norm[0::2] /= h_norm
        dp_norm[1::2] /= w_norm
        parts.append(dp_norm)

        # goal pos
        gp = obs["goal_pos"].astype(np.float32)
        parts.append(np.array([gp[0] / h_norm, gp[1] / w_norm], dtype=np.float32))

        # action mask
        parts.append(obs["action_mask"].astype(np.float32))

        # prev_action — normalized to [0, 1]
        pa = float(obs["prev_action"]) / NUM_ACTIONS
        parts.append(np.array([pa], dtype=np.float32))

        # time remaining — already [0, 1]
        parts.append(obs["time_remaining"].astype(np.float32))

        flat = np.concatenate(parts)
        assert flat.shape == (self._flat_dim,), f"Expected {self._flat_dim}, got {flat.shape}"
        return flat
