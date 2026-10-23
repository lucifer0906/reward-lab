"""Custom Gymnasium environments (founder spec Section 24: envs/)."""

import gymnasium as gym

from reward_rl.envs.dynamics import Action, EpisodeRecord, GridState, StepEvents, transition
from reward_rl.envs.key_door_env import DEFAULT_CONFIG, KeyDoorChainEnv, make_from_config
from reward_rl.envs.layout import Layout, parse_layout
from reward_rl.envs.solver import optimal_length, solve

ENV_ID = "reward_rl/KeyDoorChain-v0"

if ENV_ID not in gym.registry:
    gym.register(
        id=ENV_ID,
        entry_point="reward_rl.envs.key_door_env:make_from_config",
        kwargs={"config_path": str(DEFAULT_CONFIG)},
    )

__all__ = [
    "Action",
    "DEFAULT_CONFIG",
    "ENV_ID",
    "EpisodeRecord",
    "GridState",
    "KeyDoorChainEnv",
    "Layout",
    "StepEvents",
    "make_from_config",
    "optimal_length",
    "parse_layout",
    "solve",
    "transition",
]
