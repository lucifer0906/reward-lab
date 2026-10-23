"""Training module: PPO training loop, callbacks, and CLI.

CLI usage: python -m reward_rl.train --config configs/sparse.yaml
"""

from reward_rl.training.callbacks import (
    CheckpointAndMetricsCallback,
    RewardComponentLogger,
    SuccessRateLogger,
)
from reward_rl.training.train import main, train_single_seed

__all__ = [
    "main",
    "train_single_seed",
    "CheckpointAndMetricsCallback",
    "RewardComponentLogger",
    "SuccessRateLogger",
]
