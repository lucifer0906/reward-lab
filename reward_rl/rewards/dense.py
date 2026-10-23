"""Dense reward: hand-designed intermediate signals (Section 8, Section 10: R_dense).

Provides interpretable intermediate rewards for meaningful progress:
    - key_pickup:           reward when a key is first collected
    - door_opened:          reward when a door is unlocked
    - invalid_action_penalty: penalty for actions that have no effect
    - terminal_success:     bonus for reaching the goal with all doors open

Every component is logged separately (Section 11) so we can later
diagnose whether the policy is learning intended behavior or exploiting
one specific component (reward-hacking analysis, Stage 5).

Design rationale (Section 8):
    - Each dense component has a clear causal relationship to final success:
      keys must be collected, doors must be opened, invalid actions waste budget.
    - We deliberately do NOT reward distance-to-target or movement:
      those can create reward hacking (oscillating near a key without
      picking it up, farming distance reward, etc).
    - All weights are configurable via YAML (Section 23).
"""

from __future__ import annotations

from typing import Mapping

from reward_rl.envs.dynamics import GridState, StepEvents
from reward_rl.envs.layout import Layout


class DenseReward:
    """Callable matching the RewardFn signature, with per-component logging.

    Supports ``reset()`` for interface uniformity (stateless, like sparse).
    """

    def __init__(
        self,
        key_pickup_reward: float = 0.2,
        door_opened_reward: float = 0.3,
        invalid_action_penalty: float = -0.05,
        success_reward: float = 1.0,
        timeout_penalty: float = 0.0,
        step_cost: float = 0.0,
        max_steps: int = 100,
    ) -> None:
        self.key_pickup_reward = key_pickup_reward
        self.door_opened_reward = door_opened_reward
        self.invalid_action_penalty = invalid_action_penalty
        self.success_reward = success_reward
        self.timeout_penalty = timeout_penalty
        self.step_cost = step_cost
        self.max_steps = max_steps

    def reset(self) -> None:
        """Called by the env at episode start. Dense is stateless, so no-op."""
        pass

    def __call__(
        self,
        prev: GridState,
        action: int,
        nxt: GridState,
        events: StepEvents,
        layout: Layout,
    ) -> Mapping[str, float]:
        components: dict[str, float] = {}

        # --- Intermediate signals ---

        # Subgoal: key pickup
        if events.key_picked is not None:
            components["key_pickup"] = self.key_pickup_reward

        # Subgoal: door opened
        if events.door_opened is not None:
            components["door_opened"] = self.door_opened_reward

        # Invalid action penalty
        if not events.valid:
            components["invalid_action_penalty"] = self.invalid_action_penalty

        # --- Terminal signals ---

        # Success: reached goal with all doors open
        if events.reached_goal and all(nxt.doors_open):
            components["terminal_success"] = self.success_reward

        # Timeout
        is_timeout = not nxt.at_goal and nxt.step_count >= self.max_steps
        if is_timeout and self.timeout_penalty != 0.0:
            components["terminal_failure"] = self.timeout_penalty

        # Per-step cost
        if self.step_cost != 0.0:
            components["step_cost"] = self.step_cost

        return components
