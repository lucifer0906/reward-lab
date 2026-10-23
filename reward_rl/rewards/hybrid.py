"""Hybrid reward: dense progress + validity + efficiency + terminal (Section 10: R_hybrid).

Combines the best of dense and potential approaches:
    - Dense progress signals (key pickup, door opening)
    - Validity penalties (invalid actions waste the step budget)
    - Efficiency incentive (small per-step cost encourages shorter solutions)
    - Potential-based shaping (subgoal completion fraction)
    - Terminal success/failure

This is intentionally the "kitchen sink" variant — it has the most
components and the most tuning knobs. This makes it the most powerful
learning signal but also the most vulnerable to reward hacking (Stage 5
will stress-test this exact prediction).

Every component is logged separately (Section 11) for decomposition
analysis and reward-hacking diagnosis.
"""

from __future__ import annotations

from typing import Mapping

from reward_rl.envs.dynamics import GridState, StepEvents
from reward_rl.envs.layout import Layout


class HybridReward:
    """All-in-one reward combining dense, potential-based, and efficiency signals.

    Stateful: tracks the previous potential for the shaping component.
    ``reset()`` must be called at episode start.
    """

    def __init__(
        self,
        # Dense progress
        key_pickup_reward: float = 0.2,
        door_opened_reward: float = 0.3,
        # Validity
        invalid_action_penalty: float = -0.05,
        # Efficiency
        step_cost: float = -0.005,
        # Potential-based shaping
        gamma: float = 0.99,
        use_potential: bool = True,
        # Terminal
        success_reward: float = 1.0,
        timeout_penalty: float = -0.1,
        max_steps: int = 100,
    ) -> None:
        self.key_pickup_reward = key_pickup_reward
        self.door_opened_reward = door_opened_reward
        self.invalid_action_penalty = invalid_action_penalty
        self.step_cost = step_cost
        self.gamma = gamma
        self.use_potential = use_potential
        self.success_reward = success_reward
        self.timeout_penalty = timeout_penalty
        self.max_steps = max_steps
        self._prev_potential: float = 0.0

    def reset(self) -> None:
        """Reset potential tracking at episode start."""
        self._prev_potential = 0.0

    @staticmethod
    def _potential(state: GridState, num_pairs: int) -> float:
        """Same Φ(s) as R_potential: subgoal completion fraction ∈ [0, 1]."""
        if num_pairs == 0:
            return 0.0
        completed = state.num_keys_collected + state.num_doors_open
        return completed / (2 * num_pairs)

    def __call__(
        self,
        prev: GridState,
        action: int,
        nxt: GridState,
        events: StepEvents,
        layout: Layout,
    ) -> Mapping[str, float]:
        components: dict[str, float] = {}

        # --- Dense progress signals ---

        if events.key_picked is not None:
            components["key_pickup"] = self.key_pickup_reward

        if events.door_opened is not None:
            components["door_opened"] = self.door_opened_reward

        # --- Validity penalty ---

        if not events.valid:
            components["invalid_action_penalty"] = self.invalid_action_penalty

        # --- Potential-based shaping ---

        if self.use_potential:
            phi_next = self._potential(nxt, layout.num_pairs)
            shaping = self.gamma * phi_next - self._prev_potential
            self._prev_potential = phi_next
            if abs(shaping) > 1e-9:
                components["potential_shaping"] = shaping

        # --- Efficiency (per-step cost) ---

        if self.step_cost != 0.0:
            components["step_cost"] = self.step_cost

        # --- Terminal signals ---

        if events.reached_goal and all(nxt.doors_open):
            components["terminal_success"] = self.success_reward

        is_timeout = not nxt.at_goal and nxt.step_count >= self.max_steps
        if is_timeout and self.timeout_penalty != 0.0:
            components["terminal_failure"] = self.timeout_penalty

        return components
