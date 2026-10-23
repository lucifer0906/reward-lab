"""Potential-based reward shaping (Section 9, Section 10: R_potential).

Implements the classic potential-based shaping theorem (Ng et al., 1999):

    F(s, s') = γ · Φ(s') − Φ(s)

where Φ(s) is a *potential function* estimating progress toward the goal.
This shaping is added ON TOP of the sparse terminal reward (success/failure).

Key theoretical property: potential-based shaping preserves the optimal
policy under any MDP — it can speed up learning without introducing
reward hacking (unlike arbitrary dense shaping).

Potential function design:
    Φ(s) = (num_keys_collected + num_doors_open) / (2 * num_pairs)

    - Monotonically non-decreasing as the agent makes genuine progress.
    - Bounded in [0, 1] — reaches 1.0 only when all keys are collected
      AND all doors are opened (the two categories of subgoal in this env).
    - Does NOT leak the final answer: it measures subgoal completion count,
      not spatial proximity to specific keys/doors. The agent still has to
      figure out WHERE to go and in what ORDER.
    - Does NOT include distance-to-next-target: adding spatial information
      would risk leaking the solution path and would no longer be a pure
      "progress count" potential.

Why this is safe (Section 9's warning about not leaking the answer):
    The potential only counts *completed* subgoals, not partial progress
    toward uncompleted ones. An agent standing next to a key but not
    picking it up gets the same potential as one far from the key.
    The agent must still discover the correct key→door ordering itself.
"""

from __future__ import annotations

from typing import Mapping

from reward_rl.envs.dynamics import GridState, StepEvents
from reward_rl.envs.layout import Layout


class PotentialReward:
    """Sparse terminal reward + potential-based shaping F(s,s') = γΦ(s') − Φ(s).

    Stateful: tracks the previous potential for the shaping calculation.
    ``reset()`` must be called at episode start (the env handles this).
    """

    def __init__(
        self,
        gamma: float = 0.99,
        success_reward: float = 1.0,
        timeout_penalty: float = 0.0,
        step_cost: float = 0.0,
        max_steps: int = 100,
    ) -> None:
        self.gamma = gamma
        self.success_reward = success_reward
        self.timeout_penalty = timeout_penalty
        self.step_cost = step_cost
        self.max_steps = max_steps
        self._prev_potential: float = 0.0

    def reset(self) -> None:
        """Reset potential tracking at episode start."""
        self._prev_potential = 0.0

    @staticmethod
    def _potential(state: GridState, num_pairs: int) -> float:
        """Φ(s) = subgoal completion fraction ∈ [0, 1].

        Subgoals = keys collected + doors opened.
        Total possible = 2 * num_pairs (collect each key + open each door).
        """
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

        # --- Potential-based shaping: F(s,s') = γ·Φ(s') − Φ(s) ---
        phi_next = self._potential(nxt, layout.num_pairs)
        shaping = self.gamma * phi_next - self._prev_potential
        self._prev_potential = phi_next

        if abs(shaping) > 1e-9:
            components["potential_shaping"] = shaping

        # --- Terminal signals (same as sparse) ---

        # Success
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
