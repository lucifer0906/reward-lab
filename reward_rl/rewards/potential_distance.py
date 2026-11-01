"""Distance-potential control reward for the Stage 5b adversarial comparison."""

from __future__ import annotations

from typing import Mapping

from reward_rl.envs.dynamics import GridState, StepEvents
from reward_rl.envs.layout import Layout
from reward_rl.rewards.distance_common import distance_to_next_target


class PotentialDistanceReward:
    """Use F(s, s') = gamma * Phi(s') - Phi(s) over next-target distance."""

    def __init__(
        self,
        gamma: float = 1.0,
        distance_scale: float = 0.1,
        distance_baseline: float = 0.0,
        success_reward: float = 1.0,
        timeout_penalty: float = 0.0,
        max_steps: int = 100,
    ) -> None:
        self.gamma = gamma
        self.distance_scale = distance_scale
        self.distance_baseline = distance_baseline
        self.success_reward = success_reward
        self.timeout_penalty = timeout_penalty
        self.max_steps = max_steps

    def reset(self) -> None:
        """This reward derives both potentials directly from the transition."""

    def _potential(self, state: GridState, layout: Layout) -> float:
        """Return an affine distance potential with the configured scale.

        A baseline of zero is the legacy Phi=-scale*distance form. With a
        discounted shaping gamma, a positive baseline avoids the artificial
        positive loop drift produced by a negative fixed-state potential.
        """
        return self.distance_scale * (
            self.distance_baseline - distance_to_next_target(state, layout)
        )

    def __call__(self, prev: GridState, action: int, nxt: GridState,
                 events: StepEvents, layout: Layout) -> Mapping[str, float]:
        components: dict[str, float] = {}
        shaping = self.gamma * self._potential(nxt, layout) - self._potential(prev, layout)
        if abs(shaping) > 1e-12:
            components["potential_distance_shaping"] = shaping
        if events.reached_goal and all(nxt.doors_open):
            components["terminal_success"] = self.success_reward
        elif not nxt.at_goal and nxt.step_count >= self.max_steps and self.timeout_penalty != 0.0:
            components["terminal_failure"] = self.timeout_penalty
        return components
