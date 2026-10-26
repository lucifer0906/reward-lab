"""Deliberately farmable non-potential distance reward for Stage 5b."""

from __future__ import annotations

from typing import Mapping

from reward_rl.envs.dynamics import GridState, StepEvents
from reward_rl.envs.layout import Layout
from reward_rl.rewards.distance_common import distance_to_next_target


class NaiveDistanceReward:
    """Pay a bonus for every one-cell decrease in next-target distance.

    This intentionally does not penalize moving away. A forward/backward pair
    can therefore be repeated to collect the forward bonus indefinitely.
    """

    def __init__(
        self,
        distance_decrease_reward: float = 0.1,
        success_reward: float = 1.0,
        timeout_penalty: float = 0.0,
        max_steps: int = 100,
    ) -> None:
        self.distance_decrease_reward = distance_decrease_reward
        self.success_reward = success_reward
        self.timeout_penalty = timeout_penalty
        self.max_steps = max_steps

    def reset(self) -> None:
        """Keep the callable interface consistent with other rewards."""

    def __call__(self, prev: GridState, action: int, nxt: GridState,
                 events: StepEvents, layout: Layout) -> Mapping[str, float]:
        components: dict[str, float] = {}
        decrease = distance_to_next_target(prev, layout) - distance_to_next_target(nxt, layout)
        if decrease > 0:
            components["distance_decrease"] = self.distance_decrease_reward * decrease
        if events.reached_goal and all(nxt.doors_open):
            components["terminal_success"] = self.success_reward
        elif not nxt.at_goal and nxt.step_count >= self.max_steps and self.timeout_penalty != 0.0:
            components["terminal_failure"] = self.timeout_penalty
        return components
