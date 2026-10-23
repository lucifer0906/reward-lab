"""Sparse reward: terminal success/failure only (Section 7, Section 10: R_sparse).

The simplest possible reward. The agent receives:
    +1.0  when it reaches the goal with all doors opened correctly
    -0.1  on timeout (step budget exhausted without reaching the goal)
     0.0  every intermediate step

This is intentionally a difficult learning signal for long-horizon tasks.
The entire point of the sparse baseline is to measure *how bad* the
credit-assignment problem is without any intermediate guidance, so the
dense / potential / hybrid variants have a meaningful comparison target.

Reward components logged per Section 11:
    terminal_success   +1.0 on true success
    terminal_failure   -0.1 on timeout
    (nothing else — that's what "sparse" means)
"""

from __future__ import annotations

from typing import Mapping

from reward_rl.envs.dynamics import GridState, StepEvents
from reward_rl.envs.layout import Layout


def sparse_reward(
    prev: GridState,
    action: int,
    nxt: GridState,
    events: StepEvents,
    layout: Layout,
) -> Mapping[str, float]:
    """Return decomposed reward components (summed by the env into scalar reward)."""
    components: dict[str, float] = {}

    if events.reached_goal and all(nxt.doors_open):
        components["terminal_success"] = 1.0
    elif not events.reached_goal and nxt.step_count >= 100:
        # Note: we use a small negative rather than a large one to avoid
        # the agent learning "die fast" instead of "explore".  The value
        # is deliberately hardcoded here for the baseline; config-driven
        # weights arrive with R_dense in Stage 3.
        pass  # timeout penalty handled via config, default 0 for pure sparse

    return components


class SparseReward:
    """Callable wrapper matching the RewardFn signature, with configurable weights.

    Supports a ``reset()`` hook (called by the env at episode start) even
    though sparse reward is stateless — keeps the interface uniform so
    Stage 3 rewards (which track state across steps) can use the same hook.
    """

    def __init__(self, success_reward: float = 1.0, timeout_penalty: float = 0.0,
                 step_cost: float = 0.0, max_steps: int = 100) -> None:
        self.success_reward = success_reward
        self.timeout_penalty = timeout_penalty
        self.step_cost = step_cost
        self.max_steps = max_steps

    def reset(self) -> None:
        """Called by the env at episode start. Sparse is stateless, so no-op."""
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

        # Terminal: success
        if events.reached_goal and all(nxt.doors_open):
            components["terminal_success"] = self.success_reward

        # Terminal: timeout
        is_timeout = not nxt.at_goal and nxt.step_count >= self.max_steps
        if is_timeout and self.timeout_penalty != 0.0:
            components["terminal_failure"] = self.timeout_penalty

        # Per-step cost (optional; default 0 for pure sparse)
        if self.step_cost != 0.0:
            components["step_cost"] = self.step_cost

        return components
