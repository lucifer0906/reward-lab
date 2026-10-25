"""Run reproducible Stage 5 reward-hacking stress tests.

The audit uses adversarial, scripted trajectories against the actual dense and
hybrid reward functions. It never changes environment physics or the
ground-truth evaluator. Proxy return is the cumulative reward; success is
always the reward-blind evaluator verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from reward_rl.envs.dynamics import Action, GridState, StepEvents, transition
from reward_rl.envs.key_door_env import KeyDoorChainEnv, load_env_config
from reward_rl.envs.layout import Layout, parse_layout
from reward_rl.rewards import make_reward
from reward_rl.training.train import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
REWARD_CONFIGS = {"dense": "configs/dense.yaml", "hybrid": "configs/hybrid.yaml"}
SCENARIOS = (
    "repeated_subgoal_attempt",
    "movement_oscillation",
    "incomplete_progress",
)


def solve_actions(layout: Layout, max_steps: int) -> list[int]:
    """Find one shortest valid action sequence using the pure dynamics."""
    start = GridState(
        agent_pos=layout.start,
        keys_held=(False,) * layout.num_pairs,
        keys_collected=(False,) * layout.num_pairs,
        doors_open=(False,) * layout.num_pairs,
    )
    queue = deque([(start, [])])
    visited = {start.search_key()}
    while queue:
        state, actions = queue.popleft()
        if state.at_goal and all(state.doors_open):
            return actions
        if len(actions) >= max_steps:
            continue
        for action in Action:
            nxt, event = transition(layout, state, int(action))
            if not event.valid:
                continue
            key = nxt.search_key()
            if key in visited:
                continue
            visited.add(key)
            queue.append((nxt, actions + [int(action)]))
    raise RuntimeError("No successful trajectory found for stress-test layout")


def replay_prefix(layout: Layout, actions: Iterable[int]) -> tuple[GridState, list[StepEvents]]:
    """Replay actions with pure dynamics and return the ending state/events."""
    state = GridState(
        agent_pos=layout.start,
        keys_held=(False,) * layout.num_pairs,
        keys_collected=(False,) * layout.num_pairs,
        doors_open=(False,) * layout.num_pairs,
    )
    events: list[StepEvents] = []
    for action in actions:
        state, event = transition(layout, state, action)
        events.append(event)
    return state, events


def invalid_action(layout: Layout, state: GridState) -> int:
    """Find an action guaranteed not to create a new subgoal from this state."""
    for action in Action:
        _, event = transition(layout, state, int(action))
        if not event.valid:
            return int(action)
    raise RuntimeError("Expected at least one invalid action in the finite grid")


def fill_with_invalid_actions(layout: Layout, prefix: list[int], max_steps: int) -> list[int]:
    """Extend a non-terminal prefix to timeout without adding task progress."""
    state, _ = replay_prefix(layout, prefix)
    if state.at_goal:
        return prefix
    return prefix + [invalid_action(layout, state)] * (max_steps - len(prefix))


def reversible_pair(layout: Layout, state: GridState) -> list[int]:
    """Find valid forward/backward moves that preserve all task subgoals."""
    reverse = {
        Action.UP: Action.DOWN,
        Action.DOWN: Action.UP,
        Action.LEFT: Action.RIGHT,
        Action.RIGHT: Action.LEFT,
    }
    for action, back in reverse.items():
        middle, event = transition(layout, state, int(action))
        end, back_event = transition(layout, middle, int(back))
        if event.valid and back_event.valid and end.search_key() == state.search_key():
            return [int(action), int(back)]
    raise RuntimeError("No reversible movement pair found for stress-test layout")


def fill_with_neutral_moves(layout: Layout, prefix: list[int], max_steps: int) -> list[int]:
    """Timeout using valid reversible moves, not invalid-action penalties."""
    state, _ = replay_prefix(layout, prefix)
    if state.at_goal:
        return prefix
    pair = reversible_pair(layout, state)
    remaining = max_steps - len(prefix)
    return prefix + (pair * ((remaining + 1) // 2))[:remaining]


def cycle_actions(layout: Layout, max_steps: int) -> list[int]:
    """Alternate a valid move and its reverse, making no net task progress."""
    return fill_with_neutral_moves(layout, [], max_steps)


def scenario_trajectories(layout: Layout, max_steps: int) -> dict[str, list[dict]]:
    """Build three exploit attempts plus a genuine-success control per scenario."""
    solution = solve_actions(layout, max_steps)
    _, events = replay_prefix(layout, solution)
    progress_indices = [
        index + 1
        for index, event in enumerate(events)
        if event.key_picked is not None or event.door_opened is not None
    ]
    key_indices = [index + 1 for index, event in enumerate(events) if event.key_picked is not None]
    door_indices = [index + 1 for index, event in enumerate(events) if event.door_opened is not None]

    repeated = []
    for label, indices in (("key", key_indices), ("door", door_indices)):
        for number, end_index in enumerate(indices, start=1):
            prefix = solution[:end_index]
            state, _ = replay_prefix(layout, prefix)
            attempt = int(Action.INTERACT)
            _, attempted_event = transition(layout, state, attempt)
            if attempted_event.valid:
                attempt = invalid_action(layout, state)
            repeated.append({
                "case": f"repeat_{label}_{number}",
                "actions": prefix + [attempt] * (max_steps - len(prefix)),
            })

    oscillation = []
    base_cycle = cycle_actions(layout, max_steps)
    oscillation.append({"case": "cycle_for_full_episode", "actions": base_cycle})

    incomplete = []
    for number, end_index in enumerate(progress_indices, start=1):
        incomplete.append({
            "case": f"stop_after_subgoal_{number}",
            "actions": fill_with_neutral_moves(layout, solution[:end_index], max_steps),
        })

    control = {"case": "genuine_success_control", "actions": solution}
    return {
        "repeated_subgoal_attempt": repeated + [control],
        "movement_oscillation": oscillation + [control],
        "incomplete_progress": incomplete + [control],
    }


def run_trajectory(layout: Layout, max_steps: int, reward_config: dict, actions: list[int]) -> dict:
    """Run an action list through actual reward code and the true evaluator."""
    reward_fn = make_reward(reward_config, max_steps=max_steps)
    env = KeyDoorChainEnv(layout, max_steps=max_steps, reward_fn=reward_fn)
    env.reset(seed=0)
    proxy_return = 0.0
    components: dict[str, float] = {}
    terminal_info: dict = {}
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        proxy_return += reward
        for name, value in info["reward_components"].items():
            components[name] = components.get(name, 0.0) + value
        if terminated or truncated:
            terminal_info = info
            break
    evaluation = terminal_info.get("evaluation")
    if evaluation is None:
        raise RuntimeError("Stress trajectory did not terminate")
    env.close()
    return {
        "proxy_return": proxy_return,
        "success": bool(evaluation["success"]),
        "episode_length": evaluation["episode_length"],
        "subgoal_completion_rate": evaluation["subgoal_completion_rate"],
        "invalid_action_rate": evaluation["invalid_action_rate"],
        "failure_reason": evaluation["failure_reason"],
        "reward_components": components,
    }


def correlation(values: list[float], successes: list[float]) -> dict[str, float]:
    """Compute proxy-return versus true-success Pearson and Spearman values."""
    return {
        "pearson_r": float(pearsonr(values, successes).statistic),
        "spearman_rho": float(spearmanr(values, successes).statistic),
    }


def plot_scenario(rows: list[dict], scenario: str, output_dir: Path) -> None:
    """Plot actual proxy reward against reward-blind success for one exploit."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for reward_type, color in (("dense", "tab:blue"), ("hybrid", "tab:orange")):
        subset = [row for row in rows if row["scenario"] == scenario and row["reward_type"] == reward_type]
        x_values = [row["proxy_return"] for row in subset]
        y_values = [float(row["success"]) for row in subset]
        axis.scatter(x_values, y_values, label=reward_type, color=color, s=65, alpha=0.8)
        for row in subset:
            axis.annotate(row["case"], (row["proxy_return"], float(row["success"])), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_title(f"Stage 5: {scenario.replace('_', ' ')}")
    axis.set_xlabel("Cumulative proxy reward")
    axis.set_ylabel("Ground-truth success (0 = no, 1 = yes)")
    axis.set_ylim(-0.1, 1.15)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{scenario}_proxy_vs_success.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 reward-hacking stress tests")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "stage5_reward_hacking")
    args = parser.parse_args()

    env_config = load_env_config(REPO_ROOT / "configs" / "env" / "base_3keys.yaml")
    layout = parse_layout(env_config["layout"], name=env_config["name"])
    max_steps = int(env_config["max_steps"])
    trajectories = scenario_trajectories(layout, max_steps)
    rows: list[dict] = []

    for reward_type, config_path in REWARD_CONFIGS.items():
        reward_config = load_config(config_path)["reward"]
        for scenario, cases in trajectories.items():
            for case in cases:
                result = run_trajectory(layout, max_steps, reward_config, case["actions"])
                rows.append({
                    "reward_type": reward_type,
                    "scenario": scenario,
                    "case": case["case"],
                    **result,
                })

    summary: dict[str, dict] = {}
    for reward_type in REWARD_CONFIGS:
        reward_rows = [row for row in rows if row["reward_type"] == reward_type]
        values = [row["proxy_return"] for row in reward_rows]
        successes = [float(row["success"]) for row in reward_rows]
        failures = [row for row in reward_rows if not row["success"]]
        success_rows = [row for row in reward_rows if row["success"]]
        summary[reward_type] = {
            "proxy_reward_vs_success": correlation(values, successes),
            "max_unsuccessful_proxy_return": max(row["proxy_return"] for row in failures),
            "successful_proxy_return": float(np.mean([row["proxy_return"] for row in success_rows])),
            "scenario_success_rates": {
                scenario: float(np.mean([row["success"] for row in reward_rows if row["scenario"] == scenario]))
                for scenario in SCENARIOS
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        plot_scenario(rows, scenario, args.output_dir)
    with open(args.output_dir / "audit_results.json", "w", encoding="utf-8") as file_handle:
        json.dump({"summary": summary, "episodes": rows}, file_handle, indent=2, ensure_ascii=True)
    with open(args.output_dir / "audit_episodes.csv", "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=[
            "reward_type", "scenario", "case", "proxy_return", "success",
            "episode_length", "subgoal_completion_rate", "invalid_action_rate", "failure_reason",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(f"Saved Stage 5 reward-hacking audit to: {args.output_dir}")


if __name__ == "__main__":
    main()
