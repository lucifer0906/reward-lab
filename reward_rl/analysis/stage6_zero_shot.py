"""Stage 6: zero-shot final-checkpoint evaluation on unseen layouts and action noise.

This module only loads saved checkpoints.  It never trains or changes the
reward-blind evaluator, environment dynamics, PPO settings, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from reward_rl.envs.action_noise import ExecutedActionNoiseWrapper
from reward_rl.envs.key_door_env import KeyDoorChainEnv
from reward_rl.envs.wrappers import FlattenObsWrapper
from reward_rl.rewards import make_reward
from reward_rl.training.train import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = {
    "sparse": "configs/sparse.yaml",
    "dense": "configs/dense.yaml",
    "potential": "configs/potential.yaml",
    "hybrid": "configs/hybrid.yaml",
}
LAYOUTS = {
    "base": "configs/env/base_3keys.yaml",
    "L1_minor": "configs/env/l1_minor_start_moved.yaml",
    "L2_moderate": "configs/env/l2_moderate_pair_relocated.yaml",
    "L3_major": "configs/env/l3_major_all_pairs_relocated.yaml",
}
NOISE_CONDITIONS = {
    "base": 0.0,
    "noise_0.05": 0.05,
    "noise_0.10": 0.10,
    "noise_0.20": 0.20,
}


def bootstrap_ci(values: list[float], draws: int = 10000) -> list[float]:
    data = np.asarray(values, dtype=float)
    generator = np.random.default_rng(20261101)
    samples = generator.choice(data, size=(draws, len(data)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def retained_seeds() -> dict[str, set[int]]:
    path = REPO_ROOT / "runs" / "stage4_10seed_final_checkpoint" / "final_checkpoint_per_seed.csv"
    result = {variant: set() for variant in CORE}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["failure_taxonomy"] == "learned_and_retained":
                result[row["variant"]].add(int(row["seed"]))
    return result


def make_eval_env(config_path: str, layout_path: str, noise: float, seed: int):
    config = load_config(config_path)
    raw = KeyDoorChainEnv.from_config(
        layout_path, reward_fn=make_reward(config["reward"], max_steps=100)
    )
    noisy = ExecutedActionNoiseWrapper(raw, noise_probability=noise, rng_seed=seed * 1009)
    return FlattenObsWrapper(noisy), raw, noisy


def describe_failure(positions: list[tuple[int, int]], invalid_rate: float) -> str:
    unique = len(set(positions))
    if invalid_rate >= 0.5:
        return "repeated invalid moves into walls or closed doors"
    if unique <= 4:
        return "a short stuck loop with very little spatial progress"
    return "wanders through a repeated route without completing the required order"


def evaluate_model(
    checkpoint: Path, config_path: str, layout_path: str, noise: float, episodes: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env, raw, noisy = make_eval_env(config_path, layout_path, noise, seed)
    model = PPO.load(str(checkpoint), env=env, device="cpu")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        done = False
        proxy_return = 0.0
        requested: list[int] = []
        positions = [tuple(raw.state.agent_pos)]
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            requested.append(int(action))
            observation, reward, terminated, truncated, info = env.step(int(action))
            proxy_return += float(reward)
            positions.append(tuple(raw.state.agent_pos))
            done = bool(terminated or truncated)
        evaluation = info["evaluation"]
        row = {
            "episode": episode,
            "success": bool(evaluation["success"]),
            "proxy_return": proxy_return,
            "episode_length": int(evaluation["episode_length"]),
            "subgoals": int(evaluation["subgoals_completed"]),
            "invalid_action_rate": float(evaluation["invalid_action_rate"]),
        }
        rows.append(row)
        if not row["success"] and len(failures) < 2:
            failures.append({
                **row,
                "requested_actions": requested,
                "executed_actions": noisy.executed_actions.copy(),
                "description": describe_failure(positions, row["invalid_action_rate"]),
            })
    env.close()
    return rows, failures


def write_gif(config_path: str, actions: list[int], output: Path) -> None:
    raw = KeyDoorChainEnv.from_config(config_path, render_mode="ansi")
    raw.reset(seed=0)
    frames: list[Image.Image] = []
    for action in [None, *actions]:
        if action is not None:
            _, _, terminated, truncated, _ = raw.step(action)
        text = raw.render() or ""
        image = Image.new("RGB", (500, 190), "white")
        ImageDraw.Draw(image).text((8, 8), text, fill="black", spacing=3)
        frames.append(image)
        if action is not None and (terminated or truncated):
            break
    raw.close()
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=80, loop=0)


def aggregate(per_seed: list[dict[str, Any]], retained: dict[str, set[int]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for view in ("learned_and_retained", "all_40"):
        for variant in CORE:
            allowed = retained[variant] if view == "learned_and_retained" else set(range(10))
            for condition in sorted({row["condition"] for row in per_seed}):
                rows = [row for row in per_seed if row["variant"] == variant and row["condition"] == condition
                        and row["seed"] in allowed]
                success = [row["success_rate"] for row in rows]
                output.append({
                    "view": view, "variant": variant, "condition": condition, "n_seeds": len(rows),
                    "success_rate": float(np.mean(success)),
                    "success_ci_low": bootstrap_ci(success)[0], "success_ci_high": bootstrap_ci(success)[1],
                    "mean_episode_length": float(np.mean([row["mean_episode_length"] for row in rows])),
                    "mean_subgoals": float(np.mean([row["mean_subgoals"] for row in rows])),
                })
    return output


def write_plots(summary: list[dict[str, Any]], output: Path) -> None:
    primary = [row for row in summary if row["view"] == "learned_and_retained"]
    figure, axis = plt.subplots(figsize=(8, 5))
    levels = [("base", 0.0), ("noise_0.05", .05), ("noise_0.10", .10), ("noise_0.20", .20)]
    for variant in CORE:
        rows = {row["condition"]: row for row in primary if row["variant"] == variant}
        axis.plot([level for _, level in levels], [rows[name]["success_rate"] for name, _ in levels], marker="o", label=variant)
    axis.set(xlabel="Executed-action noise probability", ylabel="Ground-truth success rate", ylim=(-.05, 1.05), title="Zero-shot robustness: learned-and-retained checkpoints")
    axis.grid(alpha=.3); axis.legend(); figure.tight_layout(); figure.savefig(output / "success_vs_noise.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 5))
    names = ["base", "L1_minor", "L2_moderate", "L3_major"]
    x = np.arange(len(names)); width = .19
    for index, variant in enumerate(CORE):
        rows = {row["condition"]: row for row in primary if row["variant"] == variant}
        axis.bar(x + (index - 1.5) * width, [rows[name]["success_rate"] for name in names], width, label=variant)
    axis.set_xticks(x, names); axis.set(ylabel="Ground-truth success rate", ylim=(0, 1.05), title="Zero-shot layouts (base shown as reference)")
    axis.grid(axis="y", alpha=.3); axis.legend(); figure.tight_layout(); figure.savefig(output / "success_by_layout.png", dpi=160); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 saved-checkpoint-only zero-shot evaluation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "stage6_zero_shot")
    parser.add_argument("--summarize-existing", action="store_true",
                        help="Regenerate aggregates/plots from an already completed per_seed.csv.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retained = retained_seeds()
    if args.summarize_existing:
        with open(args.output_dir / "per_seed.csv", newline="", encoding="utf-8") as handle:
            per_seed = [{**row, "seed": int(row["seed"]), "success_rate": float(row["success_rate"]),
                         "mean_episode_length": float(row["mean_episode_length"]),
                         "mean_subgoals": float(row["mean_subgoals"])} for row in csv.DictReader(handle)]
        summary = aggregate(per_seed, retained)
        with open(args.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        write_plots(summary, args.output_dir)
        with open(args.output_dir / "report.json", "w", encoding="utf-8") as handle:
            json.dump({"protocol": {"episodes": 100, "deterministic_policy": True, "checkpoints": 40},
                       "retained_seeds": {k: sorted(v) for k, v in retained.items()}, "summary": summary}, handle,
                      indent=2, ensure_ascii=True)
        print(f"Regenerated Stage 6 summary in {args.output_dir}")
        return
    per_seed: list[dict[str, Any]] = []
    why: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in LAYOUTS if name != "base"}
    collapsed_actions: dict[str, dict[str, Any]] = {}
    conditions = [(name, path, 0.0) for name, path in LAYOUTS.items()] + [
        (name, LAYOUTS["base"], noise) for name, noise in NOISE_CONDITIONS.items() if name != "base"
    ]
    for variant, config_path in CORE.items():
        for seed in range(10):
            checkpoint = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            for condition, layout_path, noise in conditions:
                rows, failures = evaluate_model(checkpoint, config_path, layout_path, noise, args.episodes, seed)
                per_seed.append({
                    "variant": variant, "seed": seed, "condition": condition, "layout": Path(layout_path).stem,
                    "noise_probability": noise, "episodes": args.episodes,
                    "success_rate": float(np.mean([row["success"] for row in rows])),
                    "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
                    "mean_subgoals": float(np.mean([row["subgoals"] for row in rows])),
                })
                if condition in why and seed in retained[variant]:
                    why[condition].setdefault(variant, []).extend(failures)
    for variant, seed in {"dense": 3, "hybrid": 4, "potential": 9}.items():
        checkpoint = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "checkpoints" / "final_model.zip"
        _, failures = evaluate_model(checkpoint, CORE[variant], LAYOUTS["base"], 0.0, 1, seed)
        if failures:
            collapsed_actions[f"{variant}_seed_{seed}"] = failures[0]
    with open(args.output_dir / "per_seed.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0])); writer.writeheader(); writer.writerows(per_seed)
    summary = aggregate(per_seed, retained)
    with open(args.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with open(args.output_dir / "failure_replays.json", "w", encoding="utf-8") as handle:
        json.dump(why, handle, indent=2, ensure_ascii=True)
    with open(args.output_dir / "collapsed_replays.json", "w", encoding="utf-8") as handle:
        json.dump(collapsed_actions, handle, indent=2, ensure_ascii=True)
    first = next(iter(collapsed_actions.values()))
    with open(args.output_dir / "collapsed_replay_actions.json", "w", encoding="utf-8") as handle:
        json.dump(first, handle, indent=2, ensure_ascii=True)
    write_gif(LAYOUTS["base"], first["executed_actions"], args.output_dir / "collapsed_replay.gif")
    write_plots(summary, args.output_dir)
    with open(args.output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump({"protocol": {"episodes": args.episodes, "deterministic_policy": True, "checkpoints": 40},
                   "retained_seeds": {k: sorted(v) for k, v in retained.items()}, "summary": summary}, handle,
                  indent=2, ensure_ascii=True)
    print(f"Saved Stage 6 zero-shot report to {args.output_dir}")


if __name__ == "__main__":
    main()
