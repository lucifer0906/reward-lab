"""Evaluate saved PPO checkpoints and write reproducible experiment reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import pointbiserialr
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from reward_rl.rewards import make_reward
from reward_rl.training.train import load_config
from reward_rl.envs.key_door_env import KeyDoorChainEnv
from reward_rl.envs.wrappers import FlattenObsWrapper


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE4 = {
    "sparse": "configs/sparse.yaml", "dense": "configs/dense.yaml",
    "potential": "configs/potential.yaml", "hybrid": "configs/hybrid.yaml",
}


def evaluate_saved_model(checkpoint: Path, config_path: str, episodes: int) -> tuple[list[dict], list[int]]:
    """Evaluate one saved model against reward-blind terminal evaluator output."""
    config = load_config(config_path)
    env_cfg = config["env"]
    reward_cfg = config["reward"]
    env_path = env_cfg["config"]
    raw_env = KeyDoorChainEnv.from_config(env_path, reward_fn=make_reward(reward_cfg, max_steps=100))
    env = FlattenObsWrapper(raw_env)
    model = PPO.load(str(checkpoint), env=env, device="cpu")
    rows: list[dict] = []
    selected_actions: list[int] = []
    best_failed_return = -float("inf")
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        total_reward = 0.0
        actions: list[int] = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            scalar_action = int(action)
            actions.append(scalar_action)
            obs, reward, terminated, truncated, info = env.step(scalar_action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        ev = info["evaluation"]
        row = {
            "episode": episode,
            "proxy_return": total_reward,
            "success": bool(ev["success"]),
            "episode_length": int(ev["episode_length"]),
            "subgoals_completed": int(ev["subgoals_completed"]),
            "subgoal_completion_rate": float(ev["subgoal_completion_rate"]),
            "invalid_action_rate": float(ev["invalid_action_rate"]),
            "failure_reason": ev["failure_reason"],
        }
        rows.append(row)
        if not row["success"] and total_reward > best_failed_return:
            best_failed_return = total_reward
            selected_actions = actions
    env.close()
    return rows, selected_actions


def bootstrap_ci(values: list[float], draws: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(20261025)
    samples = generator.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def latest_event_dir(variant: str, seed: int) -> Path:
    root = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "tb_logs"
    candidates = list(root.glob(f"seed_{seed}_*"))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_window(variant: str, seed: int, start: int, end: int, tags: list[str]) -> dict:
    accumulator = EventAccumulator(str(latest_event_dir(variant, seed)))
    accumulator.Reload()
    result: dict[str, dict | None] = {}
    available = set(accumulator.Tags()["scalars"])
    for tag in tags:
        if tag not in available:
            result[tag] = None
            continue
        points = [point.value for point in accumulator.Scalars(tag) if start <= point.step <= end]
        result[tag] = None if not points else {
            "n": len(points), "first": float(points[0]), "last": float(points[-1]),
            "mean": float(np.mean(points)), "min": float(np.min(points)), "max": float(np.max(points)),
        }
    return result


def learning_taxonomy(variant: str, seed: int, final_success_rate: float) -> tuple[str, int | None]:
    """Classify a run from its rolling evaluator curve and final checkpoint."""
    accumulator = EventAccumulator(str(latest_event_dir(variant, seed)))
    accumulator.Reload()
    points = accumulator.Scalars("ground_truth/success_rate")
    first_90 = next((int(point.step) for point in points if point.value >= 0.9), None)
    if first_90 is None:
        return "never_learned", None
    if final_success_rate < 0.9:
        return "learned_then_collapsed", first_90
    return "learned_and_retained", first_90


def write_scatter(rows: list[dict], output: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    values = np.asarray([row["proxy_return"] for row in rows])
    successes = np.asarray([float(row["success"]) for row in rows])
    jitter = np.linspace(-0.025, 0.025, len(rows)) if rows else np.array([])
    axis.scatter(values, successes + jitter, alpha=0.6, s=18)
    axis.set_xlabel("Final checkpoint proxy return")
    axis.set_ylabel("Ground-truth success")
    axis.set_title(title)
    axis.set_ylim(-0.1, 1.1)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_comparison_plot(seed_rows: list[dict], output: Path) -> None:
    """Plot final deterministic seed success rates, the primary comparison metric."""
    variants = sorted({row["variant"] for row in seed_rows})
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, variant in enumerate(variants):
        rates = [row["final_success_rate"] for row in seed_rows if row["variant"] == variant]
        axis.scatter([index] * len(rates), rates, s=55, alpha=0.8)
        axis.hlines(np.median(rates), index - 0.25, index + 0.25, color="black", linewidth=2)
    axis.set_xticks(range(len(variants)), variants, rotation=20, ha="right")
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Final deterministic success rate (200 episodes per seed)")
    axis.set_title("Final saved checkpoint performance")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_weight_sweep_plot(summary: dict[str, dict], output: Path) -> None:
    """Plot final success and reward-hacking incidence for naive-distance c values."""
    rows = [
        values for values in summary.values()
        if values["reward_distance_decrease_c"] is not None
    ]
    if not rows:
        return
    rows.sort(key=lambda values: values["reward_distance_decrease_c"])
    c_values = [values["reward_distance_decrease_c"] for values in rows]
    success = [values["mean_final_success_rate"] for values in rows]
    incidence = [values["hacking_incidence_above_success_median"] for values in rows]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(c_values, success, marker="o", label="Final success")
    axis.plot(c_values, incidence, marker="s", label="Hacking incidence")
    axis.axvline(1.0 / 30.0, color="black", linestyle="--", label="Analytic break-even")
    axis.set_xlabel("Naive distance coefficient c")
    axis.set_ylabel("Fraction")
    axis.set_ylim(-0.05, 1.05)
    axis.set_title("Naive distance reward weight sweep")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_replay_gif(config_path: str, actions: list[int], output: Path) -> None:
    """Render a saved failed-policy action sequence as an ASCII-grid GIF."""
    config = load_config(config_path)
    raw_env = KeyDoorChainEnv.from_config(config["env"]["config"], render_mode="ansi")
    raw_env.reset(seed=0)
    frames: list[Image.Image] = []
    for action in [None] + actions:
        if action is not None:
            _, _, terminated, truncated, _ = raw_env.step(action)
        text = raw_env.render() or ""
        image = Image.new("RGB", (500, 190), "white")
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), text, fill="black", spacing=3)
        frames.append(image)
        if action is not None and (terminated or truncated):
            break
    raw_env.close()
    if frames:
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=80, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report deterministic final-checkpoint results")
    parser.add_argument("--variant", action="append", default=None, metavar="NAME=CONFIG")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stage4", action="store_true")
    parser.add_argument("--audit-results", type=Path, default=None,
                        help="Optional scripted audit JSON supplying control returns.")
    args = parser.parse_args()
    variants = dict(STAGE4) if args.stage4 else {}
    for item in args.variant or []:
        name, separator, config = item.partition("=")
        if not separator or not name or not config:
            raise ValueError("--variant must be NAME=CONFIG")
        variants[name] = config
    if not variants:
        raise ValueError("Specify --stage4 or at least one --variant")

    audit_controls: dict[str, float] = {}
    if args.audit_results:
        with open(args.audit_results, "r", encoding="utf-8") as handle:
            audit_data = json.load(handle)
        audit_controls = {
            name: float(values["successful_proxy_return"])
            for name, values in audit_data["summary"].items()
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict] = []
    episode_rows: list[dict] = []
    all_actions: dict[str, dict[str, list[int]]] = {}
    for variant, config in variants.items():
        all_actions[variant] = {}
        for seed in args.seeds:
            checkpoint = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            rows, actions = evaluate_saved_model(checkpoint, config, args.episodes)
            all_actions[variant][str(seed)] = actions
            for row in rows:
                episode_rows.append({"variant": variant, "seed": seed, **row})
            seed_rows.append({
                "variant": variant, "seed": seed, "checkpoint": str(checkpoint),
                "episodes": args.episodes,
                "final_success_rate": float(np.mean([row["success"] for row in rows])),
                "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
                "mean_subgoals": float(np.mean([row["subgoals_completed"] for row in rows])),
                "mean_subgoal_rate": float(np.mean([row["subgoal_completion_rate"] for row in rows])),
                "mean_proxy_return": float(np.mean([row["proxy_return"] for row in rows])),
            })
            if args.stage4:
                taxonomy, step_90 = learning_taxonomy(variant, seed, seed_rows[-1]["final_success_rate"])
                seed_rows[-1]["failure_taxonomy"] = taxonomy
                seed_rows[-1]["steps_to_90pct_rolling_success"] = step_90
        write_scatter(
            [row for row in episode_rows if row["variant"] == variant],
            args.output_dir / f"{variant}_final_proxy_vs_success.png",
            f"{variant}: final checkpoint proxy return vs success",
        )

    summary: dict[str, dict] = {}
    for variant in variants:
        reward_parameters = load_config(variants[variant])["reward"]
        rows = [row for row in seed_rows if row["variant"] == variant]
        rates = [row["final_success_rate"] for row in rows]
        all_rows = [row for row in episode_rows if row["variant"] == variant]
        returns = [row["proxy_return"] for row in all_rows]
        successes = [float(row["success"]) for row in all_rows]
        has_success_variance = len(set(successes)) > 1
        r_pb = float(pointbiserialr(successes, returns).statistic) if has_success_variance else None
        success_returns = [row["proxy_return"] for row in all_rows if row["success"]]
        control_median = float(np.median(success_returns)) if success_returns else audit_controls.get(variant)
        incidence = None if control_median is None else float(np.mean([
            row["proxy_return"] > control_median and not row["success"] for row in all_rows
        ]))
        failed_returns = [row["proxy_return"] for row in all_rows if not row["success"]]
        failed_mean = float(np.mean(failed_returns)) if failed_returns else None
        failed_gap = None if control_median is None or failed_mean is None else failed_mean - control_median
        summary[variant] = {
            "seeds_at_or_above_90pct": int(sum(rate >= 0.9 for rate in rates)),
            "total_seeds": len(rates),
            "median_final_success_rate": float(np.median(rates)),
            "mean_final_success_rate": float(np.mean(rates)),
            "bootstrap_95pct_ci_mean_success": bootstrap_ci(rates),
            "point_biserial_proxy_vs_success": r_pb,
            "mean_failed_proxy_return": failed_mean,
            "failed_minus_scripted_success_control_proxy_gap": failed_gap if not has_success_variance else None,
            "successful_control_median_proxy_return": control_median,
            "successful_control_source": "final_evaluation" if success_returns else (
                "scripted_audit" if control_median is not None else None
            ),
            "hacking_incidence_above_success_median": incidence,
            "reward_distance_decrease_c": reward_parameters.get("distance_decrease_reward"),
            "analytic_distance_break_even_c": (
                1.0 / 30.0 if reward_parameters.get("type") == "naive_distance" else None
            ),
            "reward_shaping_gamma": reward_parameters.get("gamma"),
        }

    with open(args.output_dir / "final_checkpoint_per_seed.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with open(args.output_dir / "final_checkpoint_episodes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    with open(args.output_dir / "failed_episode_actions.json", "w", encoding="utf-8") as handle:
        json.dump(all_actions, handle, indent=2, ensure_ascii=True)
    for variant, by_seed in all_actions.items():
        replay_seed, replay_actions = next(
            ((seed, actions) for seed, actions in by_seed.items() if actions), (None, [])
        )
        if replay_seed is not None:
            with open(args.output_dir / f"{variant}_hacking_replay_actions.json", "w", encoding="utf-8") as handle:
                json.dump({"variant": variant, "seed": int(replay_seed), "actions": replay_actions}, handle,
                          indent=2, ensure_ascii=True)
            write_replay_gif(variants[variant], replay_actions,
                             args.output_dir / f"{variant}_hacking_replay.gif")

    diagnostics = {}
    if args.stage4:
        tags = ["train/entropy_loss", "train/approx_kl", "train/clip_fraction", "train/value_loss", "train/explained_variance"]
        diagnostics["hybrid_seed_0_160k_190k"] = scalar_window("hybrid", 0, 160000, 190000, tags)
        diagnostics["potential_seed_1_full_training"] = scalar_window("potential", 1, 0, 200000, tags)
        for variant in variants:
            training_rates = []
            for seed in args.seeds:
                path = REPO_ROOT / "runs" / variant / f"seed_{seed}" / "results.json"
                with open(path, "r", encoding="utf-8") as handle:
                    training_rates.append(json.load(handle)["ground_truth"]["success_rate"])
            summary[variant]["secondary_whole_training_episode_weighted_success_mean"] = float(np.mean(training_rates))
            variant_rows = [row for row in seed_rows if row["variant"] == variant]
            summary[variant]["failure_taxonomy_counts"] = {
                name: sum(row["failure_taxonomy"] == name for row in variant_rows)
                for name in ("never_learned", "learned_then_collapsed", "learned_and_retained")
            }
            learned_steps = [row["steps_to_90pct_rolling_success"] for row in variant_rows
                             if row["steps_to_90pct_rolling_success"] is not None]
            summary[variant]["mean_steps_to_90pct_among_learned"] = (
                float(np.mean(learned_steps)) if learned_steps else None
            )

    report = {"protocol": {"episodes": args.episodes, "deterministic": True}, "summary": summary,
              "per_seed": seed_rows, "diagnostics": diagnostics}
    with open(args.output_dir / "final_checkpoint_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=True)
    with open(args.output_dir / "final_checkpoint_summary.csv", "w", newline="", encoding="utf-8") as handle:
        fields = ["variant", *sorted({key for values in summary.values() for key in values})]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, values in summary.items():
            writer.writerow({"variant": variant, **values})
    write_comparison_plot(seed_rows, args.output_dir / "final_checkpoint_success_comparison.png")
    write_weight_sweep_plot(summary, args.output_dir / "naive_distance_weight_sweep.png")
    print(f"Saved final-checkpoint report to: {args.output_dir}")


if __name__ == "__main__":
    main()
