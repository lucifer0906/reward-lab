"""Watch the environment and the ground-truth evaluator in action.

    python -m reward_rl.envs.demo                 # optimal solution, step by step
    python -m reward_rl.envs.demo --random 200    # random-policy baseline stats
    python -m reward_rl.envs.demo --play          # play it yourself (w/a/s/d + e)
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from reward_rl.envs.dynamics import Action
from reward_rl.envs.key_door_env import DEFAULT_CONFIG, KeyDoorChainEnv
from reward_rl.envs.solver import solve


def show_optimal(env: KeyDoorChainEnv) -> None:
    env.reset(seed=0)
    print(env.render(), "\n")
    for a in solve(env.layout):
        _, _, term, trunc, info = env.step(a)
        ev = info["events"]
        note = (
            f"picked key {'abc'[ev['key_picked']]}" if ev["key_picked"] is not None
            else f"opened door {'ABC'[ev['door_opened']]}" if ev["door_opened"] is not None
            else "reached GOAL" if ev["reached_goal"] else ""
        )
        print(f"action: {Action(a).name:<8} {note}")
        print(env.render(), "\n")
    print("Evaluator verdict:", info["evaluation"])


def random_baseline(env: KeyDoorChainEnv, episodes: int) -> None:
    rng = np.random.default_rng(0)
    reasons, subgoals, invalid = Counter(), [], []
    for ep in range(episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
            done = term or trunc
        ev = info["evaluation"]
        reasons["success" if ev["success"] else ev["failure_reason"]] += 1
        subgoals.append(ev["subgoals_completed"])
        invalid.append(ev["invalid_action_rate"])
    print(f"Random policy over {episodes} episodes (max_steps={env.max_steps}, optimal={env.optimal_length}):")
    print(f"  outcomes:               {dict(reasons)}")
    print(f"  success rate:           {reasons['success'] / episodes:.1%}")
    print(f"  mean subgoals (of {2 * env.layout.num_pairs + 1}):  {np.mean(subgoals):.2f}")
    print(f"  subgoal histogram:      {dict(sorted(Counter(subgoals).items()))}")
    print(f"  mean invalid-action rate: {np.mean(invalid):.1%}")


def play(env: KeyDoorChainEnv) -> None:
    keys = {"w": Action.UP, "s": Action.DOWN, "a": Action.LEFT, "d": Action.RIGHT, "e": Action.INTERACT}
    env.reset(seed=0)
    print(env.render())
    while True:
        k = input("move [w/a/s/d], interact [e], quit [q]: ").strip().lower()
        if k == "q":
            return
        if k not in keys:
            continue
        _, _, term, trunc, info = env.step(keys[k])
        print(env.render(), "| event:", {k: v for k, v in info["events"].items() if v not in (None, False)})
        if term or trunc:
            print("Evaluator verdict:", info["evaluation"])
            return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--random", type=int, metavar="N", help="run N random-policy episodes")
    p.add_argument("--play", action="store_true")
    args = p.parse_args()
    env = KeyDoorChainEnv.from_config(args.config, render_mode="ansi")
    if args.play:
        play(env)
    elif args.random:
        random_baseline(env, args.random)
    else:
        show_optimal(env)


if __name__ == "__main__":
    main()
