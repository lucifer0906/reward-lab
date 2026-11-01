from pathlib import Path

from reward_rl.envs.key_door_env import KeyDoorChainEnv
from reward_rl.envs.solver import solve


EXPECTED = {
    "stage7_h10_5keys.yaml": (2, 36),
    "stage7_h50_5keys.yaml": (5, 204),
    "stage7_h100_6keys.yaml": (6, 408),
}


def test_stage7_layouts_are_solvable_and_have_ordered_pairs():
    root = Path("configs/env")
    for filename, (pairs, step_limit) in EXPECTED.items():
        env = KeyDoorChainEnv.from_config(root / filename)
        plan = solve(env.layout)
        assert plan is not None
        assert env.layout.num_pairs == pairs
        assert env.optimal_length == len(plan)
        assert env.max_steps == step_limit
        assert env.optimal_length <= env.max_steps
