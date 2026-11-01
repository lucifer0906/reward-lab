import numpy as np

from reward_rl.envs.action_noise import ExecutedActionNoiseWrapper
from reward_rl.envs.key_door_env import KeyDoorChainEnv


def test_action_noise_records_the_action_sent_to_the_environment():
    raw = KeyDoorChainEnv.from_config()
    env = ExecutedActionNoiseWrapper(raw, noise_probability=1.0, rng_seed=7)
    env.reset(seed=3)
    _, _, _, _, info = env.step(0)
    assert info["action_was_replaced"]
    assert info["executed_action"] in range(raw.action_space.n)
    assert raw.record is not None
    assert raw.record.actions == env.executed_actions == [info["executed_action"]]


def test_zero_noise_preserves_requested_actions():
    raw = KeyDoorChainEnv.from_config()
    env = ExecutedActionNoiseWrapper(raw, noise_probability=0.0)
    env.reset(seed=11)
    _, _, _, _, info = env.step(2)
    assert not info["action_was_replaced"]
    assert info["executed_action"] == 2
    assert np.array_equal(raw.record.actions, [2])
