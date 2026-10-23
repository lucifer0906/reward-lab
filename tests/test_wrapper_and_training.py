"""Tests for the FlattenObsWrapper and training pipeline."""

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from reward_rl.envs.key_door_env import KeyDoorChainEnv
from reward_rl.envs.wrappers import FlattenObsWrapper
from reward_rl.rewards.sparse import SparseReward
from tests.conftest import BASE_SOLUTION


@pytest.fixture
def wrapped_env():
    reward_fn = SparseReward(success_reward=1.0, max_steps=100)
    env = KeyDoorChainEnv.from_config(reward_fn=reward_fn)
    env = FlattenObsWrapper(env)
    yield env
    env.close()


def test_wrapper_flattens_to_box(wrapped_env):
    """FlattenObsWrapper converts Dict obs to a 1-D float32 Box."""
    obs, _ = wrapped_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.ndim == 1
    assert wrapped_env.observation_space.contains(obs)


def test_wrapper_values_normalized(wrapped_env):
    """All values in the flat observation should be in [0, 1]."""
    rng = np.random.default_rng(42)
    wrapped_env.reset(seed=0)
    for _ in range(200):
        obs, _, term, trunc, _ = wrapped_env.step(int(rng.integers(wrapped_env.action_space.n)))
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"Out of range: {obs}"
        if term or trunc:
            obs, _ = wrapped_env.reset()


def test_sb3_check_env_on_wrapper(wrapped_env):
    """SB3's env checker should pass on the wrapped env."""
    sb3_check_env(wrapped_env, warn=True)


def test_ppo_can_train_one_step(wrapped_env):
    """PPO can instantiate and train for a tiny number of steps without crashing."""
    model = PPO("MlpPolicy", wrapped_env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    model.learn(total_timesteps=128)


def test_ppo_predict_after_training(wrapped_env):
    """A trained model can predict actions on the wrapped env."""
    model = PPO("MlpPolicy", wrapped_env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    model.learn(total_timesteps=128)
    obs, _ = wrapped_env.reset(seed=99)
    action, _ = model.predict(obs, deterministic=True)
    assert wrapped_env.action_space.contains(int(action))


def test_scripted_solution_with_reward(wrapped_env):
    """Scripted optimal solution through the wrapper gives reward and success."""
    obs, _ = wrapped_env.reset(seed=0)
    total_reward = 0.0
    for a in BASE_SOLUTION:
        obs, reward, term, trunc, info = wrapped_env.step(a)
        total_reward += reward
    assert term and not trunc
    assert info["is_success"]
    assert total_reward == pytest.approx(1.0)  # only terminal_success
