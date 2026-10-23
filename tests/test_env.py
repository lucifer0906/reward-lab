import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from reward_rl.envs import ENV_ID, KeyDoorChainEnv
from tests.conftest import BASE_SOLUTION


@pytest.fixture
def env():
    e = KeyDoorChainEnv.from_config(render_mode="ansi")
    yield e
    e.close()


def test_gymnasium_api_compliance(env):
    gym_check_env(env, skip_render_check=True)


def test_sb3_api_compliance():
    sb3_check_env(KeyDoorChainEnv.from_config(), warn=True)


def test_gym_make_registered():
    e = gym.make(ENV_ID)
    obs, info = e.reset(seed=0)
    assert e.observation_space.contains(obs)
    e.close()


def test_random_policy_runs_and_terminates(env):
    """Random actions never crash; every episode ends by success or timeout."""
    rng = np.random.default_rng(0)
    for ep in range(30):
        obs, _ = env.reset(seed=ep)
        done, steps = False, 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
            steps += 1
            assert env.observation_space.contains(obs)
            assert reward == 0.0 and info["reward_components"] == {}  # Stage 1: no reward logic
            done = terminated or truncated
        assert steps <= env.max_steps
        assert "is_success" in info and "evaluation" in info
        if truncated:
            assert steps == env.max_steps and info["is_success"] is False
            assert info["evaluation"]["failure_reason"] == "timeout"


def test_scripted_solution_succeeds(env):
    env.reset(seed=0)
    for i, a in enumerate(BASE_SOLUTION):
        obs, reward, terminated, truncated, info = env.step(a)
        assert info["events"]["valid"]
        if i < len(BASE_SOLUTION) - 1:
            assert not terminated and not truncated
    assert terminated and not truncated
    assert info["is_success"] is True
    ev = info["evaluation"]
    assert ev["planning_efficiency"] == pytest.approx(1.0)
    assert ev["subgoal_completion_rate"] == pytest.approx(1.0)
    assert "@" in env.render() and "G" not in env.render()  # agent is standing on the goal


def test_timeout_when_idling(env):
    env.reset(seed=0)
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step(4)  # INTERACT on empty floor, forever
    assert truncated and not terminated
    assert info["is_success"] is False
    assert info["evaluation"]["invalid_action_rate"] == pytest.approx(1.0)


def test_step_after_episode_end_raises(env):
    env.reset(seed=0)
    for a in BASE_SOLUTION:
        env.step(a)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_observation_contents(env):
    obs, _ = env.reset(seed=0)
    assert obs["agent_pos"].tolist() == [1, 1]
    assert obs["keys_held"].tolist() == [0, 0, 0]
    assert obs["prev_action"] == 5
    assert obs["action_mask"].tolist() == [0, 1, 0, 1, 0]  # only DOWN and RIGHT are possible
    assert obs["time_remaining"][0] == pytest.approx(1.0)
    obs, *_ = env.step(1)
    assert obs["prev_action"] == 1 and obs["agent_pos"].tolist() == [2, 1]


def test_unsolvable_layout_rejected():
    with pytest.raises(ValueError, match="unsolvable"):
        KeyDoorChainEnv(["#######", "#SA#aG#", "#######"])


def test_reward_fn_hook_is_used():
    """Stage 2 preview: injected reward components are summed and logged separately."""
    calls = []

    def fake_reward(prev, action, nxt, events, layout):
        calls.append(action)
        return {"a": 1.0, "b": -0.25}

    e = KeyDoorChainEnv.from_config(reward_fn=fake_reward)
    e.reset(seed=0)
    _, reward, *_, info = e.step(1)
    assert reward == pytest.approx(0.75)
    assert info["reward_components"] == {"a": 1.0, "b": -0.25}
    assert calls == [1]
