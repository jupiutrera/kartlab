import math
import warnings

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from kartlab.env import ENV_ID
from kartlab.env.kart_env import EnvConfig, KartEnv
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.tracksets import TRAIN_SEED_LIMIT

STILL = np.array([0.0, 0.0], dtype=np.float32)
NO_LIMITS = {"stuck_seconds": 1e9, "max_seconds": 1e9}


def fixed_env(**overrides) -> KartEnv:
    """Start on the line of a fixed showcase track."""
    defaults = {"track": "plateau", "random_start": False}
    env = KartEnv(**{**defaults, **overrides})
    env.reset(seed=0)
    return env


def place(env: KartEnv, s: float, lateral: float = 0.0) -> None:
    """Put the kart at rest at arc position ``s`` (kinematic move, for reward tests)."""
    kart = env.racer.kart
    kart.x, kart.y, kart.heading = env.track.pose_at(s, lateral)
    kart.vx = kart.vy = 0.0


def drive_with_autopilot(env: KartEnv, max_steps: int = 5000):
    pilot = Autopilot(env.track)
    rewards, infos = [], []
    for _ in range(max_steps):
        action = np.array(pilot.act(env.racer), dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        infos.append(info)
        if terminated or truncated:
            break
    return rewards, infos


# -- API ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [{}, {"extra_observations": False}, {"track": "harbor"}, {"terminate_off_track": False}],
    ids=["default", "spec-obs-only", "fixed-track", "no-off-track-end"],
)
def test_passes_stable_baselines3_check_env(overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(KartEnv(**overrides), warn=True)


def test_is_registered_with_gymnasium():
    env = gym.make(ENV_ID, track="harbor")
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("extra, size", [(False, 16), (True, 18)])
def test_observation_layout(extra, size):
    env = KartEnv(extra_observations=extra, ray_count=11)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (size,) and obs.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, *_, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            env.reset()


def test_observation_on_the_line_of_a_straight():
    env = fixed_env()
    obs, _ = env.reset()
    rays = obs[:11]
    assert rays[5] == pytest.approx(1.0)  # straight ahead: nothing within range
    assert rays[0] == pytest.approx(rays[-1], abs=0.02)  # centred between the edges
    speed, sin_err, cos_err, prev_steer, prev_throttle, wheel, slip = obs[11:]
    assert speed == 0.0 and prev_steer == 0.0 and prev_throttle == 0.0
    assert wheel == 0.0 and slip == 0.0
    assert sin_err == pytest.approx(0.0, abs=1e-3) and cos_err == pytest.approx(1.0)


def test_previous_action_is_observed():
    env = fixed_env()
    obs, *_ = env.step(np.array([0.3, -0.7], dtype=np.float32))
    assert obs[14:16] == pytest.approx([0.3, -0.7])


def test_same_seed_same_episode_and_new_tracks_each_episode():
    def rollout(seed):
        env = KartEnv()
        obs, info = env.reset(seed=seed)
        trace = [obs]
        for _ in range(30):
            obs, *_ = env.step(np.array([0.2, 0.8], dtype=np.float32))
            trace.append(obs)
        return info["track_seed"], np.array(trace)

    seed_a, trace_a = rollout(11)
    seed_b, trace_b = rollout(11)
    assert seed_a == seed_b
    np.testing.assert_array_equal(trace_a, trace_b)
    assert rollout(12)[0] != seed_a


def test_random_episodes_never_use_evaluation_tracks():
    env = KartEnv()
    env.reset(seed=0)
    seeds = {env.reset()[1]["track_seed"] for _ in range(30)}
    assert len(seeds) == 30
    assert all(seed < TRAIN_SEED_LIMIT for seed in seeds)


def test_random_start_spreads_the_kart_over_the_track():
    env = KartEnv(track="harbor")
    env.reset(seed=0)
    starts = [env.reset()[1]["progress"] for _ in range(5)]
    positions = []
    for _ in range(20):
        env.reset()
        positions.append(env.racer.position.s)
    assert starts == [0.0] * 5
    assert np.ptp(positions) > env.track.length / 2


def test_track_ranges_can_change_between_episodes():
    env = KartEnv(track_width=(12.0, 12.0))
    env.reset(seed=0)
    env.set_track_ranges(width=(16.0, 16.0), curviness=(0.0, 0.2))
    env.reset()
    assert env.track.width == 16.0


# -- reward ---------------------------------------------------------------------------


def test_progress_is_rewarded_per_metre():
    env = fixed_env(**NO_LIMITS)
    total = 0.0
    for step in range(1, 41):
        place(env, step * 1.5)
        total += env.step(STILL)[1]
    assert total == pytest.approx(0.1 * 40 * 1.5, abs=1e-6)


def test_driving_backwards_is_penalised():
    env = fixed_env(**NO_LIMITS)
    place(env, 200.0)
    env.step(STILL)
    total = 0.0
    for step in range(1, 21):
        place(env, 200.0 - step)
        total += env.step(STILL)[1]
    assert total == pytest.approx(-0.1 * 20, abs=1e-6)


def test_going_back_and_forth_over_the_line_adds_nothing():
    env = fixed_env(**NO_LIMITS)
    length = env.track.length
    path = [(-10.0 + k) % length for k in range(21)] + [(10.0 - k) % length for k in range(21)]
    place(env, path[0])
    env.step(STILL)
    total = 0.0
    for _ in range(15):
        for s in path:
            place(env, s)
            _, reward, terminated, truncated, info = env.step(STILL)
            total += reward
            assert not (terminated or truncated)
    assert total == pytest.approx(0.0, abs=1e-6)
    assert info["laps"] == 0


def test_back_and_forth_with_real_physics_earns_only_the_net_distance():
    env = fixed_env(track="harbor", **NO_LIMITS)
    total, travelled, direction, turns = 0.0, 0.0, 1.0, 0
    while turns < 8:  # forward to 8 m past the line, back to 8 m before it, repeat
        distance = env.racer.progress.distance
        if direction * distance > 8.0:
            direction, turns = -direction, turns + 1
        total += env.step(np.array([0.0, direction], dtype=np.float32))[1]
        travelled += abs(env.racer.progress.distance - distance)
    assert travelled > 100.0
    assert total == pytest.approx(0.1 * env.racer.progress.distance, abs=1e-6)
    assert abs(total) < 1.5
    assert env.racer.laps == 0


def test_spinning_in_circles_adds_nothing():
    env = fixed_env(terminate_off_track=False, **NO_LIMITS)
    spin = np.array([1.0, 0.3], dtype=np.float32)
    total = sum(env.step(spin)[1] for _ in range(60 * 15))  # one minute
    assert abs(total) < 1.5
    assert env.racer.laps == 0

    straight = fixed_env(**NO_LIMITS)
    forward = sum(straight.step(np.array([0.0, 1.0], dtype=np.float32))[1] for _ in range(5 * 15))
    assert forward > 5.0


def test_each_lap_earns_one_bonus():
    env = fixed_env(track="harbor", max_laps=2)
    rewards, infos = drive_with_autopilot(env)
    assert infos[-1]["laps"] == 2
    progress = 0.1 * infos[-1]["progress"]
    steering = sum(rewards) - progress - 2 * 10.0
    assert -1.0 < steering <= 0.0  # only the small jerky-steering penalty remains
    bonus_steps = [i for i in range(1, len(infos)) if infos[i]["laps"] > infos[i - 1]["laps"]]
    assert len(bonus_steps) == 2


def test_jerky_steering_costs_a_little():
    env = fixed_env(**NO_LIMITS)
    smooth = [env.step(np.array([0.5, 0.0], dtype=np.float32))[1] for _ in range(10)]
    env.reset()
    jerky = [env.step(np.array([(-1.0) ** k, 0.0], dtype=np.float32))[1] for k in range(10)]
    assert sum(smooth) == pytest.approx(-0.025 * 0.25)
    assert sum(jerky) == pytest.approx(-0.025 * (1.0 + 9 * 4.0))


def test_standing_still_earns_nothing_and_ends_as_stuck():
    env = fixed_env(stuck_seconds=4.0)
    rewards = []
    for _ in range(200):
        _, reward, terminated, truncated, info = env.step(STILL)
        rewards.append(reward)
        if terminated:
            break
    assert terminated and info["stuck"] and not truncated
    assert set(rewards) == {0.0}
    assert info["time"] == pytest.approx(4.0, abs=0.1)


def test_leaving_the_track_is_penalised_and_ends_the_episode():
    env = fixed_env()
    for _ in range(600):
        _, reward, terminated, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        if terminated:
            break
    assert terminated and info["off_track"]
    assert reward < -9.0
    lateral = abs(env.racer.position.lateral)
    assert env.track.half_width + 0.5 < lateral < env.track.half_width + 1.5


def test_without_off_track_ending_the_kart_keeps_rolling_on_the_grass():
    env = fixed_env(terminate_off_track=False, **NO_LIMITS)
    off_steps, widest = 0, 0.0
    for _ in range(600):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        off_steps += info["off_track"]
        widest = max(widest, abs(env.racer.position.lateral))
        assert not (terminated or truncated)
    assert off_steps > 10
    assert widest > env.track.half_width + 2.0


def test_time_limit_truncates():
    env = fixed_env(max_seconds=2.0)
    steps, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
        steps += 1
    assert truncated and not terminated
    assert steps == math.ceil(2.0 * 15)


def test_off_track_tolerance_is_configurable():
    strict = fixed_env(off_track_tolerance=0.0)
    lenient = fixed_env(off_track_tolerance=3.0)
    for env in (strict, lenient):
        for _ in range(600):
            *_, terminated, _, _ = env.step(np.array([1.0, 1.0], dtype=np.float32))
            if terminated:
                break
    assert abs(strict.racer.position.lateral) < abs(lenient.racer.position.lateral)


def test_config_validation():
    with pytest.raises(ValueError):
        EnvConfig(action_repeat=0)
    with pytest.raises(ValueError):
        EnvConfig(track_width=(16.0, 12.0))
