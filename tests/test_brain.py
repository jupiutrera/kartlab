"""The brain panel: what it shows, that V(s) history stays bounded, and that it costs nothing
while it is off and little while it is on. Headless, like the other viewer tests."""

import math
import pickle
import statistics
import time

import numpy as np
import pygame
import pytest

from kartlab.env.kart_env import EnvConfig
from kartlab.render.palette import RAY_RAMP
from kartlab.show.brain import BrainPanel, ray_colors
from kartlab.show.drivers import VALUE_SECONDS, PolicyDriver, Version, load_version
from kartlab.show.live import LiveApp
from kartlab.show.viewer import ViewerApp
from kartlab.sim.showcase import showcase_track
from kartlab.train.config import PPOParams, TrainConfig


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from kartlab.train.train import train

    config = TrainConfig(
        name="brain-test",
        total_steps=768,
        n_envs=2,
        vec_env="dummy",
        low_priority=False,
        ppo=PPOParams(n_steps=64, batch_size=64, n_epochs=1),
        checkpoint_every=256,
        checkpoint_at=(0,),
        eval_tracks=(),
        env={"track": "harbor"},
    )
    run = train(config, runs_dir=tmp_path_factory.mktemp("runs"), verbose=0)
    final = run.list_checkpoints()[-1]
    run.best_model.write_bytes(final.model.read_bytes())
    return run


@pytest.fixture
def viewer(tiny_run):
    created = []

    def make(**kwargs):
        app = ViewerApp(tiny_run, scale=2, realtime=False, **kwargs)
        created.append(app)
        return app

    yield make
    for app in created:
        app.close()
    pygame.quit()


class CountingValue:
    """A value function that counts its calls (and says V = steer-free constant)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        return np.full(len(obs), 2.5)


def follow(obs):
    """Keeps the heading along the track at a gentle throttle: no restarts for a while."""
    sin_error = obs[:, EnvConfig().ray_count + 1]
    return np.stack([-2.0 * sin_error, np.full(len(obs), 0.5)], axis=1).astype(np.float32)


def counting_driver(value=None, track="harbor", policy=follow):
    value = value if value is not None else CountingValue()
    driver = PolicyDriver(
        Version(0, policy, value, True), showcase_track(track), EnvConfig(), (255, 0, 0)
    )
    return driver, value


# -- what the driver keeps -------------------------------------------------------------------


def test_the_value_history_is_bounded_to_about_ten_seconds(monkeypatch):
    monkeypatch.setattr("kartlab.show.drivers.RESTART_SECONDS", math.inf)  # no restart clears it
    driver, value = counting_driver()
    driver.set_tracking(True)
    for _ in range(60 * 25):  # 25 s of driving: more than the history holds
        driver.step()
    assert driver.restarts == 0
    assert len(driver.values) == driver.values.maxlen == round(VALUE_SECONDS * 60 / 4)
    assert value.calls == 60 * 25 // 4  # one per decision: every 4 physics steps


def test_no_value_is_computed_while_tracking_is_off():
    driver, value = counting_driver()
    for _ in range(300):
        driver.step()
    assert value.calls == 0 and not driver.values
    assert driver.last_obs is not None  # the observation is kept for free
    driver.set_tracking(True)
    for _ in range(40):
        driver.step()
    calls = value.calls
    assert calls > 0
    driver.set_tracking(False)
    for _ in range(40):
        driver.step()
    assert value.calls == calls and not driver.values  # stopping clears the history


def test_a_restart_clears_the_value_history():
    driver, _ = counting_driver()
    driver.set_tracking(True)
    for _ in range(60):
        driver.step()
    assert driver.values
    driver.restart()
    assert not driver.values


def test_a_version_without_a_value_function_still_drives():
    driver = PolicyDriver(Version(0, follow), showcase_track("harbor"), EnvConfig(), (1, 2, 3))
    driver.set_tracking(True)
    for _ in range(20):
        driver.step()
    assert not driver.values


def test_the_value_is_rescaled_to_reward_units_with_the_saved_statistics(tiny_run):
    import torch

    checkpoint = tiny_run.list_checkpoints()[-1]
    version = load_version(checkpoint)
    assert version.value_in_reward_units
    obs = np.linspace(-0.5, 0.5, 18, dtype=np.float32)[None]
    shown = float(version.value(obs)[0])

    from stable_baselines3 import PPO

    model = PPO.load(checkpoint.model, device="cpu")
    with open(checkpoint.vecnormalize, "rb") as f:
        stats = pickle.load(f)
    with torch.no_grad():
        raw = float(model.policy.predict_values(torch.as_tensor(obs)).item())
    assert math.isfinite(shown)
    assert shown == pytest.approx(raw * math.sqrt(stats.ret_rms.var + stats.epsilon), rel=1e-5)


# -- the panel ---------------------------------------------------------------------------------


def test_ray_colours_run_from_warm_when_near_to_cool_when_far():
    colors = ray_colors()
    assert len(RAY_RAMP) >= 5
    near, far = colors[0], colors[-1]
    assert near[0] > near[2] and far[2] > far[0]


def test_the_panel_draws_and_changes_the_pixels():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    driver, _ = counting_driver()
    driver.set_tracking(True)
    for _ in range(80):
        driver.step()
    panel = BrainPanel(18)
    screen.fill((40, 90, 40))
    before = pygame.surfarray.array3d(screen)
    panel.draw(screen, (10, 10), driver)
    after = pygame.surfarray.array3d(screen)
    w, h = panel.size
    changed = (before != after).any(axis=2)
    assert changed[10 : 10 + w, 10 : 10 + h].mean() > 0.5
    assert not changed[10 + w :, :].any()  # nothing outside the panel
    pygame.quit()


def test_the_panel_draws_before_the_first_decision_and_without_values():
    pygame.init()
    screen = pygame.display.set_mode((640, 360))
    driver = PolicyDriver(Version(0, follow), showcase_track("harbor"), EnvConfig(), (1, 2, 3))
    panel = BrainPanel(12)
    panel.draw(screen, (0, 0), driver)  # no observation yet
    driver.step()
    panel.draw(screen, (0, 0), driver)  # no value function
    pygame.quit()


def test_panel_placement_slides_in_from_the_edge_and_stays_in_the_free_space():
    pygame.init()
    panel = BrainPanel(18)
    area = pygame.Rect(0, 0, 1280, 720)
    panel.open(0.0)
    x0, _ = panel.position(area, "left", 100, 600, 0.0)
    x1, y1 = panel.position(area, "left", 100, 600, 1.0)
    assert x0 < 0 <= x1 and 100 <= y1 <= 600 - panel.size[1] + 100
    xr, _ = panel.position(area, "right", 100, 600, 1.0)
    assert xr + panel.size[0] <= area.right and xr > area.centerx
    pygame.quit()


@pytest.mark.parametrize("unit", [12, 18, 27])
def test_drawing_the_panel_costs_well_under_a_millisecond_or_so(unit):
    """Measured (median of many frames); the bound is generous so a busy CI machine passes.
    The target in the spec is 1 ms; a normal machine takes a fraction of it."""
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    driver, _ = counting_driver()
    driver.set_tracking(True)
    for _ in range(200):
        driver.step()
    panel = BrainPanel(unit)
    for _ in range(10):
        panel.draw(screen, (20, 100), driver)
    samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        panel.draw(screen, (20, 100), driver)
        samples.append((time.perf_counter() - t0) * 1000)
    assert statistics.median(samples) < 3.0
    pygame.quit()


# -- inside the viewers ----------------------------------------------------------------------


@pytest.mark.smoke
def test_the_panel_draws_in_the_exhibition_viewer_watch_mode(viewer):
    app = viewer(mode="watch", brain=True)
    app.run_loop(120)
    driver = app.scene.driver
    assert app.brain_panel is not None
    assert driver.track_value and len(driver.values) > 5
    assert "cerebro" in app.game.timings
    app.handle_key(pygame.K_b)
    assert not app.brain
    app.run_loop(5)
    assert not driver.track_value and not driver.values
    assert "cerebro" not in app.game.timings


def test_with_the_panel_off_the_exhibition_viewer_computes_no_values(viewer):
    app = viewer(mode="watch")
    app.run_loop(30)
    counter = CountingValue()
    driver = app.scene.driver
    original = driver.version.value
    driver.version.value = counter
    try:
        app.run_loop(120)
        assert counter.calls == 0 and not driver.track_value
        app.handle_key(pygame.K_b)  # on
        app.run_loop(120)
        on_calls = counter.calls
        assert on_calls >= 120 // 4 - 1
        app.handle_key(pygame.K_b)  # off again
        app.run_loop(120)
        assert counter.calls <= on_calls + 1
    finally:
        driver.version.value = original


def test_the_panel_follows_the_kart_the_camera_follows_in_the_race(viewer):
    app = viewer(mode="race")
    app.run_loop(100)
    scene = app.scene
    assert scene.brain_side == "right"
    app.handle_key(pygame.K_b)
    app.run_loop(240)  # past the countdown
    followed = scene.entries[scene.focus].driver
    assert scene.brain_driver() is followed and followed.track_value
    assert all(not e.driver.track_value for e in scene.entries if e.driver is not followed)
    app.handle_key(pygame.K_LEFT)  # the camera moves to another version...
    app.run_loop(5)
    assert scene.brain_driver() is not followed
    assert not followed.track_value and scene.brain_driver().track_value  # ...and so does V(s)


def test_the_panel_draws_in_progression_and_survives_a_resize(viewer):
    app = viewer(mode="progression", brain=True)
    app.run_loop(90)
    assert app.scene.driver is not None and app.scene.driver.track_value
    app.toggle_fullscreen()
    app.run_loop(5)
    assert app.brain_panel is not None


def test_the_title_screen_has_no_panel(viewer):
    app = viewer(brain=True)
    app.run_loop(30)
    assert app.state == "title" and app.brain_panel is None


@pytest.mark.smoke
def test_the_panel_draws_in_the_live_viewer_in_both_views(tiny_run):
    app = LiveApp(tiny_run, track_choice="harbor", ghosts=1, scale=2, realtime=False, brain=True)
    try:
        app.preload_ghosts()
        app.loader.poll_once()
        app.run(max_frames=90)
        assert app.brain_panel is not None
        assert app.current.track_value and len(app.current.values) > 5
        assert all(not g.track_value for g in app.ghosts)
        app.handle_key(pygame.K_t)  # top-down view
        app.run(max_frames=30)
        assert app.brain_panel is not None
        app.handle_key(pygame.K_b)
        app.run(max_frames=5)
        assert not app.brain and not app.current.track_value
    finally:
        pygame.quit()


def test_the_live_viewer_draws_the_rays_with_g(tiny_run):
    app = LiveApp(tiny_run, track_choice="harbor", ghosts=0, scale=2, realtime=False)
    try:
        app.loader.poll_once()
        app.run(max_frames=10)
        assert not app.game.world.show_rays
        app.handle_key(pygame.K_g)
        app.run(max_frames=2)
        assert app.rays and app.game.world.show_rays
    finally:
        pygame.quit()
