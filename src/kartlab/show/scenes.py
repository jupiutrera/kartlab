"""The scenes of the exhibition viewer: watch one version, ghost race, progression, and the
attract loop behind the title screen.

A scene owns the karts and what is drawn on top of the 3D view. The app (``viewer.py``)
owns the window, the 3D view, the clock, the keys that are the same everywhere (camera,
pause, slow motion...) and the version library. Scenes are advanced one physics step
(1/60 s) at a time and never block: a version that is still loading is waited for while
the frame keeps drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygame

from kartlab.render.palette import KART_RAMPS
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import ACCENT, KART_COLOR, format_time
from kartlab.render.view import DIM, WHITE, HudInfo
from kartlab.show.drive import drift_pressed, keyboard_controls
from kartlab.show.drivers import PolicyDriver, format_steps
from kartlab.show.humanrace import (
    RACE_TIMEOUT,
    Entrant,
    classify,
    difficulty_of,
    place_of,
    result_lines,
)
from kartlab.show.overlays import ease_in_out, version_name, visual_of
from kartlab.show.racehud import AMBER, GOLD, RaceHud
from kartlab.show.versions import BEST, pick_checkpoints
from kartlab.show.video import (
    COUNTDOWN_SECONDS,
    FADE_IN,
    FADE_OUT,
    FINAL_FACTOR,
    FINISH_SECONDS,
    GRID_FIRST,
    GRID_ROW,
    GRID_SIDE,
    MAX_RESTARTS,
)
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.contact import START_GRACE, Body, PairState, resolve
from kartlab.sim.drift import DriftState
from kartlab.sim.kart import DT, KartParams, KartState, cruise_speed
from kartlab.sim.race import Racer

if TYPE_CHECKING:
    from kartlab.show.viewer import ViewerApp

FPS = 60
RACE_MAX_SECONDS = 240.0
RACE_RESTART_SECONDS = 8.0  # after the winner: the next race starts by itself
SEGMENT_SECONDS = 30.0  # progression: longest a version drives (twice for the last one)
SEGMENT_END_SECONDS = FADE_OUT
CHASE_FROM = 0.15  # watch: versions past this share of the training get the chase camera
JUMP = 10  # versions skipped by Page Up / Page Down


def short_label(steps: int) -> str:
    return "0" if steps == 0 else format_steps(steps)


class Scene:
    """Base class: the hooks the app calls every frame."""

    name = ""
    brain_side = "left"  # the edge of the window the brain panel sits against

    def __init__(self, app: ViewerApp) -> None:
        self.app = app

    def brain_driver(self) -> PolicyDriver | None:
        """The driver whose brain the panel shows (the kart the camera follows)."""
        return None

    def enter(self) -> None:
        """The scene becomes the active one (the 3D view and the library are ready)."""

    def rebuild(self) -> None:
        """The track changed: start again on the new one."""

    def restart(self) -> None:
        """R: start again on the same track."""
        self.rebuild()

    def default_camera(self) -> str:
        return "broadcast"

    def on_escape(self) -> None:
        """Esc: back to the title screen, or out to the main menu when embedded."""
        if self.app.embedded:
            self.app.running = False
        else:
            self.app.go_title()

    def update(self) -> None:
        """Once per frame, before the physics: loading, swapping versions."""

    def physics_step(self) -> None:
        """One step of 1/60 s."""

    def frame(self) -> tuple[list[KartVisual], int, HudInfo] | None:
        """What to draw: karts, the index of the one the camera follows, and the HUD."""
        return None

    def draw_overlays(self) -> None:
        """Panels over the 3D view."""

    def loading_text(self) -> str | None:
        return None

    def on_key(self, key: int) -> bool:
        return False

    def on_wheel(self, delta: int) -> None:
        pass

    def on_click(self, pos: tuple[int, int]) -> None:
        pass

    # -- helpers -----------------------------------------------------------------------------

    def new_driver(self, version, ramp: int = 0) -> PolicyDriver:
        app = self.app
        driver = PolicyDriver(version, app.track, app.env_config, KART_COLOR)
        driver.ramp = ramp
        return driver

    def reset_cameras(self) -> None:
        for camera in self.app.game.cameras.values():
            camera.pose = None


# -- title background ---------------------------------------------------------------------------


class AttractScene(Scene):
    """The best model driving behind the title screen (a parked kart while it loads)."""

    name = "attract"

    def __init__(self, app: ViewerApp) -> None:
        super().__init__(app)
        self.driver: PolicyDriver | None = None

    def default_camera(self) -> str:
        return "chase"

    def enter(self) -> None:
        self.app.library.request(BEST)

    def rebuild(self) -> None:
        self.driver = None

    def update(self) -> None:
        if self.driver is None:
            version = self.app.library.request(BEST) or self.app.library.cached(BEST)
            if version is not None:
                self.driver = self.new_driver(version)
                self.reset_cameras()

    def physics_step(self) -> None:
        if self.driver is not None:
            self.driver.step()

    def frame(self):
        if self.driver is None:
            parked = KartState(*self.app.track.pose_at(0.0))
            return [KartVisual(parked, 0)], 0, HudInfo()
        return [visual_of(self.driver, 0)], 0, HudInfo()

    @property
    def version_label(self) -> str | None:
        version = self.app.library.cached(BEST)
        return version_name(version.steps) if version else None


# -- watch one version ---------------------------------------------------------------------------


class WatchScene(Scene):
    """One version driving. Left/right (or the wheel, or a click on the bar) change it."""

    name = "watch"

    def __init__(self, app: ViewerApp, start_steps: int | None = None) -> None:
        super().__init__(app)
        steps = app.library.steps
        if start_steps is None:
            self.index = len(steps) - 1
        else:
            self.index = min(range(len(steps)), key=lambda i: abs(steps[i] - start_steps))
        self.driver: PolicyDriver | None = None
        self.shown: int | None = None  # step count of the version at the wheel
        self._swapping = False
        self._bar = pygame.Rect(0, 0, 0, 0)

    @property
    def steps(self) -> list[int]:
        return self.app.library.steps

    def default_camera(self) -> str:
        steps = self.steps
        wanted = steps[self.index]
        return "chase" if wanted >= CHASE_FROM * steps[-1] and wanted > 0 else "broadcast"

    def rebuild(self) -> None:
        self.driver = None
        self.shown = None
        self._swapping = False

    def restart(self) -> None:
        if self.shown is not None:
            self._install(self.shown)

    def select(self, index: int) -> None:
        index = max(0, min(index, len(self.steps) - 1))
        if index == self.index:
            return
        self.index = index
        steps = self.steps
        keep = {steps[i] for i in (index - 1, index, index + 1) if 0 <= i < len(steps)}
        self.app.library.cancel_except(keep)

    def update(self) -> None:
        library = self.app.library
        steps = self.steps
        self.index = min(self.index, len(steps) - 1)
        want = steps[self.index]
        if self.shown == want or self._swapping:
            return
        reason = library.failed(want)
        if reason is not None:
            self.app.notify(f"No se pudo cargar {version_name(want)}")
            library.clear_failure(want)
            if self.shown is not None and self.shown in steps:
                self.index = steps.index(self.shown)
            return
        library.request(want)
        if library.cached(want) is None:
            return  # still loading: the old kart keeps driving
        if self.driver is None:
            self._install(want)
        else:
            self._swapping = True
            self.app.fade(self._swap)

    def _swap(self) -> None:
        self._swapping = False
        want = self.steps[self.index]
        if self.app.library.cached(want) is not None:
            self._install(want)

    def _install(self, steps: int) -> None:
        version = self.app.library.cached(steps)
        self.driver = self.new_driver(version)
        self.shown = steps
        self.reset_cameras()
        all_steps = self.steps
        i = all_steps.index(steps)
        for near in (i + 1, i - 1):  # stepping through versions should feel instant
            if 0 <= near < len(all_steps):
                self.app.library.request(all_steps[near])

    def loading_text(self) -> str | None:
        want = self.steps[self.index]
        if self.shown != want and self.app.library.cached(want) is None:
            return f"Cargando {version_name(want)}"
        return None

    def brain_driver(self) -> PolicyDriver | None:
        return self.driver

    def physics_step(self) -> None:
        if self.driver is not None:
            self.driver.step()

    def frame(self):
        driver = self.driver
        if driver is None:
            return None
        racer = driver.racer
        steps = self.steps
        label = "Sin entrenar" if self.shown == 0 else f"Versión {version_name(self.shown)}"
        info = HudInfo(
            laps=racer.laps,
            lap_time=racer.lap_time,
            best_lap=driver.best_lap,
            speed=racer.speed,
            label=label,
            extra=[self.app.track.name, f"Versión {self.index + 1} de {len(steps)}"],
        )
        return [visual_of(driver, 0)], 0, info

    # -- overlays ----------------------------------------------------------------------------

    def draw_overlays(self) -> None:
        app, ov = self.app, self.app.ov
        if self.shown is not None:
            ov.chart_panel(app.chart, app.metrics, self.shown)
        self._draw_bar()

    def _draw_bar(self) -> None:
        """Bottom strip with a tick per version, the chosen one marked and the cached ones
        lit: a map of where you are in the training."""
        app, ov = self.app, self.app.ov
        screen, u = app.screen, ov.unit
        steps = self.steps
        n = len(steps)
        width = min(round(u * 30), round(app.game.world_rect.width * 0.42))
        text = app.game.small.get(
            f"{self.index + 1} / {n} · {version_name(steps[self.index])}", WHITE
        )
        first = app.game.small.get(short_label(steps[0]), DIM)
        last = app.game.small.get(short_label(steps[-1]), DIM)
        pad = u // 2
        mark = u // 4  # how far the markers stick out above and below the ticks
        bar_h = max(u // 2, 6)
        height = pad + text.get_height() + mark + bar_h + mark + first.get_height() + pad // 2
        x = app.game.world_rect.centerx - width // 2
        y = app.game.world_rect.bottom - pad - height
        screen.blit(ov.panel(width + 2 * pad, height), (x - pad, y))
        screen.blit(text, (x + (width - text.get_width()) // 2, y + pad // 2))
        bar_y = y + pad // 2 + text.get_height() + mark
        self._bar = pygame.Rect(x, bar_y - mark, width, bar_h + 2 * mark)
        library = app.library
        span = max(n - 1, 1)
        for i in range(n):
            tx = x + round(i / span * (width - 2))
            lit = library.cached(steps[i]) is not None
            colour = (150, 155, 175) if lit else (70, 72, 90)
            pygame.draw.line(screen, colour, (tx, bar_y), (tx, bar_y + bar_h))
        if self.shown is not None and self.shown in steps:  # the version at the wheel
            sx = x + round(steps.index(self.shown) / span * (width - 2))
            pygame.draw.rect(screen, WHITE, (sx - 2, bar_y - mark, 5, bar_h + 2 * mark), 1)
        mx = x + round(self.index / span * (width - 2))  # the one chosen
        pygame.draw.rect(screen, ACCENT, (mx - 1, bar_y - mark, 3, bar_h + 2 * mark))
        label_y = bar_y + bar_h + mark
        screen.blit(first, (x, label_y))
        screen.blit(last, (x + width - last.get_width(), label_y))

    # -- input -------------------------------------------------------------------------------

    def on_key(self, key: int) -> bool:
        n = len(self.steps)
        moves = {
            pygame.K_LEFT: self.index - 1,
            pygame.K_RIGHT: self.index + 1,
            pygame.K_PAGEDOWN: self.index + JUMP,
            pygame.K_PAGEUP: self.index - JUMP,
            pygame.K_HOME: 0,
            pygame.K_END: n - 1,
        }
        if key in moves:
            self.select(moves[key])
            return True
        return False

    def on_wheel(self, delta: int) -> None:
        self.select(self.index + delta)

    def on_click(self, pos: tuple[int, int]) -> None:
        if self._bar.width and self._bar.collidepoint(pos):
            n = len(self.steps)
            fraction = (pos[0] - self._bar.x) / max(self._bar.width - 1, 1)
            self.select(round(fraction * (n - 1)))


# -- ghost race ----------------------------------------------------------------------------------


@dataclass
class Entry:
    driver: PolicyDriver
    ramp: int
    steps: int
    laps_done: int = 0


class RaceScene(Scene):
    """Every picked version at once on the same grid, without collisions. The camera follows
    one of them (the best at first; left/right change it). A new race starts by itself."""

    name = "race"
    brain_side = "right"  # the standings take the left edge

    def __init__(self, app: ViewerApp) -> None:
        super().__init__(app)
        self.checkpoints = pick_checkpoints(app.run)
        self.entries: list[Entry] = []
        self.focus = 0
        self.frame_no = 0
        self.end_frame: int | None = None
        self._restarting = False

    @property
    def laps(self) -> int:
        return self.app.laps

    def rebuild(self) -> None:
        self.entries = []
        self.end_frame = None
        self._restarting = False

    def enter(self) -> None:
        self.rebuild()

    def update(self) -> None:
        if self.entries:
            return
        library = self.app.library
        ready = []
        for checkpoint in self.checkpoints:
            library.request(checkpoint.steps)
            if library.cached(checkpoint.steps) is not None:
                ready.append(checkpoint.steps)
        pending = [
            c.steps
            for c in self.checkpoints
            if library.cached(c.steps) is None and library.failed(c.steps) is None
        ]
        if not pending and ready:
            self._start(ready)

    def loading_text(self) -> str | None:
        if self.entries:
            return None
        done = sum(self.app.library.cached(c.steps) is not None for c in self.checkpoints)
        return f"Cargando versiones {done}/{len(self.checkpoints)}"

    def _start(self, steps_list: list[int] | None = None) -> None:
        """Put every version on the grid, the worst on pole and the best last."""
        library = self.app.library
        if steps_list is None:
            steps_list = [e.steps for e in sorted(self.entries, key=lambda e: e.steps)]
        entries = []
        for rank, steps in enumerate(reversed(sorted(steps_list))):  # best first, ramp 0
            entries.append(Entry(self.new_driver(library.cached(steps), rank), rank, steps))
        for slot, entry in enumerate(reversed(entries)):
            side = -1.0 if slot % 2 == 0 else 1.0
            entry.driver.place(-GRID_FIRST - GRID_ROW * (slot // 2), side * GRID_SIDE)
        self.entries = entries
        self.focus = min(self.focus, len(entries) - 1)
        self.frame_no = 0
        self.end_frame = None
        self._restarting = False
        self.reset_cameras()

    def restart(self) -> None:
        if self.entries:
            self._start()

    @property
    def best(self) -> Entry:
        return self.entries[0]

    def brain_driver(self) -> PolicyDriver | None:
        return self.entries[self.focus].driver if self.entries else None

    def physics_step(self) -> None:
        if not self.entries:
            return
        start_frames = round(COUNTDOWN_SECONDS * FPS)
        if self.frame_no >= start_frames:
            for entry in self.entries:
                if entry.driver.step() == "lap":
                    entry.laps_done += 1
            limit = start_frames + round(RACE_MAX_SECONDS * FPS)
            if self.end_frame is None and (
                self.best.laps_done >= self.laps or self.frame_no + 1 >= limit
            ):
                self.end_frame = self.frame_no
        self.frame_no += 1
        if (
            self.end_frame is not None
            and not self._restarting
            and self.frame_no - self.end_frame >= RACE_RESTART_SECONDS * FPS
        ):
            self._restarting = True
            self.app.fade(self._start)

    def _order(self) -> list[Entry]:
        return sorted(self.entries, key=lambda e: -e.driver.racer.progress.distance)

    def frame(self):
        if not self.entries:
            return None
        # No labels under the countdown digit: the standings panel says who is who.
        racing = self.frame_no >= round(COUNTDOWN_SECONDS * FPS)
        label = short_label if racing else (lambda _steps: None)
        visuals = [
            visual_of(e.driver, e.ramp, label(e.steps), ghost=i != self.focus)
            for i, e in enumerate(self.entries)
        ]
        followed = self.entries[self.focus]
        racer = followed.driver.racer
        info = HudInfo(
            laps=min(followed.laps_done, self.laps - 1),
            lap_time=racer.lap_time,
            best_lap=followed.driver.best_lap,
            speed=racer.speed,
            label="Carrera de versiones",
            extra=[
                self.app.track.name,
                f"{self.laps} vueltas",
                f"Cámara en {version_name(followed.steps)}",
            ],
            ramp=followed.ramp,
        )
        return visuals, self.focus, info

    def draw_overlays(self) -> None:
        if not self.entries:
            return
        app, ov = self.app, self.app.ov
        order = self._order()
        rows = [
            (e.ramp, version_name(e.steps), f"vuelta {min(e.laps_done + 1, self.laps)}/{self.laps}")
            for e in order
        ]
        focus_row = order.index(self.entries[self.focus])
        ov.standings(rows, (app.game.world_rect.x + ov.unit // 2, ov.unit * 7), focus_row)
        ov.countdown(self.frame_no, round(COUNTDOWN_SECONDS * FPS), FPS)
        if self.end_frame is not None:
            age = (self.frame_no - self.end_frame) / FPS
            ov.card(
                "Meta",
                f"Ganador: {version_name(order[0].steps)}",
                [f"{app.track.name} · {self.laps} vueltas"],
                age=age,
                life=FINISH_SECONDS,
            )

    def on_key(self, key: int) -> bool:
        if key == pygame.K_LEFT:
            self.focus = (self.focus + 1) % max(len(self.entries), 1)  # towards slower versions
        elif key == pygame.K_RIGHT:
            self.focus = (self.focus - 1) % max(len(self.entries), 1)
        elif key == pygame.K_HOME:
            self.focus = len(self.entries) - 1
        elif key == pygame.K_END:
            self.focus = 0
        else:
            return False
        return True  # the camera eases towards the new kart instead of jumping

    def on_wheel(self, delta: int) -> None:
        if self.entries:
            self.focus = (self.focus - delta) % len(self.entries)


# -- progression ---------------------------------------------------------------------------------


class ProgressionScene(Scene):
    """The same track driven by versions on a log scale of steps, one after another, and
    round again at the end. Left/right skip to the previous or next one."""

    name = "progression"

    def __init__(self, app: ViewerApp) -> None:
        super().__init__(app)
        self.checkpoints = pick_checkpoints(app.run)
        self.names = [short_label(c.steps) for c in self.checkpoints]
        self.index = 0
        self.previous = 0
        self.driver: PolicyDriver | None = None
        self.frame_no = 0
        self.end_frame: int | None = None

    @property
    def steps(self) -> int:
        return self.checkpoints[self.index].steps

    def brain_driver(self) -> PolicyDriver | None:
        return self.driver

    def rebuild(self) -> None:
        self.driver = None
        self.frame_no = 0
        self.end_frame = None

    def enter(self) -> None:
        self.previous = 0
        self.index = 0
        self.rebuild()

    def goto(self, index: int, settle: bool = True) -> None:
        index %= len(self.checkpoints)
        self.previous = self.steps
        self.index = index
        self.rebuild()
        if settle:
            self.update()  # a cached version is on the road in this very frame

    def restart(self) -> None:
        self.previous = 0
        self.rebuild()

    def update(self) -> None:
        library = self.app.library
        library.request(self.steps)
        if self.driver is None:
            version = library.cached(self.steps)
            if version is None:
                if library.failed(self.steps) is not None:
                    library.clear_failure(self.steps)
                    self.goto(self.index + 1, settle=False)  # skip a version that does not load
                return
            self.driver = self.new_driver(version)
            self.frame_no = 0
            self.end_frame = None
            self.reset_cameras()
            following = (self.index + 1) % len(self.checkpoints)
            library.request(self.checkpoints[following].steps)

    def loading_text(self) -> str | None:
        if self.driver is None:
            return f"Cargando {version_name(self.steps)}"
        return None

    def physics_step(self) -> None:
        driver = self.driver
        if driver is None:
            return
        event = driver.step()
        last = self.index == len(self.checkpoints) - 1
        limit = round(SEGMENT_SECONDS * (FINAL_FACTOR if last else 1.0) * FPS)
        if self.end_frame is None and (
            event == "lap" or self.frame_no + 1 >= limit or driver.restarts >= MAX_RESTARTS
        ):
            self.end_frame = self.frame_no + round(SEGMENT_END_SECONDS * FPS)
        if self.end_frame is not None and self.frame_no >= self.end_frame:
            if last:
                self.app.notify("Vuelta a empezar: de nuevo sin entrenar")
                self.previous = 0
                self.index = 0
                self.rebuild()
            else:
                self.goto(self.index + 1)
            return
        self.frame_no += 1

    def frame(self):
        driver = self.driver
        if driver is None:
            return None
        racer = driver.racer
        info = HudInfo(
            laps=racer.laps,
            lap_time=racer.lap_time,
            best_lap=driver.best_lap,
            speed=racer.speed,
            label="Versión sin entrenar"
            if self.steps == 0
            else f"Versión {version_name(self.steps)}",
            extra=[self.app.track.name, f"{self.index + 1} de {len(self.checkpoints)}"],
        )
        return [visual_of(driver, 0)], 0, info

    def draw_overlays(self) -> None:
        if self.driver is None:
            return
        app, ov = self.app, self.app.ov
        steps = self.steps
        shown = round(
            self.previous + (steps - self.previous) * ease_in_out(self.frame_no / (1.4 * FPS))
        )
        ov.counter(shown, "pasos de entrenamiento")
        ov.chart_panel(app.chart, app.metrics, steps)
        ov.timeline(self.names, self.index)
        ov.card(
            "Versión",
            version_name(steps),
            [f"{app.track.name} · {app.run.root.name}"],
            age=self.frame_no / FPS,
        )
        amount = 1.0 - ease_in_out(self.frame_no / (FADE_IN * FPS))
        if self.end_frame is not None:
            left = (self.end_frame - self.frame_no) / (FADE_OUT * FPS)
            amount = max(amount, ease_in_out(1.0 - left))
        ov.dim(amount)

    def on_key(self, key: int) -> bool:
        if key == pygame.K_RIGHT:
            self.goto(self.index + 1)
        elif key == pygame.K_LEFT:
            self.goto(self.index - 1)
        elif key == pygame.K_HOME:
            self.goto(0)
        elif key == pygame.K_END:
            self.goto(len(self.checkpoints) - 1)
        else:
            return False
        return True

    def on_wheel(self, delta: int) -> None:
        self.goto(self.index + delta)


# -- human against the AI --------------------------------------------------------------------------

BANNER_SECONDS = 2.5
RESULTS_DELAY = 2.2  # after the finish line: the "¡META!" banner first, then the table
RESULTS_DELAY_LOST = 0.6  # when the race ended without you finishing


def grid_slot(slot: int) -> tuple[float, float]:
    """Arc position (behind the line, so negative) and lateral offset of a grid slot, two
    karts per row, pole first."""
    return -GRID_FIRST - GRID_ROW * (slot // 2), (-1.0 if slot % 2 == 0 else 1.0) * GRID_SIDE


class HumanRaceScene(Scene):
    """You against AI karts on the same grid: a 3-2-1 countdown with the controls frozen,
    a number of laps, soft bumps between karts (the AI does not see them and recovers on its
    own) and a results table. The AI is the best checkpoint of the run, slowed down by a
    speed limiter per difficulty. Esc leaves; Enter after the results races again."""

    name = "vsai"
    brain_side = "right"  # the standings take the left edge

    def __init__(self, app: ViewerApp) -> None:
        super().__init__(app)
        self.laps = app.vs_laps
        self.rivals = app.rivals
        self.difficulty = difficulty_of(app.difficulty)
        self.collisions = app.collisions
        self.kart_ramp = app.kart_ramp
        self.hud: RaceHud | None = None
        self.drift = DriftState()
        self.autopilot_on = False
        self.controls = (0.0, 0.0)
        self.pilot: Autopilot | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.started = False
        self.human: Racer | None = None
        self.origin = 0.0
        self.drivers: list[PolicyDriver] = []
        self.entrants: list[Entrant] = []
        self.pairs: list[PairState] = []
        self.frame_no = 0
        self.race_steps = 0
        self.place: int | None = None
        self.place_since = -10.0  # race time of the last change of place (s)
        self.banner: tuple[str, float, tuple] | None = None  # text, race time, look
        self.ended_at: float | None = None  # race time at which the results were fixed
        self.final: list[tuple[str, bool]] = []
        self.final_place = 0
        self.drift.reset()

    # -- life cycle --------------------------------------------------------------------------

    def default_camera(self) -> str:
        return "chase"

    def brain_driver(self) -> PolicyDriver | None:
        return self.drivers[0] if self.drivers else None

    def on_escape(self) -> None:
        self.app.running = False

    def enter(self) -> None:
        self.app.library.request(BEST)

    def rebuild(self) -> None:
        self.pilot = None
        self._reset_state()

    def restart(self) -> None:
        self._reset_state()  # update() builds the grid again as soon as the AI is loaded

    def update(self) -> None:
        if self.started:
            return
        library = self.app.library
        version = library.request(BEST) or library.cached(BEST)
        if version is None:
            if library.failed(BEST) is not None:
                self.app.notify("No se pudo cargar la IA")
                library.clear_failure(BEST)
            return
        self._start(version)

    def loading_text(self) -> str | None:
        return None if self.started else "Cargando la IA"

    def _start(self, version) -> None:
        app, track = self.app, self.app.track
        total = 1 + self.rivals
        ramps = [r for r in range(len(KART_RAMPS)) if r != self.kart_ramp][: self.rivals]
        params = KartParams()
        cap = None if self.difficulty.speed >= 1.0 else self.difficulty.speed * cruise_speed(params)
        self.entrants = []
        self.drivers = []
        self.pairs = []
        for k in range(self.rivals):
            driver = self.new_driver(version, ramps[k])
            driver.restart_in_place = True
            driver.speed_cap = cap
            s, lateral = grid_slot(k)
            driver.place(s, lateral)
            self.drivers.append(driver)
            self.pairs.append(PairState())
            self.entrants.append(Entrant(f"IA {k + 1}", ramps[k], False, track.length, self.laps))
        s, lateral = grid_slot(total - 1)  # the human starts on the last row
        self.human = Racer(track, params, s, lateral)
        self.origin = -s
        self.entrants.insert(0, Entrant("Tú", self.kart_ramp, True, track.length, self.laps))
        self.entrants[0].distance = -self.origin
        for entrant, driver in zip(self.entrants[1:], self.drivers, strict=True):
            entrant.distance = driver.race_distance
        self.pilot = Autopilot(track, params)
        self.drift.reset()
        self.frame_no = 0
        self.race_steps = 0
        self.place = place_of(self.entrants, self.entrants[0])
        self.started = True
        self.reset_cameras()
        if self.hud is None:
            self.hud = RaceHud(app.ov)

    # -- physics -----------------------------------------------------------------------------

    @property
    def start_frames(self) -> int:
        return round(COUNTDOWN_SECONDS * FPS)

    @property
    def race_time(self) -> float:
        return self.race_steps * DT

    @property
    def me(self) -> Entrant:
        return self.entrants[0]

    def physics_step(self) -> None:
        if not self.started:
            return
        if self.frame_no < self.start_frames:  # the countdown: everybody is frozen
            self.frame_no += 1
            return
        self.frame_no += 1
        self.race_steps += 1
        t = self.race_time
        racer = self.human
        manual = not self.autopilot_on and not self.me.finished
        if manual:
            pressed = pygame.key.get_pressed()
            steer, throttle = keyboard_controls(pressed)
            self.drift.hold = drift_pressed(pressed)
            drift = self.drift
        else:
            steer, throttle = self.pilot.act(racer)
            drift = None
        self.controls = (steer, throttle)
        racer.step(steer, throttle, drift=drift)
        for driver in self.drivers:
            driver.step()
        if self.collisions and not self.me.finished and self.ended_at is None:
            self._bump(t)
        if self.ended_at is None:
            self._track(t)

    def _bump(self, t: float) -> None:
        """Soft contact between the human kart and each AI kart (see ``sim/contact.py``)."""
        racer = self.human
        height = getattr(self.drift, "height", 0.0)
        mine = Body(racer.kart, racer.level_z, height, START_GRACE)
        for driver, state in zip(self.drivers, self.pairs, strict=True):
            other = driver.racer
            grace = max(START_GRACE, driver.no_contact_until)
            contact = resolve(mine, Body(other.kart, other.level_z, 0.0, grace), state, t)
            if contact is None:
                continue
            racer.displace(*contact.push_human)
            other.displace(*contact.push_ai)
            racer.kart.vx += contact.dv_human[0]
            racer.kart.vy += contact.dv_human[1]
            other.kart.vx += contact.dv_ai[0]
            other.kart.vy += contact.dv_ai[1]
            driver.touch()

    def _track(self, t: float) -> None:
        """Laps, finish, place and the end of the race."""
        me = self.me
        crossed = me.update(self.human.progress.distance - self.origin, t)
        for entrant, driver in zip(self.entrants[1:], self.drivers, strict=True):
            entrant.update(driver.race_distance, t)
        if crossed:
            if me.finished:
                self.banner = ("¡META!", t, GOLD)
            elif me.laps_done == self.laps - 1:
                self.banner = ("ÚLTIMA VUELTA", t, AMBER)
        place = place_of(self.entrants, me)
        if place != self.place:
            self.place, self.place_since = place, t
        firsts = [e.finish_time for e in self.entrants if e.finished]
        timed_out = bool(firsts) and t - min(firsts) >= RACE_TIMEOUT
        if me.finished or timed_out:
            self.ended_at = t
            self.final = result_lines(self.entrants)
            self.final_place = place_of(self.entrants, me)

    # -- drawing -----------------------------------------------------------------------------

    def frame(self):
        if not self.started:
            return None
        racer = self.human
        manual = not self.autopilot_on and not self.me.finished
        human = KartVisual(
            racer.kart,
            self.kart_ramp,
            on_grass=racer.on_grass,
            throttle=self.controls[1],
            drift=self.drift if manual else None,
        )
        racing = self.frame_no >= self.start_frames
        visuals = [human]
        for entrant, driver in zip(self.entrants[1:], self.drivers, strict=True):
            visuals.append(visual_of(driver, entrant.ramp, entrant.name if racing else None))
        t = self.race_time
        info = HudInfo(
            laps=min(self.me.laps_done, self.laps - 1),
            lap_time=self.me.lap_time_at(t),
            best_lap=self.me.best_lap,
            speed=racer.speed,
            label="Carrera contra la IA" if manual else "Piloto automático",
            ramp=self.kart_ramp,
            extra=[
                self.app.track.name,
                f"{self.laps} vueltas · {self.difficulty.label}",
                "Choques activados" if self.collisions else "Sin choques",
            ],
            drift=self.drift if manual else None,
        )
        return visuals, 0, info

    def draw_overlays(self) -> None:
        if not self.started or self.hud is None:
            return
        app, ov, hud = self.app, self.app.ov, self.hud
        t = self.race_time
        order = classify(self.entrants)
        rows = [
            (
                e.ramp,
                e.name,
                "meta" if e.finished else f"vuelta {min(e.laps_done + 1, self.laps)}/{self.laps}",
            )
            for e in order
        ]
        ov.standings(
            rows, (app.game.world_rect.x + ov.unit // 2, ov.unit * 7), order.index(self.me)
        )
        hud.position(self.place, len(self.entrants), t - self.place_since)
        hud.lap_counter(
            min(self.me.laps_done + 1, self.laps),
            self.laps,
            format_time(self.ended_at if self.ended_at is not None else t),
        )
        hud.countdown(self.frame_no, self.start_frames, FPS)
        if self.banner is not None:
            text, since, look = self.banner
            hud.banner(text, t - since, BANNER_SECONDS, look)
        if self.ended_at is not None:
            delay = RESULTS_DELAY if self.me.finished else RESULTS_DELAY_LOST
            lines = [(text, (255, 214, 120) if human else WHITE) for text, human in self.final]
            hud.results(
                "Resultados",
                self.final_place,
                lines,
                "Intro: repetir  ·  Esc: salir",
                t - self.ended_at - delay,
            )

    # -- input -------------------------------------------------------------------------------

    def on_key(self, key: int) -> bool:
        if key == pygame.K_s:
            return True  # S brakes here; slow motion is for the other modes
        if key == pygame.K_TAB:
            self.autopilot_on = not self.autopilot_on
            self.drift.reset()
            return True
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.ended_at is not None:
            self.restart()
            self.reset_cameras()
            return True
        return False


SCENES: dict[str, type[Scene]] = {
    "watch": WatchScene,
    "race": RaceScene,
    "progression": ProgressionScene,
    "vsai": HumanRaceScene,
}
