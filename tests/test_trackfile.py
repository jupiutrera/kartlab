from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kartlab.env.kart_env import EnvConfig, KartEnv
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.kart import KartParams
from kartlab.sim.race import Racer
from kartlab.sim.showcase import SHOWCASE, showcase_track
from kartlab.sim.track import (
    Track,
    TrackParams,
    filleted_polygon,
    finish_centerline,
    generate_track,
)
from kartlab.sim.trackcli import main as cli_main
from kartlab.sim.trackfile import (
    CLEARANCE,
    FORMAT_VERSION,
    MAX_LENGTH,
    MIN_RADIUS,
    MIN_WIDTH,
    TrackFileError,
    TrackSpec,
    build_track,
    check_centerline,
    load_spec,
    load_track_file,
    save_spec,
    spec_from_dict,
    spec_from_seed,
    spec_from_showcase,
    spec_to_dict,
    validate_spec,
)
from kartlab.sim.tracksets import EVAL_SEEDS, eval_track, resolve_track
from kartlab.train.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "tracks" / "ejemplo.json"

# A valid design with a bay: 1.27 km, 12 m wide.
GOOD = TrackSpec(
    "Bahía de prueba",
    12.0,
    (
        (0.0, 0.0, 40.0),
        (320.0, -10.0, 30.0),
        (400.0, 100.0, 25.0),
        (330.0, 190.0, 16.0),
        (220.0, 150.0, 28.0),
        (190.0, 260.0, 20.0),
        (60.0, 290.0, 30.0),
        (-70.0, 200.0, 24.0),
        (-40.0, 80.0, 40.0),
    ),
    decoration_seed=7,
)


# A rectangle with a tongue hanging from the top edge. The tongue tip is `tip` metres
# above the bottom edge: close enough, the asphalt of two sections nearly touches.
def tongue(tip: float) -> TrackSpec:
    corners = [
        (0, 0, 30),
        (400, 0, 30),
        (400, 300, 30),
        (220, 300, 20),
        (220, tip, 18),
        (180, tip, 18),
        (180, 300, 20),
        (0, 300, 30),
    ]
    return TrackSpec("Lengua", 12.0, tuple((float(x), float(y), float(r)) for x, y, r in corners))


# Radii of 14 m or more at every corner, yet the smoothed centerline bends tighter
# than 14 m (generator layout of seed 22 at maximum curviness).
TIGHT = TrackSpec(
    "Apretado",
    12.0,
    (
        (70.0, -15.0, 20.8),
        (110.3, 60.0, 21.3),
        (42.2, 118.9, 18.2),
        (5.2, 97.2, 14.6),
        (-45.3, 118.8, 23.5),
        (-109.8, 125.0, 26.7),
        (-84.6, 21.7, 23.9),
        (-67.0, -6.6, 18.3),
        (-40.3, -61.2, 14.4),
        (-26.1, -145.2, 15.3),
        (1.9, -176.0, 20.5),
        (52.2, -119.1, 22.6),
        (109.0, -108.7, 14.1),
    ),
)


def problems(spec: TrackSpec) -> str:
    report = validate_spec(spec)
    assert not report.ok
    return "\n".join(report.problems)


# -- format and round trip -------------------------------------------------------------


def test_round_trip_keeps_name_width_corners_and_geometry(tmp_path):
    path = tmp_path / "t.json"
    save_spec(path, GOOD)
    loaded = load_spec(path)
    assert loaded == GOOD
    a, b = build_track(GOOD), load_track_file(path)
    np.testing.assert_array_equal(a.points, b.points)
    assert (b.name, b.width, b.decoration_seed) == ("Bahía de prueba", 12.0, 7)


def test_round_trip_of_fractional_values_is_exact(tmp_path):
    spec = replace(GOOD, width=12.375, corners=((0.1, 0.2, 40.3), *GOOD.corners[1:]))
    path = tmp_path / "t.json"
    save_spec(path, spec)
    assert load_spec(path) == spec


def test_file_is_utf8_with_lf_endings_and_plain_json(tmp_path):
    path = tmp_path / "t.json"
    save_spec(path, GOOD)
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert "Bahía de prueba" in raw.decode("utf-8")  # real UTF-8, no \u escapes
    assert json.loads(raw) == spec_to_dict(GOOD)
    assert json.loads(raw)["format"] == FORMAT_VERSION


def test_save_is_atomic_and_overwrites(tmp_path):
    path = tmp_path / "sub" / "t.json"
    save_spec(path, GOOD)
    save_spec(path, replace(GOOD, name="Otro"))
    assert load_spec(path).name == "Otro"
    assert [p.name for p in path.parent.iterdir()] == ["t.json"]  # no temporary left


def test_track_is_built_through_the_generators_path():
    for key, design in SHOWCASE.items():
        built = build_track(spec_from_showcase(key))
        np.testing.assert_array_equal(built.points, showcase_track(key).points)
        assert built.name == design.name
    corners = np.array(GOOD.corners)
    path, _ = filleted_polygon(corners[:, :2], corners[:, 2])
    expected = finish_centerline(path, GOOD.width, MIN_RADIUS, CLEARANCE)
    np.testing.assert_array_equal(build_track(GOOD).points, expected)
    assert build_track(GOOD).clearance == CLEARANCE


def test_limits_match_the_generator_and_the_kart():
    params = TrackParams()
    assert params.min_radius == MIN_RADIUS
    assert params.clearance == CLEARANCE
    assert KartParams().max_curvature >= 1.0 / MIN_RADIUS


# -- reserved and unknown fields, structure --------------------------------------------


def good_dict() -> dict:
    return spec_to_dict(GOOD)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"crossings": []}, "reservado"),
        ({"surprise": 1}, "desconocidos"),
        ({"format": 2}, "no soportado"),
        ({"format": "1"}, "'format'"),
        ({"name": ""}, "'name'"),
        ({"name": "x" * 200}, "'name'"),
        ({"name": "a\nb"}, "'name'"),
        ({"width": "12"}, "'width'"),
        ({"width": True}, "'width'"),
        ({"width": float("inf")}, "'width'"),
        ({"decoration_seed": -1}, "'decoration_seed'"),
        ({"decoration_seed": 1.5}, "'decoration_seed'"),
        ({"corners": "nope"}, "'corners'"),
        ({"corners": [[0, 0]]}, "esquina 1"),
        ({"corners": [[0, 0, "r"]]}, "esquina 1"),
        ({"corners": [[0, 0, None]]}, "esquina 1"),
    ],
)
def test_structural_errors_are_reported(change, message):
    data = {**good_dict(), **change}
    with pytest.raises(TrackFileError, match=message):
        spec_from_dict(data)


@pytest.mark.parametrize("missing", ["format", "name", "width", "corners"])
def test_missing_required_fields_are_reported(missing):
    data = good_dict()
    del data[missing]
    with pytest.raises(TrackFileError, match=missing):
        spec_from_dict(data)


def test_decoration_seed_is_optional():
    data = good_dict()
    del data["decoration_seed"]
    assert spec_from_dict(data).decoration_seed == 0


def test_crossings_are_reserved_for_a_future_version():
    with pytest.raises(TrackFileError) as error:
        spec_from_dict({**good_dict(), "crossings": [[0, 1]]})
    assert "'crossings'" in str(error.value) and "futura" in str(error.value)


def test_a_json_array_is_not_a_track():
    with pytest.raises(TrackFileError, match="objeto JSON"):
        spec_from_dict([1, 2, 3])


# -- corrupt files ---------------------------------------------------------------------


def test_truncated_file_gives_a_clear_error(tmp_path):
    path = tmp_path / "t.json"
    save_spec(path, GOOD)
    path.write_bytes(path.read_bytes()[:120])
    with pytest.raises(TrackFileError, match="incompleto") as error:
        load_spec(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "JSON"),
        (b"{not json", "JSON"),
        (b"\xff\xfe\x00garbage", "UTF-8"),
        (b'{"format": NaN}', "JSON"),
        (b"[1, 2]", "objeto"),
        (b"null", "objeto"),
    ],
)
def test_corrupt_files_give_clear_errors(tmp_path, content, message):
    path = tmp_path / "t.json"
    path.write_bytes(content)
    with pytest.raises(TrackFileError, match=message):
        load_track_file(path)


def test_missing_file_and_directory_give_clear_errors(tmp_path):
    with pytest.raises(TrackFileError, match="no existe"):
        load_track_file(tmp_path / "nope.json")
    with pytest.raises(TrackFileError):
        load_track_file(tmp_path)


def test_oversized_file_is_refused(tmp_path):
    path = tmp_path / "big.json"
    path.write_bytes(b" " * 1_500_000)
    with pytest.raises(TrackFileError, match="demasiado grande"):
        load_spec(path)


# -- validator rules -------------------------------------------------------------------


def test_a_good_design_is_valid():
    report = validate_spec(GOOD)
    assert report.ok
    assert 400 < report.length < 2500
    assert report.min_radius >= MIN_RADIUS
    assert report.gap >= CLEARANCE


@pytest.mark.parametrize("width", [MIN_WIDTH - 0.5, 0.0, -3.0, 21.0, 60.0])
def test_width_out_of_range_is_rejected(width):
    assert "anchura" in problems(replace(GOOD, width=width))


def test_minimum_width_is_accepted():
    assert validate_spec(replace(GOOD, width=MIN_WIDTH)).ok


def test_corner_radius_below_the_kart_limit_is_rejected():
    corners = list(GOOD.corners)
    corners[2] = (400.0, 100.0, MIN_RADIUS - 1.0)
    assert "esquina 3: radio" in problems(replace(GOOD, corners=tuple(corners)))


def test_radius_that_does_not_fit_between_neighbours_is_rejected():
    # A 90 degree corner on a 25 m edge cannot hold a 20 m arc and stays below the minimum.
    corners = [(0.0, 0.0, 30.0), (200.0, 0.0, 30.0), (200.0, 25.0, 20.0), (0.0, 25.0, 30.0)]
    assert "no cabe" in problems(TrackSpec("Corto", 12.0, tuple(corners)))


def test_a_large_radius_is_shrunk_to_fit_like_in_the_generator():
    corners = list(GOOD.corners)
    corners[0] = (
        0.0,
        0.0,
        400.0,
    )  # asks for more than fits: the arc shrinks, the track stays valid
    assert validate_spec(replace(GOOD, corners=tuple(corners))).ok


def test_self_intersecting_polygon_is_rejected():
    bowtie = TrackSpec(
        "Lazo",
        12.0,
        ((0.0, 0.0, 20.0), (300.0, 300.0, 20.0), (300.0, 0.0, 20.0), (0.0, 300.0, 20.0)),
    )
    assert "se cruza a sí mismo" in problems(bowtie)


def test_figure_eight_is_rejected_until_crossings_exist():
    eight = TrackSpec(
        "Ocho",
        12.0,
        (
            (-300.0, -150.0, 30.0),
            (-300.0, 150.0, 30.0),
            (300.0, -150.0, 30.0),
            (300.0, 150.0, 30.0),
        ),
    )
    assert "se cruza a sí mismo" in problems(eight)


def test_corners_too_close_are_rejected():
    corners = list(GOOD.corners)
    corners.insert(1, (3.0, 1.0, 20.0))
    assert "separa más" in problems(replace(GOOD, corners=tuple(corners)))


def test_duplicate_corner_is_rejected_without_crashing():
    corners = list(GOOD.corners)
    corners.insert(1, corners[0])
    assert "separa más" in problems(replace(GOOD, corners=tuple(corners)))


def test_too_few_or_too_many_corners_are_rejected():
    assert "esquinas" in problems(replace(GOOD, corners=GOOD.corners[:2]))
    assert "esquinas" in problems(replace(GOOD, corners=()))
    ring = tuple(
        (float(500 * np.cos(t)), float(500 * np.sin(t)), 30.0)
        for t in np.linspace(0, 2 * np.pi, 60, endpoint=False)
    )
    assert "esquinas" in problems(replace(GOOD, corners=ring))


def test_coordinates_out_of_range_are_rejected():
    corners = list(GOOD.corners)
    corners[0] = (1e6, 0.0, 40.0)
    assert "se sale" in problems(replace(GOOD, corners=tuple(corners)))


def test_too_short_track_is_rejected():
    square = TrackSpec(
        "Cuadrado",
        12.0,
        ((0.0, 0.0, 20.0), (100.0, 0.0, 20.0), (100.0, 100.0, 20.0), (0.0, 100.0, 20.0)),
    )
    assert "longitud" in problems(square)


def test_too_long_track_is_rejected():
    big = TrackSpec(
        "Enorme",
        12.0,
        ((0.0, 0.0, 40.0), (1500.0, 0.0, 40.0), (1500.0, 1500.0, 40.0), (0.0, 1500.0, 40.0)),
    )
    assert "longitud" in problems(big)
    assert MAX_LENGTH < 6000


def test_two_sections_too_close_are_rejected_with_the_gap():
    message = problems(tongue(15.0))
    assert "demasiado cerca" in message and "hierba" in message
    assert validate_spec(tongue(60.0)).ok


def test_tight_centerline_is_rejected():
    assert all(c[2] >= MIN_RADIUS for c in TIGHT.corners)
    assert "curva demasiado cerrada" in problems(TIGHT)


def test_check_centerline_rejects_tight_curves_and_crossings():
    s = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    circle = 12.7 * np.stack([np.cos(s), np.sin(s)], 1)  # radius 12.7 m, 1 m spacing
    assert "curva demasiado cerrada" in "\n".join(check_centerline(circle, 8.0).problems)

    s = 0.01 + np.linspace(0, 2 * np.pi, 600, endpoint=False)
    eight = np.stack([120 * np.sin(s), 60 * np.sin(2 * s)], 1)  # crosses itself in the middle
    assert "se cruza" in "\n".join(check_centerline(eight, 12.0).problems)


def test_build_track_refuses_invalid_specs():
    with pytest.raises(TrackFileError, match="demasiado cerca"):
        build_track(tongue(15.0))
    with pytest.raises(TrackFileError, match="anchura"):
        build_track(replace(GOOD, width=2.0))


def test_all_problems_are_listed_not_only_the_first():
    spec = replace(
        GOOD, width=2.0, corners=GOOD.corners[:1] + ((1.0, 1.0, 5.0),) + GOOD.corners[2:]
    )
    text = problems(spec)
    assert "anchura" in text and "radio" in text and "separa más" in text


# -- generated, showcase and shipped tracks pass ----------------------------------------


def passes(track: Track) -> bool:
    report = check_centerline(track.points, track.width)
    assert report.ok, report.problems
    return True


@pytest.mark.parametrize("seed", range(12))
def test_generated_training_tracks_pass_the_validator(seed):
    track = generate_track(seed)
    assert passes(track)
    assert validate_spec(spec_from_seed(seed)).ok


@pytest.mark.parametrize("index", range(20))
def test_evaluation_tracks_pass_the_validator(index):
    assert passes(eval_track(index))


@pytest.mark.parametrize("key", sorted(SHOWCASE))
def test_showcase_tracks_pass_the_validator(key):
    assert passes(showcase_track(key))
    assert validate_spec(spec_from_showcase(key)).ok


@pytest.mark.parametrize(
    "params",
    [
        TrackParams(width=10.0, length=700.0, curviness=0.8),
        TrackParams(width=14.0, length=1200.0, curviness=0.2),
    ],
)
def test_training_range_tracks_pass_the_validator(params):
    for seed in range(6):
        assert passes(generate_track(seed, params))


@pytest.mark.parametrize("seed", [0, 3, 8, 11])
def test_exported_seed_rebuilds_close_to_the_generated_track(seed):
    generated = generate_track(seed)
    rebuilt = build_track(spec_from_seed(seed))
    assert rebuilt.length == pytest.approx(generated.length, rel=1e-3)
    d = np.linalg.norm(rebuilt.points[:, None, :] - generated.points[None, ::3, :], axis=2)
    assert d.min(axis=0).max() < 0.5  # every generated sample lies on the rebuilt line


def test_exported_seed_is_deterministic():
    assert spec_from_seed(5) == spec_from_seed(5)
    assert spec_from_seed(5) != spec_from_seed(6)


def test_shipped_example_is_valid_and_drivable():
    track = load_track_file(EXAMPLE)
    assert track.name == "Bahía del Faro"
    racer, pilot = Racer(track), Autopilot(track)
    widest = 0.0
    while racer.laps < 1 and racer.time < 200.0:
        racer.step(*pilot.act(racer))
        widest = max(widest, abs(racer.position.lateral))
    assert racer.laps == 1
    assert widest < track.half_width - 1.0


def test_nothing_in_the_tracks_folder_is_invalid():
    files = sorted((ROOT / "tracks").glob("*.json"))
    assert files
    for path in files:
        assert validate_spec(load_spec(path)).ok, path


# -- loading from the rest of the code --------------------------------------------------


def test_resolve_track_loads_files_and_refuses_invalid_ones(tmp_path):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    save_spec(good, GOOD)
    save_spec(bad, replace(GOOD, width=2.0))
    assert resolve_track(str(good)).name == "Bahía de prueba"
    with pytest.raises(TrackFileError):
        resolve_track(str(bad))
    assert resolve_track("harbor").name == "Puerto"  # the other choices are untouched


def test_the_environment_never_gets_an_invalid_track(tmp_path):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    save_spec(good, GOOD)
    save_spec(bad, tongue(15.0))
    env = KartEnv(EnvConfig(track=str(good)))
    env.reset(seed=0)
    assert env.track.name == "Bahía de prueba"
    with pytest.raises(TrackFileError):
        KartEnv(EnvConfig(track=str(bad)))


def test_evaluate_runs_on_a_track_file(tmp_path):
    path = tmp_path / "good.json"
    save_spec(path, GOOD)

    def predict(obs):  # drives straight: it leaves the track, which ends the episode
        return np.tile([0.0, 1.0], (len(obs), 1)).astype(np.float32)

    result = evaluate(predict, [str(path)], EnvConfig(), laps=1)
    assert result.tracks[0].track == str(path)
    assert result.tracks[0].end in {"off_track", "stuck", "time", "laps"}


def test_evaluate_refuses_an_invalid_track_file(tmp_path):
    path = tmp_path / "bad.json"
    save_spec(path, tongue(15.0))
    with pytest.raises(TrackFileError):
        evaluate(lambda obs: obs, [str(path)], EnvConfig(), laps=1)


def test_decoration_seed_defaults_to_zero_on_generated_tracks():
    assert generate_track(1).decoration_seed == 0
    assert eval_track(0).decoration_seed == 0 and EVAL_SEEDS[0] >= 1_000_000


# -- command lines ---------------------------------------------------------------------


def test_cli_validate_exit_codes_and_messages(tmp_path, capsys):
    good, bad, cut = tmp_path / "g.json", tmp_path / "b.json", tmp_path / "c.json"
    save_spec(good, GOOD)
    save_spec(bad, tongue(15.0))
    cut.write_text('{"format": 1, "na', encoding="utf-8")

    with pytest.raises(SystemExit) as ok:
        cli_main(["validate", str(good)])
    assert ok.value.code == 0
    assert "OK" in capsys.readouterr().out

    for path, message in ((bad, "demasiado cerca"), (cut, "incompleto")):
        with pytest.raises(SystemExit) as fail:
            cli_main(["validate", str(good), str(path)])
        assert fail.value.code == 1
        assert message in capsys.readouterr().err


def test_cli_export_writes_a_valid_file(tmp_path, capsys):
    out = tmp_path / "x.json"
    with pytest.raises(SystemExit) as done:
        cli_main(["export", "--seed", "5", "--out", str(out)])
    assert done.value.code == 0
    assert validate_spec(load_spec(out)).ok
    with pytest.raises(SystemExit) as done:
        cli_main(["export", "--showcase", "canyon", "--out", str(tmp_path / "c.json")])
    assert done.value.code == 0
    with pytest.raises(SystemExit) as refused:
        cli_main(["export", "--seed", str(EVAL_SEEDS[0]), "--out", str(tmp_path / "e.json")])
    assert refused.value.code == 2 and not (tmp_path / "e.json").exists()
    assert "reservada" in capsys.readouterr().err


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )


def test_track_file_script_fails_cleanly_on_a_truncated_file(tmp_path):
    path = tmp_path / "cut.json"
    save_spec(path, GOOD)
    path.write_bytes(path.read_bytes()[:50])
    result = run_script("scripts/track_file.py", "validate", str(path))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr and "incompleto" in result.stderr


def test_evaluate_script_refuses_an_invalid_track_file(tmp_path):
    path = tmp_path / "bad.json"
    save_spec(path, tongue(15.0))
    result = run_script("scripts/evaluate.py", "--track-file", str(path))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "Circuito no válido" in result.stderr and "demasiado cerca" in result.stderr


def test_evaluate_script_refuses_a_corrupt_track_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff\xfe")
    result = run_script("scripts/evaluate.py", "--track-file", str(path))
    assert result.returncode != 0 and "Traceback" not in result.stderr
    assert "UTF-8" in result.stderr


def test_drive_accepts_a_track_file_and_refuses_an_invalid_one(tmp_path):
    from kartlab.show.drive import main as drive_main

    shot = tmp_path / "shot.png"
    drive_main(
        [
            "--track-file",
            str(EXAMPLE),
            "--view",
            "topdown",
            "--frames",
            "3",
            "--screenshot",
            str(shot),
        ]
    )
    assert shot.stat().st_size > 0
    bad = tmp_path / "bad.json"
    save_spec(bad, tongue(15.0))
    with pytest.raises(SystemExit) as error:
        drive_main(["--track-file", str(bad), "--frames", "1"])
    assert "Circuito no válido" in str(error.value)


def test_a_crossing_through_a_sample_point_is_still_rejected():
    s = np.linspace(0, 2 * np.pi, 600, endpoint=False)  # samples sit on the crossing
    eight = np.stack([120 * np.sin(s), 60 * np.sin(2 * s)], 1)
    assert not check_centerline(eight, 12.0).ok


# -- hostile files ---------------------------------------------------------------------


def write_json(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_huge_integers_are_a_clear_error_not_a_traceback(tmp_path):
    data = good_dict()
    data["corners"][0][0] = 10**400
    path = write_json(tmp_path / "t.json", json.dumps(data))
    with pytest.raises(TrackFileError, match="esquina 1"):
        load_spec(path)
    data = good_dict()
    data["width"] = 10**400
    with pytest.raises(TrackFileError, match="'width'"):
        load_spec(write_json(tmp_path / "w.json", json.dumps(data)))


def test_deeply_nested_json_is_a_clear_error(tmp_path):
    path = write_json(tmp_path / "t.json", "[" * 100_000 + "]" * 100_000)
    with pytest.raises(TrackFileError, match="anidado|JSON"):
        load_spec(path)


def test_utf8_bom_is_accepted_but_never_written(tmp_path):
    path = tmp_path / "t.json"
    save_spec(path, GOOD)
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_spec(path) == GOOD


def _timed_problems(spec: TrackSpec) -> tuple[str, float]:
    start = time.perf_counter()
    text = problems(spec)
    return text, time.perf_counter() - start


def test_enormous_tracks_are_rejected_quickly():
    star = tuple(
        (
            float(2900 * np.cos(t) * (1.0 if i % 2 else 0.3)),
            float(2900 * np.sin(t) * (1.0 if i % 2 else 0.3)),
            30.0,
        )
        for i, t in enumerate(np.linspace(0, 2 * np.pi, 40, endpoint=False))
    )
    square = ((-3000.0, -3000.0, 40.0), (3000.0, -3000.0, 40.0), (3000.0, 3000.0, 40.0))
    square += ((-3000.0, 3000.0, 40.0),)
    for corners in (star, square):
        text, seconds = _timed_problems(TrackSpec("Enorme", 12.0, corners))
        assert "longitud" in text
        assert seconds < 2.0


def test_cli_export_refuses_negative_seeds(tmp_path, capsys):
    with pytest.raises(SystemExit) as refused:
        cli_main(["export", "--seed", "-5", "--out", str(tmp_path / "n.json")])
    assert refused.value.code == 2 and "negativa" in capsys.readouterr().err
    with pytest.raises(ValueError, match="negativa"):
        spec_from_seed(-1)
