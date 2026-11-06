"""The ``bridge`` key of the track format (format 2) and the circuit that uses it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from kartlab.sim import trackfile
from kartlab.sim.levels import BridgeSpec
from kartlab.sim.trackfile import (
    TrackFileError,
    build_track,
    load_spec,
    load_track_file,
    save_spec,
    spec_from_dict,
    spec_to_dict,
    spec_to_json,
    validate_spec,
)
from kartlab.sim.trackissues import analyze

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "tracks" / "ocho_puente.json"
PLAIN = ROOT / "tracks" / "ocho.json"
# sha256 of tracks/ocho.json (LF line endings): the flat crossing must never change.
PLAIN_SHA256 = "233d33302c4fd7795956e35cd6a098903cb9150574369f2a86fc000fe9937d5f"


@pytest.fixture(scope="module")
def spec():
    return load_spec(FILE)


def data(**changes) -> dict:
    base = json.loads(FILE.read_text(encoding="utf-8"))
    base.update(changes)
    return base


def problems(raw: dict) -> str:
    with pytest.raises(TrackFileError) as error:
        spec_from_dict(raw)
    return "\n".join(error.value.problems)


# -- the circuit -----------------------------------------------------------------------------


def test_the_flat_eight_file_is_untouched():
    text = PLAIN.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(text).hexdigest() == PLAIN_SHA256
    assert b"bridge" not in text


def test_the_bridge_file_is_the_flat_eight_plus_a_bridge(spec):
    flat = load_spec(PLAIN)
    assert spec.name == "Lazo del Molino - puente"
    assert replace(spec, name=flat.name, bridge=None) == flat
    assert spec.bridge == BridgeSpec(0, "second", 3.5, 25.0)


def test_the_bridge_file_is_valid_and_builds_levels(spec):
    report = validate_spec(spec)
    assert report.ok and report.crossings == 1
    assert analyze(spec).ok
    track = load_track_file(FILE)
    assert track.levels is not None and len(track.crossings) == 1
    assert track.length >= 4 * 2 * track.levels.reach


def test_the_file_is_exactly_what_the_editor_saves(spec, tmp_path):
    assert spec_to_json(spec) == FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    save_spec(tmp_path / "copy.json", spec)
    assert (tmp_path / "copy.json").read_bytes() == spec_to_json(spec).encode("utf-8")


# -- reading and writing -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bridge",
    [
        BridgeSpec(0),
        BridgeSpec(0, "first"),
        BridgeSpec(0, "second", 2.0, 15.0),
        BridgeSpec(0, "first", 6.0, 50.0),
        BridgeSpec(0, "second", 4.25, 30.5),
    ],
)
def test_bridge_round_trips_through_dict_and_json(spec, tmp_path, bridge):
    mine = replace(spec, bridge=bridge)
    assert spec_from_dict(spec_to_dict(mine)) == mine
    assert spec_from_dict(json.loads(spec_to_json(mine))) == mine
    save_spec(tmp_path / "t.json", mine)
    assert load_spec(tmp_path / "t.json") == mine


def test_optional_fields_take_their_defaults(spec):
    raw = data(bridge={"crossing": 0})
    assert spec_from_dict(raw).bridge == BridgeSpec(0, "second", 3.5, 25.0)
    assert spec_from_dict(data(bridge={"crossing": 0, "over": "first"})).bridge.over == "first"


def test_a_spec_without_a_bridge_is_written_as_before(spec):
    plain = load_spec(PLAIN)
    assert "bridge" not in spec_to_dict(plain)
    assert "bridge" not in spec_to_json(plain)
    assert spec_to_json(plain) == PLAIN.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_bridge_is_part_of_format_2_only():
    raw = data(format=1)
    raw.pop("crossings")
    raw.pop("start")
    assert "'bridge'" in problems(raw)


def test_a_bridge_needs_crossings():
    raw = data()
    raw.pop("crossings")
    raw.pop("start")
    assert "crossings" in problems(raw)


@pytest.mark.parametrize(
    ("bridge", "message"),
    [
        ({"crossing": 1}, "solo declara 1 cruce"),
        ({"crossing": 7}, "índices de 0 a 0"),
        ({"crossing": -1}, "entero desde 0"),
        ({"crossing": 0.0}, "entero desde 0"),
        ({"crossing": True}, "entero desde 0"),
        ({}, "necesita 'crossing'"),
        ({"crossing": 0, "over": "third"}, "'bridge.over'"),
        ({"crossing": 0, "over": None}, "'bridge.over'"),
        ({"crossing": 0, "colour": "red"}, "campos desconocidos: 'colour'"),
        ({"crossing": 0, "deck_height": 1.0}, "'bridge.deck_height' = 1 fuera de rango"),
        ({"crossing": 0, "deck_height": 7}, "fuera de rango"),
        ({"crossing": 0, "deck_height": "alto"}, "debe ser un número"),
        ({"crossing": 0, "ramp": 14.9}, "'bridge.ramp' = 14.9 fuera de rango"),
        ({"crossing": 0, "ramp": 80}, "fuera de rango"),
        ({"crossing": 0, "ramp": None}, "debe ser un número"),
        ([0], "debe ser un objeto"),
        ("puente", "debe ser un objeto"),
    ],
)
def test_malformed_bridges_are_reported(bridge, message):
    assert message in problems(data(bridge=bridge))


def test_a_bridge_with_unknown_field_and_a_good_one_reports_the_unknown_field():
    text = problems(data(bridge={"crossing": 0, "spans": 2}))
    assert "'spans'" in text


# -- validation ----------------------------------------------------------------------------------


def test_a_bridge_on_a_crossing_that_does_not_exist_is_a_problem(spec):
    mine = replace(spec, bridge=BridgeSpec(3))
    assert "usa el cruce 3" in "\n".join(validate_spec(mine).problems)
    with pytest.raises(TrackFileError):
        build_track(mine)


def test_the_ramp_may_not_reach_the_start_line(spec, monkeypatch):
    # The real start is 290 m from the bridge; pretend the safety margin is enormous.
    monkeypatch.setattr(trackfile, "START_CLEARANCE", 400.0)
    text = "\n".join(validate_spec(spec).problems)
    assert "llega a la línea de meta" in text
    located = [i for i in analyze(spec).issues if "llega a la línea de meta" in i.message]
    assert located and located[0].points  # the editor can mark the spot


def test_the_track_must_be_much_longer_than_the_bridge(spec, monkeypatch):
    monkeypatch.setattr(trackfile, "MIN_LENGTH_RATIO", 50.0)
    assert "al menos 50 veces más largo" in "\n".join(validate_spec(spec).problems)


def test_the_deck_must_be_straight(spec, monkeypatch):
    monkeypatch.setattr(trackfile, "CROSSING_MAX_CURVATURE", -1.0)
    text = "\n".join(validate_spec(spec).problems)
    assert "no es recto" in text


def test_issues_carry_a_place_for_the_editor(spec, monkeypatch):
    monkeypatch.setattr(trackfile, "MIN_LENGTH_RATIO", 50.0)
    issues = [i for i in analyze(spec).issues if "veces más largo" in i.message]
    assert issues and issues[0].points
