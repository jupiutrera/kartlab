import dataclasses
import random

import pytest

from kartlab.sim.trackfile import build_track, load_spec, validate_spec
from kartlab.sim.trackissues import analyze


@pytest.fixture(scope="module")
def example():
    return load_spec("tracks/ejemplo.json")


@pytest.fixture(scope="module")
def eight():
    return load_spec("tracks/ocho.json")


def moved(spec, index, x, y):
    corners = list(spec.corners)
    corners[index] = (x, y, corners[index][2])
    return dataclasses.replace(spec, corners=tuple(corners))


def messages(analysis):
    return [issue.message for issue in analysis.issues]


def test_a_valid_track_has_no_issues_and_the_loaders_centerline(example):
    analysis = analyze(example)
    assert analysis.ok and analysis.complete
    assert analysis.report.ok
    assert (analysis.points == build_track(example).points).all()


def test_a_quick_analysis_of_a_valid_track_is_not_a_verdict(example):
    quick = analyze(example, full=False)
    assert quick.ok and not quick.complete
    assert quick.points is not None


@pytest.mark.parametrize("name", ["example", "eight"])
def test_issue_messages_are_exactly_the_validators(name, request):
    """``analyze`` adds locations, never messages: fuzz both files and compare."""
    base = request.getfixturevalue(name)
    rng = random.Random(11)
    for _ in range(60):
        corners = list(base.corners)
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(corners))
            x, y, r = corners[i]
            roll = rng.random()
            if roll < 0.6:
                corners[i] = (x + rng.uniform(-150, 150), y + rng.uniform(-150, 150), r)
            elif roll < 0.8:
                corners[i] = (x, y, rng.choice([5, 10, 60, 120, 600]))
            else:
                corners.insert(i, (x + rng.uniform(-3, 3), y + rng.uniform(-3, 3), r))
        spec = dataclasses.replace(base, corners=tuple(corners), width=rng.choice([6, 12, 20, 25]))
        full = analyze(spec)
        assert messages(full) == list(validate_spec(spec).problems)
        assert full.ok == validate_spec(spec).ok
        # the quick level reports a subset of the same problems
        assert set(messages(analyze(spec, full=False))) <= set(messages(full))


def test_a_corner_that_does_not_fit_names_its_corner(example):
    analysis = analyze(moved(example, 4, 330, 150))
    corner_issues = [i for i in analysis.issues if i.message.startswith("esquina")]
    assert corner_issues
    for issue in corner_issues:
        number = int(issue.message.split()[1].rstrip(":"))
        assert issue.corners == (number - 1,)
        assert issue.points == (example.corners[number - 1][:2],) or issue.points


def test_a_self_crossing_polygon_names_the_four_corners_and_the_crossing_point(example):
    analysis = analyze(moved(example, 0, 330, 250))
    (issue,) = [i for i in analysis.issues if "se cruza a sí mismo" in i.message]
    assert len(issue.corners) == 4 and len(issue.edges) == 2 and len(issue.points) == 1
    assert analysis.points is None  # a crossing polygon is never built
    assert analysis.complete  # nothing more a full check could add


def test_a_corner_too_close_to_the_next_names_both(example):
    analysis = analyze(moved(example, 1, 2.0, -1.0))
    (issue,) = [i for i in analysis.issues if "están a" in i.message]
    assert set(issue.corners) == {0, 1} and issue.edges == ((0, 1),)


def test_too_little_grass_between_sections_marks_the_closest_pair(example):
    spec = moved(example, 0, 160, 235)
    full = analyze(spec)
    (issue,) = [i for i in full.issues if i.message.startswith("dos partes")]
    (ax, ay), (bx, by) = issue.points
    assert 0 < ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < spec.width + 8
    assert issue.corners
    assert full.points is not None  # still drawable
    # the quick level cannot see it: that is what the complete validation is for
    quick = analyze(spec, full=False)
    assert quick.ok and not quick.complete


def test_a_tight_curve_marks_the_tightest_point(eight):
    spec = moved(eight, 1, -215, -200)
    for full in (True, False):
        analysis = analyze(spec, full=full)
        (issue,) = [i for i in analysis.issues if i.message.startswith("curva demasiado")]
        assert len(issue.points) == 1 and len(issue.corners) == 1
        (x, y) = issue.points[0]
        # the point is on the centerline
        distance = ((analysis.points[:, 0] - x) ** 2 + (analysis.points[:, 1] - y) ** 2) ** 0.5
        assert distance.min() < 1e-6


def test_a_width_or_corner_count_problem_has_no_place(example):
    analysis = analyze(dataclasses.replace(example, width=30.0))
    (issue,) = analysis.issues
    assert issue.message.startswith("anchura") and not issue.corners and not issue.points
    few = analyze(dataclasses.replace(example, corners=example.corners[:2]))
    assert any("esquinas: debe haber" in i.message for i in few.issues)


def test_declared_crossings_are_kept_in_the_analysis_of_a_figure_eight(eight):
    analysis = analyze(eight)
    assert analysis.ok and analysis.report.crossings == 1
    broken = analyze(moved(eight, 0, -60, -130))  # the crossing moves away from the declared one
    assert any("cruce" in i.message for i in broken.issues)
    located = [i for i in broken.issues if "cruce" in i.message and i.points]
    assert located
