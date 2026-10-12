"""widening.widening_row on synthetic pinhole lines (WWH-34): a widening is found,
a grade change and a bend are not."""

import numpy as np

from libs.inference.widening import widening_row
from tests import synthetic as syn


def _project(X_of_z, rise_of_z=None):
    """Image curve {"y", "x"} of a lane line at lateral offset X(z) (m, + right) on a road
    whose height above the wheel plane is rise(z) (m; None = flat)."""
    z = np.linspace(3.0, 80.0, 4000)
    X = X_of_z(z)
    Y = rise_of_z(z) if rise_of_z is not None else np.zeros_like(z)
    v = syn.CY + syn.F_Y * (syn.CAM_H - Y) / z
    u = syn.CX + syn.F_X * X / z
    k = (v > syn.CY + 1) & (v <= syn.IMG_H - 1)
    o = np.argsort(v[k])
    rows = np.arange(np.ceil(v[k][o][0]), np.floor(v[k][o][-1]) + 1)
    return {"y": rows, "x": np.interp(rows, v[k][o], u[k][o])}


def _raw(c):
    return np.column_stack([c["x"], c["y"]])


def _row_depth(row):
    return syn.F_Y * syn.CAM_H / (row - syn.CY)


def _run(left, right, extra=()):
    lanes = [_raw(left), _raw(right)] + [_raw(c) for c in extra]
    return widening_row(left, right, lanes, syn.F_X, syn.F_Y, syn.CAM_H, syn.IMG_H)


const = lambda x: (lambda z: np.full_like(z, x))
hill = lambda z: 0.05 * np.clip(z - 12.0, 0, None)               # 5 % grade from 12 m on


def test_constant_lane_no_change():
    assert _run(_project(const(-1.7)), _project(const(1.7))) == (None, "A")


def test_one_sided_widening_found_by_a():
    """Right line tapers out 1 m over 20-30 m (a turn bay): A finds it near 20 m."""
    right = _project(lambda z: 1.7 + np.clip((z - 20.0) / 10.0, 0, 1))
    row, cue = _run(_project(const(-1.7)), right)
    assert cue == "A" and row is not None and 18.0 <= _row_depth(row) <= 30.0


def test_grade_change_is_not_a_widening():
    """A 5 % climb moves both lines in the image, but their vanishing-point ratio stays."""
    assert _run(_project(const(-1.7), hill), _project(const(1.7), hill)) == (None, "A")


def test_bend_is_gated():
    """Both lines shift right together from 20 m (a bend): no cut."""
    bend = lambda x0: (lambda z: x0 + 0.004 * np.clip(z - 20.0, 0, None) ** 2)
    assert _run(_project(bend(-1.7)), _project(bend(1.7))) == (None, "A")


def test_neighbour_line_used_and_finds_widening():
    left, nb = _project(const(-1.7)), _project(const(-5.1))
    right = _project(lambda z: 1.7 + np.clip((z - 20.0) / 10.0, 0, 1))
    row, cue = _run(left, right, extra=(nb,))
    assert cue == "B" and row is not None and 18.0 <= _row_depth(row) <= 32.0


def test_neighbour_line_on_a_hill_no_change():
    left, right, nb = (_project(const(x), hill) for x in (-1.7, 1.7, -5.1))
    assert _run(left, right, extra=(nb,)) == (None, "B")
