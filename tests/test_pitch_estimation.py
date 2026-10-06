"""Pitch estimation against roads whose grade is known by construction.

These say nothing about accuracy on real imagery — the batch MAE sweep owns
that. What they pin is that the metric stage inverts its own forward model,
and that it declines to answer instead of guessing when the input degenerates.
"""

import numpy as np
import pytest

from libs.inference.lane_fitting import lane_curve
from libs.inference.pitch_estimation import (NEARFIELD_POOL_K,
                                             NEARFIELD_POOL_MIN_N,
                                             NearfieldWidthCalibrator,
                                             back_project_widths,
                                             estimate_pitch_from_curves,
                                             estimate_w_real_nearfield,
                                             resolve_lane_width,
                                             sample_widths_from_curves)
from tests import synthetic as syn


def _estimate(pitch_deg, method, **kw):
    left, right = syn.road_points(pitch_deg, **kw)
    return estimate_pitch_from_curves(
        lane_curve(left), lane_curve(right),
        syn.F_X, syn.F_Y, syn.IMG_H, syn.W_REAL,
        num_samples=80, samples_per_meter=6, method=method)


@pytest.mark.parametrize("method", ["windowed", "spline"])
@pytest.mark.parametrize("pitch_deg", [-10.0, -3.0, 0.0, 3.0, 10.0])
def test_recovers_a_constant_grade(method, pitch_deg):
    """A road of constant grade must read back as that grade at every depth."""
    result = _estimate(pitch_deg, method)
    assert result["pitch_at"] is not None
    for z in np.linspace(result["z_visible_min"], result["z_visible_max"], 25):
        assert result["pitch_at"](z) == pytest.approx(pitch_deg, abs=0.1)


def test_reports_the_visible_depth_range():
    result = _estimate(0.0, "windowed", z_near=12.0, z_far=28.0)
    assert result["z_visible_min"] == pytest.approx(12.0, abs=0.1)
    assert result["z_visible_max"] == pytest.approx(28.0, abs=0.1)


def test_back_projection_round_trips():
    """(y, w) -> (z, Y_3d) must invert the forward model exactly."""
    z = np.array([8.0, 15.0, 30.0])
    ys = syn.row_for_depth(z)
    widths = np.column_stack([ys, syn.width_for_depth(z)])
    depths, Y_3d = back_project_widths(widths, syn.F_X, syn.F_Y,
                                       syn.IMG_H, syn.W_REAL)
    assert depths == pytest.approx(z)
    assert Y_3d == pytest.approx(np.full(3, -syn.CAM_H))


def test_sample_density_follows_visible_depth():
    """samples_per_meter is a density: doubling the visible span doubles n."""
    def n_samples(z_near, z_far):
        left, right = syn.road_points(0.0, z_near=z_near, z_far=z_far)
        return len(sample_widths_from_curves(
            lane_curve(left), lane_curve(right), num_samples=80,
            f_x=syn.F_X, w_real=syn.W_REAL, samples_per_meter=6))

    assert n_samples(10.0, 20.0) == pytest.approx(60, abs=1)
    assert n_samples(10.0, 30.0) == pytest.approx(120, abs=1)


@pytest.mark.parametrize("method", ["windowed", "spline"])
def test_declines_when_the_depth_range_is_too_short(method):
    """Below min_valid_range_m there is no slope to measure — say so."""
    result = _estimate(0.0, method, z_near=20.0, z_far=20.2, n=40)
    assert result["pitch_at"] is None
    assert len(result["z_samples"]) == 0


@pytest.mark.parametrize("method", ["windowed", "spline"])
def test_declines_on_a_missing_side(method):
    """One curve absent (junction, abstained side) must not fabricate a pitch."""
    left, _ = syn.road_points(0.0)
    result = estimate_pitch_from_curves(
        lane_curve(left), None, syn.F_X, syn.F_Y, syn.IMG_H, syn.W_REAL,
        num_samples=80, samples_per_meter=6, method=method)
    assert result["pitch_at"] is None


def _nearfield(theta0_deg=0.0, **kw):
    return estimate_w_real_nearfield(
        syn.nearfield_widths(theta0_deg, **kw),
        syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)


def test_nearfield_recovers_w_real_on_the_support_plane():
    """Level camera: every row must read back the physical width exactly."""
    result = _nearfield(0.0)
    assert result is not None
    assert result["w_real_med"] == pytest.approx(syn.W_REAL, rel=1e-9)
    assert result["w_real_z0"] == pytest.approx(syn.W_REAL, rel=1e-6)
    assert result["theta0_deg"] == pytest.approx(0.0, abs=1e-6)
    assert result["z_lo"] >= 2.0 and result["z_hi"] <= 5.0


def test_nearfield_separates_mounting_pitch_from_width():
    """A pitched camera trends w(z); intercept and slope split θ0 from w_real."""
    result = _nearfield(1.5)
    assert result["theta0_deg"] == pytest.approx(1.5, abs=0.1)
    assert result["w_real_z0"] == pytest.approx(syn.W_REAL, abs=0.02)
    # the plain median cannot see θ0 and must carry the +w·θ0·z̄/h bias
    assert result["w_real_med"] > syn.W_REAL + 0.2


def test_quality_gate_rejects_a_tilted_near_field():
    """|θ0| beyond what body pitch gives a level camera means the window is
    off the wheel plane (a grade change): rejected, however clean the fit."""
    assert _nearfield(0.5)["quality_ok"]
    assert not _nearfield(1.5)["quality_ok"]


def test_nearfield_declines_without_enough_near_rows():
    """Curves that stop short of the near window must not fabricate a width."""
    assert _nearfield(0.0, z_near=6.0, z_far=20.0) is None      # all beyond window
    assert _nearfield(0.0, z_near=2.0, z_far=5.0, n=5) is None  # too few rows
    horizon_up = syn.nearfield_widths(0.0)
    horizon_up[:, 0] = syn.IMG_H / 2.0 - 10.0                   # above the horizon
    assert estimate_w_real_nearfield(
        horizon_up, syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H) is None


def _curves(pitch_deg, z_near, z_far):
    left, right = syn.road_points(pitch_deg, z_near=z_near, z_far=z_far)
    return lane_curve(left), lane_curve(right)


def _fed(cal, dist, pitch_deg, z_near=2.0):
    cal.advance_to(dist)
    return cal.update(*_curves(pitch_deg, z_near, 30.0))


def test_calibrator_has_no_width_until_it_measures_one():
    """No configured fallback (stage C): before the first adoption there is
    no width, and the reason says why this frame was not measured."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    # curves that never reach the near window: nothing to measure
    assert _fed(cal, 0.0, 0.0, z_near=8.0) is None
    assert (cal.status, cal.reason) == ("no_anchor", "no_nearfield_rows")
    # gate opens, but adoption waits for MIN_RUN metres of open gate
    assert _fed(cal, 0.1, 0.0) is None
    assert (cal.status, cal.reason) == ("no_anchor", "run_too_short")
    assert _fed(cal, 0.7, 0.0) == pytest.approx(syn.W_REAL, rel=1e-6)
    assert (cal.status, cal.reason, cal.hold_frames) == ("measured", None, 0)


def test_calibrator_holds_for_the_rest_of_the_sequence():
    """A rejected frame reuses the last measured width however far back it
    was measured — and reports how far back that is."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    _fed(cal, 0.0, 0.0)
    _fed(cal, 0.7, 0.0)                                     # adopted here
    # grade knee at the camera (road_points pitches away from the support
    # plane at z=0): the near-field fit is no longer a straight w(z) trend, so
    # the quality gate rejects it — hold the adopted value
    assert _fed(cal, 1.0, 8.0) == pytest.approx(syn.W_REAL, rel=1e-6)
    assert (cal.status, cal.reason) == ("held", "quality_gate")
    # far past the old hold bound (the window's far edge): still held
    assert _fed(cal, 200.0, 8.0) == pytest.approx(syn.W_REAL, rel=1e-6)
    assert cal.status == "held"
    assert cal.hold_frames == 2
    assert cal.hold_m == pytest.approx(199.3)


def test_a_run_survives_one_failed_frame():
    """A near field that flickers (OpenLane 17612: every other frame without
    rows) still completes a run — one failure is a dropped observation."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    _fed(cal, 0.0, 0.0)                              # run starts
    _fed(cal, 0.5, 0.0, z_near=8.0)                  # one frame without rows
    assert _fed(cal, 1.0, 0.0) == pytest.approx(syn.W_REAL, rel=1e-6)
    assert cal.status == "measured"


def test_two_failed_frames_break_the_run():
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    _fed(cal, 0.0, 0.0)
    _fed(cal, 0.3, 0.0, z_near=8.0)
    _fed(cal, 0.6, 0.0, z_near=8.0)                  # second failure in a row
    assert _fed(cal, 1.0, 0.0) is None               # run restarted here
    assert cal.reason == "run_too_short"


def test_without_odometry_the_run_counts_frames():
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    good, bad = _curves(0.0, 2.0, 30.0), _curves(0.0, 8.0, 30.0)
    assert cal.update(*good) is None
    assert cal.update(*bad) is None
    assert cal.update(*bad) is None                  # two failures: run broken
    assert cal.update(*good) is None
    assert cal.update(*good) == pytest.approx(syn.W_REAL, rel=1e-6)


def test_calibrator_ignores_an_isolated_gate_pass():
    """One clean-looking frame between rejects (curvature inflection) must
    not be adopted — the θ0 gate is blind exactly there."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    _fed(cal, 0.0, 8.0)
    _fed(cal, 0.2, 0.0)          # isolated pass: run restarts, span 0 m
    assert _fed(cal, 0.4, 8.0) is None
    assert cal.status == "no_anchor"


def test_a_lone_image_uses_its_own_measurement():
    """sequence=False (single-image inference): no run to wait for."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H,
                                   sequence=False)
    assert cal.update(*_curves(0.0, 2.0, 30.0)) == pytest.approx(
        syn.W_REAL, rel=1e-6)
    assert cal.status == "measured" and cal.hold_m is None
    lone = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H,
                                    sequence=False)
    assert lone.update(*_curves(0.0, 8.0, 30.0)) is None
    assert lone.reason == "no_nearfield_rows"


def _fed_w(cal, dist, w_real):
    """Feed a flat road whose lane is w_real wide."""
    cal.advance_to(dist)
    left, right = syn.road_points(0.0, z_near=2.0, z_far=30.0, w_real=w_real)
    return cal.update(lane_curve(left), lane_curve(right))


def test_pooled_width_outvotes_a_wrong_pass():
    """From the POOL_MIN_N-th pass on the width is the median of the passes,
    so one wrong pass no longer replaces a good value — the single-frame
    policy, still in force before that, would have adopted it."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    _fed_w(cal, 0.0, 3.25)
    assert _fed_w(cal, 0.7, 3.25) == pytest.approx(3.25, rel=1e-6)
    assert cal.n_pooled == 1                          # the run adopted one frame
    assert NEARFIELD_POOL_MIN_N == 3
    assert _fed_w(cal, 1.4, 4.0) == pytest.approx(3.25, rel=1e-6)
    assert (cal.status, cal.n_pooled) == ("measured", 3)


def test_pool_forgets_all_but_the_last_k_passes():
    """A changed lane width takes over once it is the majority of the last
    POOL_K passes."""
    k = NEARFIELD_POOL_K
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    for i in range(2 * k):
        _fed_w(cal, 0.7 * i, 3.25)
    for i in range(k // 2 - 1):
        _fed_w(cal, 0.7 * (2 * k + i), 3.75)
    assert cal.w_real == pytest.approx(3.25, rel=1e-6)
    _fed_w(cal, 0.7 * 3 * k, 3.75)
    _fed_w(cal, 0.7 * 3 * k + 0.7, 3.75)
    assert cal.w_real == pytest.approx(3.75, rel=1e-6)
    assert cal.n_pooled == k


def test_last_resort_only_before_the_first_measurement():
    """The constant stands in only while the sequence has measured nothing,
    and says so; once measured, a failing frame holds the measurement."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    unreachable = _curves(0.0, 8.0, 30.0)         # nothing in the near window
    cal.advance_to(0.0)
    assert resolve_lane_width(cal, *unreachable, 3.5) == (3.5, "last_resort")
    cal.advance_to(0.1)
    resolve_lane_width(cal, *_curves(0.0, 2.0, 30.0), 3.5)
    cal.advance_to(0.7)
    w, status = resolve_lane_width(cal, *_curves(0.0, 2.0, 30.0), 3.5)
    assert status == "measured" and w == pytest.approx(syn.W_REAL, rel=1e-6)
    cal.advance_to(50.0)
    w, status = resolve_lane_width(cal, *unreachable, 3.5)
    assert status == "held" and w == pytest.approx(syn.W_REAL, rel=1e-6)


def test_no_last_resort_means_no_width():
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)
    assert resolve_lane_width(cal, *_curves(0.0, 8.0, 30.0)) == (None, "no_anchor")


def test_declines_on_crossed_curves():
    """Curves that cross give non-positive widths: unphysical, not a pitch."""
    left, right = syn.road_points(0.0)
    result = estimate_pitch_from_curves(
        lane_curve(right), lane_curve(left),   # swapped: right is now inner-left
        syn.F_X, syn.F_Y, syn.IMG_H, syn.W_REAL,
        num_samples=80, samples_per_meter=6, method="windowed")
    assert result["pitch_at"] is None
