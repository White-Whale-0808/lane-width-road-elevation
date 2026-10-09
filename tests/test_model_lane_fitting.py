"""Detector front end: ego pick, per-row densify, centre refinement, tail rule.

Synthetic images only (no weights, no GPU): a road of ROAD_GRAY with painted
stripes of PAINT_GRAY whose width follows the pinhole model at each row.
"""

import numpy as np
import pytest

from libs.inference.model_lane_fitting import (_GROUP_M, apply_tail, densify,
                                               guard_ego, trim_pitch_to_depth,
                                               model_curve, near_pair_width, pick_ego,
                                               refine_center, src_at)
from libs.inference.paint_evidence import _STRIPE_M
from tests import synthetic as syn

CAM = (syn.F_X, syn.F_Y, syn.CAM_H)
ROWS = np.arange(300, 480, dtype=float)       # z ≈ 11 m .. 2.2 m


def _stripe_px(y, metres=_STRIPE_M):
    return syn.F_X * metres / (syn.F_Y * syn.CAM_H / (y - syn.CY))


def _image(centres_by_row, metres=_STRIPE_M):
    """Road image with one stripe per (row -> [centre x, ...]) entry, each of
    real width `metres` at that row's depth."""
    img = np.full((syn.IMG_H, syn.IMG_W, 3), syn.ROAD_GRAY, dtype=np.uint8)
    cols = np.arange(syn.IMG_W)
    for y, centres in centres_by_row.items():
        half = _stripe_px(y, metres) / 2.0
        for c in centres:
            img[int(y), (cols >= c - half) & (cols <= c + half)] = syn.PAINT_GRAY
    return img


def test_pick_ego_takes_nearest_row_then_closest_to_centre():
    ego_l = np.array([[300.0, 511.0], [480.0, 300.0]])
    ego_r = np.array([[760.0, 511.0], [560.0, 300.0]])
    # neighbour lines leave the image sideways, so they only start higher up;
    # higher up they are still farther from the centre than the ego lines
    nb_l = np.array([[5.0, 420.0], [330.0, 300.0]])
    nb_r = np.array([[1020.0, 420.0], [700.0, 300.0]])
    left, right, yl, yr = pick_ego([nb_l, ego_l, nb_r, ego_r], syn.IMG_W)
    assert left is ego_l and right is ego_r
    assert yl == 511 and yr == 511


def test_pick_ego_known_risk_missing_ego_side_takes_neighbour():
    """pick_ego's documented risk: with the ego left line missed, the walk goes
    up and finds the neighbour's. guard_ego is what catches it (tests below)."""
    ego_r = np.array([[760.0, 511.0], [560.0, 300.0]])
    nb_l = np.array([[5.0, 420.0], [330.0, 300.0]])
    left, right, yl, _ = pick_ego([nb_l, ego_r], syn.IMG_W)
    assert left is nb_l and yl == 420


def test_pick_ego_empty():
    assert pick_ego([], syn.IMG_W) == (None, None, None, None)


def test_pick_ego_never_takes_one_line_for_both_sides():
    # a line left of centre near the car that crosses the centre farther up
    # (curve / lane change); with no right line it must not also become 'right'
    crossing = np.array([[400.0, 511.0], [700.0, 300.0]])
    left, right, _, _ = pick_ego([crossing], syn.IMG_W)
    assert left is crossing and right is None


def test_densify_every_integer_row_linear():
    rows, x = densify(np.array([[100.0, 410.4], [140.0, 400.2]]))
    assert rows[0] == 401 and rows[-1] == 410
    assert np.all(np.diff(rows) == 1)
    assert np.allclose(x, np.interp(rows, [400.2, 410.4], [140.0, 100.0]))


def test_refine_snaps_to_stripe_centre_subpixel():
    centre = 400.3
    img = _image({y: [centre] for y in ROWS})
    prior = np.full(len(ROWS), centre + 3.0)      # detector off by 3 px
    x, paint = refine_center(img, ROWS, prior, *CAM)
    assert paint.all()
    assert np.abs(x - centre).max() < 0.6


def test_refine_no_paint_keeps_prior_and_flags_model():
    img = _image({})
    prior = np.full(len(ROWS), 400.0)
    x, paint = refine_center(img, ROWS, prior, *CAM)
    assert not paint.any()
    assert np.array_equal(x, prior)


def test_refine_double_marking_uses_group_centre():
    """Two stripes with a gap, whole group narrower than _GROUP_M: the centre
    is the group's, as the detector labels a double line."""
    gap_m = 0.1
    rows = ROWS[ROWS > 380]                       # near rows: stripes resolved
    by_row = {}
    for y in rows:
        off = syn.F_X * (_STRIPE_M + gap_m) / 2 / (syn.F_Y * syn.CAM_H / (y - syn.CY))
        by_row[y] = [400.0 - off, 400.0 + off]
    img = _image(by_row)
    x, paint = refine_center(img, rows, np.full(len(rows), 402.0), *CAM)
    assert (_STRIPE_M * 2 + gap_m) < _GROUP_M
    assert paint.all()
    assert np.abs(x - 400.0).max() < 1.0


def test_refine_rejects_bright_plateau():
    """A bright surface far wider than any stripe is not paint."""
    img = _image({y: [400.0] for y in ROWS}, metres=1.5)
    x, paint = refine_center(img, ROWS, np.full(len(ROWS), 400.0), *CAM)
    assert not paint.any()


def test_apply_tail():
    rows = np.array([300.0, 310, 320, 330, 340])
    x = np.arange(5.0)
    paint = np.array([False, False, True, False, True])
    r, _, p = apply_tail(rows, x, paint, "keep")
    assert len(r) == 5
    r, _, p = apply_tail(rows, x, paint, "last_paint")
    assert list(r) == [320, 330, 340] and list(p) == [True, False, True]
    r, _, _ = apply_tail(rows, x, np.zeros(5, bool), "last_paint")
    assert len(r) == 0
    with pytest.raises(ValueError):
        apply_tail(rows, x, paint, "bogus")


def _line_at_offset(X_m, y_top=300.0):
    """Straight flat-road line at lateral offset X_m (m), image rows y_top..511."""
    ys = np.arange(y_top, 512.0)
    z = syn.F_Y * syn.CAM_H / (ys - syn.CY)
    return np.column_stack([syn.CX + syn.F_X * X_m / z, ys])


def _guard(lanes, left, right):
    return guard_ego(lanes, left, right, syn.F_X, syn.F_Y, syn.CAM_H, syn.IMG_W, syn.IMG_H)


def test_guard_ok_pair():
    l, r = _line_at_offset(-1.7), _line_at_offset(1.7)
    assert _guard([l, r], l, r) == (l, r, "ok")


def test_guard_wide_drops_the_outer_side():
    """Ego left missed → neighbour's line (-5.1 m) picked: width 6.8 m."""
    nb, r = _line_at_offset(-5.1), _line_at_offset(1.7)
    left, right, why = _guard([nb, r], nb, r)
    assert left is None and right is r and why == "wide_dropped"


def test_guard_narrow_replaces_phantom_with_next_line_out():
    phantom, l, r = _line_at_offset(-0.5), _line_at_offset(-1.7), _line_at_offset(1.7)
    left, right, why = _guard([phantom, l, r], phantom, r)
    assert left is l and right is r and why == "narrow_replaced"


def test_guard_narrow_without_alternative_drops():
    phantom, r = _line_at_offset(-0.5), _line_at_offset(1.7)
    left, right, why = _guard([phantom, r], phantom, r)
    assert left is None and right is r and why == "narrow_dropped"


def test_guard_one_side_untouched():
    r = _line_at_offset(1.7)
    assert _guard([r], None, r) == (None, r, "one_side")


def test_guard_wide_replaces_with_ego_line_that_starts_higher():
    """The ego left line starts above the neighbour's lowest row, so pick_ego
    took the neighbour; the guard finds the ego line instead of dropping the side."""
    nb, r = _line_at_offset(-5.1), _line_at_offset(1.7)
    ego_l = _line_at_offset(-1.7)
    ego_l = ego_l[ego_l[:, 1] <= 450.0]
    left, right, why = _guard([nb, ego_l, r], nb, r)
    assert left is ego_l and right is r and why == "wide_replaced"


def _guard_kw(lanes, left, right, keep_wide=True):
    return guard_ego(lanes, left, right, syn.F_X, syn.F_Y, syn.CAM_H, syn.IMG_W, syn.IMG_H,
                     keep_wide=keep_wide)


def _line_diverging(X0, dX_per_m, y_top=300.0):
    """Flat-road line whose lateral offset drifts dX_per_m per metre of depth."""
    ys = np.arange(y_top, 512.0)
    z = syn.F_Y * syn.CAM_H / (ys - syn.CY)
    X = X0 + dX_per_m * (z - z.min())
    return np.column_stack([syn.CX + syn.F_X * X / z, ys])


def test_guard_keep_wide_keeps_a_parallel_wide_lane():
    """WWH-33: a 5.3 m lane is kept; without keep_wide its outer line is dropped as before."""
    l, r = _line_at_offset(-2.65), _line_at_offset(2.65)
    assert _guard_kw([l, r], l, r) == (l, r, "wide_kept")
    left, right, why = _guard_kw([l, r], l, r, keep_wide=False)
    assert why == "wide_dropped" and (left is None) != (right is None)


def test_guard_keep_wide_keeps_two_lanes_too():
    """By width alone two 3.4 m lanes (6.8 m) are a wide lane: kept. The scale
    check in pipeline_clrnet is what drops it when the sequence's width differs."""
    nb, r = _line_at_offset(-5.1), _line_at_offset(1.7)
    assert _guard_kw([nb, r], nb, r) == (nb, r, "wide_kept")


def test_guard_beyond_the_absolute_width_still_dropped():
    nb, r = _line_at_offset(-6.5), _line_at_offset(1.7)            # 8.2 m
    left, right, why = _guard_kw([nb, r], nb, r)
    assert left is None and right is r and why == "wide_dropped"


def test_guard_wide_pair_that_diverges_is_not_kept():
    l, r = _line_diverging(-2.65, -1.0), _line_at_offset(2.65)    # 5.3 m -> ~7.2 m at twice the depth
    left, right, why = _guard_kw([l, r], l, r)
    assert left is None and right is r and why == "wide_dropped"


def test_near_pair_width_on_flat_road():
    l, r = _line_at_offset(-1.7), _line_at_offset(1.7)
    w, z = near_pair_width(l, r, syn.F_X, syn.F_Y, syn.CAM_H, syn.IMG_W, syn.IMG_H)
    assert abs(w - 3.4) < 0.05 and z > 0


def test_guard_far_shared_row_left_unchecked():
    """Near the horizon a pixel is metres of width: no verdict, no drop."""
    rows = np.arange(syn.CY + 2.0, syn.CY + 6.0)
    z = syn.F_Y * syn.CAM_H / (rows - syn.CY)
    l = np.column_stack([syn.CX - syn.F_X * 5.0 / z, rows])      # a "10 m lane"
    r = np.column_stack([syn.CX + syn.F_X * 5.0 / z, rows])
    assert _guard([l, r], l, r) == (l, r, "far_unchecked")


def test_guard_lines_sharing_no_row():
    l, r = _line_at_offset(-1.7), _line_at_offset(1.7)
    l, r = l[l[:, 1] >= 450.0], r[r[:, 1] <= 400.0]
    assert _guard([l, r], l, r) == (l, r, "no_overlap")


def _pitch_result():
    zs = np.linspace(3.0, 40.0, 38)
    return {"pitch_at": lambda z: float(z), "z_samples": zs, "pitch_samples": zs.copy(),
            "y_samples": -zs, "z_points": zs, "y_points": -zs,
            "z_visible_min": 3.0, "z_visible_max": 40.0, "widths": np.ones((5, 2))}


def test_trim_pitch_keeps_near_output_identical():
    full = _pitch_result()
    t = trim_pitch_to_depth(full, 20.0)
    near = full["z_samples"] <= 20.0
    assert np.array_equal(t["pitch_samples"], full["pitch_samples"][near])
    assert t["z_samples"].max() <= 20.0 and t["z_visible_max"] == 20.0
    assert t["pitch_at"](35.0) == t["pitch_at"](t["z_samples"][-1])   # clamped
    assert t["widths"] is full["widths"]
    assert trim_pitch_to_depth(full, None) is full
    assert trim_pitch_to_depth(full, 100.0) is full
    empty = trim_pitch_to_depth(full, 1.0)
    assert empty["pitch_at"] is None and len(empty["z_samples"]) == 0
    assert np.isnan(empty["z_visible_min"]) and np.isnan(empty["z_visible_max"])   # estimator contract
    assert empty["widths"] is full["widths"]


def test_model_curve_and_src_at():
    c = model_curve([12.0, 10.0, 11.0], [3.0, 1.0, 2.0], [True, False, True])
    assert list(c["y"]) == [10, 11, 12] and list(c["x"]) == [1, 2, 3]
    assert list(src_at(c, [10, 11, 12, 11.4])) == [False, True, True, True]
    assert model_curve([1.0], [1.0], [True]) is None


def test_smooth_refine_corrects_gap_rows_of_a_dashed_line():
    """A steady 0.10 m lateral detector error, paint seen on every other 20-row
    block: the smoothed offset moves the gap rows onto the line too."""
    from libs.inference.model_lane_fitting import smooth_refine
    rows = np.arange(300.0, syn.IMG_H)
    z = syn.F_Y * syn.CAM_H / (rows - syn.CY)
    x_true = syn.CX - syn.F_X * 1.6 / z
    x_prior = x_true + syn.F_X * 0.10 / z
    paint = ((rows - rows[0]) // 20) % 2 == 0                         # dashes at both ends
    x_snap = np.where(paint, x_true, x_prior)
    x = smooth_refine(rows, x_prior, x_snap, paint, syn.F_X, syn.F_Y, syn.CAM_H, syn.IMG_H)
    assert np.abs(x - x_true).max() < 1e-6
    few = np.zeros_like(paint); few[:5] = True                          # too little paint
    assert np.array_equal(smooth_refine(rows, x_prior, x_snap, few, syn.F_X, syn.F_Y, syn.CAM_H, syn.IMG_H), x_prior)


def test_smooth_refine_ignores_occluder_snaps():
    """Snaps onto a car body (0.8 m off) or scattered snaps are not paint of this
    line: the detector line, already right, must stay put."""
    from libs.inference.model_lane_fitting import smooth_refine
    rows = np.arange(300.0, syn.IMG_H)
    z = syn.F_Y * syn.CAM_H / (rows - syn.CY)
    x = syn.CX - syn.F_X * 1.6 / z
    paint = np.ones(len(rows), dtype=bool)
    car = (rows >= 400) & (rows < 460)
    snap = np.where(car, x + syn.F_X * 0.8 / z, x)                      # car body beside the line
    assert np.allclose(smooth_refine(rows, x, snap, paint, syn.F_X, syn.F_Y, syn.CAM_H, syn.IMG_H), x)
    rng = np.random.default_rng(0)
    noisy = x + syn.F_X * rng.uniform(-0.25, 0.25, len(rows)) / z         # scattered, under the cap
    assert np.abs(smooth_refine(rows, x, noisy, paint, syn.F_X, syn.F_Y, syn.CAM_H, syn.IMG_H) - x).max() < 0.5

