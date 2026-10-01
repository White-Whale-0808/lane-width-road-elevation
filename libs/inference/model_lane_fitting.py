import numpy as np

from libs.inference.geometry import CameraGeometry
from libs.inference.paint_evidence import _RIDGE_THR, _STRIPE_M, _gray
from libs.inference.pitch_estimation import _empty_result, nearfield_window

"""
Lane fitting for a learned 2D lane detector front end (WWH-25).

The detector (CLRNet, lane_detector.py) returns whole lane-line instances —
through dashed gaps and unpainted road — at ~72 points each. This module turns
the two ego lines into the per-row curves the metric stage consumes, keeping
what the image can verify apart from what the model inferred:

    pick_ego        the ego pair, chosen as NEAR the vehicle as possible
    guard_ego       reject a pair whose near-field width is not a lane width
    densify         the detector polyline at every integer image row
    refine_center   per row: find the painted stripe's CENTRE when the image
                    shows one (src = paint), else src = model. The pipeline
                    uses it for the FLAG only by default — snapping the
                    position made pitch worse (pipeline_clrnet `refine`)
    apply_tail      optionally drop model-only rows beyond the farthest paint
    model_curve     {"y", "x", "src"} — lane_fitting.lane_curve's format plus
                    the source flag, so the existing metric stage reads it as is
    trim_pitch_to_depth  no pitch beyond a depth the weights are trusted to

Why the CENTRE and not the inner edge the ELSED front end measures: the
detector is trained on centre-line labels, so its rows (src = model) are
centres. Refining paint rows to the inner edge would make the reference
point jump by half a stripe at every paint/model switch — a width step along
depth, which the two-plane stage reads as grade (WWH-25 §3). With both on the
centre the near-field width is centre-to-centre, which is also what OpenLane's
lane_width_gt measures.

Nothing here reads the lane width; the only geometry is the row depth
f_y*h/(y-cy) (with paint_evidence's grade margin), as in paint_evidence.
"""

# Double marking (double yellow etc.): outer extent of the whole group, used
# to (a) size the search window so the group fits in it and (b) decide that
# two stripes belong to one marking. 2 stripes x 0.125 m + a gap of up to
# 0.2 m. Assumed from road-marking practice, not measured on our data.
_GROUP_M = 0.45

# Search slack around the detector prior, px, on top of half the group width.
# Step 1 measured the detector's centre error at |dx| median 2.0 px (OpenLane
# updown_straight) / 2.4 px (Town05); twice that.
_PRIOR_TOL_PX = 4

# Hard cap on the half-window, px (same order as paint_evidence._FAR_CAP_PX).
_WIN_CAP_PX = 40

# Background probe: this many px, starting 2 px beyond the stripe-width bound
# measured from the centre.
_BG_PX = 6

# Plausible ego-lane width (m, centre to centre) for guard_ego — the project's
# existing range, see guard_ego.
_EGO_W_MIN, _EGO_W_MAX = 2.5, 4.5

# guard_ego checks the width only where the flat-ground depth is trustworthy:
# up to this multiple of the near-field window's far end (OpenLane ~20 m,
# CARLA ~8 m). Farther out a pixel of noise is a large width error and the road
# need not be the wheel plane, so the pair is left unchecked instead.
_GUARD_MAX_Z_FACTOR = 2.0


def trim_pitch_to_depth(pitch_curve, max_depth_m):
    """The pitch result with everything beyond max_depth_m removed — the
    estimate itself is made on the FULL curves, only the output is trimmed.

    Trimming the output rather than the curves keeps the near field unchanged
    (cutting the curves first changes the sampling and made 5-10 m worse).
    The depth is the pitch stage's own z. pitch_at is clamped at the new far
    end. Nothing left ⇒ the estimator's empty result. None max_depth_m: unchanged.

    Why a limit at all: the CULane weights draw the far road as flat (grade-change
    ratio k 0.55-0.64 within 20 m, 0.09 at 30-50 m on Town05, while the estimator
    on ground-truth lines keeps k >= 0.96). The value is a property of the
    WEIGHTS; re-measure for other weights.
    """
    if max_depth_m is None or pitch_curve.get("pitch_at") is None:
        return pitch_curve
    zs = np.asarray(pitch_curve["z_samples"])
    keep = zs <= max_depth_m
    if keep.all():
        return pitch_curve
    out = dict(pitch_curve)
    if not keep.any():
        out.update(_empty_result())           # same contract as the estimator's empty result
        return out
    for key in ("z_samples", "pitch_samples", "y_samples"):
        out[key] = np.asarray(pitch_curve[key])[keep]
    zp = np.asarray(pitch_curve.get("z_points", []))
    if len(zp):
        kp = zp <= max_depth_m
        out["z_points"] = zp[kp]
        out["y_points"] = np.asarray(pitch_curve["y_points"])[kp]
    z_far = float(out["z_samples"][-1])
    out["z_visible_max"] = min(pitch_curve.get("z_visible_max", z_far), max_depth_m)
    inner = pitch_curve["pitch_at"]
    out["pitch_at"] = lambda z, f=inner, lim=z_far: f(min(z, lim))
    return out


def pick_ego(lanes, image_width):
    """The ego pair: per side, the line nearest the image centre at the NEAREST
    image row that side has any line.

    Walks rows from the image bottom upward; for the left side, the first row
    covered by a line lying left of the centre decides, and among the lines
    there it takes the one closest to the centre (same for the right). Deciding
    near the vehicle keeps curves out of it — far ends of curved lanes cross
    the centre. The detector extrapolates its lines down to the image border,
    so near rows are available for every line that reaches the near field.

    Known risk: if the ego line on one side is missed, the walk goes on up and
    can pick the neighbour lane's line (WWH-25 step 3: 4-10 % of sides).
    guard_ego checks the result.

    lanes : list of (N, 2) arrays of (x, y), any order
    Returns (left, right, y_left, y_right); a missing side is (None, None).
    """
    cx = image_width / 2.0
    lanes = [np.asarray(l, dtype=np.float64) for l in lanes if len(l) >= 2]
    if not lanes:
        return None, None, None, None
    spans = []
    for ln in lanes:
        o = np.argsort(ln[:, 1])
        spans.append((ln[o, 1], ln[o, 0]))
    y_hi = int(np.floor(max(ys[-1] for ys, _ in spans)))
    y_lo = int(np.ceil(min(ys[0] for ys, _ in spans)))
    picked = {"L": (None, None), "R": (None, None)}
    taken = {"L": None, "R": None}             # index picked per side: never both sides
    for y in range(y_hi, y_lo - 1, -1):
        for side in ("L", "R"):
            if picked[side][0] is not None:
                continue
            other = taken["R" if side == "L" else "L"]
            best, best_d = None, np.inf
            for i, (ys, xs) in enumerate(spans):
                if i == other or not ys[0] <= y <= ys[-1]:
                    continue
                x = float(np.interp(y, ys, xs))
                if (side == "L") != (x < cx):
                    continue
                if abs(x - cx) < best_d:
                    best, best_d = i, abs(x - cx)
            if best is not None:
                picked[side] = (lanes[best], y)
                taken[side] = best
        if picked["L"][0] is not None and picked["R"][0] is not None:
            break
    return picked["L"][0], picked["R"][0], picked["L"][1], picked["R"][1]


def _x_at(lane, y):
    ln = np.asarray(lane, dtype=np.float64)
    o = np.argsort(ln[:, 1])
    ys, xs = ln[o, 1], ln[o, 0]
    return float(np.interp(y, ys, xs)) if ys[0] <= y <= ys[-1] else None


def guard_ego(lanes, left, right, f_x, f_y, camera_height, image_width, image_height):
    """Reject an ego pair whose near-field width is not a lane width.

    At the nearest row both picked lines cover, the ground-plane width
    (x_R - x_L)·z/f_x with z = f_y·h/(y - cy) must lie in
    [_EGO_W_MIN, _EGO_W_MAX]. The failure it catches is pick_ego's known one —
    an ego line missed, so a line of the next lane (or a hallucinated one) was
    taken instead:

      too narrow: the side nearer the centre is suspect (a phantom inside the
                  lane). Replace it by the next line farther out on that side
                  if that makes the width plausible, else drop it.
      too wide:   the side farther from the centre is suspect (the neighbour
                  lane's line). Replace it by a line on that side nearer the
                  centre that gives a plausible width at the nearest row it
                  shares with the other side — pick_ego only compared lines
                  covering its own decision row, so the real ego line may start
                  higher up — else drop it.

    The width is checked only at rows no deeper than _GUARD_MAX_Z_FACTOR times
    the near-field window's far end; beyond that the pair is returned unchecked
    ("far_unchecked"). The bounds are the project's plausible lane-width range
    (openlane_module.convert_openlane WIDTH_MIN/MAX, the range WWH-21 proposes).

    Returns (left, right, reason): reason is "ok", "narrow_replaced",
    "narrow_dropped", "wide_replaced", "wide_dropped", "far_unchecked",
    "no_overlap" (the two lines share no row) or "one_side" (nothing to check).
    """
    if left is None or right is None:
        return left, right, "one_side"
    cx, cy = image_width / 2.0, image_height / 2.0
    z_max = _GUARD_MAX_Z_FACTOR * nearfield_window(f_y, camera_height, image_height)[1]

    def near_row(a, b):
        """Nearest image row both lines cover, or None."""
        y = min(np.max(np.asarray(a)[:, 1]), np.max(np.asarray(b)[:, 1]), image_height - 1.0)
        return y if _x_at(a, y) is not None and _x_at(b, y) is not None else None

    def depth(y):
        return f_y * camera_height / (y - cy) if y - cy > 1.0 else np.inf

    def width_at(l, r, y):
        return (_x_at(r, y) - _x_at(l, y)) * depth(y) / f_x

    y = near_row(left, right)
    if y is None:
        return left, right, "no_overlap"
    if depth(y) > z_max:
        return left, right, "far_unchecked"
    w = width_at(left, right, y)
    if _EGO_W_MIN <= w <= _EGO_W_MAX:
        return left, right, "ok"
    xl, xr = _x_at(left, y), _x_at(right, y)
    if w > _EGO_W_MAX:
        suspect_left = abs(xl - cx) > abs(xr - cx)
        kept = right if suspect_left else left
        suspect = left if suspect_left else right
        best, best_y = None, -np.inf
        for ln in lanes:
            if ln is left or ln is right:
                continue
            yc = near_row(ln, kept)
            if yc is None or depth(yc) > z_max or yc <= best_y:
                continue
            x = _x_at(ln, yc)
            if (x >= cx) if suspect_left else (x <= cx):
                continue                        # not on the suspect side
            xs = _x_at(suspect, yc)
            if xs is not None and abs(x - cx) >= abs(xs - cx):
                continue                        # not nearer the centre than the suspect
            l2, r2 = (ln, kept) if suspect_left else (kept, ln)
            if _EGO_W_MIN <= width_at(l2, r2, yc) <= _EGO_W_MAX:
                best, best_y = ln, yc
        if best is not None:
            return (best, right, "wide_replaced") if suspect_left else (left, best, "wide_replaced")
        return (None, right, "wide_dropped") if suspect_left else (left, None, "wide_dropped")

    def width(l, r):
        return width_at(l, r, y)

    suspect_left = abs(xl - cx) < abs(xr - cx)
    x_s = xl if suspect_left else xr
    alts = []
    for ln in lanes:
        x = _x_at(ln, y)
        if x is None or ln is left or ln is right:
            continue
        if (x < x_s) if suspect_left else (x > x_s):
            alts.append((abs(x - x_s), ln))
    for _, ln in sorted(alts, key=lambda a: a[0]):
        l2, r2 = (ln, right) if suspect_left else (left, ln)
        if _EGO_W_MIN <= width(l2, r2) <= _EGO_W_MAX:
            return l2, r2, "narrow_replaced"
        break                                  # only the next line out
    return (None, right, "narrow_dropped") if suspect_left else (left, None, "narrow_dropped")


def densify(lane):
    """Detector polyline at every integer row it spans (linear, no
    extrapolation). Returns (rows, x), rows ascending."""
    ln = np.asarray(lane, dtype=np.float64)
    o = np.argsort(ln[:, 1])
    ys, xs = ln[o, 1], ln[o, 0]
    rows = np.arange(np.ceil(ys[0]), np.floor(ys[-1]) + 1.0)
    return rows, np.interp(rows, ys, xs)


def _stripes(profile, lo, hi, k_max):
    """Bright stripes whose peak lies in [lo, hi] on a 1-D profile.

    A stripe is a local maximum that rises _RIDGE_THR above the darkest road
    on BOTH sides (probed _BG_PX wide, starting 2 px past the stripe-width
    bound k_max/2 from the peak — the min, like paint_evidence's base_far, so
    the gap of a double marking counts as road) and falls to the half level
    within k_max/2 + 2 px on both sides (a wider plateau is a bright surface,
    not paint). Returns [(x_left, x_right)] half-level extents, sub-pixel,
    merged when they overlap.
    """
    n = len(profile)
    half_bound = int(np.ceil(k_max / 2.0)) + 2
    out = []
    for p in range(max(lo, 1), min(hi, n - 2) + 1):
        v = profile[p]
        if v < profile[p - 1] or v < profile[p + 1]:
            continue
        a, b = p - half_bound - _BG_PX, p + half_bound + _BG_PX
        if a < 0 or b >= n:
            continue
        base = max(profile[a:p - half_bound].min(),
                   profile[p + half_bound + 1:b + 1].min())
        if v - base < _RIDGE_THR:
            continue
        level = 0.5 * (v + base)
        edges = []
        for step in (-1, 1):
            i = p
            while abs(i - p) <= half_bound and profile[i + step] > level:
                i += step
            if abs(i - p) > half_bound:
                break
            j = i + step                      # first sample at/below level
            f = (profile[i] - level) / (profile[i] - profile[j])
            edges.append(i + step * f)
        if len(edges) != 2:
            continue
        xl, xr = edges
        if out and xl <= out[-1][1]:          # same stripe (flat top)
            out[-1] = (min(out[-1][0], xl), max(out[-1][1], xr))
        else:
            out.append((xl, xr))
    return out


def refine_center(image_rgb, rows, x_prior, f_x, f_y, camera_height):
    """Per row: the centre of the painted marking nearest the detector prior,
    or the prior itself when the image shows no paint there.

    Stripes (see _stripes) within one _GROUP_M-wide extent are one marking and
    its centre is the centre of the whole group — the detector labels a double
    line by the pair's centre. The window is half a group plus _PRIOR_TOL_PX
    around the prior; the chosen marking is the one whose centre is nearest
    the prior. Widths come from the row depth with paint_evidence's grade
    margin (z_min), so they bound the stripe on any grade within ±15°.

    Returns (x, is_paint): x as float array, is_paint as bool array.
    """
    gray = _gray(image_rgb)
    h, w = gray.shape
    geom = CameraGeometry.without_lane_width(f_x, f_y, camera_height, w, h)
    rows = np.asarray(rows, dtype=np.float64)
    x_out = np.asarray(x_prior, dtype=np.float64).copy()
    paint = np.zeros(len(rows), dtype=bool)
    for i, (y, xp) in enumerate(zip(rows, x_out)):
        yi = int(round(y))
        if not 1 <= yi <= h - 2:
            continue
        z = geom.z_min(y)
        k_max = f_x * _STRIPE_M / z
        group_px = f_x * _GROUP_M / z
        s = min(int(np.ceil(group_px / 2.0)) + _PRIOR_TOL_PX, _WIN_CAP_PX)
        xi = int(round(xp))
        profile = gray[yi - 1:yi + 2, :].mean(axis=0)
        stripes = _stripes(profile, xi - s, xi + s, k_max)
        if not stripes:
            continue
        groups = [[stripes[0]]]
        for st in stripes[1:]:
            if st[1] - groups[-1][0][0] <= group_px:
                groups[-1].append(st)
            else:
                groups.append([st])
        centres = [0.5 * (g[0][0] + g[-1][1]) for g in groups]
        c = min(centres, key=lambda cc: abs(cc - xp))
        if abs(c - xp) <= s:
            x_out[i] = c
            paint[i] = True
    return x_out, paint


def apply_tail(rows, x, is_paint, tail):
    """`keep`: every row. `last_paint`: drop rows farther (smaller y) than the
    farthest paint row — the detector draws lines smoothly over a crest, and
    beyond the last verified paint the image cannot tell unpainted road from
    road hidden behind the crest (WWH-25 §6). Gaps between paint rows are kept
    either way. With no paint row at all, `last_paint` keeps nothing."""
    if tail == "keep":
        return rows, x, is_paint
    if tail != "last_paint":
        raise ValueError(f"tail must be 'keep' or 'last_paint', got {tail!r}")
    if not is_paint.any():
        empty = np.empty(0)
        return empty, empty, np.empty(0, dtype=bool)
    keep = rows >= rows[is_paint].min()
    return rows[keep], x[keep], is_paint[keep]


def model_curve(rows, x, is_paint):
    """{"y" ascending, "x", "src"} — lane_curve's format plus src (True =
    paint). None with < 2 rows, as lane_curve."""
    rows = np.asarray(rows, dtype=np.float64)
    if len(rows) < 2:
        return None
    o = np.argsort(rows)
    return {"y": rows[o], "x": np.asarray(x, dtype=np.float64)[o],
            "src": np.asarray(is_paint, dtype=bool)[o]}


def src_at(curve, ys):
    """Source flag of the curve row nearest each y (curve rows are integer
    rows, so this is exact for integer ys inside the curve)."""
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
    idx = np.clip(np.searchsorted(curve["y"], ys), 1, len(curve["y"]) - 1)
    left, right = curve["y"][idx - 1], curve["y"][idx]
    idx = np.where(ys - left <= right - ys, idx - 1, idx)
    return curve["src"][idx]
