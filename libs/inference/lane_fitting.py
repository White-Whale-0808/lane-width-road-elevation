import numpy as np

# Segment shadowing: a segment consistently more OUTER than a parallel kept
# segment (by > margin px over >= min-overlap rows) is an outer paint edge or
# the outer line of a double marking. It never defines the inner envelope —
# where the inner line is absent (dashed gaps, image bottom) the envelope must
# yield a GAP, not fall back outward: fallback steps in x_inner(y) become width
# steps and blow up the pitch at the ends.
_SHADOW_MARGIN_PX = 3.0
_SHADOW_MIN_OVERLAP_ROWS = 8

# Fragment split: a 1-row x step in the envelope beyond the steepest legitimate
# lane slope means the envelope switched to a different segment — fragment
# boundary. Measured, not derived: the ego lane's inner edge never exceeds
# ~3.3 px/row on these datasets (the geometric bound, ~16 px/row, is too loose).
_FRAG_MAX_STEP_PX = 4.0

# Junction consistency: bridging from one fragment to the next must stay within
# this off-direction displacement (bridge slope vs either fragment's own end
# slope, times the gap dy). Legitimate bridges stay under 20 px; the bottom
# outer-edge fallback lands 27-93 px outward.
_JUNCTION_TOL_PX = 20.0

# End slopes near a junction use up to this many row pairs of the fragment.
_JUNCTION_SLOPE_ROWS = 10

# Edge refinement: search window (px) around the ELSED-derived envelope x.
# Small on purpose — the prior is already on the edge, and on a double
# marking the far-range gap between the two stripes shrinks to a few px;
# a wider window would let the OUTER stripe's flank (same gradient sign)
# capture the point.
_REFINE_SEARCH_PX = 3

# Minimum |dI/dx| (gray levels / px, 3-row averaged) to accept a refined
# edge. Paint-road contrast in CARLA is ~25+; below this there is no clear
# edge under the window (mask boundary, washed-out far range) and the
# ELSED prior is kept unchanged.
_REFINE_MIN_GRAD = 4.0


def inner_chain_points(segments, is_left, return_debug=False):
    """Clean inner-lane-line points for one side, from tracked segments.

    The lane width (measured per frame in pitch_estimation) is inner edge to
    inner edge, but the tracker deliberately keeps the whole marking group (all
    parallel paint edges — evidence for tracking). This recovers the inner edge:

    1. Per-row inner envelope (no tunable): for every image row y covered by
       at least one segment, x_inner(y) = innermost x among the covering
       segments (max x on the left side, min x on the right). Rows with no
       coverage stay gaps — nothing is fabricated, and density is NOT shaped
       here: the lane curve is a continuous model and pitch_estimation
       resamples widths z-uniformly on its own.
    2. Fragments: the envelope splits at coverage gaps and at 1-row x steps
       beyond _FRAG_MAX_STEP_PX (envelope switched segments).
    3. Junction consistency: a bridge between adjacent fragments whose
       direction deviates from both fragments' own end directions by more
       than _JUNCTION_TOL_PX (scaled by the gap dy) marks a break; the
       largest consistent fragment group (by row count) survives. One rule
       drops both mid-chain off-lane runs (next to a dashed gap) and bottom
       outer-edge fallback rows (inner segment ending above the image bottom).

    Returns an (N, 2) array of (x, y) points, near (large y) first.
    With return_debug=True returns (points, dbg) where dbg exposes the
    intermediates for step visualization: shadowed / kept segments, envelope
    rows, fragments, junction groups and the index of the surviving group.
    """
    def _ret(points, dbg):
        return (points, dbg) if return_debug else points

    dbg = {"shadowed": np.empty((0, 4)), "kept_segments": np.empty((0, 4)),
           "rows": [], "fragments": [], "groups": [], "best_group": -1}
    segs = np.asarray(segments, dtype=np.float64)
    if segs.size == 0:
        return _ret(np.empty((0, 2)), dbg)
    y_lo = np.minimum(segs[:, 1], segs[:, 3])
    y_hi = np.maximum(segs[:, 1], segs[:, 3])

    def x_at(seg, y):
        x1, sy1, x2, sy2 = seg
        return (x1 + x2) / 2 if sy2 == sy1 else x1 + (y - sy1) * (x2 - x1) / (sy2 - sy1)

    # 0. shadowing: exclude outer paint edges / outer double-marking lines so
    # the envelope never falls back outward where the inner line is absent
    inner_sign = 1.0 if is_left else -1.0
    keep = np.ones(len(segs), dtype=bool)
    for i in range(len(segs)):
        for j in range(len(segs)):
            if i == j:
                continue
            lo = max(y_lo[i], y_lo[j])
            hi = min(y_hi[i], y_hi[j])
            if hi - lo < _SHADOW_MIN_OVERLAP_ROWS:
                continue
            d_lo = (x_at(segs[j], lo) - x_at(segs[i], lo)) * inner_sign
            d_hi = (x_at(segs[j], hi) - x_at(segs[i], hi)) * inner_sign
            if d_lo > _SHADOW_MARGIN_PX and d_hi > _SHADOW_MARGIN_PX:
                keep[i] = False
                break
    if keep.any():
        dbg["shadowed"] = segs[~keep]
        segs, y_lo, y_hi = segs[keep], y_lo[keep], y_hi[keep]
    dbg["kept_segments"] = segs

    # 1. dense per-row inner envelope, bottom (near) row first
    rows = []  # (y, x_inner), y descending
    for y in range(int(np.floor(y_hi.max())), int(np.ceil(y_lo.min())) - 1, -1):
        xs = []
        for (x1, y1, x2, y2), lo, hi in zip(segs, y_lo, y_hi):
            if not (lo <= y <= hi):
                continue
            xs.append((x1 + x2) / 2 if y1 == y2
                      else x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        if xs:
            rows.append((float(y), max(xs) if is_left else min(xs)))
    dbg["rows"] = rows
    if not rows:
        return _ret(np.empty((0, 2)), dbg)

    # 2. fragments: contiguous runs with per-row x steps within the slope gate
    frags = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        if prev[0] - cur[0] == 1 and abs(cur[1] - prev[1]) <= _FRAG_MAX_STEP_PX:
            frags[-1].append(cur)
        else:
            frags.append([cur])

    # 3. junction consistency -> keep the largest consistent fragment group
    def _end_slope(frag, head):
        """Median dx-per-row near one fragment end; None for a single row."""
        pairs = list(zip(frag, frag[1:]))
        if not pairs:
            return None
        pairs = pairs[:_JUNCTION_SLOPE_ROWS] if head else pairs[-_JUNCTION_SLOPE_ROWS:]
        slopes = sorted((b[1] - a[1]) / (a[0] - b[0]) for a, b in pairs)
        return slopes[len(slopes) // 2]

    all_slopes = sorted((b[1] - a[1]) / (a[0] - b[0])
                        for f in frags for a, b in zip(f, f[1:]))
    global_med = all_slopes[len(all_slopes) // 2] if all_slopes else 0.0

    groups = [[frags[0]]]
    for prev_frag, frag in zip(frags, frags[1:]):
        dy = prev_frag[-1][0] - frag[0][0]  # >= 1
        bridge = (frag[0][1] - prev_frag[-1][1]) / dy
        refs = [s for s in (_end_slope(prev_frag, head=False),
                            _end_slope(frag, head=True)) if s is not None]
        if not refs:
            refs = [global_med]
        # consistent with EITHER side is enough: across a two-plane hinge the
        # bridge matches the far side long before it matches the near side
        dev = min(abs(bridge - r) for r in refs) * dy
        if dev > _JUNCTION_TOL_PX:
            groups.append([])
        groups[-1].append(frag)
    best = max(groups, key=lambda g: sum(len(f) for f in g))
    dbg["fragments"] = frags
    dbg["groups"] = groups
    dbg["best_group"] = groups.index(best)

    return _ret(np.array([(x, y) for f in best for y, x in f]), dbg)

def refine_inner_points(image_rgb, points, is_left):
    """Sub-pixel edge refinement of inner-chain points on the ORIGINAL image.

    inner_chain_points is limited by ELSED's straight-segment approximation
    (its information ceiling is the segment endpoints). This snaps each point
    to the actual paint→road transition: sample a 3-row-averaged horizontal
    gradient profile around the prior x, take the strongest gradient of the
    expected sign (inner edge of the LEFT marking is bright→dark going right,
    of the RIGHT marking dark→bright), and localize it to sub-pixel with a
    parabola fit over the peak's 3 gradient samples.

    Uses the unmasked image — the road mask boundary creates fake gradients
    and can cut through markings. Points with no clear edge in the window
    (|dI/dx| < _REFINE_MIN_GRAD) or too close to the border keep their prior.

    Parameters
    ----------
    image_rgb : PIL.Image or ndarray (H, W, 3), RGB
    points : (N, 2) ndarray of (x, y) from inner_chain_points
    is_left : bool

    Returns a new (N, 2) ndarray; y values are unchanged.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return pts
    gray = np.asarray(image_rgb, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    height, width = gray.shape

    sign = -1.0 if is_left else 1.0  # expected dI/dx at the inner edge
    s = _REFINE_SEARCH_PX
    refined = pts.copy()
    for i, (x, y) in enumerate(pts):
        yi, xi = int(round(y)), int(round(x))
        if not (1 <= yi <= height - 2 and s + 1 <= xi <= width - s - 2):
            continue
        strip = gray[yi - 1:yi + 2, xi - s - 1:xi + s + 2].mean(axis=0)
        grad = sign * 0.5 * (strip[2:] - strip[:-2])  # grad[m] ↔ column xi-s+m
        # nearest qualifying local max, not the strongest: the prior is
        # already on the edge, and a stronger same-sign edge in the window
        # (e.g. the other stripe of a double marking at far range) must not
        # capture the point
        k, best = None, None
        for m in range(len(grad)):
            if grad[m] < _REFINE_MIN_GRAD:
                continue
            if m > 0 and grad[m] < grad[m - 1]:
                continue
            if m < len(grad) - 1 and grad[m] < grad[m + 1]:
                continue
            key = (abs(m - s), -grad[m])
            if best is None or key < best:
                best, k = key, m
        if k is None:
            continue
        delta = 0.0
        if 0 < k < len(grad) - 1:
            denom = grad[k - 1] - 2 * grad[k] + grad[k + 1]
            if abs(denom) > 1e-9:
                delta = float(np.clip(0.5 * (grad[k - 1] - grad[k + 1]) / denom,
                                      -0.5, 0.5))
        refined[i, 0] = xi - s + k + delta
    return refined


# Depth-jump truncation: paired-row depth (one value per row where BOTH chains
# have a real point — no continuity assumption) can only jump between adjacent
# measured rows when the surface in between is hidden (crest occlusion): the jump
# is the hidden distance, 10-30 m on the Town03 crests, while continuous road
# moves <= ~2 m per paired row. Guards the one case the photometric gate cannot:
# real paint beyond the crest that is not the same continuous lane line.
#
# Depth is in LANE WIDTHS (z/w = f_x/width), so no lane width is needed; the
# absolute floor is 3.0 m on a 3.25 m lane, expressed as 3.0/3.25 lane widths.
_ZJUMP_ABS_LANES = 3.0 / 3.25   # minimum jump that can never be continuous road
_ZJUMP_FRAC = 0.3     # relative gate: dz > 0.3*z at larger depths
# Continuous-road bound across a ROW GAP: extrapolating the local plane
# (height matched at the near pair) to the far row gives the depth a
# continuous visible surface would reach — z_exp = z1*(y1-cy)/(y2-cy), the
# f_y cancels. A visible continuous surface stays near it; an occlusion jump
# far exceeds it (without this a legitimate long pairing gap trips the floor).
_ZJUMP_EXTRAP_FACTOR = 1.5


def truncate_at_depth_jump(left_points, right_points, f_x, image_height):
    """Cut BOTH chains at the first paired-row depth jump (crest occlusion).

    Rows where both chains have a point give independent per-row depths, in
    lane widths (see _ZJUMP_ABS_LANES); walking near to far, a jump that
    exceeds BOTH the continuity gate
    max(_ZJUMP_ABS_LANES, _ZJUMP_FRAC*z) AND _ZJUMP_EXTRAP_FACTOR times the
    local-plane extrapolation marks a hidden interval — everything beyond
    belongs to a disconnected road section and must not be joined to the
    near chain (lane_curve would bridge it). Both chains keep only rows at
    or below the near side of the jump. Returns (left_points, right_points),
    unchanged when no jump.
    """
    lp = np.asarray(left_points, dtype=np.float64)
    rp = np.asarray(right_points, dtype=np.float64)
    if lp.ndim != 2 or rp.ndim != 2 or len(lp) == 0 or len(rp) == 0:
        return lp, rp
    cy = image_height / 2.0
    xl = {int(round(y)): x for x, y in lp}
    xr = {int(round(y)): x for x, y in rp}
    pairs = []                                    # (row, z in lane widths), near first
    for y in sorted(set(xl) & set(xr), reverse=True):
        width = xr[y] - xl[y]
        if width > 0:
            pairs.append((y, f_x / width))
    for (y1, z1), (y2, z2) in zip(pairs, pairs[1:]):
        if z2 - z1 <= max(_ZJUMP_ABS_LANES, _ZJUMP_FRAC * z1):
            continue
        if y2 - cy <= 1e-6:
            continue      # far row at/above the local horizon: no bound, stay put
        z_exp = z1 * (y1 - cy) / (y2 - cy)
        if z2 <= _ZJUMP_EXTRAP_FACTOR * z_exp:
            continue      # reachable by a continuous surface across the row gap
        # keep the near side of the jump only; rows above y1 (including
        # unpaired single-side rows inside the gap) are unverified
        return lp[lp[:, 1] >= y1], rp[rp[:, 1] >= y1]
    return lp, rp


def lane_curve(points):
    """Continuous image-space lane-line curve from inner-chain points.

    The polyline through the chain points IS the lane-line model: chain gaps
    (dashed markings, shadow-excluded rows) are bridged linearly between the
    neighbouring points, and kinks (two-plane hinge) are preserved exactly.
    Smoothing is deliberately NOT done here — the pitch spline downstream
    owns smoothing, with its physically-motivated 1/z² weights.

    Returns {"y": ascending ndarray, "x": ndarray}, or None with < 2 points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return None
    order = np.argsort(pts[:, 1])
    return {"y": pts[order, 1], "x": pts[order, 0]}


def curve_x_at(curve, ys):
    """x of the lane curve at ys (scalar or array). Callers must stay inside
    [curve.y[0], curve.y[-1]] — np.interp clamps, it never extrapolates."""
    return np.interp(ys, curve["y"], curve["x"])
