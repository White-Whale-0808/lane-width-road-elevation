import numpy as np

"""
Lane segmentation via near-to-far continuity tracking, with geometry-derived
thresholds.

Tracking: seed at the bottom (innermost rule), track upward band by band using
a local line model, re-seed past dead ends (junctions).

Thresholds come from the flat-ground pinhole model with camera height h:

    x - cx = f_x * X / z          z(y) = f_y * h / (y - cy)

so a lateral distance X has the pixel form px(X, y) = f_x * X / z(y), and a
lane line at lateral offset X has image slope dy/dx = f_y*h/(f_x*X). Association
tolerance, cross-lane cap, slope gates and model memory are all stated in metres
and projected per row. The lane WIDTH is not needed: every threshold is a lateral
distance, so the tracker runs on a road of unknown width. Camera geometry is
required; there is no un-calibrated mode.

Known facts to keep in mind before changing it:
  - There is no outer bound on seed / ROI x. Such bounds were written against
    z_min (±15° grade slack), so on a high camera (OpenLane) they never
    constrained anything; removing them left the output bit-identical. They do
    NOT prevent adjacent-lane seeding (5.5 % of dashed-lane OpenLane frames seed
    on the next lane, with or without them); a fix needs a bound derived from
    the flat-road depth (WWH-21).
  - The slope gate is load-bearing on urban footage (stop lines, crosswalks):
    disabling it loses whole frames on OpenLane. CARLA routes have almost no
    junctions, so it cannot be evaluated there.
  - z(y) is a flat-ground approximation; on the second plane it drifts, so
    (y - cy) is clamped and band-count fallbacks are kept as safety nets.
"""

# Every threshold below is a LATERAL DISTANCE IN METRES, projected to pixels at
# each row by geom.px_at / px_max_at (sweeping an assumed width 2.60-4.40 m left
# >= 95 % of frames bit-identical, so a fixed distance costs nothing).
_TOL_X_M             = 0.325  # association tol as a lateral distance
_TOL_PX_FLOOR        = 3.0    # ELSED endpoint noise floor (px)
_CROSS_LANE_FRACTION = 0.40   # association may not search beyond this fraction of
                              # the lane width measured this frame (_measure_lane_width_m)
_SLOPE_GATE_X_M      = 3.25   # noise slope gate: |dy/dx| of a line this far aside.
                              # The one lateral scale that cannot be measured: it runs
                              # before any line has been found. Load-bearing (see above).
_SEED_X_MAX          = 8.0    # seed slope gate: lines beyond 8 m lateral are noise
_NOISE_X_MAX         = 16.0   # prefilter slope gate: beyond 16 m lateral
_MODEL_MEMORY_M      = 4.0    # local model fits points within 4 m of depth
_RESET_GAP_M         = 2.0    # detection gap > 2 m => possible plane change
_SUPPORT_MIN_LEN_PX  = 60.0   # lone seed-only re-seeded segments shorter than this
                              # are isolated blobs (poles/hillside: 32-54 px;
                              # legit single-seg far sections: >= 77 px)

# Camera model (z_at / z_min / lane_px bounds) lives in geometry.py, shared with paint_evidence.
from libs.inference.geometry import CameraGeometry as _Geometry


def _segment_info(segments, geom):
    """Precompute per-segment geometry; drop only clearly-horizontal noise."""
    # slope gate: dy/dx of a line _SLOPE_GATE_X_M to the side (a lane line at
    # lateral X has image slope f_y*h/(f_x*X)). Anything flatter cannot be a
    # lane line under normal driving — filters stop lines and crosswalks,
    # which have dy/dx ≈ 0.
    slope_gate = geom.f_y * geom.h / (geom.f_x * _SLOPE_GATE_X_M)

    infos = []
    for seg in np.asarray(segments, dtype=np.float64):
        x1, y1, x2, y2 = seg
        if x2 == x1:
            slope = np.inf  # vertical: perfectly valid lane candidate
        else:
            slope = (y2 - y1) / (x2 - x1)
        mid_y = (y1 + y2) / 2

        mid_x = (x1 + x2) / 2

        if np.isfinite(slope) and abs(slope) < slope_gate:
            continue

        infos.append({
            "seg": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            "p1": (x1, y1), "p2": (x2, y2),
            "slope": slope,
            "mid_x": mid_x,
            "mid_y": mid_y,
            "y_min": min(y1, y2),
            "y_max": max(y1, y2),
        })
    return infos


def _x_on_segment_line(info, y):
    """x of the segment's own infinite line at height y."""
    x1, y1 = info["p1"]
    if not np.isfinite(info["slope"]) or info["slope"] == 0:
        return info["mid_x"]
    return x1 + (y - y1) / info["slope"]


def _fit_x_of_y(points, geom, y_now, last_n=8):
    """Fit x = a*y + b on the lane's recent direction.

    Normally the model spans points within _MODEL_MEMORY_M metres of the
    current row's depth — one road section, not the whole lane. Above the
    flat-ground validity limit z(y) saturates and cannot select a depth
    window, so the most recent `last_n` endpoints stand in.
    """
    pts = None
    if y_now is not None and geom.z_valid(y_now):
        z_now = geom.z_at(y_now)
        recent = [p for p in points
                  if geom.z_valid(p[1]) and z_now - geom.z_at(p[1]) <= _MODEL_MEMORY_M]
        if len(recent) >= 4:
            pts = recent
    if pts is None:
        pts = points[-last_n:]

    ys = np.array([p[1] for p in pts])
    xs = np.array([p[0] for p in pts])
    if len(pts) < 2 or np.ptp(ys) < 1e-6:
        return None
    a, b = np.polyfit(ys, xs, 1)
    return a, b


def _find_seed(infos, selected, is_left, center_x,
               band_edges, search_from_band, geom, first_seed=True):
    """Lowest band (index <= search_from_band) with an innermost seed group."""
    sign = -1.0 if is_left else 1.0

    # slope gate: a lane-parallel line within X m lateral distance.
    # The first seed sits on the ego plane where the flat model holds
    # (X_max = 8 m); re-seeds may land on the second plane whose lines
    # are flatter, so only the basic noise gate (16 m) is applied.
    x_gate = _SEED_X_MAX if first_seed else _NOISE_X_MAX
    seed_slope_gate = geom.f_y * geom.h / (geom.f_x * x_gate)

    def is_seed_candidate(info):
        s = info["slope"]
        if not np.isfinite(s):
            return True  # vertical is fine on either side near the camera
        on_side = info["mid_x"] < center_x if is_left else info["mid_x"] > center_x
        if not (on_side and s * sign > 0):
            return False
        if abs(s) < seed_slope_gate:
            return False
        # Do not gate re-seeds by direction consistency with the dying track:
        # any tolerance tight enough to block kerbs / stop lines also rejects
        # legitimate re-seeds at crest plane changes (tried, 60+ frames regressed).
        return True

    for i in range(search_from_band, -1, -1):
        lo, hi = band_edges[i], band_edges[i + 1]
        y_c = (lo + hi) / 2

        # Above the flat-ground validity limit the seed x-window is derived
        # from a collapsed z_min and spans half the image; poles and hillside
        # edges get seeded there. Never SEED in that region (association may
        # still track into it from below).
        if not geom.z_valid(y_c):
            continue

        # The seed must lie on this side of the camera axis — the whole
        # constraint (no outer bound; see the module docstring before adding one).
        group_tol = max(_TOL_PX_FLOOR, geom.px_max_at(_TOL_X_M, y_c))

        cands = []
        for info in infos:
            if info["seg"] in selected:
                continue
            if not (info["y_max"] >= lo and info["y_min"] <= hi):
                continue
            if not is_seed_candidate(info):
                continue
            x_c = _x_on_segment_line(info, y_c)
            if (x_c > center_x) if is_left else (x_c < center_x):
                continue
            cands.append((x_c, info))
        if cands:
            # innermost + tolerance
            if is_left:
                best = max(c[0] for c in cands)
                picked = [inf for x, inf in cands if x >= best - group_tol]
            else:
                best = min(c[0] for c in cands)
                picked = [inf for x, inf in cands if x <= best + group_tol]
            return picked, i
    return None, -1


def _measure_lane_width_m(infos, center_x, band_edges, track_bands, geom):
    """Lane width in METRES, measured from the first seed on each side.

    The seed rule is innermost-first and needs no width of its own, so this is
    a bootstrap WITHIN one frame — seed, measure, then track — not a
    cross-frame feedback loop, and it cannot diverge.

    Driving in lane puts the camera between the two markings, so the width is
    the sum of the two lateral offsets, each evaluated at its OWN seed row.
    Summing per-side offsets instead of differencing two x values at a shared
    row is what removes the extrapolation: neither side is ever evaluated at a
    row its own seed does not cover. One side alone gives 2x its offset, the
    ego being roughly centred.

    Returns None when neither side seeds. The caller then leaves the
    cross-lane cap OFF rather than substituting an assumed width — a guess
    would be exactly the failure this function exists to remove.
    """
    halves = []
    for is_left in (True, False):
        items, band = _find_seed(infos, {}, is_left, center_x, band_edges,
                                 track_bands - 1, geom, True)
        if items is None:
            continue
        y_c = (band_edges[band] + band_edges[band + 1]) / 2.0
        if not geom.z_valid(y_c):
            continue
        xs = [_x_on_segment_line(i, y_c) for i in items]
        x_inner = max(xs) if is_left else min(xs)
        halves.append(abs(x_inner - center_x) * geom.z_at(y_c) / geom.f_x)
    if not halves:
        return None
    return sum(halves) if len(halves) == 2 else 2.0 * halves[0]


def _track_side(infos, is_left, center_x, track_bands, geom, lane_m):
    """Seed at the bottom, track upward by continuity; re-seed past dead ends."""
    sign = -1.0 if is_left else 1.0
    y_lo = min(i["y_min"] for i in infos)
    y_hi = max(i["y_max"] for i in infos)
    if y_hi - y_lo < 1:
        return []
    band_edges = np.linspace(y_lo, y_hi, track_bands + 1)

    def band_overlap(info, lo, hi):
        return info["y_max"] >= lo and info["y_min"] <= hi

    def assoc_window(y, missed):
        """Search half-width around the prediction at row y.

        The uphill worst-case bound keeps the window wide enough on
        ascending rows; the cross-lane cap uses the flat-ground scale,
        preserving the 4x headroom between the two.
        """
        # tolerance scale: worst-case uphill (largest pixel scale there);
        # cross-lane cap: worst-case flat (other lane closest in pixels).
        base = max(_TOL_PX_FLOOR, geom.px_max_at(_TOL_X_M, y))
        if lane_m is None:
            return base * (1.0 + missed)      # unmeasurable: no cap, no guess
        cap = max(_TOL_PX_FLOOR, geom.px_at(_CROSS_LANE_FRACTION * lane_m, y))
        return min(cap, base * (1.0 + missed))

    def group_tol(y):
        """Same-physical-marking grouping width."""
        return max(_TOL_PX_FLOOR, geom.px_max_at(_TOL_X_M, y))

    selected = {}
    sections = []  # per seed: {"segs": [...], "first": bool, "extra_bands": int}
    search_from_band = track_bands - 1
    first_seed = True

    while search_from_band >= 0:
        seed_items, seed_band = _find_seed(
            infos, selected, is_left, center_x,
            band_edges, search_from_band, geom, first_seed)
        if seed_items is None:
            break
        section = {"segs": [], "first": first_seed, "extra_bands": 0}
        sections.append(section)
        first_seed = False

        track_points = []  # accepted (x, y) endpoints, bottom -> top
        for info in seed_items:
            selected[info["seg"]] = True
            section["segs"].append(info["seg"])
            for p in (info["p1"], info["p2"]):
                track_points.append(p)
        track_points.sort(key=lambda p: -p[1])  # by y descending (near first)
        last_accept_y = max(p[1] for p in track_points)

        # --- track upward from the seed band ---
        missed = 0
        stop_band = -1  # band index where this track gave up (-1: reached top)
        for i in range(seed_band - 1, -1, -1):
            lo, hi = band_edges[i], band_edges[i + 1]
            y_c = (lo + hi) / 2

            model = _fit_x_of_y(track_points, geom, y_c)
            if model is None:
                stop_band = i
                break
            a, b = model
            x_pred = a * y_c + b
            tol = assoc_window(y_c, missed)

            accepted = []
            for info in infos:
                if info["seg"] in selected or not band_overlap(info, lo, hi):
                    continue
                # a lane line on this side can never have the opposite slope
                # sign (seeds already enforce this; kerb and pole segments
                # were slipping in through association)
                if np.isfinite(info["slope"]) and info["slope"] * sign <= 0:
                    continue
                x_c = _x_on_segment_line(info, y_c)
                if abs(x_c - x_pred) > tol:
                    continue
                # never cross the centre line
                if is_left and x_c > center_x:
                    continue
                if not is_left and x_c < center_x:
                    continue
                accepted.append((abs(x_c - x_pred), x_c, info))

            if not accepted:
                missed += 1
                if missed > max(4, track_bands // 3):
                    stop_band = i
                    break
                continue

            # Plane-change reset: after a real gap the lane has likely kinked;
            # old points would drag the model toward the previous plane.
            # (above the flat-ground limit z(y) saturates and no depth gap can
            # be computed — fall back to counting missed bands)
            if geom.z_valid(y_c) and geom.z_valid(last_accept_y):
                gap_m = geom.z_at(y_c) - geom.z_at(last_accept_y)
                do_reset = gap_m > _RESET_GAP_M
            else:
                do_reset = missed >= 2
            if do_reset and missed > 0:
                track_points = track_points[-2:]
            missed = 0

            # keep everything close to the best match (parallel double markings
            # within the grouping width are the same physical lane)
            accepted.sort(key=lambda t: t[0])
            best_x = accepted[0][1]
            gtol = group_tol(y_c)
            for _, x_c, info in accepted:
                if abs(x_c - best_x) <= gtol:
                    selected[info["seg"]] = True
                    section["segs"].append(info["seg"])
                    for p in (info["p1"], info["p2"]):
                        track_points.append(p)
            section["extra_bands"] += 1
            last_accept_y = y_c

        if stop_band < 0:
            break  # reached the top of the road area
        # Re-seed strictly above the failure point for the next road section.
        search_from_band = stop_band - 1

    # A re-seeded section whose track never accepted a band beyond its own
    # seed group AND that is a lone short segment is an isolated blob (pole,
    # hillside edge, mask hole), not a road section. Multi-segment or long
    # seed-only sections are legitimate far lane sections that fit entirely
    # inside one seed group.
    kept = []
    for sec in sections:
        if sec["first"] or sec["extra_bands"] > 0 or len(sec["segs"]) >= 2:
            kept.extend(sec["segs"])
            continue
        x1, y1, x2, y2 = sec["segs"][0]
        if np.hypot(x2 - x1, y2 - y1) >= _SUPPORT_MIN_LEN_PX:
            kept.extend(sec["segs"])
    return kept


def split_left_right_lines(
    segments,
    image_width: int,
    img_height: int,
    track_bands: int = 16,
    *,
    f_x: float,
    f_y: float,
    camera_height: float,
):
    """Split ELSED segments into inner left / right lane segments.

    Parameters
    ----------
    segments : array-like, shape (N, 4)
        Raw ELSED output: each row is (x1, y1, x2, y2).
    image_width, img_height : int
    track_bands : int
        Tracking band count (internally clamped to >= 16). Independent of
        lane_fitting's num_bands — tracking steps and fitting knots are
        separate concepts.
    f_x, f_y, camera_height : float, keyword-only, REQUIRED
        Camera focal lengths (px) and camera height above road (m).
        Association tolerance, seed window, slope gates and model memory are
        all derived from these; there is no un-calibrated mode. The lane WIDTH
        is deliberately not taken: every threshold here is a lateral distance
        in metres, so the tracker works on a road whose width it does not know.

    Returns
    -------
    inner_left, inner_right : list of (x1, y1, x2, y2) tuples
    """
    segments = np.asarray(segments)
    if segments.size == 0:
        return [], []

    geom = _Geometry.without_lane_width(f_x, f_y, camera_height,
                                        image_width, img_height)

    infos = _segment_info(segments, geom)
    if not infos:
        return [], []

    center_x = image_width / 2
    track_bands = max(int(track_bands), 16)

    # Measure this road's lane width from the seeds before tracking, so the
    # cross-lane cap scales with the road in front of the camera rather than
    # with a number carried over from whichever dataset it was tuned on.
    y_lo = min(i["y_min"] for i in infos)
    y_hi = max(i["y_max"] for i in infos)
    lane_m = None
    if y_hi - y_lo >= 1:
        band_edges = np.linspace(y_lo, y_hi, track_bands + 1)
        lane_m = _measure_lane_width_m(infos, center_x, band_edges,
                                       track_bands, geom)

    inner_left = _track_side(infos, True, center_x, track_bands, geom, lane_m)
    inner_right = _track_side(infos, False, center_x, track_bands, geom, lane_m)

    return inner_left, inner_right
