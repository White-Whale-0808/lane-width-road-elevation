import numpy as np

from libs.inference.geometry import CameraGeometry

"""
Paint-evidence photometric checks (WWH-15).

Both downhill failure modes are the tracker following an edge that is NOT a
painted lane marking (crest: kerb / retaining wall / disconnected far road;
sag bottom: shadow boundary lying on the road). Every geometric detector
tried against them failed for the same reason: the quantities it tested were
themselves computed from the contaminated chain. These checks instead ask
the IMAGE whether each tracked location actually looks like paint — a signal
the failure cannot pollute.

A painted marking is a bright RIDGE of bounded width: intensity rises
immediately at the edge, stays high across the stripe (_STRIPE_M = 0.125 m
in CARLA), then falls back to road level (for a double marking, into the
gap between stripes).
Per probe location (x, y) on a 3-row-averaged horizontal profile I:

    peak      = max I[x + dir·(1.._PEAK_PX)]        bright side of the edge
    base_opp  = median I[x − dir·(2..7)]            road on the other side
    base_far  = min I[x + dir·(lo..hi)]             beyond the whole stripe
    ridge     = min(peak − base_opp, peak − base_far)

with [lo, hi] = [k+2, 3k+8] px and k = f_x·_STRIPE_M/z_min(y) the stripe-width
upper bound from projection geometry (z_min carries the ±15° grade margin,
see geometry.CameraGeometry). Real paint scores ~25+ gray levels; a shadow boundary fails
(one side all dark, the other a plateau that never drops), and so do kerbs
and walls (insufficient contrast against the road).

The peak window is deliberately FIXED at 1.._PEAK_PX: the probe sits on the
edge, so the bright side starts immediately regardless of stripe width.
Widening it with the geometric stripe bound breaks the check near the
horizon, where the grade margin inflates k to ~23 px and the window swallows
bright far-field surfaces.

Two layers use this score (validated on all three Town03 datasets):

1. filter_paint_segments — drops non-paint ELSED segments BEFORE lane
   tracking, direction-agnostic (a segment may be either edge of any
   stripe). With the shadow boundary gone the tracker re-seeds on the true
   white line and recovers the full visible range (sag case: MAE 1.3-1.6°
   → 0.17°, z_max 6 → 20 m).
2. truncate_at_evidence_break — walks the refined inner-chain points near
   to far, direction-aware (paint lies OUTWARD of the inner edge), and cuts
   the chain at the first _TRUNC_MIN_RUN consecutive failures: beyond a
   sustained evidence break (the crest occlusion boundary) nothing the
   chain caught is trustworthy. Contaminated crest tails fail 18-42 points
   in a row while normal frames show at most 1 isolated failure, so the
   run-length rule separates them cleanly (crest case: MAE 2.4-9.8° → < 0.8°).

Calibration needed: f_x, f_y and camera_height only. The one geometric
quantity these checks use is the row depth z_min(y) = f_y*h/(y-cy+grade),
so the lane width never enters — CameraGeometry is built via
without_lane_width(). This is why the guard stays valid on a camera whose
lane width is unknown or being estimated.
"""

_STRIPE_M = 0.125       # painted stripe real width (m), single stripe (CARLA)
_RIDGE_THR = 10.0       # min ridge score (gray levels); real paint ~25+
_PEAK_PX = 4            # bright side of an edge starts within this many px
_FAR_CAP_PX = 60        # drop-probe window cap (px)
_SEG_SAMPLES = 9        # probe points along one segment
_TRUNC_MIN_RUN = 5      # consecutive failing points = evidence break


def _gray(image_rgb):
    g = np.asarray(image_rgb, dtype=np.float32)
    if g.ndim == 3:
        g = g @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return g


def _drop_window(geom, y):
    """Probe window [lo, hi] px that starts past the whole stripe."""
    k = int(np.clip(np.ceil(geom.f_x * _STRIPE_M / geom.z_min(y)), 2, 40))
    return k + 2, min(3 * k + 8, _FAR_CAP_PX)


def filter_paint_segments(image_rgb, segments, f_x, f_y, camera_height):
    """Keep only ELSED segments that lie on a painted-marking edge.

    Direction-agnostic: probes both sides of each segment and keeps the
    better ridge score, then takes the median over _SEG_SAMPLES probe points
    along the segment. Uses the UNMASKED image (the mask boundary fabricates
    edges). Returns the kept subset as an (M, 4) ndarray.
    """
    segs = np.asarray(segments, dtype=np.float64)
    if segs.size == 0:
        return segs
    gray = _gray(image_rgb)
    h, w = gray.shape
    geom = CameraGeometry.without_lane_width(f_x, f_y, camera_height, w, h)

    keep = np.zeros(len(segs), dtype=bool)
    for si, (x1, y1, x2, y2) in enumerate(segs):
        vals = []
        for t in np.linspace(0.0, 1.0, _SEG_SAMPLES):
            x, y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            yi, xi = int(round(y)), int(round(x))
            if not (1 <= yi <= h - 2):
                continue
            lo, hi = _drop_window(geom, y)
            if not (hi + 1 <= xi <= w - hi - 2):
                continue
            strip = gray[yi - 1:yi + 2, :].mean(axis=0)
            best = -np.inf
            for d in (1, -1):
                peak = max(strip[xi + d * j] for j in range(1, _PEAK_PX + 1))
                base_opp = float(np.median([strip[xi - d * j] for j in range(2, 8)]))
                base_far = min(strip[xi + d * j] for j in range(lo, hi + 1))
                best = max(best, min(peak - base_opp, peak - base_far))
            vals.append(best)
        keep[si] = bool(vals) and float(np.median(vals)) >= _RIDGE_THR
    return segs[keep]


def truncate_at_evidence_break(image_rgb, points, is_left,
                               f_x, f_y, camera_height):
    """Cut refined inner-chain points at the first sustained paint-evidence
    failure, near to far. Direction-aware: paint lies OUTWARD of the inner
    edge.

    Points are ternary: pass / fail / neutral (too close to the border to
    probe — no evidence either way, they neither fail nor extend a run).
    Failing points BEFORE the first passing point are dropped individually:
    a failing run at the chain's near end is a local artifact (image-bottom
    dark strip, hood shadow), not an occlusion boundary, which by construction
    lies beyond verified road. From the first pass onward, the first _TRUNC_MIN_RUN
    consecutive failures cut the chain there. Returns the surviving points,
    near (large y) first.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return pts
    gray = _gray(image_rgb)
    h, w = gray.shape
    geom = CameraGeometry.without_lane_width(f_x, f_y, camera_height, w, h)
    out_sign = -1 if is_left else 1

    pts = pts[np.argsort(-pts[:, 1])]        # near (large y) first
    keep = np.ones(len(pts), dtype=bool)
    seen_pass = False
    run = 0
    cut = len(pts)
    for i, (x, y) in enumerate(pts):
        yi, xi = int(round(y)), int(round(x))
        lo, hi = _drop_window(geom, y)
        if not (1 <= yi <= h - 2 and hi + 1 <= xi <= w - hi - 2):
            run = 0
            continue                          # neutral
        strip = gray[yi - 1:yi + 2, :].mean(axis=0)
        peak = max(strip[xi + out_sign * j] for j in range(1, _PEAK_PX + 1))
        road = float(np.median([strip[xi - out_sign * j] for j in range(2, 8)]))
        far = min(strip[xi + out_sign * j] for j in range(lo, hi + 1))
        if min(peak - road, peak - far) < _RIDGE_THR:
            if not seen_pass:
                keep[i] = False               # leading failure: drop the point
                continue
            run += 1
            if run >= _TRUNC_MIN_RUN:
                cut = i - run + 1
                break
        else:
            seen_pass = True
            run = 0
    keep[cut:] = False
    return pts[keep]
