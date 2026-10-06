import numpy as np
from scipy.stats import theilslopes
from scipy.interpolate import UnivariateSpline

# Windowed-estimator defaults, exposed for the step visualization: the local
# Theil-Sen window is |z - zc| <= max(WINDOW_MIN_M, WINDOW_FRAC * zc).
WINDOW_FRAC = 0.15
WINDOW_MIN_M = 1.0

# ---------------------------------------------------------------- near field
# The window is derived from the camera (f_y·h), not hand-set:
#
#     z(y) = f_y·h / (y − cy)      bottom row  ->  z_bottom = f_y·h / (H/2)
#
# Lower bound: a little past the bottom row. Upper bound, the nearer of:
#   curvature — the anchor assumes the road ahead is the wheel plane; a vertical
#               curve of radius R gives a relative depth error z²/(2·R·h), so
#               z_hi <= sqrt(2·R·h·tol)
#   sampling  — the window holds f_y·h·(1/z_lo − 1/z_hi) rows; the intercept
#               needs enough of them
# When they do not intersect sampling wins and `curvature_ok` is False: that
# camera's anchor carries a curvature bias (OpenLane: window 8.22–10.11 m, ~1.6 %).
NEARFIELD_Z_LO_FACTOR = 1.2
NEARFIELD_MIN_ROWS = 40
NEARFIELD_CURV_R_M = 1500.0    # worst-case vertical curve (OpenLane 60 m rise p90)
NEARFIELD_CURV_TOL = 0.005     # tolerable anchor depth error from that curvature
NEARFIELD_Z_HI_FACTOR_CAP = 3.0
# Quality gate: rejects a merely noisy fit — too few rows, too short a z span,
# residual scatter beyond ~2 px of edge noise (1 px is z/f_x metres) — and a
# near field off the wheel plane. The camera sits level to within body pitch
# (OpenLane extrinsics −0.3…+0.8°), so |θ0| > 1° means the window lies on a
# grade change, where the intercept is biased about 4 % per degree of θ0
# (WWH-27: per-frame width error vs θ0 corr −0.90 on CARLA, −0.48 on OpenLane).
# ⚠ It cannot reject a confidently wrong fit (OpenLane tram tracks: residual MAD
# below a healthy frame's, width 1.58 m vs 3.17 true). Catch that in line selection.
NEARFIELD_MIN_POINTS = 8
NEARFIELD_MIN_SPAN_FRAC = 0.5
NEARFIELD_RESID_PX = 2.0
NEARFIELD_THETA0_MAX_DEG = 1.0
# Adopt policy: a width is adopted only after passes span NEARFIELD_MIN_RUN_M of
# travel — at a curvature inflection an isolated frame passes with its bias at
# its largest. A run survives one failed frame (a flickering near field is a
# dropped observation; two failures in a row mean the gate really closed).
# Without `advance_to` the run is judged on frame count alone.
NEARFIELD_MIN_RUN_M = 0.5
NEARFIELD_MIN_RUN_FRAMES = 2       # a run is at least two frames, whatever the spacing
NEARFIELD_RUN_MAX_DROPOUT = 1      # consecutive failed frames a run survives
# Pooling: one frame's intercept carries the line-position drift across the
# window, amplified ~5x by the extrapolation to z=0 (WWH-26), and that error is
# largely independent from frame to frame. Once the sequence holds
# NEARFIELD_POOL_MIN_N passing estimates, the width is the median of the last
# NEARFIELD_POOL_K of them, and the run policy above no longer applies (the
# median already outvotes an isolated inflection-point pass). The minimum keeps
# one or two early wrong values from being held as a "median".
# K counts passes, not metres: 40 passes span ~33 m on OpenLane, ~8 m on CARLA.
# A metre cap was worse on OpenLane (too few passes left); an unbounded pool
# fails where the width really changes (CARLA full_road 3.29 / 3.55 m: 6.5 %).
NEARFIELD_POOL_K = 40
NEARFIELD_POOL_MIN_N = 3
# Hold policy: within one continuous sequence the last adopted value is held
# however stale, and its age is reported (`hold_m` / `hold_frames`). Before the
# first adoption there is no value (`status` "no_anchor"); the pipeline then uses
# the flagged last-resort constant or outputs no pitch. The calibrator knows no constant.


def back_project_widths(widths, f_x, f_y, image_height, w_real):
    """Back-project (y_pixel, pixel_width) pairs to camera-frame (z, Y_3d).

    Inverse perspective: z = f_x·w_real/width, Y_3d = -z·(y_pixel - cy)/f_y
    with cy = image_height/2. Non-positive widths are unphysical (lane fits
    crossing near the vanishing point) and are dropped; no other filtering.
    Canonical implementation shared by the estimator and the debug plots.

    Returns (z, Y_3d) 1-D arrays, empty on malformed/empty input.
    """
    widths = np.asarray(widths)
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        return np.array([]), np.array([])
    widths = widths[widths[:, 1] > 0]
    if len(widths) == 0:
        return np.array([]), np.array([])
    z = f_x * w_real / widths[:, 1]
    Y_3d = -z * (widths[:, 0] - image_height / 2) / f_y
    return z, Y_3d


def nearfield_window(f_y, camera_height, image_height):
    """Near-field anchor window (z_lo, z_hi, curvature_ok) for this camera.

    See the NEARFIELD_* constants for the derivation. `curvature_ok` is False
    when the sampling requirement forces the window past the depth at which the
    support-plane assumption holds to NEARFIELD_CURV_TOL — the estimate is
    still the best available on that camera, but it carries a known bias.

        CARLA    f_y·h =  491  ->  z ∈ [2.30, 4.02] m, curvature_ok=True
        OpenLane f_y·h = 1753  ->  z ∈ [8.22, 10.11] m, curvature_ok=False
    """
    z_bottom = f_y * camera_height / (image_height / 2.0)
    z_lo = NEARFIELD_Z_LO_FACTOR * z_bottom
    z_curv = float(np.sqrt(2.0 * NEARFIELD_CURV_R_M * camera_height
                           * NEARFIELD_CURV_TOL))
    denom = 1.0 / z_lo - NEARFIELD_MIN_ROWS / (f_y * camera_height)
    z_rows = 1.0 / denom if denom > 0 else np.inf
    if z_curv >= z_rows:
        return z_lo, min(z_curv, NEARFIELD_Z_HI_FACTOR_CAP * z_lo), True
    return z_lo, z_rows, False


def estimate_w_real_nearfield(widths, f_x, f_y, image_height, camera_height,
                              z_near_min=None, z_near_max=None,
                              min_points=NEARFIELD_MIN_POINTS):
    """Per-frame w_real from the near-field ground plane (camera-height anchor).

    The near road is the plane the wheels sit on, and the camera is rigid on
    the vehicle, so in the camera frame that plane is at Y = -camera_height by
    construction — no flatness assumption about the road ahead. That gives a
    second, w_real-free depth equation valid only there:

        z_h = f_y·h / (y - cy)                     (ground-plane inverse perspective)

    and combining it with the pinhole width model w_px = f_x·w_real/z yields a
    per-row measurement of the physical lane width:

        w_real(y) = w_px · f_y·h / (f_x·(y - cy))

    Rows are kept where z_h ∈ [z_near_min, z_near_max]; the window defaults to
    `nearfield_window(f_y, camera_height, image_height)`, derived from the
    camera rather than hand-set.

    A Theil-Sen fit of w_real(y) against z_h separates the two calibration
    unknowns, because their error signatures differ: a wrong w_real shifts all
    rows equally, while an uncalibrated camera mounting pitch θ0 (positive =
    pitched down) scales z_h by (1 + θ0·z/h), a trend ∝ z. Hence the intercept
    at z=0 is the θ0-free width and the slope reads the mounting pitch:

        w_est(z) = w_real·(1 + θ0·z/h)  →  θ0 = slope·h/intercept

    Returns None when fewer than min_points rows land in the window (short
    curves, bottom occlusion); otherwise a dict:

        w_real_z0  : HEADLINE estimate — Theil-Sen intercept, θ0-free
        w_real_med : median over the window; kept for diagnostics only. It
                     carries a +w·θ0·z̄/h bias, and on a sustained grade the
                     body sits at a fixed pitch relative to the road, so that
                     bias is a whole-section offset rather than noise.
        theta0_deg : implied camera mounting pitch (deg, positive = down)
        n_points   : rows used
        z_lo, z_hi : ground-plane depth range actually used (m)
        resid_mad  : robust scatter about the Theil-Sen line (m)
        quality_ok : passes the NEARFIELD_* quality gate (see the constants)

    Per-segment width error vs GT depth (CARLA / OpenLane): fixed constant
    108 / 173 mm, median 51 / 189 mm, intercept 46 / 35 mm. A quadratic
    intercept also absorbs curvature but diverges when the window starts far
    out (OpenLane), so the linear intercept is used for both cameras.
    ⚠ The extrapolation to z=0 amplifies line-position drift across the window
    about 5x on OpenLane (window 1.9 m long, 9 m out) — the dominant per-frame
    width error there (WWH-26).

    Input `widths` is the usual (y_pixel, pixel_width) array; the pipeline uses
    this through NearfieldWidthCalibrator (quality gate + hold).
    """
    if z_near_min is None or z_near_max is None:
        z_auto_lo, z_auto_hi, _ = nearfield_window(f_y, camera_height,
                                                   image_height)
        z_near_min = z_auto_lo if z_near_min is None else z_near_min
        z_near_max = z_auto_hi if z_near_max is None else z_near_max
    widths = np.asarray(widths, dtype=float)
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        return None
    y, w_px = widths[:, 0], widths[:, 1]
    dy = y - image_height / 2.0
    ok = (dy > 0) & (w_px > 0)          # below the horizon, physical width
    if not ok.any():
        return None
    z_h = f_y * camera_height / dy[ok]
    sel = (z_h >= z_near_min) & (z_h <= z_near_max)
    if sel.sum() < min_points:
        return None
    z = z_h[sel]
    w_est = z * w_px[ok][sel] / f_x
    if np.ptp(z) < 1e-9:
        return None
    fit = theilslopes(w_est, z)
    w_med = float(np.median(w_est))
    w_z0 = float(fit.intercept)
    theta0 = float(np.degrees(np.arctan(fit.slope * camera_height / w_z0))) \
        if w_z0 > 0 else np.nan
    resid = w_est - (fit.intercept + fit.slope * z)
    resid_mad = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    # 1 px of edge noise on a width sample is z/f_x metres at that depth.
    resid_allowed = NEARFIELD_RESID_PX * float(np.median(z)) / f_x
    quality_ok = bool(
        w_z0 > 0
        and np.isfinite(theta0) and abs(theta0) <= NEARFIELD_THETA0_MAX_DEG
        and np.ptp(z) >= NEARFIELD_MIN_SPAN_FRAC * (z_near_max - z_near_min)
        and resid_mad <= resid_allowed)
    return {
        "w_real_med": w_med,
        "w_real_z0": w_z0,
        "theta0_deg": theta0,
        "resid_mad": resid_mad,
        "quality_ok": quality_ok,
        "n_points": int(sel.sum()),
        "z_lo": float(z.min()),
        "z_hi": float(z.max()),
    }


def nearfield_widths_from_curves(left_curve, right_curve, f_y, camera_height,
                                 image_height, z_lo=None, z_hi=None):
    """Per-row (y, w_px) over the near-field window, straight off the curves.

    Every integer image row whose ground-plane depth f_y·h/(y−cy) falls in
    [z_lo, z_hi] and that both curves cover — denser than the z-uniform pitch
    samples, which is what the Theil-Sen θ0/width split wants. The window
    defaults to `nearfield_window` for this camera.
    """
    if left_curve is None or right_curve is None:
        return np.empty((0, 2))
    if z_lo is None or z_hi is None:
        auto_lo, auto_hi, _ = nearfield_window(f_y, camera_height, image_height)
        z_lo = auto_lo if z_lo is None else z_lo
        z_hi = auto_hi if z_hi is None else z_hi
    cy = image_height / 2.0
    y_lo = max(left_curve["y"][0], right_curve["y"][0],
               cy + f_y * camera_height / z_hi)
    y_hi = min(left_curve["y"][-1], right_curve["y"][-1],
               cy + f_y * camera_height / z_lo)
    if not y_hi > y_lo:
        return np.empty((0, 2))
    rows = np.arange(np.ceil(y_lo), np.floor(y_hi) + 1.0)
    xl = np.interp(rows, left_curve["y"], left_curve["x"])
    xr = np.interp(rows, right_curve["y"], right_curve["x"])
    w = xr - xl
    keep = w > 0
    return np.column_stack([rows[keep], w[keep]])


class NearfieldWidthCalibrator:
    """Per-frame w_real from the near field: quality-gated, held, no fallback.

    Wraps `estimate_w_real_nearfield` into the stateful policy the pipeline
    uses. A frame passes when the fit is of usable quality (`quality_ok`, see
    the NEARFIELD_* constants); its width — the θ0-free intercept — is adopted
    once a run of passes spans NEARFIELD_MIN_RUN_M of travel AND at least
    NEARFIELD_MIN_RUN_FRAMES passing frames; the run survives up to
    NEARFIELD_RUN_MAX_DROPOUT consecutive failed frames. From the
    NEARFIELD_POOL_MIN_N-th pass of the sequence on, every pass adopts the
    median of the last NEARFIELD_POOL_K passes instead (see the pooling note
    above the constants). Any other frame reuses the last
    adopted value, however long ago it was adopted; before the first adoption
    there is no value at all and update() returns None (see the hold-policy
    note above the constants — the last-resort constant is the pipeline's
    business, not this class's).

    One instance per CONTINUOUS image sequence: the instance is what defines
    "continuous", so a held value never crosses into another sequence. Call
    `advance_to(dist_m)` with the cumulative travel distance before each frame;
    it is needed for the minimum run in metres and for `hold_m`.

    `sequence=False` is for a lone image (single-image inference): there is no
    run to wait for, so a frame that passes the quality gate is used for
    itself. The inflection protection the run gives is lost — there is nothing
    else to judge that frame against.

    After each update(), for reporting:
      status       "measured" (adopted this frame) | "held" | "no_anchor"
      reason       why this frame was not measured: "no_nearfield_rows"
                   (nothing reached the near window), "quality_gate",
                   "run_too_short" (passed, run not long enough yet);
                   None when measured
      hold_frames  frames since the adoption in use (0 when measured)
      hold_m       metres since it; None without odometry
      n_pooled     passes behind the value in use (1 = a single frame's)

    Only the metric stage consumes the value: stages 1-3 take no lane width,
    and truncate_at_depth_jump works in lane-width units.
    """

    def __init__(self, f_x, f_y, image_height, camera_height, *, sequence=True):
        self.f_x = f_x
        self.f_y = f_y
        self.image_height = image_height
        self.camera_height = camera_height
        self.z_lo, self.z_hi, self.curvature_ok = nearfield_window(
            f_y, camera_height, image_height)
        self.sequence = bool(sequence)
        self.w_real = None                     # value the last update() returned
        self.status = None
        self.reason = None
        self.hold_frames = None
        self.hold_m = None
        self.last_estimate = None              # raw estimator dict, last frame
        self._dist = None                      # advance_to state (m)
        self._frame = -1
        self._run_start = None                 # (dist, frame) of gate-run start
        self._run_passes = 0                   # passing frames in the run
        self._dropout = 0                      # consecutive failures inside it
        self._last_adopt = None                # (dist, frame) of last adoption
        self._held = None
        self._pool = []                        # w_real_z0 of the last POOL_K passes
        self.n_pooled = None                   # estimates behind the value in use

    def advance_to(self, dist_m):
        """Cumulative travel distance (m) of the frame about to be fed."""
        self._dist = float(dist_m)

    def _span_m(self, since):
        """Travel in metres since a (dist, frame) mark; None when unknown."""
        if self._dist is None or since[0] is None:
            return None
        return self._dist - since[0]

    def _pooled(self, w):
        """Record a passing estimate; the estimates the pooled median takes."""
        self._pool.append(w)
        del self._pool[:-NEARFIELD_POOL_K]
        return self._pool

    def update(self, left_curve, right_curve):
        """Feed one frame's lane curves; returns the w_real to use for it, or
        None when this sequence has not produced one yet (see `status`)."""
        self._frame += 1
        widths = nearfield_widths_from_curves(
            left_curve, right_curve, self.f_y, self.camera_height,
            self.image_height, z_lo=self.z_lo, z_hi=self.z_hi)
        est = estimate_w_real_nearfield(
            widths, self.f_x, self.f_y, self.image_height, self.camera_height,
            z_near_min=self.z_lo, z_near_max=self.z_hi)
        self.last_estimate = est
        if est is None or not est["quality_ok"]:
            self.reason = "no_nearfield_rows" if est is None else "quality_gate"
            self._dropout += 1
            if self._dropout > NEARFIELD_RUN_MAX_DROPOUT:
                self._run_start = None
        else:
            if self._run_start is None:
                self._run_start = (self._dist, self._frame)
                self._run_passes = 0
            self._dropout = 0
            self._run_passes += 1
            span_m = self._span_m(self._run_start)
            pool = self._pooled(est["w_real_z0"]) if self.sequence else []
            pooled = len(pool) >= NEARFIELD_POOL_MIN_N
            run_ok = (pooled or not self.sequence
                      or (self._run_passes >= NEARFIELD_MIN_RUN_FRAMES
                          and (span_m is None or span_m >= NEARFIELD_MIN_RUN_M)))
            if run_ok:
                self._held = float(np.median(pool)) if pooled else est["w_real_z0"]
                self.n_pooled = len(pool) if pooled else 1
                self._last_adopt = (self._dist, self._frame)
                self.reason = None
            else:
                self.reason = "run_too_short"
        if self._held is None:
            self.status = "no_anchor"
            self.w_real = self.hold_frames = self.hold_m = self.n_pooled = None
            return None
        self.hold_frames = self._frame - self._last_adopt[1]
        self.hold_m = self._span_m(self._last_adopt)
        self.status = "measured" if self.hold_frames == 0 else "held"
        self.w_real = self._held
        return self.w_real


def resolve_lane_width(calibrator, left_curve, right_curve,
                       last_resort_lane_width=None):
    """This frame's metric lane width and where it came from.

    Feeds the calibrator (measured / held), and only when its sequence has no
    measurement at all falls to the last-resort constant, marked
    "last_resort" so a pitch on an assumed scale is never passed off as a
    measured one. Returns (width or None, status); status is the calibrator's,
    or "last_resort". None with "no_anchor" means: output no pitch.
    """
    w = calibrator.update(left_curve, right_curve)
    if w is None and last_resort_lane_width is not None:
        return float(last_resort_lane_width), "last_resort"
    return w, calibrator.status


def sample_widths_from_curves(left_curve, right_curve, num_samples, *,
                              f_x, w_real, samples_per_meter=None):
    """(y, width) samples over the y-overlap of two continuous lane curves.

    The curves come from lane_fitting.lane_curve: {"y": ascending, "x": ...},
    already gap-bridged. Width is a metric quantity (only meaningful through
    f_x / w_real), so sampling it belongs to this stage, not lane_fitting.

    Dense candidate sweep → depth via z = f_x·w_real/width → resample uniformly
    in z, so each metre of visible depth gets equal sample density. The count
    is `samples_per_meter` × the visible depth range, or a fixed `num_samples`
    when samples_per_meter is unset.

    (Sampling uniformly in y would spend most samples on the near few metres,
    z ∝ 1/(y−cy), where the pitch profile needs them least.)
    """
    if left_curve is None or right_curve is None:
        return np.empty((0, 2))
    y_lo = max(left_curve["y"][0], right_curve["y"][0])
    y_hi = min(left_curve["y"][-1], right_curve["y"][-1])
    if not y_hi > y_lo:
        return np.empty((0, 2))

    def width_at(ys):
        return (np.interp(ys, right_curve["y"], right_curve["x"])
                - np.interp(ys, left_curve["y"], left_curve["x"]))

    candidate_ys = np.linspace(y_lo, y_hi, 2000)
    ws = width_at(candidate_ys)
    valid = ws > 0
    if valid.sum() < 2:
        return np.empty((0, 2))
    ys_v, ws_v = candidate_ys[valid], ws[valid]
    zs = f_x * w_real / ws_v
    order = np.argsort(zs)
    zs_s, ys_s, ws_s = zs[order], ys_v[order], ws_v[order]
    if samples_per_meter is not None:
        # Scene-invariant density: n scales with the visible depth range,
        # capped at the candidate sweep resolution.
        z_range = zs_s[-1] - zs_s[0]
        n_samples = int(np.clip(np.ceil(z_range * samples_per_meter),
                                2, len(candidate_ys)))
    else:
        n_samples = num_samples
    target_zs = np.linspace(zs_s[0], zs_s[-1], n_samples)
    target_ys = np.interp(target_zs, zs_s, ys_s)
    target_ws = np.interp(target_zs, zs_s, ws_s)
    return np.column_stack([target_ys, target_ws])


def estimate_pitch_from_curves(left_curve, right_curve, f_x, f_y, image_height,
                               w_real, *, num_samples, samples_per_meter=None,
                               method="windowed", **kwargs):
    """Metric stage entry point: two continuous lane curves in, pitch out.

    Samples lane widths from the curves, then runs the continuous pitch(z)
    estimator — `method="spline"` (weighted UnivariateSpline, global) or
    `method="windowed"` (local z-window Theil-Sen). Returns the estimator
    dict plus "widths" (the sampled (y, w) array, for visualization /
    Y_3d profiling).
    """
    widths = sample_widths_from_curves(
        left_curve, right_curve, num_samples,
        f_x=f_x, w_real=w_real, samples_per_meter=samples_per_meter)
    est = estimate_pitch_windowed if method == "windowed" else estimate_pitch_from_widths
    result = est(widths, f_x, f_y, image_height, w_real, **kwargs)
    result["widths"] = widths
    return result


def _empty_result():
    return {
        "pitch_at": None,
        "z_samples": np.array([]),
        "pitch_samples": np.array([]),
        "y_samples": np.array([]),
        "z_points": np.array([]),
        "y_points": np.array([]),
        "z_visible_min": np.nan,
        "z_visible_max": np.nan,
    }


def _preprocess_widths(widths, f_x, f_y, image_height, w_real, z_cap_m):
    """Shared width→(z, Y_3d) preprocessing: IQR width filter, back-projection,
    depth cap, sort by depth. Returns (depths, Y_3d), possibly empty."""
    widths = np.asarray(widths)
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        return np.array([]), np.array([])
    w = widths[:, 1]
    q1, q3 = np.percentile(w, [25, 75])
    iqr = q3 - q1
    widths = widths[(w >= q1 - 1.5 * iqr) & (w <= q3 + 1.5 * iqr)]
    if len(widths) == 0:
        return np.array([]), np.array([])
    depths, Y_3d = back_project_widths(widths, f_x, f_y, image_height, w_real)
    if len(depths) == 0:
        return depths, Y_3d
    # Depth cap: z ~ 1/pixel_width, so spurious few-pixel widths near the
    # vanishing point explode to unphysical depths.
    in_range = depths <= z_cap_m
    if in_range.sum() >= 2:
        depths, Y_3d = depths[in_range], Y_3d[in_range]
    order = np.argsort(depths)
    return depths[order], Y_3d[order]


def estimate_pitch_windowed(widths, f_x, f_y, image_height, w_real,
                            min_valid_range_m: float = 0.5,
                            z_cap_m: float = 45.0,
                            window_frac: float = WINDOW_FRAC,
                            window_min_m: float = WINDOW_MIN_M,
                            min_window_points: int = 4,
                            n_pitch_samples: int = 200):
    """Continuous pitch(z) from local z-window Theil-Sen slopes.

    Explicitly-local alternative to the global spline: at each output depth
    z_c, pitch is the Theil-Sen slope of the (z, Y_3d) points within
    |z - z_c| <= max(window_min_m, window_frac·z_c). The window IS the
    spatial resolution of the profile — it grows with z for the same reason
    the spline down-weights far points (depth noise ∝ z²). No global
    residual filter: the local median is robust by itself, and a global one
    chops contiguous near/far tails. Windows short on points expand to the
    min_window_points nearest samples.

    Same input/return contract as estimate_pitch_from_widths.
    """
    depths, Y_3d = _preprocess_widths(widths, f_x, f_y, image_height, w_real,
                                      z_cap_m)
    z_range = float(depths[-1] - depths[0]) if len(depths) >= 2 else 0.0
    if z_range < min_valid_range_m or len(depths) < 2:
        return _empty_result()

    z_vis_min, z_vis_max = float(depths[0]), float(depths[-1])
    z_samps = np.linspace(z_vis_min, z_vis_max, n_pitch_samples)
    pitch_samps = np.empty(n_pitch_samples)
    y_samps = np.empty(n_pitch_samples)

    for i, zc in enumerate(z_samps):
        half = max(window_min_m, window_frac * zc)
        lo = np.searchsorted(depths, zc - half)
        hi = np.searchsorted(depths, zc + half, side="right")
        if hi - lo < min_window_points:
            k = min(min_window_points, len(depths))
            idx = np.argpartition(np.abs(depths - zc), k - 1)[:k]
            zw, yw = depths[idx], Y_3d[idx]
        else:
            zw, yw = depths[lo:hi], Y_3d[lo:hi]
        if np.ptp(zw) < 1e-9:
            # all window points at one depth — no slope information; carry
            # the previous sample (z_samps ascends, windows overlap)
            pitch_samps[i] = pitch_samps[i - 1] if i else 0.0
            y_samps[i] = float(yw.mean())
            continue
        fit = theilslopes(yw, zw)
        pitch_samps[i] = np.degrees(np.arctan(fit.slope))
        y_samps[i] = fit.intercept + fit.slope * zc

    def pitch_at(z):
        z_c = np.clip(z, z_vis_min, z_vis_max)
        return float(np.interp(z_c, z_samps, pitch_samps))

    return {
        "pitch_at": pitch_at,
        "z_samples": z_samps,
        "pitch_samples": pitch_samps,
        "y_samples": y_samps,
        "z_points": depths,
        "y_points": Y_3d,
        "z_visible_min": z_vis_min,
        "z_visible_max": z_vis_max,
    }


def estimate_pitch_from_widths(widths, f_x, f_y, image_height, w_real,
                               min_profile_range_m: float = 3.0,
                               min_valid_range_m: float = 0.5,
                               z_cap_m: float = 45.0,
                               resid_mad_k: float = 5.0,
                               s: float = None,
                               k: int = 3,
                               n_pitch_samples: int = 200):
    """Estimate a continuous pitch(z) curve from per-band lane widths.

    Preprocessing: IQR width filter → depth/Y_3d → depth cap → sort → global
    Theil-Sen MAD residual filter. A weighted UnivariateSpline on Y(z), then
    analytical differentiation, gives a smooth, continuous pitch(z).

    Physical weights: w_i = 1/z_i^2 because depth uncertainty scales as z^2
    (z = f·W/width, so dz ∝ z²·dwidth). Far points are naturally down-weighted,
    making the spline smooth at range without hard truncation.

    Parameters
    ----------
    widths : np.ndarray, shape (N, 2)
        (y_pixel, pixel_width) pairs from sample_widths_from_curves.
    s : float, optional
        Spline smoothing factor (UnivariateSpline `s`). Defaults to len(depths)
        after preprocessing. Increase to suppress far-range noise further.
    k : int
        Spline degree (default 3 = cubic). Auto-reduced when too few points.
    n_pitch_samples : int
        Number of uniformly spaced z samples in the returned arrays.

    Returns
    -------
    dict with keys:
        pitch_at      : callable z -> pitch_deg, clamped to [z_visible_min, z_visible_max]
        z_samples     : 1-D ndarray of n_pitch_samples depth values
        pitch_samples : corresponding pitch angles (deg), continuous
        y_samples     : fitted Y_3d curve evaluated at z_samples (spline or
                        Theil-Sen line) — for profile debugging
        z_points      : depths of the points that survived all filters and
                        were actually fitted
        y_points      : corresponding Y_3d values
        z_visible_min : float
        z_visible_max : float
    On degenerate/short input pitch_at is None and the sample arrays are empty.
    """
    depths, Y_3d = _preprocess_widths(widths, f_x, f_y, image_height, w_real,
                                      z_cap_m)

    # Robust residual filter: global Theil-Sen fit, drop points > resid_mad_k
    # robust sigmas. Catches off-plane junk (kerbs, crosswalks) that the
    # width-IQR filter cannot see.
    if len(depths) >= 4:
        fit = theilslopes(Y_3d, depths)
        resid = Y_3d - (fit.intercept + fit.slope * depths)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad > 1e-9:
            keep = np.abs(resid - med) <= resid_mad_k * 1.4826 * mad
            if keep.sum() >= 2:
                depths, Y_3d = depths[keep], Y_3d[keep]

    z_range = float(depths[-1] - depths[0]) if len(depths) >= 2 else 0.0
    if z_range < min_valid_range_m or len(depths) < 2:
        return _empty_result()

    z_vis_min = float(depths[0])
    z_vis_max = float(depths[-1])
    z_samps = np.linspace(z_vis_min, z_vis_max, n_pitch_samples)

    # Short range: too little depth coverage for a multi-knot spline —
    # fall back to a single global Theil-Sen slope (constant pitch).
    if z_range < min_profile_range_m or len(depths) < k + 1:
        res = theilslopes(Y_3d, depths)
        pitch_const = float(np.degrees(np.arctan(res.slope)))
        return {
            "pitch_at": lambda z, p=pitch_const: p,
            "z_samples": z_samps,
            "pitch_samples": np.full(n_pitch_samples, pitch_const),
            "y_samples": res.intercept + res.slope * z_samps,
            "z_points": depths,
            "y_points": Y_3d,
            "z_visible_min": z_vis_min,
            "z_visible_max": z_vis_max,
        }

    # Aggregate duplicate depths so UnivariateSpline gets a strictly
    # monotone x array (multiple y_pixels can map to the same z).
    unique_depths, inv = np.unique(depths, return_inverse=True)
    unique_Y = np.array([Y_3d[inv == i].mean() for i in range(len(unique_depths))])
    depths, Y_3d = unique_depths, unique_Y

    # Physical weights: far-depth uncertainty ∝ z², so weight by 1/z²
    # → far points get lower influence → spline is naturally smoother at range.
    weights = 1.0 / np.clip(depths, 1e-6, None) ** 2
    s_eff   = float(len(depths)) if s is None else float(s)
    k_eff   = min(k, len(depths) - 1)

    spl  = UnivariateSpline(depths, Y_3d, w=weights, k=k_eff, s=s_eff)
    dspl = spl.derivative()

    def pitch_at(z):
        z_c = float(np.clip(z, z_vis_min, z_vis_max))
        return float(np.degrees(np.arctan(float(dspl(z_c)))))

    pitch_samps = np.degrees(np.arctan(dspl(z_samps)))

    return {
        "pitch_at": pitch_at,
        "z_samples": z_samps,
        "pitch_samples": pitch_samps,
        "y_samples": spl(z_samps),
        "z_points": depths,
        "y_points": Y_3d,
        "z_visible_min": z_vis_min,
        "z_visible_max": z_vis_max,
    }
