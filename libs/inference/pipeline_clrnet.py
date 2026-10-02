"""
Inference pipeline with a learned 2D lane detector front end (WWH-25).

A separate entry point, NOT a mode of pipeline.infer_one: the ELSED pipeline is
left untouched so the two front ends can be compared on the same frames.
Stages 1-3 of the ELSED pipeline (road mask, ELSED, tracker) are replaced; the
metric stage is the same code, called the same way.

    1. Read + resize           same PIL bilinear resize as predict_road, no PIDNet
    2. Lane detector           CLRNet (lane_detector.py): whole lane instances
    3. Ego pair                model_lane_fitting.pick_ego, as near as possible,
                               checked by guard_ego (near-field width plausible)
    4. Per-row curves          detector position at every row; each row flagged
                               paint / model by the image (optionally snapped to
                               the paint centre - off by default, see `refine`);
                               optional tail rule and depth cut; lane_fitting's
                               depth-jump guard
    5. Pitch estimation        unchanged: NearfieldWidthCalibrator →
                               resolve_lane_width → estimate_pitch_from_curves

The width is now centre-to-centre (see model_lane_fitting). The near-field
calibrator measures it per frame, so nothing downstream needs to know — except
`last_resort_lane_width`, whose 3.25 is inner-edge-to-inner-edge; pass the
centre-to-centre value for this front end.

Extra outputs, for evaluation (reported, not used as gates):
    width_paint_frac      share of the pitch stage's width samples whose row is
                          paint on BOTH sides
    nearfield_paint_frac  same, over the near-field window rows
    refined               True when the ego lines were snapped to paint (refine)
"""

import numpy as np
from PIL import Image

from libs.inference.lane_fitting import truncate_at_depth_jump
from libs.inference.model_lane_fitting import (apply_tail, densify, trim_pitch_to_depth,
                                               guard_ego, model_curve, pick_ego,
                                               refine_center, smooth_refine, src_at)
from libs.inference.pitch_estimation import (NearfieldWidthCalibrator,
                                             _empty_result,
                                             estimate_pitch_from_curves,
                                             nearfield_widths_from_curves,
                                             resolve_lane_width)


def load_resized(image_path, resize_size):
    """RGB image resized exactly as predict_road does ([height, width])."""
    image = Image.open(image_path).convert("RGB")
    return image.resize((resize_size[1], resize_size[0]), Image.BILINEAR)


# refine="solid": snap only when BOTH ego lines are continuous paint over their
# whole drawn length — solid and unoccluded. Then every row has paint, so the
# reference point never jumps between the snapped and the detector centre (what
# made "center" worse on dashed lines). Measured on the CARLA OpenLane-camera
# routes (current settings): solid Town03 lines 0.94-1.00 paint rows, longest
# paint-free run 2-4 rows; dashed Town05 0.30-0.38 and ~77 rows.
SOLID_MIN_PAINT_FRAC = 0.9
SOLID_MAX_GAP_ROWS = 10


def _is_solid(paint):
    """True when the line is paint on >= SOLID_MIN_PAINT_FRAC of its rows and
    no paint-free run is longer than SOLID_MAX_GAP_ROWS."""
    if len(paint) == 0 or paint.mean() < SOLID_MIN_PAINT_FRAC:
        return False
    gap = run = 0
    for p in paint:
        run = 0 if p else run + 1
        gap = max(gap, run)
    return gap <= SOLID_MAX_GAP_ROWS


def _side(image, lane, f_x, f_y, camera_height):
    """(rows, detector x, paint-snapped x, is_paint) for one ego line; None without one."""
    if lane is None:
        return None
    rows, x_prior = densify(lane)
    x_snap, paint = refine_center(image, rows, x_prior, f_x, f_y, camera_height)
    return rows, x_prior, x_snap, paint


def _points(side, how, tail, f_x=None, f_y=None, camera_height=None, image_height=None):
    """(points (N, 2) of (x, y), is_paint (N,)) with the chosen x and the tail rule.
    how: "prior" (detector), "snap" (paint centre per row) or "smooth" (smooth_refine)."""
    if side is None:
        return np.empty((0, 2)), np.empty(0, dtype=bool)
    rows, x_prior, x_snap, paint = side
    if how == "snap":
        x = x_snap
    elif how == "smooth":
        x = smooth_refine(rows, x_prior, x_snap, paint, f_x, f_y, camera_height, image_height)
    else:
        x = x_prior
    rows, x, paint = apply_tail(rows, x, paint, tail)
    return np.column_stack([x, rows]), paint


def _keep_src(points_before, paint_before, points_after):
    """Source flags of the rows truncate_at_depth_jump kept (it keeps rows by
    y, never moves them)."""
    if len(points_after) == 0:
        return np.empty(0, dtype=bool)
    lookup = dict(zip(np.round(points_before[:, 1]).astype(int), paint_before))
    return np.array([lookup[int(round(y))] for y in points_after[:, 1]], dtype=bool)


def _paint_only(curve):
    """The curve's paint rows only (None with < 2): what the near-field
    calibrator sees when nearfield_source = "paint"."""
    if curve is None:
        return None
    keep = curve["src"]
    return model_curve(curve["y"][keep], curve["x"][keep], keep[keep])


def _both_paint(left_curve, right_curve, ys):
    if left_curve is None or right_curve is None or len(ys) == 0:
        return np.zeros(len(ys), dtype=bool)
    return src_at(left_curve, ys) & src_at(right_curve, ys)


def infer_one_clrnet(
    detector, image_path, resize_size,
    num_samples,
    f_x, f_y, camera_height,
    *,
    cut_frac: float,
    detector_mode: str = "culane",
    tail: str = "keep",
    refine: str = "none",
    nearfield_source: str = "paint",
    ego_guard: bool = True,
    max_depth_m: float = None,
    samples_per_meter: float = None,
    method: str = "windowed",
    w_real_calibrator=None,
    last_resort_lane_width: float = None,
    return_debug: bool = False,
):
    """Run the detector-front-end pipeline on one image.

    detector
        lane_detector.CLRNet instance.
    cut_frac
        Share of the image top cropped before the detector, as it was trained
        (CULane: 270/590). Step 1 measured that running uncropped LOSES lines
        (86 % -> 66.5 % on OpenLane updown_straight), so keep the training value.
    detector_mode
        lane_detector preprocessing: "culane" (pad to CULane's aspect — the
        released weights) or "naive" (stretch the cropped image to the network
        input — how CLRNet itself trains, so use it for fine-tuned weights).
        Must match how the weights were trained, as must cut_frac.
    tail
        "keep" or "last_paint" — see model_lane_fitting.apply_tail.
    refine
        "none" (default: detector positions everywhere; the paint flags are
        still computed), "center" (snap paint rows to the stripe centre) or
        "solid" (snap only when both ego lines are solid and unoccluded, see
        SOLID_*; otherwise as "none") or "smooth" (every row shifted by the
        paint-measured offset smoothed along the line, model_lane_fitting.
        smooth_refine — dashed lines too). "center" makes pitch WORSE at every depth
        (Town05 5-10 m 0.262° vs 0.133°): the reference point jumps between the
        snapped and the detector centre at every dash end.
    nearfield_source
        "paint" (default: the calibrator sees only paint rows, bridged between
        paint rows as lane_curve would; no paint in the window ⇒ not measured
        this frame, the sequence holds) or "all" (the full curves). Paint-only
        cuts the OpenLane width error p90 from 0.59 m to 0.23 m.
    ego_guard
        Run model_lane_fitting.guard_ego on the picked pair.
    max_depth_m
        No pitch output beyond this depth (m); the estimate is made on the full
        curves and only its output is trimmed (trim_pitch_to_depth). None: no
        limit. The config sets it from the measured flattening of the weights.
    ⚠ Tried and rejected (WWH-28): a trust threshold — lines scoring below it
    used only their own frame's width, never a held one. Its gain on OpenLane
    up&down (near Z-error −0.6 cm for −10 points of output) was one segment
    (133368: 97 of the 135 frames it dropped), whose real fault is a held width
    6 % low that the high-score frames there share; on two held-out sets it
    moved ±0.1 cm and on CARLA it was no better. The fault to fix is a held
    width nothing can check (to-do), not the score of the line.

    Everything else as pipeline.infer_one, and the same result keys, plus
    width_paint_frac / nearfield_paint_frac (module docstring).
    """
    image = load_resized(image_path, resize_size)
    rgb = np.asarray(image)
    H, W = rgb.shape[:2]

    lanes, score = detector(rgb, f_x, f_y, cut=int(round(cut_frac * H)), mode=detector_mode)
    left, right, y_pick_l, y_pick_r = pick_ego(lanes, W)
    guard = "off"
    if ego_guard:
        left, right, guard = guard_ego(lanes, left, right, f_x, f_y, camera_height, W, H)

    if refine not in ("none", "center", "solid", "smooth"):
        raise ValueError(f"refine must be 'none', 'center', 'solid' or 'smooth', got {refine!r}")
    sl = _side(image, left, f_x, f_y, camera_height)
    sr = _side(image, right, f_x, f_y, camera_height)
    snapped = refine == "center" or (refine == "solid" and sl is not None and sr is not None
                                     and _is_solid(sl[3]) and _is_solid(sr[3]))
    how = "smooth" if refine == "smooth" else ("snap" if snapped else "prior")
    lp, lsrc = _points(sl, how, tail, f_x, f_y, camera_height, H)
    rp, rsrc = _points(sr, how, tail, f_x, f_y, camera_height, H)
    snapped = snapped or refine == "smooth"
    lp2, rp2 = truncate_at_depth_jump(lp, rp, f_x, H)
    lsrc, rsrc = _keep_src(lp, lsrc, lp2), _keep_src(rp, rsrc, rp2)
    left_curve = model_curve(lp2[:, 1], lp2[:, 0], lsrc) if len(lp2) else None
    right_curve = model_curve(rp2[:, 1], rp2[:, 0], rsrc) if len(rp2) else None

    cal = w_real_calibrator
    if cal is None:
        cal = NearfieldWidthCalibrator(f_x, f_y, H, camera_height, sequence=False)
    if nearfield_source == "all":
        cal_l, cal_r = left_curve, right_curve
    elif nearfield_source == "paint":
        cal_l, cal_r = _paint_only(left_curve), _paint_only(right_curve)
    else:
        raise ValueError(f"nearfield_source must be 'all' or 'paint', got {nearfield_source!r}")
    w_real_metric, status = resolve_lane_width(
        cal, cal_l, cal_r, last_resort_lane_width)
    reason, hold_frames, hold_m = cal.reason, cal.hold_frames, cal.hold_m
    if w_real_metric is None:
        pitch_curve = {**_empty_result(), "widths": np.empty((0, 2))}
    else:
        pitch_curve = estimate_pitch_from_curves(
            left_curve, right_curve, f_x, f_y, H, w_real_metric,
            num_samples=num_samples, samples_per_meter=samples_per_meter,
            method=method)
        pitch_curve = trim_pitch_to_depth(pitch_curve, max_depth_m)

    # share of the reported pitch output's width samples that are paint on both
    # sides: only rows within the (possibly trimmed) output depth, None without output
    width_paint_frac = None
    w_rows = np.empty(0)
    widths = np.asarray(pitch_curve["widths"])
    if pitch_curve["pitch_at"] is not None and widths.ndim == 2 and len(widths):
        z = f_x * w_real_metric / widths[:, 1]
        w_rows = widths[z <= pitch_curve["z_visible_max"], 0]
        both = _both_paint(left_curve, right_curve, w_rows)
        width_paint_frac = float(both.mean()) if len(both) else None

    # same rows the calibrator's near-field window takes, from the same function
    nf = nearfield_widths_from_curves(left_curve, right_curve, f_y, camera_height, H,
                                      z_lo=cal.z_lo, z_hi=cal.z_hi)
    nearfield_paint_frac = (float(_both_paint(left_curve, right_curve, nf[:, 0]).mean())
                            if len(nf) else None)

    result = {"pitch_curve": pitch_curve, "w_real_used": w_real_metric,
              "w_real_status": status, "w_real_reason": reason,
              "w_real_hold_frames": hold_frames, "w_real_hold_m": hold_m,
              "width_paint_frac": width_paint_frac,
              "nearfield_paint_frac": nearfield_paint_frac,
              "ego_guard": guard, "refined": bool(snapped)}
    if return_debug:
        degenerate = pitch_curve["pitch_at"] is None or len(pitch_curve["z_samples"]) == 0
        result["debug"] = {
            "n_lanes": len(lanes),
            "lane_score": score,
            "lanes": lanes,
            "y_pick": (y_pick_l, y_pick_r),
            "n_width_samples": int(len(w_rows)),
            "pitch_degenerate": degenerate,
            "left_curve": left_curve,
            "right_curve": right_curve,
        }
    return result
