"""
Inference pipeline — unified lane segmentation with perspective-adaptive ROI.
Stages
------
1. Road segmentation   (PIDNet)
2. ELSED line detection
3. Lane segmentation   split_left_right_lines (ROI-based, innermost)
4. Lane fitting        inner-chain points → continuous lane curves
5. Pitch estimation    widths from curves → continuous pitch(z); the default
                       estimator is `windowed` (local z-window Theil-Sen),
                       `spline` is opt-in via the `method` argument
"""

import cv2
import numpy as np
import pyelsed

from libs.inference.road_segmentation import predict_road, apply_road_mask
from libs.inference.lane_segmentation import split_left_right_lines
from libs.inference.line_segmentation import detect_lines_with_elsed
from libs.inference.lane_fitting      import (inner_chain_points, refine_inner_points,
                                              lane_curve, truncate_at_depth_jump)
from libs.inference.paint_evidence    import filter_paint_segments, truncate_at_evidence_break
from libs.inference.pitch_estimation  import (estimate_pitch_from_curves,
                                              NearfieldWidthCalibrator, _empty_result,
                                              resolve_lane_width)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def infer_one(
    model, image_path, device, resize_size,
    min_segment_length_near, min_segment_length_far,
    num_samples,
    f_x, f_y, camera_height,
    *,
    samples_per_meter: float = None,
    track_bands: int = 16,
    method: str = "windowed",
    w_real_calibrator=None,
    last_resort_lane_width: float = None,
    return_debug: bool = False,
):
    """Run the full pipeline on a single image.

    Parameters
    ----------
    f_x, f_y, camera_height
        Camera calibration. Required — the tracker's thresholds and the
        paint-evidence probe windows are all derived from these three.
    w_real_calibrator
        pitch_estimation.NearfieldWidthCalibrator, one instance per continuous
        image sequence: the lane width is MEASURED from the near field, never
        configured. Omit it for a lone image and a
        one-shot calibrator (sequence=False) is used. Only the metric stage
        needs the width — stages 1-3 take none, truncate_at_depth_jump works
        in lane-width units.
    last_resort_lane_width
        Assumed lane width (m) for a frame whose sequence has not measured
        one yet — the config's `last_resort_lane_width`. Such frames are
        marked w_real_status = "last_resort" so their pitch, on an assumed
        scale, is never mistaken for a measured one. None: output no pitch
        for them instead (status "no_anchor").

    Returns {"pitch_curve", "w_real_used", "w_real_status", "w_real_reason",
    "w_real_hold_frames", "w_real_hold_m"}. When the sequence has no measured
    width yet, the last-resort width is used (status "last_resort"), or with
    none given w_real_used is None, the status is "no_anchor" and the pitch
    curve is the empty (degenerate) result. The reason says why this frame
    could not be measured itself (see NearfieldWidthCalibrator).
    return_debug
        If True, also return a dict with intermediate values.
    """
    # 1. road segmentation
    resized_image, pred_mask = predict_road(model, image_path, device, resize_size)
    masked_road = apply_road_mask(resized_image, pred_mask)

    # 2. ELSED line detection
    segments = detect_lines_with_elsed(masked_road, min_segment_length_near, min_segment_length_far)

    # 2b. paint-evidence segment gate: drop segments that are not painted-
    # marking edges (shadow boundaries, kerbs, walls) so the tracker seeds
    # on real paint — see paint_evidence.py
    if len(segments):
        segments = filter_paint_segments(
            resized_image, segments, f_x, f_y, camera_height)

    # 3. lane segmentation (per-band innermost selection)
    inner_left, inner_right = split_left_right_lines(
        segments, resized_image.width, resized_image.height,
        track_bands=track_bands,
        f_x=f_x, f_y=f_y, camera_height=camera_height)

    # 4. lane fitting — per-row inner-envelope chain (the width is inner-edge
    # to inner-edge; the tracker keeps whole marking groups for evidence),
    # refined to sub-pixel, cut at the first sustained paint-evidence break
    # (crest occlusion boundary), then a continuous gap-bridged curve per side
    left_points = refine_inner_points(
        resized_image, inner_chain_points(inner_left,  True),  True)
    right_points = refine_inner_points(
        resized_image, inner_chain_points(inner_right, False), False)
    left_points = truncate_at_evidence_break(
        resized_image, left_points, True, f_x, f_y, camera_height)
    right_points = truncate_at_evidence_break(
        resized_image, right_points, False, f_x, f_y, camera_height)
    # depth-continuity guard: real paint beyond a crest occlusion is
    # still a DIFFERENT lane section — never join it to the near chain
    left_points, right_points = truncate_at_depth_jump(
        left_points, right_points, f_x, resized_image.height)
    left_curve  = lane_curve(left_points)
    right_curve = lane_curve(right_points)

    # 5. pitch estimation — widths from the curves → continuous pitch(z)
    # (`method` selects windowed Theil-Sen, the default, or the global spline).
    # The metric scale is the near-field width; before the sequence has one,
    # the last-resort constant (flagged), or no pitch at all.
    cal = w_real_calibrator
    if cal is None:
        cal = NearfieldWidthCalibrator(f_x, f_y, resized_image.height,
                                       camera_height, sequence=False)
    w_real_metric, status = resolve_lane_width(
        cal, left_curve, right_curve, last_resort_lane_width)
    if w_real_metric is None:
        pitch_curve = {**_empty_result(), "widths": np.empty((0, 2))}
    else:
        pitch_curve = estimate_pitch_from_curves(
            left_curve, right_curve, f_x, f_y, resized_image.height, w_real_metric,
            num_samples=num_samples, samples_per_meter=samples_per_meter,
            method=method)

    result = {"pitch_curve": pitch_curve, "w_real_used": w_real_metric,
              "w_real_status": status, "w_real_reason": cal.reason,
              "w_real_hold_frames": cal.hold_frames, "w_real_hold_m": cal.hold_m}
    if return_debug:
        degenerate = pitch_curve["pitch_at"] is None or len(pitch_curve["z_samples"]) == 0
        result["debug"] = {
            "n_segments": int(len(segments)),
            "n_left": len(inner_left),
            "n_right": len(inner_right),
            "n_width_samples": int(len(pitch_curve["widths"])),
            "pitch_degenerate": degenerate,
            "left_curve": left_curve,
            "right_curve": right_curve,
        }
    return result
