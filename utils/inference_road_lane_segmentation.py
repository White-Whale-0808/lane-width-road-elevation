from utils.env_setup import setup_env
setup_env()

import time
from pathlib import Path
import pandas as pd
import yaml
from libs.inference.road_segmentation import load_pidnet, predict_road, apply_road_mask
from libs.inference.line_segmentation import detect_lines_with_elsed
from libs.inference.lane_segmentation import split_left_right_lines
from libs.visualization.lane_visualization import draw_lane_lines, create_overlay, draw_line_segments, save_lane_fitting_steps
from libs.inference.lane_fitting import (inner_chain_points, refine_inner_points,
                                         lane_curve, truncate_at_depth_jump)
from libs.inference.paint_evidence import filter_paint_segments, truncate_at_evidence_break
from libs.visualization.lane_visualization import draw_lane_curves
from libs.inference.pitch_estimation import (estimate_pitch_from_curves, NearfieldWidthCalibrator,
                                             resolve_lane_width)
from libs.visualization.pitch_visualization import plot_pitch_profile, plot_y3d_profile, save_pitch_estimation_steps
from libs.road_profile_gt import load_profile_gt
from libs.dataset_camera import resolve_camera, describe as describe_camera

with open("config/inference_road_lane_segmentation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = config["model"]["device"]
model_name = config["model"]["model_name"]
weight_path = config["model"]["weight_path"]
image_path = config["input"]["image_path"]
resize_size = tuple(config["input"]["resize_size"])
min_segment_length_near = config["line_segmentation"]["min_segment_length_near"]
min_segment_length_far  = config["line_segmentation"]["min_segment_length_far"]
track_bands = config["lane_segmentation"].get("track_bands", 16)
alpha = config["visualization"]["alpha"]
save_path = config["visualization"]["save_path"]
num_samples = config["lane_fitting"]["num_samples"]
samples_per_meter = config["lane_fitting"].get("samples_per_meter")
# 相機：影像所屬資料集的 metadata.json 描述的相機跟 config 不同時用資料集自己的
camera, camera_source = resolve_camera(config["pitch_estimation"], Path(image_path).parent.parent,
                                       resize_size)
f_x, f_y, camera_height = camera["f_x"], camera["f_y"], camera["camera_height"]
camera_offset_m  = camera["camera_forward_offset"]
last_resort_lane_width = config["pitch_estimation"].get("last_resort_lane_width")
gt_height_source = config.get("ground_truth", {}).get("height_source", "auto")
pitch_method     = config["pitch_estimation"].get("method", "windowed")
# GT 取自影像自己的資料集（<dataset>/images/000200.png -> <dataset>/measurements.csv），
# 不看 csv_io.measurements_csv —— 那個鍵是 batch 用的；兩邊指到不同資料集時，
# 會把影像配上另一份資料集的同編號幀而不報錯。
measurements_csv = Path(image_path).parent.parent / "measurements.csv"  # .parent 對短路徑不會爆


def compute_pitch_mae(pitch_curve, frame_id, gt):
    """Compute MAE between predicted continuous pitch(z) and the GT profile."""
    import numpy as np
    if pitch_curve["pitch_at"] is None or len(pitch_curve["z_samples"]) == 0:
        return None
    zs = pitch_curve["z_samples"]
    ps = pitch_curve["pitch_samples"]
    gt_deg = gt.pitch_at(frame_id, zs)
    valid = ~np.isnan(gt_deg)
    if not valid.any():
        return None
    return float(np.abs(ps[valid] - gt_deg[valid]).mean())


def main():
    """
    Road segementation
    """
    print(describe_camera(camera, camera_source))
    model = load_pidnet(model_name, weight_path, device)
    t0 = time.perf_counter()
    resized_image, pred_mask = predict_road(model, image_path, device, resize_size)
    masked_road = apply_road_mask(resized_image, pred_mask)
    t1 = time.perf_counter()

    """
    Line segementation
    """
    segments = detect_lines_with_elsed(masked_road, min_segment_length_near, min_segment_length_far)

    # paint-evidence segment gate — same as pipeline.py
    if len(segments):
        segments = filter_paint_segments(
            resized_image, segments, f_x, f_y, camera_height)
    t2 = time.perf_counter()

    """
    Lane segementation
    """
    inner_left, inner_right = split_left_right_lines(
        segments, resized_image.width, resized_image.height,
        track_bands=track_bands,
        f_x=f_x, f_y=f_y, camera_height=camera_height)
    t3 = time.perf_counter()

    """
    Lane fitting
    """
    left_points = refine_inner_points(
        resized_image, inner_chain_points(inner_left, True), True)
    right_points = refine_inner_points(
        resized_image, inner_chain_points(inner_right, False), False)
    left_points = truncate_at_evidence_break(
        resized_image, left_points, True, f_x, f_y, camera_height)
    right_points = truncate_at_evidence_break(
        resized_image, right_points, False, f_x, f_y, camera_height)
    left_points, right_points = truncate_at_depth_jump(
        left_points, right_points, f_x, resized_image.height)
    left_curve = lane_curve(left_points)
    right_curve = lane_curve(right_points)
    t4 = time.perf_counter()

    """
    Pitch estimation
    """
    # 車道寬由近場量測（單張沒有前後幀可沿用）；量不到才用 config 的
    # last_resort_lane_width 並明白標出，沒設就不輸出 pitch
    calibrator = NearfieldWidthCalibrator(f_x, f_y, resized_image.height,
                                          camera_height, sequence=False)
    w_real, w_status = resolve_lane_width(
        calibrator, left_curve, right_curve, last_resort_lane_width)
    pitch_curve = None
    if w_status == "last_resort":
        print(f"lane width: not measurable ({calibrator.reason}) — "
              f"LAST RESORT {w_real:.3f} m (assumed, not measured)")
    elif w_real is not None:
        print(f"lane width (near field): {w_real:.3f} m")
    if w_real is None:
        print(f"lane width: not measurable ({calibrator.reason}) — no pitch output")
    else:
        pitch_curve = estimate_pitch_from_curves(
            left_curve, right_curve, f_x, f_y, resized_image.height, w_real,
            num_samples=num_samples, samples_per_meter=samples_per_meter,
            method=pitch_method)
        widths = pitch_curve["widths"]
        zs, ps = pitch_curve["z_samples"], pitch_curve["pitch_samples"]
        if len(zs) == 0:
            print("pitch: (degenerate — no valid depth range)")
        else:
            step = max(1, len(zs) // 10)
            for z, theta in zip(zs[::step], ps[::step]):
                print(f"  z={z:.1f}m  pitch={theta:.2f}°")
    t5 = time.perf_counter()

    print(f"road segmentation:   {(t1-t0)*1000:.1f} ms")
    print(f"line segmentation:      {(t2-t1)*1000:.1f} ms")
    print(f"lane segmentation:   {(t3-t2)*1000:.1f} ms")
    print(f"lane fitting:        {(t4-t3)*1000:.1f} ms")
    print(f"pitch estimation:    {(t5-t4)*1000:.1f} ms")

    gt, frame_id = None, None
    if pitch_curve is None:
        pass                                  # no pitch to score or plot
    elif not measurements_csv.exists():
        print(f"GT: skipped (no {measurements_csv})")
    else:
        frame_id = int(Path(image_path).stem)
        gt = load_profile_gt(measurements_csv, camera_offset_m=camera_offset_m,
                             camera_height=camera_height,
                             height_source=gt_height_source)
        print(f"GT source: {gt.describe()}")

        mae = compute_pitch_mae(pitch_curve, frame_id, gt)
        if mae is not None:
            print(f"pitch MAE: {mae:.4f}°")

        pitch_plot_path = save_path.replace(".png", "_pitch_profile.png")
        plot_pitch_profile(frame_id, pitch_curve, gt,
                           save_path=pitch_plot_path,
                           camera_offset_m=camera_offset_m)
        print(f"pitch profile: {pitch_plot_path}")

        y3d_plot_path = save_path.replace(".png", "_y3d_profile.png")
        plot_y3d_profile(frame_id, widths, pitch_curve, gt,
                         f_x, f_y, resized_image.height, w_real, camera_height,
                         save_path=y3d_plot_path,
                         camera_offset_m=camera_offset_m)
        print(f"y3d profile: {y3d_plot_path}")

    overlay_save_path = save_path.replace(".png", "_overlay.png")
    create_overlay(resized_image, pred_mask, alpha, overlay_save_path)
    draw_line_save_path = save_path.replace(".png", "_line_segments.png")
    draw_line_segments(resized_image, segments, draw_line_save_path)
    draw_lane_save_path = save_path.replace(".png", "_lanes_seg.png")
    draw_lane_lines(resized_image, inner_left, inner_right, draw_lane_save_path)
    draw_lane_curves_save_path = save_path.replace(".png", "_lane_fits.png")
    draw_lane_curves(resized_image, left_curve, right_curve, draw_lane_curves_save_path)

    steps_dir = save_lane_fitting_steps(resized_image, inner_left, inner_right,
                                        "outputs/lane_fit_step")
    print(f"lane fitting steps: {steps_dir}")
    if pitch_curve is None:
        return
    steps_dir = save_pitch_estimation_steps(
        resized_image, left_curve, right_curve, pitch_curve,
        f_x, f_y, resized_image.height, w_real, "outputs/pitch_est_step",
        gt=gt, frame_id=frame_id,
        camera_offset_m=camera_offset_m)
    print(f"pitch estimation steps: {steps_dir}")

if __name__ == "__main__":
    main()
