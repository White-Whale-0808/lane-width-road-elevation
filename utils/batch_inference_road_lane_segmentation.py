from utils.env_setup import setup_env
setup_env()

import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from libs.inference.road_segmentation import load_pidnet
from libs.inference.pipeline import infer_one
from libs.inference.pitch_estimation import NearfieldWidthCalibrator
from libs.road_profile_gt import load_profile_gt
from libs.dataset_camera import resolve_camera, describe as describe_camera
from libs.visualization.profile_mae_visualization import plot_profile_mae
from libs.visualization.route_profile_visualization import plot_route_profile
import traceback

with open("config/inference_road_lane_segmentation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device                    = config["model"]["device"]
model_name                = config["model"]["model_name"]
weight_path               = config["model"]["weight_path"]
image_batch_path          = config["input"]["image_batch_path"]
resize_size               = tuple(config["input"]["resize_size"])
min_segment_length_near   = config["line_segmentation"]["min_segment_length_near"]
min_segment_length_far    = config["line_segmentation"]["min_segment_length_far"]
track_bands               = config["lane_segmentation"].get("track_bands", 16)
num_samples               = config["lane_fitting"]["num_samples"]
samples_per_meter         = config["lane_fitting"].get("samples_per_meter")
last_resort_lane_width    = config["pitch_estimation"].get("last_resort_lane_width")
pitch_method              = config["pitch_estimation"].get("method", "windowed")
gt_height_source          = config.get("ground_truth", {}).get("height_source", "auto")
input_csv                 = config["csv_io"]["input_dir"]
output_csv                = config["csv_io"]["output_dir"]
measurements_csv          = config["csv_io"]["measurements_csv"]
problem_csv               = config["csv_io"]["problem_csv"]
problem_mae_threshold     = config["csv_io"]["problem_mae_threshold"]

IMG_FMT = "{:06d}.png"


def _tagged(path, tag):
    """foo.csv + tag 'x' -> foo_x.csv（tag 為空則原樣返回）"""
    if not tag:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{tag}{p.suffix}"))


def _parse_args(argv=None):
    """CLI overrides for the config defaults.

    Everything defaults to the config, so a bare run is unchanged. The
    overrides exist so one acceptance sweep can cover several routes
    without editing the config between runs (and without a second copy of
    the pipeline — see infer_one's comment). One dataset dir is one
    continuous sequence: the lane-width calibrator lives for exactly one run.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=None,
                    help="資料集目錄（含 images/ 與 measurements.csv）；"
                         "省略則用 config 的路徑")
    ap.add_argument("--tag", default="",
                    help="輸出檔名後綴，連跑多組才不會互相覆蓋")
    return ap.parse_args(argv)


def _compute_mae(pitch_curve, frame_id, gt):
    """Return profile MAE (float) or None if degenerate or no valid GT samples."""
    if pitch_curve["pitch_at"] is None or len(pitch_curve["z_samples"]) == 0:
        return None
    zs = pitch_curve["z_samples"]
    ps = pitch_curve["pitch_samples"]
    gt_deg = gt.pitch_at(frame_id, zs)
    valid = ~np.isnan(gt_deg)
    if not valid.any():
        return None
    return round(float(np.abs(ps[valid] - gt_deg[valid]).mean()), 4)


def _report_width_status(df):
    """車道寬來源統計：量到 / 沿用 / 最後手段常數 / 無錨（後兩者＝本片段還沒量到過）。"""
    st = df["w_real_status"].dropna()
    if not len(st):
        return
    counts = st.value_counts()
    print("lane width: " + "  ".join(
        f"{k}={int(counts.get(k, 0))}" for k in ("measured", "held", "last_resort", "no_anchor")))
    not_measured = df.loc[df["w_real_status"] != "measured", "w_real_reason"].dropna()
    if len(not_measured):
        print("  not measured this frame, by reason: " + "  ".join(
            f"{k}={v}" for k, v in not_measured.value_counts().items()))
    for st, what in (("last_resort", "pitch on the ASSUMED last_resort_lane_width"),
                     ("no_anchor", "no pitch output")):
        ids = df.loc[df["w_real_status"] == st, "frame_id"].astype(int)
        if len(ids):
            print(f"  {st} frames ({what}): {ids.tolist()}")
    hold_m = pd.to_numeric(df["w_real_hold_m"], errors="coerce")
    held = hold_m[df["w_real_status"] == "held"].dropna()
    if len(held):
        print(f"  held value age: median={held.median():.1f} m  "
              f"p90={held.quantile(0.9):.1f} m  max={held.max():.1f} m")


def main(argv=None):
    args = _parse_args(argv)
    # CLI 覆寫（沒給就是 config 的值）
    if args.dataset:
        in_csv   = str(args.dataset / "measurements.csv")
        meas_csv = in_csv
        image_dir_str = str(args.dataset / "images")
    else:
        in_csv, meas_csv = input_csv, measurements_csv
        image_dir_str = image_batch_path
    # 相機：資料集 metadata.json 描述的相機跟 config 不同時用資料集自己的
    cam, cam_src = resolve_camera(config["pitch_estimation"], Path(image_dir_str).parent, resize_size)
    print(describe_camera(cam, cam_src))
    f_x, f_y, camera_height = cam["f_x"], cam["f_y"], cam["camera_height"]
    camera_offset_m = cam["camera_forward_offset"]
    out_csv  = _tagged(output_csv, args.tag)
    prob_csv = _tagged(problem_csv, args.tag)

    df = pd.read_csv(in_csv)
    gt = load_profile_gt(meas_csv, camera_offset_m=camera_offset_m,
                         camera_height=camera_height,
                         height_source=gt_height_source)
    print(f"GT source: {gt.describe()}")
    model = load_pidnet(model_name, weight_path, device)

    n_total         = len(df)
    z_vis_min_col   = [pd.NA] * n_total
    z_vis_max_col   = [pd.NA] * n_total
    profile_mae_col = [pd.NA] * n_total
    w_real_used_col = [pd.NA] * n_total
    w_status_col    = [pd.NA] * n_total
    w_reason_col    = [pd.NA] * n_total
    w_hold_f_col    = [pd.NA] * n_total
    w_hold_m_col    = [pd.NA] * n_total
    n_skip_pipeline = 0
    skip_frame_ids  = []

    # 車道寬一律由近場量測：一個資料集目錄＝一段連續片段＝
    # 一個 calibrator。量不到就沿用本片段上一次量到的值；本片段從沒量到過的幀
    # 才用 config 的 last_resort_lane_width（沒設就不輸出 pitch）。狀態與原因
    # 寫進 CSV 並在結尾統計，用了最後手段的幀一眼可辨。
    calibrator = NearfieldWidthCalibrator(
        f_x, f_y, resize_size[0], camera_height)

    image_batch_dir = Path(image_dir_str)
    n_missing_image = 0

    for i, row in enumerate(df.itertuples(index=False)):
        frame_id   = int(row.frame_id)
        image_path = image_batch_dir / IMG_FMT.format(frame_id)

        if not image_path.exists():
            n_missing_image += 1
            continue

        if hasattr(row, "collect_dist_m"):
            calibrator.advance_to(float(row.collect_dist_m))

        try:
            # Full pipeline (road seg → ELSED → paint gate → tracking →
            # fitting + evidence/depth truncation → pitch); the batch runner
            # only needs the pitch curve, so it goes through infer_one and
            # never re-implements the stages (that duplication silently ran
            # the pre-WWH-15 pipeline here once already).
            result = infer_one(
                model, str(image_path), device, resize_size,
                min_segment_length_near, min_segment_length_far,
                num_samples, f_x, f_y, camera_height,
                samples_per_meter=samples_per_meter,
                track_bands=track_bands, method=pitch_method,
                w_real_calibrator=calibrator,
                last_resort_lane_width=last_resort_lane_width)
            pitch_curve = result["pitch_curve"]
            if result["w_real_used"] is not None:
                w_real_used_col[i] = round(float(result["w_real_used"]), 4)
            w_status_col[i] = result["w_real_status"]
            w_reason_col[i] = result["w_real_reason"] or pd.NA
            if result["w_real_hold_frames"] is not None:
                w_hold_f_col[i] = result["w_real_hold_frames"]
            if result["w_real_hold_m"] is not None:
                w_hold_m_col[i] = round(float(result["w_real_hold_m"]), 2)

            if pitch_curve["pitch_at"] is not None:
                z_vis_min_col[i] = round(float(pitch_curve["z_visible_min"]), 2)
                z_vis_max_col[i] = round(float(pitch_curve["z_visible_max"]), 2)
                mae = _compute_mae(pitch_curve, frame_id, gt)
                if mae is not None:
                    profile_mae_col[i] = mae

        except Exception as e:
            skip_frame_ids.append(frame_id)
            n_skip_pipeline += 1
            print(f"  skip frame {frame_id}: {type(e).__name__}: {e}")

    df["z_visible_min"] = z_vis_min_col
    df["z_visible_max"] = z_vis_max_col
    df["profile_mae"]   = profile_mae_col
    df["w_real_used"]        = w_real_used_col
    df["w_real_status"]      = w_status_col
    df["w_real_reason"]      = w_reason_col
    df["w_real_hold_frames"] = w_hold_f_col
    df["w_real_hold_m"]      = w_hold_m_col
    df.to_csv(out_csv, index=False)

    print(f"pipeline_skip={n_skip_pipeline}, missing_image={n_missing_image}, total={n_total}")
    _report_width_status(df)
    print(f"skipped frame ids: {skip_frame_ids}")

    mae = pd.to_numeric(df["profile_mae"], errors="coerce").dropna()
    if len(mae):
        print(f"profile MAE over {len(mae)} frames: "
              f"mean={mae.mean():.4f}  median={mae.median():.4f}  "
              f"p90={mae.quantile(0.9):.4f}  max={mae.max():.4f}")

    problem_df = df.loc[mae[mae > problem_mae_threshold].index, ["frame_id", "profile_mae"]].copy()
    problem_df["image_path"] = problem_df["frame_id"].apply(
        lambda fid: str(image_batch_dir / IMG_FMT.format(int(fid)))
    )
    problem_df = problem_df.sort_values("profile_mae", ascending=False)
    Path(prob_csv).parent.mkdir(parents=True, exist_ok=True)
    problem_df.to_csv(prob_csv, index=False)
    print(f"problem frames (mae > {problem_mae_threshold}): {len(problem_df)} -> {prob_csv}")

    # 圖檔帶資料集名，連續跑多份資料集才不會互相覆蓋（CSV 仍照 config 走）
    dataset_dir = Path(meas_csv).parent
    plot_path = plot_profile_mae(
        df, save_path=_tagged(f"outputs/profile_mae_{dataset_dir.name}.png",
                              args.tag),
        title=f"Per-frame profile MAE — {dataset_dir.name}")
    print(f"MAE plot: {plot_path}")

    if (dataset_dir / "road_profile.csv").exists():
        route_png = _tagged(f"outputs/route_profile_{dataset_dir.name}.png",
                            args.tag)
        print("route profile plot: "
              f"{plot_route_profile(dataset_dir, df, save_path=route_png)}")


if __name__ == "__main__":
    main()
