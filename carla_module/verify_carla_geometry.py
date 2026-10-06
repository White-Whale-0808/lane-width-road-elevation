"""
carla_module/verify_carla_geometry.py
一次性幾何驗證：直接向 CARLA 量測 `w_real` 與 `camera_height`，解除影像的 W/h 簡併。

為什麼需要這支腳本
    影像只能約束比值：w_px = f_x·W / (f_y·h) · (y − cy)。
    所以「車道寬 W 太大」和「相機高 h 太小」在影像上完全分不出來。
    2026-07-27 的診斷是用 A/B 的 MAE 排序否決了後者 —— 論證很強，但終究不是
    直接量測。這支腳本直接問地圖與模擬器，一槌定音。

    推論端 config 目前是 w_real=3.216 / camera_height=1.08，比值 2.9778。
    這個比值是影像量出來的（`debug/carla_gt/check_width_calibration.py`，std 0.003），
    可信；未定的是它該怎麼拆成 W 和 h。本腳本量的就是拆法。

量測項目（對應 to-do.md「第 1 組：一次性驗證」）
    1. `waypoint.lane_width` —— 確認 CARLA 的 3.5 m 是否為車道邊界中心到中心
    2. 標線寬 `left/right_lane_marking.width` —— 驗證式
       inner_to_inner = lane_width − (left.width + right.width)/2 =?= 3.216
       （同時印出標線 type/color，雙黃線那側的 .width 語意要人眼確認）
    3. 相機到路面的實際高度 —— 三種獨立量法交叉比對：
       相機所在處 waypoint、車輛所在處 waypoint、向下射線（cast_ray /
       ground_projection）。三者一致才算數
    4. 相機無畸變、主點在正中心 —— dump 相機 blueprint 的 lens_* 屬性，
       並用 fov 反推 f 與 resize 後的 f_x/f_y，對照 config

量測分兩個階段，因為懸吊會壓縮：
    static  —— 手煞車停住、物理落穩後取樣（幾何理想值）
    driving —— 用與 `get_carlaDataset.py` 完全相同的 TM 對齊 + PID+FF 定速 +
               pure-pursuit 橫向修正行駛取樣（資料集真正的拍攝條件，這才是
               該寫進 config 的值）。橫向修正（`apply_pid_ff_control_with_steering`
               / `--lookahead-m`，預設 6m）取代寫死 `steer=0`——TM 對齊只保證
               轉向輸出連續幾幀 < 0.05，不保證航向零誤差，取樣視窗有數十公尺，
               路稍有彎或殘留航向誤差就會不受控漂移、可能跨到隔壁車道。報告
               第⑥項印出 `lane_offset_m` 與經過的 `lane_id` 證明有沒有守住車道。

使用方式（在跑 CARLA server 的那台機器上）
    uv run --no-sync python carla_module/verify_carla_geometry.py
        [--host HOST] [--port PORT] [--speed KMH] [--camera-fps N]
        [--spawn spectator|map] [--spawn-index N] [--z-offset N]
        [--settle-frames N] [--drive-frames N] [--lookahead-m N]
        [--no-drive] [--out PATH]

    預設從 spectator 位置生成車輛（與 get_carlaDataset.py 相同），
    所以先把 CARLA 視窗的鏡頭移到要量的路面上。不需要顯示視窗、不存圖。

輸出
    console 報告 + `outputs/carla_geometry_verification_<時間戳>.{txt,json}`
    把這兩個檔帶回開發機即可。JSON 含每一幀的原始量測值。
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.env_setup import setup_env
setup_env()

_whl = os.environ.get("CARLA_WHL_PATH", "")
if _whl:
    _carla_api_dir = str(pathlib.Path(_whl).parent.parent)
    if _carla_api_dir not in sys.path:
        sys.path.insert(0, _carla_api_dir)

import argparse
import datetime
import json
import math
import statistics
import time
from typing import Any, Optional

import carla

from carla_module.get_carlaDataset import (
    CAMERA_FOV,
    CAMERA_FWD_X,
    CAMERA_HEIGHT,
    GROUND_LABELS,
    IMG_HEIGHT,
    IMG_WIDTH,
    PHYSICS_WARMUP_TICKS,
    PIDController,
    _wrap_deg,
    apply_pid_ff_control_with_steering,
    ray_ground_z,
)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# 雙線標記：CARLA 的 LaneMarking.width 對這些型別回報的是**單條線寬**
# （實測 SolidSolid 與 Solid 同樣回 0.125），不含第二條線也不含中間間隙
_DOUBLE_MARKING_TYPES = {"SolidSolid", "BrokenBroken", "SolidBroken", "BrokenSolid"}

# 判定用容差
_TOL_WIDTH_M  = 0.02   # 車道寬 / 標線寬推導值
_TOL_HEIGHT_M = 0.02   # 相機高
_TOL_RATIO    = 0.02   # W/h 比值


# ── 引數 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CARLA 幾何一次性驗證（w_real / camera_height）")
    p.add_argument("--host",       default="127.0.0.1", help="CARLA 伺服器位址")
    p.add_argument("--port",       type=int,   default=2000, help="埠號")
    p.add_argument("--timeout",    type=float, default=20.0, help="連線逾時秒數")
    p.add_argument("--speed",      type=float, default=18.0,
                   help="driving 階段目標車速 km/h（預設 18，同資料集）")
    p.add_argument("--camera-fps", type=int,   default=40,
                   help="同步模式 FPS（預設 40，同資料集）")
    p.add_argument("--spawn",      choices=["spectator", "map"], default="spectator",
                   help="車輛生成位置：spectator 鏡頭處（預設）或地圖內建 spawn point")
    p.add_argument("--spawn-index", type=int, default=0,
                   help="--spawn map 時使用第幾個 spawn point（預設 0）")
    p.add_argument("--z-offset",   type=float, default=0.0,
                   help="spectator 生成點的 z 偏移（預設 0）")
    p.add_argument("--settle-frames", type=int, default=60,
                   help="static 階段取樣幀數（預設 60；統計只取後半段，前半留給懸吊落穩）")
    p.add_argument("--drive-frames",  type=int, default=400,
                   help="driving 階段取樣幀數（預設 400 ≈ 10 s ≈ 50 m @18km/h）")
    p.add_argument("--align-frames",  type=int, default=20,
                   help="TM 對齊判定：連續幾幀 steer<0.05（預設 20）")
    p.add_argument("--align-timeout-frames", type=int, default=600,
                   help="TM 對齊最多等幾幀，超過就放棄對齊直接開（預設 600）")
    p.add_argument("--warmup-frames", type=int, default=10,
                   help="TM→手動切換後等幾幀才開始取樣（預設 10）")
    p.add_argument("--no-drive", action="store_true",
                   help="只做 static 階段，不行駛")
    p.add_argument("--lookahead-m", type=float, default=6.0,
                   help="driving 階段 pure-pursuit 轉向修正的前視距離（預設 6 m）；"
                        "取代寫死 steer=0 的假設，避免路稍有彎或 TM 對齊未完全歸零時"
                        "在取樣視窗內持續橫向漂移")
    p.add_argument("--time-scale", type=float, default=0.0,
                   help="每個 tick 後額外 sleep 的倍率（預設 0＝不節流，同步模式"
                        "算多快跑多快，畫面常常像快轉）。1.0＝真實時間播放速度，"
                        "2.0＝一半速度（慢動作），方便肉眼確認車子有沒有開對。"
                        "只影響畫面播放節奏，不影響量測數據（PID/物理仍以"
                        "fixed_delta_seconds 為準）")
    p.add_argument("--out", default=None,
                   help="輸出檔前綴路徑（預設 outputs/carla_geometry_verification_<時間戳>）")
    return p.parse_args()


# ── 小工具 ────────────────────────────────────────────────────────────────────

def _tick(world: carla.World, dt: float, time_scale: float) -> None:
    """
    world.tick() + 選擇性節流。同步模式下 tick() 算完就回傳，不等真實時間，
    伺服器算得比 dt 快就會變成視覺上的快轉。time_scale<=0 維持原行為（不節流，
    量測用最快跑法）；>0 時額外 sleep(dt * time_scale) 讓畫面播放速度貼近真實
    時間，純粹方便肉眼確認，不影響物理／PID（那些都是用 fixed_delta_seconds
    這個模擬時間在算，跟這裡的 sleep 無關）。
    """
    world.tick()
    if time_scale > 0.0:
        time.sleep(dt * time_scale)


def _setup_console() -> None:
    """
    把 stdout/stderr 切成 UTF-8。Windows 主控台預設 cp950，報告裡的 → ① ⚠
    會直接丟 UnicodeEncodeError 讓腳本在最後一步崩掉（輸出檔本身一律 utf-8）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:                                      # noqa: BLE001
            pass


def _load_repo_config() -> dict:
    """讀推論 config 的 pitch_estimation 段，用來對照量測值。讀不到就回空 dict。"""
    try:
        import yaml
        path = _PROJECT_ROOT / "config" / "inference_road_lane_segmentation.yaml"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:                                   # noqa: BLE001
        print(f"[警告] 讀不到 config，判定改用內建預期值：{exc}")
        return {}


def _stats(values: list[Optional[float]]) -> Optional[dict]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return {
        "n":      len(vals),
        "mean":   statistics.fmean(vals),
        "median": statistics.median(vals),
        "std":    statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min":    min(vals),
        "max":    max(vals),
    }


def _fmt_stats(label: str, s: Optional[dict], unit: str = "m") -> str:
    if s is None:
        return f"    {label:<30} —（無有效樣本）"
    return (f"    {label:<30} median {s['median']:+.4f} {unit}   "
            f"mean {s['mean']:+.4f}   std {s['std']:.4f}   "
            f"[{s['min']:+.4f}, {s['max']:+.4f}]   n={s['n']}")


def _marking_info(marking: Any) -> Optional[dict]:
    if marking is None:
        return None
    try:
        return {
            "type":        str(marking.type),
            "color":       str(marking.color),
            "width":       float(marking.width),
            "lane_change": str(marking.lane_change),
        }
    except Exception:                                          # noqa: BLE001
        return None


def _local_to_world(vehicle_tf: carla.Transform,
                    local: carla.Location) -> carla.Location:
    """把車輛座標系的點轉到世界座標。相容 transform() 就地修改 / 回傳兩種語意。"""
    pt  = carla.Location(x=local.x, y=local.y, z=local.z)
    res = vehicle_tf.transform(pt)
    return res if isinstance(res, carla.Location) else pt


def _projection_ground_z(world: carla.World,
                         loc: carla.Location,
                         depth: float = 6.0) -> Optional[float]:
    """
    world.ground_projection：官方的向下投影 API（0.9.12+）。
    同樣要檢查 label —— 它底層也是射線，一樣會打到車身。
    """
    try:
        pt = world.ground_projection(loc, depth)
    except Exception:                                          # noqa: BLE001
        return None
    if pt is None:
        return None
    label = str(getattr(pt, "label", "")).split(".")[-1]
    if label and label not in GROUND_LABELS:
        return None
    return float(pt.location.z)


# ── 單幀量測 ──────────────────────────────────────────────────────────────────

def _measure(world: carla.World,
             carla_map: carla.Map,
             vehicle: carla.Vehicle,
             camera: carla.Sensor,
             phase: str,
             frame_idx: int) -> dict:
    """量一幀。所有高度都以「相機世界 z 減去路面 z」為準。"""
    veh_tf  = vehicle.get_transform()
    cam_loc = camera.get_transform().location

    # 交叉檢查：附掛 sensor 的 get_transform() 是否真的給世界座標
    expect_loc = _local_to_world(
        veh_tf, carla.Location(x=CAMERA_FWD_X, z=CAMERA_HEIGHT))
    mount_mismatch = math.dist(
        (cam_loc.x, cam_loc.y, cam_loc.z),
        (expect_loc.x, expect_loc.y, expect_loc.z))

    wp_cam = carla_map.get_waypoint(cam_loc, project_to_road=True)
    wp_veh = carla_map.get_waypoint(veh_tf.location, project_to_road=True)

    ray_z, ray_label = ray_ground_z(world, cam_loc)
    proj_z           = _projection_ground_z(world, cam_loc)

    rec: dict[str, Any] = {
        "phase":     phase,
        "frame_idx": frame_idx,
        "cam_world": [cam_loc.x, cam_loc.y, cam_loc.z],
        "veh_world": [veh_tf.location.x, veh_tf.location.y, veh_tf.location.z],
        "mount_mismatch_m": mount_mismatch,
        # 姿態：車身 vs 路面（第 2 組 road_pitch_deg 的證據）
        # 一律 wrap 到 (-180, 180]：CARLA 的平路 waypoint pitch 實測會回 360.0，
        # 不 wrap 的話原始統計那欄會印出「路面 pitch +360°」這種讀不懂的值
        "body_pitch_deg":  _wrap_deg(float(veh_tf.rotation.pitch)),
        "body_roll_deg":   _wrap_deg(float(veh_tf.rotation.roll)),
        # 高度：三種獨立量法（h_perp 是 h_wp_cam 去掉坡度假象後的真值）
        "h_wp_cam":  None,
        "h_perp":    None,
        "h_wp_veh":  None,
        "h_ray":     None if ray_z  is None else cam_loc.z - ray_z,
        "h_proj":    None if proj_z is None else cam_loc.z - proj_z,
        "ray_label": ray_label,
        # 車道中心線橫向偏移（證明 driving 階段有沒有漂出車道，見 _pure_pursuit_steer）
        "lane_offset_m":  None,
        # 車道幾何
        "lane_width":     None,
        "inner_to_inner": None,
        "road_pitch_deg": None,
        "lane_id":        None,
        "road_id":        None,
        "left_marking":   None,
        "right_marking":  None,
        "marking_valid":  False,   # 兩側都有真實標記（排除路口的 NONE）
        "marking_double": False,   # 任一側是雙線 → inner_to_inner 會少扣
    }

    if wp_veh is not None:
        rec["h_wp_veh"] = cam_loc.z - float(wp_veh.transform.location.z)
        # get_waypoint(project_to_road=True) 回傳車道中心線上最近點，
        # 車輛位置到這個點的水平距離就是橫向（cross-track）偏移
        rec["lane_offset_m"] = math.dist(
            (veh_tf.location.x, veh_tf.location.y),
            (wp_veh.transform.location.x, wp_veh.transform.location.y))

    if wp_cam is not None:
        rec["h_wp_cam"]       = cam_loc.z - float(wp_cam.transform.location.z)
        rec["lane_width"]     = float(wp_cam.lane_width)
        rec["road_pitch_deg"] = _wrap_deg(float(wp_cam.transform.rotation.pitch))
        rec["lane_id"]        = int(wp_cam.lane_id)
        rec["road_id"]        = int(wp_cam.road_id)

        # 針孔模型要的 h 是「相機到路面的垂直距離」，但 cam_z − road_z 是
        # 世界座標的**垂直高差**。上坡 θ 時兩者差 1/cos(θ)：實測 12° 路段
        # h_raw 漲到 1.1069，乘上 cos 後回到 1.0813，與平路的 1.0818 一致。
        # 不修正的話，坡度越大量到的相機越「高」，是純粹的幾何假象。
        rec["h_perp"] = rec["h_wp_cam"] * math.cos(
            math.radians(rec["road_pitch_deg"]))

        left  = _marking_info(getattr(wp_cam, "left_lane_marking",  None))
        right = _marking_info(getattr(wp_cam, "right_lane_marking", None))
        rec["left_marking"]  = left
        rec["right_marking"] = right
        rec["marking_valid"] = bool(
            left and right
            and left["type"] != "NONE" and right["type"] != "NONE"
            and left["width"] > 0.0 and right["width"] > 0.0)
        # 雙線標記（SolidSolid 等）的 .width 實測是「單條線寬」，不含第二條
        # 也不含中間間隙 —— 見 _verdicts 第②項的說明
        rec["marking_double"] = bool(
            left and right and any(
                m["type"] in _DOUBLE_MARKING_TYPES for m in (left, right)))

        if rec["marking_valid"]:
            # 車道寬是邊界中心到中心；內側邊到內側邊要各扣半個標線寬。
            # 注意：這條公式只對單線標記成立，雙線那側會少扣（見第②項）
            rec["inner_to_inner"] = (
                rec["lane_width"] - (left["width"] + right["width"]) / 2.0)

    return rec


# ── 相機內參 / 畸變（項目 4）───────────────────────────────────────────────────

def _camera_report(cam_bp: carla.ActorBlueprint, cfg: dict) -> dict:
    """dump blueprint 屬性 + 用 fov 反推內參，對照 config。"""
    def _attr_value(attr: Any) -> str:
        for getter in ("as_str", "as_float", "as_int", "as_bool"):
            try:
                return str(getattr(attr, getter)())
            except Exception:                                  # noqa: BLE001
                continue
        return str(attr)

    attrs: dict[str, str] = {}
    try:
        for attr in cam_bp:
            attrs[str(attr.id)] = _attr_value(attr)
    except Exception:                                          # noqa: BLE001
        for name in ("fov", "image_size_x", "image_size_y", "lens_circle_falloff",
                     "lens_circle_multiplier", "lens_k", "lens_kcube",
                     "chromatic_aberration_intensity", "lens_x_size", "lens_y_size"):
            if cam_bp.has_attribute(name):
                attrs[name] = _attr_value(cam_bp.get_attribute(name))

    lens_attrs = {k: v for k, v in attrs.items()
                  if "lens" in k or "distort" in k or "chromatic" in k}

    f_native = IMG_WIDTH / (2.0 * math.tan(math.radians(CAMERA_FOV) / 2.0))

    resize = (cfg.get("input", {}) or {}).get("resize_size") or [512, 1024]
    r_h, r_w = int(resize[0]), int(resize[1])   # config 是 [height, width]

    pitch_cfg = cfg.get("pitch_estimation", {}) or {}
    return {
        "blueprint_attributes": attrs,
        "lens_attributes":      lens_attrs,
        "native": {
            "width": IMG_WIDTH, "height": IMG_HEIGHT, "fov_deg": CAMERA_FOV,
            "f_px": f_native, "cx": IMG_WIDTH / 2.0, "cy": IMG_HEIGHT / 2.0,
        },
        "resized": {
            "width": r_w, "height": r_h,
            "f_x": f_native * (r_w / IMG_WIDTH),
            "f_y": f_native * (r_h / IMG_HEIGHT),
            "cx":  r_w / 2.0, "cy": r_h / 2.0,
        },
        "config": {
            "f_x": pitch_cfg.get("f_x"),
            "f_y": pitch_cfg.get("f_y"),
        },
    }


# ── 判定 ──────────────────────────────────────────────────────────────────────

def _verdicts(records: list[dict], cfg: dict) -> tuple[list[str], dict]:
    """把量測值和 config 對照，產生人可讀的判定。"""
    pitch_cfg  = cfg.get("pitch_estimation", {}) or {}
    cfg_w      = float(pitch_cfg.get("last_resort_lane_width",        3.216))
    cfg_h      = float(pitch_cfg.get("camera_height", 1.08))
    cfg_ratio  = cfg_w / cfg_h

    # 以行駛階段為準（資料集真正的拍攝條件）；沒有就退回落穩後的 static
    drive  = [r for r in records if r["phase"] == "driving"]
    static = [r for r in records if r["phase"] == "static"]
    prim      = drive or static or records
    prim_name = "driving" if drive else ("static" if static else "all")

    lines: list[str] = []
    out: dict[str, Any] = {"primary_phase": prim_name,
                           "config": {"w_real": cfg_w, "camera_height": cfg_h,
                                      "ratio": cfg_ratio}}

    # ① 車道寬定義
    lw = _stats([r["lane_width"] for r in prim])
    if lw is not None:
        out["lane_width"] = lw
        lines.append(
            f"① waypoint.lane_width = {lw['median']:.4f} m "
            f"(std {lw['std']:.4f}, n={lw['n']})")
        if abs(lw["median"] - 3.5) <= _TOL_WIDTH_M:
            lines.append("   → 確認為 3.5 m，與先前假設一致。")
        else:
            lines.append(f"   → 不是 3.5 m！這條路的車道寬是 {lw['median']:.4f} m，"
                         "推導與 config 都要重算。")

    # ② 標線寬 → inner-to-inner（只採兩側都有真實標記的樣本；路口的 NONE
    #    會讓 inner_to_inner 退化成 lane_width 本身，混進統計就是污染）
    valid  = [r for r in prim if r.get("marking_valid")]
    n_skip = len(prim) - len(valid)
    ii = _stats([r["inner_to_inner"] for r in valid])
    mk = _stats([r["left_marking"]["width"]  for r in valid] +
                [r["right_marking"]["width"] for r in valid])
    double = [r for r in valid if r.get("marking_double")]
    if ii is not None:
        out["inner_to_inner"] = ii
        out["marking_width"]  = mk
        out["n_skipped_no_marking"] = n_skip
        out["n_double_marking"]     = len(double)
        if mk is not None:
            lines.append(f"② 標線寬 median {mk['median']:.4f} m "
                         f"(std {mk['std']:.4f}, n={mk['n']})"
                         + (f"   [略過 {n_skip} 幀無標記]" if n_skip else ""))
        lines.append(f"   inner-to-inner = lane_width - (l+r)/2 = "
                     f"{ii['median']:.4f} m (std {ii['std']:.4f})")

        if double:
            # 實測：SolidSolid（雙黃）與 Solid（單白）都回報 width=0.125，
            # 也就是 .width 是「單條線寬」，不含第二條線也不含中間間隙。
            # 因此上面那條公式在雙線那側**少扣**了 (間隙/2 + 一條線寬)，
            # 算出來的 inner-to-inner 偏大。這不是量測誤差，是公式不適用。
            types = sorted({m["type"] for r in double
                            for m in (r["left_marking"], r["right_marking"])
                            if m["type"] in _DOUBLE_MARKING_TYPES})
            lines.append(f"   ⚠ 有 {len(double)}/{len(valid)} 幀是雙線標記"
                         f"（{', '.join(types)}）。CARLA 的 .width 回報的是"
                         "單條線寬，不含第二條與間隙，所以上面這個值在雙線那側"
                         "**少扣**、偏大。**不可直接當成 w_real。**")
            lines.append("      → 雙線路段請改以第④項反推：由直接量到的 h 與"
                         "影像約束的 W/h 得到 W，再回頭推間隙。")
        else:
            delta = ii["median"] - cfg_w
            if abs(delta) <= _TOL_WIDTH_M:
                lines.append(f"   → 與 config w_real={cfg_w} 相符（差 {delta:+.4f} m）。"
                             "反推值直接被地圖證實。")
            else:
                lines.append(f"   → 與 config w_real={cfg_w} 不符（差 {delta:+.4f} m）。"
                             "w_real 要改，且整批 MAE 基準需重跑。")

    # ③ 相機高：多種量法。主量測用 h_perp（去掉坡度假象）
    h_perp   = _stats([r["h_perp"]   for r in prim])
    h_wp_cam = _stats([r["h_wp_cam"] for r in prim])
    h_wp_veh = _stats([r["h_wp_veh"] for r in prim])
    h_ray    = _stats([r["h_ray"]    for r in prim])
    h_proj   = _stats([r["h_proj"]   for r in prim])
    out["height"] = {"perp": h_perp, "wp_cam": h_wp_cam, "wp_veh": h_wp_veh,
                     "ray": h_ray, "proj": h_proj}

    # h_perp 是主量測：cam_z − road_z 只是世界座標的垂直高差，針孔模型要的是
    # 到路面的垂直距離，上坡時前者被 1/cos(θ) 放大。其餘量法留作交叉驗證。
    best, best_name = None, None
    for cand, name in ((h_perp, "waypoint@camera×cos(坡度)"),
                       (h_wp_cam, "waypoint@camera"), (h_ray, "cast_ray"),
                       (h_proj, "ground_projection")):
        if cand is not None:
            best, best_name = cand, name
            break

    if best is not None:
        out["height_primary"] = {"source": best_name, **best}
        lines.append(f"③ 相機到路面高度（採用 {best_name}）= "
                     f"{best['median']:.4f} m (std {best['std']:.4f}, n={best['n']})")
        for cand, name in ((h_perp, "垂直距離(×cos坡度)"), (h_wp_cam, "垂直高差 cam_z-road_z"),
                           (h_ray, "cast_ray"), (h_proj, "ground_projection"),
                           (h_wp_veh, "waypoint@vehicle")):
            tag = "（無樣本，可能全被車身擋住或此版無此 API）" if cand is None else ""
            val = "—" if cand is None else f"{cand['median']:.4f} m"
            lines.append(f"      {name:<24} {val} {tag}")
        if h_perp is not None and h_wp_cam is not None:
            lines.append(f"      坡度修正讓 std 由 {h_wp_cam['std']:.4f} 收斂到 "
                         f"{h_perp['std']:.4f} —— 收斂即證實原始值的變動是坡度假象")
        if h_wp_veh is not None and h_perp is not None and \
                abs(h_wp_veh["median"] - h_perp["median"]) > 0.05:
            lines.append("      （waypoint@vehicle 在坡上本來就會偏大：相機前移 "
                         f"{CAMERA_FWD_X} m，那裡的路面比車輛所在處高，僅供對照）")
        spread = [c["median"] for c in (h_wp_cam, h_ray, h_proj) if c is not None]
        if len(spread) > 1:
            lines.append(f"      三種量法全距 {max(spread) - min(spread):.4f} m"
                         "（>0.02 m 就要懷疑路拱或射線打到車身）")
        d_cfg = best["median"] - cfg_h
        if abs(d_cfg) <= _TOL_HEIGHT_M:
            lines.append(f"   → 與 config camera_height={cfg_h} 相符（差 {d_cfg:+.4f} m）"
                         "，1.18 假設確定被否決。")
        elif abs(best["median"] - 1.18) <= _TOL_HEIGHT_M:
            lines.append("   → 竟然接近 1.18 m！那簡併是反過來的，"
                         "w_real 的修正需整個重新檢討。")
        else:
            lines.append(f"   → 與 config camera_height={cfg_h} 差 {d_cfg:+.4f} m，"
                         "兩個假設都不成立，需重新推導。")

    # ④ W/h 比值 —— 只是自洽性檢查，**不能**用來判斷拆法
    #
    #    注意不要誤讀這一項：比值就是簡併本身。3.216/1.08 = 2.9778 與
    #    3.5/1.18 = 2.9660 只差 0.4%，兩個競爭假設都會通過這個檢查。
    #    真正定案拆法的是 ②（地圖直接給 W）和 ③（地圖直接給 h）。
    #    ④ 的作用只有一個：若比值和影像對不上，說明 ②③ 或掛載假設有問題。
    if best is not None and best["median"] > 1e-6:
        # 直接量到 h 之後，影像的 W/h 就把 W 定死了 —— 這條路徑完全繞開
        # 標線 .width 的語意問題，在雙線路段是唯一可信的 W
        w_from_h = cfg_ratio * best["median"]
        out["w_from_measured_h"] = w_from_h
        lines.append(f"④ 由直接量到的 h 反推 W = {cfg_ratio:.4f} × "
                     f"{best['median']:.4f} = {w_from_h:.4f} m"
                     f"（config w_real={cfg_w}，差 {w_from_h - cfg_w:+.4f} m）")
        if abs(w_from_h - cfg_w) <= _TOL_WIDTH_M:
            lines.append("   → config 的 w_real 成立。h 是直接量的、W/h 是影像量的，"
                         "兩者相乘不經過標線語意，所以這條推導最可信。")
        else:
            lines.append(f"   → 與 config 差 {w_from_h - cfg_w:+.4f} m 超過容差 "
                         f"{_TOL_WIDTH_M}，w_real 要改，且整批 MAE 基準需重跑。")

        if ii is not None:
            ratio = ii["median"] / best["median"]
            out["measured_ratio"] = ratio
            lines.append(f"   對照：地圖 .width 公式給的 W/h = {ii['median']:.4f} / "
                         f"{best['median']:.4f} = {ratio:.4f}"
                         f"（影像約束值 {cfg_ratio:.4f}，差 {ratio - cfg_ratio:+.4f}）")
            if abs(ratio - cfg_ratio) <= _TOL_RATIO:
                lines.append("   → 兩條路徑一致。（注意比值本身分不出拆法 —— "
                             "競爭假設 3.5/1.18=2.9660 只差 0.4%，也會通過。）")
            elif double and lw is not None:
                # 已知原因：雙線那側的 .width 少扣。把差額拆回幾何，驗證這個
                # 解釋在物理上說不說得通。
                #   total_sub   = lane_width − W          （兩側內縮總量）
                #   單線側內縮  = 線寬/2
                #   雙線側內縮  = total_sub − 單線側      = 間隙/2 + 線寬
                w_dbl = _stats([m["width"] for r in double
                                for m in (r["left_marking"], r["right_marking"])
                                if m["type"] in _DOUBLE_MARKING_TYPES])
                w_sgl = _stats([m["width"] for r in double
                                for m in (r["left_marking"], r["right_marking"])
                                if m["type"] not in _DOUBLE_MARKING_TYPES])
                if w_dbl is not None and w_sgl is not None:
                    total_sub = lw["median"] - w_from_h
                    sub_sgl   = w_sgl["median"] / 2.0
                    sub_dbl   = total_sub - sub_sgl
                    gap       = 2.0 * (sub_dbl - w_dbl["median"])
                    lines.append(
                        f"   → 差額由第②項的雙線 .width 語意解釋。拆回幾何："
                        f"兩側內縮總量 {total_sub:.4f} m，單線側 {sub_sgl:.4f} m，"
                        f"故雙線側內縮 {sub_dbl:.4f} m = 間隙/2 + 線寬 "
                        f"{w_dbl['median']:.4f} → 隱含間隙 {gap:.4f} m。")
                    if 0.0 <= gap <= 0.4:
                        lines.append("      間隙落在合理範圍，解釋成立。"
                                     "以第④項的 W 為準。")
                    else:
                        lines.append(f"      ⚠ 隱含間隙 {gap:.4f} m 不合理，"
                                     "雙線語意之外還有別的誤差來源，需再查。")
                else:
                    lines.append("   → 差額推測來自雙線 .width 語意，"
                                 "但兩側都是雙線／都非雙線，無法拆回間隙。")
            else:
                lines.append(f"   → 地圖與影像矛盾（差 {ratio - cfg_ratio:+.4f}，"
                             f"容差 {_TOL_RATIO}）且標線非雙線，無現成解釋。"
                             "回頭查 mount_mismatch、是否量在路拱上。")

    # 額外：車身姿態 vs 路面坡度（第 2 組的動機證據）
    bp = _stats([r["body_pitch_deg"] for r in prim])
    rp = _stats([r["road_pitch_deg"] for r in prim])
    if bp is not None and rp is not None:
        diff = _stats([_wrap_deg(r["body_pitch_deg"] - r["road_pitch_deg"]) for r in prim
                       if r["road_pitch_deg"] is not None])
        out["pitch"] = {"body": bp, "road": rp, "body_minus_road": diff}
        lines.append(f"⑤ 車身 pitch median {bp['median']:+.4f}° "
                     f"(std {bp['std']:.4f})   路面 pitch median "
                     f"{rp['median']:+.4f}° (std {rp['std']:.4f})")
        if diff is not None:
            lines.append(f"   車身 - 路面 = {diff['median']:+.4f}° "
                         f"(std {diff['std']:.4f}, 全距 "
                         f"{diff['min']:+.3f}~{diff['max']:+.3f})")
            lines.append("   → 這個差就是懸吊/加速造成的假象，"
                         "第 2 組要把 road_pitch_deg 獨立記一欄。")

    # 掛載一致性檢查
    mm = _stats([r["mount_mismatch_m"] for r in prim])
    if mm is not None:
        out["mount_mismatch"] = mm
        if mm["max"] > 0.01:
            lines.append(f"⚠ camera.get_transform() 與手算掛載位置差 "
                         f"max {mm['max']:.4f} m —— 高度量測的前提要重新確認。")

    # ⑥ driving 階段是否守在同一車道（pure-pursuit 修正是否真的有效）
    if drive:
        off = _stats([r["lane_offset_m"] for r in drive])
        lane_ids = sorted({r["lane_id"] for r in drive if r["lane_id"] is not None})
        out["lane_tracking"] = {"offset": off, "lane_ids": lane_ids}
        if off is not None:
            lines.append(f"⑥ driving 階段車道中心線橫向偏移 median {off['median']:.4f} m "
                         f"(std {off['std']:.4f})   全距 [{off['min']:.4f}, {off['max']:.4f}] m"
                         f"   經過 lane_id={lane_ids}")
            if len(lane_ids) > 1:
                lines.append("   → ⚠ driving 階段 lane_id 不只一個，車子中途換了車道／"
                             "壓到對向車道，這批樣本混到不同車道的資料，不能直接採信。")
            elif off["max"] > 0.5:
                lines.append(f"   → ⚠ 最大偏移 {off['max']:.4f} m 偏大，"
                             "考慮調小 --lookahead-m 讓修正更即時，或檢查對齊/路段是否有急彎。")
            else:
                lines.append("   → 全程守在同一車道、偏移量小，driving 階段的取樣位置可信。")

    return lines, out


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    _setup_console()
    args = parse_args()
    cfg  = _load_repo_config()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world     = client.get_world()
    carla_map = world.get_map()
    map_name  = carla_map.name.split("/")[-1]
    print(f"[初始化] 連線至地圖：{carla_map.name}")

    bp_lib     = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")

    if args.spawn == "spectator":
        spawn_tf = world.get_spectator().get_transform()
        spawn_tf.location.z    += args.z_offset
        spawn_tf.rotation.pitch = 0.0
        spawn_tf.rotation.roll  = 0.0
    else:
        points = carla_map.get_spawn_points()
        if not points:
            raise RuntimeError("這張地圖沒有內建 spawn point，請用 --spawn spectator")
        spawn_tf = points[args.spawn_index % len(points)]

    vehicle: Optional[carla.Vehicle] = world.try_spawn_actor(vehicle_bp, spawn_tf)
    if vehicle is None:
        raise RuntimeError(
            "無法生成車輛。--spawn spectator 時請把鏡頭移到可通行路面，"
            "或用 --z-offset 調整，或改用 --spawn map")
    vehicle.set_autopilot(False)
    print(f"[初始化] 車輛生成於：{spawn_tf.location}")

    original_settings = world.get_settings()
    camera: Optional[carla.Sensor] = None
    tm = None
    records:  list[dict] = []
    cam_info: dict       = {}

    dt = 1.0 / args.camera_fps
    if args.time_scale > 0.0:
        print(f"[初始化] --time-scale={args.time_scale}：畫面節流到約真實時間的 "
              f"{1.0 / args.time_scale:.2f}x，方便肉眼確認（不影響量測數據）")

    try:
        settings = world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = dt
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)

        # 相機：與資料集完全相同的規格與掛載，但不 listen（本腳本不需要影像）
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        cam_bp.set_attribute("fov", str(CAMERA_FOV))
        cam_tf = carla.Transform(carla.Location(x=CAMERA_FWD_X, z=CAMERA_HEIGHT))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        cam_info = _camera_report(cam_bp, cfg)

        print(f"[初始化] 物理預熱 {PHYSICS_WARMUP_TICKS} ticks...")
        for _ in range(PHYSICS_WARMUP_TICKS):
            _tick(world, dt, args.time_scale)

        # ── static 階段：手煞車停住，量幾何理想值 ────────────────────────────
        print(f"[static] 停止取樣 {args.settle_frames} 幀（統計只取後半段）...")
        for i in range(args.settle_frames):
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
            _tick(world, dt, args.time_scale)
            records.append(_measure(world, carla_map, vehicle, camera, "static", i))

        # 前半段留給懸吊落穩，只保留後半段進統計
        cut = args.settle_frames // 2
        for rec in records[:cut]:
            rec["phase"] = "static_settling"

        # ── driving 階段：與資料集相同的 TM 對齊 + PID+FF 定速 ───────────────
        if not args.no_drive:
            target_mps = args.speed / 3.6
            pid = PIDController(kp=1.0, ki=0.25, kd=0.15, dt=dt,
                                integral_limit=3.0)

            # static 階段拉了手煞車，交給 TM 前先放掉
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))
            vehicle.set_autopilot(True, tm.get_port())
            tm.ignore_lights_percentage(vehicle, 100.0)
            tm.ignore_signs_percentage(vehicle,  100.0)
            tm.set_desired_speed(vehicle, args.speed)

            print(f"[driving] TM 對齊車道中（需連續 {args.align_frames} 幀 steer<0.05）...")
            aligned = 0
            for _ in range(args.align_timeout_frames):
                _tick(world, dt, args.time_scale)
                aligned = aligned + 1 if abs(vehicle.get_control().steer) < 0.05 else 0
                if aligned >= args.align_frames:
                    break
            else:
                print("[警告] 對齊逾時，仍繼續（橫向位置可能不在車道中央）")

            vehicle.set_autopilot(False)
            pid.reset()
            for _ in range(args.warmup_frames):
                apply_pid_ff_control_with_steering(
                    vehicle, carla_map, target_mps, pid, 0.015, args.lookahead_m)
                _tick(world, dt, args.time_scale)

            print(f"[driving] 開始取樣 {args.drive_frames} 幀 "
                  f"(@{args.speed} km/h ≈ {args.drive_frames / args.camera_fps:.1f} s "
                  f"≈ {args.speed / 3.6 * args.drive_frames / args.camera_fps:.0f} m)...")
            for i in range(args.drive_frames):
                apply_pid_ff_control_with_steering(
                    vehicle, carla_map, target_mps, pid, 0.015, args.lookahead_m)
                _tick(world, dt, args.time_scale)
                records.append(_measure(world, carla_map, vehicle, camera, "driving", i))
                if (i + 1) % 100 == 0:
                    print(f"    ...{i + 1}/{args.drive_frames}")

    finally:
        print("[清理] 恢復非同步模式...")
        try:
            world.apply_settings(original_settings)
            if tm is not None:
                tm.set_synchronous_mode(False)
        except Exception as exc:                                # noqa: BLE001
            print(f"[警告] 還原世界設定失敗：{exc}")
        for actor in (camera, vehicle):
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:                                   # noqa: BLE001
                pass

    if not records:
        print("[錯誤] 沒有取到任何樣本，不產生報告。")
        return

    _write_report(args, cfg, map_name, records, cam_info)


def _write_report(args: argparse.Namespace,
                  cfg: dict,
                  map_name: str,
                  records: list[dict],
                  cam_info: dict) -> None:
    _setup_console()
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (pathlib.Path(args.out) if args.out else
              _PROJECT_ROOT / "outputs" / f"carla_geometry_verification_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    verdict_lines, verdict = _verdicts(records, cfg)

    L: list[str] = []
    L.append("=" * 78)
    L.append("CARLA 幾何驗證報告（to-do.md 第 1 組：一次性驗證）")
    L.append("=" * 78)
    L.append(f"時間      : {ts}")
    L.append(f"地圖      : {map_name}")
    L.append(f"車速      : {args.speed} km/h    FPS: {args.camera_fps}")
    L.append(f"相機掛載  : x={CAMERA_FWD_X} m  z={CAMERA_HEIGHT} m  "
             f"fov={CAMERA_FOV}°  {IMG_WIDTH}x{IMG_HEIGHT}")
    phases = {}
    for r in records:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    L.append("樣本      : " + "  ".join(f"{k}={v}" for k, v in phases.items()))
    lanes = sorted({(r["road_id"], r["lane_id"]) for r in records
                    if r["road_id"] is not None})
    L.append(f"經過車道  : {len(lanes)} 個 (road_id, lane_id) 組合")
    L.append("")

    L.append("── 判定 " + "─" * 68)
    if verdict_lines:
        L.extend("  " + line for line in verdict_lines)
    else:
        L.append("  ⚠ 沒有任何項目量到有效值 —— waypoint 投影全部失敗？"
                 "確認車輛生成在 OpenDRIVE 有定義的路面上（--spawn map 試試）。")
    L.append("")

    L.append("── 各階段原始統計 " + "─" * 59)
    for phase in ("static", "driving"):
        sub = [r for r in records if r["phase"] == phase]
        if not sub:
            continue
        L.append(f"  [{phase}] n={len(sub)}")
        L.append(_fmt_stats("lane_offset",      _stats([r["lane_offset_m"]  for r in sub])))
        L.append(_fmt_stats("lane_width",       _stats([r["lane_width"]     for r in sub])))
        L.append(_fmt_stats("inner_to_inner",   _stats([r["inner_to_inner"] for r in sub])))
        L.append(_fmt_stats("h_perp (×cos坡度, 主值)",
                            _stats([r.get("h_perp") for r in sub])))
        L.append(_fmt_stats("h_ray (cast_ray)", _stats([r["h_ray"]     for r in sub])))
        L.append(_fmt_stats("h_proj (ground)",  _stats([r["h_proj"]    for r in sub])))
        L.append(_fmt_stats("h_wp_cam",         _stats([r["h_wp_cam"]  for r in sub])))
        L.append(_fmt_stats("h_wp_veh",         _stats([r["h_wp_veh"]  for r in sub])))
        L.append(_fmt_stats("body_pitch",  _stats([r["body_pitch_deg"] for r in sub]), "°"))
        L.append(_fmt_stats("road_pitch",  _stats([r["road_pitch_deg"] for r in sub]), "°"))
        L.append("")

    L.append("── 標線語意（雙黃線那側請人眼確認 .width 是否含間隙）" + "─" * 24)
    seen: set[tuple] = set()
    for r in records:
        for side in ("left_marking", "right_marking"):
            m = r.get(side)
            if not m:
                continue
            key = (side, m["type"], m["color"], round(m["width"], 4))
            if key in seen:
                continue
            seen.add(key)
            L.append(f"    {side:<14} type={m['type']:<28} "
                     f"color={m['color']:<14} width={m['width']:.4f} m")
    L.append("")

    if cam_info:
        native, resized = cam_info["native"], cam_info["resized"]
        L.append("── 相機內參與畸變（項目 4）" + "─" * 50)
        L.append(f"    原生 {native['width']}x{native['height']} fov={native['fov_deg']}° "
                 f"→ f={native['f_px']:.2f} px  (cx,cy)=({native['cx']:.1f},{native['cy']:.1f})")
        L.append(f"    resize {resized['width']}x{resized['height']} "
                 f"→ f_x={resized['f_x']:.2f}  f_y={resized['f_y']:.2f}  "
                 f"(cx,cy)=({resized['cx']:.1f},{resized['cy']:.1f})")
        c_fx, c_fy = cam_info["config"]["f_x"], cam_info["config"]["f_y"]
        if c_fx is not None:
            L.append(f"    config  f_x={c_fx}  f_y={c_fy}  → "
                     f"差 {resized['f_x'] - float(c_fx):+.2f} / "
                     f"{resized['f_y'] - float(c_fy):+.2f} px")
        # CARLA 的 rgb 相機只有在 lens_circle_multiplier > 0 時才真的套用鏡頭
        # 畸變；lens_k / lens_kcube 單獨存在不代表有畸變。所以要看這一項，
        # 不能只因為「有 lens_* 屬性」就下結論。
        lens = cam_info["lens_attributes"]

        def _num(key: str) -> Optional[float]:
            try:
                return float(lens[key])
            except (KeyError, TypeError, ValueError):
                return None

        mult   = _num("lens_circle_multiplier")
        chroma = _num("chromatic_aberration_intensity")
        active = [f"{k}={v}" for k, v in ((("lens_circle_multiplier"), mult),
                                          ("chromatic_aberration_intensity", chroma))
                  if v is not None and abs(v) > 1e-9]
        if active:
            L.append(f"    ⚠ 畸變/色差**有開啟**：{', '.join(active)} —— "
                     "針孔假設不成立，f_x/f_y 與主點都要重新標定。")
        elif mult is not None:
            L.append(f"    畸變關閉（lens_circle_multiplier={mult}），理想針孔成立："
                     f"主點嚴格在影像中心 cy={resized['cy']:.1f}。")
            L.append("    → 平坦幀量到 253.3 px 的那 2.7 px 差不是內參問題，"
                     "與「車身俯仰假象」的判斷一致。")
        else:
            L.append("    （blueprint 沒有 lens_circle_multiplier，"
                     "無法自動判定畸變；請人眼看下方屬性清單）")

        if lens:
            L.append("    lens/畸變屬性:")
            for k, v in sorted(lens.items()):
                L.append(f"        {k:<34} = {v}")
        L.append("")

    report = "\n".join(L)
    print()
    print(report)

    txt_path  = prefix.with_suffix(".txt")
    json_path = prefix.with_suffix(".json")
    txt_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps({
        "timestamp":   ts,
        "map":         map_name,
        "args":        vars(args),
        "mount":       {"forward_x": CAMERA_FWD_X, "height": CAMERA_HEIGHT,
                        "fov_deg": CAMERA_FOV,
                        "img_width": IMG_WIDTH, "img_height": IMG_HEIGHT},
        "camera_info": cam_info,
        "verdict":     verdict,
        "records":     records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[輸出] {txt_path}")
    print(f"[輸出] {json_path}")
    print("       把這兩個檔帶回開發機即可。")


if __name__ == "__main__":
    main()
