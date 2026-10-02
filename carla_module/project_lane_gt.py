"""
carla_module/project_lane_gt.py
把 CARLA 地圖的車道幾何即時投影到相機影像上，用來目視驗證 w_real。

背景（carla_module/verify_carla_geometry.py 在 Town03 上坡段已確立，不重推）
    - waypoint.lane_width = 3.5000 m，是「車道邊界中心到中心」
    - LaneMarking.width 是「單條線寬」：SolidSolid（雙黃）與 Solid（單白）都
      回報 0.125，不含第二條線也不含中間間隙
    - 相機到路面垂直距離 h = 1.0816 m（已做坡度修正）
    - 邊界中心（lane_width/2 = 1.75）落在標線油漆的正中央，實測差 9 mm。
      左側是雙黃線，所以 1.75 落在兩條黃線中間的**間隙中央**
    - 以間隙 ≈ 標線寬 0.125 推得真實內緣：
          左（雙黃）= 1.75 - (0.125 + 0.125/2) = 1.5625
          右（單白）= 1.75 - 0.125/2           = 1.6875
          inner-to-inner = 3.25
      與 debug/carla_gt/measure_paint_edges.py 直接量到的 3.243~3.255 相符
    - 但 config 仍維持 w_real=3.216：A/B 實測 3.245 會讓 batch MAE 由
      0.2331 惡化到 0.2492（退步 388/451 幀）。w_real 只做尺度變換
      （pitch 值不變、只移 z 軸），3.216 目前在補償 GT 距離軸的系統誤差。
      本腳本不改 config

這支腳本要回答的問題
    「地圖說的車道內側邊，到底落在影像的哪裡？」
    把互相競爭的假設同時畫出來，讓影像自己裁決，四組偏移量（沿
    transform.get_right_vector() 方向，± 對稱）：
        1. 車道中心線              offset = 0                （白）
        2. 邊界中心                offset = ±lane_width/2      （黃）
        3. 天真公式的內側邊        offset = ±(lane_width/2 − marking.width/2)
                                                                （青，故意不修正
           雙線少扣的已知瑕疵——這正是要在畫面上暴露出來的東西）
        4. config w_real 的內側邊  offset = ±w_real/2           （洋紅）
    第 3 與第 4 誰壓在標線漆的內緣上，誰就是對的；這條路徑完全不經過寬度
    回歸，是獨立證據。雙線那側（已知 SolidSolid）預期第 3 組會蓋不到真正
    的內緣（因為 .width 少算了第二條線與間隙），單線那側則預期會準。

投影數學（按規格寫，不猜軸向）
    f = IMG_WIDTH / (2 * tan(radians(fov)/2))
    K = [[f,0,W/2],[0,f,H/2],[0,0,1]]
    M = np.array(camera.get_transform().get_inverse_matrix())   # world -> sensor
    p_sensor = M @ [x,y,z,1]
    CARLA sensor 軸 x 前 / y 右 / z 上 → 標準相機 x 右 / y 下 / z 前：
        std = [p_sensor[1], -p_sensor[2], p_sensor[0]]
    u = f*std[0]/std[2] + cx ; v = f*std[1]/std[2] + cy ; 只畫 std[2] > 0.1 的點

取樣車道點
    從相機所在處（不是車輛所在處——相機前移 1.5m，坡上兩處路面高度不同）
    的 waypoint 出發，用 cur = cur.next(step)[0] 逐步往前串（step 預設
    0.5m，最遠 60m），不用 wp.next(d) 直接跳絕對距離，避免分岔路口跳錯。

車輛控制
    骨架沿用 verify_carla_geometry.py：spectator 生成、同步模式、TM 對齊後
    交給 PIDController + apply_pid_ff_control_with_steering（PID+FF 縱向 +
    pure-pursuit 橫向），跟資料集蒐集條件一致。

輸出
    outputs/lane_gt_overlay/{map}_{時間戳}/
        images/000000.png ...   —— driving 階段每幀一張，畫在原生
                                    1280x720 上（cv2 BGR，未經 resize）
        summary.json             —— 每幀 lane_width、雙側標線 type/width、
                                    第 3／第 4 兩組線在 --rows 指定的影像列
                                    上的像素寬度與差值
    不開即時視窗——這是產生檔案讓人事後逐張檢視的批次工具，不是互動視窗。

使用方式（在跑 CARLA server 的那台機器上）
    uv run --no-sync python carla_module/project_lane_gt.py
        [--host HOST] [--port PORT] [--speed KMH] [--camera-fps N]
        [--spawn spectator|map] [--spawn-index N] [--z-offset N]
        [--sample-step-m N] [--sample-lookahead-m N] [--steer-lookahead-m N]
        [--align-frames N] [--align-timeout-frames N] [--warmup-frames N]
        [--drive-frames N] [--w-real M] [--double-gap-m M] [--lines A,B,C]
        [--rows V1,V2,...] [--out-dir PATH]

    --lines 預設只畫 center,map,perside 三組（最乾淨的判讀組合）：
        center  參考基準
        map     lane_width/2 = ±1.75，已驗證落在標線油漆的正中央（差 9 mm）
        perside 真實內緣的預測位置，就是要看它貼不貼在漆緣上
    需要對照舊的天真公式或對稱 w_real 時再加 naive / wreal。
    挑線只影響目視，summary.json 的 row_widths 一律涵蓋全部線族。

    預設從 spectator 位置生成車輛，先把 CARLA 視窗鏡頭移到要看的路面。
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
import queue
from typing import Any, Optional

import carla
import cv2
import numpy as np

from carla_module.get_carlaDataset import (
    CAMERA_FOV,
    CAMERA_FWD_X,
    CAMERA_HEIGHT,
    IMG_HEIGHT,
    IMG_WIDTH,
    PHYSICS_WARMUP_TICKS,
    PIDController,
    _wrap_deg,
    apply_pid_ff_control_with_steering,
)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# 雙線標記：CARLA 的 LaneMarking.width 對這些型別回報的是**單條線寬**
# （verify_carla_geometry.py 實測 SolidSolid 與 Solid 同樣回 0.125），
# 不含第二條線也不含中間間隙——第 3 組「天真內側邊」在這種標記上會失準
_DOUBLE_MARKING_TYPES = {"SolidSolid", "BrokenBroken", "SolidBroken", "BrokenSolid"}

# 顏色刻意避開路面油漆本身的顏色（白線、黃線）——第一版用黃色畫邊界中心，
# 結果整條線疊在雙黃漆上，正好把最需要判讀的那一側蓋掉，量不出東西
_COLOR_CENTER = (160, 160, 160)  # 灰：車道中心線
_COLOR_MAP    = (255, 80, 0)     # 藍：邊界中心 ±lane_width/2
_COLOR_NAIVE  = (220, 220, 0)    # 青：天真內側邊（雙線側故意不修正）
_COLOR_WREAL  = (220, 0, 220)    # 洋紅：config w_real（對稱畫，只驗間距不驗位置）
_COLOR_PERSIDE = (0, 160, 0)     # 綠：逐側修正的真實內緣

# --lines 可選的線族 → (欄位名, 顏色, 線寬)
_LINE_GROUPS = ("center", "map", "naive", "wreal", "perside")
_GROUP_SPEC = {
    "center":  [("center",    _COLOR_CENTER,  1)],
    "map":     [("map_l",     _COLOR_MAP,     2), ("map_r",     _COLOR_MAP,     2)],
    "naive":   [("naive_l",   _COLOR_NAIVE,   2), ("naive_r",   _COLOR_NAIVE,   2)],
    "wreal":   [("wreal_l",   _COLOR_WREAL,   2), ("wreal_r",   _COLOR_WREAL,   2)],
    "perside": [("perside_l", _COLOR_PERSIDE, 2), ("perside_r", _COLOR_PERSIDE, 2)],
}


# ── 引數 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA 車道幾何即時投影（目視驗證 w_real）")
    p.add_argument("--host",       default="127.0.0.1", help="CARLA 伺服器位址")
    p.add_argument("--port",       type=int,   default=2000, help="埠號")
    p.add_argument("--timeout",    type=float, default=20.0, help="連線逾時秒數")
    p.add_argument("--speed",      type=float, default=18.0,
                   help="driving 階段目標車速 km/h（預設 18，同資料集）")
    p.add_argument("--camera-fps", type=int,   default=40, help="同步模式 FPS（預設 40）")
    p.add_argument("--spawn",      choices=["spectator", "map"], default="spectator",
                   help="車輛生成位置：spectator 鏡頭處（預設）或地圖內建 spawn point")
    p.add_argument("--spawn-index", type=int, default=0,
                   help="--spawn map 時使用第幾個 spawn point（預設 0）")
    p.add_argument("--z-offset",   type=float, default=0.0,
                   help="spectator 生成點的 z 偏移（預設 0）")
    p.add_argument("--sample-step-m",      type=float, default=0.5,
                   help="沿車道中心線逐步取樣的間距，公尺（預設 0.5）")
    p.add_argument("--sample-lookahead-m", type=float, default=60.0,
                   help="車道幾何往前畫多遠，公尺（預設 60）")
    p.add_argument("--steer-lookahead-m",  type=float, default=6.0,
                   help="pure-pursuit 橫向控制的前視距離，公尺（預設 6，"
                        "跟 --sample-lookahead-m 是兩件事：一個是開車用的轉向"
                        "前視，一個是畫線畫多遠）")
    p.add_argument("--align-frames",  type=int, default=20,
                   help="TM 對齊判定：連續幾幀 steer<0.05（預設 20）")
    p.add_argument("--align-timeout-frames", type=int, default=600,
                   help="TM 對齊最多等幾幀，超過就放棄對齊直接開（預設 600）")
    p.add_argument("--warmup-frames", type=int, default=10,
                   help="TM→手動切換後等幾幀才開始存檔（預設 10）")
    p.add_argument("--drive-frames",  type=int, default=400,
                   help="driving 階段取樣＋存檔幀數（預設 400 ≈ 10s ≈ 50m@18km/h，"
                        "足以涵蓋平路到坡頂）")
    p.add_argument("--w-real", type=float, default=None,
                   help="覆寫 config 的 w_real（公尺）；不指定就讀 "
                        "config/inference_road_lane_segmentation.yaml 的 "
                        "pitch_estimation.w_real（讀不到用 3.216）")
    p.add_argument("--double-gap-m", type=float, default=0.125,
                   help="雙線標記兩條線之間的間隙 (m)，per-side 線用。預設 0.125 "
                        "（＝標線寬，目視估計）。此時 per-side 的間距 = 3.5 - "
                        "(0.125/2+0.125) - 0.125/2 = 3.25，與量到的 3.243~3.255 相符。"
                        "早期版本預設 0.1834 是在「渲染邊緣＝名目邊緣」的錯誤假設下"
                        "反推的，已廢棄")
    p.add_argument("--lines", default="center,map,perside",
                   help="要畫哪幾組線，逗號分隔，可選 "
                        f"{'/'.join(_LINE_GROUPS)}（預設 center,map,perside）。"
                        "只影響目視，summary.json 的 row_widths 一律涵蓋全部")
    p.add_argument("--rows", default="380,420,460,500,540,580,620,660,700",
                   help="量第 3／第 4 組像素寬度的影像列（v 座標，逗號分隔）")
    p.add_argument("--out-dir", default=None,
                   help="輸出目錄（預設 outputs/lane_gt_overlay/{map}_{時間戳}/）")
    return p.parse_args()


# ── 小工具 ────────────────────────────────────────────────────────────────────

def _setup_console() -> None:
    """Windows 主控台預設 cp950，print 到中文/符號會丟 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:                                      # noqa: BLE001
            pass


def _load_w_real(override: Optional[float]) -> float:
    if override is not None:
        return override
    try:
        import yaml
        path = _PROJECT_ROOT / "config" / "inference_road_lane_segmentation.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return float((cfg.get("pitch_estimation", {}) or {}).get("last_resort_lane_width", 3.216))
    except Exception as exc:                                    # noqa: BLE001
        print(f"[警告] 讀不到 config，w_real 改用內建預設 3.216：{exc}")
        return 3.216


def _marking_info(marking: Any) -> Optional[dict]:
    if marking is None:
        return None
    try:
        return {"type": str(marking.type), "color": str(marking.color),
                "width": float(marking.width)}
    except Exception:                                          # noqa: BLE001
        return None


# ── 投影數學（照規格寫，不猜軸向）───────────────────────────────────────────────

def _build_K(width: int, height: int, fov: float) -> tuple[float, float, float]:
    """回傳 (f, cx, cy)；K 矩陣是 [[f,0,cx],[0,f,cy],[0,0,1]]，這裡直接拆開用。"""
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return f, width / 2.0, height / 2.0


def _project(loc: carla.Location, f: float, cx: float, cy: float,
            M: np.ndarray) -> Optional[tuple[float, float]]:
    p_sensor = M @ np.array([loc.x, loc.y, loc.z, 1.0])
    std = np.array([p_sensor[1], -p_sensor[2], p_sensor[0]])
    if std[2] <= 0.1:
        return None
    u = f * std[0] / std[2] + cx
    v = f * std[1] / std[2] + cy
    return u, v


# ── 車道取樣與四組假設 ────────────────────────────────────────────────────────

def _lane_samples(carla_map: carla.Map, start_wp: carla.Waypoint,
                  step_m: float, max_dist_m: float) -> list[carla.Waypoint]:
    """從 start_wp 逐步往前串（每次只走 step_m，不直接跳絕對距離），避免在
    分岔路口跳錯車道。"""
    samples: list[carla.Waypoint] = [start_wp]
    cur = start_wp
    traveled = 0.0
    while traveled < max_dist_m:
        nxt = cur.next(step_m)
        if not nxt:
            break
        cur = nxt[0]
        samples.append(cur)
        traveled += step_m
    return samples


def _hypothesis_points(wp: carla.Waypoint, w_real: float,
                       double_gap_m: float) -> dict:
    """單一 waypoint 的四組假設世界座標 + 標線 metadata。"""
    tf    = wp.transform
    loc   = tf.location
    right = tf.get_right_vector()

    half_lane = wp.lane_width / 2.0
    left_m  = _marking_info(getattr(wp, "left_lane_marking",  None))
    right_m = _marking_info(getattr(wp, "right_lane_marking", None))
    left_w  = left_m["width"]  if left_m  else 0.0
    right_w = right_m["width"] if right_m else 0.0

    # 第 3 組：天真公式，兩側各用自己的 marking.width，**不**修正雙線少扣
    naive_left_dist  = half_lane - left_w / 2.0
    naive_right_dist = half_lane - right_w / 2.0
    half_w_real = w_real / 2.0

    # 第 5 組：逐側修正的真實內緣。單線側內縮 = 線寬/2；雙線側內縮 =
    # 間隙/2 + 一整條線寬（因為 .width 只算單條，邊界中心在兩條線之間）。
    # 這組是不對稱的 —— 這正是重點：真實內緣本來就不對稱於車道中心，
    # 所以第 4 組那種 ±w_real/2 的對稱畫法永遠不可能同時壓在兩側漆上。
    # 第 4 組驗的是「兩線間距」，第 5 組才驗「位置」。
    def _inner_dist(width: float, is_double: bool) -> float:
        return half_lane - (width + double_gap_m / 2.0 if is_double else width / 2.0)

    l_double = bool(left_m  and left_m["type"]  in _DOUBLE_MARKING_TYPES)
    r_double = bool(right_m and right_m["type"] in _DOUBLE_MARKING_TYPES)
    per_left_dist  = _inner_dist(left_w,  l_double)
    per_right_dist = _inner_dist(right_w, r_double)

    def _offset(dist: float) -> carla.Location:
        return carla.Location(x=loc.x + right.x * dist,
                              y=loc.y + right.y * dist,
                              z=loc.z + right.z * dist)

    return {
        "center":   loc,
        "map_l":    _offset(-half_lane),
        "map_r":    _offset(half_lane),
        "naive_l":  _offset(-naive_left_dist),
        "naive_r":  _offset(naive_right_dist),
        "wreal_l":  _offset(-half_w_real),
        "wreal_r":  _offset(half_w_real),
        "perside_l": _offset(-per_left_dist),
        "perside_r": _offset(per_right_dist),
        "lane_width":   wp.lane_width,
        "left_marking":  left_m,
        "right_marking": right_m,
        "left_double":  l_double,
        "right_double": r_double,
        # 第 5 組隱含的內側邊到內側邊距離（不對稱，但總距離才是 w_real 該對上的量）
        "perside_width": per_left_dist + per_right_dist,
        "per_left_dist":  per_left_dist,
        "per_right_dist": per_right_dist,
    }


# ── 畫圖 ──────────────────────────────────────────────────────────────────────

def _draw_polyline(frame: np.ndarray, world_pts: list[Optional[carla.Location]],
                   f: float, cx: float, cy: float, M: np.ndarray,
                   color: tuple, thickness: int = 2,
                   draw: bool = True) -> list[Optional[tuple[float, float]]]:
    """投影（＋選擇性畫線），回傳每點的影像座標。

    draw=False 時只算座標不畫 —— 讓 --lines 挑線只影響目視，
    summary.json 的 row_widths 仍然涵蓋所有線族。
    """
    img_pts: list[Optional[tuple[float, float]]] = []
    prev: Optional[tuple[int, int]] = None
    for loc in world_pts:
        p = _project(loc, f, cx, cy, M) if loc is not None else None
        img_pts.append(p)
        if p is not None:
            pt = (int(round(p[0])), int(round(p[1])))
            if draw:
                cv2.circle(frame, pt, 2, color, -1, cv2.LINE_AA)
                if prev is not None:
                    cv2.line(frame, prev, pt, color, thickness, cv2.LINE_AA)
            prev = pt
        else:
            prev = None
    return img_pts


def _legend_text(name: str, hyp: dict, w_real: float, gap: float) -> str:
    half = hyp["lane_width"] / 2.0
    return {
        "center":  "center line   offset=0",
        "map":     f"lane_width={hyp['lane_width']:.3f}  offset=+-{half:.4f}",
        "naive":   "naive inner   offset=+-(lane_width/2 - marking.width/2)",
        "wreal":   f"w_real={w_real:.3f} SYMMETRIC  offset=+-{w_real/2:.4f}  "
                   "(spacing only)",
        "perside": f"per-side inner edge (gap={gap:.3f})  "
                   f"L=-{half - hyp['per_left_dist']:.4f}  "
                   f"R=-{half - hyp['per_right_dist']:.4f}  "
                   f"spacing={hyp['perside_width']:.4f}",
    }[name]


def _draw_legend(frame: np.ndarray, hyp: dict, w_real: float, gap: float,
                 groups: tuple[str, ...]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = {g: _GROUP_SPEC[g][0][1] for g in _LINE_GROUPS}
    for i, name in enumerate(groups):
        cv2.putText(frame, f"{i+1}) {_legend_text(name, hyp, w_real, gap)}",
                    (10, 24 + i * 22), font, 0.55, colors[name], 2, cv2.LINE_AA)


def _draw_frame(frame: np.ndarray, hyps: list[dict], f: float, cx: float, cy: float,
                M: np.ndarray, w_real: float, gap: float,
                groups: tuple[str, ...]) -> dict[str, list[Optional[tuple[float, float]]]]:
    """畫選定的假設線，回傳**所有**線族的影像座標（供量 row-width 用）。

    未選中的線族仍照算座標、只是不畫，所以 --lines 只影響目視，
    summary.json 的數據不受影響。
    """
    img: dict[str, list[Optional[tuple[float, float]]]] = {}
    for group, entries in _GROUP_SPEC.items():
        for key, color, thick in entries:
            img[key] = _draw_polyline(frame, [h[key] for h in hyps], f, cx, cy, M,
                                      color, thick, draw=group in groups)
    _draw_legend(frame, hyps[0], w_real, gap, groups)

    near = hyps[0]
    cv2.putText(frame,
                f"lane_width={near['lane_width']:.3f}m  "
                f"L={near['left_marking']['type'] if near['left_marking'] else 'NONE'}"
                f"({near['left_marking']['width'] if near['left_marking'] else 0:.3f}m)  "
                f"R={near['right_marking']['type'] if near['right_marking'] else 'NONE'}"
                f"({near['right_marking']['width'] if near['right_marking'] else 0:.3f}m)",
                (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _interp_u_at_v(uv: list[Optional[tuple[float, float]]], target_v: float) -> Optional[float]:
    """在一串 (u,v) 折線上，線性內插出 v=target_v 時的 u。"""
    pts = [p for p in uv if p is not None]
    for (u0, v0), (u1, v1) in zip(pts, pts[1:]):
        if v0 == v1:
            continue
        if (v0 - target_v) * (v1 - target_v) <= 0.0:
            t = (target_v - v0) / (v1 - v0)
            return u0 + t * (u1 - u0)
    return None


def _row_widths(img: dict[str, list[Optional[tuple[float, float]]]],
                rows: list[int]) -> list[dict]:
    out = []
    for v in rows:
        u_naive_l = _interp_u_at_v(img["naive_l"], v)
        u_naive_r = _interp_u_at_v(img["naive_r"], v)
        u_wreal_l = _interp_u_at_v(img["wreal_l"], v)
        u_wreal_r = _interp_u_at_v(img["wreal_r"], v)
        u_per_l   = _interp_u_at_v(img["perside_l"], v)
        u_per_r   = _interp_u_at_v(img["perside_r"], v)
        width3 = (u_naive_r - u_naive_l) if None not in (u_naive_l, u_naive_r) else None
        width4 = (u_wreal_r - u_wreal_l) if None not in (u_wreal_l, u_wreal_r) else None
        width5 = (u_per_r - u_per_l)     if None not in (u_per_l,   u_per_r)   else None
        diff   = (width3 - width4) if None not in (width3, width4) else None
        # 第 4 與第 5 的「間距」若一致，就證明 w_real 這個數字對，
        # 差別只在對稱與否（位置），那對 pipeline 的深度估計無影響
        d45    = (width5 - width4) if None not in (width4, width5) else None
        out.append({"row": v, "width3_px": width3, "width4_px": width4,
                    "width5_px": width5, "diff_px": diff, "diff45_px": d45,
                    "u_naive_l": u_naive_l, "u_naive_r": u_naive_r,
                    "u_wreal_l": u_wreal_l, "u_wreal_r": u_wreal_r,
                    "u_perside_l": u_per_l, "u_perside_r": u_per_r})
    return out


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    _setup_console()
    args   = parse_args()
    w_real = _load_w_real(args.w_real)
    rows   = [int(x) for x in args.rows.split(",") if x.strip()]
    groups = tuple(g.strip() for g in args.lines.split(",") if g.strip())
    bad    = [g for g in groups if g not in _LINE_GROUPS]
    if bad:
        raise SystemExit(f"--lines 有無效的線族 {bad}，可選：{list(_LINE_GROUPS)}")
    print(f"[初始化] 畫出的線族：{', '.join(groups)}"
          f"（雙線間隙 {args.double_gap_m} m）")
    print(f"[初始化] w_real={w_real:.4f} m（{'命令列覆寫' if args.w_real is not None else '讀自 config'}）")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world     = client.get_world()
    carla_map = world.get_map()
    map_name  = carla_map.name.split("/")[-1]
    print(f"[初始化] 連線至地圖：{carla_map.name}")
    bp_lib = world.get_blueprint_library()

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

    f, cx, cy = _build_K(IMG_WIDTH, IMG_HEIGHT, CAMERA_FOV)
    dt = 1.0 / args.camera_fps

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = pathlib.Path(args.out_dir) if args.out_dir else \
        _PROJECT_ROOT / "outputs" / "lane_gt_overlay" / f"{map_name}_{ts}"
    img_dir   = out_root / "images"
    clean_dir = out_root / "images_clean"
    img_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    print(f"[初始化] 輸出目錄：{out_root}")

    original_settings = world.get_settings()
    camera: Optional[carla.Sensor] = None
    tm = None
    frame_records: list[dict] = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = dt
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        cam_bp.set_attribute("fov", str(CAMERA_FOV))
        cam_tf = carla.Transform(carla.Location(x=CAMERA_FWD_X, z=CAMERA_HEIGHT))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

        image_queue: "queue.Queue" = queue.Queue()
        camera.listen(image_queue.put)

        print(f"[初始化] 物理預熱 {PHYSICS_WARMUP_TICKS} ticks...")
        for _ in range(PHYSICS_WARMUP_TICKS):
            world.tick()

        # ── TM 對齊 ──────────────────────────────────────────────────────────
        vehicle.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(vehicle, 100.0)
        tm.ignore_signs_percentage(vehicle,  100.0)
        tm.set_desired_speed(vehicle, args.speed)

        print(f"[對齊] TM 對齊車道中（需連續 {args.align_frames} 幀 steer<0.05）...")
        aligned = 0
        for _ in range(args.align_timeout_frames):
            world.tick()
            aligned = aligned + 1 if abs(vehicle.get_control().steer) < 0.05 else 0
            if aligned >= args.align_frames:
                break
        else:
            print("[警告] 對齊逾時，仍繼續（橫向位置可能不在車道中央）")

        vehicle.set_autopilot(False)
        pid = PIDController(kp=1.0, ki=0.25, kd=0.15, dt=dt, integral_limit=3.0)
        target_mps = args.speed / 3.6

        print(f"[暖身] 切手動控制，等 {args.warmup_frames} 幀物理穩定...")
        for _ in range(args.warmup_frames):
            apply_pid_ff_control_with_steering(
                vehicle, carla_map, target_mps, pid, 0.015, args.steer_lookahead_m)
            world.tick()

        print(f"[取樣] 開始 {args.drive_frames} 幀，每幀存一張 overlay PNG...")
        for i in range(args.drive_frames):
            apply_pid_ff_control_with_steering(
                vehicle, carla_map, target_mps, pid, 0.015, args.steer_lookahead_m)
            target_frame = world.tick()

            image: Optional[carla.Image] = None
            try:
                while True:
                    temp = image_queue.get(timeout=2.0)
                    if temp.frame == target_frame:
                        image = temp
                        break
                    if temp.frame > target_frame:
                        break
            except Exception:                                    # noqa: BLE001
                print(f"[警告] 幀 {target_frame} 影像遺失，跳過")
                continue
            if image is None:
                continue

            cam_tf_now = camera.get_transform()
            # 取樣起點用相機所在處，不是車輛所在處（見檔頭「已知陷阱」）
            start_wp = carla_map.get_waypoint(cam_tf_now.location, project_to_road=True)
            if start_wp is None:
                print(f"[警告] 幀 {i} 相機位置投影不到路面，跳過")
                continue

            samples = _lane_samples(carla_map, start_wp, args.sample_step_m, args.sample_lookahead_m)
            hyps    = [_hypothesis_points(wp, w_real, args.double_gap_m)
                       for wp in samples]

            frame_bgr = np.frombuffer(image.raw_data, dtype=np.uint8) \
                .reshape((image.height, image.width, 4))[:, :, :3].copy()

            png_name = f"{i:06d}.png"
            # 先存一份沒有任何疊圖的乾淨原圖。疊圖的線正好畫在要判讀的漆緣上，
            # 只有疊圖版就沒辦法量油漆的實際邊緣位置（第一版就卡在這）
            cv2.imwrite(str(clean_dir / png_name), frame_bgr)

            M = np.array(cam_tf_now.get_inverse_matrix())
            img_lines = _draw_frame(frame_bgr, hyps, f, cx, cy, M, w_real,
                                    args.double_gap_m, groups)
            cv2.imwrite(str(img_dir / png_name), frame_bgr)

            near = hyps[0]
            veh_tf = vehicle.get_transform()
            frame_records.append({
                "frame_idx": i,
                "png": f"images/{png_name}",
                "png_clean": f"images_clean/{png_name}",
                "cam_world": [cam_tf_now.location.x, cam_tf_now.location.y, cam_tf_now.location.z],
                "veh_world": [veh_tf.location.x, veh_tf.location.y, veh_tf.location.z],
                # CARLA 平路的 waypoint pitch 會回 360.0 而不是 0.0，一律 wrap
                "road_pitch_deg": _wrap_deg(float(start_wp.transform.rotation.pitch)),
                "lane_width":     near["lane_width"],
                "left_marking":   near["left_marking"],
                "right_marking":  near["right_marking"],
                "left_double":    near["left_double"],
                "right_double":   near["right_double"],
                "perside_width_m": near["perside_width"],
                "row_widths":     _row_widths(img_lines, rows),
            })

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

    if not frame_records:
        print("[錯誤] 沒有存到任何幀，不寫 summary。")
        return

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps({
        "timestamp": ts,
        "map": map_name,
        "args": vars(args),
        "w_real": w_real,
        "camera": {"fov_deg": CAMERA_FOV, "width": IMG_WIDTH, "height": IMG_HEIGHT,
                   "forward_x": CAMERA_FWD_X, "height_m": CAMERA_HEIGHT},
        "double_gap_m": args.double_gap_m,
        "offsets_legend": {
            "map_boundary": "±lane_width/2",
            "naive_inner":  "±(lane_width/2 - marking.width/2)  [雙線側未修正，故意的]",
            "config_w_real": "±w_real/2（對稱：只驗兩線間距，不驗位置）",
            "per_side":     "雙線側 lane_width/2-(width+gap/2)、單線側 "
                            "lane_width/2-width/2（不對稱：驗位置）",
        },
        "frames": frame_records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[輸出] {img_dir}        ({len(frame_records)} 張疊圖 PNG)")
    print(f"[輸出] {clean_dir}  ({len(frame_records)} 張乾淨原圖，量漆緣用這個)")
    print(f"[輸出] {summary_path}")


if __name__ == "__main__":
    main()
