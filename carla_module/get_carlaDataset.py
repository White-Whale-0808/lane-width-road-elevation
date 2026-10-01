"""
carla_module/get_carlaDataset.py
CARLA 資料集蒐集腳本：蒐集 RGB 影像與 GT pitch／速度／行駛距離

使用方式：
    uv run python carla_module/get_carlaDataset.py [--host HOST] [--port PORT]
        [--speed KMH] [--camera-fps N] [--z-offset N]
        [--align-frames N] [--warmup-frames N] [--lookahead-m N]
        [--route-length-m N] [--route-dest X,Y[,Z]]
        [--kp F] [--ki F] [--kd F] [--ff-gain F]
        [--camera-preset {default,openlane}] [--cam-z M] [--cam-fov DEG]
        [--img-width PX] [--img-height PX]

狀態機流程：
    生成車輛
       ↓
    [ALIGNING]   TM autopilot 對齊車道（連續 --align-frames 幀 steer < 0.05）
       ↓
    〈規劃路線〉 以對齊後的位置為起點，一次算完整條路線（見下）
       ↓
    [WARMUP]     關閉 TM，PID+FF 控制 + 沿路線 pure-pursuit，等 --warmup-frames 幀讓物理穩定
       ↓
    [COLLECTING] PID+FF 控制 + 沿路線 pure-pursuit，存圖 + GT 量測，
                 按 'q' 或路線走完即停止

    橫向控制說明：ALIGNING 只保證 TM 轉向輸出連續幾幀 < 0.05，不保證航向零
    誤差；最早的版本切到手動後直接 steer=0 寫死，假設接下來的路完全筆直，路
    稍有彎或殘留航向誤差就會在整段收集距離上不受控漂移，可能跨到隔壁車道
    （carla_module/verify_carla_geometry.py 的驗證報告第⑥項可以量出這個問題）。
    改成 pure-pursuit 之後漂移解決了，但又冒出第二個問題：它每幀都重新問地圖
    「我現在在哪條車道」，到路口會被轉彎銜接車道吸走（詳見 `Route` 的
    docstring）。現在**路線在開始行駛前就一次決定好**，行駛時只沿著那條折線
    走、不再問地圖，路口才真的不會轉錯。縱向（throttle/brake）仍是同一套
    PID+FF 不變。

    路線怎麼來：預設 `build_route_straight` —— 從對齊後的位置沿車道往前
    `--route-length-m` 公尺，每遇分岔都選最接近直走的分支。想採「會轉彎」的
    路線就用 `--route-dest X,Y[,Z]` 指定終點，改由 CARLA 內建的
    GlobalRoutePlanner 做 A* 規劃。

儲存路徑：outputs/carla_dataset_{Map}_{YYYYMMDD_HHMMSS}/
    ├── images/         000000.png, 000001.png, ...
    ├── measurements.csv  逐幀：車身姿態 / 路面姿態 / 速度 / 距離 / 車輛與相機世界座標
    ├── road_profile.csv  逐幀 × 逐採樣點：相機前方路面的世界座標（採集當下直接問地圖），
    │                     解析中心線高度 `z` 與向下射線量到的網格高度 `z_mesh` 兩欄並存
    └── metadata.json     相機掛載參數、fov、解析度、地圖名、採集設定

GT 的設計原則：**存最生的量，之後再推導**。前方剖面存 waypoint 的世界座標而
不是預先算好的 pitch，相機存完整 transform 而不是只有位置 —— 這樣「沿光軸的
深度」「高度剖面」「相對哪個座標系的 pitch」都能離線重算，不必為了換一種
推導方式而重採。詳見 DatasetWriter 與 sample_road_profile 的 docstring，
以及 to-do.md 第 2/3 組。
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
import csv
import datetime
import json
import math
import queue
import time
from collections import deque
from typing import Optional

import carla
import cv2
import numpy as np


# ── 相機規格（比照 get_data.py）────────────────────────────────────────────────
IMG_WIDTH     = 1280
IMG_HEIGHT    = 720
CAMERA_FOV    = 90.0
CAMERA_HEIGHT = 1.08   # 相對「車輛原點」的掛載高度，車輛原點不一定在地面
CAMERA_FWD_X  = 1.5    # 相機前移量：GT 距離記在車輛原點，推論端需用此值對齊
PHYSICS_WARMUP_TICKS = 30   # 生成後讓物理系統落穩的預熱 tick 數

# --camera-preset：整組換掉相機規格（WWH-25）。
#   openlane  仿 OpenLane（Waymo 前鏡頭）：1920×1280、焦距 2071.4 px（202 段中位數，
#             見 openlane_module/convert_openlane._F_SRC_MEDIAN）→ 水平 FOV
#             2·atan(960/2071.4) = 49.73°；離地 2.116 m（OpenLane 轉換後的
#             camera_height）。CARLA 是方像素，f_x = f_y，與 OpenLane 一致。
#             縮到 pipeline 的 1024×512 後 f_x=1104.75、f_y=828.56，和 OpenLane
#             的虛擬內參完全相同 —— 兩份資料集可以共用同一組相機參數。
#   ⚠ project_lane_gt.py / verify_carla_geometry.py 直接 import 下面的常數，
#   不會跟著換；它們讀舊資料集沒問題，要用在新資料集得改成讀 metadata.json。
CAMERA_PRESETS = {
    "default":  dict(IMG_WIDTH=1280, IMG_HEIGHT=720, CAMERA_FOV=90.0, CAMERA_HEIGHT=1.08),
    "openlane": dict(IMG_WIDTH=1920, IMG_HEIGHT=1280, CAMERA_FOV=49.73, CAMERA_HEIGHT=2.116),
}


# ── 狀態機 ────────────────────────────────────────────────────────────────────

class State:
    ALIGNING   = "ALIGNING"    # TM autopilot 對齊車道中
    WARMUP     = "WARMUP"      # 關閉 TM 後等物理穩定
    COLLECTING = "COLLECTING"  # PID+FF 控制，開始存資料


# ── PID + Feed-forward 控制器 ─────────────────────────────────────────────────

class PIDController:
    """
    針對坡道定速設計的 PID 控制器。
    - P 項：即時反應速差
    - I 項：消除上坡重力造成的穩態誤差（取代舊版固定偏置 0.3）
    - D 項：抑制超調與震盪
    搭配 Feed-forward 坡度補償，進坡瞬間不掉速。
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.25, kd: float = 0.15,
                 dt: float = 0.025, integral_limit: float = 3.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_limit = integral_limit  # anti-windup

        self._integral  = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = None

    def step(self, error: float) -> float:
        """輸入速差 (m/s)，輸出 [-1, 1] 控制量（正=油門，負=煞車）"""
        self._integral += error * self.dt
        self._integral  = max(-self.integral_limit,
                              min(self.integral_limit, self._integral))

        # 首步（或 reset 後）沒有前值可差分，D 項歸零，
        # 避免 (error-0)/dt 的 derivative kick 造成油門/煞車瞬間打滿
        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(max(-1.0, min(1.0, output)))


def _get_slope_deg(vehicle: carla.Vehicle) -> float:
    """回傳車輛當前坡度（度），上坡為正、下坡為負。"""
    # CARLA/UE 慣例：上坡（機頭朝上）pitch 為正值，直接沿用
    return vehicle.get_transform().rotation.pitch


def _slope_feedforward(slope_deg: float, ff_gain: float) -> float:
    """
    坡度前饋補償量（油門側）。
    物理原理：上坡阻力 ∝ sin(θ)，直接預補油門，
    不需等積分累積，進坡瞬間不掉速。
    下坡回傳 0（不主動提前煞車）。
    """
    return float(max(0.0, math.sin(math.radians(slope_deg)) * ff_gain * 100))


def _compute_pid_ff(
    vehicle:    carla.Vehicle,
    target_mps: float,
    pid:        PIDController,
    ff_gain:    float,
) -> tuple[float, float, float, float]:
    """算 PID + Feed-forward 縱向控制量，不呼叫 apply_control。回傳 (throttle, brake, 當前速度 m/s, 坡度 °)。"""
    vel       = vehicle.get_velocity()
    speed     = vel.length()
    error     = target_mps - speed
    slope_deg = _get_slope_deg(vehicle)

    pid_out     = pid.step(error)
    feedforward = _slope_feedforward(slope_deg, ff_gain)
    raw_output  = pid_out + feedforward

    if raw_output >= 0.0:
        throttle = min(1.0, raw_output)
        brake    = 0.0
    else:
        throttle = 0.0
        brake    = min(1.0, abs(raw_output) * 0.6)

    return throttle, brake, speed, slope_deg


def _wrap_deg(angle: float) -> float:
    """正規化角度差到 (-180, 180]，避免 0°/360° 之類的等價角度被算成 -360°。"""
    return (angle + 180.0) % 360.0 - 180.0


# ── 路線：開始行駛前就決定好，行駛中不再查地圖 ────────────────────────────────

_ROUTE_STEP_M = 0.5   # 路線點間距。steer 會沿折線內插，不需要更密

#: 開始採集時離路線多遠就警告（m）。3.5 m 車道的一半 —— 超過這個值，車八成
#: 已經在隔壁車道上了
_LATERAL_WARN_M = 1.75


def _straightest_next(nxts: list, fwd: carla.Vector3D) -> carla.Waypoint:
    """從 `wp.next()` 的候選裡挑最接近「直走」的分支。

    路口一次會跳出好幾個銜接車道，取 `[0]` 等於隨機選一條。
    `build_route_straight` 與 `sample_road_profile` 共用這個規則。
    """
    return max(nxts, key=lambda w: w.transform.get_forward_vector().dot(fwd))


class Route:
    """一條**在開始行駛前就決定好**的路線，行駛時只沿著它走。

    為什麼非得預先決定：pure-pursuit 的前一版每幀都用
    `carla_map.get_waypoint(車輛位置, project_to_road=True)` 重新找參考車道。
    路口裡的**轉彎銜接車道也是 Driving 車道**，車子只要稍微偏離中心，這個
    查詢就可能吸附到左/右轉那條銜接車道上；pure-pursuit 於是朝它轉，車子更
    靠過去，下一幀吸得更死 —— 正回饋，車就轉走了。只在 `wp.next()` 的候選裡
    挑「最直」的分支救不回來，因為選錯發生在更前面的 `get_waypoint()` 那步。

    這裡把路線先算好存成折線，行駛時 cursor **只單調前進**、不再問地圖
    「我現在在哪條車道」，路口自然拉不走。
    """

    # cursor 每幀最多往前推進的弧長（m）。18 km/h @ 40 fps 每幀只前進
    # 0.125 m，5 m 已經是 40 倍的餘裕；上限存在的理由是**路線可能自我交叉**
    # —— 繞一圈回到同一個地方、或同一個路口進出兩次，若不設限就會挑到後半段
    # 那個更近的點，cursor 一次跳過去，車子跟著抄近路切過去。
    # （立體交叉不受影響：`carla.Location.distance` 是 3D 的，高度差就分得開。）
    _MAX_ADVANCE_M = 5.0
    #: 距離連續變大幾個點就停止搜尋 —— 取第一個局部極小，而不是視窗內的全域
    #: 極小。點距 0.5 m、每幀只走 0.125 m，距離變化是平滑的，不必怕雜訊誤停
    _RISE_TO_STOP = 4

    def __init__(self, points: list) -> None:
        """`points` 是一串 carla.Location（車道中心線上的點）。

        存 Location 而不是 Waypoint：路線可能來自地圖規劃，也可能來自
        `pick_route.py` 存下來的 JSON —— 後者根本沒有 Waypoint 物件可用。
        """
        self._pts = list(points)
        self._s   = [0.0]                      # 累積弧長
        for a, b in zip(self._pts, self._pts[1:]):
            self._s.append(self._s[-1] + a.distance(b))
        self._cursor = 0
        #: 最近一次 `advance()` 量到的橫向偏差（m）。HUD 直接讀這個 ——
        #: 顯示程式碼不該自己再呼叫一次 `advance()`，那會推進 cursor
        self.last_lateral = 0.0

    def __len__(self) -> int:
        return len(self._pts)

    @property
    def points(self) -> list:
        """路線折線點（carla.Location），給存檔／繪圖用。"""
        return self._pts

    @property
    def length_m(self) -> float:
        return self._s[-1] if self._s else 0.0

    def remaining_m(self) -> float:
        """cursor 之後還剩多少路線 —— 用來判斷該不該收工。"""
        return self.length_m - self._s[self._cursor] if self._pts else 0.0

    def advance(self, loc: carla.Location) -> float:
        """把 cursor 推進到離 loc 最近的路線點，回傳橫向偏差（m）。

        只往前找、不回頭：這正是它在路口不會被拉走的原因。
        """
        best_i = self._cursor
        best_d = self._pts[best_i].distance(loc)
        limit_s = self._s[self._cursor] + self._MAX_ADVANCE_M
        rising  = 0
        i = self._cursor + 1
        while i < len(self._pts) and self._s[i] <= limit_s:
            d = self._pts[i].distance(loc)
            if d < best_d:
                best_i, best_d, rising = i, d, 0
            else:
                rising += 1
                if rising >= self._RISE_TO_STOP:
                    break
            i += 1
        self._cursor       = best_i
        self.last_lateral  = best_d
        return best_d

    def lookahead_point(
        self, loc: carla.Location, lookahead_m: float
    ) -> tuple[Optional[carla.Location], float]:
        """回傳 (前視目標點, 橫向偏差)。空路線回傳 (None, 0.0)。

        快走完時目標點就停在路線終點 —— 由 `remaining_m()` 負責喊停，
        這裡不做外插。
        """
        if not self._pts:
            return None, 0.0
        lateral  = self.advance(loc)
        s_target = self._s[self._cursor] + lookahead_m
        j = self._cursor
        while j + 1 < len(self._s) and self._s[j] < s_target:
            j += 1
        return self._pts[j], lateral


def build_route_straight(
    carla_map: carla.Map,
    start_loc: carla.Location,
    length_m:  float,
    step_m:    float = _ROUTE_STEP_M,
    start_fwd: Optional[carla.Vector3D] = None,
) -> Route:
    """從 start_loc 沿目前車道往前 length_m，每遇分岔都選最接近直走的分支。

    選路只做一次，而且是從**乾淨的車道朝向**做的。行駛中車身會因懸吊、殘留
    航向誤差而歪，但那時候路線早就定案了，歪不到選路頭上。

    `start_fwd` 給車輛當下的前向量：起點萬一落在路口裡，`get_waypoint()` 可能
    傳回某條轉彎銜接車道，那條車道的朝向不能拿來判斷「哪邊算直走」，要用車頭
    自己的朝向。不給就退回用起點 waypoint 的朝向。
    """
    wp = carla_map.get_waypoint(start_loc, project_to_road=True)
    if wp is None:
        return Route([])
    if wp.is_junction:
        print("[警告] 路線起點就在路口內，規劃出來的方向可能不是你要的；"
              "建議把起點移到路口前後的直路段再採集")

    route    = [wp]
    prev_fwd = start_fwd if start_fwd is not None else wp.transform.get_forward_vector()
    d = 0.0
    while d < length_m - 1e-6:
        nxts = wp.next(step_m)
        if not nxts:
            break                     # 路盡頭：路線到此為止，不猜測
        wp       = _straightest_next(nxts, prev_fwd)
        prev_fwd = wp.transform.get_forward_vector()
        route.append(wp)
        d += step_m
    return Route([w.transform.location for w in route])


def build_route_to(
    carla_map: carla.Map,
    start_loc: carla.Location,
    dest_loc:  carla.Location,
    step_m:    float = _ROUTE_STEP_M,
) -> Route:
    """用 CARLA 內建的 GlobalRoutePlanner 從 start_loc 規劃到 dest_loc。

    與 `build_route_straight` 的差別是**可以含轉彎**：想採一段特定的、會過
    路口的路線時用這個指定終點。路線一樣是行駛前就定好的，所以 `Route` 講的
    那個路口吸附問題照樣不會發生。

    延後 import：GRP 需要 CARLA 內建的 agents 套件與 networkx，而預設的
    `build_route_straight` 兩者都不需要 —— 不指定終點就不該為此裝東西。
    """
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ImportError as exc:                                  # noqa: BLE001
        raise RuntimeError(
            "--route-dest 需要 CARLA 內建的 agents 套件與 networkx："
            "CARLA_WHL_PATH 往上兩層要有 agents/ 目錄（隨 CARLA 附帶）。"
            f"原始錯誤：{exc}"
        ) from exc

    grp  = GlobalRoutePlanner(carla_map, step_m)
    plan = grp.trace_route(start_loc, dest_loc)   # [(waypoint, RoadOption), ...]
    return Route([wp.transform.location for wp, _ in plan])


# 路線 JSON 的格式版本。`pick_route.py` 寫、這裡讀，兩邊都認這個號碼
ROUTE_FILE_VERSION = 1


def load_route_file(path: pathlib.Path, carla_map: carla.Map) -> Route:
    """讀 `pick_route.py` 存下來的路線 JSON。

    會擋掉地圖不符：路線點是世界座標，套到另一張地圖上會落在莫名其妙的地方，
    而且多半還是「看起來能跑」的那種錯 —— 寧可在這裡直接停下來。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到路線檔：{path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"路線檔不是合法 JSON：{path}（{exc}）") from exc

    version = data.get("version")
    if version != ROUTE_FILE_VERSION:
        raise RuntimeError(
            f"路線檔版本不符：檔案是 {version}，這支腳本認得 {ROUTE_FILE_VERSION}。"
            f"請用現在的 pick_route.py 重新選一次路線"
        )

    file_map = data.get("map")
    if file_map and file_map != carla_map.name:
        raise RuntimeError(
            f"路線檔是在地圖 {file_map} 上選的，現在連到的是 {carla_map.name}。"
            f"世界座標不通用，請切換地圖或重新選路線"
        )

    pts = data.get("points") or []
    if len(pts) < 2:
        raise RuntimeError(f"路線檔只有 {len(pts)} 個點，至少要 2 個：{path}")

    route = Route([carla.Location(x=p[0], y=p[1], z=p[2]) for p in pts])
    print(f"[路線] 載入 {path.name}：{route.length_m:.1f} m / {len(route)} 點"
          f"（{data.get('mode', '?')} 模式，行駛中不再查地圖）")
    return route


def route_start_transform(route: Route, heading_span_m: float = 3.0) -> carla.Transform:
    """路線起點的生成姿態：位置取第一個點，車頭朝向路線前方。

    朝向不是拿相鄰兩點算的 —— 點距只有 0.5 m，量化誤差會讓朝向抖得很厲害。
    改成往前找約 `heading_span_m` 公尺的點再算方位角，穩定得多。
    """
    pts   = route.points
    start = pts[0]

    ahead = pts[-1]
    for p in pts[1:]:
        if start.distance(p) >= heading_span_m:
            ahead = p
            break

    yaw = math.degrees(math.atan2(ahead.y - start.y, ahead.x - start.x))
    # 抬高一點再生成：起點 z 是車道**路面**高度，直接放車子會卡進地面
    return carla.Transform(
        carla.Location(x=start.x, y=start.y, z=start.z + 0.5),
        carla.Rotation(yaw=yaw),
    )


def _pure_pursuit_steer(vehicle: carla.Vehicle, carla_map: carla.Map,
                        lookahead_m: float,
                        route: Optional[Route] = None) -> float:
    """
    取代最早版本寫死的 steer=0.0。ALIGNING 只保證 TM 轉向輸出
    連續幾幀 < 0.05，不保證航向零誤差；WARMUP/COLLECTING 加起來往往是幾十到
    上百公尺，路稍有彎或殘留一點航向誤差，steer=0 會讓車子不受控地橫向漂移，
    可能跨到隔壁車道，把整批資料的 GT 對應關係搞錯。

    做法：往前 lookahead_m 取一個目標點，算它相對車頭朝向的夾角，轉成
    [-1, 1] 的 steer 修正量（簡化版 pure-pursuit）。

    給了 `route` 就沿那條預先規劃好的折線取目標點 —— 這是採集時該走的路，
    路口不會被拉走。沒給就退回每幀查地圖的舊行為（`verify_carla_geometry.py`
    與 `project_lane_gt.py` 走這條，它們只在直路上短程行駛）；舊行為在路口
    會轉錯，原因見 `Route` 的 docstring。
    """
    tf  = vehicle.get_transform()
    fwd = tf.get_forward_vector()

    if route is not None:
        target, _ = route.lookahead_point(tf.location, lookahead_m)
        if target is None:
            return 0.0
    else:
        wp = carla_map.get_waypoint(tf.location, project_to_road=True)
        if wp is None:
            return 0.0
        nxts = wp.next(lookahead_m)
        if not nxts:
            return 0.0
        target = _straightest_next(nxts, fwd).transform.location

    heading      = math.degrees(math.atan2(fwd.y, fwd.x))
    to_target_x  = target.x - tf.location.x
    to_target_y  = target.y - tf.location.y
    target_angle = math.degrees(math.atan2(to_target_y, to_target_x))

    angle_err_deg = _wrap_deg(target_angle - heading)
    # 經驗值：滿舵對應 ~30° 航向誤差，避免小誤差被過度放大造成震盪
    return max(-1.0, min(1.0, angle_err_deg / 30.0))


# ── 前方路面剖面（採集當下直接問地圖）──────────────────────────────────────────

_PROFILE_MAX_D_M = 50.0   # 採樣到相機前方幾公尺（略大於 pitch_estimation 的 z_cap 45）

# 採樣間距。0.125 m 是對齊 legacy GT 的密度 —— 那條是逐幀取樣的，18 km/h @ 40 fps
# 下相機每幀前進 0.1246 m。
#
# 原本是 1.0 m，實測太稀：GT 的 pitch 由 np.gradient 逐點微分求得，1 m 間距等於
# 把剖面當折線，畫出來也是折線。試過離線補救（把所有幀的剖面點在世界座標下合併，
# 可到 1 cm 間距）但更差 —— 單幀那批點是同一瞬間、同一個相機姿態打出來的，內部
# 一致；跨幀合併會混入各幀 transform 的配準誤差。密度要在採集時給。
#
# 代價：每幀的 next() 與 cast_ray 都從 51 次變 401 次。同步模式下這只花牆鐘時間，
# 不影響物理與車速（見 --no-profile-ray 的說明），但整趟採集會明顯變久。
_PROFILE_STEP_M  = 0.125

# 射線模式：從解析曲面**上方** _UP_M 處起打，一路搜到曲面**下方** _DEPTH_M
# （射線總長 = 兩者相加），找渲染網格的實際高度。
# 起點必須抬高 —— waypoint 的 z 就在解析曲面上，從那裡起算會落在網格的正負
# 兩側，往下打有可能整條射線都在網格下方而漏掉。抬 2 m 遠比任何可能的網格
# 偏差大（相機腳下實測僅 mm 級），又低到不會打到天橋/隧道頂。
_PROFILE_RAY_UP_M    = 2.0
_PROFILE_RAY_DEPTH_M = 6.0

# 向下射線可接受的地面語意標籤（label 名稱字串比對，跨 CARLA 版本較穩）。
# verify_carla_geometry.py 也用這份，改這裡兩邊一起動。
GROUND_LABELS = {"Roads", "RoadLines", "Ground", "Terrain", "Sidewalks"}


def ray_ground_z(world: carla.World, loc: carla.Location,
                 depth: float = _PROFILE_RAY_DEPTH_M) -> tuple[Optional[float], Optional[str]]:
    """
    從 loc 垂直向下射線，回傳第一個「地面語意」命中點的 z 與其標籤。

    **必須**用 label 過濾：從相機處起算的射線會先穿過引擎蓋/底盤，剖面上的
    射線則可能打到路上的車輛或雜物。找不到地面標籤就回 None —— 不做退而求
    其次的猜測，寧缺勿錯（否則會把車身高度當成路面高度）。

    回傳的第二項在失敗時是 `"no-ground:<看到的標籤>"`，方便離線分辨「射線沒
    命中」與「命中了但都不是地面」。
    """
    try:
        end    = carla.Location(x=loc.x, y=loc.y, z=loc.z - depth)
        points = world.cast_ray(loc, end)
    except Exception:                                          # noqa: BLE001
        return None, None
    if not points:
        return None, None

    labels_seen: list[str] = []
    for pt in points:
        label = str(getattr(pt, "label", "")).split(".")[-1]
        z     = float(pt.location.z)
        if z > loc.z + 1e-6:          # 射線可能回傳起點上方的命中，忽略
            continue
        labels_seen.append(label)
        if label in GROUND_LABELS:
            return z, label
    return None, ("no-ground:" + ",".join(labels_seen) if labels_seen else None)


def sample_road_profile(
    carla_map:  carla.Map,
    cam_loc:    carla.Location,
    max_d_m:    float = _PROFILE_MAX_D_M,
    step_m:     float = _PROFILE_STEP_M,
    world:      Optional[carla.World] = None,
) -> list[tuple[float, carla.Waypoint, Optional[float], Optional[str]]]:
    """相機前方路面剖面：沿車道往前走，回傳
    (要求距離, waypoint, 網格高度, 網格標籤) 序列。

    給 `world` 就多打一輪向下射線，量**渲染網格**的高度；不給則後兩項為 None。
    兩種高度都存、不是二選一：waypoint 給的是 OpenDRIVE 的**解析中心線**，
    但相機看到的、車子壓過的都是由那條曲線三角化出來的**網格**。兩者在相機
    腳下實測只差 mm 級（`verify_carla_geometry` 那趟，median −0.28 mm、
    std 3.59 mm），但那是**在腳下量的** —— 相機真正看的是前方 5-40 m，遠處的
    三角化密度與 LOD 從來沒有量過，而 pitch 對深度方向的高度變化極敏感。
    兩欄並存才能離線比較，也不會弄壞既有的解析式 GT。

    取代「用行駛歷史回推前方 GT」的做法（`pitch_visualization.gt_pitch_profile`
    是拿車子後來開到那裡時的 pitch 當前方 GT）。歷史回推有三個問題：車身/懸吊
    的遲滯（實測 body = 1.0213 x road - 0.48 度，且逐坡度帶差值不是常數）、
    前移量的對齊、以及開到終點的幀沒有前方 GT。直接問地圖三者一次消失。

    起點是**相機所在處**而非車輛所在處：相機前移 CAMERA_FWD_X = 1.5 m，
    在 9.5 度坡上那裡的路面高 0.25 m，用車輛 waypoint 會系統性量錯
    （見 carla_module/verify_carla_geometry.py 的第 1 組驗證）。

    逐步走而不是直接 `wp.next(d)`：後者在路口會分岔，且每次都從頭走一遍。
    分岔時選航向最接近前一段的分支，避免剖面拐進岔路。
    """
    wp = carla_map.get_waypoint(cam_loc, project_to_road=True)
    if wp is None:
        return []

    walked = [(0.0, wp)]
    prev_fwd = wp.transform.get_forward_vector()
    d = 0.0
    while d < max_d_m - 1e-6:
        nxts = wp.next(step_m)
        if not nxts:
            break   # 路盡頭：剖面就到此為止，不猜測
        wp = _straightest_next(nxts, prev_fwd)
        prev_fwd = wp.transform.get_forward_vector()
        d += step_m
        walked.append((d, wp))

    if world is None:
        return [(d_req, w, None, None) for d_req, w in walked]

    out = []
    for d_req, w in walked:
        loc   = w.transform.location
        start = carla.Location(x=loc.x, y=loc.y, z=loc.z + _PROFILE_RAY_UP_M)
        z_mesh, label = ray_ground_z(
            world, start, depth=_PROFILE_RAY_UP_M + _PROFILE_RAY_DEPTH_M)
        out.append((d_req, w, z_mesh, label))
    return out


def apply_pid_ff_control_with_steering(
    vehicle:     carla.Vehicle,
    carla_map:   carla.Map,
    target_mps:  float,
    pid:         PIDController,
    ff_gain:     float,
    lookahead_m: float,
    route:       Optional[Route] = None,
) -> tuple[float, float]:
    """
    PID+FF 縱向 + pure-pursuit 橫向，同一輪算好後一次 apply_control 送出。

    `route` 是選用的：給了就沿預先規劃的路線走（採集用），沒給就退回每幀
    查地圖的舊行為。參數擺在最後且有預設值，是為了讓既有以位置引數呼叫的
    `verify_carla_geometry.py` / `project_lane_gt.py` 不用改。

    早期版本是「先送出縱向控制（steer 寫死 0）→ 用 vehicle.get_control()
    讀回來 → 疊加 steer → 再 apply_control 一次」，看似安全（CARLA 只認每個
    tick 最後一次 apply_control），實際上有 race condition：同步模式下
    get_control() 要等下一次 world.tick() 才會反映剛剛 apply_control() 送出的
    值，讀回來的其實是上一輪的舊值。實測這會讓油門被舊值鎖住、車速衝過頭
    （曾經衝到 37km/h，目標只有 18），才被大幅延遲的煞車拉回來變成 bang-bang
    震盪。修法是完全不讀 get_control()，縱向/橫向控制量在這裡當場算好、
    一次送出。
    """
    throttle, brake, speed, slope_deg = _compute_pid_ff(vehicle, target_mps, pid, ff_gain)
    steer = _pure_pursuit_steer(vehicle, carla_map, lookahead_m, route)
    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle, steer=steer, brake=brake,
        hand_brake=False, manual_gear_shift=False))
    return speed, slope_deg


# ── 引數 ──────────────────────────────────────────────────────────────────────

def _parse_dest(s: str) -> carla.Location:
    """`--route-dest` 的 "X,Y" 或 "X,Y,Z"（世界座標，公尺）。

    只給 X,Y 時 z 填 0：GlobalRoutePlanner 會把終點投影到路面，高度不重要。
    """
    try:
        parts = [float(v) for v in s.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--route-dest 需要數字：{s}") from exc
    if len(parts) == 2:
        parts.append(0.0)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--route-dest 格式為 X,Y 或 X,Y,Z，收到：{s}")
    return carla.Location(x=parts[0], y=parts[1], z=parts[2])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA 資料集蒐集")
    p.add_argument("--host",          default="127.0.0.1", help="CARLA 伺服器位址")
    p.add_argument("--port",          type=int,   default=2000,  help="埠號")
    p.add_argument("--timeout",       type=float, default=20.0,  help="連線逾時秒數")
    p.add_argument("--speed",         type=float, default=18.0,  help="目標車速 km/h")
    p.add_argument("--camera-fps",    type=int,   default=40,    help="同步模式 FPS（預設：40）")
    p.add_argument("--z-offset",      type=float, default=0.0,
                   help="spectator z 偏移量（預設 0，依場景高度調整）")
    p.add_argument("--align-frames",  type=int,   default=20,
                   help="連續幾幀 steer<0.05 才視為對齊完成（預設：20）")
    p.add_argument("--warmup-frames", type=int,   default=10,
                   help="切換 TM→手動後再等幾幀才開始存檔（預設：10）")
    p.add_argument("--profile-step-m", type=float, default=_PROFILE_STEP_M,
                   help=f"前方剖面的採樣間距（公尺，預設 {_PROFILE_STEP_M}，"
                        f"對齊 legacy GT 的逐幀密度）。調大可換取採集速度")
    p.add_argument("--no-profile-ray", action="store_true",
                   help="關閉剖面向下射線（只存 waypoint 解析高度）。"
                        "每幀多一輪 cast_ray（點數 = max_d/step + 1）：同步模式下"
                        "這只拉長採集的牆鐘時間，不影響物理與車速，趕時間才需要關")
    p.add_argument("--lookahead-m",   type=float, default=6.0,
                   help="pure-pursuit 橫向修正的前視距離（預設 6 m），"
                        "取代寫死 steer=0 的假設")
    # ── 路線（行駛前就決定好，見 Route 的 docstring）──────────────────────
    p.add_argument("--route-length-m", type=float, default=1000.0,
                   help="預設（直走）路線要規劃多長，公尺（預設 1000）。"
                        "路線走完就自動停止採集；路提早到盡頭則路線提早結束")
    p.add_argument("--route-dest", type=_parse_dest, default=None, metavar="X,Y[,Z]",
                   help="指定終點世界座標，改用 CARLA 的 GlobalRoutePlanner "
                        "規劃路線（可含轉彎）。不給則從對齊後的位置一路直走")
    p.add_argument("--route-file", default=None, metavar="JSON",
                   help="載入 carla_module/pick_route.py 存下來的路線 JSON。"
                        "車輛會生成在路線起點（不看 spectator），路線直接用檔案裡"
                        "的點，不重新規劃。優先於 --route-dest / --route-length-m")
    # ── PID + FF 參數（可在命令列覆寫）──────────────────────────────────────
    p.add_argument("--kp",      type=float, default=1.0,
                   help="PID 比例增益（預設 1.0，調大反應更快但易震盪）")
    p.add_argument("--ki",      type=float, default=0.25,
                   help="PID 積分增益（預設 0.25）")
    p.add_argument("--kd",      type=float, default=0.15,
                   help="PID 微分增益（預設 0.15）")
    p.add_argument("--ff-gain", type=float, default=0.015,
                   help="坡度前饋增益（預設 0.015）")
    p.add_argument("--camera-preset", choices=sorted(CAMERA_PRESETS), default="default",
                   help="相機規格組（預設 default＝1280×720、FOV 90°、1.08 m；"
                        "openlane＝仿 OpenLane 前鏡頭，見 CAMERA_PRESETS）")
    p.add_argument("--cam-z",      type=float, default=None, help="覆寫相機離車輛原點高度（m）")
    p.add_argument("--cam-fov",    type=float, default=None, help="覆寫水平 FOV（度）")
    p.add_argument("--img-width",  type=int,   default=None, help="覆寫影像寬（px）")
    p.add_argument("--img-height", type=int,   default=None, help="覆寫影像高（px）")
    return p.parse_args()


def apply_camera_args(args) -> None:
    """把 --camera-preset 與個別覆寫寫進模組常數；main() 之後讀到的都是新值，
    metadata.json 也照實記下（推論端的 f_x / f_y / camera_height 從那裡推）。"""
    g = globals()
    g.update(CAMERA_PRESETS[args.camera_preset])
    for arg, name in (("cam_z", "CAMERA_HEIGHT"), ("cam_fov", "CAMERA_FOV"),
                      ("img_width", "IMG_WIDTH"), ("img_height", "IMG_HEIGHT")):
        if getattr(args, arg) is not None:
            g[name] = type(g[name])(getattr(args, arg))
    f = IMG_WIDTH / (2.0 * np.tan(np.radians(CAMERA_FOV) / 2.0))
    print(f"[相機] {IMG_WIDTH}×{IMG_HEIGHT}  FOV {CAMERA_FOV}°（f = {f:.1f} px）  "
          f"離車輛原點 {CAMERA_HEIGHT} m  前移 {CAMERA_FWD_X} m  preset={args.camera_preset}")


# ── 資料集寫入器 ───────────────────────────────────────────────────────────────

class DatasetWriter:
    """資料集寫入器：影像 + 逐幀量測 + 前方路面剖面 + 掛載中繼資料。

    設計原則是**存最生的量，之後再推導**。前方剖面存 waypoint 的世界座標
    (x, y, z) 而不是預先算好的 pitch —— 存 pitch 會把「相對哪個座標系」這個
    選擇燒死在資料裡，選錯就得重採；存世界座標的話，深度／高度剖面／pitch
    都能離線重算，連「弧長 vs 沿光軸深度」這種問題都變成算術而非假說。
    同理相機存的是完整 transform（位置 + 旋轉）而非只有位置：pipeline 的 z
    是沿光軸的距離，光軸方向由相機旋轉定義。
    """

    _CSV_HEADER = [
        "frame_id",
        # 車身姿態（含懸吊）—— 原有欄位，語意與名稱刻意不變，舊工具仍可讀
        "gt_pitch_deg",
        "gt_speed_mps",
        "collect_dist_m",
        # 路面姿態（相機所在處）—— 與車身分開記，才能量化懸吊造成的誤差。
        # 實測 body = 1.0213 x road - 0.48 度，車身振幅超出 2.13%
        "road_pitch_deg",
        "body_roll_deg",
        # 世界座標：有了它才能直接重建高度剖面，取代「積分 tan(pitch)」的近似
        "veh_x", "veh_y", "veh_z",
        "cam_x", "cam_y", "cam_z",
        "cam_pitch_deg", "cam_yaw_deg", "cam_roll_deg",
        "lane_id", "road_id",
    ]

    # 每幀 × 每個採樣點一列
    # `z` 是 waypoint 的解析高度（欄名不動，舊 reader 與舊資料集照常）；
    # `z_mesh` / `mesh_label` 是向下射線量到的渲染網格高度，射線模式關閉或
    # 沒打到地面語意時留空。
    _PROFILE_HEADER = [
        "frame_id", "d_req_m", "x", "y", "z", "wp_pitch_deg", "lane_id", "road_id",
        "z_mesh", "mesh_label",
    ]

    def __init__(self, root: pathlib.Path, map_name: str, metadata: dict) -> None:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = root / "outputs" / f"carla_dataset_{map_name}_{ts_str}"
        self.img_dir  = self.save_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.save_dir / "measurements.csv"
        self._csv_f  = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_f)
        self._writer.writerow(self._CSV_HEADER)

        prof_path = self.save_dir / "road_profile.csv"
        self._prof_f = open(prof_path, "w", newline="", encoding="utf-8")
        self._prof_writer = csv.writer(self._prof_f)
        self._prof_writer.writerow(self._PROFILE_HEADER)

        # 掛載參數寫進資料集本身：現在推論 config 的 camera_forward_offset /
        # camera_height 是手抄過去的，兩邊沒有任何連結，改一邊另一邊不會知道
        self._metadata = metadata
        self._flush_metadata()

        self._count = 0
        print(f"[資料集] 儲存至：{self.save_dir}")

    def _flush_metadata(self) -> None:
        with open(self.save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)

    def update_metadata(self, patch: dict) -> None:
        """補寫 metadata。路線要等 ALIGNING 結束、車輛位置定了才算得出來，
        但資料集目錄在那之前就得建好（ALIGNING 期間的 print 要有地方指），
        所以分兩次寫。"""
        self._metadata.update(patch)
        self._flush_metadata()

    @property
    def count(self) -> int:
        return self._count

    def save(
        self,
        image:          carla.Image,
        transform:      carla.Transform,
        speed_mps:      float,
        collect_dist_m: float,
        cam_tf:         carla.Transform,
        profile:        list,
    ) -> None:
        """存一幀。`profile` 來自 `sample_road_profile`，是
        (要求距離, waypoint, 網格高度, 網格標籤) 序列；其第 0 點就是相機
        所在處的 waypoint，所以不必再查一次地圖。"""
        img_path = str(self.img_dir / f"{self._count:06d}.png")
        image.save_to_disk(img_path)

        wp_cam = profile[0][1] if profile else None
        vl, cl, cr = transform.location, cam_tf.location, cam_tf.rotation

        # pitch 一律 wrap 到 (-180, 180]：CARLA 的平路 waypoint pitch 實測會
        # 回 360.0，不 wrap 的話下游會讀到「路面 pitch +360 度」
        row = [
            self._count,
            f"{_wrap_deg(transform.rotation.pitch):.4f}",
            f"{speed_mps:.4f}",
            f"{collect_dist_m:.4f}",
            "" if wp_cam is None else f"{_wrap_deg(wp_cam.transform.rotation.pitch):.4f}",
            f"{_wrap_deg(transform.rotation.roll):.4f}",
            f"{vl.x:.4f}", f"{vl.y:.4f}", f"{vl.z:.4f}",
            f"{cl.x:.4f}", f"{cl.y:.4f}", f"{cl.z:.4f}",
            f"{_wrap_deg(cr.pitch):.4f}", f"{_wrap_deg(cr.yaw):.4f}", f"{_wrap_deg(cr.roll):.4f}",
            "" if wp_cam is None else int(wp_cam.lane_id),
            "" if wp_cam is None else int(wp_cam.road_id),
        ]
        self._writer.writerow(row)
        self._csv_f.flush()

        for d_req, wp, z_mesh, mesh_label in profile:
            loc = wp.transform.location
            self._prof_writer.writerow([
                self._count, f"{d_req:.2f}",
                f"{loc.x:.4f}", f"{loc.y:.4f}", f"{loc.z:.4f}",
                f"{_wrap_deg(wp.transform.rotation.pitch):.4f}",
                int(wp.lane_id), int(wp.road_id),
                "" if z_mesh is None else f"{z_mesh:.4f}",
                "" if mesh_label is None else mesh_label,
            ])
        self._prof_f.flush()

        self._count += 1

    def close(self) -> None:
        self._csv_f.close()
        self._prof_f.close()
        print(f"[資料集] 共儲存 {self._count} 幀。路徑：{self.save_dir}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    apply_camera_args(args)
    root_dir = pathlib.Path(__file__).parent.parent

    # ── 連線 ──────────────────────────────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world     = client.get_world()
    carla_map = world.get_map()
    print(f"[初始化] 連線至地圖：{carla_map.name}")
    bp_lib = world.get_blueprint_library()

    # ── 路線檔（有的話，起點與路線都由它決定）────────────────────────────────
    preplanned: Optional[Route] = None
    if args.route_file:
        preplanned = load_route_file(pathlib.Path(args.route_file), carla_map)

    # ── 生成車輛 ──────────────────────────────────────────────────────────────
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    if preplanned is not None:
        # 生成在路線起點、車頭朝路線方向。走這條就不看 spectator 了 ——
        # 路線是在俯視圖上挑的，起點理應由路線決定，再要求使用者手動把鏡頭
        # 對到同一個地方只是多一個出錯的機會
        spawn_transform = route_start_transform(preplanned)
        origin_desc     = f"路線檔起點（{args.route_file}）"
    else:
        spectator       = world.get_spectator()
        spawn_transform = spectator.get_transform()
        origin_desc     = "spectator 位置"
    spawn_transform.location.z    += args.z_offset
    spawn_transform.rotation.pitch = 0.0
    spawn_transform.rotation.roll  = 0.0

    vehicle: carla.Vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if vehicle is None:
        raise RuntimeError(
            f"無法在{origin_desc}生成車輛（可能不在可通行路面上、或該處已有物體），"
            "請換個起點，或以 --z-offset 調整偏移量（預設 0）"
        )
    vehicle.set_autopilot(False)
    print(f"[初始化] 車輛生成於{origin_desc}：{spawn_transform.location}")

    TARGET_SPEED_MPS = args.speed / 3.6

    # ── 同步模式 ──────────────────────────────────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / args.camera_fps
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.CloudySunset)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # ── PID + FF 控制器（dt 對齊同步模式 FPS）────────────────────────────────
    pid = PIDController(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        dt=1.0 / args.camera_fps,   # 與 tick 週期一致，微分項才準確
        integral_limit=3.0,
    )
    ff_gain = args.ff_gain
    print(f"[PID+FF] kp={args.kp}  ki={args.ki}  kd={args.kd}  ff_gain={ff_gain}")

    # ── 感測器 ────────────────────────────────────────────────────────────────
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
    cam_bp.set_attribute("fov", str(CAMERA_FOV))
    cam_tf = carla.Transform(carla.Location(x=CAMERA_FWD_X, z=CAMERA_HEIGHT))
    camera: carla.Sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    image_queue: queue.Queue = queue.Queue()
    camera.listen(image_queue.put)

    # ── 物理預熱 ──────────────────────────────────────────────────────────────
    print(f"[初始化] 物理預熱 {PHYSICS_WARMUP_TICKS} ticks...")
    for _ in range(PHYSICS_WARMUP_TICKS):
        world.tick()

    # ── 啟動 TM 對齊 ──────────────────────────────────────────────────────────
    vehicle.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle, 100.0)
    tm.ignore_signs_percentage(vehicle,  100.0)
    tm.set_desired_speed(vehicle, args.speed)
    # TM 是完整導航，遇路口會照自己的邏輯選路 —— ALIGNING 通常只有不到一秒，
    # 但起點就在路口前的話它照樣會轉走，那之後規劃的路線就不是你要的那條了。
    # 這裡叫它一路直走；此 API 不是每個版本都有，沒有就只是回到舊行為
    try:
        tm.set_route(vehicle, ["Straight"] * 20)
    except (AttributeError, RuntimeError) as exc:                # noqa: BLE001
        print(f"[警告] TM set_route 不可用（{exc}）；"
              f"ALIGNING 期間若遇路口可能被 TM 轉走")

    state          = State.ALIGNING
    align_counter  = 0
    warmup_counter = 0
    route: Optional[Route] = None

    # ── 距離追蹤 ──────────────────────────────────────────────────────────────
    total_distance_m   = 0.0   # 累計總行駛距離（含 ALIGNING/WARMUP）
    collect_distance_m = 0.0   # 僅 COLLECTING 階段的距離
    prev_location: Optional[carla.Location] = None
    # 剖面取樣耗時（最近 50 幀）：射線模式每幀多一輪 cast_ray（點數 =
    # max_d/step + 1，預設 401 次）。這是**牆鐘**
    # 成本，不是模擬預算 —— 同步模式下每次 world.tick() 推進固定的
    # fixed_delta_seconds，RPC 再慢也不會改變物理、PID 的 dt 或車速曲線，只會
    # 讓整趟採集在現實時間裡變久。顯示出來是為了估採集要花多久，不是為了看
    # 有沒有超時
    profile_ms = deque(maxlen=50)

    map_name = carla_map.name.split("/")[-1]
    metadata = {
        "created":    datetime.datetime.now().isoformat(timespec="seconds"),
        "map":        carla_map.name,
        "vehicle_bp": vehicle_bp.id,
        # 推論端的 camera_forward_offset / camera_height / f_x / f_y 都從這裡推
        "camera": {
            "x_forward_m": CAMERA_FWD_X,
            "z_height_m":  CAMERA_HEIGHT,
            "fov_deg":     CAMERA_FOV,
            "img_width":   IMG_WIDTH,
            "img_height":  IMG_HEIGHT,
            "preset":      args.camera_preset,
        },
        "road_profile": {
            "max_d_m":     _PROFILE_MAX_D_M,
            "step_m":      args.profile_step_m,
            "origin":      "camera",   # waypoint 起點是相機所在處，不是車輛原點
            "stored":      "world xyz of each waypoint (raw); pitch is a convenience column",
            "ray_mode":    not args.no_profile_ray,
            "ray_up_m":    _PROFILE_RAY_UP_M,
            "ray_depth_m": _PROFILE_RAY_DEPTH_M,
            "ray_labels":  sorted(GROUND_LABELS),
            "z_mesh":      "downward-ray hit on the rendered mesh; blank when "
                           "ray mode is off or no ground-semantic hit",
        },
        "collection": {
            "target_speed_kmh": args.speed,
            "camera_fps":       args.camera_fps,
            "lookahead_m":      args.lookahead_m,
            "pid":              {"kp": args.kp, "ki": args.ki, "kd": args.kd},
            "ff_gain":          args.ff_gain,
        },
    }
    writer   = DatasetWriter(root_dir, map_name, metadata)
    win_name = "CARLA Dataset Collection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print(f"[{state}] TM 對齊車道中（需連續 {args.align_frames} 幀 steer < 0.05）")

    try:
        while True:
            target_frame = world.tick()

            # ── 取出與本 tick 對齊的相機幀 ────────────────────────────────────
            image: Optional[carla.Image] = None
            try:
                while True:
                    temp = image_queue.get(timeout=2.0)
                    if temp.frame == target_frame:
                        image = temp
                        break
                    if temp.frame > target_frame:
                        break
            except queue.Empty:
                print(f"[警告] 幀 {target_frame} 影像遺失，跳過")
                continue

            if image is None:
                continue

            transform    = vehicle.get_transform()
            vel          = vehicle.get_velocity()
            speed_mps    = vel.length()
            cur_location = transform.location

            # ── 距離累加（每幀位移，同步模式下精度高）────────────────
            if prev_location is not None:
                frame_dist = cur_location.distance(prev_location)
                total_distance_m += frame_dist
                if state == State.COLLECTING:
                    collect_distance_m += frame_dist
            prev_location = cur_location

            # ── 狀態機 ────────────────────────────────────────────────────────
            slope_deg = 0.0

            if state == State.ALIGNING:
                steer = vehicle.get_control().steer
                if abs(steer) < 0.05:
                    align_counter += 1
                else:
                    align_counter = 0

                if align_counter >= args.align_frames:
                    vehicle.set_autopilot(False)
                    pid.reset()   # 切換後清除積分，避免舊值干擾

                    # 路線在這裡定案：起點是**對齊完成後**的車輛位置，車頭已
                    # 經順著車道，選分岔時的朝向才乾淨。從生成點規劃不行 ——
                    # ALIGNING 期間 TM 已經把車開走了一段
                    if preplanned is not None:
                        # 路線檔的點就是要走的路，重新規劃只會把選好的路蓋掉。
                        # ALIGNING 期間 TM 會把車往前開一小段，Route 的 cursor
                        # 自己會前進到對應位置，不需要特別處理
                        route      = preplanned
                        route_mode = f"file {args.route_file}"
                    elif args.route_dest is not None:
                        route = build_route_to(
                            carla_map, cur_location, args.route_dest)
                        route_mode = f"dest {args.route_dest}"
                    else:
                        route = build_route_straight(
                            carla_map, cur_location, args.route_length_m,
                            start_fwd=transform.get_forward_vector())
                        route_mode = f"straight {args.route_length_m:.0f} m"

                    if len(route) < 2:
                        raise RuntimeError(
                            f"路線規劃失敗（{route_mode}）：只得到 {len(route)} 個點。"
                            f"起點可能不在可行駛路面上，或終點無法到達"
                        )
                    print(f"[路線] {route_mode} → 實際 {route.length_m:.1f} m / "
                          f"{len(route)} 點（行駛中不再查地圖）")
                    writer.update_metadata({"route": {
                        "mode":       ("file" if preplanned is not None else
                                       "dest" if args.route_dest else "straight"),
                        "requested":  (args.route_file if preplanned is not None else
                                       f"{args.route_dest.x},{args.route_dest.y},"
                                       f"{args.route_dest.z}" if args.route_dest
                                       else args.route_length_m),
                        "planned_m":  round(route.length_m, 2),
                        "points":     len(route),
                        "step_m":     _ROUTE_STEP_M,
                        "start":      [round(cur_location.x, 3),
                                       round(cur_location.y, 3),
                                       round(cur_location.z, 3)],
                    }})

                    state          = State.WARMUP
                    warmup_counter = 0
                    print(f"[{state}] TM 已關閉，等待 {args.warmup_frames} 幀物理穩定...")

            elif state == State.WARMUP:
                speed_mps, slope_deg = apply_pid_ff_control_with_steering(
                    vehicle, carla_map, TARGET_SPEED_MPS, pid, ff_gain,
                    args.lookahead_m, route)
                warmup_counter += 1
                if warmup_counter >= args.warmup_frames:
                    # ALIGNING 是 TM 在開，它可能把車帶到隔壁車道；--route-file
                    # 模式尤其明顯（生成點與路線是選好的，TM 卻自己走一段）。
                    # 橫向偏差超過半個車道就代表接下來整批影像拍的不是路線上
                    # 那條車道 —— 這種錯只會在事後對 GT 時才發現，先喊出來
                    if route is not None and route.last_lateral > _LATERAL_WARN_M:
                        print(f"[警告] 開始採集時離路線 {route.last_lateral:.2f} m"
                              f"（>{_LATERAL_WARN_M} m，約半個車道）—— 車可能在隔壁"
                              f"車道上。建議按 q 中止，把起點移到路口前後的直路段重來")
                    state = State.COLLECTING
                    print(f"[{state}] 開始存檔！按 'q' 停止")

            elif state == State.COLLECTING:
                speed_mps, slope_deg = apply_pid_ff_control_with_steering(
                    vehicle, carla_map, TARGET_SPEED_MPS, pid, ff_gain,
                    args.lookahead_m, route)
                cam_tf   = camera.get_transform()
                _t0      = time.perf_counter()
                profile  = sample_road_profile(
                    carla_map, cam_tf.location, step_m=args.profile_step_m,
                    world=None if args.no_profile_ray else world)
                profile_ms.append((time.perf_counter() - _t0) * 1e3)
                writer.save(image, transform, speed_mps, collect_distance_m,
                            cam_tf, profile)

                # 路線走完就收工。繼續開下去就沒有預定路線可循了，等於退回
                # 每幀查地圖的舊行為，正是路口會轉錯的那種狀態
                if route is not None and route.remaining_m() < args.lookahead_m:
                    print(f"[路線] 已走完 {route.length_m:.1f} m，停止採集")
                    break

            # ── 顯示視窗 ──────────────────────────────────────────────────────
            arr     = np.frombuffer(image.raw_data, dtype=np.uint8)
            display = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()
            h, font = display.shape[0], cv2.FONT_HERSHEY_SIMPLEX

            state_color = {
                State.ALIGNING:   (0, 165, 255),
                State.WARMUP:     (0, 255, 255),
                State.COLLECTING: (0, 255, 0),
            }[state]

            cv2.putText(display,
                        f"[{state}]  Saved: {writer.count}",
                        (10, 30),     font, 0.8, state_color,   2, cv2.LINE_AA)
            cv2.putText(display,
                        f"GT Pitch: {transform.rotation.pitch:+.2f} deg  "
                        f"Slope: {slope_deg:+.1f} deg",
                        (10, 62),     font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display,
                        f"Speed: {speed_mps * 3.6:.1f} km/h  "
                        f"Steer: {vehicle.get_control().steer:+.3f}",
                        (10, 94),     font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display,
                        f"Dist(total): {total_distance_m:.1f} m  "
                        f"Dist(collect): {collect_distance_m:.1f} m",
                        (10, 126),    font, 0.8, (180, 255, 180), 2, cv2.LINE_AA)
            if route is not None:
                # 橫向偏差是「有沒有乖乖跟著路線」的直接指標：正常應該是幾公分，
                # 持續變大就代表 pure-pursuit 沒跟上（不是路口選錯了）
                cv2.putText(display,
                            f"Route: {route.remaining_m():.0f} m left  "
                            f"Lateral: {route.last_lateral * 100:.0f} cm",
                            (10, 158),    font, 0.8, (255, 200, 120), 2, cv2.LINE_AA)
            if profile_ms:
                cv2.putText(display,
                            f"Profile: {sum(profile_ms) / len(profile_ms):.1f} ms/frame "
                            f"wall-clock  (ray {'off' if args.no_profile_ray else 'on'})",
                            (10, 190),    font, 0.8, (200, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(display,
                        "Press Q to stop",
                        (10, h - 12), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        print("[清理] 恢復非同步模式...")
        s = world.get_settings()
        s.synchronous_mode    = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        writer.close()
        print(f"[距離統計] 總行駛：{total_distance_m:.1f} m  "
              f"採集階段：{collect_distance_m:.1f} m")


if __name__ == "__main__":
    main()
