"""OpenLane 2D 車道標註 → CULane 格式（CLRNet 等 2D 偵測器的訓練／評估輸入，WWH-25）。

每一幀輸出（``<out>/<split>/<segment>/``）：

    <stem>.jpg          影像，NTFS 硬連結指向原檔（不同磁碟區才複製）
    <stem>.lines.txt    每條車道線一列 "x1 y1 x2 y2 ..."（原圖像素，y 由下而上）
    <stem>.png          分割遮罩：背景 0，車道線依位置 1–4，線寬 --mask-width px

以及 ``<out>/list/<split>_gt.txt``：每列 ``/<split>/<segment>/<stem>.jpg
/<split>/<segment>/<stem>.png e1 e2 e3 e4``（e = 該位置有沒有線）。CLRNet 的
CULane loader 讀 ``list/train_gt.txt``；train split 的輸出檔名正好是它。

對應規則
    * 用 OpenLane 的 **2D 標註 ``uv``**（原圖像素，官方 2D 評測用的就是它），
      不是 3D ``xyz`` 的投影。
    * 只收 ``attribute`` 1–4 的線：OpenLane 把它定義為 左左／左／右／右右，
      正好是 CULane 的四個車道線位置，遮罩值直接用 attribute。
      attribute 0 的線（第 5 條以後、以及**所有路緣** —— 實測路緣從不帶
      attribute，見 reference 記憶／WWH-25）不輸出。
    * 影像範圍外的點丟掉、重複點去掉；少於 2 點的線丟掉。
    * 影像不縮放、不裁切：解析度與 cut_height 是訓練設定的事，不是資料的事。
    * 空標籤幀（沒有 attribute 1–4 的線）依原因分類，見 frame_kind。``--stats-only``
      只統計不輸出（驗證集 42.7% 空：路緣 19.3%、沒線 9.2%、遠處漆線 11.4%、
      自車道位置上有漆線但沒標 2.7%）。空標籤幀**預設全部保留**。
      ``ego_unlabelled`` **不是漏標**：驗證集 96% 帶官方「路口」標籤（far_paint
      也是 96%，有標註的幀只有 33%），是 OpenLane 的慣例——車子在路口裡時不定義
      自車道。保留它，模型學到「在路口裡不畫自車道線」，與同樣在路口的 far_paint
      一致（使用者 2026-09-26 決定）。``--drop-ego-unlabelled`` 可排除。
    * ``--extend-bottom``：OpenLane 標註通常從 8–14 m 才開始（光達＋未來軌跡），
      影像底部那段看得到卻沒標。這個選項把每條線依最近那段直線延伸到底部，
      跟 CULane 的標註習慣一致（見 extend_to_bottom）。預設**不延伸**；
      微調時兩種都試，用近場量寬的準確度決定。

⚠ 授權 CC BY-NC-SA ＋ Waymo 非商業：輸出放在 repo 之外，影像不進版控。

    python -m openlane_module.convert_openlane_culane --split training \\
        --openlane <OpenLane 根目錄> --out <輸出根目錄>
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.env_setup import setup_env  # noqa: E402

setup_env()
import cv2  # noqa: E402

SLOTS = (1, 2, 3, 4)          # OpenLane attribute: left-left, left, right, right-right
MASK_WIDTH_PX = 16            # CULane laneseg_label_w16 (its own images are 1640 wide)
# --extend-bottom (see extend_to_bottom): fit on the lowest 20 % of a lane's
# vertical span, at least 5 points; emit a point every 10 px down to the bottom.
EXTEND_FIT_FRAC = 0.2
EXTEND_MIN_PTS = 5
EXTEND_STEP_PX = 10
CURB = (20, 21)
# ego_unlabelled: an attribute-0 paint line within this lateral distance (m) of
# the camera somewhere 5-20 m ahead — i.e. where an ego line would be. Half a
# 3.5 m lane plus margin; the near range is where the labels normally exist.
EGO_LAT_M = 2.5
EGO_Z_RANGE_M = (5.0, 20.0)
SRC_W, SRC_H = 1920, 1280      # OpenLane front camera (--stats-only reads no images)


def lanes_to_culane(lane_lines, width, height, extend_bottom=False):
    """{slot: (N, 2) float array of (x, y), y descending} for attribute-1..4 lanes;
    with extend_bottom, each lane continued straight to the image bottom."""
    out = {}
    for ln in lane_lines:
        slot = ln.get('attribute', 0)
        if slot not in SLOTS:
            continue
        uv = np.asarray(ln['uv'], dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] != 2 or uv.shape[1] < 2:
            continue
        pts = uv.T
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
        pts = np.unique(pts[ok], axis=0)
        if len(pts) < 2:
            continue
        pts = pts[np.argsort(-pts[:, 1])]            # CULane: bottom of the image first
        out[slot] = extend_to_bottom(pts, width, height) if extend_bottom else pts
    return out


def frame_kind(lane_lines, width, height, lanes=None):
    """Why a frame has (or lacks) attribute-1..4 labels:

    labelled        at least one attribute-1..4 lane inside the image
    out_of_image    attribute-1..4 lanes exist but none inside the image
    no_lanes        no lane annotation at all
    curb_only       only curbs (category 20/21)
    ego_unlabelled  an attribute-0 PAINT line beside the ego position
                    (|lateral| ≤ EGO_LAT_M somewhere in EGO_Z_RANGE_M ahead)
    far_paint       attribute-0 paint lines, none beside the ego position

    `lanes`: this frame's lanes_to_culane result if the caller already has it
    (only whether it is empty matters), so it is not converted twice.
    """
    if lanes is None:
        lanes = lanes_to_culane(lane_lines, width, height)
    if lanes:
        return 'labelled'
    if any(ln.get('attribute', 0) in SLOTS for ln in lane_lines):
        return 'out_of_image'
    if not lane_lines:
        return 'no_lanes'
    paint = [ln for ln in lane_lines if ln.get('category') not in CURB]
    if not paint:
        return 'curb_only'
    lo, hi = EGO_Z_RANGE_M
    for ln in paint:
        xyz = np.asarray(ln['xyz'], dtype=np.float64)   # [forward, lateral, up], as inner_pair
        m = (xyz[0] >= lo) & (xyz[0] <= hi)
        if m.any() and np.abs(xyz[1][m]).min() <= EGO_LAT_M:
            return 'ego_unlabelled'
    return 'far_paint'


def extend_to_bottom(pts, width, height, frac=EXTEND_FIT_FRAC, step=EXTEND_STEP_PX):
    """Lane points (y descending) extended straight down to the image bottom.

    OpenLane's labels come from lidar + the future trajectory, so they usually
    start 8-14 m ahead: the road between the image bottom (~6.9 m on this
    camera) and the first label is visible but unlabelled. CULane labels —
    what the pretrained weights learned — run to the bottom. Fine-tuning on
    the raw labels would leave that band unsupervised, and the near field is
    exactly what this project measures the lane width in (WWH-25).

    The extension is a straight line fitted (least squares, x on y) to the
    lowest `frac` of the lane's vertical span (at least EXTEND_MIN_PTS points):
    a straight lane on a plane projects to a straight image line, so over the
    few metres involved this is the CULane annotators' convention made exact
    for straight flat road. It stops at the image border. Returns the input
    unchanged when it already reaches within `step` px of the bottom or has
    too few points to fit.
    """
    y_low = pts[0, 1]                       # lowest point = largest y
    if y_low >= height - 1 - step or len(pts) < EXTEND_MIN_PTS:
        return pts
    span = y_low - pts[-1, 1]
    near = pts[pts[:, 1] >= y_low - max(frac * span, 1.0)]
    if len(near) < EXTEND_MIN_PTS:
        near = pts[:EXTEND_MIN_PTS]
    a, b = np.polyfit(near[:, 1], near[:, 0], 1)
    ext = []
    y = y_low + step
    while y <= height - 1:                  # walk down from the lane's end
        x = a * y + b
        if not 0 <= x < width:
            break                           # stop at the image border
        ext.append((x, y))
        y += step
    if not ext:
        return pts
    return np.vstack([np.array(ext[::-1]), pts])     # still y descending


def lines_txt(lanes):
    return ''.join(' '.join(f'{x:.2f} {y:.2f}' for x, y in pts) + '\n'
                   for _, pts in sorted(lanes.items()))


def mask_png(lanes, width, height, mask_width=MASK_WIDTH_PX):
    m = np.zeros((height, width), dtype=np.uint8)
    for slot, pts in lanes.items():
        cv2.polylines(m, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False,
                      int(slot), thickness=int(mask_width))
    return m


def _link(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--openlane', type=Path, default=Path('D:/datasets/openlane'),
                    help='OpenLane 根目錄：底下有 <split>/<segment>/*.json，影像依 json 的 file_path')
    ap.add_argument('--split', default='validation', help='training 或 validation')
    ap.add_argument('--out', type=Path, default=Path('D:/datasets/openlane_culane'))
    ap.add_argument('--mask-width', type=int, default=MASK_WIDTH_PX)
    ap.add_argument('--skip-empty', action='store_true', help='不輸出任何沒有 attribute 1–4 車道線的幀')
    ap.add_argument('--drop-ego-unlabelled', action='store_true',
                    help='排除 ego_unlabelled 幀（自車道位置上有漆線但沒標；96%% 在路口，見 frame_kind）')
    ap.add_argument('--stats-only', action='store_true',
                    help='只統計各類幀數（frame_kind），不讀影像、不輸出')
    ap.add_argument('--extend-bottom', action='store_true',
                    help='每條線依最近那段直線延伸到影像底部（OpenLane 近處沒標註，見 extend_to_bottom）')
    ap.add_argument('--limit-segments', type=int, default=0, help='只轉前 N 段（冒煙測試）')
    ap.add_argument('--segment', action='append', default=[], help='只轉這些段（可重複）')
    a = ap.parse_args(argv)

    segs = sorted(p for p in (a.openlane / a.split).glob('segment-*') if p.is_dir())
    if a.segment:
        segs = [p for p in segs if p.name in set(a.segment)]
    if a.limit_segments:
        segs = segs[:a.limit_segments]
    kinds = {}
    if a.stats_only:
        for i, seg in enumerate(segs):
            for js in sorted(seg.glob('*.json')):
                if not js.name.startswith('._'):
                    k = frame_kind(json.loads(js.read_text(encoding='utf-8'))['lane_lines'],
                                   SRC_W, SRC_H)
                    kinds[k] = kinds.get(k, 0) + 1
            if i % 100 == 0:
                print(f'[{i}/{len(segs)}]', flush=True)
        n = sum(kinds.values())
        print(f'{n} frames, {len(segs)} segments ({a.split}):')
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f'  {k:15s} {v:7d}  {v / n:6.1%}')
        return

    (a.out / 'list').mkdir(parents=True, exist_ok=True)
    listing, n_img, n_missing, n_empty, per_slot = [], 0, 0, 0, dict.fromkeys(SLOTS, 0)
    for i, seg in enumerate(segs):
        dst_dir = a.out / a.split / seg.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for js in sorted(seg.glob('*.json')):
            if js.name.startswith('._'):
                continue
            data = json.loads(js.read_text(encoding='utf-8'))
            lanes = lanes_to_culane(data['lane_lines'], SRC_W, SRC_H, a.extend_bottom)
            kind = frame_kind(data['lane_lines'], SRC_W, SRC_H, lanes=lanes)
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind == 'ego_unlabelled' and a.drop_ego_unlabelled:
                continue
            src_img = a.openlane / data['file_path']
            if not src_img.exists():
                n_missing += 1
                continue
            img = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            if (w, h) != (SRC_W, SRC_H):             # OpenLane is 1920x1280; convert again only if not
                lanes = lanes_to_culane(data['lane_lines'], w, h, a.extend_bottom)
            if not lanes:
                n_empty += 1
                if a.skip_empty:
                    continue
            stem = js.stem
            _link(src_img, dst_dir / f'{stem}.jpg')
            (dst_dir / f'{stem}.lines.txt').write_text(lines_txt(lanes), encoding='utf-8')
            cv2.imwrite(str(dst_dir / f'{stem}.png'), mask_png(lanes, w, h, a.mask_width))
            rel = f'/{a.split}/{seg.name}/{stem}'
            listing.append(f'{rel}.jpg {rel}.png ' + ' '.join('1' if s in lanes else '0' for s in SLOTS))
            for s in lanes:
                per_slot[s] += 1
            n_img += 1
        if i % 20 == 0:
            print(f'[{i}/{len(segs)}] {seg.name[:40]} images={n_img}', flush=True)
    list_name = 'train_gt.txt' if a.split == 'training' else f'{a.split}_gt.txt'
    (a.out / 'list' / list_name).write_text('\n'.join(listing) + '\n', encoding='utf-8')
    print(f'{n_img} images from {len(segs)} segments -> {a.out}  '
          f'(list/{list_name}; no image file {n_missing}; no attribute-1..4 lane {n_empty}'
          f'{" skipped" if a.skip_empty else " kept"}; lanes per slot {per_slot})')
    print(f'frame kinds (before skipping): {kinds}'
          f'{"  (ego_unlabelled skipped)" if a.drop_ego_unlabelled else ""}')


if __name__ == '__main__':
    main()
