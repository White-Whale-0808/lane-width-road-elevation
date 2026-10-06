"""OpenLane 官方「上下坡」標籤（test/1000_updown.txt）的所有幀 -> 本專案格式，逐片段一個目錄。

與 ``convert_openlane.py`` 的差別：**不篩選**。官方標了 up&down 的每一幀都輸出，
轉換器原本會踢掉的條件改成欄位記下來（``drop_reasons``），讓下游自己決定：

    no_left / no_right   PROBE_X 處找不到該側內側線（inner_pair 同規則）
    category             內側線不是漆線（例如 curbside、unknown）
    width                兩條內側線間距不在 [WIDTH_MIN, WIDTH_MAX]
    short                GT 剖面深度跨度 < 5 m

剖面（road_profile.csv）只用**漆線**（PAINT 類別）算：兩條內側線都是漆線 -> 高度
平均（＝車道中心，與轉換器同）；只有一條 -> 用那一條（``profile_source`` 欄記是哪個）；
都沒有 -> 該幀沒有剖面列，影像照樣輸出。
⚠ 路緣（left/right-curbside）不拿來當路面高度：實測片段 191862526745161106 只有
右側路緣標註，高度一路爬到 +20.8 m（沿牆／路緣頂標的），算出 60 m 內起伏 15.5 m。
幀不會因此被剔除 —— 這是真值來源的選擇，不是篩選。

影像、虛擬內參、相機位姿寫法、collect_dist_m 的算法都與轉換器相同
（里程累加整段所有幀，不只 up&down 那些）。

另外輸出 ``<out>/index.csv``：每片段一列（幀數、有剖面的幀數、兩側都有的幀數、
起伏量、其他官方標籤的幀數），以及 ``<out>/frames.csv``：所有片段的逐幀表合併。

用法::

    python -m openlane_module.export_updown --openlane D:/datasets/openlane \\
        --out D:/datasets/openlane_updown
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from openlane_module.convert_openlane import (  # noqa: E402  （會先呼叫 setup_env）
    inner_pair, warp_image, CAT, PAINT, SOLID, WIDTH_MIN, WIDTH_MAX,
    GRADE_SPAN, PROFILE_MAX_M, PROFILE_STEP_M, VIRT_FX, VIRT_FY, VIRT_CX, VIRT_CY,
    OUT_W, OUT_H)
from openlane_module.frame_tags import load_cases  # noqa: E402

MIN_SPAN_M = 5.0     # 與 frame_record 的 z1 - z0 < 5 同一條


def frame_record(data):
    """一幀 -> (meta, depth, height)；沒有任何內側線時 depth/height 為 None。"""
    left, right = inner_pair(data['lane_lines'])
    reasons = []
    if left is None:
        reasons.append('no_left')
    if right is None:
        reasons.append('no_right')
    if any(g is not None and g[1] not in PAINT for g in (left, right)):
        reasons.append('category')

    width = (left[0] - right[0]) if (left and right) else np.nan
    if left and right and not (WIDTH_MIN <= width <= WIDTH_MAX):
        reasons.append('width')

    paint = [g if (g is not None and g[1] in PAINT) else None for g in (left, right)]
    lines = [g[2] for g in paint if g is not None]
    depth = height = None
    z0 = z1 = grade = np.nan
    if lines:
        z0 = max(x[0].min() for x in lines)
        z1 = min(min(x[0].max() for x in lines), PROFILE_MAX_M)
        if z1 > z0:
            depth = np.arange(z0, z1 + 1e-9, PROFILE_STEP_M)
            height = np.mean([np.interp(depth, x[0], x[2]) for x in lines], axis=0)
            near = depth <= z0 + GRADE_SPAN
            grade = float(height[near].max() - height[near].min())
        if not (z1 - z0 >= MIN_SPAN_M):
            reasons.append('short')

    cat = lambda g: CAT.get(g[1], g[1]) if g is not None else ''
    meta = dict(
        left_category=cat(left), right_category=cat(right),
        left_solid=int(left is not None and left[1] in SOLID),
        right_solid=int(right is not None and right[1] in SOLID),
        lane_width_gt=round(width, 4) if np.isfinite(width) else np.nan,
        z_gt_min=round(float(z0), 2) if np.isfinite(z0) else np.nan,
        z_gt_max=round(float(z1), 2) if np.isfinite(z1) else np.nan,
        grade_60m=round(grade, 4) if np.isfinite(grade) else np.nan,
        profile_source=('both' if paint[0] and paint[1] else
                        'left' if paint[0] else 'right' if paint[1] else 'none'),
        drop_reasons='|'.join(reasons),
    )
    return meta, depth, height


def export_segment(seg_dir, out_dir, image_root, keep_stems, tags, want_images,
                   subset='official up&down tag (test/1000_updown.txt), unfiltered'):
    rows, prof_rows = [], []
    travelled, prev = 0.0, None
    fid = 0
    for path in sorted(seg_dir.glob('*.json')):
        if path.name.startswith('._'):
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        pos = np.asarray(data['pose'], dtype=float)[:3, 3]
        if prev is not None:
            travelled += float(np.linalg.norm(pos - prev))
        prev = pos
        if path.stem not in keep_stems:
            continue
        meta, depth, height = frame_record(data)
        img_ok = False
        if want_images:
            src = Path(image_root) / data['file_path']
            img_ok = src.exists() and warp_image(
                src, out_dir / 'images' / ('%06d.png' % fid), data['intrinsic'])
        rows.append(dict(
            frame_id=fid, cam_x=0.0, cam_y=0.0, cam_z=0.0,
            cam_pitch_deg=0.0, cam_yaw_deg=0.0, cam_roll_deg=0.0,
            collect_dist_m=round(travelled, 4), source_frame=path.stem,
            has_image=int(img_ok), **meta,
            **{f'tag_{t}': int((seg_dir.name, path.stem) in s) for t, s in tags.items()}))
        if depth is not None:
            for d, h in zip(depth, height):
                prof_rows.append((fid, round(float(d), 3), round(float(d), 4), 0.0,
                                  round(float(h), 5)))
        fid += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'measurements.csv', index=False)
    pd.DataFrame(prof_rows, columns=['frame_id', 'd_req_m', 'x', 'y', 'z']
                 ).to_csv(out_dir / 'road_profile.csv', index=False)
    with open(out_dir / 'metadata.json', 'w') as fh:
        json.dump(dict(source='OpenLane', subset=subset,
                       segment=seg_dir.name, frames=len(rows),
                       f_x=round(VIRT_FX, 4), f_y=round(VIRT_FY, 4), c_x=VIRT_CX, c_y=VIRT_CY,
                       resize_size=[OUT_H, OUT_W], camera_height=2.116,
                       camera_forward_offset=0.0,
                       note='virtual intrinsics; camera pose written as identity; '
                            'no filtering - see drop_reasons / profile_source'),
                  fh, indent=2)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--openlane', type=Path, default=Path('D:/datasets/openlane'))
    ap.add_argument('--out', type=Path, default=Path('D:/datasets/openlane_updown'))
    ap.add_argument('--no-images', action='store_true')
    ap.add_argument('--images', type=Path, default=None,
                    help='影像根目錄（其下 validation/<段>/<幀>.jpg）；預設＝--openlane')
    ap.add_argument('--limit-segments', type=int, default=None)
    ap.add_argument('--segments', type=Path, default=None,
                    help='一行一個 validation 片段名：改成轉出這些片段的**全部**幀（不看 up&down 標籤），'
                         '例如保留驗證集（select_holdout.py）')
    a = ap.parse_args(argv)

    cases = load_cases(a.openlane)
    updown = cases.pop('updown')
    by_seg = {}
    if a.segments is None:
        for seg, stem in updown:
            by_seg.setdefault(seg, set()).add(stem)
        subset = 'official up&down tag (test/1000_updown.txt), unfiltered'
        print(f'up&down: {len(updown)} frames in {len(by_seg)} segments')
    else:
        for seg in a.segments.read_text(encoding='utf-8').split():
            by_seg[seg] = {p.stem for p in (a.openlane / 'validation' / seg).glob('*.json')
                           if not p.name.startswith('._')}
        subset = f'all frames of the segments in {a.segments.name}, unfiltered'
        print(f'{a.segments.name}: {sum(map(len, by_seg.values()))} frames in {len(by_seg)} segments')
    segs = sorted(by_seg)[:a.limit_segments]

    all_frames, index = [], []
    for i, seg in enumerate(segs):
        seg_dir = a.openlane / 'validation' / seg
        df = export_segment(seg_dir, a.out / seg, a.images or a.openlane, by_seg[seg], cases,
                            not a.no_images, subset)
        df.insert(0, 'segment', seg)
        all_frames.append(df)
        index.append(dict(
            segment=seg, frames=len(df), with_image=int(df.has_image.sum()),
            with_profile=int((df.profile_source != 'none').sum()),
            both_lines=int((df.profile_source == 'both').sum()),
            converter_would_keep=int((df.drop_reasons == '').sum()),
            route_m=round(float(df.collect_dist_m.max() - df.collect_dist_m.min()), 1),
            grade_60m_median=round(float(df.grade_60m.median()), 3),
            grade_60m_max=round(float(df.grade_60m.max()), 3),
            **{f'tag_{t}': int(df[f'tag_{t}'].sum()) for t in cases}))
        print(f'[{i + 1}/{len(segs)}] {seg[:40]}  {len(df)} frames, '
              f'images {index[-1]["with_image"]}, profile {index[-1]["with_profile"]}', flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index).to_csv(a.out / 'index.csv', index=False)
    pd.concat(all_frames).to_csv(a.out / 'frames.csv', index=False)
    print(f'-> {a.out}/index.csv, frames.csv')


if __name__ == '__main__':
    main()
