"""從 export_updown 的輸出**逐幀**挑「兩側都有漆線、沒有轉彎、白天、非路口」的幀，組成新資料集。

與 ``select_updown.py`` 的差別：那支整段取捨（段內仍有單邊/路緣/彎道幀），
這支逐幀篩，只留下條件全部成立的幀：

    兩側都有線   12 m 處左右內側線都是漆線，寬度在 [WIDTH_MIN, WIDTH_MAX]，
                 GT 剖面深度跨度 ≥ 5 m（frames.csv: profile_source == both
                 且 drop_reasons 為空；與 convert_openlane 的收錄條件相同）
    沒有轉彎     50 m 內車道中心線 sagitta ≤ SAGITTA_MAX（沿用 Tier A 的實測門檻，
                 不用官方 curve tag —— tag 只有 0/1 無法調鬆緊；兩者的交叉表會印出來）
    不是夜間、不是路口   官方 case tag（tag_night / tag_intersection）；``--keep-night`` /
                 ``--keep-intersection`` 可放回

坡度**不另外篩**：來源本身就是官方 up&down 標籤，`grade_60m` 留在欄位裡當分層標籤。

輸出沿用本專案格式（一個片段一個目錄）：影像以 NTFS 硬連結指向來源、不複製；
frame_id 重新編號，``collect_dist_m`` 保留原里程，所以幀間的缺口在里程上看得出來。
⚠ 逐幀篩選會讓一個目錄不再是連續序列（跳過的幀＝缺口），NearfieldWidthCalibrator
的 hold 會跨缺口沿用 —— 與 Tier A 相同的性質。``selection.csv`` 記每段保留幀數與連續 run 數。

    python -m openlane_module.select_updown_straight
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from openlane_module.convert_openlane import inner_pair  # noqa: E402  （會先呼叫 setup_env）
from openlane_module.frame_tags import curvature, TIER_A_SAGITTA_MAX  # noqa: E402

SAGITTA_MAX = TIER_A_SAGITTA_MAX
MIN_FRAMES = 10     # 一段留下不到這麼多幀就整段不要（撐不起近場量寬的採用 run）


def sagitta_of(json_path):
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    L, R = inner_pair(data['lane_lines'])
    if not (L and R):
        return np.nan
    xl, xr = L[2], R[2]
    return curvature(xl, xr, max(xl[0].min(), xr[0].min()))[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--openlane', type=Path, default=Path('D:/datasets/openlane'))
    ap.add_argument('--src', type=Path, default=Path('D:/datasets/openlane_updown'))
    ap.add_argument('--out', type=Path, default=Path('D:/datasets/openlane_updown_straight'))
    ap.add_argument('--sagitta-max', type=float, default=SAGITTA_MAX)
    ap.add_argument('--min-frames', type=int, default=MIN_FRAMES)
    ap.add_argument('--keep-night', action='store_true')
    ap.add_argument('--keep-intersection', action='store_true')
    a = ap.parse_args(argv)

    f = pd.read_csv(a.src / 'frames.csv', dtype={'source_frame': str})
    f['two_side'] = (f.profile_source == 'both') & f.drop_reasons.isna()
    f['sagitta'] = np.nan
    for i in f.index[f.two_side]:
        f.at[i, 'sagitta'] = sagitta_of(
            a.openlane / 'validation' / f.at[i, 'segment'] / (f.at[i, 'source_frame'] + '.json'))
    f['straight'] = f.sagitta <= a.sagitta_max
    f['day'] = (f.tag_night == 0) | a.keep_night
    f['no_junction'] = (f.tag_intersection == 0) | a.keep_intersection
    f['keep'] = f.two_side & f.straight & f.day & f.no_junction & (f.has_image == 1)

    n1 = f.two_side & f.straight
    n2 = n1 & f.day
    print(f'全部 {len(f)} 幀 -> 兩側漆線 {f.two_side.sum()} -> 且沒轉彎 {n1.sum()}'
          f' -> 且非夜間 {n2.sum()} -> 且非路口 {(n2 & f.no_junction).sum()}')
    t = f[f.two_side]
    print('兩側漆線幀：sagitta 門檻 vs 官方 curve tag')
    print(pd.crosstab(t.straight.map({True: 'sag<=%.2f' % a.sagitta_max, False: 'sag>'}),
                      t.tag_curve.map({0: 'tag_curve=0', 1: 'tag_curve=1'})))

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg, g in f.groupby('segment', sort=True):
        k = g[g.keep]
        runs = int((np.diff(k.frame_id.to_numpy()) != 1).sum() + 1) if len(k) else 0
        row = dict(segment=seg, frames=len(g), two_side=int(g.two_side.sum()),
                   kept=len(k), runs=runs,
                   grade_60m_median=round(float(k.grade_60m.median()), 3) if len(k) else np.nan,
                   grade_60m_max=round(float(k.grade_60m.max()), 3) if len(k) else np.nan,
                   written=len(k) >= a.min_frames)
        rows.append(row)
        if (a.out / seg).exists():      # 重跑時清掉舊結果（影像是硬連結，刪的只是連結）
            shutil.rmtree(a.out / seg)
        if not row['written']:
            continue
        src, dst = a.src / seg, a.out / seg
        (dst / 'images').mkdir(parents=True, exist_ok=True)
        old = k.frame_id.to_numpy()
        new_id = dict(zip(old, range(len(old))))
        for o, n in new_id.items():
            os.link(src / 'images' / ('%06d.png' % o), dst / 'images' / ('%06d.png' % n))
        m = pd.read_csv(src / 'measurements.csv', dtype={'source_frame': str})
        m = m[m.frame_id.isin(old)].copy()
        m['frame_id'] = m.frame_id.map(new_id)
        m.to_csv(dst / 'measurements.csv', index=False)
        p = pd.read_csv(src / 'road_profile.csv')
        p = p[p.frame_id.isin(old)].copy()
        p['frame_id'] = p.frame_id.map(new_id)
        p.to_csv(dst / 'road_profile.csv', index=False)
        meta = json.loads((src / 'metadata.json').read_text())
        meta.update(frames=len(k), subset='official up&down tag, per-frame: both inner lines '
                    f'paint + sagitta <= {a.sagitta_max} m'
                    + ('' if a.keep_night else ' + not night')
                    + ('' if a.keep_intersection else ' + not intersection')
                    + ' (select_updown_straight)',
                    note='virtual intrinsics; camera pose written as identity; frame_id renumbered, '
                         'collect_dist_m kept -> gaps between kept frames are visible')
        (dst / 'metadata.json').write_text(json.dumps(meta, indent=2))

    s = pd.DataFrame(rows).sort_values(['written', 'kept'], ascending=False)
    s.to_csv(a.out / 'selection.csv', index=False)
    f.to_csv(a.out / 'frames.csv', index=False)
    w = s[s.written]
    print(f'written {len(w)}/{len(s)} segments, {int(w.kept.sum())} frames, '
          f'{int(w.runs.sum())} continuous runs -> {a.out}')


if __name__ == '__main__':
    main()
