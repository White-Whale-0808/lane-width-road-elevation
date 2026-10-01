"""自車道 Z-error：用 OpenLane 官方評估程式（official_eval/）評 zerror_predict.py 或 LATR 的預測，另存逐點誤差（WWH-26）。

讀 <out-dir>/pred_<tag>.jsonl，對每幀：
  1. 讀原始 GT JSON，只留自車道兩條線（attribute 2＝左、3＝右，官方定義；路緣不帶 attribute）
  2. 外參照官方 bench_one_submit 改成地面座標（x 右、y 前、z 上，原點在相機正下方地面，
     x/y 平移歸零）；GT 與預測套**同一個**轉換。預測檔每行若帶 coords='ground'
     （LATR 直接輸出官方地面座標）就不再轉
  3. 丟進官方 LaneEval.bench

官方 Z-error 的規則（eval_3D_lane.py）：取樣 y＝3..102 m 每 1 m；近段 y ≤ 40（38 點）、遠段 41–102；
配對總成本 < 1.5×100 的線（**不必是 TP**）才算，先在每條線上對「兩邊都可見」的點取平均，
再對所有線平均。所以只輸出到 45 m 的線也會算進 Z-error，只是沒輸出的遠處不算。

另外印覆蓋，避免「只在容易的地方算誤差」：有輸出的幀比例、近段算進 Z-error 的 GT 線比例、
近段／遠段 GT 可見取樣點中也有預測點的比例。逐點誤差（之後跟別的方法在同一批點上配對比，
zerror_paired.py）存 points_<tag>[_scope].csv，逐幀存 frames_<tag>[_scope].csv；
另外自己重做一次逐點配對、彙總後跟官方數字對帳（應逐位相同）。

--scope
  all        有官方自車道線的幀（預設）
  both       只評 GT 左右兩條自車道線都有可見點的幀（比較範圍 B，使用者 2026-09-27 定：
             我們的方法要兩側都有線才算得出深度）
  all-lanes  官方原規則：所有 GT 線（含路緣）、所有幀，印官方 F-score。只用來驗證別人的模型
             （例如 LATR 移植後）在這個子集的數字跟論文對得上；只輸出兩條線的方法在這裡不公平

    python -m openlane_module.zerror_eval --tag ep10 --scope both
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from openlane_module.official_eval.eval_3D_lane import LaneEval
from openlane_module.official_eval.utils.utils import (prune_3d_lane_by_range, prune_3d_lane_by_visibility,
                                                       resample_laneline_in_y)
from openlane_module.official_eval.utils.MinCostFlow import SolveMinCostFlow

EGO = (2, 3)
# 官方 bench_one_submit：GT 的 xyz（相機座標：前、左、上）→ 光學座標（右、下、前）
CAM_REP = np.linalg.inv(np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float))


def make_evaluator():
    ev = LaneEval(argparse.Namespace(dataset_dir=None, pred_dir=None, test_list=None))
    ev.dataset_name = 'openlane'          # 官方 bench 會讀、__init__ 卻沒設
    return ev


def ground_extrinsic(E):
    """官方 bench_one_submit 的外參改寫（逐字照抄）。"""
    E = np.array(E, float)
    R_vg = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    R_gc = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    E[:3, :3] = np.linalg.inv(R_vg) @ E[:3, :3] @ R_vg @ R_gc
    E[0:2, 3] = 0.0
    return E


def gt_to_ground(E, xyz):
    """GT 的 lane_lines[i]['xyz']（3×N，相機座標前、左、上）→ 地面座標 N×3（官方寫法）。"""
    xyz = np.asarray(xyz, float)
    return (E @ (CAM_REP @ np.vstack([xyz, np.ones((1, xyz.shape[1]))])))[:3].T


def to_ground(E, pts_optical):
    """預測點（N×3，光學座標右、下、前）→ 地面座標，依 y（前方距離）遞增。"""
    p = np.asarray(pts_optical, float).T
    g = (E @ np.vstack([p, np.ones((1, p.shape[1]))]))[:3].T
    return g[np.argsort(g[:, 1])]


def point_errors(ev, pred_lanes, gt_lanes, gt_vis, gt_attr):
    """重做 bench 的重取樣與配對，回傳 (逐點列, 逐 GT 線列)；與官方同規則，用來跟官方彙總數字對帳。"""
    ys = ev.y_samples
    close = np.where(ys > ev.close_range)[0][0]
    G = []
    for lane, vis, at in zip(gt_lanes, gt_vis, gt_attr):
        lane = prune_3d_lane_by_visibility(np.array(lane), np.array(vis))
        if lane.shape[0] < 2 or not (lane[0, 1] < ys[-1] and lane[-1, 1] > ys[0]):
            continue
        lane = prune_3d_lane_by_range(lane, ev.x_min, ev.x_max)
        if lane.shape[0] < 2:
            continue
        x, z, v = resample_laneline_in_y(lane, ys, out_vis=True)
        m = (x >= ev.x_min) & (x <= ev.x_max) & (ys >= lane[:, 1].min()) & (ys <= lane[:, 1].max()) & (v > 0.5)
        if m.sum() > 1:
            G.append((x, z, m, at))
    P = []
    for lane in pred_lanes:
        if not (lane[0, 1] < ys[-1] and lane[-1, 1] > ys[0]):
            continue
        lane = prune_3d_lane_by_range(lane, ev.x_min, ev.x_max)
        if lane.shape[0] < 2:
            continue
        x, z, v = resample_laneline_in_y(lane, ys, out_vis=True)
        m = (x >= ev.x_min) & (x <= ev.x_max) & (ys >= lane[:, 1].min()) & (ys <= lane[:, 1].max()) & (v > 0.5)
        if m.sum() > 1:
            P.append((x, z, m))
    rows, gt_rows = [], []
    cost = np.full((len(G), len(P)), 1000, int)
    for i, (gx, gz, gm, _) in enumerate(G):
        for j, (px, pz, pm) in enumerate(P):
            both, none = gm & pm, ~gm & ~pm
            d = np.sqrt((gx - px) ** 2 + (gz - pz) ** 2)
            d[none], d[~(both | none)] = 0, ev.dist_th
            c = d.sum()
            cost[i, j] = 1 if 0 < c < 1 else int(c)
    matched = {}
    for i, j, c in SolveMinCostFlow(np.ones_like(cost), cost) if len(G) and len(P) else []:
        if c < ev.dist_th * len(ys):
            matched[i] = j
    for i, (gx, gz, gm, at) in enumerate(G):
        j = matched.get(i)
        both = (gm & P[j][2]) if j is not None else np.zeros_like(gm)
        gt_rows.append(dict(attr=at, n_close=int(gm[:close].sum()), n_far=int(gm[close:].sum()),
                            matched=j is not None,
                            # counted in the official near Z-error: matched AND >= 1 both-visible near point
                            near_counted=bool(both[:close].any())))
        if j is None:
            continue
        px, pz, pm = P[j]
        for k in np.where(both)[0]:
            rows.append(dict(attr=at, y=float(ys[k]), far=bool(k >= close),
                             x_err=float(abs(gx[k] - px[k])), z_err=float(abs(gz[k] - pz[k])),
                             z_gt=float(gz[k]), z_pred=float(pz[k])))
    return rows, gt_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--openlane', type=pathlib.Path, default=pathlib.Path('D:/datasets/openlane/validation'))
    ap.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path('debug/outputs/zerror'))
    ap.add_argument('--scope', choices=['all', 'both', 'all-lanes'], default='all')
    a = ap.parse_args()
    suffix = '' if a.scope == 'all' else '_' + a.scope
    ev = make_evaluator()

    recs = [json.loads(l) for l in open(a.out_dir / f'pred_{a.tag}.jsonl', encoding='utf-8')]
    zc, zf, pts, gts, frames, stats = [], [], [], [], [], []
    for r in recs:
        gt = json.loads((a.openlane / r['segment'] / f"{r['source_frame']}.json").read_text(encoding='utf-8'))
        E = ground_extrinsic(gt['extrinsic'])
        g_l, g_v, g_c, g_a = [], [], [], []
        for L in gt['lane_lines']:
            if a.scope != 'all-lanes' and L.get('attribute') not in EGO:
                continue
            g_l.append(gt_to_ground(E, L['xyz'])); g_v.append(np.asarray(L['visibility']))
            g_c.append(L['category']); g_a.append(L.get('attribute', 0))
        if not g_l and a.scope != 'all-lanes':
            continue                          # 這幀官方沒有定義自車道線，不評
        if a.scope == 'both' and {at for at, v in zip(g_a, g_v) if (v > 0.5).any()} != set(EGO):
            continue
        if r.get('coords') == 'ground':
            pred = [np.asarray(l, float)[np.argsort(np.asarray(l, float)[:, 1])] for l in r['lanes'] if len(l) >= 2]
        else:
            pred = [to_ground(E, l) for l in r['lanes'] if len(l) >= 2]
        out = ev.bench(pred, [0] * len(pred), g_l, g_v, g_c, None, E[2, 3], 0, False, None)
        stats.append(out[:6])
        c, f = np.array(out[8]), np.array(out[9])
        zc += list(c[c > -1 + 1e-6]); zf += list(f[f > -1 + 1e-6])
        rows, grow = point_errors(ev, pred, g_l, g_v, g_a)
        for x in rows + grow:
            x.update(segment=r['segment'], frame_id=r['frame_id'])
        pts += rows; gts += grow
        frames.append(dict(segment=r['segment'], frame_id=r['frame_id'], out=bool(pred), status=r.get('status'),
                           z_close=np.mean(c[c > -1 + 1e-6]) if (c > -1 + 1e-6).any() else np.nan))

    if not frames:
        print(f'== {a.tag}（scope={a.scope}）：沒有可評估的幀（預測檔 {len(recs)} 行）')
        return
    S = np.array(stats, float)
    R_, P_ = S[:, 0].sum() / max(S[:, 3].sum(), 1), S[:, 1].sum() / max(S[:, 4].sum(), 1)
    P, G, F = pd.DataFrame(pts), pd.DataFrame(gts), pd.DataFrame(frames)
    if P.empty:
        P = pd.DataFrame(columns=['segment', 'frame_id', 'attr', 'y', 'far', 'x_err', 'z_err', 'z_gt', 'z_pred'])
        P['far'] = P['far'].astype(bool)
    if G.empty:
        G = pd.DataFrame(columns=['attr', 'n_close', 'n_far', 'matched', 'near_counted'])
    P.to_csv(a.out_dir / f'points_{a.tag}{suffix}.csv', index=False)
    F.to_csv(a.out_dir / f'frames_{a.tag}{suffix}.csv', index=False)

    # 對帳：自己重做的逐點 → 每條線平均 → 全部平均，要等於官方 bench 的彙總
    # （all-lanes 模式下非自車道線的 attr 都是 0，逐點重算會把同幀的它們併成一條，這裡只看自車道模式）
    if len(P):
        per_lane = P[~P.far].groupby(['segment', 'frame_id', 'attr']).z_err.mean()
        print(f'對帳 近段 Z-error：官方 {np.mean(zc):.5f}  逐點重算 {per_lane.mean():.5f}  '
              f'（線數 {len(zc)} / {len(per_lane)}）')
    n_close = G.n_close.sum()
    print(f'\n== {a.tag}（scope={a.scope}）：{F.segment.nunique()} 段、評估的幀 {len(F)}、有輸出 {F.out.mean():.1%}')
    print(f'官方 F-score {2 * R_ * P_ / max(R_ + P_, 1e-9):.4f}（recall {R_:.4f}、precision {P_:.4f}；'
          f'GT {int(S[:, 3].sum())} 條、預測 {int(S[:, 4].sum())} 條）')
    print(f'官方 Z-error 近段（3–40 m）{np.mean(zc):.4f} m（{len(zc)} 條線）   '
          f'遠段（41–102 m）{np.mean(zf) if zf else float("nan"):.4f} m（{len(zf)} 條線）')
    near_cov = G.near_counted.astype(float).mean() if len(G) else float('nan')
    print(f'覆蓋：GT 線 {len(G)} 條，近段算進 Z-error 的 {near_cov:.1%}；'
          f'近段可見取樣點 {n_close}，有預測點 {(~P.far).sum() / max(n_close, 1):.1%}；'
          f'遠段可見點 {G.n_far.sum()}，有預測點 {P.far.sum() / max(G.n_far.sum(), 1):.1%}')
    if len(P):
        P['yb'] = pd.cut(P.y, [3, 10, 20, 30, 40.5, 60, 103], right=False)
        print('\n逐點 |Δz|（m）依距離：')
        print(P.groupby('yb', observed=True).z_err.agg(['count', 'mean', 'median']).round(4).to_string())


if __name__ == '__main__':
    main()
