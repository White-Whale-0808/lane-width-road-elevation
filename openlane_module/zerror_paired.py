"""兩個方法的 Z-error 配對比較（zerror_eval.py 的逐點輸出 points_<tag>[_scope].csv；WWH-26）。

「配對」＝同一幀、同一條 GT 自車道線（attr）、同一個取樣距離 y，兩個方法**都**有點（都被官方
配對規則算進 Z-error）。只在這些點上比，才不會因為一方只在容易的地方有輸出而佔便宜。

印：
  1. 各自的官方式 Z-error（先線內平均、再跨線平均）與點數——各自範圍，不配對
  2. 配對點上的 Z-error：官方式平均、逐點平均／中位數，依距離分段
  3. 配對點上逐點誰比較準的比例

    python -m openlane_module.zerror_paired --a ep10 --b latr --scope both
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

K = ['segment', 'frame_id', 'attr', 'y']


def official(p, col='z_err'):
    """官方式：每條線（幀×attr）近／遠各自平均，再跨線平均。回傳 (近, 遠, 近線數, 遠線數)。"""
    g = p.groupby(['segment', 'frame_id', 'attr', 'far'])[col].mean().reset_index()
    return g[~g.far][col].mean(), g[g.far][col].mean(), (~g.far).sum(), g.far.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--scope', default='both')
    ap.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path('debug/outputs/zerror'))
    a = ap.parse_args()
    suf = '' if a.scope == 'all' else '_' + a.scope

    def load(tag):
        p = pd.read_csv(a.out_dir / f'points_{tag}{suf}.csv')
        return p[K + ['far', 'z_err', 'x_err']]

    A, B = load(a.a), load(a.b)
    print(f'== 各自範圍（{a.scope}）')
    for tag, p in ((a.a, A), (a.b, B)):
        c, f, nc, nf = official(p)
        print(f'{tag:>8s}  近段 {c:.4f} m（{nc} 條線、{(~p.far).sum()} 點）  遠段 {f:.4f} m（{nf} 條線、{p.far.sum()} 點）')

    M = A.merge(B, on=K + ['far'], suffixes=('_a', '_b'))
    print(f'\n== 配對點：{len(M)} 點（{M.groupby(["segment", "frame_id"]).ngroups} 幀）')
    for tag, col in ((a.a, 'z_err_a'), (a.b, 'z_err_b')):
        c, f, _, _ = official(M, col)
        print(f'{tag:>8s}  官方式 近段 {c:.4f} m  遠段 {f:.4f} m')
    M['yb'] = pd.cut(M.y, [3, 10, 20, 30, 40.5, 60, 103], right=False)
    t = M.groupby('yb', observed=True).agg(n=('y', 'size'),
                                            a_mean=('z_err_a', 'mean'), b_mean=('z_err_b', 'mean'),
                                            a_med=('z_err_a', 'median'), b_med=('z_err_b', 'median'))
    t['a_better'] = M.groupby('yb', observed=True)[['z_err_a', 'z_err_b']].apply(
        lambda g: (g.z_err_a < g.z_err_b).mean())
    t.columns = ['點數', f'{a.a} 平均', f'{a.b} 平均', f'{a.a} 中位', f'{a.b} 中位', f'{a.a} 較準的比例']
    print('\n依距離（m）：')
    print(t.round(4).to_string())


if __name__ == '__main__':
    main()
