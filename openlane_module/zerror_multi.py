"""多個方法的 Z-error 共同點比較（zerror_paired.py 的多方版）。

「共同點」＝同一幀、同一條 GT 自車道線（attr）、同一個取樣距離 y，**所有**方法都有點。
只在這些點上比，任一方都不能因為只在容易的地方輸出而佔便宜；方法越多，共同點越少，
所以另外印各自範圍（不配對）的數字與每個方法的覆蓋。

印：
  1. 各自範圍：官方式 Z-error 近／遠段、有點的幀數
  2. 共同點：官方式近／遠段，與依距離分段的逐點平均

    python -m openlane_module.zerror_multi --tags ep10_w40_c03_t04b_solid,latr,a3dpp,persformer,sclane --scope both
"""
import argparse
import pathlib

import pandas as pd

from openlane_module.zerror_paired import K, official


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', required=True, help='逗號分隔；第一個當主角')
    ap.add_argument('--names', default='', help='逗號分隔的顯示名稱（省略＝tag）')
    ap.add_argument('--scope', default='both')
    ap.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path('debug/outputs/zerror'))
    a = ap.parse_args()
    tags = a.tags.split(',')
    names = a.names.split(',') if a.names else tags
    suf = '' if a.scope == 'all' else '_' + a.scope

    P = {n: pd.read_csv(a.out_dir / f'points_{t}{suf}.csv')[K + ['far', 'z_err']] for t, n in zip(tags, names)}
    w = max(len(n) for n in names)

    print(f'== 各自範圍（{a.scope}）')
    for n, p in P.items():
        c, f, nc, nf = official(p)
        print(f'{n:>{w}s}  近段 {c * 100:5.2f} cm（{nc} 條線）  遠段 {f * 100:5.2f} cm（{nf} 條線）  '
              f'有點的幀 {p.groupby(["segment", "frame_id"]).ngroups}')

    M = None
    for n, p in P.items():
        q = p.rename(columns={'z_err': n})
        M = q if M is None else M.merge(q, on=K + ['far'])
    print(f'\n== 共同點：{len(M)} 點（{M.groupby(["segment", "frame_id"]).ngroups} 幀）')
    rows = []
    for n in names:
        c, f, _, _ = official(M, n)
        rows.append(dict(方法=n, 近段_cm=c * 100, 遠段_cm=f * 100))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    M['yb'] = pd.cut(M.y, [3, 10, 20, 30, 40.5, 60, 103], right=False)
    t = M.groupby('yb', observed=True)[names].mean() * 100
    t.insert(0, '點數', M.groupby('yb', observed=True).size())
    print('\n依距離（m），逐點平均 cm：')
    print(t.round(2).to_string())


if __name__ == '__main__':
    main()
