"""保留驗證集：從 OpenLane validation 抽「從沒拿來調過參數」的片段，檢查結論換一批資料還成不成立。

排除：之前任何實驗用過的片段——官方上下坡子集（Z-error 與 WWH-27/28 的參數都在它上面選）、
Tier A（新舊兩版）、虛線集、以及 --exclude 給的其他資料夾（每個資料夾的 segment-* 子目錄名）。
剩下的依固定種子隨機抽 N 段（不看內容挑，避免挑到對我們有利的路）。

``--min-rise R``（坡道保留集，WWH-32）：不隨機抽，改成收「段的起伏中位 ≥ R」的**全部**段。
起伏＝每幀 ``export_updown.frame_record`` 的 ``grade_60m``（真值漆線前方 60 m 內最高減最低），
段的起伏＝全部幀的中位數；只看真值、不看任何方法的結果。這個模式另外排除兩組隨機保留集
（``HOLDOUTS``：結果已看過，不再獨立）。清單旁另寫 ``<name>_rise.csv``（每段的起伏與幀數）。

輸出：
  <list>                         一行一段的清單（export_updown.py --segments 吃這個）
  <images>/validation/<段>/*.jpg 從 images_validation_{0..3}.tar 只解這些段（LATR 等讀原圖）
  <labels>/<段>/*.json           官方標註的複本（LATR／其他模型的推論腳本用 --labels 指到這裡）

    python -m openlane_module.select_holdout --n 30 --seed 0
    python -m openlane_module.select_holdout --min-rise 0.3 --name holdout_slope   # WWH-32
    python -m openlane_module.export_updown --segments D:/datasets/openlane/holdout30.txt \\
        --out D:/datasets/openlane_holdout30
"""
import argparse
import json
import random
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openlane_module.export_updown import frame_record  # noqa: E402

EXCLUDE = ['D:/datasets/openlane_updown', 'D:/datasets/openlane_tierA', 'D:/datasets/openlane_tierA_OLD',
           'D:/datasets/openlane_dashed', 'D:/datasets/openlane_updown_straight',
           'D:/datasets/openlane_updown_twoside', 'D:/datasets/openlane_converted']
# 已看過結果的隨機保留集（WWH-29）：--min-rise 模式一併排除
HOLDOUTS = ['D:/datasets/openlane_holdout30', 'D:/datasets/openlane_holdout30b']


def segment_rise(seg_dir):
    """(段的起伏中位 m, 幀數)：每幀 grade_60m 的中位數，沒有任何幀算得出來時是 NaN。"""
    g = [frame_record(json.loads(f.read_text(encoding='utf-8')))[0]['grade_60m']
         for f in sorted(seg_dir.glob('*.json')) if not f.name.startswith('._')]
    g = np.asarray(g, float)
    return (float(np.nanmedian(g)) if np.isfinite(g).any() else float('nan')), len(g)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--openlane', type=Path, default=Path('D:/datasets/openlane'))
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--exclude', nargs='*', default=EXCLUDE)
    ap.add_argument('--name', default=None, help='預設 holdout<n>（--min-rise 時 holdout_rise<R>）')
    ap.add_argument('--min-rise', type=float, default=None,
                    help='收段的起伏中位 ≥ 這個值（m）的全部段，不隨機抽；另外排除 HOLDOUTS')
    a = ap.parse_args(argv)
    name = a.name or (f'holdout_rise{a.min_rise:g}' if a.min_rise is not None else f'holdout{a.n}')

    used = set()
    for d in a.exclude + (HOLDOUTS if a.min_rise is not None else []):
        if Path(d).is_dir():
            used |= {p.name for p in Path(d).iterdir() if p.is_dir() and p.name.startswith('segment-')}
    pool = sorted(p.name for p in (a.openlane / 'validation').iterdir()
                  if p.is_dir() and p.name.startswith('segment-') and p.name not in used)
    n_val = len(pool) + len(used & {p.name for p in (a.openlane / 'validation').iterdir()})
    lst = a.openlane / f'{name}.txt'
    if a.min_rise is None:
        pick = sorted(random.Random(a.seed).sample(pool, a.n))
        how = f'抽 {len(pick)} 段（seed {a.seed}）'
    else:
        rise = {seg: segment_rise(a.openlane / 'validation' / seg) for seg in pool}
        pick = sorted(s for s, (r, _) in rise.items() if r >= a.min_rise)
        how = f'起伏中位 ≥ {a.min_rise:g} m 的 {len(pick)} 段'
        lst.with_name(f'{name}_rise.csv').write_text(
            'segment,rise_median_m,frames,picked\n' + ''.join(
                f'{s},{r:.4f},{n},{int(s in pick)}\n' for s, (r, n) in sorted(rise.items())), encoding='utf-8')
    print(f'validation {n_val} 段，排除用過的 {len(used)} 段，剩 {len(pool)} 段，{how}')
    lst.write_text('\n'.join(pick) + '\n', encoding='utf-8')

    labels = a.openlane / 'test' / name
    for seg in pick:
        dst = labels / seg
        dst.mkdir(parents=True, exist_ok=True)
        for f in (a.openlane / 'validation' / seg).glob('*.json'):
            if not f.name.startswith('._'):
                shutil.copy2(f, dst / f.name)

    want = set(pick)
    out = a.openlane / 'images' / 'validation'
    n_img = 0
    for tp in sorted(a.openlane.glob('images_validation_*.tar')):
        with tarfile.open(tp) as t:
            for m in t:
                parts = m.name.split('/')
                if len(parts) == 3 and parts[1] in want and m.isfile():
                    dst = out / parts[1] / parts[2]
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with t.extractfile(m) as src, open(dst, 'wb') as fo:
                            shutil.copyfileobj(src, fo)
                    n_img += 1
        print(f'{tp.name}: 累計 {n_img} 張', flush=True)
    print(f'-> {lst}\n-> {labels}\n-> {out}（{n_img} 張）')


if __name__ == '__main__':
    main()
