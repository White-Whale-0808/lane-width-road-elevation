"""保留驗證集：從 OpenLane validation 抽「從沒拿來調過參數」的片段，檢查結論換一批資料還成不成立。

排除：之前任何實驗用過的片段——官方上下坡子集（Z-error 與 WWH-27/28 的參數都在它上面選）、
Tier A（新舊兩版）、虛線集、以及 --exclude 給的其他資料夾（每個資料夾的 segment-* 子目錄名）。
剩下的依固定種子隨機抽 N 段（不看內容挑，避免挑到對我們有利的路）。

輸出：
  <list>                         一行一段的清單（export_updown.py --segments 吃這個）
  <images>/validation/<段>/*.jpg 從 images_validation_{0..3}.tar 只解這些段（LATR 等讀原圖）
  <labels>/<段>/*.json           官方標註的複本（LATR／其他模型的推論腳本用 --labels 指到這裡）

    python -m openlane_module.select_holdout --n 30 --seed 0
    python -m openlane_module.export_updown --segments D:/datasets/openlane/holdout30.txt \\
        --out D:/datasets/openlane_holdout30
"""
import argparse
import random
import shutil
import tarfile
from pathlib import Path

EXCLUDE = ['D:/datasets/openlane_updown', 'D:/datasets/openlane_tierA', 'D:/datasets/openlane_tierA_OLD',
           'D:/datasets/openlane_dashed', 'D:/datasets/openlane_updown_straight',
           'D:/datasets/openlane_updown_twoside', 'D:/datasets/openlane_converted']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--openlane', type=Path, default=Path('D:/datasets/openlane'))
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--exclude', nargs='*', default=EXCLUDE)
    ap.add_argument('--name', default=None, help='預設 holdout<n>')
    a = ap.parse_args(argv)
    name = a.name or f'holdout{a.n}'

    used = set()
    for d in a.exclude:
        if Path(d).is_dir():
            used |= {p.name for p in Path(d).iterdir() if p.is_dir() and p.name.startswith('segment-')}
    pool = sorted(p.name for p in (a.openlane / 'validation').iterdir()
                  if p.is_dir() and p.name.startswith('segment-') and p.name not in used)
    pick = sorted(random.Random(a.seed).sample(pool, a.n))
    print(f'validation {len(pool) + len(used & {p.name for p in (a.openlane / "validation").iterdir()})} 段，'
          f'排除用過的 {len(used)} 段，剩 {len(pool)} 段，抽 {len(pick)} 段（seed {a.seed}）')
    lst = a.openlane / f'{name}.txt'
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
