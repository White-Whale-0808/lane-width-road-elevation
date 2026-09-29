"""用 LATR 公開權重（OpenLane-1000 full，md5 d8ecb900…）在 OpenLane 官方上下坡子集推論。

前處理直接用 LATR 自己的 LaneDataset（裁切／縮放 720×960／正規化／lidar2img 全照原版），
模型用同目錄 latr_stubs.py 的替身跑，不需要編譯任何運算子（環境與資料準備見 README.md）。

輸出 <out-dir>/pred_<tag>.jsonl（預設 debug/outputs/zerror/pred_latr.jsonl），一幀一行：
（評分用 python -m openlane_module.zerror_eval --tag latr --scope both|all-lanes）
  segment, source_frame, frame_id（對到 D:/datasets/openlane_updown 的編號，跟我們的預測對得上）,
  coords='ground'（LATR 直接輸出官方地面座標，不用再轉）,
  lanes＝通過 LATR 自己評估門檻（最大類別機率 > pos_threshold 0.3）的每條線 [[x, y, z], ...],
  probs＝各線的最大類別機率

    uv run --no-sync --with D:/models/latr/wheels/mmcv-1.7.2-py2.py3-none-any.whl \\
        --with mmdet==2.28.2 --with fvcore --with addict --with yapf==0.40.1 \\
        python baselines/latr/latr_infer.py [--limit N]
"""
import os, sys, json, pathlib, argparse, time
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
LATR_ROOT = pathlib.Path(os.environ.get('LATR_ROOT', 'D:/models/latr/LATR'))
sys.path.insert(0, str(HERE))
import latr_stubs
latr_stubs.install()
sys.path.insert(0, str(LATR_ROOT))          # LATR 的 utils / data / models（與本專案 utils 同名，要排前面）

import numpy as np, pandas as pd, torch
from mmcv.utils import Config
from data.Load_Data import LaneDataset
from models.latr import LATR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(LATR_ROOT / 'config/release_iccv/latr_1000_baseline.py'))
    ap.add_argument('--ckpt', default=str(LATR_ROOT / 'pretrained_models/openlane.pth'))
    ap.add_argument('--labels', default='D:/datasets/openlane/test/up_down_case/')
    ap.add_argument('--images', default='D:/datasets/openlane/images/')
    ap.add_argument('--ours', default='D:/datasets/openlane_updown', help='拿 source_frame→frame_id 對照')
    ap.add_argument('--tag', default='latr')
    ap.add_argument('--out-dir', type=pathlib.Path, default=REPO / 'debug/outputs/zerror')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    cwd = os.getcwd()
    os.chdir(LATR_ROOT)                      # config 裡有相對路徑
    args = Config.fromfile(a.config)
    os.chdir(cwd)
    args.merge_from_dict(dict(distributed=False, local_rank=0, gpu=0, world_size=1, no_cuda=False,
                              evaluate=True, batch_size=1, nworkers=0))
    ds = LaneDataset(a.images, a.labels, args)
    order = sorted(range(len(ds._label_list)), key=lambda i: ds._label_list[i])
    if a.limit:
        order = order[:a.limit]

    model = LATR(args).cuda().eval()
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'權重：缺 {len(missing)}、多 {len(unexpected)}', missing[:5], unexpected[:5], flush=True)
    assert not missing and not unexpected, '替身的參數名稱跟權重對不上'

    fid = {}
    for seg in pathlib.Path(a.ours).iterdir():
        if (seg / 'measurements.csv').exists():
            m = pd.read_csv(seg / 'measurements.csv', usecols=['frame_id', 'source_frame'])
            fid.update({(seg.name, str(s)): int(f) for f, s in zip(m.frame_id, m.source_frame)})

    ny = args.num_y_steps
    out = a.out_dir / f'pred_{a.tag}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, 'w', encoding='utf-8') as fo, torch.no_grad():
        for n, i in enumerate(order, 1):
            d = ds[i]
            jf = pathlib.Path(d.pop('idx_json_file'))
            ex = {k: (torch.as_tensor(v)[None].cuda() if not isinstance(v, str) else v) for k, v in d.items()}
            o = model(image=ex['image'].float(), extra_dict=ex, is_training=False)
            lp, cs = o['all_line_preds'][-1][0].cpu().numpy(), o['all_cls_scores'][-1][0]
            cls = torch.argmax(cs, -1).cpu().numpy()
            prob = torch.softmax(cs[cls > 0], -1).cpu().numpy()
            lanes, probs = [], []
            for lane, p in zip(lp[cls > 0], prob):
                vis = lane[2 * ny:3 * ny] > 0
                if vis.sum() < 2 or p.max() <= args.pos_threshold:
                    continue
                pts = np.stack([lane[:ny][vis], args.anchor_y_steps[vis], lane[ny:2 * ny][vis]], 1)
                lanes.append(np.round(pts, 4).tolist()); probs.append(float(p.max()))
            seg, sf = jf.parent.name, jf.stem
            fo.write(json.dumps(dict(segment=seg, source_frame=sf, frame_id=fid.get((seg, sf), -1),
                                     coords='ground', lanes=lanes, probs=probs, status='latr')) + '\n')
            if n % 200 == 0 or n == len(order):
                print(f'{n}/{len(order)} {time.time() - t0:.0f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB',
                      flush=True)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
