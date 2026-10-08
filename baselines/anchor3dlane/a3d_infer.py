"""用 Anchor3DLane++（Huang et al., 2024；ResNet-50、720×960，OpenLane v1.2 權重）在 OpenLane 官方上下坡子集推論。

不需要編譯任何運算子：ResNet 的 DCNv2 與 MSDA neck 的可變形注意力都換成純 PyTorch 等價實作
（替身沿用 baselines/latr/latr_stubs.py 的 DCNv2 與 mmcv 純 PyTorch 版可變形注意力；
原版 Deformable-DETR 的 MultiScaleDeformableAttention 擴充給一個同介面的假模組）。

前處理照原版 test_pipeline：cv2 讀圖（BGR→RGB）、不保持比例縮到 960×720（雙線性）、ImageNet 正規化、
全 False 遮罩；投影矩陣照原版 tools/convert_datasets/openlane.py 把外參轉到地面座標（R_vg、R_gc、
x/y 平移歸零）再 K·E⁻¹。後處理照 mmseg/apis/test_openlane.py：softmax，分數＝1−背景機率，
模型內 NMS（門檻 0.1、conf 0.2、refine_vis）；輸出只留分數 > test_conf 0.5 的線（原版評分的 prob_th）。

輸出 <out-dir>/pred_<tag>.jsonl（格式同 baselines/latr：coords='ground'）。

    uv run --no-sync --with D:/models/latr/wheels/mmcv-1.7.2-py2.py3-none-any.whl \\
        --with mmdet==2.28.2 --with addict --with yapf==0.40.1 --with prettytable --with terminaltables \\
        python baselines/anchor3dlane/a3d_infer.py [--limit N]
"""
import os, sys, json, glob, pathlib, argparse, time, types
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
A3D_ROOT = pathlib.Path(os.environ.get('A3D_ROOT', 'D:/models/anchor3dlane/Anchor3DLane_pp'))
import importlib.util
_spec = importlib.util.spec_from_file_location('mono3d_env_setup', REPO / 'utils' / 'env_setup.py')
_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env)
_env.setup_env()

import numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, str(REPO / 'baselines' / 'latr'))
import latr_stubs


def install_stubs():
    """latr_stubs 的 mmcv.ops／DCNv2／NumPy 別名，但不碰 mmseg（這裡要用 Anchor3DLane 自帶的 mmseg）。"""
    for k, v in dict(float=float, int=int, bool=bool, object=object, long=int).items():
        if k not in np.__dict__:
            setattr(np, k, v)
    ops = latr_stubs._Loose('mmcv.ops')
    ops.__path__ = []
    ops.MultiScaleDeformableAttnFunction = latr_stubs.MultiScaleDeformableAttnFunction
    ops.ModulatedDeformConv2dPack = latr_stubs.ModulatedDeformConv2dPack
    sys.modules['mmcv.ops'] = ops
    sys.meta_path.insert(0, latr_stubs._LooseFinder())
    sys.modules.setdefault('geffnet', types.ModuleType('geffnet'))

    # 推論用不到、但 mmseg 一 import 就要的模組（光達 spconv、Linux 才有的 resource、訓練／評估工具）
    import importlib.abc, importlib.machinery, importlib.util as iu
    absent = [n for n in ('spconv', 'jarvis', 'resource', 'cityscapesscripts', 'ortools') if iu.find_spec(n) is None]

    class _AbsentFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname.split('.')[0] in absent:
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            m = latr_stubs._Loose(spec.name)
            m.__path__ = []
            return m

        def exec_module(self, module):
            pass
    sys.meta_path.insert(0, _AbsentFinder())

    # Deformable-DETR 的編譯擴充 → 同介面的純 PyTorch 版
    msda = types.ModuleType('MultiScaleDeformableAttention')

    def ms_deform_attn_forward(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return latr_stubs.multi_scale_deformable_attn_pytorch(
            value, [(int(h), int(w)) for h, w in shapes], sampling_locations, attention_weights)
    msda.ms_deform_attn_forward = ms_deform_attn_forward
    sys.modules['MultiScaleDeformableAttention'] = msda

    import mmcv
    mmcv.ops = ops
    mmcv.__version__ = '1.7.1'          # mmseg 0.26 要求 mmcv < 1.7.2；1.7.2 只是版本檢查擋住
    from mmcv.cnn import CONV_LAYERS
    CONV_LAYERS.register_module('DCNv2', module=latr_stubs.ModulatedDeformConv2dPack, force=True)


R_VG = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
R_GC = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


def project_matrix(ext, K):
    E = np.array(ext, dtype=np.float64)
    E[:3, :3] = np.linalg.inv(R_VG) @ E[:3, :3] @ R_VG @ R_GC
    E[0:2, 3] = 0.0
    return (np.array(K) @ np.linalg.inv(E)[0:3, :]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(A3D_ROOT / 'configs_v2/openlane/anchor3dlane++_r50x2.py'))
    ap.add_argument('--ckpt', default='D:/models/anchor3dlane/ckpt/openlane_anchor3dlane++_r50x2.pth')
    ap.add_argument('--labels', default='D:/datasets/openlane/test/up_down_case/')
    ap.add_argument('--images', default='D:/datasets/openlane/images/')
    ap.add_argument('--ours', default='D:/datasets/openlane_updown', help='拿 source_frame→frame_id 對照')
    ap.add_argument('--tag', default='a3dpp')
    ap.add_argument('--out-dir', type=pathlib.Path, default=REPO / 'debug/outputs/zerror')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    install_stubs()
    sys.path.insert(0, str(A3D_ROOT))
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmseg.models import build_lanedetector

    cfg = Config.fromfile(a.config)
    cfg.model.pretrained = None
    model = build_lanedetector(cfg.model)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'權重：缺 {len(missing)}、多 {len(unexpected)}', missing[:5], unexpected[:5], flush=True)
    assert not missing and not unexpected, '替身的參數名稱跟權重對不上'
    model = model.cuda().eval()

    H, W = cfg.input_size
    mean = np.array(cfg.img_norm_cfg['mean'], np.float32)
    std = np.array(cfg.img_norm_cfg['std'], np.float32)
    test_conf = cfg.model.test_cfg['test_conf']
    ys = np.array(cfg.anchor_y_steps, np.float32)
    L = len(ys)

    fid = {}
    for seg in pathlib.Path(a.ours).iterdir():
        if (seg / 'measurements.csv').exists():
            m = pd.read_csv(seg / 'measurements.csv', usecols=['frame_id', 'source_frame'])
            fid.update({(seg.name, str(s)): int(f) for f, s in zip(m.frame_id, m.source_frame)})

    files = sorted(glob.glob(os.path.join(a.labels, '*', '*.json')))
    if a.limit:
        files = files[:a.limit]
    out = a.out_dir / f'pred_{a.tag}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    mask = torch.zeros(1, 1, H, W, dtype=torch.bool).cuda()
    t0 = time.time()
    with open(out, 'w', encoding='utf-8') as fo, torch.no_grad():
        for n, jf in enumerate(files, 1):
            d = json.load(open(jf, encoding='utf-8'))
            img = cv2.imread(os.path.join(a.images, d['file_path']))
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) - mean) / std
            x = torch.from_numpy(img.transpose(2, 0, 1))[None].cuda()
            P = torch.from_numpy(project_matrix(d['extrinsic'], d['intrinsic']))[None, None].cuda()
            o = model(img=x, img_metas=[{}], mask=mask, return_loss=False, gt_project_matrix=P)
            props = o['proposals_list'][0][0]
            lanes, probs = [], []
            if props.shape[0]:
                logit = props[:, 5 + 3 * L:]
                score = (1 - torch.softmax(logit, 1)[:, 0]).cpu().numpy()
                props = props.cpu().numpy()
                for p, s in zip(props, score):
                    vis = p[5 + 2 * L:5 + 3 * L] > 0
                    if vis.sum() < 2 or s <= test_conf:
                        continue
                    pts = np.stack([p[5:5 + L][vis], ys[vis], p[5 + L:5 + 2 * L][vis]], 1)
                    lanes.append(np.round(pts, 4).tolist()); probs.append(float(s))
            p = pathlib.Path(jf)
            seg, sf = p.parent.name, p.stem
            fo.write(json.dumps(dict(segment=seg, source_frame=sf, frame_id=fid.get((seg, sf), -1),
                                     coords='ground', lanes=lanes, probs=probs, status='a3dpp')) + '\n')
            if n % 200 == 0 or n == len(files):
                print(f'{n}/{len(files)} {time.time() - t0:.0f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB',
                      flush=True)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
