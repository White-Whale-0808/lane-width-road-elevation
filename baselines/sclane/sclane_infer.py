"""用 SC-Lane（Park et al., ICCV 2025；作者釋出的 OpenLane 權重 ckpt.pth）在 OpenLane 官方上下坡子集推論。

照原版 tools/val.py 與 loader/bev_road/openlane_data.py（OpenLane_dataset_with_offset_val）：
  * 影像：cv2 讀（保持 BGR，原版就是這樣餵 albumentations）、雙線性縮到 800×600、
    A.Normalize（/255 後減 ImageNet 均值除標準差）—— 這裡直接寫出同樣的運算，不裝 albumentations
  * 內參依縮放比例調整；road2cam＝inv(地面外參 @ inv(相機軸換向))，地面外參同官方（R_vg、R_gc、x/y 平移歸零）
  * 單幀推論（forward 的 prev=False），不讀 heightmap（原版推論那行就註解掉了）
  * 後處理：embedding_post（conf −1.5、margin 6、最少 15 點）→ bev_instance2points_with_offset_z
    → (−橫, 縱, z) ＝ 官方地面座標
SC-Lane 沒有逐條信心分數（分群出來的都算），probs 一律記 1.0。

不需要編譯：Deformable-DETR 的 MultiScaleDeformableAttention 擴充換成 mmcv 純 PyTorch 版。
ResNet-50 的 ImageNet 預訓練不下載（整個模型由權重覆蓋）。

輸出 <out-dir>/pred_<tag>.jsonl（格式同 baselines/latr：coords='ground'）。

    uv run --no-sync python baselines/sclane/sclane_infer.py [--limit N]
"""
import os, sys, json, glob, pathlib, argparse, time, types
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
SC_ROOT = pathlib.Path(os.environ.get('SCLANE_ROOT', 'D:/models/sclane/SC-Lane'))
import importlib.util
_spec = importlib.util.spec_from_file_location('mono3d_env_setup', REPO / 'utils' / 'env_setup.py')
_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env)
_env.setup_env()

import numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, str(REPO / 'baselines' / 'latr'))
import latr_stubs

INPUT_H, INPUT_W = 600, 800
X_RANGE, Y_RANGE, MPP = (3, 103), (-12, 12), 0.5
OUTPUT_2D = (144, 256)
POST_CONF, POST_MARGIN, POST_MIN = -1.5, 6.0, 15
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CAM2CAMW = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
R_VG = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
R_GC = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def install_stubs():
    msda = types.ModuleType('MultiScaleDeformableAttention')

    def ms_deform_attn_forward(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return latr_stubs.multi_scale_deformable_attn_pytorch(
            value, [(int(h), int(w)) for h, w in shapes], sampling_locations, attention_weights)
    msda.ms_deform_attn_forward = ms_deform_attn_forward
    sys.modules['MultiScaleDeformableAttention'] = msda

    import torchvision as tv
    r50 = tv.models.resnet50
    tv.models.resnet50 = lambda *a, **k: r50(weights=None)


def road2cam(ext):
    E = np.array(ext, dtype=float)
    Ep = E.copy()
    Ep[:3, :3] = np.linalg.inv(R_VG) @ Ep[:3, :3] @ R_VG @ R_GC
    Ep[0:2, 3] = 0.0
    return np.linalg.inv(Ep @ np.linalg.inv(CAM2CAMW))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(SC_ROOT / 'ckpt.pth'))
    ap.add_argument('--labels', default='D:/datasets/openlane/test/up_down_case/')
    ap.add_argument('--images', default='D:/datasets/openlane/images/')
    ap.add_argument('--ours', default='D:/datasets/openlane_updown', help='拿 source_frame→frame_id 對照')
    ap.add_argument('--tag', default='sclane')
    ap.add_argument('--out-dir', type=pathlib.Path, default=REPO / 'debug/outputs/zerror')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    install_stubs()
    sys.path.insert(0, str(SC_ROOT))
    from models.model.sc_lane import SCLane
    from models.util.cluster import embedding_post
    from models.util.post_process import bev_instance2points_with_offset_z

    bev_shape = (int((X_RANGE[1] - X_RANGE[0]) / MPP), int((Y_RANGE[1] - Y_RANGE[0]) / MPP))
    model = SCLane(bev_shape=bev_shape, image_shape=(INPUT_H, INPUT_W), output_2d_shape=OUTPUT_2D,
                   train=False, use_img=False, x_range=X_RANGE, y_range=Y_RANGE, meter_per_pixel=MPP)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('model_state', ck.get('state_dict', ck))
    md = model.state_dict()
    # 原版 load_model：先試去掉前綴 'module'（6 字元），對不上再原名；沒對上的參數只印出來
    m = {k[6:]: v for k, v in sd.items() if k[6:] in md} or {k: v for k, v in sd.items() if k in md}
    missing = [k for k in md if k not in m]
    print(f'權重：模型 {len(md)} 個參數、對上 {len(m)}、缺 {len(missing)}', missing[:5], flush=True)
    assert not missing, '替身的參數名稱跟權重對不上'
    md.update(m)
    model.load_state_dict(md)
    model = model.cuda().eval()

    fid = {}
    for seg in pathlib.Path(a.ours).iterdir():
        if (seg / 'measurements.csv').exists():
            mm = pd.read_csv(seg / 'measurements.csv', usecols=['frame_id', 'source_frame'])
            fid.update({(seg.name, str(s)): int(f) for f, s in zip(mm.frame_id, mm.source_frame)})

    files = sorted(glob.glob(os.path.join(a.labels, '*', '*.json')))
    if a.limit:
        files = files[:a.limit]
    out = a.out_dir / f'pred_{a.tag}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, 'w', encoding='utf-8') as fo, torch.no_grad():
        for n, jf in enumerate(files, 1):
            d = json.load(open(jf, encoding='utf-8'))
            img = cv2.imread(os.path.join(a.images, d['file_path']))
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            img = (img.astype(np.float32) / 255.0 - MEAN) / STD
            x = torch.from_numpy(img.transpose(2, 0, 1))[None].cuda()
            K = np.array(d['intrinsic'], dtype=float)
            K[0] *= INPUT_W / w0
            K[1] *= INPUT_H / h0
            K = torch.tensor(K)[None].float().cuda()
            R2C = torch.tensor(road2cam(d['extrinsic']))[None].float().cuda()
            pred, height = model(x, K, R2C)
            seg = pred[0].detach().cpu().numpy()
            emb = pred[1].detach().cpu().numpy()
            offset_y = torch.sigmoid(pred[2]).detach().cpu().numpy()[0][0]
            z = height.detach().cpu().numpy()[0][0]
            canvas, ids = embedding_post((seg, emb), conf=POST_CONF, emb_margin=POST_MARGIN,
                                         min_cluster_size=POST_MIN, canvas_color=False)
            lines = bev_instance2points_with_offset_z(canvas, max_x=X_RANGE[1], meter_per_pixal=(MPP, MPP),
                                                      offset_y=offset_y, Z=z)
            lanes = [np.round(np.array([-1 * ln[1], ln[0], ln[2]]).T, 4).tolist() for ln in lines]
            lanes = [sorted(l, key=lambda q: q[1]) for l in lanes if len(l) >= 2]     # 由近到遠，同其他模型
            p = pathlib.Path(jf)
            sg, sf = p.parent.name, p.stem
            fo.write(json.dumps(dict(segment=sg, source_frame=sf, frame_id=fid.get((sg, sf), -1),
                                     coords='ground', lanes=lanes, probs=[1.0] * len(lanes), status='sclane')) + '\n')
            if n % 200 == 0 or n == len(files):
                print(f'{n}/{len(files)} {time.time() - t0:.0f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB',
                      flush=True)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
