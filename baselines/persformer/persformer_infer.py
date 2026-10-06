"""用 PersFormer（Chen et al., ECCV 2022；OpenLane 官方基準，v1.2 權重）在 OpenLane 官方上下坡子集推論。

前處理、解碼都直接用 PersFormer 自己的程式：LaneDataset（讀 test/up_down_case 的標註 json、縮放、
正規化、外參）、unit_update_projection_extrinsic、unormalize_lane_anchor、nms_bev、
compute_3d_lanes_all_category；輸出照它評分的規則留 max(類別機率) > prob_th 0.5 的線。
⚠ 輸出還原用的正規化標準差：原版用訓練集的，本機沒有訓練集標註 → 用完整驗證集（--std-from）的。
不換的話會用上下坡子集自己的真值算（F 0.415，README 46.8）。

不需要編譯任何運算子：
  * 可變形注意力（Deformable-DETR 的 MultiScaleDeformableAttention 擴充）→ mmcv 的純 PyTorch 版
  * nms 擴充（LaneATT 的車道線 NMS）：utils.nms_bev 在 3D 輸出上會用到，照 nms_kernel.cu 逐行改寫成
    Python（lane_nms）；2D 輔助頭不呼叫它
  * geffnet（EfficientNet-B7 骨幹）：用 Anchor3DLane repo 附的 gen-efficientnet-pytorch 原始碼，
    不 pip 安裝（pip 版會把 torch 換成 CPU 版）；ImageNet 預訓練不下載，整個模型由權重覆蓋

輸出 <out-dir>/pred_<tag>.jsonl（格式同 baselines/latr：coords='ground'）。

    uv run --no-sync --with D:/models/latr/wheels/mmcv-1.7.2-py2.py3-none-any.whl --with addict --with yapf==0.40.1 \\
        python baselines/persformer/persformer_infer.py [--limit N]
"""
import os, sys, json, pathlib, argparse, time, types
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
PF_ROOT = pathlib.Path(os.environ.get('PERSFORMER_ROOT', 'D:/models/persformer/PersFormer_3DLane'))
GEFFNET = pathlib.Path(os.environ.get('GEFFNET_SRC', 'D:/models/anchor3dlane/Anchor3DLane_pp/gen-efficientnet-pytorch'))
import importlib.util
_spec = importlib.util.spec_from_file_location('mono3d_env_setup', REPO / 'utils' / 'env_setup.py')
_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env)
_env.setup_env()

import numpy as np, pandas as pd, torch
sys.path.insert(0, str(REPO / 'baselines' / 'latr'))
import latr_stubs


N_OFFSETS = 72                    # nms_kernel.cu 寫死的 N_OFFSETS（PROP_SIZE = 5 + 72）


def lane_nms(boxes, scores, overlap, top_k):
    """models/nms/src/nms_kernel.cu（LaneATT 的車道線 NMS）逐行改寫成 Python，給 utils.nms_bev 用。

    每列 [.., .., start/(N_OFFSETS-1), .., length, x_0 .. x_71]。兩條線在共同的取樣索引
    [max(start), min(end)] 上 Σ|Δx| < overlap × 點數 ＝ 重疊；依分數由高到低貪婪保留、刪掉
    與已保留者重疊的，留滿 top_k 就停。回傳 (keep, num_to_keep, None) 同擴充模組。
    """
    b = boxes.detach().float().cpu().numpy()
    n = len(b)
    assert b.shape[1] == 5 + N_OFFSETS, 'nms_kernel.cu 只接受 5 + 72 欄'
    order = torch.sort(scores.detach().float().cpu(), 0, descending=True)[1].numpy()

    def span(r):
        s = int(r[2] * (N_OFFSETS - 1) + 0.5)
        e = int(s + r[4] - 1 + 0.5 - ((r[4] - 1) < 0))
        return s, e

    def overlaps(a, c):
        (sa, ea), (sc, ec) = span(a), span(c)
        s, e = max(sa, sc), min(ea, ec, N_OFFSETS - 1)
        if e < s:
            return False
        return np.abs(a[5 + s:6 + e] - c[5 + s:6 + e]).sum() < overlap * (e - s + 1)

    removed = np.zeros(n, bool)
    keep = []
    for ii in range(n):
        if removed[ii]:
            continue
        keep.append(int(order[ii]))
        for jj in range(ii + 1, n):
            if not removed[jj] and overlaps(b[order[ii]], b[order[jj]]):
                removed[jj] = True
        if len(keep) == top_k:
            break
    out = torch.zeros(n, dtype=torch.long)
    out[:len(keep)] = torch.tensor(keep, dtype=torch.long)
    return out, torch.tensor(min(top_k, len(keep))), None


def install_stubs():
    for k, v in dict(float=float, int=int, bool=bool, object=object, long=int).items():
        if k not in np.__dict__:
            setattr(np, k, v)
    if 'RankWarning' not in np.__dict__:
        np.RankWarning = np.exceptions.RankWarning
    msda = types.ModuleType('MultiScaleDeformableAttention')

    def ms_deform_attn_forward(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return latr_stubs.multi_scale_deformable_attn_pytorch(
            value, [(int(h), int(w)) for h, w in shapes], sampling_locations, attention_weights)
    msda.ms_deform_attn_forward = ms_deform_attn_forward
    sys.modules['MultiScaleDeformableAttention'] = msda

    nms = types.ModuleType('nms')
    nms.nms = lane_nms
    sys.modules['nms'] = nms

    sys.path.insert(0, str(GEFFNET))
    import geffnet
    for name in dir(geffnet):
        f = getattr(geffnet, name)
        if name.startswith(('tf_efficientnet', 'efficientnet')) and callable(f):
            setattr(geffnet, name, (lambda f: lambda *a, **k: f(*a, **{**k, 'pretrained': False}))(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='D:/models/persformer/ckpt/persformer_openlane_v12.pth')
    ap.add_argument('--data-dir', default='D:/datasets/openlane/lane3d_1000/', help='其下 test/up_down_case/ 是標註；路徑要含 lane3d_1000（PersFormer 用它判斷版本，D:/datasets/openlane/lane3d_1000 是指回 openlane 的目錄連結）')
    ap.add_argument('--images', default='D:/datasets/openlane/images/')
    ap.add_argument('--ours', default='D:/datasets/openlane_updown', help='拿 source_frame→frame_id 對照')
    ap.add_argument('--tag', default='persformer')
    ap.add_argument('--out-dir', type=pathlib.Path, default=REPO / 'debug/outputs/zerror')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--std-from', default='validation/',
                    help='算正規化標準差的標註目錄（相對 --data-dir）；原版用訓練集，本機只有驗證集')
    a = ap.parse_args()

    install_stubs()
    sys.path.insert(0, str(PF_ROOT))           # PersFormer 的 utils / data / models 排前面
    cwd = os.getcwd()
    os.chdir(PF_ROOT)
    from config import persformer_openlane
    from utils.utils import define_args, unit_update_projection_extrinsic
    from data.Load_Data import LaneDataset, unormalize_lane_anchor, nms_bev, compute_3d_lanes_all_category
    from models.PersFormer import PersFormer
    os.chdir(cwd)

    args = define_args().parse_args([])
    persformer_openlane.config(args)
    args.dataset_dir, args.data_dir = a.images, a.data_dir
    args.evaluate, args.distributed, args.no_cuda = True, False, False
    args.proc_id, args.local_rank, args.world_size, args.gpu = 0, 0, 1, 0
    args.batch_size = 1
    args.save_json_path = args.save_path = str(PF_ROOT.parent / 'work')     # Runner.__init__ 原本設的
    os.makedirs(args.save_path, exist_ok=True)

    # 輸出還原成公尺用的正規化標準差（x 偏移、z）：原版 Runner 換成**訓練集**的（runner.py
    # _get_valid_dataset 的 set_x_off_std / set_z_std），LaneDataset 預設卻是用「當下載入那批」的真值算。
    # 本機沒有訓練集標註，改用完整驗證集（202 段）的；先建它，再建上下坡那份——
    # 2D anchor 取最後建的資料集（同原版：先訓練集、後驗證集）。
    # 它建資料集時會 assert 每張影像存在；算標準差只讀標註，本機只解了上下坡的影像 → 這份暫時不檢查
    import data.Load_Data as LD
    real_exists = LD.ops.exists
    LD.ops.exists = lambda p: True if str(p).endswith('.jpg') else real_exists(p)
    try:
        std_ds = LaneDataset(args.dataset_dir, os.path.join(args.data_dir, a.std_from), args, seg_bev=args.seg_bev)
    finally:
        LD.ops.exists = real_exists
    print(f'正規化標準差取自 {a.std_from}：x_off {np.round(std_ds._x_off_std, 3)}  z {np.round(std_ds._z_std, 3)}', flush=True)
    ds = LaneDataset(args.dataset_dir, os.path.join(args.data_dir, 'test/up_down_case/'), args, seg_bev=args.seg_bev)
    print(f'（上下坡子集自己的：x_off {np.round(ds._x_off_std, 3)}  z {np.round(ds._z_std, 3)}，不用）', flush=True)
    ds.set_x_off_std(std_ds._x_off_std)
    ds.set_z_std(std_ds._z_std)
    del std_ds
    model = PersFormer(args)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'權重：缺 {len(missing)}、多 {len(unexpected)}', missing[:5], unexpected[:5], flush=True)
    assert not missing and not unexpected, '替身的參數名稱跟權重對不上'
    model = model.cuda().eval()

    fid = {}
    for seg in pathlib.Path(a.ours).iterdir():
        if (seg / 'measurements.csv').exists():
            m = pd.read_csv(seg / 'measurements.csv', usecols=['frame_id', 'source_frame'])
            fid.update({(seg.name, str(s)): int(f) for f, s in zip(m.frame_id, m.source_frame)})

    n_all = len(ds) if not a.limit else min(a.limit, len(ds))
    out = a.out_dir / f'pred_{a.tag}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, 'w', encoding='utf-8') as fo, torch.no_grad():
        for n in range(n_all):
            (json_file, inp, seg_maps, gt, gt_img, idx, gt_hcam, gt_pitch, K, E, seg_name, seg_bev) = ds[n]
            inp = torch.as_tensor(inp)[None].cuda().float()
            K = torch.as_tensor(K)[None].cuda().float()
            E = torch.as_tensor(E)[None].cuda().float()
            M_inv = unit_update_projection_extrinsic(args, E, K)
            _, output_net, _, _, _, _ = model(input=inp, _M_inv=M_inv)
            output_net = output_net.data.cpu().numpy()
            unormalize_lane_anchor(output_net[0], ds)
            if not args.use_default_anchor:
                output_net = nms_bev(output_net, args)
            lanes, _, probs, _ = compute_3d_lanes_all_category(output_net[0], ds, args.anchor_y_steps,
                                                               E[0].cpu().numpy()[2, 3], args.model_name)
            keep = [i for i, p in enumerate(probs) if max(p) > args.prob_th]
            p = pathlib.Path(json_file)
            seg, sf = p.parent.name, p.stem
            fo.write(json.dumps(dict(segment=seg, source_frame=sf, frame_id=fid.get((seg, sf), -1),
                                     coords='ground', lanes=[np.round(lanes[i], 4).tolist() for i in keep],
                                     probs=[float(max(probs[i])) for i in keep], status='persformer')) + '\n')
            if (n + 1) % 200 == 0 or n + 1 == n_all:
                print(f'{n + 1}/{n_all} {time.time() - t0:.0f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB',
                      flush=True)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
