"""CLRNet (CVPR 2022, CULane weights) as the lane-detector front end (WWH-25).

Copied from debug/clrnet_infer.py (the step-1 evaluation wrapper, left as is)
and extended to return each lane's confidence. The rest of that module's notes
apply unchanged and are kept below.

Running it needs CLRNet's pure-Python dependencies, which are deliberately NOT
project dependencies (a plain `uv sync` would also swap the CUDA torch):

    uv run --no-sync --with addict --with shapely --with yapf python ...

`setup_env()` must have run before this module is imported (cv2).

Paths (environment variables, machine-specific):
    CLRNET_ROOT     CLRNet source checkout (the directory holding `clrnet/`),
                    default D:/models/clrnet
    CLRNET_WEIGHTS  culane_dla34.pth, default $CLRNET_ROOT/culane_dla34.pth

CLRNet（CVPR 2022，CULane 權重）推論包裝，不安裝它的依賴。

CLRNet 原始碼放在 repo 外（``CLRNET_ROOT``，預設 D:/models/clrnet，
git clone https://github.com/Turoad/CLRNet + releases/models/culane_dla34.pth）。
它的依賴在本機裝不起來或不需要，這裡用最小替身塞進 sys.modules：

- ``mmcv.cnn.ConvModule``：conv → BN → ReLU，bias='auto'（有 norm 就不帶 bias），
  屬性名 ``conv`` / ``bn`` 與 checkpoint 的 key 對得上。
- ``clrnet.ops.nms``：原本是要編譯的 CUDA 擴充（本機沒有 nvcc / MSVC）。照
  ``nms_kernel.cu`` 的 devIoU 逐行移植成 Python：兩條線在共同的 strip 範圍內
  逐點 |Δx| 總和 < threshold × 點數 就算重疊，依分數貪婪保留，最多 top_k 條。
- ``mmcv.jit``：訓練時的 accuracy 才用得到，給恆等 decorator。

前處理（``mode``）：
  naive    裁掉 cut 以上、直接縮成 800×320（寬高各自縮）。
  culane   也裁掉 cut 以上，但把寬度縮成讓 f_x'/f_y' 等於 CULane 訓練時的
           0.488（1640→800 橫向、縱向不縮），不足 800 的左右補中性灰（ImageNet 均值色）。
           理由：網路學到的是 CULane 影像裡車道線的「斜率分布」，這個比值決定斜率。
正規化照 CLRNet 自己的 val pipeline：BGR / 255，**不減均值**（config 的
img_norm 在 CULane pipeline 裡沒被用到）。
輸出一律換回輸入影像的像素座標（x, y），每條線一個 (N, 2) 陣列，y 遞增。
"""
import os, sys, types, pathlib
import numpy as np
import torch
import torch.nn as nn
import cv2

CLRNET_ROOT = pathlib.Path(os.environ.get('CLRNET_ROOT', 'D:/models/clrnet'))
# 權重可以和原始碼分開放（5080 那台：原始碼在 D:/models/clrnet/CLRNet、權重在上一層）
CLRNET_WEIGHTS = pathlib.Path(os.environ.get('CLRNET_WEIGHTS',
                                             str(CLRNET_ROOT / 'culane_dla34.pth')))
CULANE_FXFY = 800 / 1640          # CULane: 1640 寬縮到 800，高 320 不縮（方像素相機）
IMG_W, IMG_H = 800, 320
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


# ------------------------------------------------------------------ 替身
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias='auto', conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='ReLU'), inplace=True, **_):
        super().__init__()
        with_norm = norm_cfg is not None
        if bias == 'auto':
            bias = not with_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation, groups, bias=bias)
        if with_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.with_norm = with_norm
        self.act = nn.ReLU(inplace=inplace) if act_cfg is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        return self.act(x) if self.act is not None else x


def _lane_overlap(a, b, thr, n_strips=71, n_offsets=72):
    sa = int(a[2] * n_strips + 0.5); sb = int(b[2] * n_strips + 0.5)
    start = max(sa, sb)
    ea = int(sa + a[4] - 1 + 0.5 - ((a[4] - 1) < 0))
    eb = int(sb + b[4] - 1 + 0.5 - ((b[4] - 1) < 0))
    end = min(ea, eb, n_offsets - 1)
    if end < start:
        return False
    d = np.abs(a[5 + start:5 + end + 1] - b[5 + start:5 + end + 1]).sum()
    return d < thr * (end - start + 1)


def nms(boxes, scores, overlap, top_k):
    """nms_kernel.cu 的 Python 版：回傳 (keep, num_to_keep, parent)。"""
    order = torch.argsort(scores, descending=True)
    B = boxes.detach().cpu().numpy()
    idx = order.cpu().numpy()
    keep, removed = [], np.zeros(len(idx), bool)
    for i in range(len(idx)):
        if removed[i]:
            continue
        keep.append(idx[i])
        if len(keep) == top_k:
            break
        for j in range(i + 1, len(idx)):
            if not removed[j] and _lane_overlap(B[idx[i]], B[idx[j]], overlap):
                removed[j] = True
    k = torch.zeros(len(idx), dtype=torch.long, device=boxes.device)
    k[:len(keep)] = torch.as_tensor(keep, dtype=torch.long)
    return k, len(keep), None


def _install_stubs():
    mm = types.ModuleType('mmcv'); mm.jit = lambda **kw: (lambda f: f)
    cnn = types.ModuleType('mmcv.cnn'); cnn.ConvModule = ConvModule
    par = types.ModuleType('mmcv.parallel'); par.DataContainer = object; par.collate = None
    par.MMDataParallel = None
    run = types.ModuleType('mmcv.runner'); run.auto_fp16 = lambda *a, **k: (lambda f: f)
    mm.cnn, mm.parallel, mm.runner = cnn, par, run
    sys.modules.update({'mmcv': mm, 'mmcv.cnn': cnn, 'mmcv.parallel': par,
                        'mmcv.runner': run})
    ops = types.ModuleType('clrnet.ops'); ops.nms = nms
    sys.modules['clrnet.ops'] = ops
    if not hasattr(np, 'bool'):
        np.bool = bool          # predictions_to_pred 用了 np.bool（numpy ≥1.24 已移除）


class CLRNet:
    def __init__(self, weights=None, device='cuda', conf=0.4, img_w=IMG_W, img_h=IMG_H):
        """img_w / img_h: the network input size the weights were trained at —
        800×320 for the CULane release; a fine-tune may use another (WWH-25)."""
        self.img_w, self.img_h = int(img_w), int(img_h)
        _install_stubs()
        sys.path.insert(0, str(CLRNET_ROOT))
        from addict import Dict
        import clrnet.models  # noqa: F401  (註冊 backbone / neck / head)
        from clrnet.models.registry import build_net
        cfg = Dict(dict(
            net=dict(type='Detector'),
            backbone=dict(type='DLAWrapper', dla='dla34', pretrained=False),
            num_points=72, max_lanes=4, sample_y=range(589, 230, -20),
            heads=dict(type='CLRHead', num_priors=192, refine_layers=3,
                       fc_hidden_dim=64, sample_points=36),
            neck=dict(type='FPN', in_channels=[128, 256, 512], out_channels=64,
                      num_outs=3, attention=False),
            test_parameters=dict(conf_threshold=conf, nms_thres=50, nms_topk=4),
            img_w=self.img_w, img_h=self.img_h, ori_img_w=1640, ori_img_h=590, cut_height=270,
            num_classes=5, ignore_label=255, bg_weight=0.4,
            iou_loss_weight=2., cls_loss_weight=2., xyt_loss_weight=0.2, seg_loss_weight=1.))
        cfg.haskey = lambda k: k in cfg
        self.cfg = cfg
        net = build_net(cfg)
        sd = torch.load(weights or CLRNET_WEIGHTS,
                        map_location='cpu', weights_only=False)['net']
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # 這四個是 head 在 __init__ 依設定算出的 buffer，舊 checkpoint 沒存
        derived = {'heads.sample_x_indexs', 'heads.prior_feat_ys', 'heads.prior_ys',
                   'heads.criterion.weight'}
        missing = [k for k in missing if k not in derived]
        assert not missing and not unexpected, f'missing {missing[:5]} unexpected {unexpected[:5]}'
        self.net = net.to(device).eval()
        self.device = device

    @classmethod
    def from_config(cls, clrnet_cfg, device='cuda'):
        """(detector, cut_frac, mode) from the `clrnet:` block of
        config/lane_detector_clrnet.yaml. Weights, network input size, top crop
        and preprocessing belong together — a fine-tune only works with the
        settings it was trained with — so they are read from one place; pass
        cut_frac and mode on to pipeline_clrnet.infer_one_clrnet."""
        c = clrnet_cfg
        det = cls(weights=c.get('weights'), device=device, conf=c['conf_threshold'],
                  img_w=c.get('img_w', IMG_W), img_h=c.get('img_h', IMG_H))
        return det, float(c['cut_frac']), c.get('mode', 'culane')

    @torch.no_grad()
    def __call__(self, img_rgb, f_x, f_y, cut, mode='culane'):
        """img_rgb: H×W×3 uint8。回傳 (lanes, conf)：lanes 是 [ (N,2) ndarray (x, y)
        原圖像素 ]，conf 是每條線的前景 logit（CLRNet Lane.metadata['conf']，
        未經 softmax；只用來排序／比較，門檻是建構時的 conf）。"""
        H, W = img_rgb.shape[:2]
        crop = cv2.cvtColor(img_rgb[cut:], cv2.COLOR_RGB2BGR).astype(np.float32)
        sy = self.img_h / (H - cut)
        if mode == 'naive':
            sx, x0 = self.img_w / W, 0
            body = cv2.resize(crop, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            canvas = body
        else:
            sx = sy * CULANE_FXFY * f_y / f_x
            new_w = int(round(W * sx))
            body = cv2.resize(crop, (new_w, self.img_h), interpolation=cv2.INTER_AREA)
            canvas = np.tile(MEAN_BGR, (self.img_h, self.img_w, 1))
            if new_w <= self.img_w:
                x0 = (self.img_w - new_w) // 2
                canvas[:, x0:x0 + new_w] = body
            else:                                   # 比 800 寬就裁中間
                c = (new_w - self.img_w) // 2
                canvas = body[:, c:c + self.img_w]; x0 = -c
        t = torch.from_numpy((canvas / 255.0).astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        # CLRNet 的 head 把 y 換回 CULane 原圖座標：這裡讓它輸出「網路輸入」座標
        self.cfg.ori_img_h, self.cfg.cut_height, self.cfg.ori_img_w = self.img_h, 0, self.img_w
        out = self.net(t)
        lanes = self.net.heads.get_lanes(out)[0]
        res, conf = [], []
        for ln in lanes:
            p = np.asarray(ln.points, dtype=np.float64)       # 正規化到 800×320
            x = (p[:, 0] * self.img_w - x0) / sx
            y = p[:, 1] * self.img_h / sy + cut
            ok = (x >= 0) & (x < W)
            if ok.sum() >= 2:
                o = np.argsort(y[ok])
                res.append(np.stack([x[ok][o], y[ok][o]], 1))
                conf.append(float(ln.metadata['conf']))
        return res, conf
