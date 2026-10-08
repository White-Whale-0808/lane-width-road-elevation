# baselines/anchor3dlane

在本機跑 **Anchor3DLane++**（Huang et al.，TPAMI，[原始碼](https://github.com/tusen-ai/Anchor3DLane) 的
`anchor3dlane++` 分支）的公開 OpenLane 權重（ResNet-50、720×960），當 Z-error 的對照組之一。
**不需要編譯任何運算子**，用專案原本的 torch（CUDA）即可。

替身沿用 `baselines/latr/latr_stubs.py`：ResNet 的 DCNv2 → `torchvision.ops.deform_conv2d`、MSDA neck 的
可變形注意力 → mmcv 的純 PyTorch 版（Deformable-DETR 的 `MultiScaleDeformableAttention` 擴充給同介面的假模組）。
它自帶的 mmseg 0.26 要求 mmcv < 1.7.2，而本機 wheel 是 1.7.2 —— 只是版本檢查，載入前把版本號改報 1.7.1。
推論用不到但 mmseg 一 import 就要的模組（光達 `spconv`、Linux 才有的 `resource`、`ortools` 等）給寬鬆替身。

驗證：權重載入缺 0、多 0；官方上下坡子集（5,518 幀）全部車道官方 F-score **0.5410**（論文表 II R50† 上下坡 54.1）。

## 準備（一次）

```bash
git clone --depth 1 -b anchor3dlane++ https://github.com/tusen-ai/Anchor3DLane.git D:/models/anchor3dlane/Anchor3DLane_pp
curl -L -o D:/models/anchor3dlane/ckpt/openlane_anchor3dlane++_r50x2.pth \
  "https://huggingface.co/nowherespyfly/anchor3dlane/resolve/main/Openlane/openlane_anchor3dlane%2B%2B_r50x2.pth"
# md5 207ef11efde651fb3b49f0a67fc6fb34
```

mmcv 1.7.2 wheel 與 OpenLane 原圖的準備同 `baselines/latr/README.md`。原始碼位置可用 `A3D_ROOT` 改。

## 執行

```bash
uv run --no-sync --with D:/models/latr/wheels/mmcv-1.7.2-py2.py3-none-any.whl \
    --with mmdet==2.28.2 --with addict --with yapf==0.40.1 --with prettytable --with terminaltables \
    --with numba --with shapely --with ujson --with tqdm --with tabulate --with munkres \
    python baselines/anchor3dlane/a3d_infer.py            # -> debug/outputs/zerror/pred_a3dpp.jsonl

python -m openlane_module.zerror_eval --tag a3dpp --scope all-lanes   # 驗證：F-score 約 0.541
python -m openlane_module.zerror_eval --tag a3dpp --scope both        # 自車道 Z-error
```

GTX 1650：約 0.6 秒／幀（單獨跑），顯存峰值 1.4 GB。輸出格式同 LATR（`coords='ground'`），
只留分數 > `test_conf` 0.5 的線（原版評分的 prob_th）。
