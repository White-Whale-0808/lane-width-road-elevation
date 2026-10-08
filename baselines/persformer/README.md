# baselines/persformer

在本機跑 **PersFormer**（Chen et al., ECCV 2022，OpenLane 官方基準，[原始碼](https://github.com/OpenDriveLab/PersFormer_3DLane)）
的公開 OpenLane v1.2 權重，當 Z-error 的對照組之一。**不需要編譯任何運算子**。

前處理與解碼直接用 PersFormer 自己的程式（`LaneDataset`、`unormalize_lane_anchor`、`nms_bev`、
`compute_3d_lanes_all_category`）。要編譯的兩個擴充：

- 可變形注意力（Deformable-DETR 的 `MultiScaleDeformableAttention`）→ mmcv 的純 PyTorch 版
- `nms`（LaneATT 的車道線 NMS）：**3D 輸出的 `nms_bev` 會用到**，照 `models/nms/src/nms_kernel.cu`
  逐行改寫成 Python（`persformer_infer.lane_nms`）
- EfficientNet-B7 骨幹的 `geffnet`：用 Anchor3DLane repo 附的 `gen-efficientnet-pytorch` 原始碼
  （pip 版會把 torch 換成 CPU 版），ImageNet 預訓練不下載

⚠ **輸出還原成公尺用的正規化標準差**：原版 Runner 換成**訓練集**的（`runner.py` `_get_valid_dataset` 的
`set_x_off_std` / `set_z_std`），`LaneDataset` 預設卻用「當下載入那批」的真值算。本機沒有訓練集標註、
權重檔也沒存 → 用**完整驗證集**（202 段，`--std-from validation/`）算。不換的話上下坡 F-score 0.415；
換了 **0.4642**（README v1.2 上下坡 46.8）。仍是近似：要完全照原版得拿訓練集標註算。

PersFormer 用標註路徑裡有沒有 `lane3d_1000` 判斷版本，所以 `--data-dir` 指到
`D:/datasets/openlane/lane3d_1000`（指回 `D:/datasets/openlane` 的目錄連結，Windows junction）。

## 準備（一次）

```bash
git clone --depth 1 https://github.com/OpenDriveLab/PersFormer_3DLane.git D:/models/persformer/PersFormer_3DLane
uv run --no-project --with gdown gdown 1FHbko2ocdxZYaxfG8a7m9qJMtHHMmwGQ -O D:/models/persformer/ckpt/persformer_openlane_v12.pth
# md5 d6f4543d94c2ba66572d3889168021a0
# PowerShell：New-Item -ItemType Junction -Path D:\datasets\openlane\lane3d_1000 -Target D:\datasets\openlane
```

`gen-efficientnet-pytorch` 來自 Anchor3DLane repo（見 `baselines/anchor3dlane/README.md`），位置可用
`GEFFNET_SRC` 改；PersFormer 原始碼位置可用 `PERSFORMER_ROOT` 改。

## 執行

```bash
uv run --no-sync --with D:/models/latr/wheels/mmcv-1.7.2-py2.py3-none-any.whl --with addict --with yapf==0.40.1 \
    python baselines/persformer/persformer_infer.py       # -> debug/outputs/zerror/pred_persformer.jsonl

python -m openlane_module.zerror_eval --tag persformer --scope all-lanes   # 驗證：F-score 約 0.464
python -m openlane_module.zerror_eval --tag persformer --scope both
```

建完整驗證集算標準差約 11 分鐘；之後約 0.4–0.9 秒／幀，顯存 0.7 GB。只留 max(類別機率) > 0.5 的線（原版 prob_th）。
