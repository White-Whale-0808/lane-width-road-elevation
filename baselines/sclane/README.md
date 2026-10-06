# baselines/sclane

在本機跑 **SC-Lane**（Park et al., ICCV 2025，[原始碼](https://github.com/parkchaesong/SC-Lane)）的公開
OpenLane 權重，當 Z-error 的對照組之一（它主打坡度感知的路面高度，題目跟我們最接近）。**不需要編譯任何運算子**。

照原版 `tools/val.py` 與 `OpenLane_dataset_with_offset_val`：cv2 讀圖（保持 BGR，原版就是這樣餵
albumentations）、雙線性縮到 800×600、ImageNet 正規化；內參依縮放調整；road2cam 用官方地面外參。
單幀推論（不讀 heightmap —— 原版推論那行就註解掉了）；後處理 `embedding_post` → `bev_instance2points_with_offset_z`
→ 官方地面座標。SC-Lane 沒有逐條信心分數（分群出來的都算），`probs` 一律記 1.0。

替身：Deformable-DETR 的 `MultiScaleDeformableAttention` → mmcv 的純 PyTorch 版；ResNet-50 的 ImageNet
預訓練不下載（整個模型由權重覆蓋）。

驗證：506 個參數全部對上；官方上下坡子集全部車道官方 F-score **0.5373**（論文上下坡 54.6；
它沒有分數可以調門檻、論文用作者自己的評分腳本）。

## 準備（一次）

```bash
git clone --depth 1 https://github.com/parkchaesong/SC-Lane.git D:/models/sclane/SC-Lane
uv run --no-project --with gdown gdown 1UwiKDp8WzGMRd_cLYOdCs8jiuqKQ6i4Z -O D:/models/sclane/SC-Lane/ckpt.pth
# md5 325b5dc68b6e15e28d433bf15a7ea1c7
```

原始碼位置可用 `SCLANE_ROOT` 改。

## 執行

```bash
uv run --no-sync python baselines/sclane/sclane_infer.py   # -> debug/outputs/zerror/pred_sclane.jsonl

python -m openlane_module.zerror_eval --tag sclane --scope all-lanes   # 驗證：F-score 約 0.537
python -m openlane_module.zerror_eval --tag sclane --scope both
```

GTX 1650：單獨跑約 0.85 秒／幀（幾乎全是前向），顯存峰值 1.6 GB。
