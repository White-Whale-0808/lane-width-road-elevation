# baselines/latr

在本機跑 **LATR**（Luo et al., ICCV 2023，[原始碼](https://github.com/JMoonr/LATR)）的公開 OpenLane 權重，
當 Z-error 的對照組（Linear WWH-26）。**不需要編譯任何 CUDA 運算子**，用專案原本的 torch（CUDA）即可。

LATR 原版需要 torch 1.8＋自己編譯的 mmcv-full 1.5／mmdet3d 1.0.0rc3。這裡改用：

- 純 Python 的 mmcv 1.7.2（lite）＋ mmdet 2.28.2 ＋ fvcore
- `latr_stubs.py`：兩個要編譯的運算子換成等價實作——ResNet 的可變形卷積 v2 → `torchvision.ops.deform_conv2d`，
  可變形注意力 → mmcv 自己的純 PyTorch 版；其餘 `mmcv.ops` 給空類別讓 mmdet 能 import；
  `mmdet3d` 的 build_backbone／build_neck 轉給 mmdet；補回 NumPy 2 拿掉的別名

驗證：權重載入時參數**缺 0、多 0**；官方上下坡子集（5,518 幀）用官方規則算全部車道，F-score **0.564**
（論文 0.552，README 說公開權重比論文好）。

## 準備（一次）

```bash
# 1. LATR 原始碼與權重（OpenLane-1000 full，md5 d8ecb900c34fd23a9e7af840aff00843）
git clone https://github.com/JMoonr/LATR.git D:/models/latr/LATR
uv run --no-project --with gdown gdown 1jThvqnJ2cUaAuKdlTuRKjhLCH0Zq62A1 -O D:/models/latr/LATR/pretrained_models/openlane.pth

# 2. mmcv 1.7.2（lite）的 wheel：它的 setup.py 要 pkg_resources，新版 setuptools 沒有 → 用舊版自己建
uv venv --python 3.12 D:/models/latr/buildenv
uv pip install --python D:/models/latr/buildenv "setuptools<70" wheel pip
MMCV_WITH_OPS=0 D:/models/latr/buildenv/Scripts/python.exe -m pip wheel "mmcv==1.7.2" --no-build-isolation --no-deps -w D:/models/latr/wheels

# 3. 原始影像：LATR 讀 1920×1280 原圖（D:/datasets/openlane/images/validation/<segment>/<frame>.jpg）。
#    官方上下坡清單是 D:/datasets/openlane/test/up_down_case/（lane3d_1000_validation_test.tar 解出）；
#    影像從 images_validation_{0..3}.tar 只抽這 5,518 張即可（Python tarfile 串流，約 20 秒）
```

LATR 原始碼位置可用環境變數 `LATR_ROOT` 改（預設 `D:/models/latr/LATR`）。

## 執行

```bash
uv run --no-sync --with D:/models/latr/wheels/mmcv-1.7.2-py2.py3-none-any.whl \
    --with mmdet==2.28.2 --with fvcore --with addict --with yapf==0.40.1 \
    python baselines/latr/latr_infer.py            # -> debug/outputs/zerror/pred_latr.jsonl

python -m openlane_module.zerror_eval --tag latr --scope all-lanes   # 驗證：F-score 應約 0.564
python -m openlane_module.zerror_eval --tag latr --scope both        # 自車道 Z-error
```

⚠ **不要加 `--with geffnet`**：它依賴 torch，uv 會把 overlay 的 torch 換成 CPU 版。
LATR 只有 EfficientNet 抽特徵器 import 它，ResNet 設定用不到，`latr_stubs.py` 給空模組。

GTX 1650（4 GB）：約 0.33 秒／幀（含讀原圖），顯存峰值 0.4 GB。純 PyTorch 的可變形注意力
在前向 218 ms 裡只佔約 7 ms，所以移植沒有明顯拖慢。

## 輸出格式

`pred_<tag>.jsonl` 一幀一行：`segment`、`source_frame`、`frame_id`（對到 `D:/datasets/openlane_updown`
的編號）、`coords='ground'`（官方地面座標，評分時不再轉）、`lanes`（通過 LATR 自己評估門檻
「最大類別機率 > 0.3」的每條線 `[[x, y, z], ...]`）、`probs`。
