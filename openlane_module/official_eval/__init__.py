"""OpenLane 官方 3D 車道線評估程式（第三方，Apache License 2.0，版權聲明保留在各檔頭）。

來源：https://github.com/OpenDriveLab/OpenLane/tree/main/eval/LANE_evaluation/lane3d
（`eval_3D_lane.py` 最後一次上游修改 2022-09-27「prune invisible gt points before evaluation」）。

本 repo 只做了三處改動，都不影響評分數字：
  1. `eval_3D_lane.py` 的兩行 import 改成相對匯入（官方寫 `from utils.utils import *`，
     會跟本 repo 的 `utils/` 撞名）。
  2. `utils/MinCostFlow.py` 整檔換成等價的 scipy 版：官方用的 `ortools.graph.pywrapgraph`
     在新版 ortools 已移除。官方解的是「完全二分圖、容量 1、流量＝min(列, 欄)」的最小成本流，
     就是 `scipy.optimize.linear_sum_assignment` 的長方形指派；Google 官方範例的最小成本 265
     一致（tests/test_zerror_eval.py）。
  3. 官方 `LaneEval.bench` 讀 `self.dataset_name` 但 `__init__` 沒設，照原樣跑會 AttributeError；
     呼叫端（openlane_module/zerror_eval.py）補 `ev.dataset_name = 'openlane'`，本檔不改。
"""
