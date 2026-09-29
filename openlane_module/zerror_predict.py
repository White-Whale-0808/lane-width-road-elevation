"""Z-error 評估第一步：在 OpenLane 官方上下坡子集跑偵測器前端，輸出每條自車道線的 3D 點（WWH-26）。

每幀寫一行 JSON 到 <out-dir>/pred_<tag>.jsonl（預設 debug/outputs/zerror）：
  segment, frame_id, source_frame, status（w_real 來源）, lanes
  lanes = [左, 右]，每條是 [[X, Y, Z], ...]，**相機光學座標**（右、下、前，公尺），依 Z 遞增；
  沒輸出就是空串列。轉到官方地面座標在 zerror_eval.py 做（要讀 GT JSON 的外參）。
資料是 export_updown.py 轉出的官方上下坡子集（預設 D:/datasets/openlane_updown），一段一個
NearfieldWidthCalibrator，設定與 WWH-25 的評估相同（不精修、近場只用漆的列量寬、有寬度檢查）。

每一列的 3D 點怎麼來（跟 pitch_estimation 同一套幾何）：
  widths 的每一列 (v, w_px) → 深度 Z = f_x·w_real/w_px
  左右線在該列的 u → X = (u − c_x)·Z/f_x
  高度用方法本身的輸出：pitch 剖面的 Y_3d（y_samples）在 Z 的內插，兩條線共用 → Y = −Y_3d
  只取剖面有定義的深度範圍（z_visible_min 到 z_samples 最遠處，含 z_cap 45 m 與 max_depth 修剪）

虛擬相機與原相機只差內參（convert_openlane.warp_image：H = K_virt·K_src⁻¹），所以這裡的
相機座標就是原相機座標，可以直接套官方的外參轉換。

    uv run --no-sync --with addict --with shapely --with yapf python -m openlane_module.zerror_predict \\
        --weights D:/models/clrnet/ft_full/ft_B_full_ep10.pth --tag ep10
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.env_setup import setup_env
setup_env()

import argparse
import numpy as np, pandas as pd, yaml

from libs.inference.pipeline_clrnet import infer_one_clrnet
from libs.inference.lane_detector import CLRNet
from libs.inference.pitch_estimation import NearfieldWidthCalibrator


def lanes_3d(res, f_x, W):
    """[左, 右] 光學座標點列；沒有 pitch 輸出回 []。"""
    pc, w_real = res['pitch_curve'], res['w_real_used']
    dbg = res['debug']
    lc, rc = dbg['left_curve'], dbg['right_curve']
    if pc['pitch_at'] is None or not len(pc['z_samples']) or lc is None or rc is None:
        return []
    widths = np.asarray(pc['widths'])
    widths = widths[widths[:, 1] > 0]
    Z = f_x * w_real / widths[:, 1]
    zs = np.asarray(pc['z_samples'])
    keep = (Z >= zs.min()) & (Z <= zs.max())
    v, Z = widths[keep, 0], Z[keep]
    if len(Z) < 2:
        return []
    Y_up = np.interp(Z, zs, np.asarray(pc['y_samples']))
    o = np.argsort(Z)
    out = []
    for c in (lc, rc):
        u = np.interp(v, c['y'], c['x'])
        X = (u - W / 2.0) * Z / f_x
        pts = np.stack([X, -Y_up, Z], 1)[o]
        out.append(np.round(pts, 4).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, default=pathlib.Path('D:/datasets/openlane_updown'))
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path('debug/outputs/zerror'))
    ap.add_argument('--mode', default='naive')
    ap.add_argument('--cut-frac', type=float, default=0.0)
    ap.add_argument('--img-w', type=int, default=800)
    ap.add_argument('--img-h', type=int, default=320)
    ap.add_argument('--max-depth', type=float, default=None, help='pitch 輸出最遠深度；預設不修剪（仍有 z_cap 45 m）')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open('config/inference_road_lane_segmentation.yaml', encoding='utf-8'))
    ccfg = yaml.safe_load(open('config/lane_detector_clrnet.yaml', encoding='utf-8'))
    pe, mo, lf = cfg['pitch_estimation'], cfg['model'], cfg['lane_fitting']
    cc = ccfg['clrnet']
    det = CLRNet(weights=a.weights, device=mo['device'], conf=cc['conf_threshold'],
                 img_w=a.img_w, img_h=a.img_h)

    out = a.out_dir / f'pred_{a.tag}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, 'w', encoding='utf-8') as fo:
        for seg_dir in sorted(p for p in a.root.iterdir() if p.is_dir()):
            meta = json.loads((seg_dir / 'metadata.json').read_text(encoding='utf-8'))
            f_x, f_y, h = meta['f_x'], meta['f_y'], meta['camera_height']
            H, W = meta['resize_size']
            cal = NearfieldWidthCalibrator(f_x, f_y, H, h)
            for r in pd.read_csv(seg_dir / 'measurements.csv').itertuples():
                img = seg_dir / 'images' / f'{r.frame_id:06d}.png'
                if not img.exists():
                    continue
                if a.limit and n >= a.limit:
                    break
                n += 1
                cal.advance_to(float(r.collect_dist_m))
                rec = dict(segment=seg_dir.name, frame_id=int(r.frame_id),
                           source_frame=str(r.source_frame), lanes=[], status=None)
                try:
                    res = infer_one_clrnet(det, str(img), (H, W), lf['num_samples'], f_x, f_y, h,
                                           cut_frac=a.cut_frac, detector_mode=a.mode, tail='keep',
                                           refine='none', nearfield_source='paint', ego_guard=True,
                                           max_depth_m=a.max_depth,
                                           samples_per_meter=lf.get('samples_per_meter'),
                                           method=pe.get('method', 'windowed'),
                                           w_real_calibrator=cal,
                                           last_resort_lane_width=ccfg['fitting']['last_resort_lane_width'],
                                           return_debug=True)
                    rec['status'] = res['w_real_status']
                    rec['lanes'] = lanes_3d(res, f_x, W)
                except Exception as e:
                    rec['status'] = f'error:{type(e).__name__}:{e}'[:120]
                fo.write(json.dumps(rec) + '\n')
                if n % 250 == 0:
                    print(f'{n} {seg_dir.name[:30]} f{r.frame_id}', flush=True)
            if a.limit and n >= a.limit:
                break
    print(f'{n} frames -> {out}')


if __name__ == '__main__':
    main()
