"""Z-error 評估第一步：在 OpenLane 官方上下坡子集跑偵測器前端，輸出每條自車道線的 3D 點（WWH-26）。

每幀寫一行 JSON 到 <out-dir>/pred_<tag>.jsonl（預設 debug/outputs/zerror）：
  segment, frame_id, source_frame, status（w_real 來源）, lanes
  lanes = [左, 右]，每條是 [[X, Y, Z], ...]，**相機光學座標**（右、下、前，公尺），依 Z 遞增；
  沒輸出就是空串列。轉到官方地面座標在 zerror_eval.py 做（要讀 GT JSON 的外參）。
資料是 export_updown.py 轉出的官方上下坡子集（預設 D:/datasets/openlane_updown），一段一個
量寬器（config fitting.width_filter：kalman／median，pitch_estimation.width_calibrator），其餘設定與 WWH-25 的評估相同（不精修、近場只用漆的列量寬、有寬度檢查）。
相機取自每段的 metadata.json（libs.dataset_camera）。偵測器的權重與前處理預設都讀
config/lane_detector_clrnet.yaml；用 --weights 換權重時必須同時給 --mode、--cut-frac 與 --conf，
前處理要跟權重的訓練方式一致（CULane 權重配錯前處理會掉線，86% → 66.5%），分數尺度也跟權重走
（ep10 用 0.4 會丟掉抓對的線，要 --conf 0.3，WWH-28）。

每一列的 3D 點怎麼來（跟 pitch_estimation 同一套幾何）：
  widths 的每一列 (v, w_px) → 深度 Z = f_x·w_real/w_px
  左右線在該列的 u → X = (u − c_x)·Z/f_x
  高度用方法本身的輸出：pitch 剖面的 Y_3d（y_samples）在 Z 的內插，兩條線共用 → Y = −Y_3d
  只取剖面有定義的深度範圍（z_visible_min 到 z_samples 最遠處，含 z_cap 45 m 與 max_depth 修剪）

虛擬相機與原相機只差內參（convert_openlane.warp_image：H = K_virt·K_src⁻¹），所以這裡的
相機座標就是原相機座標，可以直接套官方的外參轉換。

    uv run --no-sync --with addict --with shapely --with yapf python -m openlane_module.zerror_predict \\
        --weights D:/models/clrnet/ft_full/ft_B_full_ep10.pth --mode naive --cut-frac 0 --tag ep10
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.env_setup import setup_env
setup_env()

import argparse
import numpy as np, pandas as pd, yaml

from libs.inference.pipeline_clrnet import infer_one_clrnet
from libs.inference.lane_detector import CLRNet
from libs.inference.pitch_estimation import width_calibrator
from libs.dataset_camera import camera_from_metadata


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
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path('debug/outputs/zerror'))
    ap.add_argument('--weights', default=None, help='省略＝config 的權重與前處理；給了就要一起給 --mode 與 --cut-frac')
    ap.add_argument('--mode', default=None, help='culane / naive')
    ap.add_argument('--cut-frac', type=float, default=None)
    ap.add_argument('--img-w', type=int, default=None)
    ap.add_argument('--img-h', type=int, default=None)
    ap.add_argument('--conf', type=float, default=None, help='偵測門檻；省略＝config 的 conf_threshold（給 --weights 時必給）')
    ap.add_argument('--refine', default=None, choices=['none', 'center', 'solid', 'smooth'], help='線位置精修（pipeline_clrnet refine）；省略＝config 的 fitting.refine')
    ap.add_argument('--keep-wide', default=None, choices=['on', 'off'],
                    help='保留平行的寬線對（WWH-33）；省略＝config')
    ap.add_argument('--widening', default=None, choices=['on', 'off'],
                    help='前方車道變寬處截斷輸出（WWH-34）；省略＝config')
    ap.add_argument('--scale-tol', default=None,
                    help='尺度一致性容許比例（WWH-33）；省略＝config，off＝不檢查')
    ap.add_argument('--max-depth', type=float, default=None, help='pitch 輸出最遠深度；預設不修剪（仍有 z_cap 45 m）')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dump', action='store_true',
                    help='另存每幀所有偵測線、分數、選出的左右曲線與相機 → <out-dir>/dump_<tag>.pkl（離線實驗用）')
    a = ap.parse_args()
    if a.weights and (a.mode is None or a.cut_frac is None or a.conf is None):
        ap.error('--weights 要搭配 --mode、--cut-frac 與 --conf（前處理與分數尺度都跟權重走）')

    cfg = yaml.safe_load(open('config/inference_road_lane_segmentation.yaml', encoding='utf-8'))
    ccfg = yaml.safe_load(open('config/lane_detector_clrnet.yaml', encoding='utf-8'))
    pe, mo, lf = cfg['pitch_estimation'], cfg['model'], cfg['lane_fitting']
    cc = ccfg['clrnet']
    if a.conf is not None:
        cc = {**cc, 'conf_threshold': a.conf}
    if a.weights is None:
        det, cut_frac, mode = CLRNet.from_config(cc, device=mo['device'])
    else:
        det = CLRNet(weights=a.weights, device=mo['device'], conf=cc['conf_threshold'],
                     img_w=a.img_w or cc.get('img_w', 800), img_h=a.img_h or cc.get('img_h', 320))
        cut_frac, mode = a.cut_frac, a.mode
    refine = a.refine or ccfg['fitting'].get('refine', 'none')

    def opt(arg, key):
        v = ccfg['fitting'].get(key) if arg is None else (None if arg == 'off' else float(arg))
        return None if v is None else float(v)
    keep_wide = (bool(ccfg['fitting'].get('keep_wide', False)) if a.keep_wide is None
                 else a.keep_wide == 'on')
    scale_tol = opt(a.scale_tol, 'scale_tolerance')
    widening = (bool(ccfg['fitting'].get('widening_cut', False)) if a.widening is None
                else a.widening == 'on')
    print(f'detector: weights={a.weights or cc.get("weights") or "(default CULane)"} mode={mode} '
          f'cut_frac={cut_frac} conf={cc["conf_threshold"]} refine={refine} '
          f'keep_wide={keep_wide} scale_tol={scale_tol} widening={widening}')

    out = a.out_dir / f'pred_{a.tag}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    dump = []
    with open(out, 'w', encoding='utf-8') as fo:
        for seg_dir in sorted(p for p in a.root.iterdir() if p.is_dir()):
            H, W = cfg['input']['resize_size']
            cam = camera_from_metadata(seg_dir, (H, W))
            if cam is None:
                raise SystemExit(f'{seg_dir} 沒有 metadata.json 的相機參數')
            f_x, f_y, h = cam['f_x'], cam['f_y'], cam['camera_height']
            cal = width_calibrator(ccfg['fitting'].get('width_filter', 'median'), f_x, f_y, H, h)
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
                                           cut_frac=cut_frac, detector_mode=mode, tail='keep',
                                           refine=refine, nearfield_source='paint', ego_guard=True,
                                           keep_wide=keep_wide, scale_tolerance=scale_tol,
                                           widening_cut=widening,
                                           max_depth_m=a.max_depth,
                                           samples_per_meter=lf.get('samples_per_meter'),
                                           method=pe.get('method', 'windowed'),
                                           w_real_calibrator=cal,
                                           last_resort_lane_width=ccfg['fitting']['last_resort_lane_width'],
                                           return_debug=True)
                    rec['status'] = res['w_real_status']
                    rec['refined'] = res.get('refined', False)
                    # 寬度從哪來、沿用多久、選線檢查結果——分析輸出幀的誤差用
                    rec.update(w_used=res['w_real_used'], w_reason=res['w_real_reason'],
                               hold_frames=res['w_real_hold_frames'], hold_m=res['w_real_hold_m'],
                               guard=res['ego_guard'], scale=res.get('scale_check'),
                               pair_w=res.get('pair_width'), widen_cue=res.get('widening_cue'),
                               widen_cut=res.get('widening_cut_m'))
                    est = cal.last_estimate
                    if est is not None:          # 這幀自己的近場量測（沒過門檻也記）
                        rec.update(est_w=est['w_real_z0'], est_theta0=est['theta0_deg'],
                                   est_resid=est['resid_mad'], est_ok=est['quality_ok'])
                    rec['lanes'] = lanes_3d(res, f_x, W)
                    if a.dump:
                        dbg = res['debug']
                        cv = lambda c: None if c is None else {k: np.asarray(c[k]) for k in ('y', 'x', 'src') if k in c}
                        dump.append(dict(segment=seg_dir.name, frame_id=int(r.frame_id),
                                         dist=float(r.collect_dist_m), status=rec['status'],
                                         w_used=res['w_real_used'], guard=res['ego_guard'],
                                         lanes=[np.asarray(l, np.float32) for l in dbg['lanes']],
                                         score=np.asarray(dbg['lane_score'], np.float32),
                                         left=cv(dbg['left_curve']), right=cv(dbg['right_curve']),
                                         cam=dict(f_x=f_x, f_y=f_y, h=h, H=H, W=W)))
                except Exception as e:
                    rec['status'] = f'error:{type(e).__name__}:{e}'[:120]
                fo.write(json.dumps(rec) + '\n')
                if n % 250 == 0:
                    print(f'{n} {seg_dir.name[:30]} f{r.frame_id}', flush=True)
            if a.limit and n >= a.limit:
                break
    if a.dump:
        import pickle
        with open(a.out_dir / f'dump_{a.tag}.pkl', 'wb') as fo:
            pickle.dump(dict(frames=dump, num_samples=lf['num_samples'],
                             samples_per_meter=lf.get('samples_per_meter'),
                             method=pe.get('method', 'windowed')), fo)
    print(f'{n} frames -> {out}')


if __name__ == '__main__':
    main()
