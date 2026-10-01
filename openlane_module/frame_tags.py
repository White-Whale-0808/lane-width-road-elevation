"""逐幀變因表：天氣/時段/地景、OpenLane 六個 case tag（逐幀）、GT 實測曲率、
標線組合、GT 車道寬、60 m 起伏。同時統計轉換器各道篩選踢掉多少幀。

順便產出 **Tier A 幀清單**（`--tier-a-out`）。Tier A 是「乾淨集合」：晴天白天、
兩側實線、非路口、幾乎直線 —— 用來在把變因壓到最低的條件下量車道寬準確率。
`gt_width_noise.py` / `measure_paint_inset.py` / `nearfield_feasibility.py` 都吃它。
在此之前那個 CSV 沒有任何程式產生，是個孤兒檔。

⚠ **坡度不是篩選，是分層標籤**（2026-09-12）。`--grade-min` 預設 0，也就是
全收；`grade_60m` 欄位留著讓下游自己分層。理由：平路的幀是**陰性對照** ——
在沒有起伏的地方報出起伏就是假陽性，把它們篩掉等於把「會不會無中生有」這個
問題從資料裡刪掉。實測 10,134 幀裡有 49% 起伏不到 0.3 m，而且平路幀與起伏幀
大量共存在同一段內，所以分層比較不會被場景混淆。

用法::

    python -m openlane_module.frame_tags                    # 全收（預設）
    python -m openlane_module.frame_tags --grade-min 0.3    # 只要有坡的（舊行為）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from openlane_module.convert_openlane import (inner_pair, PAINT, SOLID, CAT,  # noqa: E402
                                              WIDTH_MIN, WIDTH_MAX,
                                              GRADE_SPAN, PROFILE_MAX_M,
                                              PROFILE_STEP_M)

CURV_SPAN = 50.0        # 量 sagitta 的前視距離（m）
# Tier A：壓低變因的乾淨集合。sagitta 門檻用**實測值**而不是官方 curve tag，
# 因為實測值是連續的（官方 tag=1 的 sagitta 中位 0.542 m、tag=0 是 0.032 m，
# 對得上，但 tag 只有 0/1 無法調鬆緊）。
TIER_A_SAGITTA_MAX = 0.15


def load_cases(ol_root):
    """OpenLane 官方逐幀 case tag（curve / night / intersection / ...）。

    找不到就直接拋。空的 dict 會讓 `tier_a` 在存取 `df.tag_intersection` 時才
    掛掉 —— 那已經是掃完整個 validation split（約一萬幀、數分鐘）之後的事了。
    能在開始前查的就不要留到結束後炸。
    """
    case_dir = Path(ol_root) / "test"
    files = sorted(case_dir.glob("1000_*.txt"))
    if not files:
        raise FileNotFoundError(
            f"找不到官方 case tag 檔：{case_dir}/1000_*.txt。"
            " tier_a() 需要 tag_intersection / tag_curve / tag_night，"
            " 請確認 --openlane 指到的是解開後的 OpenLane 根目錄。")
    cases = {}
    for p in files:
        tag = p.stem.replace("1000_", "").replace("_case", "")
        s = set()
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                parts = line.split("/")
                s.add((parts[1], Path(parts[2]).stem))
        cases[tag] = s
    return cases


def curvature(xl, xr, z0):
    """50 m 內車道中心線偏離弦線的最大橫向量（sagitta, m）與航向變化（度）。

    ``xl`` / ``xr`` 是 inner_pair 回傳、已依前向排序的 xyz；``z0`` 為兩線共同起點。
    共同可見不足 10 m 時回 (nan, nan)。
    """
    c1 = min(xl[0].max(), xr[0].max(), z0 + CURV_SPAN)
    if c1 - z0 < 10.0:
        return np.nan, np.nan
    x = np.arange(z0, c1, 0.5)
    yc = 0.5 * (np.interp(x, xl[0], xl[1]) + np.interp(x, xr[0], xr[1]))
    chord = np.interp(x, [x[0], x[-1]], [yc[0], yc[-1]])
    sag = float(np.abs(yc - chord).max())
    co = np.polyfit(x, yc, 2)
    head = float(np.degrees(np.arctan(np.polyval(np.polyder(co), x[-1]))
                            - np.arctan(np.polyval(np.polyder(co), x[0]))))
    return sag, head


def scan(ol_root, grade_min=0.0, quiet=False):
    """掃 validation split，回傳 (逐幀表, 篩選漏斗)。"""
    ol_root = Path(ol_root)
    scene = json.load(open(ol_root / "scene/scene.json", encoding="utf-8"))
    cases = load_cases(ol_root)
    if not quiet:
        print("case tags:", {k: len(v) for k, v in cases.items()})

    reject = dict(no_pair=0, not_paint=0, width=0, short_span=0, flat=0, ok=0)
    rows = []
    segs = sorted((ol_root / "validation").glob("segment-*"))
    for i, seg in enumerate(segs):
        if not quiet and i % 40 == 0:
            print(f"  [{i}/{len(segs)}]", flush=True)
        for jf in sorted(seg.glob("*.json")):
            if jf.name.startswith("._"):
                continue
            data = json.load(open(jf, encoding="utf-8"))
            L, R = inner_pair(data["lane_lines"])
            if not (L and R):
                reject["no_pair"] += 1; continue
            if L[1] not in PAINT or R[1] not in PAINT:
                reject["not_paint"] += 1; continue
            width = L[0] - R[0]
            xl, xr = L[2], R[2]
            z0 = max(xl[0].min(), xr[0].min())
            z1 = min(xl[0].max(), xr[0].max(), PROFILE_MAX_M)
            if not (WIDTH_MIN <= width <= WIDTH_MAX):
                reject["width"] += 1; continue
            if z1 - z0 < 5.0:
                reject["short_span"] += 1; continue
            depth = np.arange(z0, z1 + 1e-9, PROFILE_STEP_M)
            h = 0.5 * (np.interp(depth, xl[0], xl[2]) + np.interp(depth, xr[0], xr[2]))
            near = depth <= z0 + GRADE_SPAN
            grade = float(h[near].max() - h[near].min())
            if grade < grade_min:
                reject["flat"] += 1; continue
            reject["ok"] += 1

            sag, head = curvature(xl, xr, z0)
            key = (seg.name, jf.stem)
            sc = scene.get(seg.name, {})
            rows.append(dict(
                segment=seg.name, frame=jf.stem,
                weather=sc.get("weather", "?"), hours=sc.get("hours", "?"),
                scene=sc.get("scene", "?"),
                **{f"tag_{t}": int(key in s) for t, s in sorted(cases.items())},
                left_cat=CAT.get(L[1], L[1]), right_cat=CAT.get(R[1], R[1]),
                both_solid=int(L[1] in SOLID and R[1] in SOLID),
                w_gt=round(width, 4), sagitta=round(sag, 4) if sag == sag else np.nan,
                heading_deg=round(head, 3) if head == head else np.nan,
                z_gt_min=round(float(z0), 2), z_gt_max=round(float(z1), 2),
                grade_60m=round(grade, 4)))
    return pd.DataFrame(rows), reject


def tier_a(df, sagitta_max=TIER_A_SAGITTA_MAX):
    """乾淨集合：晴天白天 ＋ 兩側實線 ＋ 非路口 ＋ 幾乎直線。

    ⚠ 不含坡度條件。Tier A 是用來量**車道寬**的，而量寬度不需要坡度 ——
    先前它繼承了母體的 `grade >= 0.3`，等於在一個與待測量無關的條件上自我
    設限（308 幀；放掉之後是 488 幀 / 16 段，且沒有任何一段掉出來）。
    """
    return df[(df.weather == "Clear") & (df.hours == "Daytime")
              & (df.both_solid == 1) & (df.tag_intersection == 0)
              & (df.sagitta <= sagitta_max)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openlane", type=Path, default=Path("D:/datasets/openlane"))
    ap.add_argument("--grade-min", type=float, default=0.0,
                    help="60 m 起伏下限（m）。預設 0＝全收；坡度是分層標籤不是篩選")
    ap.add_argument("--out", type=Path,
                    default=REPO / "debug/outputs/openlane_frame_tags.csv")
    ap.add_argument("--tier-a-out", type=Path,
                    default=REPO / "debug/outputs/tierA_frames.csv")
    args = ap.parse_args(argv)

    df, reject = scan(args.openlane, args.grade_min)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n篩選漏斗:", reject)
    print(f"{len(df)} frames / {df.segment.nunique()} segments -> {args.out}")

    ta = tier_a(df)
    # mkdir before the write, not before the scan: a missing folder used to
    # surface only here, after the whole OpenLane pass had already run
    args.tier_a_out.parent.mkdir(parents=True, exist_ok=True)
    ta.to_csv(args.tier_a_out, index=False)
    print(f"Tier A: {len(ta)} frames / {ta.segment.nunique()} segments "
          f"-> {args.tier_a_out}")
    if len(ta):
        print("  起伏分層: 平路<0.1 %d、邊緣0.1-0.3 %d、有坡>=0.3 %d"
              % ((ta.grade_60m < .1).sum(),
                 ((ta.grade_60m >= .1) & (ta.grade_60m < .3)).sum(),
                 (ta.grade_60m >= .3).sum()))


if __name__ == "__main__":
    main()
