"""Where the lane ahead starts to widen (or narrow), so the pitch output can stop there (WWH-34).

The metric stage turns pixel width into depth with ONE lane width, measured in the
near field. Where the lane ahead widens — a turn bay, a merge — the lines spread
apart and the stage reads it as the road rising. Width alone cannot tell the two
apart (Dickmanns' 4-D approach notes the same limit); a different cue is needed:

  B  a neighbour lane's line. A grade change scales every lane's pixel width by
     the same factor at a row (same depth), so ego width / neighbour width stays
     put; only the ego lane changing its own width moves the ratio.
  A  when no neighbour line is seen: the two ego lines' pixel distances to their
     vanishing point, (u_vp − u_L) / (u_R − u_vp) = X_L / X_R, the ratio of the
     lines' lateral offsets — the depth cancels, so a grade change leaves it
     alone; a one-sided widening moves it. The vanishing point is where the two
     lines' straight near-field (8–15 m) fits meet, which also removes heading.
     With two lines a one-sided widening can be imitated exactly by a grade
     change plus a bend, so A also fires on bends: the gate drops a change where
     both lines move the same way by similar amounts (a bend), keeping the
     one-sided (widening) and opposite (grade, which A ignores anyway) patterns.

The change row is the first where the ratio leaves its near-field reference by
more than the threshold over RUN_M metres of flat-ground depth.

Measured offline on the development data (OpenLane up&down + two random hold-out
sets now used for development, CARLA): the cut points are 2–12 times worse than
the points kept at the same distance on the hold-out sets, ~1.2 on up&down,
which has few widenings; A fires on 0 % of CARLA's slope routes (grade never
reads as widening) and, with the gate, 2.8 % of Town05's bends (19 % without).
"""

import numpy as np

# B: a neighbour line is one whose lateral offset from the ego line, at flat-ground
# depth over the shared rows, is a lane width — the project's ego-lane range.
NB_MIN_M, NB_MAX_M = 2.5, 4.5
# B compares ratios over these flat-ground depths (closer, the neighbour line is
# rarely in view) and takes its reference over the nearest REF_M of them.
B_Z_LO, B_Z_HI, B_REF_M = 10.0, 45.0, 5.0
# A: vanishing point from the lines' straight fits over this near-field depth range,
# changes looked for beyond it.
A_NEAR_LO, A_NEAR_HI, A_FAR_HI = 8.0, 15.0, 45.0
# Both: a change must hold over this many metres of flat-ground depth.
RUN_M = 3.0
# Gate: a change where both lines move the same way and the smaller move is at
# least this share of the larger is a bend, not a widening.
BEND_SHARE = 0.5


def _x_at(c, ys):
    ys = np.asarray(ys, float)
    v = np.interp(ys, c["y"], c["x"])
    v[(ys < c["y"][0]) | (ys > c["y"][-1])] = np.nan
    return v


def _flat_depth(ys, f_y, h, H):
    return f_y * h / (np.asarray(ys, float) - H / 2.0)


def _curve(ln):
    p = np.asarray(ln, float)
    o = np.argsort(p[:, 1])
    return {"y": p[o, 1], "x": p[o, 0]}


def _first_run(z, dev, tail_half=False):
    """Index of the first depth where `dev` holds over RUN_M (z ascending), or None.
    tail_half: a change still running at the farthest row counts from RUN_M / 2 (B)."""
    for i in np.where(dev)[0]:
        span = z[i:][~dev[i:]]
        end = span[0] if len(span) else z[-1] + 1e-6
        if end - z[i] >= RUN_M or (tail_half and not len(span) and z[-1] - z[i] >= RUN_M / 2):
            return int(i)
    return None


def _b_rows(f_y, h, H):
    ys = np.arange(H / 2.0 + 2, H, 1.0)
    z = _flat_depth(ys, f_y, h, H)
    k = (z >= B_Z_LO) & (z <= B_Z_HI)
    return ys[k], z[k]


def _neighbour(lanes, ref, side, others, f_x, f_y, h, H):
    """Detector line one lane beyond `ref` on `side` (-1 left, +1 right), or None."""
    ys, z = _b_rows(f_y, h, H)
    xr = _x_at(ref, ys)
    taken = [np.nanmedian(_x_at(e, ys)) for e in others]
    best, best_off = None, np.inf
    for ln in lanes:
        if len(ln) < 2:
            continue
        c = _curve(ln)
        xc = _x_at(c, ys)
        med = np.nanmedian(xc) if np.isfinite(xc).any() else np.nan
        if any(np.isfinite(t) and abs(med - t) <= 3.0 for t in taken):
            continue                                    # an ego line itself
        off = side * (xc - xr) * z / f_x
        k = np.isfinite(off)
        if k.sum() < 10:
            continue
        m = float(np.median(off[k]))
        if NB_MIN_M <= m <= NB_MAX_M and m < best_off:
            best, best_off = c, m
    return best


def _b_change(left, right, lanes, f_x, f_y, h, H, thr):
    """(neighbour seen?, change row or None) from the ego/neighbour width ratio."""
    best = None
    for side in (-1, 1):
        nb = _neighbour(lanes, right if side > 0 else left, side, (left, right), f_x, f_y, h, H)
        if nb is None:
            continue
        ys, z = _b_rows(f_y, h, H)
        xl, xr, xn = _x_at(left, ys), _x_at(right, ys), _x_at(nb, ys)
        ego = xr - xl
        nbw = (xn - xr) if side > 0 else (xl - xn)
        k = np.isfinite(ego) & np.isfinite(nbw) & (ego > 2) & (nbw > 2)
        ys, z, r = ys[k], z[k], ego[k] / nbw[k]
        if len(z) < 10:
            continue
        o = np.argsort(z)
        ys, z, r = ys[o], z[o], r[o]
        ref = z <= z[0] + B_REF_M
        if ref.sum() < 5:
            continue
        i = _first_run(z, np.abs(r / np.median(r[ref]) - 1) > thr, tail_half=True)
        cand = (len(z), None if i is None else float(ys[i]))
        if best is None or cand[0] > best[0]:
            best = cand
    return best is not None, (best[1] if best else None)


def _a_change(left, right, f_y, h, H, thr):
    """Change row from the vanishing-point distance ratio, bends gated out; or None."""
    ys = np.arange(np.ceil(max(left["y"][0], right["y"][0])),
                   np.floor(min(left["y"][-1], right["y"][-1])) + 1)
    ys = ys[ys > H / 2.0 + 2]
    if len(ys) < 10:
        return None
    z = _flat_depth(ys, f_y, h, H)
    uL, uR = _x_at(left, ys), _x_at(right, ys)
    near = (z >= A_NEAR_LO) & (z <= A_NEAR_HI)
    if near.sum() < 5:
        return None
    aL, bL = np.polyfit(ys[near], uL[near], 1)
    aR, bR = np.polyfit(ys[near], uR[near], 1)
    if abs(aL - aR) < 1e-9:
        return None
    v_vp = (bR - bL) / (aL - aR)
    u_vp = aL * v_vp + bL
    rho = (u_vp - uL) / (uR - u_vp)
    ok = rho > 0
    if (near & ok).sum() < 5:
        return None
    r0 = np.median(rho[near & ok])
    far = (z > A_NEAR_HI) & (z <= A_FAR_HI) & ok
    if far.sum() < 5:
        return None
    o = np.argsort(z[far])
    zf, yf = z[far][o], ys[far][o]
    i = _first_run(zf, np.abs(np.log(rho[far][o] / r0)) > thr)
    if i is None:
        return None
    yy = yf[(zf >= zf[i]) & (zf <= zf[i] + RUN_M)]
    dL = np.median(_x_at(left, yy) - (aL * yy + bL))
    dR = np.median(_x_at(right, yy) - (aR * yy + bR))
    if dL * dR > 0 and min(abs(dL), abs(dR)) >= BEND_SHARE * max(abs(dL), abs(dR)):
        return None                                      # both lines bend the same way
    return float(yf[i])


def widening_row(left, right, lanes, f_x, f_y, camera_height, image_height,
                 thr_b=0.10, thr_a=0.15):
    """(image row where the ego lane starts to change width, "B" / "A"), or (None, cue).

    left, right : the ego lines as {"y" ascending, "x"} curves
    lanes       : every detector line, (N, 2) arrays of (x, y) — neighbours come from here
    B is used when a neighbour line is found on either side, A otherwise.
    """
    if left is None or right is None:
        return None, None
    seen, row = _b_change(left, right, lanes, f_x, f_y, camera_height, image_height, thr_b)
    if seen:
        return row, "B"
    return _a_change(left, right, f_y, camera_height, image_height, thr_a), "A"
