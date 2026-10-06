"""infer_one_clrnet end to end with a stub detector: no weights, no GPU."""

import numpy as np
from PIL import Image

from libs.inference.paint_evidence import _STRIPE_M
from libs.inference.pipeline_clrnet import infer_one_clrnet
from tests import synthetic as syn


def _no_lanes(rgb, f_x, f_y, cut, mode):
    return [], []


def test_debug_output_on_a_frame_without_pitch(tmp_path):
    """return_debug on a frame with no pitch output must still return: the
    width rows it reports are only computed when there is an output."""
    img = tmp_path / "blank.png"
    Image.new("RGB", (syn.IMG_W, syn.IMG_H), (syn.ROAD_GRAY,) * 3).save(img)
    res = infer_one_clrnet(_no_lanes, str(img), (syn.IMG_H, syn.IMG_W), 50,
                           syn.F_X, syn.F_Y, syn.CAM_H, cut_frac=0.0,
                           return_debug=True)
    assert res["pitch_curve"]["pitch_at"] is None
    assert res["w_real_status"] == "no_anchor"
    assert res["debug"]["n_width_samples"] == 0


def _lanes(w_real):
    """Ego line centres of a flat road whose lane is w_real wide (centre to centre)."""
    y = np.arange(300.0, syn.IMG_H, 1.0)
    z = syn.F_Y * syn.CAM_H / (y - syn.CY)
    half = syn.F_X * w_real / 2.0 / z
    return [np.column_stack([syn.CX - half, y]), np.column_stack([syn.CX + half, y])]


def _painted(tmp_path, lanes, name):
    """Road image with a real-width stripe under every lane, saved as PNG."""
    img = np.full((syn.IMG_H, syn.IMG_W, 3), syn.ROAD_GRAY, dtype=np.uint8)
    cols = np.arange(syn.IMG_W)
    for ln in lanes:
        for x, y in ln:
            half = syn.F_X * _STRIPE_M / (syn.F_Y * syn.CAM_H / (y - syn.CY)) / 2.0
            img[int(y), (cols >= x - half) & (cols <= x + half)] = syn.PAINT_GRAY
    p = tmp_path / name
    Image.fromarray(img).save(p)
    return str(p)


def _run_refine(path, lanes, refine):
    detector = lambda rgb, f_x, f_y, cut, mode: (lanes, [0.9] * len(lanes))
    return infer_one_clrnet(detector, path, (syn.IMG_H, syn.IMG_W), 50,
                            syn.F_X, syn.F_Y, syn.CAM_H, cut_frac=0.0,
                            refine=refine, return_debug=True)


def test_solid_refine_snaps_only_when_both_lines_are_solid(tmp_path):
    """refine="solid": two continuous painted lines -> snap to the stripe
    centre; a dashed line on either side -> keep the detector positions."""
    left, right = _lanes(3.25)
    off = [ln + np.array([1.5, 0.0]) for ln in (left, right)]      # detector 1.5 px off
    solid = _painted(tmp_path, [left, right], "solid.png")
    res = _run_refine(solid, off, "solid")
    assert res["refined"]
    c = res["debug"]["left_curve"]
    near = c["y"] > syn.IMG_H - 60
    x_paint = np.interp(c["y"][near], left[:, 1], left[:, 0])
    assert np.abs(c["x"][near] - x_paint).mean() < 0.5

    dashed_left = left[(left[:, 1] // 20) % 2 == 0]                 # every other 20-row block
    dashed = _painted(tmp_path, [dashed_left, right], "dashed.png")
    res = _run_refine(dashed, off, "solid")
    assert not res["refined"]
    c = res["debug"]["left_curve"]
    assert np.allclose(c["x"], np.interp(c["y"], off[0][:, 1], off[0][:, 0]), atol=1e-6)
