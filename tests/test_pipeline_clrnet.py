"""infer_one_clrnet end to end with a stub detector: no weights, no GPU."""

from PIL import Image

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
