"""Dataset camera from metadata.json: both formats, and when the config camera is kept."""

import json

import pytest

from libs.dataset_camera import camera_from_metadata, resolve_camera

CONFIG = {"f_x": 512, "f_y": 455, "camera_height": 1.08, "camera_forward_offset": 1.5}
RESIZE = (512, 1024)


def _write(tmp_path, meta):
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_no_metadata_uses_config(tmp_path):
    cam, src = resolve_camera(CONFIG, tmp_path, RESIZE)
    assert src == "config" and cam["f_x"] == 512


def test_carla_format_is_scaled_to_the_resized_image(tmp_path):
    d = _write(tmp_path, {"camera": {"fov_deg": 90.0, "img_width": 1280, "img_height": 720,
                                     "z_height_m": 1.08, "x_forward_m": 1.5}})
    cam = camera_from_metadata(d, RESIZE)
    assert cam["f_x"] == pytest.approx(512.0) and cam["f_y"] == pytest.approx(640 * 512 / 720)


def test_matching_dataset_keeps_config_values_exactly(tmp_path):
    # the original CARLA sets: f_y 455.11 vs the config's 455 — results must not move
    d = _write(tmp_path, {"camera": {"fov_deg": 90.0, "img_width": 1280, "img_height": 720,
                                     "z_height_m": 1.08, "x_forward_m": 1.5}})
    cam, src = resolve_camera(CONFIG, d, RESIZE)
    assert src.startswith("config") and cam["f_y"] == 455.0


def test_different_camera_switches_to_metadata(tmp_path):
    d = _write(tmp_path, {"f_x": 1104.7467, "f_y": 828.56, "camera_height": 2.116,
                          "camera_forward_offset": 0.0, "resize_size": [512, 1024]})
    cam, src = resolve_camera(CONFIG, d, RESIZE)
    assert src == "metadata.json" and cam["camera_height"] == 2.116 and cam["camera_forward_offset"] == 0.0


def test_openlane_format_for_another_resize_is_rejected(tmp_path):
    d = _write(tmp_path, {"f_x": 1104.7, "f_y": 828.6, "camera_height": 2.116, "resize_size": [256, 512]})
    with pytest.raises(ValueError):
        camera_from_metadata(d, RESIZE)
