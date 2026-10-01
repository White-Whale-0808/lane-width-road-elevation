"""Camera of a dataset directory, in the pipeline's resized image.

Datasets carry their own camera in metadata.json, in one of two formats:

  OpenLane converter   f_x, f_y, camera_height, camera_forward_offset, resize_size —
                       already the virtual camera of the resized image
  CARLA collector      camera: {fov_deg, img_width, img_height, z_height_m, x_forward_m}
                       — square pixels at capture, resized non-uniformly to the
                       pipeline size, so f_x = f·W/img_width and f_y = f·H/img_height

The config's pitch_estimation block holds one camera (the original CARLA set).
`resolve_camera` keeps it for datasets that agree with it and switches to the
dataset's own camera when it differs, so runners never apply one camera's
geometry to another camera's images.
"""
import json
import math
from pathlib import Path

KEYS = ("f_x", "f_y", "camera_height", "camera_forward_offset")


def camera_from_metadata(dataset_dir, resize_size):
    """{f_x, f_y, camera_height, camera_forward_offset} from dataset_dir/metadata.json,
    or None when the file is missing or describes no camera. resize_size is
    [height, width] of the pipeline image."""
    p = Path(dataset_dir) / "metadata.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text(encoding="utf-8"))
    H, W = resize_size
    if "f_x" in meta:
        if list(meta.get("resize_size", resize_size)) != list(resize_size):
            raise ValueError(f"{p}: intrinsics are for resize_size {meta['resize_size']}, "
                             f"the pipeline uses {list(resize_size)}")
        return {"f_x": float(meta["f_x"]), "f_y": float(meta["f_y"]),
                "camera_height": float(meta["camera_height"]),
                "camera_forward_offset": float(meta.get("camera_forward_offset", 0.0))}
    cam = meta.get("camera")
    if not cam:
        return None
    f = cam["img_width"] / (2.0 * math.tan(math.radians(cam["fov_deg"]) / 2.0))
    return {"f_x": f * W / cam["img_width"], "f_y": f * H / cam["img_height"],
            "camera_height": float(cam["z_height_m"]),
            "camera_forward_offset": float(cam.get("x_forward_m", 0.0))}


def resolve_camera(pitch_cfg, dataset_dir, resize_size, rel_tol=0.01, offset_tol_m=0.05):
    """(camera dict, source) for a dataset.

    The config camera, unless the dataset's metadata describes a different one
    (f_x, f_y or camera_height off by more than rel_tol, or the forward offset
    by more than offset_tol_m) — then the metadata camera. A dataset that agrees
    with the config keeps the config values exactly, so its results do not move
    (the original CARLA sets agree to 0.02 %: f_y 455 vs 455.11).
    source is "config", "config (matches metadata.json)" or "metadata.json".
    """
    conf = {"f_x": float(pitch_cfg["f_x"]), "f_y": float(pitch_cfg["f_y"]),
            "camera_height": float(pitch_cfg["camera_height"]),
            "camera_forward_offset": float(pitch_cfg.get("camera_forward_offset", 0.0))}
    meta = camera_from_metadata(dataset_dir, resize_size)
    if meta is None:
        return conf, "config"
    differs = any(abs(meta[k] - conf[k]) > rel_tol * abs(conf[k]) for k in KEYS[:3]) \
        or abs(meta["camera_forward_offset"] - conf["camera_forward_offset"]) > offset_tol_m
    return (meta, "metadata.json") if differs else (conf, "config (matches metadata.json)")


def describe(cam, source):
    return (f"camera ({source}): f_x={cam['f_x']:.2f} f_y={cam['f_y']:.2f} "
            f"h={cam['camera_height']:.3f} m forward_offset={cam['camera_forward_offset']:.2f} m")
