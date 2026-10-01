import numpy as np

"""
Shared flat-ground pinhole projection model, used by lane_segmentation
(tracking thresholds) and paint_evidence / model_lane_fitting (probe windows).
"""

# Grade-uncertainty constants (used by z_min / lane_px_max bounds)
_MAX_GRADE_DEG   = 15.0   # worst-case road grade the thresholds must absorb
_GRADE_RAMP_Z0   = 6.0    # within 6 m the road is the ego plane: no grade slack
_GRADE_RAMP_SPAN = 6.0    # grade slack ramps to full between 6 m and 12 m


class CameraGeometry:
    """Projection helpers.

    z_at(y) assumes a flat road. The road may however be tilted by up to
    ±_MAX_GRADE_DEG: a plane of grade θ passing under the camera satisfies
        y - cy = f_y*h/z - f_y*tanθ   =>   z = f_y*h / ((y-cy) + f_y*tanθ)
    so z_min(y) (taken at +θ_max, uphill) lower-bounds the true depth, and is
    also finite for rows ABOVE the flat horizon — exactly where uphill road
    appears. Pixel windows derived with z_min are valid for any grade within
    the bound.
    """

    def __init__(self, f_x, f_y, camera_height, w_real,
                 image_width, img_height, min_y_margin=0.05):
        self.f_x = float(f_x)
        self.f_y = float(f_y)
        self.h = float(camera_height)
        self.w = float(w_real) if w_real is not None else None
        self.cx = image_width / 2.0
        self.cy = img_height / 2.0
        self.min_dy = min_y_margin * img_height  # clamp near/above horizon
        self.grade_dy = self.f_y * np.tan(np.radians(_MAX_GRADE_DEG))

    def z_at(self, y):
        """Flat-ground (nominal) depth of image row y, clamped at the horizon."""
        return self.f_y * self.h / max(y - self.cy, self.min_dy)

    def z_min(self, y):
        """Lower depth bound at row y over grades within ±_MAX_GRADE_DEG.

        The ego vehicle stands on the near plane, so rows closer than
        _GRADE_RAMP_Z0 carry no grade uncertainty; the slack ramps in
        linearly and is fully active beyond _GRADE_RAMP_Z0 + _GRADE_RAMP_SPAN
        (where a second, tilted plane may appear).
        """
        z_flat = self.z_at(y)
        w = min(max((z_flat - _GRADE_RAMP_Z0) / _GRADE_RAMP_SPAN, 0.0), 1.0)
        return self.f_y * self.h / max(y - self.cy + w * self.grade_dy, self.min_dy)

    @classmethod
    def without_lane_width(cls, f_x, f_y, camera_height,
                           image_width, img_height, min_y_margin=0.05):
        """Geometry for callers that only need depth (z_at / z_min / z_valid).

        Those are f_y*h/(y-cy) — the lane width never enters. lane_px and
        lane_px_max are the only members that need it, and they raise here.
        Constructing without it keeps the fact visible at the call site
        instead of passing a value that is silently never read.
        """
        return cls(f_x, f_y, camera_height, None,
                   image_width, img_height, min_y_margin)

    def _require_w(self):
        if self.w is None:
            raise ValueError(
                "lane width was not supplied to CameraGeometry "
                "(built via without_lane_width); lane_px/lane_px_max need it")
        return self.w

    def px_at(self, x_m, y):
        """Pixel offset of a LATERAL distance x_m (metres) at row y, flat road.

        x - cx = f_x*X/z, so every threshold that is really "a distance to
        the side of the camera axis" has an exact pixel form at each row.
        This is the primitive; lane_px below is just the case X = one lane.
        """
        return self.f_x * x_m / self.z_at(y)

    def px_max_at(self, x_m, y):
        """Upper bound of that offset at row y (worst-case uphill grade)."""
        return self.f_x * x_m / self.z_min(y)

    def lane_px(self, y):
        """Nominal (flat-ground) pixel width of one real lane at row y."""
        return self.px_at(self._require_w(), y)

    def lane_px_max(self, y):
        """Upper bound of the lane pixel width at row y (worst-case uphill)."""
        return self.px_max_at(self._require_w(), y)

    def z_valid(self, y):
        """True when flat-ground z(y) is not saturated by the horizon clamp."""
        return (y - self.cy) > self.min_dy
