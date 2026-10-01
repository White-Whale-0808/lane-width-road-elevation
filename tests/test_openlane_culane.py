"""OpenLane → CULane label conversion (pure functions, no dataset needed)."""

import numpy as np

from openlane_module.convert_openlane_culane import (EXTEND_STEP_PX, frame_kind, lanes_to_culane,
                                                     lines_txt, mask_png)

W, H = 1920, 1280


def _lane(attr, us, vs, category=1):
    return {'attribute': attr, 'category': category, 'uv': [list(us), list(vs)]}


def test_only_attribute_slots_kept_and_curbs_dropped():
    lanes = lanes_to_culane([
        _lane(2, [800, 900, 1000], [1200, 1000, 800]),
        _lane(0, [100, 200, 300], [1200, 1000, 800]),                 # 5th lane
        _lane(0, [1500, 1600, 1700], [1200, 1000, 800], category=21),  # curb
    ], W, H)
    assert list(lanes) == [2]


def test_points_outside_image_and_duplicates_removed_bottom_first():
    lanes = lanes_to_culane([_lane(3, [10, 10, -5, 20, 30], [700, 700, 900, 1300, 600])], W, H)
    pts = lanes[3]
    assert len(pts) == 2                       # (-5, 900) and (20, 1300) out, dup merged
    assert pts[0, 1] > pts[-1, 1]              # CULane: bottom of the image first


def test_short_lane_dropped():
    assert lanes_to_culane([_lane(1, [5, 5], [700, 700])], W, H) == {}


def test_extend_bottom_continues_the_straight_line():
    # x = 1000 - 0.5·(y - 800): a straight lane ending at y = 800
    vs = np.arange(500.0, 801.0, 10.0)
    lanes = lanes_to_culane([_lane(2, 1000 - 0.5 * (vs - 800), vs)], W, H, extend_bottom=True)
    pts = lanes[2]
    assert pts[0, 1] > H - 1 - EXTEND_STEP_PX            # reaches the bottom
    assert np.all(np.diff(pts[:, 1]) < 0)                # still bottom first
    assert np.allclose(pts[:, 0], 1000 - 0.5 * (pts[:, 1] - 800))


def test_extend_bottom_stops_at_image_border():
    vs = np.arange(500.0, 801.0, 10.0)                   # heads off the left edge
    pts = lanes_to_culane([_lane(1, 200 - 2.0 * (vs - 800), vs)], W, H,
                          extend_bottom=True)[1]
    assert pts[:, 0].min() >= 0 and pts[0, 1] < H - 1


def test_extend_bottom_off_by_default():
    vs = np.arange(500.0, 801.0, 10.0)
    pts = lanes_to_culane([_lane(2, 1000 - 0.5 * (vs - 800), vs)], W, H)[2]
    assert pts[0, 1] == 800


def _lane3d(attr, category, lateral_m, z0=5.0, z1=40.0):
    z = np.linspace(z0, z1, 20)
    return {'attribute': attr, 'category': category,
            'xyz': [list(z), [lateral_m] * 20, [0.0] * 20],
            'uv': [[900.0, 950.0], [1200.0, 900.0]]}


def test_frame_kind():
    assert frame_kind([], W, H) == 'no_lanes'
    assert frame_kind([_lane3d(2, 1, 1.7)], W, H) == 'labelled'
    assert frame_kind([_lane3d(0, 21, 1.7)], W, H) == 'curb_only'
    # an unlabelled paint line where an ego line would be
    assert frame_kind([_lane3d(0, 1, -1.6)], W, H) == 'ego_unlabelled'
    # unlabelled paint lines only far ahead or far to the side
    assert frame_kind([_lane3d(0, 1, 1.6, z0=29.0), _lane3d(0, 2, 6.0)], W, H) == 'far_paint'
    off = _lane3d(3, 1, 1.7); off['uv'] = [[-50.0, -10.0], [1200.0, 900.0]]
    assert frame_kind([off], W, H) == 'out_of_image'


def test_lines_txt_and_mask_values():
    lanes = lanes_to_culane([_lane(4, [1000, 1100], [1200, 900]),
                             _lane(1, [300, 400], [1200, 900])], W, H)
    txt = lines_txt(lanes).splitlines()
    assert len(txt) == 2 and txt[0].startswith('300.00 1200.00')   # slot order 1, 4
    m = mask_png(lanes, W, H)
    assert set(np.unique(m)) == {0, 1, 4}
    assert m[1050, 350] == 1 and m[1050, 1050] == 4
