"""Z-error evaluation against the official OpenLane scorer (pure functions, no dataset needed)."""

import numpy as np
from scipy.spatial.transform import Rotation

from openlane_module.official_eval.utils.MinCostFlow import SolveMinCostFlow
from openlane_module.zerror_eval import (gt_to_ground, ground_extrinsic, make_evaluator, point_errors,
                                         to_ground)


def test_min_cost_flow_matches_google_example():
    # Google OR-tools assignment example the official MinCostFlow.py ships in its main(): optimum 265
    cost = np.array([90, 76, 75, 70, 35, 85, 55, 65, 125, 95, 90, 105, 45, 110, 95, 115]).reshape(4, 4)
    res = SolveMinCostFlow(np.ones((4, 4)), cost)
    assert len(res) == 4
    assert sum(c for _, _, c in res) == 265
    assert sorted(i for i, _, _ in res) == [0, 1, 2, 3] and sorted(j for _, j, _ in res) == [0, 1, 2, 3]


def test_min_cost_flow_rectangular_and_empty():
    cost = np.array([[5, 1, 9], [2, 8, 7]])
    res = SolveMinCostFlow(np.ones_like(cost), cost)
    assert sorted((i, j) for i, j, _ in res) == [(0, 1), (1, 0)]      # min(rows, cols) pairs, cost 3
    assert SolveMinCostFlow(np.ones((0, 2)), np.zeros((0, 2))) == []


def _extrinsic():
    E = np.eye(4)
    E[:3, :3] = Rotation.from_euler('xyz', [0.4, -1.1, 0.7], degrees=True).as_matrix()
    E[:3, 3] = [1.5, -0.02, 2.115]
    return E


def test_prediction_and_gt_share_one_ground_transform():
    # GT xyz is camera (forward, left, up); predictions are optical (right, down, forward).
    # The same physical point has to land on the same ground coordinates either way.
    E = ground_extrinsic(_extrinsic())
    fwd, left, up = np.array([9.0, 25.0]), np.array([1.6, -1.7]), np.array([-2.0, -1.4])
    g_gt = gt_to_ground(E, np.stack([fwd, left, up]))
    g_pred = to_ground(E, np.stack([-left, -up, fwd], 1))
    np.testing.assert_allclose(g_gt[np.argsort(g_gt[:, 1])], g_pred, atol=1e-9)
    assert abs(E[2, 3] - 2.115) < 1e-12 and np.all(E[0:2, 3] == 0)   # official: x/y translation dropped


def _lane(x, y0, y1, z=lambda y: 0.02 * y):
    y = np.arange(y0, y1 + 1e-9, 1.0)
    return np.stack([np.full_like(y, x), y, z(y)], 1)


def test_point_errors_reconcile_with_official_bench():
    ev = make_evaluator()
    gt = [_lane(-1.6, 3, 90), _lane(1.7, 3, 70)]
    vis = [np.ones(len(g)) for g in gt]
    pred = [_lane(-1.5, 6, 45, lambda y: 0.02 * y + 0.05),       # 5 cm high, stops at 45 m
            _lane(1.8, 6, 45, lambda y: 0.021 * y - 0.03),
            _lane(8.0, 6, 45)]                                    # another lane, unmatched
    out = ev.bench(pred, [0] * len(pred), gt, vis, [1, 1], None, 2.1, 0, False, None)
    z_close = np.array(out[8])
    rows, gt_rows = point_errors(ev, pred, gt, vis, [2, 3])
    assert all(r['matched'] for r in gt_rows)
    for attr, official in zip((2, 3), z_close):
        ours = np.mean([r['z_err'] for r in rows if r['attr'] == attr and not r['far']])
        assert abs(ours - official) < 1e-12
    left_close = [r['z_err'] for r in rows if r['attr'] == 2 and not r['far']]
    np.testing.assert_allclose(left_close, 0.05, atol=1e-9)
