"""Drop-in for the official OpenLane MinCostFlow.py (which needs the removed
ortools.graph.pywrapgraph API). Not part of the upstream code — see
openlane_module/official_eval/__init__.py.

The official solver sends min(#non-empty rows, #non-empty cols) units of flow
through a complete bipartite graph with unit capacities, i.e. it finds the
maximum-cardinality assignment of minimum total cost. With every adj entry 1
(as eval_3D_lane.bench sets it) that is exactly scipy's rectangular
linear_sum_assignment. Returns [[gt_i, pred_j, cost], ...] like the original.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def SolveMinCostFlow(adj_mat, cost_mat):
    adj_mat, cost_mat = np.asarray(adj_mat), np.asarray(cost_mat)
    if adj_mat.size == 0 or not adj_mat.any():
        return []
    big = cost_mat.max() * 10 + 10**6
    c = np.where(adj_mat > 0, cost_mat, big)
    rows, cols = linear_sum_assignment(c)
    return [[int(i), int(j), int(cost_mat[i, j])] for i, j in zip(rows, cols) if adj_mat[i, j] > 0]
