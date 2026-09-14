"""Sparse SR agrees with the dense definition, including directed/absorbing chains."""
import numpy as np
import pytest
from scipy import sparse
from successor_sparse import successor_matrix


@pytest.mark.parametrize('gamma', [0.1, 0.95, 0.999])
@pytest.mark.parametrize('block_size', [1, 7, 32])
def test_sparse_sr(gamma, block_size):
    rng = np.random.default_rng(7)
    P = rng.random((19, 19))
    P[P < .8] = 0
    P[np.arange(19), np.arange(19)] += 1
    P /= P.sum(axis=1, keepdims=True)
    P[-1] = 0
    P[-1, -1] = 1
    actual = successor_matrix(sparse.csr_matrix(P), gamma, block_size)
    np.testing.assert_allclose(actual, np.linalg.solve(np.eye(19)-gamma*P, np.eye(19)), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual.sum(axis=1), 1/(1-gamma), rtol=1e-11)


@pytest.mark.parametrize('algorithm', ['feas2', 'feas2s', 'feas_thresh_s'])
@pytest.mark.parametrize('round_decimals', [3, None])
def test_timing_preserves_solver_outputs(algorithm, round_decimals):
    from feas_two_phase import feas_iter_two_phase
    from feas_two_phase_sparse import feas_iter_two_phase_sparse
    from feas_thresh_sparse import feas_iter_two_phase_sparse_thresh
    solver = {'feas2': feas_iter_two_phase, 'feas2s': feas_iter_two_phase_sparse,
              'feas_thresh_s': feas_iter_two_phase_sparse_thresh}[algorithm]
    P = np.array([[[.2,.8,0],[0,.3,.7],[0,0,1]], np.eye(3)])
    g = np.zeros((2,3)); g[:,2] = 1
    c = np.ones((2,3))
    baseline = solver(P, g, c, round_decimals=round_decimals)
    timing = {}
    actual = solver(P, g, c, round_decimals=round_decimals, timing=timing)
    for old, new in zip(baseline, actual):
        if hasattr(old, 'to_dense'):
            old, new = old.to_dense(), new.to_dense()
        np.testing.assert_array_equal(old, new)
    assert timing['policy'] > 0 and timing['stok'] > 0


def test_api_work_does_not_overlap(monkeypatch):
    import tutorial_server as server
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    state = {'active': 0, 'peak': 0}
    guard = threading.Lock()
    def work(self):
        with guard:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
        time.sleep(.02)
        with guard:
            state['active'] -= 1
    monkeypatch.setattr(server.Handler, '_post_serial', work)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: server.Handler.do_POST(object.__new__(server.Handler)), range(4)))
    assert state['peak'] == 1
