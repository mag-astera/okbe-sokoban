"""feas_two_phase must reach the optimum where the interleaved original does not.

The ground truth is MAXIMAL REACHABILITY PROBABILITY -- V = max_a sum_x' P(x'|x,a) V(x'),
V(goal)=1, V(fire)=0, infinite horizon -- solved independently here. That is by definition the
best achievable probability of reaching the goal without ever entering a constraint, so no policy
can beat it and kappa cannot legitimately exceed it.
"""
import os
import sys

import numpy as np
import pytest

# tutorial_server.py lives one level up (main_files/); feas_two_phase.py is a sibling here in
# llm_solvers/. pytest's own rootless-import insertion covers this file's own directory already, but
# main_files/ needs an explicit add for tutorial_server.
_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

import tutorial_server as TS
from feas_two_phase import feas_iter_two_phase

ROWS = COLS = 9
GOAL, FIRE, X0 = 0, 4 * COLS, 8 * COLS
NOISE = [0.9, 0.8, 0.7, 0.42]


def world(p_main):
    _sp, P = TS.get_kernel(ROWS, COLS, [], p_main)
    na, nx, _ = P.shape
    g = np.zeros((na, nx)); g[:, GOAL] = 1.0
    c = np.ones((na, nx)); c[:, FIRE] = 0.0
    return P, g, c, na, nx


def best_possible(P, nx):
    V = np.zeros(nx); V[GOAL] = 1.0
    for _ in range(200000):
        Vn = np.einsum('anx,x->an', P, V).max(axis=0)
        Vn[GOAL] = 1.0; Vn[FIRE] = 0.0
        if np.abs(Vn - V).max() < 1e-14:
            return Vn
        V = Vn
    return V


def simulate(P, pi, nx, steps=900):
    """Run the returned policy as a STATIONARY policy and see what actually happens."""
    d = np.zeros(nx); d[X0] = 1.0
    goal = fire = 0.0
    for _ in range(steps):
        nd = np.zeros(nx)
        for x in range(nx):
            if d[x] > 0:
                nd += d[x] * P[pi[x], x, :]
        goal += nd[GOAL]; fire += nd[FIRE]
        nd[GOAL] = 0; nd[FIRE] = 0; d = nd
    return goal, fire, d.sum()


@pytest.mark.parametrize("p_main", NOISE)
def test_reaches_the_optimum(p_main):
    P, g, c, _na, nx = world(p_main)
    V = best_possible(P, nx)
    kappa, pi, _ep, _en, _b = feas_iter_two_phase(P, g, c, round_decimals=None)
    assert np.abs(kappa - V).max() < 1e-4, (
        f"kappa differs from the converged optimum by {np.abs(kappa - V).max():.4f}")
    goal, _fire, _w = simulate(P, pi, nx)
    assert goal >= V[X0] - 1e-3, (
        f"the returned policy reaches the goal {goal:.4f} of the time, "
        f"against a best possible {V[X0]:.4f}")


@pytest.mark.parametrize("p_main", NOISE)
def test_no_self_loop_traps(p_main):
    """A policy that elects to stay put forever loses all its mass. Actions 0 and 1 self-loop."""
    P, g, c, _na, nx = world(p_main)
    _k, pi, _ep, _en, _b = feas_iter_two_phase(P, g, c, round_decimals=None)
    parked = [x for x in range(nx) if pi[x] in (0, 1) and x != GOAL and x != FIRE]
    assert not parked, f"policy self-loops at cells {parked}"
    _g, _f, wander = simulate(P, pi, nx)
    assert wander < 1e-3, f"{wander:.4f} of the mass never terminates"


@pytest.mark.parametrize("p_main", NOISE)
def test_eta_is_the_eta_of_the_returned_pi(p_main):
    """The original builds eta while pi is still changing, so the two disagree. These must not."""
    P, g, c, _na, nx = world(p_main)
    _k, pi, ep, _en, _b = feas_iter_two_phase(P, g, c, round_decimals=None)
    goal, _f, _w = simulate(P, pi, nx)
    assert abs(ep[X0].sum() - goal) < 2e-3, (
        f"eta+ says {ep[X0].sum():.4f} but simulating the returned pi gives {goal:.4f}")


@pytest.mark.parametrize("p_main", NOISE)
def test_eta_invariants(p_main):
    P, g, c, _na, nx = world(p_main)
    kappa, _pi, ep, en, _b = feas_iter_two_phase(P, g, c, round_decimals=None)
    # eta+ sums to kappa; the small tolerance is horizon truncation, not an inconsistency
    assert np.abs(ep.sum(axis=(1, 2)) - kappa).max() < 5e-3
    # the two halves together account for all the mass
    assert np.abs(ep.sum(axis=(1, 2)) + en.sum(axis=(1, 2)) - 1.0).max() < 5e-3
    assert (ep >= 0).all() and (en >= 0).all()


@pytest.mark.parametrize("walls", [(), (40,), (40, 41, 42), (30, 31, 32, 39, 41, 48, 49, 50)])
def test_walls_do_not_stall_ET(walls):
    """A wall is absorbing AND unreachable, so ET = 1 + ET there and never converges.

    Unfrozen, that ran the phase-2 loop to its 20000-iteration cap on EVERY solve with a wall in
    it -- 1.3s instead of 0.1s -- with max(ET) landing on exactly the iteration count.
    """
    import time
    _sp, P = TS.get_kernel(ROWS, COLS, list(walls), 0.42)
    na, nx, _ = P.shape
    g = np.zeros((na, nx)); g[:, GOAL] = 1.0
    c = np.ones((na, nx)); c[:, FIRE] = 0.0
    t0 = time.time()
    kappa, pi, _ep, _en, _b = feas_iter_two_phase(P, g, c, round_decimals=None)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"{len(walls)} walls took {elapsed:.2f}s -- phase 2 is not converging"
    # and the answer is unaffected: walls are unreachable, so they cannot change any real state
    for w in walls:
        assert kappa[w] == 0.0, f"wall {w} should be infeasible"
    goal, _fire, wander = simulate(P, pi, nx)
    assert wander < 1e-3, f"{wander:.4f} of the mass never terminates"
    assert goal > 0.9, f"only {goal:.4f} reaches the goal"


def test_beats_the_original_where_the_original_fails():
    """The two documented failures, pinned so a regression is visible."""
    import io, contextlib
    import TGMDP_methods as tg
    TS._stub_solver_plots()
    for p_main, what in ((0.8, "self-loop parking"), (0.42, "truncated kappa")):
        P, g, c, _na, nx = world(p_main)
        _sp, _P = TS.get_kernel(ROWS, COLS, [], p_main)
        with contextlib.redirect_stdout(io.StringIO()):
            _k1, pi1, _e1, _n1, _b1 = tg.feas_iter_stationary_flat_dense(
                P, g, c, use_ET=True, xspace=_sp)
        _k2, pi2, _e2, _n2, _b2 = feas_iter_two_phase(P, g, c, round_decimals=None)
        old = simulate(P, pi1, nx)[0]
        new = simulate(P, pi2, nx)[0]
        assert new > old, f"{what} at p_main={p_main}: two-phase {new:.4f} vs original {old:.4f}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
