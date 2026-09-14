"""feas_two_phase_sparse must return EXACTLY what feas_two_phase returns.

It is a representation change, not a method change: sparse CSR for the kernel, and eta packed by
terminal state instead of by state. So every returned array is compared elementwise against the
dense implementation, on a spread of worlds -- walls, multiple goals, multiple constraints,
state-action goals, unreachable pockets, and a world with no constraints at all.
"""
import os
import sys

import numpy as np
import pytest

# tutorial_server.py lives one level up (main_files/); the feas_two_phase* modules are siblings
# here in llm_solvers/, already covered by pytest's own rootless-import insertion -- main_files/
# needs an explicit add for tutorial_server.
_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

import tutorial_server as TS
from feas_two_phase import feas_iter_two_phase
from feas_two_phase_sparse import feas_iter_two_phase_sparse, kernel_to_sparse

ROWS = COLS = 9
N = ROWS * COLS


def build(walls=(), constraints=(4 * COLS,), goals=((0, None),), p_main=0.42):
    _sp, P = TS.get_kernel(ROWS, COLS, list(walls), p_main)
    na, nx, _ = P.shape
    g = np.zeros((na, nx))
    for li, a in goals:
        if a is None:
            g[:, li] = 1.0
        else:
            g[a, li] = 1.0
    c = np.ones((na, nx))
    for li in constraints:
        c[:, li] = 0.0
    return P, g, c


# Each world is solved ONCE by each implementation and reused: a dense solve materialises a
# [nx, nx, T] tensor and runs a 300-step loop, so re-solving per test made the suite unusable.
_CACHE = {}


# A pinned horizon for the comparison. These tests are about the REPRESENTATION being equal, not
# about how the horizon is chosen, and the dense reference costs 2*nx^2*T*8 bytes plus a T-step
# Python loop -- at the auto horizon (578 at noise 0.7) that is 60 MB and several seconds per
# world. test_auto_horizon_also_matches covers the default path separately.
T_TEST = 60


def solved(name):
    if name not in _CACHE:
        P, g, c = build(**WORLDS[name])
        _CACHE[name] = (P, g, c,
                        feas_iter_two_phase(P, g, c, max_time=T_TEST, round_decimals=None),
                        feas_iter_two_phase_sparse(P, g, c, max_time=T_TEST, round_decimals=None))
    return _CACHE[name]


WORLDS = {
    "plain":            dict(),
    "walls":            dict(walls=(3 * COLS + 3, 3 * COLS + 4, 4 * COLS + 3, 5 * COLS + 3)),
    "pocket":           dict(walls=(COLS, COLS + 1, COLS + 2, 2 * COLS + 2, 3 * COLS + 2)),
    "two goals":        dict(goals=((0, None), (8, None))),
    "two constraints":  dict(constraints=(4 * COLS, 2 * COLS + 6)),
    "state-action goal": dict(goals=((0, 3),)),
    "no constraints":   dict(constraints=()),
    "low noise":        dict(p_main=0.9),
    "high noise":       dict(p_main=0.3),
}


@pytest.mark.parametrize("name", sorted(WORLDS))
def test_sparse_matches_dense(name):
    _P, _g, _c, (dk, dpi, dep, den, db), (sk, spi, sep, sen, sb) = solved(name)

    assert np.allclose(dk, sk, rtol=0, atol=1e-12), f"kappa differs by {np.abs(dk-sk).max()}"
    assert (dpi == spi).all(), f"pi differs at {np.nonzero(dpi != spi)[0]}"
    # eta must match in FULL, after reinflating the packed form
    sep_d, sen_d = sep.to_dense(), sen.to_dense()
    assert sep_d.shape == dep.shape and sen_d.shape == den.shape
    assert np.allclose(dep, sep_d, rtol=0, atol=1e-12), f"eta+ differs by {np.abs(dep-sep_d).max()}"
    assert np.allclose(den, sen_d, rtol=0, atol=1e-12), f"eta- differs by {np.abs(den-sen_d).max()}"
    assert np.allclose(db, sb, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(WORLDS))
def test_packed_form_loses_nothing(name):
    """Every final state the DENSE eta uses must be in the packed terminal set."""
    _P, _g, _c, (_dk, _dpi, dep, den, _db), (_sk, _spi, sep, sen, _sb) = solved(name)
    for dense, packed, what in ((dep, sep, "eta+"), (den, sen, "eta-")):
        used = set(np.nonzero(dense.sum(axis=(0, 2)) > 0)[0].tolist())
        kept = set(packed.terms.tolist())
        assert used <= kept, f"{what}: dropped final states {sorted(used - kept)}"


def test_kernel_conversion_is_exact():
    P, _g, _c = build()
    Psp = kernel_to_sparse(P)
    for a in range(P.shape[0]):
        assert np.array_equal(np.asarray(Psp[a].todense()), P[a]), f"action {a} kernel changed"


@pytest.mark.parametrize("name", sorted(WORLDS))
def test_sparse_stok_handoff_is_exact(name):
    """to_sparse_stok must round-trip through stok_chain's format without loss."""
    _P, _g, _c, _dense, (_k, _pi, sep, sen, _b) = solved(name)
    for packed, what in ((sep, "eta+"), (sen, "eta-")):
        assert np.allclose(packed.to_dense(), packed.to_sparse_stok().to_dense(),
                           rtol=0, atol=0), f"{what} changed passing through SparseSTOK"


def test_auto_horizon_also_matches():
    """The pinned horizon above is for speed; the DEFAULT horizon path must agree too."""
    P, g, c = build(p_main=0.9)
    dk, dpi, dep, den, _db = feas_iter_two_phase(P, g, c, round_decimals=None)
    sk, spi, sep, sen, _sb = feas_iter_two_phase_sparse(P, g, c, round_decimals=None)
    assert dep.shape[2] == sep.to_dense().shape[2], "the two chose different horizons"
    assert np.allclose(dk, sk, rtol=0, atol=1e-12)
    assert (dpi == spi).all()
    assert np.allclose(dep, sep.to_dense(), rtol=0, atol=1e-12)
    assert np.allclose(den, sen.to_dense(), rtol=0, atol=1e-12)


def test_terminal_axis_is_actually_small():
    """The point of the packing: only terminal states can be final states."""
    _P, _g, _c, _dense, (_k, _pi, sep, sen, _b) = solved("plain")
    assert sep.terms.size <= 3, f"eta+ kept {sep.terms.size} terminal columns of {N}"
    assert sen.terms.size <= 3, f"eta- kept {sen.terms.size} terminal columns of {N}"
    saved = 1 - (sep.nbytes() + sen.nbytes()) / float(sep.dense_nbytes() + sen.dense_nbytes())
    assert saved > 0.9, f"packing saved only {saved:.1%}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
