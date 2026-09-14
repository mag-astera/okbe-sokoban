"""Correctness checks for feas_thresh_sparse.feas_iter_two_phase_sparse_thresh, the sparse
two-phase risk-thresholded solver (see its own module docstring for the two-pass math).

There is no dense TWO-PHASE thresholded solver to compare against elementwise the way
test_feas_sparse.py compares feas_iter_two_phase_sparse to feas_iter_two_phase -- the only dense
thresholded solver that exists, TGMDP_methods.feas_iter_stationary_flat_dense_thresh, is the
ORIGINAL interleaved one, which this module's pass 1 deliberately does NOT reuse (see the module
docstring: pass 1 uses the FIXED two-phase solver instead, specifically so kappa_c does not
inherit the plain 'feas' solver's truncation/frozen-ET defects). So instead of one blanket
"matches the reference" test, this file checks the properties that must hold regardless of which
solver computes them:

  - self-consistency: eta_pos summed over final state and time must equal kappa exactly (the same
    invariant TGMDP_methods.feas_iter_stationary_flat_dense_thresh itself asserts with a
    warnings.warn if it fails)
  - theta = 0 (no risk tolerated) must hard-gate every state pass 1 found any risk at
  - kappa must be monotone non-decreasing in theta (more risk tolerance cannot hurt)
  - theta = 1 (unlimited risk tolerance) matches feas2s's own plain two-phase answer ONLY on a
    world with no constraints -- see below for why this is NOT true in general, which an earlier
    draft of this file's own docstring (and the ALGORITHMS "about" text) got wrong before this
    file caught it
  - one apples-to-apples comparison against the DENSE thresholded solver IS still done, on a
    small deterministic world, feeding it the SAME externally-computed kappa_c this module's own
    pass 1 would produce -- the two passes' MATH should agree even though pass 1's SOURCE differs

WHY theta = 1 does not equal the plain solve in general: f_thresh = g * r_thresh and
h_thresh = (1-g) * r_thresh use ONLY the risk indicator r_thresh, never the raw constraint
function c, in the achievement/continuation objective -- c only re-enters at the eta- boundary via
c_thresh = max(c, r_thresh). At theta=1, r_thresh is 1 everywhere (kappa_c <= 1 always holds), so
f_thresh/h_thresh become g/(1-g) -- NOT g*c/(1-g)*c, which is what the plain solver's f/h actually
are. The two are the same only when c is already 1 everywhere, i.e. the world has no reachable
constraints, which is exactly what the constraint-free test below checks. Verified directly: on a
world WITH a constraint, feas_iter_two_phase_sparse_thresh(theta=1) does NOT match feas2s
(kappa differs), confirming this is a real distinction and not a rounding artifact.
"""
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # feas_iter_stationary_flat_dense_thresh calls plt.imshow unconditionally

# tutorial_server.py and TGMDP_methods.py live one level up (main_files/); the feas_* solver
# modules are siblings here in llm_solvers/, already covered by pytest's own rootless-import
# insertion -- main_files/ needs an explicit add for the other two.
_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

import tutorial_server as TS
import TGMDP_methods as tg
from feas_two_phase_sparse import feas_iter_two_phase_sparse
from feas_thresh_sparse import feas_iter_two_phase_sparse_thresh

ROWS = COLS = 9  # matches utilities.ROW_COUNT/COLUMN_COUNT -- see the dense-comparison test's
                 # own comment for why a different grid size crashes the dense reference solver
N = ROWS * COLS


def build(walls=(), constraints=(), goals=((20, None),), start=0, p_main=1.0):
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


# ---- self-consistency, gating, monotonicity: solver-internal, no reference needed --------------

WORLDS = {
    "no constraints, deterministic": dict(constraints=(), p_main=1.0),
    "one constraint, deterministic": dict(constraints=(10,), p_main=1.0),
    "one constraint, noisy":         dict(constraints=(10,), p_main=0.85),
    "two constraints, noisy":        dict(constraints=(10, 44), p_main=0.7),
}


@pytest.mark.parametrize("name", sorted(WORLDS))
@pytest.mark.parametrize("theta", [0.0, 0.2, 0.6, 1.0])
def test_eta_pos_sums_to_kappa(name, theta):
    # max_time is pinned, not left on the auto horizon: "two constraints, noisy" at theta=0.2/0.6
    # showed a genuine 1.6e-4 shortfall on the auto horizon (T_f ~ 79), which max_time=300 (T_f
    # ~ 300) collapses to 6e-9 -- the eta rollout was simply cut off before all the mass had
    # arrived. Same class of auto-horizon under-estimate already documented for feas2/feas2s
    # itself (T_f = 4*sqrt(nx) can truncate kappa); it is a property of the HORIZON HEURISTIC
    # shared with the plain two-phase solver, not something specific to thresholding, so it is
    # pinned out of THIS test rather than papered over with a looser tolerance.
    P, g, c = build(**WORLDS[name])
    kappa, _pi, ep, _en, _beta = feas_iter_two_phase_sparse_thresh(
        P, g, c, risk_thresh=theta, round_decimals=None, max_time=300)
    eta_sum = np.asarray(ep.sum(axis=(1, 2)))
    assert np.allclose(eta_sum, kappa, rtol=0, atol=1e-6), \
        f"eta_pos.sum() vs kappa differs by {np.abs(eta_sum - kappa).max()}"


@pytest.mark.parametrize("name", ["one constraint, deterministic", "one constraint, noisy",
                                  "two constraints, noisy"])
def test_theta_zero_gates_every_risky_state(name):
    """No risk tolerated: any state pass 1 finds even a hair of violation risk under must end
    up with kappa == 0 -- it has nowhere legal left to go (f_thresh = h_thresh = 0 there)."""
    P, g, c = build(**WORLDS[name])
    _k1, _pi1, _ep1, en1, _b1 = feas_iter_two_phase_sparse(P, g, c, round_decimals=None)
    kappa_c = np.asarray(en1.sum(axis=(1, 2)))
    kappa0, _pi0, _ep0, _en0, _beta0 = feas_iter_two_phase_sparse_thresh(
        P, g, c, risk_thresh=0.0, round_decimals=None)
    kappa0 = np.asarray(kappa0)
    risky = kappa_c > 1e-9
    assert risky.any(), "world has no risky states -- test is not exercising the gate"
    assert np.allclose(kappa0[risky], 0.0, atol=1e-9), \
        f"states with kappa_c > 0 still show kappa > 0 at theta=0: {np.nonzero(kappa0[risky] > 1e-9)}"


@pytest.mark.parametrize("name", ["one constraint, noisy", "two constraints, noisy"])
def test_kappa_is_monotone_in_theta(name):
    """More risk tolerance can only ever help (or do nothing) -- never hurt."""
    P, g, c = build(**WORLDS[name])
    thetas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]
    kappas = []
    for theta in thetas:
        k, _pi, _ep, _en, _beta = feas_iter_two_phase_sparse_thresh(
            P, g, c, risk_thresh=theta, round_decimals=None)
        kappas.append(np.asarray(k))
    for i in range(1, len(kappas)):
        drop = kappas[i - 1] - kappas[i]
        assert drop.max() <= 1e-6, \
            (f"kappa dropped raising theta {thetas[i-1]} -> {thetas[i]} at states "
             f"{np.nonzero(drop > 1e-6)[0]}, by up to {drop.max()}")


def test_kappa_in_unit_interval():
    for name, cfg in WORLDS.items():
        P, g, c = build(**cfg)
        for theta in (0.0, 0.3, 1.0):
            kappa, _pi, _ep, _en, _beta = feas_iter_two_phase_sparse_thresh(
                P, g, c, risk_thresh=theta, round_decimals=None)
            kappa = np.asarray(kappa)
            assert kappa.min() >= -1e-9 and kappa.max() <= 1 + 1e-9, \
                f"{name} theta={theta}: kappa out of [0,1], range {kappa.min()}..{kappa.max()}"


def test_returns_terminal_eta_like_feas2s():
    """The app's caching layer (tutorial_server.get_eta_sparse) dispatches on TYPE
    (hasattr(ep, 'to_dense')), not on algorithm id -- eta must actually be a TerminalEta."""
    P, g, c = build(**WORLDS["one constraint, deterministic"])
    _kappa, _pi, ep, en, _beta = feas_iter_two_phase_sparse_thresh(
        P, g, c, risk_thresh=0.5, round_decimals=None)
    assert hasattr(ep, "to_dense") and hasattr(en, "to_dense")


# ---- theta = 1 degenerate case: constraint-free world only --------------------------------------

def test_theta_one_matches_plain_two_phase_when_world_has_no_constraints():
    P, g, c = build(**WORLDS["no constraints, deterministic"])
    ref_kappa, ref_pi, ref_ep, ref_en, ref_beta = feas_iter_two_phase_sparse(
        P, g, c, round_decimals=None)
    kappa, pi, ep, en, beta = feas_iter_two_phase_sparse_thresh(
        P, g, c, risk_thresh=1.0, round_decimals=None)
    assert np.allclose(np.asarray(kappa), np.asarray(ref_kappa), rtol=0, atol=1e-9)
    assert np.array_equal(np.asarray(pi), np.asarray(ref_pi))
    assert np.allclose(ep.to_dense(), ref_ep.to_dense(), rtol=0, atol=1e-9)
    assert np.allclose(en.to_dense(), ref_en.to_dense(), rtol=0, atol=1e-9)


def test_theta_one_does_NOT_match_plain_two_phase_when_a_constraint_exists():
    """The negative of the test above -- documents that the degenerate case really is special,
    not the general rule, so a future reader does not assume theta=1 is always a safe stand-in
    for the unconstrained solve on a world with real constraints in it."""
    P, g, c = build(**WORLDS["one constraint, deterministic"])
    ref_kappa, _ref_pi, _ref_ep, _ref_en, _ref_beta = feas_iter_two_phase_sparse(
        P, g, c, round_decimals=None)
    kappa, _pi, _ep, _en, _beta = feas_iter_two_phase_sparse_thresh(
        P, g, c, risk_thresh=1.0, round_decimals=None)
    assert not np.allclose(np.asarray(kappa), np.asarray(ref_kappa), rtol=0, atol=1e-9), \
        "theta=1 matched the plain solve even though the world has a constraint -- if this " \
        "starts passing, the degenerate-case reasoning above needs re-checking"


# ---- apples-to-apples against the DENSE thresholded solver --------------------------------------
# Deterministic (p_main=1.0) so the interleaved dense solver's slow probabilistic mixing does not
# apply -- it converges in ~12 steps here against >100 (and still not converged) on a noisy 9x9
# world, which is what this file uses a small fast-converging world for rather than reusing the
# WORLDS above. Grid must stay 9x9: feas_iter_stationary_flat_dense_thresh ends with an
# unconditional utilities.plotgrid2(kappa, ...) call that only reshapes correctly (and so only
# does not crash) when kappa.size == utilities.ROW_COUNT * utilities.COLUMN_COUNT == 81.

@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_matches_dense_reference_given_the_same_kappa_c(theta):
    P, g, c = build(constraints=(10,), p_main=1.0)
    sp, _P2 = TS.get_kernel(ROWS, COLS, [], 1.0)
    _k1, _pi1, _ep1, en1, _b1 = feas_iter_two_phase_sparse(P, g, c, round_decimals=None)
    kappa_c = np.asarray(en1.sum(axis=(1, 2)))

    kd, pid, _epd, _end = tg.feas_iter_stationary_flat_dense_thresh(
        P, P, g, c, kappa_c, risk_thresh=theta, use_ET=True, xspace=sp, max_time=40)
    ks, pis, _eps, _ens, _betas = feas_iter_two_phase_sparse_thresh(
        P, g, c, risk_thresh=theta, round_decimals=None)
    ks = np.asarray(ks)

    assert np.allclose(kd, ks, rtol=0, atol=1e-9), \
        f"theta={theta}: kappa differs from the dense reference by {np.abs(kd - ks).max()}"
    assert np.array_equal(pid, np.asarray(pis)), \
        f"theta={theta}: pi differs at {np.nonzero(pid != np.asarray(pis))[0]}"
