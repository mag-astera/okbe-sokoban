"""Sparse, two-phase risk-thresholded feasibility iteration.

Mirrors feas_two_phase_sparse.feas_iter_two_phase_sparse's structure (sparse kernel matrices,
kappa/ET converged separately, eta built once, TerminalEta packing) for the risk-thresholded
solver, TGMDP_methods.feas_iter_stationary_flat_dense_thresh, which exists only in dense,
single-pass-interleaved form -- there was no sparse counterpart before this file.

WHAT RISK-THRESHOLDING IS (paper sec 1.F, "Safety Parameter"): the kappa-OKBE is risk-seeking by
default -- any probability mass that touches a constraint can never contribute to kappa, so an
agent will take arbitrarily long, cautious paths to avoid ANY chance of one. Thresholding caps
that: a state is "safe enough" if its cumulative constraint-violation probability under the
UNCONSTRAINED optimal policy is within a budget theta; continuation and achievement are then gated
by that per-state safety indicator instead of by the raw constraint function.

TWO PASSES, exactly the dense original's structure, ported:

  Pass 1 (UNCONSTRAINED) solves the plain problem to get kappa_c(x), the cumulative probability of
      hitting a constraint from x under PASS 1's OWN optimal policy. This uses
      feas_iter_two_phase_sparse (the FIXED two-phase solver) rather than
      TGMDP_methods.feas_iter_stationary_flat_dense (the interleaved original the dense thresholded
      solver calls) -- pass 1's own documented defects (the 4*sqrt(nx) truncation cap; kappa_c
      built from a non-stationary eta, since the original accumulates eta while pi is still
      changing) would otherwise leak into r_thresh and hence into every downstream decision. This
      is a deliberate, documented deviation from the dense original: the same fix feas2/feas2s
      already make for the plain (non-thresholded) problem, applied here to pass 1 specifically.

  Pass 2 (THRESHOLDED): r_thresh(x) = 1[kappa_c(x) <= theta] (TGMDP_methods.risk_threshold).
      f_thresh = g * r_thresh, h_thresh = (1 - g) * r_thresh -- achievement/continuation are gated
      by the SAFETY INDICATOR, not by c: a state can be treated as a valid waypoint even inside
      what c would call a hard constraint, as long as the cumulative risk of the whole naive route
      from it is within budget. c only reenters at the eta- boundary, via
      c_thresh = max(c, r_thresh) (an OR of "not a raw constraint" with "within budget") -- ported
      AS-IS from the dense original. Note this is NOT the paper's own c_epsilon = 1_eta^eps * c
      (a product/AND); if that discrepancy ever needs reconciling with the paper, it belongs in
      both the dense and sparse solvers together, not silently fixed in only one.

INTERFACE: matches feas_iter_two_phase_sparse's shape exactly -- (kappa, pi, eta_pos, eta_neg,
beta) with eta_pos/eta_neg as TerminalEta -- rather than the dense thresholded solver's own
4-tuple (no beta). Callers that also want pass 1's kappa_c can get it from a separate
feas_iter_two_phase_sparse(...) call; it is not threaded through here to keep the two solvers'
signatures interchangeable.
"""
import os
from time import perf_counter
import sys

import numpy as np
import scipy.sparse as sps

# TGMDP_methods.py lives one level up (main_files/); feas_two_phase_sparse.py is a sibling in
# this same llm_solvers/ folder -- neither is on sys.path automatically when this module is imported
# (only the __main__ script's own directory gets that for free).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(_HERE)
for _p in (_HERE, _MAIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from TGMDP_methods import feas, infeas, risk_threshold
from feas_two_phase_sparse import feas_iter_two_phase_sparse, kernel_to_sparse


def feas_iter_two_phase_sparse_thresh(p_ax_x, g, c, risk_thresh=1.0, max_time=None, use_ET=True,
                                      xspace=None, kappa_epsilon=1e-9, ET_epsilon=1e-9,
                                      kappa_iters=20000, ET_iters=20000, round_decimals=3, timing=None, compute_eta=None):
    """Risk-thresholded feasibility iteration, sparse + two-phase (see module docstring).

    p_ax_x     : [na, nx, nx] p(x'|x,a), or an already-sparse list of CSR (kernel_to_sparse
                 accepts either, so pass-1's own call and pass 2's own contract() both just work).
    g, c       : [na, nx] achievement and constraint functions, as in the original.
    risk_thresh: theta, the cumulative constraint-violation budget. At 1.0, r_thresh is 1
                 everywhere (every state is "safe enough"), so f_thresh/h_thresh reduce to g/(1-g)
                 -- NOT the plain solver's g*c/(1-g)*c, since c never enters the achievement/
                 continuation objective here, only the eta- boundary (c_thresh = max(c, r_thresh)).
                 theta=1 therefore matches feas_iter_two_phase_sparse's own answer only on a world
                 with c == 1 everywhere, i.e. no reachable constraints -- verified by
                 test_feas_thresh_sparse.py, which checks both that this holds on a
                 constraint-free world and that it does NOT hold once one is added.
    round_decimals, kappa_epsilon, ET_epsilon: see feas_iter_two_phase_sparse's docstring -- same
                 parameters, same defaults, applied identically in both passes.

    compute_eta: optional function that computes eta_pos and eta_neg together.
        None uses propagate_termination_probabilities separately for each outcome.

    timing: optional output dictionary of policy and STOK elapsed seconds.

    Returns (kappa, pi, eta_pos, eta_neg, beta), matching feas_iter_two_phase_sparse.
    """
    _started = perf_counter()
    _pass1 = {}
    Psp = kernel_to_sparse(p_ax_x)
    na = len(Psp)
    nx = Psp[0].shape[0]
    g = np.asarray(g, dtype=float)
    c = np.asarray(c, dtype=float)
    x_inds = np.arange(nx)

    def contract(vec):
        """[na, nx] where row a is P[a] @ vec -- the sparse form of tensordot(P, vec)."""
        out = np.empty((na, nx))
        for a in range(na):
            out[a] = Psp[a] @ vec
        return out

    def rnd(v):
        return np.round(v, round_decimals) if round_decimals is not None else v

    # ---- Pass 1: UNCONSTRAINED two-phase solve, for kappa_c ------------------------------------
    _k1, _pi1, _ep1, en1, _b1 = feas_iter_two_phase_sparse(
        Psp, g, c, max_time=max_time, use_ET=use_ET, xspace=xspace,
        kappa_epsilon=kappa_epsilon, ET_epsilon=ET_epsilon,
        kappa_iters=kappa_iters, ET_iters=ET_iters, round_decimals=round_decimals, timing=_pass1, compute_eta=compute_eta)
    kappa_c = en1.sum(axis=(1, 2))                          # TerminalEta.sum -- no densifying

    # ---- Pass 2: THRESHOLDED two-phase solve ---------------------------------------------------
    r_thresh = risk_threshold(kappa_c, risk_thresh).astype(float)     # [nx], 1 = within budget
    c_thresh = np.maximum(c, r_thresh[np.newaxis, :])                  # [na, nx], ported as-is
    f_thresh = g * r_thresh[np.newaxis, :]
    h_thresh = (1.0 - g) * r_thresh[np.newaxis, :]
    q = 1.0 - g

    # Phase 1: kappa, to convergence, under f_thresh/h_thresh.
    kappa = rnd(np.amax(f_thresh, axis=0))
    for _ in range(kappa_iters):
        new_kappa = rnd((f_thresh + h_thresh * contract(kappa)).max(axis=0))
        if np.abs(new_kappa - kappa).max() <= kappa_epsilon:
            kappa = new_kappa
            break
        kappa = new_kappa
    kappa_ax = f_thresh + h_thresh * contract(kappa)
    if round_decimals is not None:
        kappa_ax = rnd(kappa_ax)
        kappa = kappa_ax.max(axis=0)
    best = (kappa_ax == kappa[np.newaxis, :]) if round_decimals is not None \
        else (kappa_ax >= kappa[np.newaxis, :] - 1e-12)
    pi = np.argmax(kappa_ax, axis=0).astype('int')

    # Phase 2: minimum expected time within A*_x. Same freeze-on-unreachable trick as
    # feas_iter_two_phase_sparse -- a state kappa==0 (or absorbing/unreachable under f_thresh/
    # h_thresh) has no finite-time action, so it is held rather than left to drift under ET = 1+ET.
    live = kappa > 0
    ET = np.zeros([nx])
    if use_ET:
        for _ in range(ET_iters):
            at = np.where(best, q + contract(ET), np.inf)
            new_ET = at.min(axis=0)
            new_ET = np.where(np.isfinite(new_ET), new_ET, ET)
            new_ET = np.where(live, new_ET, ET)
            if np.abs(new_ET - ET).max() <= ET_epsilon:
                ET = new_ET
                break
            ET = new_ET
        pi = np.argmin(np.where(best, q + contract(ET), np.inf), axis=0).astype('int')

    _policy_done = perf_counter()
    # Phase 3: eta under the final, fixed pi, packed by terminal state. The eta- boundary uses
    # c_thresh (matching the dense original), NOT h_thresh's own r_thresh and NOT the raw c.
    if max_time is None:
        finite = ET[np.isfinite(ET) & (kappa > 0)]
        ET_max = finite.max() if finite.size else 0.0
        T_f = int(max(4 * np.sqrt(nx), 3 * ET_max + 10))
    else:
        ET_max = float(max_time)
        T_f = int(max_time)

    feas_indic = feas(kappa)
    h_pi = h_thresh[pi, x_inds]
    if compute_eta is None:
        P_pi = sps.vstack([Psp[pi[x]].getrow(x) for x in range(nx)], format='csr')
    else:
        from eta_fast import policy_kernel
        P_pi = policy_kernel(Psp, pi)

    seed_pos = feas_indic * f_thresh[pi, x_inds]
    seed_neg = feas_indic * (1 - c_thresh[pi, x_inds]) + infeas(kappa)

    tp = np.nonzero(seed_pos > 0)[0]
    tn = np.nonzero(seed_neg > 0)[0]

    # Early-stop tail tolerance, pegged to round_decimals -- see feas_two_phase_sparse.roll's
    # docstring for the full reasoning and the empirical numbers behind this choice.
    _tail_tol = 0.5 * 10 ** (-round_decimals) if round_decimals is not None else 1e-12

    def propagate_termination_probabilities(seed, terms, gate):
        # Time-major build -- see feas_two_phase_sparse.roll's docstring for why: a [nx, n_term,
        # T]-shaped array makes every per-step slice non-contiguous, forcing scipy to copy it on
        # every sparse matvec. Same fix, same math, applied here too. Also early-stops once the
        # flux (m[t].sum()) has been below _tail_tol for 5 consecutive steps, past the ET_max
        # floor -- same guard against the early non-monotonic rise, see the sibling docstring.
        from feas_two_phase_sparse import TerminalEta
        T = T_f
        n_term = terms.size
        if n_term == 0:
            return TerminalEta(np.zeros((nx, 0, T)), terms, nx)
        m = np.zeros((T, nx, n_term))
        m[0, terms, np.arange(n_term)] = seed[terms]
        below = 0
        for t in range(1, T):
            m[t] = gate[:, np.newaxis] * (P_pi @ m[t - 1])
            if t >= ET_max:
                if m[t].sum() < _tail_tol:
                    below += 1
                    if below >= 5:
                        break
                else:
                    below = 0
        return TerminalEta(np.transpose(m, (1, 2, 0)), terms, nx)

    if compute_eta is None:
        eta_pos = propagate_termination_probabilities(seed_pos, tp, h_pi)
        eta_neg = propagate_termination_probabilities(seed_neg, tn, feas_indic * h_pi)
    else:
        eta_pos, eta_neg = compute_eta(P_pi, (seed_pos, seed_neg),
                                       (h_pi, feas_indic * h_pi), T_f, ET_max, _tail_tol)
    beta = seed_pos + seed_neg
    if timing is not None:
        timing.update(policy=_policy_done - _started, stok=perf_counter() - _policy_done)
        # Pass 1's STOK is required to determine the risk gate, but remains STOK work.
        timing["policy"] -= _pass1["stok"]
        timing["stok"] += _pass1["stok"]
    return kappa, pi, eta_pos, eta_neg, beta
