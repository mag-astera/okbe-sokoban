"""Sparse feasibility iteration: sparse KERNEL (matrices) and sparse ETA (tensor).

PROVENANCE
    Written by Claude. Numerically the same algorithm as feas_two_phase.feas_iter_two_phase, which
    is itself a restructured copy of Tom's feas_iter_stationary_flat_dense. Nothing here is a new
    method; it is a change of REPRESENTATION only, and test_feas_sparse.py asserts every returned
    array matches the dense version elementwise.

WHAT IS SPARSE, AND WHY EACH ONE IS SPARSE

1. THE KERNEL, as one CSR matrix per action.
   p(x'|x,a) on a grid has at most five destinations per row -- 4.16% nonzero at 9x9, and the
   fraction falls as 1/nx. Phases 1 and 2 only ever use it as `P[a] @ vec`, which is a sparse
   matvec. The dense np.tensordot was doing nx^2 multiplies per action to add up five numbers.

2. ETA'S FINAL-STATE AXIS, indexed by TERMINAL STATE rather than by state.
   This is the big one. eta(x_f, t | x) is dense in x but almost empty in x_f, because x_f is
   PRESERVED by the propagation: the contraction sums over x', leaving x_f alone. So the only
   final states that can ever carry mass are the ones the boundary conditions seed --
       eta_pos: states where f[pi(x), x] > 0        (a goal is achieved there)
       eta_neg: states where c[pi(x), x) < 1        (a constraint is violated there)
                plus states with kappa == 0         (absorbed where nothing can succeed)
   On a 9x9 world with one goal and one fire that is 2 columns out of 81. Storing [nx, nx, T]
   instead of [nx, n_term, T] wastes a factor of nx / n_term -- 40x here, and it grows with the
   grid while the terminal set does not.

   NB this only holds once the eta^- boundary artifact is out of the way. Tom's original evaluates
   infeas(kappa) against a kappa that has not converged, so on the first iteration it seeds 80 of
   81 states and 77 columns end up carrying mass -- the structure that makes eta compressible is
   destroyed by that bug. Computing kappa to convergence first is what makes this representation
   available at all.

3. ETA'S TIME AXIS, via stok_chain.SparseSTOK on the way out.
   Roughly 1% of (start, end, t) entries are nonzero; SparseSTOK stores the nonzero (start, end)
   pairs and their time series. Offered by `to_sparse_stok()`, not forced, so callers that want a
   dense array still get one.

MEMORY, 9x9 with one goal and one constraint, T=303:
    dense eta_pos + eta_neg   31.8 MB
    terminal-indexed          0.79 MB      (2 terminal columns, not 81)
"""

import os
from time import perf_counter
import sys

import numpy as np
import scipy.sparse as sps

# TGMDP_methods.py lives one level up (main_files/), not alongside this file (main_files/llm_solvers/).
_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

from TGMDP_methods import feas, infeas


def kernel_to_sparse(p_ax_x):
    """[na, nx, nx] -> list of CSR, one per action. Exact: no thresholding."""
    if isinstance(p_ax_x, (list, tuple)):
        return [m if sps.isspmatrix_csr(m) else sps.csr_matrix(m) for m in p_ax_x]
    return [sps.csr_matrix(p_ax_x[a]) for a in range(p_ax_x.shape[0])]


class TerminalEta(object):
    """eta(x_f, t | x) stored [nx, n_term, T], with `terms` naming the columns.

    Written by Claude. A storage format only -- `to_dense()` reinflates to the [nx, nx, T] array
    the dense solver returns, exactly.
    """

    __slots__ = ("mat", "terms", "nx")

    def __init__(self, mat, terms, nx):
        self.mat = np.asarray(mat, dtype=float)      # [nx, n_term, T]
        self.terms = np.asarray(terms, dtype=np.intp)
        self.nx = int(nx)

    @property
    def T(self):
        return self.mat.shape[2]

    def to_dense(self):
        out = np.zeros((self.nx, self.nx, self.mat.shape[2]))
        if self.terms.size:
            out[:, self.terms, :] = self.mat
        return out

    def to_sparse_stok(self, tol=0.0):
        """Hand off to stok_chain.SparseSTOK, which the CK convolution already consumes."""
        from stok_chain import SparseSTOK
        nzp = np.any(np.abs(self.mat) > tol, axis=2)
        i, jj = np.nonzero(nzp)
        return SparseSTOK(i, self.terms[jj], self.mat[i, jj, :], self.nx)

    # --- enough of the ndarray surface to be a DROP-IN wherever the server reduces eta ------
    # The server does en.sum(axis=(1, 2)), en.sum(axis=2) and ep[x0]. Supporting exactly those
    # keeps the packed form usable without densifying, which is the whole point: reinflating on
    # the way out of the solver would hand back the 233 MB this exists to avoid.

    @property
    def shape(self):
        return (self.nx, self.nx, self.mat.shape[2])

    def sum(self, axis=None):
        if axis == (1, 2) or axis == [1, 2]:
            return self.mat.sum(axis=(1, 2))              # [nx], total mass per start state
        if axis == 2:
            # [nx, nx] over time. Dense in x_f, but that is only nx^2 floats -- the T axis, which
            # is what makes eta large, is already gone.
            out = np.zeros((self.nx, self.nx))
            if self.terms.size:
                out[:, self.terms] = self.mat.sum(axis=2)
            return out
        if axis == (0, 1) or axis == [0, 1]:
            return self.mat.sum(axis=(0, 1))              # [T], the arrival-time profile
        if axis is None:
            return float(self.mat.sum())
        raise NotImplementedError("TerminalEta.sum(axis=%r)" % (axis,))

    def __getitem__(self, x0):
        # One start state's slice, [x_f, t]: dense in x_f, which is cheap for a single row.
        out = np.zeros((self.nx, self.mat.shape[2]))
        if self.terms.size:
            out[self.terms, :] = self.mat[x0]
        return out

    def nbytes(self):
        return int(self.mat.nbytes + self.terms.nbytes)

    def dense_nbytes(self):
        return int(self.nx * self.nx * self.mat.shape[2] * 8)


def feas_iter_two_phase_sparse(p_ax_x, g, c, max_time=None, use_ET=True, xspace=None,
                               kappa_epsilon=1e-9, ET_epsilon=1e-9, kappa_iters=20000,
                               ET_iters=20000, round_decimals=3, kappa_tol=0.0, timing=None, compute_eta=None):
    """Same result as feas_two_phase.feas_iter_two_phase, on sparse matrices and a packed eta.

    round_decimals, kappa_tol, kappa_epsilon, ET_epsilon: see feas_two_phase.feas_iter_two_phase's
        docstring -- same parameters, same defaults, same effect. Kept in lockstep with the dense
        version so the two never disagree on which actions tie (test_feas_sparse.py asserts
        elementwise equality).

    compute_eta: optional function that computes eta_pos and eta_neg together.
        None uses propagate_termination_probabilities separately for each outcome.

    timing: optional output dictionary of policy and STOK elapsed seconds.

    Returns (kappa, pi, eta_pos, eta_neg, beta) where the two etas are TerminalEta. Call
    `.to_dense()` for the [nx, nx, T] arrays, or `.to_sparse_stok()` for the CK-convolution form.
    """
    _started = perf_counter()
    Psp = kernel_to_sparse(p_ax_x)
    na = len(Psp)
    nx = Psp[0].shape[0]
    g = np.asarray(g, dtype=float)
    c = np.asarray(c, dtype=float)
    f = g * c
    h = (1 - g) * c
    q = 1.0 - g
    x_inds = np.arange(nx)

    def contract(vec):
        """[na, nx] where row a is P[a] @ vec -- the sparse form of tensordot(P, vec)."""
        out = np.empty((na, nx))
        for a in range(na):
            out[a] = Psp[a] @ vec
        return out

    def rnd(v):
        return np.round(v, round_decimals) if round_decimals is not None else v

    # ---- Phase 1: kappa, to convergence ------------------------------------------------------
    kappa = rnd(np.amax(f, axis=0))
    for _ in range(kappa_iters):
        new_kappa = rnd((f + h * contract(kappa)).max(axis=0))
        if np.abs(new_kappa - kappa).max() <= kappa_epsilon:
            kappa = new_kappa
            break
        kappa = new_kappa
    kappa_ax = f + h * contract(kappa)
    if round_decimals is not None:
        kappa_ax = rnd(kappa_ax)
        kappa = kappa_ax.max(axis=0)
    best = (kappa_ax == kappa[np.newaxis, :]) if round_decimals is not None \
        else (kappa_ax >= kappa[np.newaxis, :] - max(1e-12, kappa_tol))
    pi = np.argmax(kappa_ax, axis=0).astype('int')

    # ---- Phase 2: minimum expected time within A*_x -------------------------------------------
    # ET is EXPECTED TIME TO THE GOAL, so it is only defined where the goal can be reached.
    #
    # At a state with kappa == 0 there is no goal to absorb at, and if that state is also absorbing
    # -- which every WALL is: P(wall|wall,a) = 1 for all a -- the update degenerates to
    # ET = 1 + ET and climbs by exactly 1 per iteration, forever. The loop then always ran its full
    # 20000 iterations: adding a single wall took the solve from 0.05s to 1.3s, with max(ET)
    # landing on exactly the iteration count, which is the signature.
    #
    # Those states are frozen instead. It costs nothing in correctness: a wall has no inbound
    # transitions, so no other state's expected time can depend on it, and pi at an infeasible
    # state is never used -- feas(kappa) gates it out of eta. The convergence test then sees only
    # states where ET actually converges.
    live = kappa > 0
    ET = np.zeros([nx])
    if use_ET:
        for _ in range(ET_iters):
            at = np.where(best, q + contract(ET), np.inf)
            new_ET = at.min(axis=0)
            new_ET = np.where(np.isfinite(new_ET), new_ET, ET)
            new_ET = np.where(live, new_ET, ET)      # infeasible states are frozen, see above
            if np.abs(new_ET - ET).max() <= ET_epsilon:
                ET = new_ET
                break
            ET = new_ET
        pi = np.argmin(np.where(best, q + contract(ET), np.inf), axis=0).astype('int')

    _policy_done = perf_counter()
    # ---- Phase 3: eta under the final pi, packed by terminal state -----------------------------
    if max_time is None:
        finite = ET[np.isfinite(ET) & (kappa > 0)]
        ET_max = finite.max() if finite.size else 0.0
        T_f = int(max(4 * np.sqrt(nx), 3 * ET_max + 10))
    else:
        ET_max = float(max_time)
        T_f = int(max_time)

    feas_indic = feas(kappa)
    h_pi = h[pi, x_inds]
    # the policy's own kernel, as ONE sparse matrix: row x is p(.|x, pi(x))
    if compute_eta is None:
        P_pi = sps.vstack([Psp[pi[x]].getrow(x) for x in range(nx)], format='csr')
    else:
        from eta_fast import policy_kernel
        P_pi = policy_kernel(Psp, pi)

    seed_pos = feas_indic * f[pi, x_inds]
    seed_neg = feas_indic * (1 - c[pi, x_inds]) + infeas(kappa)

    # THE TERMINAL SETS: the only final states either half can ever hold. Taken from the seeds
    # themselves rather than assumed, so a state-ACTION goal or an odd constraint cannot be missed.
    tp = np.nonzero(seed_pos > 0)[0]
    tn = np.nonzero(seed_neg > 0)[0]

    # EARLY-STOP TOLERANCE for propagate_termination_probabilities()'s own tail. T_f is already only a heuristic horizon (3x the
    # slowest state's own expected time, plus padding) -- measured on a 40x40/80-constraint world,
    # 99.999% of all eta mass has already landed by 51% of T_f, and the last step's flux is 1e-14
    # (pure float noise). round_decimals already commits to discarding anything past its own grain
    # (0.5 * 10**-round_decimals) EVERYWHERE ELSE in this solver (kappa gets rounded to it every
    # iteration) -- pegging the truncation here to that SAME grain, rather than an arbitrary
    # tighter constant, was checked empirically: going tighter than the grain buys accuracy nobody
    # can see (it gets rounded away regardless) while giving up real speed (5e-4 stops the rollout
    # ~1.8x sooner than 1e-9 does, since the flux-vs-t decay curve has a sharp elbow then a long,
    # nearly-flat tail). round_decimals=None (exact mode) gets a tight fixed floor instead, since
    # there is no rounding grain to inherit and "exact" should mean it.
    _tail_tol = 0.5 * 10 ** (-round_decimals) if round_decimals is not None else 1e-12

    def propagate_termination_probabilities(seed, terms, gate):
        """Propagate the boundary seed forward, keeping only the terminal columns.

        Built TIME-MAJOR ([T, nx, n_term]) so each step's slice m[t] is a C-contiguous (nx,
        n_term) block -- the sparse matvec can use it directly. Built [nx, n_term, T] (as the
        TerminalEta contract promises) that same slice would stride across the T axis, and
        scipy silently copies/ravels a fresh (nx, n_term) buffer on every single step to cope --
        measured at 0.7s of a 2.7s solve on a 40x40 world with a 190-wide eta- terminal set,
        for a copy that carries no information (the result was already going to be written back
        into a differently-shaped array). Transposed once at the end, which is O(nx*n_term*T) but
        happens ONCE per propagate_termination_probabilities() call rather than T times.

        STOPS EARLY once the flux landing at time t (m[t].sum(), NOT yet an accumulated/cumulative
        quantity) has been below _tail_tol for 5 consecutive steps -- the array was zero-init'd, so
        skipping the remaining steps just leaves their true (already below _tail_tol) values as 0
        instead of computing them. Guarded by t >= ET_max: the flux is NOT monotonic early on (it
        rises as the probability wavefront spreads out from the seed before any real absorption has
        happened -- measured 8.7 at t=50 vs 11.4 at t=100 on the same test world), so a bare
        below-tolerance check could trigger during an early trough before the real decay has even
        started. ET_max (the slowest state's own expected time) is a natural floor: by definition
        most starting states' mass has resolved by then, so it is already well past that rise.
        """
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
    return kappa, pi, eta_pos, eta_neg, beta
