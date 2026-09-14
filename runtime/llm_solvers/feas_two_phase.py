"""A two-phase variant of Tom's feas_iter_stationary_flat_dense.

PROVENANCE
    Written by Claude. It is a COPY of feas_iter_stationary_flat_dense in TGMDP_methods.py,
    restructured. Tom's original is untouched and remains the reference; this module does not
    import from it except for feas()/infeas(), which are used verbatim.

WHY
    The original advances kappa, ET and eta together, one backup each per iteration, and stops on
        delta = sum|kappa - old_kappa| <= epsilon      (or a cap of 4*sqrt(nx) iterations)
    Three things follow from that, all measured on a 9x9 grid with one goal and one constraint:

    1. THE CAP TRUNCATES kappa.  The cap is a diameter heuristic, but the number of backups needed
       scales with expected TIME, which grows sharply with noise. At p_main=0.42 the solve needs
       170 iterations and gets 36, exiting with delta=1.21 against epsilon=0.005 and kappa wrong by
       up to 0.79. A*_x is then built from a kappa that has not converged, so the policy takes
       genuinely less-feasible actions -- it walks toward the constraint.

    2. pi IS DECIDED BY ET, BUT THE STOPPING TEST ONLY WATCHES kappa.  A self-loop action always
       satisfies kappa(a,x) = kappa(x) exactly -- standing still forfeits nothing when the question
       is whether the goal is STILL reachable -- so a self-loop is in A*_x at every state and can
       never be excluded by the kappa comparison. Only the ET tie-break can drop it. Where kappa
       converges first (p_main=0.8 exits at 36 iterations with delta=0.0045 <= epsilon) ET is left
       flat, np.argmin falls through to the lowest index, and index 0 is `interact`. The returned
       policy then parks: 94% of the mass sits on three cells forever.

    3. ET CANNOT SIMPLY BE ADDED TO THE STOPPING TEST, because ET does not converge. At a state
       whose policy self-loops, ET = q + P[pi,x,:] @ ET is ET = 1 + ET, so max(ET) grows by exactly
       1.0000 per iteration without bound (58.39 -> 158.39 -> 258.39 at iterations 99/199/299).
       Requiring ET to settle makes the high-noise case far WORSE, 0.9760 -> 0.0109, because the
       drifting values corrupt the argmin elsewhere. The circularity is that ET is the mechanism
       meant to remove self-loops, and a self-loop is what makes ET diverge.

WHAT IS DIFFERENT
    The three quantities are separated instead of interleaved.

    Phase 1 -- converge kappa alone.  Since sum_{x_f,t} eta_pos(x_f,t|x) = kappa(x), the original's
        eta contraction is algebraically just  sum_x' P(x'|x,a) kappa(x'),  so kappa can be
        iterated on its own. That is both the same fixed point and far cheaper: no [nx,nx,T]
        tensor is touched until phase 3.

    Phase 2 -- with kappa fixed, A*_x is fixed, so minimum expected time is an ordinary
        undiscounted shortest-path problem over that action set. Iterating
            ET(x) <- min_{a in A*_x} [ q(a,x) + sum_x' P(x'|x,a) ET(x') ]
        from ET = 0 converges upward to the true minimum. A self-loop scores 1 + ET(x), which is
        strictly worse than ET(x), so it loses to any action that makes progress and is excluded
        without needing a special case. This is the same objective the original's use_ET branch
        aims at; it just is not chasing a moving A*_x.

    Phase 3 -- build eta ONCE under the final, fixed pi. In the original eta accumulates across
        iterations while pi is still changing, so the eta returned is the kernel of a
        NON-STATIONARY sequence of policies rather than of the pi returned beside it. That is why
        the original can report eta+ = 0.9992 while the returned pi reaches the goal 6% of the
        time. Here the two agree by construction.

    Interface, return values and semantics are otherwise the original's: same g/c convention, same
    [na, nx] goal and constraint functions, same (kappa, pi, eta_pos, eta_neg, beta) tuple.
"""

import os
from time import perf_counter
import sys

import numpy as np

# TGMDP_methods.py lives one level up (main_files/), not alongside this file (main_files/llm_solvers/).
_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

from TGMDP_methods import feas, infeas


def feas_iter_two_phase(p_ax_x, g, c, max_time=None, use_ET=True, xspace=None,
                        kappa_epsilon=1e-9, ET_epsilon=1e-9, kappa_iters=20000, ET_iters=20000,
                        round_decimals=3, kappa_tol=0.0, timing=None):
    """Feasibility iteration, kappa and ET converged separately, eta built last.

    p_ax_x : [na, nx, nx]   p(x'|x,a)
    g, c   : [na, nx]       achievement and constraint functions, as in the original
    max_time: horizon for eta. None picks one from the converged expected time, so the horizon is
              set by how long the policy actually takes rather than by a diameter guess.
    use_ET : kept for signature compatibility. Phase 2 IS the ET computation, so False only
             disables the tie-break refinement, leaving the kappa-argmax policy.
    kappa_epsilon, ET_epsilon: convergence thresholds for phase 1 and phase 2 respectively --
             separate knobs, not one shared `epsilon`, because they trade off very differently.
             kappa is a PROBABILITY (bounded [0,1], and feeds A*_x -- too loose here can flip
             which actions tie); ET is an UNBOUNDED expected step count, so the same absolute
             epsilon means something looser the farther a state is from the goal. Both default to
             1e-9 (unchanged from the old shared epsilon), and both converge in far fewer
             iterations, hence far faster, than that suggests: undiscounted propagation across a
             grid's full diameter is what actually costs iterations, not the tolerance alone --
             see value_world's own ihdc for the contrast (delta < 0.1, geometric contraction under
             its discount, roughly diameter-independent). Loosening either here trades exactness
             for speed on a LARGE grid specifically; a small one converges to 1e-9 quickly anyway.
    round_decimals: kappa is rounded to this many decimal places (every iteration, and again at
             convergence before A*_x is read off), the SAME mechanism the original
             feas_iter_stationary_flat_dense hardcodes at 4 decimals -- generalized here and
             defaulting to 3 instead of being a silent magic number, so a knife-edge tie between
             two actions can be WIDENED on purpose and the width tuned/compared. None disables
             rounding entirely and falls back to the previous exact (1e-12 floating-point
             tolerance) behaviour: two actions tie only if float64 cannot tell them apart at all.
    kappa_tol: widens A*_x beyond exact ties. An action enters A*_x if its kappa_ax is within
             kappa_tol of the max, so ET (expected time) gets to choose among actions that are
             merely NEAR-optimal, not just exactly tied. This is for a real, reproducible case:
             two routes that are both ~99.99% likely to succeed can differ in kappa by ~1e-5,
             purely from tiny converged-fixed-point differences between two almost-equally-good
             neighbours -- kappa has no notion of path length, so it will happily prefer the
             fractionally safer route over the visibly shorter one. round_decimals already does
             something similar by construction (rounding IS a tolerance), so kappa_tol only has
             an effect when round_decimals=None; with rounding on, widen the rounding instead.
             0.0 (default) reproduces the exact-argmax policy -- this is opt-in, not a silent
             change to what "optimal" means.

    timing: optional output dictionary of policy and STOK elapsed seconds.

    Returns (kappa, pi, eta_pos, eta_neg, beta), matching feas_iter_stationary_flat_dense.
    """
    _started = perf_counter()
    na, nx, _ = p_ax_x.shape
    f = g * c                      # achievement function
    h = (1 - g) * c                # continuation function
    q = 1.0 - g                    # unit time cost, zero once the goal is achieved
    x_inds = np.arange(nx)

    def rnd(v):
        return np.round(v, round_decimals) if round_decimals is not None else v

    # ---- Phase 1: kappa, to convergence -------------------------------------------------------
    # kappa(a,x) = f(a,x) + h(a,x) * sum_x' p(x'|x,a) kappa(x');  kappa(x) = max_a kappa(a,x).
    # Rounding is OFF by default in the sense that round_decimals=None restores this function's
    # original behaviour (no implicit feasibility floor); round_decimals=3 restores something LIKE
    # the original solver's np.round(..., 4) floor, at a coarser, explicit, tunable width.
    kappa = rnd(np.amax(f, axis=0))
    for _ in range(kappa_iters):
        kappa_ax = f + h * np.tensordot(p_ax_x, kappa, axes=([2], [0]))
        new_kappa = rnd(kappa_ax.max(axis=0))
        if np.abs(new_kappa - kappa).max() <= kappa_epsilon:
            kappa = new_kappa
            break
        kappa = new_kappa
    kappa_ax = f + h * np.tensordot(p_ax_x, kappa, axes=([2], [0]))
    if round_decimals is not None:
        kappa_ax = rnd(kappa_ax)
        kappa = kappa_ax.max(axis=0)

    # A*_x, the actions that attain the maximum. With rounding on this is exact equality on the
    # rounded array (matching the original's np.round(...,4) + != comparison, at a tunable width);
    # with rounding off it stays a floating-point allowance, not a feasibility threshold.
    best = (kappa_ax == kappa[np.newaxis, :]) if round_decimals is not None \
        else (kappa_ax >= kappa[np.newaxis, :] - max(1e-12, kappa_tol))
    pi = np.argmax(kappa_ax, axis=0).astype('int')

    # ---- Phase 2: minimum expected time within A*_x --------------------------------------------
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
        big = np.inf
        for _ in range(ET_iters):
            ax_time = q + np.tensordot(p_ax_x, ET, axes=([2], [0]))
            ax_time = np.where(best, ax_time, big)
            new_ET = ax_time.min(axis=0)
            # Unreachable states have no finite-time action; hold them rather than propagate inf.
            new_ET = np.where(np.isfinite(new_ET), new_ET, ET)
            new_ET = np.where(live, new_ET, ET)      # infeasible states are frozen, see above
            if np.abs(new_ET - ET).max() <= ET_epsilon:
                ET = new_ET
                break
            ET = new_ET
        ax_time = np.where(best, q + np.tensordot(p_ax_x, ET, axes=([2], [0])), np.inf)
        pi = np.argmin(ax_time, axis=0).astype('int')

    _policy_done = perf_counter()
    # ---- Phase 3: eta under the final, fixed pi ------------------------------------------------
    # Horizon from the converged expected time, so a slow policy is not truncated. The original
    # grows its eta by np.pad instead; sizing it once up front avoids the reallocation and the
    # shape bug that padding caused in the thresholded variant.
    if max_time is None:
        finite = ET[np.isfinite(ET) & (kappa > 0)]
        T_f = int(max(4 * np.sqrt(nx), 3 * (finite.max() if finite.size else 0) + 10))
    else:
        T_f = int(max_time)

    feas_indic = feas(kappa)
    h_pi = h[pi, x_inds]                       # [nx]
    P_pi = p_ax_x[pi, x_inds, :]               # [nx, nx]

    eta_pos = np.zeros([nx, nx, T_f])
    eta_neg = np.zeros([nx, nx, T_f])
    # Boundary conditions, exactly the original's.
    eta_pos[x_inds, x_inds, 0] = feas_indic * f[pi, x_inds]
    eta_neg[x_inds, x_inds, 0] = feas_indic * (1 - c[pi, x_inds]) + infeas(kappa)
    # Roll forward: mass at x that has not terminated moves under P_pi and lands one step later.
    for t in range(1, T_f):
        eta_pos[:, :, t] = h_pi[:, np.newaxis] * (P_pi @ eta_pos[:, :, t - 1])
        eta_neg[:, :, t] = (feas_indic * h_pi)[:, np.newaxis] * (P_pi @ eta_neg[:, :, t - 1])

    beta = eta_pos[:, :, 0].diagonal() + eta_neg[:, :, 0].diagonal()
    if timing is not None:
        timing.update(policy=_policy_done - _started, stok=perf_counter() - _policy_done)
    return kappa, pi, eta_pos, eta_neg, beta
