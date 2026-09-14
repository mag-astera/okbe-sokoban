"""stok.py -- the STOK factorization numerics, in one place.

Pure functions over arrays. NO dependency on state_spaces, arcade, or TGMDP_methods, so this module
is cheap to import and testable without building a world. state_spaces imports THIS (never the
reverse) -- keep it that way so the direction stays acyclic.

The OKBE pipeline, in order (region resolution and caching stay in state_spaces.AffordanceSet,
because they need the affordance DAG and the space registry):

    h_r         region indicator/continuation vector        [AffordanceSet.build_h_r]
    -> eta      STEF, state-time event function             stef_from_hr
    -> bundle   kappa (CEF), sf (SF), xi (TED), nu (hazard) event_functions_from_stef
    -> xi_S     global TEF over the first system event      (tef_from_survivals -- step 3)
    -> sigma    which spaces fired at t_f                   sed_sample / sed_sample_reject
    -> landing  terminal state per space                    (sample_stok -- step 3)
"""
import numpy as np


# ---------------------------------------------------------------------------------------------
# STEF: the state-time event function eta[s_i, s_f, t_f]
# ---------------------------------------------------------------------------------------------

def stef_from_hr(P_alpha, h_r, max_time, ns=None):
    """STEF from the continuation vector h_r (the region-indicator product from build_h_r), for the
    region driven by the action whose kernel is P_alpha. Kernel-generic: densifies P_alpha when it is
    sparse. Extracted verbatim from OKBEKernelMixin.compute_STEF_from_hr -- the host's (P_a_s_s[alpha],
    max_time, ns) are now explicit arguments.
      P_alpha : [ns, ns] kernel for the driving action (dense ndarray or anything with .todense())
      h_r     : length-ns continuation vector (1 = stay in region, 0 = event)
      max_time: time-axis chunk size; the axis is padded in max_time-sized blocks until convergence
      ns      : |S|; defaults to P_alpha.shape[0]
    Returns eta[s_i, s_f, t_f]."""
    t_const = max_time
    h_r = np.asarray(h_r, float)
    P_alpha = np.asarray(P_alpha.todense() if hasattr(P_alpha, 'todense') else P_alpha, float)
    ns = P_alpha.shape[0] if ns is None else ns
    eta_event = np.zeros([ns, ns, t_const])
    delta = np.inf; epsilon = 0.0001; tau = 0
    while delta > epsilon:
        old_eta_event = eta_event.copy()
        eta_contraction = np.tensordot(P_alpha, eta_event, axes=([1], [0]))
        new_eta_event = np.multiply(h_r[:, np.newaxis, np.newaxis], eta_contraction)
        eta_event[:, :, 1:] = new_eta_event[:, :, :-1].copy()           # t_f-1 time shift
        eta_event[range(ns), range(ns), 0] = 1 - h_r.copy()             # boundary at t_0
        tau += 1
        if tau >= t_const - 2:                                          # extend the time axis if needed
            eta_event     = np.pad(eta_event,     ((0, 0), (0, 0), (0, t_const)), 'constant', constant_values=0)
            old_eta_event = np.pad(old_eta_event, ((0, 0), (0, 0), (0, t_const)), 'constant', constant_values=0)
        delta = np.abs(np.sum(eta_event - old_eta_event))
    return eta_event


def event_functions_from_stef(eta):
    """Derive the rest of the event-function bundle from a STEF eta[s_i, s_f, t_f]:
    kappa (CEF), sf (SF), xi (TED), nu (hazard). Pure function of eta; small entries thresholded.
    Returns {eta, kappa, sf, xi, nu}. (Subsumes the old comptute_TEF: xi is stef.sum(axis=1).)"""
    eta = np.asarray(eta).copy()
    eta[eta < 1e-4] = 0
    kappa = np.cumsum(eta, axis=2).sum(axis=1)                   # CEF
    sf = 1 - kappa                                               # SF (survival)
    xi = eta.sum(axis=1)                                         # TED (hazard numerator)
    sf_prev = np.ones_like(sf); sf_prev[:, 1:] = sf[:, :-1]      # sf(t_f-1), with sf(-1)=1
    nu = np.divide(xi, sf_prev, out=np.zeros_like(xi), where=sf_prev > 1e-12)   # hazard = xi / sf(t-1)
    return {'eta': eta, 'kappa': kappa, 'sf': sf, 'xi': xi, 'nu': nu}


# ---------------------------------------------------------------------------------------------
# Global TED: the first SYSTEM-wide event time
# ---------------------------------------------------------------------------------------------

def tef_from_survivals(sfs, max_time=None):
    """Global TEF xi_S^r(t_f | s) from the per-space survival curves, by the paper's telescoping
    product (the factorization's first factor):

        xi_S^r(t_f | s) = prod_k sf_k^r(s^k, t_f - 1) - prod_k sf_k^r(s^k, t_f)

    = P(every space still in-region through t_f-1) - P(... through t_f).
      sfs      : list of 1-D survival curves, one per space, each already indexed at that space's
                 own current state (i.e. bundle['sf'][s_k]).
      max_time : truncate to this many t_f; default = the shortest curve.
    Returns xi over t_f = 0..T-1. It sums to 1 - S[T-1] where S(t) = prod_k sf_k(t) is the joint
    survival, so it is DEFECTIVE by design -- the leftover mass is "no event within the horizon".

    Free-function form of the math half of AffordanceSet.global_tef, which needs the class only to
    RESOLVE which survival curve each space is on. Callers that already hold the curves (e.g. a BL
    space with no OKBEKernelMixin, whose STEF came straight from stef_from_hr) can use this directly.
    """
    T = min(len(s) for s in sfs) if max_time is None else int(max_time)
    joint_sf = np.ones(T)
    for sf_k in sfs:
        joint_sf = joint_sf * np.asarray(sf_k)[:T]                  # S(t) = prod_k sf_k(t)
    return -np.diff(np.concatenate(([1.0], joint_sf)))              # [1-S0, S0-S1, ...]


# ---------------------------------------------------------------------------------------------
# SED: sigma_f | t_f -- which spaces make the first system event
# ---------------------------------------------------------------------------------------------

def sed_sample(nu, rng=None):
    """Sample the SED sigma-vector varphi_r(sigma_f | s, t_f): a multi-Bernoulli with per-space
    hazards nu_k CONDITIONED on sigma != 0 (at least one first-event fires at t_f). Exact, no rejection.

      sigma_k in {0,1};  nu_k = nu_k^r(s^k, t_f) is space k's local first-event hazard at t_f.

    Deterministic bits are read off for free -- nu_k >= 1 -> sigma_k = 1, nu_k <= 0 -> 0 -- and are
    never 'sampled'. The conditioning ("forcing") runs ONLY over the stochastic bits (0 < nu < 1),
    and ONLY when no nu_k >= 1 bit already guarantees the event. Among the stochastic bits it uses the
    exact sequential rule: before the first 1, draw bit k with p_k = nu_k / (1 - prod_{j>=k}(1-nu_j))
    (so the last stochastic bit -> nu/nu = 1, i.e. forced); after the first 1, draw the rest at raw nu.
    O(#stochastic bits): the suffix products prod_{j>=i}(1-nu_j) are precomputed once (reverse cumprod)
    and the uniforms are drawn in one batched call. See sed_sample_reject for the rejection variant.
    Returns an int ndarray sigma of length len(nu)."""
    rng = np.random if rng is None else rng
    nu = np.asarray(nu, float)
    sigma = (nu >= 1.0).astype(int)                     # deterministic firings (nu >= 1)
    stoch = np.where((nu > 0.0) & (nu < 1.0))[0]        # only these ever need a random draw
    nus = nu[stoch]

    if sigma.any():                                     # a nu>=1 bit already makes sigma != 0:
        sigma[stoch] = (rng.random(len(stoch)) < nus).astype(int)   # event guaranteed -> raw, vectorized
        return sigma

    if len(stoch) == 0:                                 # nothing can fire, yet an event is required
        raise ValueError("SED undefined: no bit can fire (all nu deterministic-off) under sigma != 0")

    # no deterministic firing -> condition the stochastic bits on 'not all zero' (exact sequential)
    tail = np.cumprod((1.0 - nus)[::-1])[::-1]          # tail[i] = prod_{j>=i}(1-nu_j), O(n) once
    u = rng.random(len(stoch))                          # one batched RNG call, one uniform per bit
    satisfied = False
    for i, k in enumerate(stoch):
        if satisfied:
            sigma[k] = int(u[i] < nus[i])               # raw once the event is already guaranteed
        elif u[i] < nus[i] / (1.0 - tail[i]):           # adjusted prob; last stochastic bit -> nu/nu = 1
            sigma[k] = 1
            satisfied = True
    return sigma


def sed_sample_reject(nu, rng=None, max_tries=100000):
    """Rejection variant of sed_sample (for METHOD COMPARISON): draw the plain multi-Bernoulli over the
    stochastic bits and reject all-zero draws until sigma != 0. Exact, but its expected number of draws
    is 1 / (1 - prod_k(1-nu_k)), which blows up as every nu_k -> 0 (p(0) -> 1). Deterministic bits are
    set directly and, if any nu_k >= 1, the event is guaranteed so no rejection happens. Same output
    distribution as sed_sample; raises if it can't satisfy sigma != 0 within max_tries."""
    rng = np.random if rng is None else rng
    nu = np.asarray(nu, float)
    base = (nu >= 1.0).astype(int)
    stoch = np.where((nu > 0.0) & (nu < 1.0))[0]
    nus = nu[stoch]

    if base.any():                                      # event guaranteed -> no rejection
        base[stoch] = (rng.random(len(stoch)) < nus).astype(int)
        return base
    if len(stoch) == 0:
        raise ValueError("SED undefined: no bit can fire (all nu deterministic-off) under sigma != 0")

    for _ in range(max_tries):
        draw = (rng.random(len(stoch)) < nus).astype(int)
        if draw.any():
            base[stoch] = draw
            return base
    raise RuntimeError(f"sed_sample_reject exceeded {max_tries} tries (p(0)~{np.prod(1.0-nus):.3g})")
