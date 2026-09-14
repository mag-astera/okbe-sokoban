"""
dbn_product.py -- produce a factored DBN  lambda(P, F)  into its FULL sparse joint transition
operator, as a GROUND-TRUTH harness for validating the factored (OKBE) feasibility / kappa.

    P_joint[a, s, s']  ,  shape [na, N, N] ,  N = prod_k |S_k|
    = prod_k  P_k( s_k' | s_k, alpha^k )   with  alpha^k = F_k( parents(k), a )  induced per space.

DESIGN CHOICE: EXPLICIT ENUMERATION + sparse.COO, NOT sparse.einsum. Enumeration lets us
  (a) know the exact nonzero count BEFORE building -> a hard memory pre-flight check, and
  (b) guarantee a sparse result with no dense intermediates (sparse.einsum cannot promise either).
Reliability over elegance -- which is what a ground-truth harness needs.

ASSUMPTIONS: affordances read parent STATES only (no alpha->alpha edges; those need a topological
alpha-resolve pass first). Each space exposes  .name, .ns, and a per-driving-action kernel
.P_a_s_s [na_k, ns_k, ns_k].  A space that must read ANOTHER space's STATE (e.g. X reading a door
mode) provides next_row_fns[name] = fn(state_dict, a) -> 1D next-state vector instead.
"""
import itertools
import numpy as np
import sparse


class JointOperatorTooLarge(MemoryError):
    """Raised (before allocating) when the joint operator would exceed a memory/size budget."""
    pass


def _dense_row(P, alpha, s):
    """P_a_s_s[alpha, s, :] as a dense 1-D vector, for dense ndarray or sparse.COO kernels."""
    row = P[alpha, s]
    return np.asarray(row.todense() if hasattr(row, "todense") else row).ravel()


def _induced_alpha(sp, F, state_dict, a, root):
    """Driving action of space sp given the joint state + free action a."""
    if sp.name == root:
        return int(a)                                          # root's action IS the free action
    aff = F.by_driven.get(sp.name)
    if aff is None:
        return int(getattr(sp, "default_action_ind", 0))       # undriven -> its default
    parent_state = {p: int(state_dict[p]) for p in aff.parents}
    return int(aff.induced_alpha_at(parent_state, a if aff.has_action else None))


def estimate_joint_size(ordered_spaces, root=None, na=None):
    """Pre-flight, WITHOUT building anything:
    returns (N, na, nnz_upper_bound, bytes_upper_bound, ns_list). Uses exact big-int products so
    N never silently overflows. nnz bound = na * N * prod_k(max nonzeros in any P_k row)."""
    root = root or ordered_spaces[0].name
    ns_list = [int(sp.ns) for sp in ordered_spaces]
    N = int(np.prod(ns_list, dtype=object))                    # exact python-int product
    if na is None:
        root_sp = next(sp for sp in ordered_spaces if sp.name == root)
        na = int(np.asarray(root_sp.P_a_s_s).shape[0]) if hasattr(root_sp, "P_a_s_s") else None
        if na is None:
            raise ValueError("cannot infer na from root; pass na=...")
    max_row = []
    for sp in ordered_spaces:
        P = getattr(sp, "P_a_s_s", None)
        if P is None:
            max_row.append(int(sp.ns))                         # unknown (custom fn) -> worst case
        else:
            Pd = P.todense() if hasattr(P, "todense") else np.asarray(P)
            max_row.append(int((Pd > 0).sum(axis=2).max()))   # max fan-out over next-state axis
    fan = int(np.prod(max_row, dtype=object))
    nnz_ub = na * N * fan
    bytes_ub = nnz_ub * (3 * 8 + 8)                            # COO 3-D: 3 int64 coords + 1 float64
    return N, na, nnz_ub, bytes_ub, ns_list


def dbn_to_joint_operator(ordered_spaces, F, root=None, na=None, next_row_fns=None,
                          max_states=500_000, max_nnz=50_000_000, bytes_budget=4_000_000_000,
                          dtype=np.float64, verbose=True):
    """Build the full joint transition operator of the DBN as a sparse.COO [na, N, N].

    ordered_spaces : spaces in canonical order (the joint state is row-major over this list).
    F              : AffordanceSet; induces each non-root space's alpha from its parent STATES (+ a
                     iff its affordance has_action). The ROOT space's action IS the free action a.
    root           : name of the free/BL space (default ordered_spaces[0].name).
    next_row_fns   : {name: fn(state_dict, a) -> 1-D next-state vector} overrides for spaces that
                     read other spaces' STATE (e.g. X reading a door mode). Optional.
    max_states/max_nnz/bytes_budget : memory guardrails. Raises JointOperatorTooLarge BEFORE
                     allocating if the pre-flight estimate exceeds any of them; a running nnz guard
                     also aborts mid-build if a stochastic fan-out beats the estimate.
    """
    root = root or ordered_spaces[0].name
    next_row_fns = next_row_fns or {}
    N, na, nnz_ub, bytes_ub, ns_list = estimate_joint_size(ordered_spaces, root, na)

    # ---- memory pre-flight: abort BEFORE building anything ----
    reasons = []
    if N > max_states:          reasons.append(f"N_joint={N:,} > max_states={max_states:,}")
    if nnz_ub > max_nnz:        reasons.append(f"nnz<={nnz_ub:,} > max_nnz={max_nnz:,}")
    if bytes_ub > bytes_budget: reasons.append(f"bytes<=~{bytes_ub/1e9:.2f}GB > budget={bytes_budget/1e9:.2f}GB")
    if reasons:
        raise JointOperatorTooLarge("refuse to build joint operator:\n  " + "\n  ".join(reasons)
                                    + "\n  (raise max_states/max_nnz/bytes_budget to override, or shrink the DBN)")
    if verbose:
        print(f"[dbn_join] N={N:,} na={na} nnz<= {nnz_ub:,} (~{bytes_ub/1e9:.3f} GB upper bound); building...")

    # row-major strides for flattening the joint state over ordered_spaces
    nsp = len(ordered_spaces)
    strides = np.ones(nsp, dtype=np.int64)
    for i in range(nsp - 2, -1, -1):
        strides[i] = strides[i + 1] * ns_list[i + 1]

    a_co, s_co, sp_co, dat = [], [], [], []
    running_nnz = 0
    ranges = [range(sp.ns) for sp in ordered_spaces]
    for a in range(na):
        for combo in itertools.product(*ranges):
            state_dict = {sp.name: combo[i] for i, sp in enumerate(ordered_spaces)}
            s_flat = int(np.dot(strides, combo))
            # nonzero next-state entries for each space
            per_space = []
            for i, sp in enumerate(ordered_spaces):
                if sp.name in next_row_fns:
                    row = np.asarray(next_row_fns[sp.name](state_dict, a)).ravel()
                else:
                    alpha = _induced_alpha(sp, F, state_dict, a, root)
                    row = _dense_row(sp.P_a_s_s, alpha, combo[i])
                nz = np.nonzero(row)[0]
                per_space.append([(int(j), float(row[j])) for j in nz])
            # outer product over spaces -> joint next-state distribution (nonzeros only)
            for nxt in itertools.product(*per_space):
                prob, sp_flat = 1.0, 0
                for i, (j, p) in enumerate(nxt):
                    prob *= p
                    sp_flat += int(strides[i]) * j
                if prob == 0.0:
                    continue
                a_co.append(a); s_co.append(s_flat); sp_co.append(sp_flat); dat.append(prob)
                running_nnz += 1
                if running_nnz > max_nnz:
                    raise JointOperatorTooLarge(
                        f"exceeded max_nnz={max_nnz:,} mid-build (stochastic fan-out beat the estimate)")

    coords = np.array([a_co, s_co, sp_co], dtype=np.int64)
    data = np.array(dat, dtype=dtype)
    P_joint = sparse.COO(coords, data, shape=(na, N, N))
    if verbose:
        print(f"[dbn_join] built sparse.COO shape={P_joint.shape} nnz={P_joint.nnz:,} "
              f"(~{P_joint.nnz*32/1e6:.1f} MB)")
    return P_joint


# ======================================================================================
# CROSS-ALPHA version: supports alpha->alpha edges (an affordance reading another space's
# INDUCED ACTION, not just its state) by resolving the alpha-vector in TOPOLOGICAL order.
# The base dbn_to_joint_operator above is left untouched.
# ======================================================================================

def _alpha_topo_order(ordered_spaces, alpha_parents):
    """Topological order of spaces by the alpha-dependency graph (edge p->k iff k reads alpha^p).
    Raises ValueError on a cycle (the forbidden instantaneous-equilibrium corner)."""
    from collections import defaultdict, deque
    names = [sp.name for sp in ordered_spaces]
    nameset = set(names)
    indeg = {n: 0 for n in names}
    adj = defaultdict(list)
    for k in names:
        for p in alpha_parents.get(k, ()):
            if p not in nameset:
                raise ValueError(f"alpha-parent {p!r} of {k!r} is not in ordered_spaces")
            adj[p].append(k); indeg[k] += 1
    q = deque([n for n in names if indeg[n] == 0]); order = []
    while q:
        n = q.popleft(); order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    if len(order) != len(names):
        cyc = [n for n in names if indeg[n] > 0]
        raise ValueError(f"alpha->alpha CYCLE among {cyc}: the forbidden instantaneous-equilibrium "
                         f"corner. Aggregate them into one space, or add a 1-tick delay (route the "
                         f"coupling through STATE instead of alpha).")
    return order


def _resolve_alpha_vector(by_name, F, state_dict, a, root, alpha_parents, topo_order):
    """Resolve every space's driving alpha at this joint state, in alpha-topological order.
    An alpha-parent p contributes alpha[p] (already resolved) to the context; every other
    affordance parent contributes its STATE. Returns {space_name: alpha}."""
    alpha = {}
    for name in topo_order:
        sp = by_name[name]
        if name == root:
            alpha[name] = int(a); continue                       # root's action IS the free action
        aff = F.by_driven.get(name)
        if aff is None:
            alpha[name] = int(getattr(sp, "default_action_ind", 0)); continue
        aps = set(alpha_parents.get(name, ()))
        context = {p: (alpha[p] if p in aps else int(state_dict[p])) for p in aff.parents}
        alpha[name] = int(aff.induced_alpha_at(context, a if aff.has_action else None))
    return alpha


def dbn_to_joint_operator_cross_alpha(ordered_spaces, F, alpha_parents, root=None, na=None,
                                      next_row_fns=None, max_states=500_000, max_nnz=50_000_000,
                                      bytes_budget=4_000_000_000, dtype=np.float64, verbose=True):
    """Like dbn_to_joint_operator, but SUPPORTS alpha->alpha (cross-conditioning) edges.

    alpha_parents : {space_name: [parent names whose INDUCED ALPHA (not state) its affordance reads]}.
        The affordance's H_psi axis for such a parent must be sized to the parent's ACTION count na_p
        (it is indexed by the parent's resolved alpha). Any affordance parent NOT listed here is read
        as a STATE, exactly as in the base version.  alpha_parents={} reproduces the base builder.
    next_row_fns : {name: fn(state_dict, a, alpha_dict) -> 1-D next row}. NOTE the extra `alpha_dict`
        arg (the resolved alpha-vector) vs the base version's fn(state_dict, a).

    Resolves each joint state's alpha-vector in TOPOLOGICAL order (raises on an alpha-cycle). Same
    memory pre-flight, sparsity, and COO output as the base version.
    """
    root = root or ordered_spaces[0].name
    next_row_fns = next_row_fns or {}
    N, na, nnz_ub, bytes_ub, ns_list = estimate_joint_size(ordered_spaces, root, na)

    reasons = []
    if N > max_states:          reasons.append(f"N_joint={N:,} > max_states={max_states:,}")
    if nnz_ub > max_nnz:        reasons.append(f"nnz<={nnz_ub:,} > max_nnz={max_nnz:,}")
    if bytes_ub > bytes_budget: reasons.append(f"bytes<=~{bytes_ub/1e9:.2f}GB > budget={bytes_budget/1e9:.2f}GB")
    if reasons:
        raise JointOperatorTooLarge("refuse to build joint operator:\n  " + "\n  ".join(reasons)
                                    + "\n  (raise max_states/max_nnz/bytes_budget to override, or shrink the DBN)")

    topo = _alpha_topo_order(ordered_spaces, alpha_parents)       # raises on cycle
    by_name = {sp.name: sp for sp in ordered_spaces}
    if verbose:
        print(f"[dbn_join/xalpha] N={N:,} na={na} nnz<= {nnz_ub:,}; alpha topo-order={topo}; building...")

    nsp = len(ordered_spaces)
    strides = np.ones(nsp, dtype=np.int64)
    for i in range(nsp - 2, -1, -1):
        strides[i] = strides[i + 1] * ns_list[i + 1]

    a_co, s_co, sp_co, dat = [], [], [], []
    running_nnz = 0
    ranges = [range(sp.ns) for sp in ordered_spaces]
    for a in range(na):
        for combo in itertools.product(*ranges):
            state_dict = {sp.name: combo[i] for i, sp in enumerate(ordered_spaces)}
            alpha_vec = _resolve_alpha_vector(by_name, F, state_dict, a, root, alpha_parents, topo)
            s_flat = int(np.dot(strides, combo))
            per_space = []
            for i, sp in enumerate(ordered_spaces):
                if sp.name in next_row_fns:
                    row = np.asarray(next_row_fns[sp.name](state_dict, a, alpha_vec)).ravel()
                else:
                    row = _dense_row(sp.P_a_s_s, alpha_vec[sp.name], combo[i])
                nz = np.nonzero(row)[0]
                per_space.append([(int(j), float(row[j])) for j in nz])
            for nxt in itertools.product(*per_space):
                prob, sp_flat = 1.0, 0
                for i, (j, p) in enumerate(nxt):
                    prob *= p; sp_flat += int(strides[i]) * j
                if prob == 0.0:
                    continue
                a_co.append(a); s_co.append(s_flat); sp_co.append(sp_flat); dat.append(prob)
                running_nnz += 1
                if running_nnz > max_nnz:
                    raise JointOperatorTooLarge(f"exceeded max_nnz={max_nnz:,} mid-build")

    coords = np.array([a_co, s_co, sp_co], dtype=np.int64)
    P_joint = sparse.COO(coords, np.array(dat, dtype=dtype), shape=(na, N, N))
    if verbose:
        print(f"[dbn_join/xalpha] built sparse.COO shape={P_joint.shape} nnz={P_joint.nnz:,}")
    return P_joint
