import timeit

import numpy as np
from scipy.sparse import csr_matrix as csr
import itertools
# torch / torch.nn.functional deferred into the 3 methods that use them (saves ~1.2s at import)
import scipy as sp
import scipy.sparse.linalg  # ensure sp.sparse.linalg.spsolve is accessible (submodule isn't auto-imported by `import scipy`)
from bisect import bisect_left
import warnings
import sparse
from functools import reduce
import stok
# STOK numerics live in stok.py (numpy-only, no cycle). Re-exported here so existing callers and
# `from state_spaces import sed_sample, ...` keep working; prefer importing from stok in new code.
from stok import sed_sample, sed_sample_reject

# `arcade` used to be imported at module scope purely for arcade.color.BLACK, which is the plain
# tuple (0, 0, 0). That pulled pyglet/OpenGL (~0.58s, 41% of this module's import time) into a
# pure-numerics module and made it unusable headless. The renderer still uses arcade directly;
# state_spaces does not need it. Same discipline as the deferred imports in Task.__init__.
BLACK = (0, 0, 0)


# class General_Statespace:
#     def __init__(self, size, name, max_time, action_type_ind, theme_color=arcade.Color.BLACK, picture_dict=None):
#         self.ns = size
#         self.name = name
#         self.max_time = max_time
#         self.action_type_ind = action_type_ind
#         self.theme_color = theme_color
#         self.picture_dict = picture_dict
#         self.P_a_s_s = None

class Affordance:
    """F_k(alpha^k | parents) = sum_psi H_alpha(alpha^k | psi) * H_psi(psi | parents)."""
    def __init__(self, driven, parents, H_psi, H_alpha, spec=None):
        self.driven  = driven
        self.parents = list(parents)
        self.H_psi   = np.asarray(H_psi,   float)   # H_k^psi   : [n_feat, *parent_dims]
        self.H_alpha = np.asarray(H_alpha, float)    # H_k^alpha : [n_alpha, n_feat]
        self.spec    = spec   # optional authored boolean gates [(alpha, gate), ...]; None if hand-built.
        #                       When set (by gates.make_affordance), describe() prints it verbatim.
        # TODO(factored-affordance): this materializes a DENSE [n_alpha, *parent_dims] tensor = O(prod
        # parent_dims). Fine for a few small parents (a door); an affordance linking MANY/large parents
        # explodes. A conjunctive gate is an OUTER PRODUCT of per-parent factors -> store the FACTORS
        # (O(sum parent_dims)) and AND on demand; that also makes region_indicator_or trivial (= the
        # over_parent's own factor), no dense tensor / no argmax. Same move as declarative guards. Left
        # dense ON PURPOSE for now (2026-07-11). grep this tag; see memory okbe-unimplemented / okbe-dof-class.
        self.tensor  = np.tensordot(H_alpha, H_psi, axes=([1], [0]))
        # action axis present iff H_psi has an extra dim beyond [feature, *parents]:
        #   [feature, *parents]          -> H(psi | s)     (action-independent)
        #   [feature, action, *parents]  -> H(psi | s, a)  (action-dependent)
        self.has_action = (self.H_psi.ndim == len(self.parents) + 2)
        self._induced = self.tensor.argmax(0)   # cached deterministic induced-alpha; the tensor is FIXED
        # (= H_alpha o H_psi), so this is a constant [*parents] / [action,*parents]. ALL lookups index it
        # -> no per-call argmax in the tree-search hot path (region_of / induced_alpha_at / region_indicator).

    def region_indicator(self, over_parent, alpha_r, action=None, clamp=None):
        """1[ s^over_parent keeps the driven var at its region default alpha_r ] (at BL action `action`
        iff this affordance has an action axis), as a float vector over over_parent's states. Other
        parents held at their region defaults via clamp. Handles H(psi|s) and H(psi|s,a)."""
        induced = self._induced          # [*parents] or [action, *parents]
        idx = [int(action)] if self.has_action else []   # clamp the BL action axis only if present
        for p in self.parents:
            if p == over_parent:
                idx.append(slice(None))
            elif clamp is not None and p in clamp:
                idx.append(clamp[p])
            else:
                raise ValueError(f"clamp needed for parent '{p}' of '{self.driven}'")
        return (induced[tuple(idx)] == alpha_r).astype(float)

    def region_indicator_or(self, over_parent, alpha_r, action=None):
        """CONSERVATIVE / OR region indicator over `over_parent`'s states -- CONTEXT-FREE (no clamp).
        Marginalizes the OTHER parents by asking 'does the driven var leave its default alpha_r for ANY
        of their settings?'. h=1 only where the driven stays at alpha_r for EVERY other-parent config =
        the INTERSECTION of the per-context drift regions = a SUBSET of the true (clamped) region. So you
        segment EARLY, never late: never miss an event; extra exits are benign (the exact AND is re-checked
        at the concrete state in tree search). Equals region_indicator exactly when over_parent is the only
        parent (no other axes to reduce), so single-parent affordances are unchanged. This is what lets a
        space read by a MULTI-parent affordance (e.g. money read by a priced door) get a context-free h_r."""
        induced = self._induced                 # [*parents] or [action, *parents]
        if self.has_action:
            induced = induced[int(action)]                   # clamp the BL action axis -> [*parents]
        keep = self.parents.index(over_parent)
        other_axes = tuple(i for i in range(induced.ndim) if i != keep)
        leaves = (induced != alpha_r)
        if other_axes:                                       # OR over the other parents' states
            leaves = leaves.any(axis=other_axes)
        return (~leaves).astype(float)                       # over `over_parent`'s states

    def induced_alpha(self):
        """Deterministic alpha induced at each parent cell: argmax over the alpha axis. Shape = parent_dims."""
        return self._induced

    def induced_alpha_at(self, state, action=None):
        """The alpha this affordance induces given the current parents (and the BL action if it has one).
        state: {parent_key: state_index}; action: BL action index (required iff self.has_action).
        Handles BOTH H(psi|s) (tensor [alpha,*parents]) and H(psi|s,a) (tensor [alpha,action,*parents])."""
        induced = self._induced          # [*parents] or [action, *parents]
        parent_idx = tuple(int(state[p]) for p in self.parents)
        idx = ((int(action),) + parent_idx) if self.has_action else parent_idx
        return int(induced[idx])

    def describe(self):
        """Human-readable summary of which parent-values (and the free action, if read) induce each
        NON-default alpha, in DISJUNCTIVE NORMAL FORM."""
        spec = getattr(self, "spec", None)
        if spec is not None:
            lines = [f"Affordance drives {self.driven!r}, parents={self.parents}, "
                     f"reads_action={self.has_action}  [authored spec]"]
            for alpha, gate in spec:
                lines.append(f"    induces alpha={alpha}  when:  {gate!r}")
            return "\n".join(lines)
        import itertools
        ind = self._induced
        axes = (["action"] if self.has_action else []) + list(self.parents)
        vals, counts = np.unique(ind, return_counts=True)
        default = int(vals[counts.argmax()])                       # drift = most common induced alpha

        def rect_cover(points):
            """Greedily cover the firing tuples with maximal axis-aligned rectangles (a valid DNF cover:
            union == the firing set, each rectangle a subset of it)."""
            pts = set(map(tuple, points)); terms = []
            while pts:
                seed = min(pts); sets = [{seed[j]} for j in range(len(axes))]; grew = True
                while grew:                                        # grow each axis while the product stays inside
                    grew = False
                    for j in range(len(axes)):
                        for v in sorted({p[j] for p in pts} - sets[j]):
                            trial = [sets[k] | ({v} if k == j else set()) for k in range(len(axes))]
                            if all(t in pts for t in itertools.product(*trial)):
                                sets[j].add(v); grew = True
                terms.append([set(s) for s in sets]); pts -= set(itertools.product(*sets))
            return terms

        lines = [f"Affordance drives {self.driven!r}, parents={self.parents}, "
                 f"reads_action={self.has_action}, default_alpha={default}"]
        for a in sorted(set(int(v) for v in vals) - {default}):
            terms = rect_cover(np.argwhere(ind == a))
            strs = ["(" + " AND ".join(f"{axes[j]} in {sorted(t[j])}" for j in range(len(axes))) + ")"
                    for t in terms]
            cond = strs[0][1:-1] if len(strs) == 1 else "  OR  ".join(strs)     # drop parens if single term
            lines.append(f"    induces alpha={a}  when:  {cond}")
        return "\n".join(lines)

class AffordanceSet:
    """F = {F_j}. Owns the parent/child graph."""
    def __init__(self, affordances=(), spaces=()):
        self.by_driven = {a.driven: a for a in affordances}
        self.spaces = {s.name: s for s in spaces}   # registry name->StateSpace, so region_of can read defaults

    def add(self, aff):
        self.by_driven[aff.driven] = aff
        return aff

    def parents_of(self, k):
        """Spaces that dictate alpha^k (drive k)  ->  feeds alpha_ind."""
        return self.by_driven[k].parents if k in self.by_driven else []

    def children_of(self, k):
        """Spaces k drives: j with k in Pa(j)  ->  feeds the h_k^r indicators."""
        return [j for j, a in self.by_driven.items() if k in a.parents]

    def build_h_r(self, k, ns_k, default_alpha=None, g_k=None, c_k=None, clamp=None, action=0):
        """h_k^r = (1-g_k)*c_k * prod_{j in children_of(k)} 1[ s^k keeps j at default ] (at BL action `action`).
        ns_k = |S_k|; default_alpha = {child_j: alpha_r^j}; g_k,c_k length-ns_k (default 0,1)."""
        g = np.zeros(ns_k) if g_k is None else np.asarray(g_k, float)
        c = np.ones(ns_k)  if c_k is None else np.asarray(c_k, float)
        h = (1.0 - g) * c
        default_alpha = default_alpha or {}
        for j in self.children_of(k):                 # empty for spaces that drive nothing
            # Conservative / context-free OR region (per-space feature-partition): marginalize the
            # child's OTHER parents instead of clamping them, so a MULTI-parent child (e.g. a priced door
            # reading money) yields a clamp-free h_r. Identical to the exact region_indicator when the
            # child has a single parent, so single-parent spaces (physio driven by X) are unchanged.
            ind = self.by_driven[j].region_indicator_or(k, default_alpha[j], action)
            h = h * ind
        return h

    def region_of(self, k, state=None, action=None):
        """Region key for space k = (driving action of k, driving actions of the spaces k drives).
        The driving alpha is read from k's affordance EVALUATED at the current (parent states, BL action)
        (`state` = {space_name: state_index}, `action` = BL action index); falls back to the space's
        default_action_ind when k has no affordance or no state/action given. So this returns the actual
        (s,a)-region for any situation -- e.g. F_xw drives hydration up only at (lake, drink). Hashable."""
        own = self._driving_alpha(k, state, action)
        children = tuple(self._driving_alpha(j, state, action) for j in self.children_of(k))
        return (own,) + children

    def _driving_alpha(self, k, state, action):
        aff = self.by_driven.get(k)
        if aff is not None and state is not None and (action is not None or not aff.has_action):
            return aff.induced_alpha_at(state, action)   # action ignored if F_k is action-independent
        return self.spaces[k].default_action_ind if k in self.spaces else 0

    def regions_from_state(self, x, na, k='X'):
        """Partition space k's actions AT its state x by the induced HL alpha-vector over children_of(k).
        Each distinct alpha-vector = one region reachable BY ACTION CHOICE from x (the only regions the
        controller of k can pick between). Returns {alpha_vector (region key): [actions inducing it]}.
        The all-default key (drift) holds the non-triggering navigation actions; non-default keys hold
        the firing actions (each = a region-exit / option boundary). NB: only captures ACTION-selected
        regions; regions set by the HL STATE (not k's action) are node properties, not here."""
        from collections import defaultdict
        children = self.children_of(k)
        regions = defaultdict(list)
        for a in range(na):
            key = tuple(int(self.by_driven[j].induced_alpha_at({k: x}, action=a)) for j in children)
            regions[key].append(a)
        return dict(regions)

    def default_action_vector(self, ordered_spaces, state_vector=None, bl_action=None):
        """Driving-action vector over the system (canonical order = ordered_spaces, e.g. ss_list).
        state_vector is None -> DRIFT defaults (each space's default_action_ind / no-op).
        state_vector given    -> the alpha INDUCED at that state.  state_vector[i] indexes
        ordered_spaces[i]; returns an int array av where av[i] = driving alpha of ordered_spaces[i].
        NB: assumes affordances read parent STATES (no alpha->alpha edges); for cross-alpha
        conditioning, iterate ordered_spaces in TOPOLOGICAL order and feed already-set av in.
        X (root) has no affordance -> its slot falls to default_action_ind; overwrite with the
        free BL action if you want it there."""
        idx = {s.name: i for i, s in enumerate(ordered_spaces)}
        av = np.array([getattr(s, 'default_action_ind', 0) for s in ordered_spaces], dtype=int)
        if state_vector is None:
            return av                                             # drift / no-op defaults
        for i, sp in enumerate(ordered_spaces):
            aff = self.by_driven.get(sp.name)
            if aff is None:
                continue                                          # undriven -> keep its default
            parent_state = {p: int(state_vector[idx[p]]) for p in aff.parents}
            av[i] = aff.induced_alpha_at(parent_state,
                                         action=(bl_action if aff.has_action else None))
        return av

    def region_functions_for(self, space, region_key, bl_action=0):
        """Get-or-build the event-function bundle for `space` in region `region_key`, the SINGLE
        SOURCE OF TRUTH for a bundle (both the pre-build loop and region_functions_at route through it).
          region_key = (own_driving_alpha,) + (child driving alphas, ordered by children_of(space)),
                       i.e. exactly the tuple region_of returns.
          bl_action  = BL action at which the child region-indicators are evaluated (irrelevant for
                       childless spaces, where h_r is just the constraint vector).
        Builds + caches in space.region_functions on first request; returns the cached bundle thereafter.
        Returns {eta (STEF), kappa (CEF), sf (SF), xi (TED), nu (hazard)}, all [s_i, (s_f,) t_f]."""
        ef = space.region_functions
        if region_key in ef:
            return ef[region_key]                                    # cached
        own_alpha = int(region_key[0])
        child_alpha = {j: int(a) for j, a in zip(self.children_of(space.name), region_key[1:])}
        h_r = self.build_h_r(space.name, space.ns, default_alpha=child_alpha,
                             c_k=space.constraint_vec, action=(bl_action or 0))
        eta = space.compute_STEF_from_hr(h_r, own_alpha)             # STEF eta[s_i, s_f, t_f] (from the mixin)
        bundle = space.event_functions_from_stef(eta)               # kappa/sf/xi/nu derived in OKBEKernelMixin
        bundle['h_r'] = h_r                                          # region indicator: also gates this region's SPK
        bundle['spk'] = {}                                          # lazy per-action SPK cache; fill via spk_for
        ef[region_key] = bundle
        return bundle

    def spk_for(self, space, region_key, a, bl_action=0):
        """Get-or-build the lazily-extendable SPK for `space`, action `a`, in region `region_key`.
        Gated by the SAME region indicator h_r that defines the region's event functions, so the SPK
        and the {eta,kappa,sf,xi,nu} bundle are consistent. Cached in bundle['spk'][a]; index the
        result as spk[0,t] (in) / spk[1,t] (out), extending in time on demand."""
        bundle = self.region_functions_for(space, region_key, bl_action=bl_action)
        cache = bundle['spk']
        if a not in cache:
            # Reuse: when h_r == the space's constraint_vec (childless / no region indicator), this SPK is
            # identical to the one the space already built at construction as spk_alpha_list[a] -- same
            # kernel, same h, same max_time. Reuse it instead of rebuilding (avoids the SPK-build cost).
            built = getattr(space, 'spk_alpha_list', None)
            cvec = getattr(space, 'constraint_vec', None)
            if (built is not None and a < len(built) and cvec is not None
                    and np.array_equal(np.asarray(bundle['h_r'], float), np.asarray(cvec, float))):
                cache[a] = built[a]
            else:
                Pdict = getattr(space, "P_s_s_given_g_dict", None)
                P_a = Pdict[a] if Pdict is not None else space.P_a_s_s[a]
                cache[a] = SPK(csr(P_a), bundle['h_r'], max_time=space.max_time)
        return cache[a]

    def spk_at(self, space, state, a, bl_action=None, ordered_spaces=None):
        """One-call lookup: the SPK for `space`, action `a`, in the region it's in at the current
        system state (mirrors region_functions_at). Routes through spk_for -> region built on demand."""
        if not isinstance(state, dict):
            if ordered_spaces is None:
                raise ValueError("state is a vector -> also pass ordered_spaces (e.g. ss_list)")
            state = {sp.name: int(state[i]) for i, sp in enumerate(ordered_spaces)}
        key = self.region_of(space.name, state, bl_action)
        return self.spk_for(space, key, a, bl_action=(bl_action or 0))

    def region_functions_at(self, space, state, bl_action=None, ordered_spaces=None):
        """One-call lookup: the event-function bundle for `space` in the region it's in at the current
        system state. Hides the state->dict conversion, region_of, and region_functions indexing.
          space          : the space object (has .name and .region_functions).
          state          : EITHER a name-dict {space_name: state_index} OR a positional vector
                           (then also pass ordered_spaces, e.g. ss_list).
          bl_action      : the free BL action (int).
        Returns the region's event-function bundle {eta (STEF), kappa (CEF), sf (SF), xi (TED), nu (hazard)}.
        Routes through region_functions_for, so an unbuilt region is built + cached on demand (no KeyError)."""
        if not isinstance(state, dict):
            if ordered_spaces is None:
                raise ValueError("state is a vector -> also pass ordered_spaces (e.g. ss_list)")
            state = {sp.name: int(state[i]) for i, sp in enumerate(ordered_spaces)}
        key = self.region_of(space.name, state, bl_action)
        return self.region_functions_for(space, key, bl_action=bl_action)

    def global_tef(self, state, ordered_spaces, bl_action=0, max_time=None):
        """Global TEF xi_S^r(t_f | s): temporal distribution over the first system-wide event time,
        given the full initial state vector s = (s^k)_k. Computed as a product of per-space
        survival functions in their current regions).  From the paper:

            xi_S^r(t_f | s) = prod_k sf_k^r(s^k, t_f - 1) - prod_k sf_k^r(s^k, t_f)

        = P(every space still in-region through t_f-1) - P(... through t_f). Each space's survival
        function sf_k(s^k, .) is read from the region it is in at `state` (via region_functions_at),
        indexed at that space's own state s^k. Returns a 1D array xi over t_f = 0..T-1 (index xi[t_f]);
        T = min per-space horizon unless `max_time` is given. xi sums to 1 - S[T-1] where
        S(t) = prod_k sf_k(t) is the joint survival (defective if survival never hits 0).

        This method's job is RESOLUTION -- find which survival curve each space is on; the telescoping
        product itself is stok.tef_from_survivals, so callers holding curves already (e.g. a BL space
        with no OKBEKernelMixin) can compute the same xi_S without an AffordanceSet."""
        if not isinstance(state, dict):
            state = {sp.name: int(state[i]) for i, sp in enumerate(ordered_spaces)}
        per_space_survival = []
        for sp in ordered_spaces:
            bundle = self.region_functions_at(sp, state, bl_action=bl_action, ordered_spaces=ordered_spaces)
            per_space_survival.append(np.asarray(bundle['sf'])[state[sp.name]])   # sf_k(s^k, .) over t_f
        return stok.tef_from_survivals(per_space_survival, max_time=max_time)

    def sample_option(self, state, ordered_spaces, bl_action=0, n_samples=1, rng=None,
                      max_time=None, sed_sampler=None):
        """Forward-sample STOK option OUTCOMES from an initial joint state under one option (bl_action).
        Built for option-tree search: the shared structures (xi_S, each space's nu and region SPK)
        depend only on the initial state + option, so they are computed ONCE and reused across all
        n_samples draws. Each draw:
          1. t_f ~ xi_S  (self.global_tef); if the horizon is reached first the draw is 'censored'.
          2. sigma ~ SED(nu_k(t_f))  (which spaces make the first event at t_f; sed_sampler defaults
             to sed_sample, pass sed_sample_reject to compare).
          3. each space's terminal state ~ its region SPK at t_f: IN(event) if sigma_k=1 else OUT(survive).
        Does NOT classify death/goal/region-exit -- that is the caller's decision (the seam for your
        termination semantics). Returns a struct-of-arrays (columnar; ~5x lighter than a list of dicts,
        and vectorizable downstream), all indexed by draw n = 0..n_samples-1:
          {'t_f':      int32[n]   (-1 == censored, no event within the horizon),
           'sigma':    int8[n,K]  (which of the K spaces made the first event),
           'state':    int32[n,K] (terminal joint state, in ordered_spaces order),
           'censored': bool[n]}.
        View one draw with e.g. state[n], sigma[n]; dedup children with np.unique(state, axis=0)."""
        rng = np.random if rng is None else rng
        sed_fn = sed_sample if sed_sampler is None else sed_sampler
        if not isinstance(state, dict):
            state = {sp.name: int(state[i]) for i, sp in enumerate(ordered_spaces)}
        K = len(ordered_spaces)

        # shared precompute (fixed for this initial state + option)
        xi = self.global_tef(state, ordered_spaces, bl_action=bl_action, max_time=max_time)
        xi_total = float(xi.sum())                 # = 1 - S[-1]; leftover mass = no event in horizon
        cxi = np.cumsum(xi)                         # inverse-CDF sampling of t_f: O(log T)/draw, no realloc
        T = len(xi)
        # per space: (s_k, hazard_t, spk), in ordered_spaces order.
        # hazard_t = nu_k^r(s_k, .) over t_f -- the DISCRETE-TIME HAZARD of space k's first region-exit
        # from s_k: hazard_t[t] = P(k's first event fires at exactly t | it has not fired before t)
        # = xi_k^r(t|s_k) / sf_k^r(s_k, t-1). One row of bundle['nu'] ([s_i, t_f]), fixed for this option.
        per_space = []
        for sp in ordered_spaces:
            key = self.region_of(sp.name, state, bl_action)
            own_alpha = int(key[0])
            bundle = self.region_functions_for(sp, key, bl_action=bl_action)
            spk = self.spk_for(sp, key, own_alpha, bl_action=bl_action)
            s_k = state[sp.name]
            per_space.append((s_k, np.asarray(bundle['nu'])[s_k], spk))

        def landing(spk, s_k, io, t):              # sample a terminal state from SPK row s_k
            sl = spk[io, t]                        # 2D COO [ns, ns] (cached slice; list lookup, no numba)
            rows, cols, data = sl.coords[0], sl.coords[1], sl.data
            m = rows == s_k                        # plain-numpy row extract -- avoids pydata-sparse's
            idx, p = cols[m], np.asarray(data[m], float)   #   numba-JIT'd COO fancy indexing (the ~4s warmup)
            tot = p.sum()
            return s_k if tot <= 0 else int(rng.choice(idx, p=p / tot))   # no mass -> stay put

        # pre-allocated struct-of-arrays. fill row n per draw (no per-draw object churn)
        t_f_out = np.full(n_samples, -1, dtype=np.int32)     # -1 == censored
        sigma_out = np.zeros((n_samples, K), dtype=np.int8)
        state_out = np.empty((n_samples, K), dtype=np.int32)
        censored = np.zeros(n_samples, dtype=bool)
        hazard_vec = np.empty(K)   # the SED's argument: (nu_1(t_f), ..., nu_K(t_f)), i.e. every space's
                                   # hazard AT the drawn t_f. Scratch, overwritten per draw (sed_sample reads,
                                   # never mutates) -- so hazard_t is a ROW of nu (one space, all t),
                                   # hazard_vec is a COLUMN (all spaces, one t).
        for n in range(n_samples):
            u = rng.random()                                   # one uniform drives censoring AND the t_f draw
            if u >= xi_total:                                  # no event within the horizon -> censored
                censored[n] = True
                for k, (s_k, _, spk) in enumerate(per_space):
                    state_out[n, k] = landing(spk, s_k, SPK.OUT, T - 1)
                continue
            t_f = int(np.searchsorted(cxi, u, side='right'))   # inverse-CDF of the (defective) xi
            t_f_out[n] = t_f
            for k, (_, hazard_t, _) in enumerate(per_space):
                hazard_vec[k] = hazard_t[t_f]
            sig = np.asarray(sed_fn(hazard_vec, rng))
            sigma_out[n, :] = sig
            for k, (s_k, _, spk) in enumerate(per_space):
                state_out[n, k] = landing(spk, s_k, SPK.IN if sig[k] else SPK.OUT, t_f)
        return {'t_f': t_f_out, 'sigma': sigma_out, 'state': state_out, 'censored': censored}


# sed_sample / sed_sample_reject now live in stok.py (imported at the top of this module and
# re-exported, so `from state_spaces import sed_sample` still resolves).


# SPK family: when mass_eps>0, the O(nnz) mass-drain sum() is checked only every K steps (nnz==0,
# the free exact-drain check, still runs every step). mass_eps=0 (default) never calls sum() at all.
SPK_DRAIN_CHECK_EVERY = 16


class SPK:
    """Lazily-extendable conditional state-prediction kernel for one action's null kernel.

    Holds the normalized in/out SPK stacks over time and can extend past the initial horizon on
    demand. The full sequence is the Markov power series (diags(h) @ null_mat)^t; the only state
    needed to extend is the single running unnormalized power (the cursor), so growing from T to
    T+d costs d sparse matmuls with no recompute.

    Index convention IN=1 / OUT=0 matches the nu-Bernoulli draw (1 = a within-space event fired):
      spk[1, t]  ->  in-SPK  at time t: event fired WITHIN the space ((1-h) mass, normalized) [lazy-extends]
      spk[0, t]  ->  out-SPK at time t: survived in-region (h mass, normalized)                [lazy-extends]
      spk.in_(t) / spk.out(t)      readable aliases for spk[1, t] (event) / spk[0, t] (survive)
      spk.in_stack(up_to) / spk.out_stack(up_to)   COO stack [0..up_to] over time
      spk.as_dict(dense=False)     {IN: in_stack, OUT: out_stack} up to current horizon

    Values match the old eager compute_cond_spk_normalized exactly (t=0 = raw identity for both;
    t>=1 = row-normalized projections of the cursor) -- only the IN/OUT index labels differ.
    """
    IN, OUT = 1, 0

    def __init__(self, null_mat, h, max_time=1, mass_eps=0.0):
        self.h = np.asarray(h).astype(float)
        self.mass_eps = mass_eps                       # stop extending once cursor mass <= this
        self._Dh = sp.sparse.diags(self.h)
        self._D1mh = sp.sparse.diags(1.0 - self.h)
        self._step = self._Dh @ csr(null_mat)          # diags(h) @ null_mat: the per-step operator
        self._cursor = csr(self._step ** 0)            # running unnormalized power == (step)^built
        self._built = 0                                # highest time index materialized
        self._exhausted_at = None                      # index where mass drained (all further slices ~0)
        ident = sparse.COO.from_scipy_sparse(csr(self._step ** 0))
        self._in = [ident]                             # t=0: raw identity (matches legacy)
        self._out = [ident]
        if max_time > 1:
            self._extend_to(max_time - 1)

    @staticmethod
    def _normalize(m):
        row_sums = np.array(m.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        # division yields np.matrix; csr() coerces back so COO.from_scipy_sparse accepts it
        return sparse.COO.from_scipy_sparse(csr(m / row_sums[:, np.newaxis]))

    def _drained(self):
        # nnz==0 (free) catches exact drain every step; the O(nnz) sum() runs only when eps-drain is
        # requested AND at every K-th step. default mass_eps=0.0 -> no sum() cost at all (lossless).
        if self._cursor.nnz == 0:
            return True
        if self.mass_eps > 0 and self._built % SPK_DRAIN_CHECK_EVERY == 0:
            return self._cursor.sum() <= self.mass_eps
        return False

    def _extend_to(self, t):
        while self._built < t:
            if self._exhausted_at is not None:                # drained: no more nonzero slices to build
                break
            self._cursor = self._cursor.dot(self._step)       # one sparse matmul: cursor = step^(built+1)
            self._built += 1
            self._out.append(self._normalize(self._cursor @ self._Dh))
            self._in.append(self._normalize(self._cursor @ self._D1mh))
            if self._drained():
                self._exhausted_at = self._built              # this slice and all beyond are ~0

    def ensure(self, t):
        """Make time index t available (extend if needed)."""
        if t > self._built:
            self._extend_to(t)

    @property
    def built(self):
        return self._built

    @property
    def exhausted_at(self):
        """Time index at which the region drained (all slices >= this are ~0), or None."""
        return self._exhausted_at

    @property
    def exhausted(self):
        return self._exhausted_at is not None

    def _idx(self, t):
        # extend if possible, then clamp: past the drain point every slice equals the zero slice,
        # so we never materialize (or matmul) beyond it.
        self.ensure(t)
        return min(t, self._built)

    def __getitem__(self, key):
        io, t = key
        i = self._idx(t)
        return (self._in if io == self.IN else self._out)[i]

    def in_(self, t):
        return self._in[self._idx(t)]

    def out(self, t):
        return self._out[self._idx(t)]

    def in_stack(self, up_to=None):
        up_to = self._built if up_to is None else self._idx(up_to)
        return sparse.stack(self._in[:up_to + 1], axis=0)

    def out_stack(self, up_to=None):
        up_to = self._built if up_to is None else self._idx(up_to)
        return sparse.stack(self._out[:up_to + 1], axis=0)

    def as_dict(self, up_to=None, dense=False):
        """{IN: in_stack, OUT: out_stack} as [T, ns, ns] COO (or dense), keyed by the IN/OUT convention."""
        i, o = self.in_stack(up_to), self.out_stack(up_to)
        return {self.IN: (i.todense() if dense else i), self.OUT: (o.todense() if dense else o)}


class SPKAdaptive:
    """ADAPTIVE per-source SPK, kept SEPARATE for A/B comparison (NOT wired into the pipeline).

    For tree search where you discover start states as you go: sources are added lazily on first
    query, and each source is extended lazily in time -- two-level laziness (source x time). ONE
    shared step operator; per-source state = {cursor (1 x ns), built, in/out slice lists, drain}.
    Each source propagates as a single row (matvec), so cost is paid only for states you actually
    visit and only up to the times you actually ask for. Same IN=1(event)/OUT=0(survive) convention
    and drain guard as SPK; a source's row at time t equals row s of the full SPK at t.

      spk.get(s, io, t)   -> 1 x ns slice for start-state s (builds source s + extends time on demand)
      spk.has_source(s)   -> whether source s has been created yet   <- the 'computed already?' check
      spk.built(s)        -> current time horizon for source s (-1 if never touched)
      spk.sources()       -> list of sources built so far
    """
    IN, OUT = 1, 0

    def __init__(self, null_mat, h, mass_eps=0.0):
        self.h = np.asarray(h).astype(float)
        self.mass_eps = mass_eps
        self.ns = int(csr(null_mat).shape[0])
        self._Dh = sp.sparse.diags(self.h)
        self._D1mh = sp.sparse.diags(1.0 - self.h)
        self._step = self._Dh @ csr(null_mat)      # one shared operator for ALL sources
        self._src = {}                             # s -> {cursor, built, in, out, exhausted_at}

    @staticmethod
    def _normalize(m):
        row_sums = np.array(m.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        return sparse.COO.from_scipy_sparse(csr(m / row_sums[:, np.newaxis]))

    def _state(self, s):
        st = self._src.get(s)
        if st is None:
            e = sp.sparse.csr_matrix(([1.0], ([0], [int(s)])), shape=(1, self.ns))   # e_s row
            coo = sparse.COO.from_scipy_sparse(csr(e))
            st = {'cursor': csr(e), 'built': 0, 'in': [coo], 'out': [coo], 'exhausted_at': None}
            self._src[s] = st
        return st

    def _extend(self, st, t):
        while st['built'] < t:
            if st['exhausted_at'] is not None:
                break
            st['cursor'] = st['cursor'].dot(self._step)         # 1 x ns matvec
            st['built'] += 1
            st['out'].append(self._normalize(st['cursor'] @ self._Dh))
            st['in'].append(self._normalize(st['cursor'] @ self._D1mh))
            if st['cursor'].nnz == 0 or (self.mass_eps > 0 and st['built'] % SPK_DRAIN_CHECK_EVERY == 0
                                         and st['cursor'].sum() <= self.mass_eps):
                st['exhausted_at'] = st['built']

    def get(self, s, io, t):
        st = self._state(s)                 # lazily create source s
        self._extend(st, t)                 # lazily extend it in time
        i = min(t, st['built'])             # clamp past drain
        return (st['in'] if io == self.IN else st['out'])[i]

    def has_source(self, s):
        return s in self._src

    def built(self, s):
        return self._src[s]['built'] if s in self._src else -1

    def exhausted_at(self, s):
        return self._src[s]['exhausted_at'] if s in self._src else None

    def sources(self):
        return list(self._src)


class SPKEval:
    """MEMORYLESS SPK evaluator, kept SEPARATE for A/B comparison (NOT wired into the pipeline).

    For when you rarely revisit a state, so caching trajectories isn't worth it: holds ONLY the shared
    step operator and computes the in/out distribution from a source at a single time by a fresh O(t)
    row propagation -- nothing is stored between calls (no per-source state, no slice history). Same
    IN=1(event)/OUT=0(survive) convention; the result equals SPK[io,t][source] exactly.

      ev.eval(source, io, t)  -> 1 x ns distribution (source = state index OR an entry-distribution row)
      ev.eval_both(source, t) -> (in_dist, out_dist)
    """
    IN, OUT = 1, 0

    def __init__(self, null_mat, h, mass_eps=0.0):
        self.h = np.asarray(h).astype(float)
        self.mass_eps = mass_eps
        self.ns = int(csr(null_mat).shape[0])
        self._Dh = sp.sparse.diags(self.h)
        self._D1mh = sp.sparse.diags(1.0 - self.h)
        self._step = self._Dh @ csr(null_mat)

    @staticmethod
    def _normalize_row(v):
        s = v.sum()
        if s == 0:
            s = 1.0
        return sparse.COO.from_scipy_sparse(csr(v / s))

    def _propagate(self, source, t):
        if np.isscalar(source):
            v = csr(sp.sparse.csr_matrix(([1.0], ([0], [int(source)])), shape=(1, self.ns)))
        else:
            v = csr(np.atleast_2d(np.asarray(source, float)))
        for i in range(t):
            if v.nnz == 0:                                   # ~free: exact drain / nilpotent, every step
                break
            if self.mass_eps > 0 and i % SPK_DRAIN_CHECK_EVERY == 0 and v.sum() <= self.mass_eps:
                break                                        # O(nnz) sum(): only when eps-drain, every K
            v = v.dot(self._step)                            # one 1 x ns matvec, no storage
        return v

    def eval(self, source, io, t):
        v = self._propagate(source, t)
        if t == 0:
            return sparse.COO.from_scipy_sparse(csr(v))      # raw seed row (matches SPK t=0 identity)
        proj = v @ (self._D1mh if io == self.IN else self._Dh)   # IN=event=(1-h), OUT=survive=h
        return self._normalize_row(proj)

    def eval_both(self, source, t):
        return self.eval(source, self.IN, t), self.eval(source, self.OUT, t)


class OKBEKernelMixin:
    """Capability (mixin): OKBE quantities computed from a per-action kernel. A host class must set up
    self.P_a_s_s [na, ns, ns] (dense ndarray OR sparse per-action), self.ns, and self.max_time. Provides
    the kernel-generic OKBE methods so every state-space (satiation, bit, wealth, task, grid, ...) shares
    ONE implementation instead of duplicating it. This is capability, not identity -- it holds no state
    and defines no __init__; each host builds its own kernel and 'mixes in' these methods."""

    def compute_STEF_from_hr(self, h_r, alpha_ind):
        """STEF from the continuation vector h_r (the region-indicator product from build_h_r), for the
        region driven by alpha_ind. Kernel-generic: densifies the per-action slice when it is sparse.
        Unifies the former Internal_StateSpace (dense) and Task (sparse-densifying) copies -- with the
        sparse-safe densify AND the correct both-array padding when the time axis is extended.
        The recursion itself is stok.stef_from_hr; this method only supplies the host's kernel/sizes."""
        return stok.stef_from_hr(self.P_a_s_s[alpha_ind], h_r, self.max_time, ns=self.ns)

    def compute_STEF(self, region_state, alpha_ind):
        """STEF for the region CONTAINING region_state, deriving the continuation from self.zeta (the
        mode partition): h_r = 1 off the region, 0 on it. Then defers to the unified compute_STEF_from_hr
        recursion (no duplicated math). Unifies the former Internal_StateSpace/Task copies."""
        unique_values = np.unique(self.zeta)
        indices_dict = {val: np.where(self.zeta == val)[0] for val in unique_values}
        region = next((k for k, v in indices_dict.items() if region_state in v), None)
        h_r = np.ones(self.ns)
        h_r[indices_dict[region]] = 0
        return self.compute_STEF_from_hr(h_r, alpha_ind)

    def compute_SPK_tensor(self, max_time):
        """SPK tensor [na, ns, ns, max_time] with SPK[a,:,:,t] = P_a^t. Reads the per-action kernel dict
        under whichever name the host uses (P_s_s_given_g_dict on StateSpace_1D, P_a_s_s_dict on Task)."""
        kdict = getattr(self, "P_s_s_given_g_dict", None)
        if kdict is None:
            kdict = getattr(self, "P_a_s_s_dict", None)
        matrix_powers = [
            [sp.sparse.identity(mat.shape[0])] + [reduce(lambda acc, _: acc.dot(mat), range(t), mat) for t in range(max_time)]
            for a, mat in kdict.items()
        ]
        tensor_coords, tensor_data = [], []
        for a in range(self.na):
            for t in range(max_time):
                coo_mat = matrix_powers[a][t].tocoo()
                coords = np.vstack((np.full_like(coo_mat.row, a), coo_mat.row, coo_mat.col,
                                    np.full_like(coo_mat.row, t)))
                tensor_coords.append(coords)
                tensor_data.append(coo_mat.data)
        tensor_coords = np.concatenate(tensor_coords, axis=1)
        tensor_data = np.concatenate(tensor_data)
        return sparse.COO(tensor_coords, tensor_data, shape=(self.na, self.ns, self.ns, max_time))

    def event_functions_from_stef(self, eta):
        """Derive the rest of the event-function bundle from a STEF eta[s_i, s_f, t_f]:
        kappa (CEF), sf (SF), xi (TED), nu (hazard). Pure function of eta -- so it lives in
        stok.event_functions_from_stef; this method is kept as the mixin-facing alias."""
        return stok.event_functions_from_stef(eta)


class StateSpace_1D(OKBEKernelMixin):
    """1D counter/line state space (states 0..size-1) that supplies the common kernel setup for the
    OKBE family. With an explicit transition_kernel it accepts any [na, ns, ns] kernel; with no kernel
    it defaults to the decrement-by-one drift + jump-up dynamics (the satiation default). Physio /
    Bit / Wealth are specific instances. OKBE methods come from OKBEKernelMixin."""
    def __init__(self, size, name, max_time, action_type_ind, constraint_vec, jump='max', defective_states=[0], compute_tcv_cdf=False, initial_state=None, zero_thresh=1E-4, color=BLACK, identity_epsilon=0, transition_kernel=None, default_action_ind=0):
        # transition_kernel (optional): a general P(s'|s, alpha) as a [na, ns, ns] array (dense
        #   ndarray or anything csr() accepts per-action). If provided, na is derived from it and
        #   the legacy 2-action internal chain below is bypassed. If None, the original
        #   decrement-by-one default + jump-up task-action chain is built (na = 2).
        # default_action_ind: which action index acts as the region default dynamics (omega_r);
        #   defaults to 0, matching the legacy null-action.
        if jump == 'max':
            self.jump = size
        else:
            self.jump = jump
        if initial_state == None:
            self.current_state = size-2  # Initial state
        self.ns = size
        self.color = color
        self.identity_epsilon = identity_epsilon  # Modified chance of the state transitioning to itself, defaults to zero.  (This does not overwrite the abosrbing state at the bottom of the chain)
        self.action_type_ind = action_type_ind
        self.zeta = np.ones(size) # Maps y -> e for every possible y, where y is an internal state and e is a mode parameter. 0 is alive mode, 1 is death mode,
        self.zeta[defective_states] = 0  # Sets state w_0 to the death state.
        self.mode_to_states_dict = dict(zip(np.array(np.sort(np.unique(self.zeta)), dtype=int), [np.where(self.zeta == num)[0] for num in np.unique(self.zeta)]))
        self.states_to_mode_dict = dict()
        self.all_inds = np.arange(self.ns)
        self.constraint_vec = constraint_vec
        self.constraint_inds = np.where(constraint_vec == 0)[0]
        self.non_constraint_inds = np.setdiff1d(self.all_inds, self.constraint_inds)
        for i in range(len(self.zeta)):
            self.states_to_mode_dict[i] = self.zeta[i]
        self.name = name
        self.max_time = max_time
        # self.na is derived from the transition operator after it is built (see below).
        self.defective_states = defective_states
        self.null_action = 0
        self.task_action = 1
        self.task_ind_2_goal = {action_type_ind: self.task_action}  # This map takes the task_ind and returns the goal (action) ind for P_gv_v
        # Task action is the index for the operator P_y, task_action_ind is the index over all task-goals
        self.action_type_ind = action_type_ind  # This is the index of the active action in the set of active actions.
        # self.P_gv_v = np.zeros([self.na, self.ns, self.ns])  # v is the generic letter for an internal state.
        # self.P_s_s_given_g_dict = dict([])

        # self.P_gv_v[0, 0, 0] = 1
        data = np.ones(self.ns)
        row = np.array(range(self.ns))
        self.P_s_s_given_g_dict = dict()
        if transition_kernel is not None:
            # General path: accept any P(s'|s, alpha) given as a [na, ns, ns] kernel.
            tk = np.asarray(transition_kernel)
            for a in range(tk.shape[0]):
                self.P_s_s_given_g_dict[a] = csr(tk[a])
        else:
            # Legacy 2-action internal-state chain: action 0 = decrement-by-one default
            # (the region default dynamics), action 1 = jump-up task action.
            col = np.maximum(row-1, np.zeros(len(row))).astype(int)
            null_mat = csr((data, (row, col)), shape=(self.ns, self.ns))
            null_mat = (1 - self.identity_epsilon) * null_mat + self.identity_epsilon * sp.sparse.identity(self.ns)
            self.P_s_s_given_g_dict[0] = null_mat
            col = np.minimum(row + self.jump, np.ones(len(row)) * (self.ns - 1)).astype(int)
            self.P_s_s_given_g_dict[1] = csr((data, (row, col)), shape=(self.ns, self.ns))

        self.na = len(self.P_s_s_given_g_dict)        # na is derived from the operator, not hardcoded
        self.default_action_ind = default_action_ind  # which action is the region default (omega_r)

        coo_mats = map(sparse.COO.from_scipy_sparse, self.P_s_s_given_g_dict.values())
        self.P_g_v_v_sparse_tensor = sparse.stack(list(coo_mats), axis=0)

        if self.ns * self.ns * max_time < 5e7:
            self.P_a_s_s = np.zeros([self.na, self.ns, self.ns])
            for a in range(self.na):
                self.P_a_s_s[a, :, :] = self.P_s_s_given_g_dict[a].todense()

        # self.P_s_s_given_g_dict[0] = null_mat
        # for i in range(1, self.ns):
        #     self.P_gv_v[0, i, i - 1] = 1

        # self.P_gv_v[1, 0, np.min([i + jump, self.ns - 1])] = 1
        # for i in range(0, self.ns):
        #     self.P_gv_v[1, i, np.min([i + jump, self.ns - 1])] = 1
        # RETIRED (2026-07-10): the omega Pᵗ-stack (null_omega / SPK_lst_CSR_unnormalized) and the
        # expected-time-to-death solve (time_to_mode_switch_dict). NONE are used by the STOK option
        # sampler: the lazy SPK class (spk_alpha_list, below) supersedes null_omega, and the STEF's
        # sf(t) subsumes the expected hitting time (E[T] = sum_t sf(t), verified equal). They ran at
        # EVERY space construction for nobody, and the time-to-death spsolve threw a "singular matrix"
        # warning on no-death spaces (e.g. money, whose hold-kernel makes I-P = 0). Only the legacy
        # PNAS/SPA/TGMDP scripts read these -- un-comment the three lines below to revive one. Left as
        # None (not deleted) so attribute access resolves instead of AttributeError.
        self.null_omega = self.null_omega_dense = self.SPK_lst_CSR_unnormalized = None
        self.time_to_mode_switch_dict = None
        # self.null_omega, self.null_omega_dense = self.compute_omega_prediction(csr(self.P_s_s_given_g_dict[self.default_action_ind]), self.max_time, with_dense=True)  # [t_d,y_s,y_f]
        # self.SPK_lst_CSR_unnormalized = [self.compute_omega_prediction(csr(self.P_s_s_given_g_dict[i]), self.max_time, with_dense=False) for i in range(self.na)]
        # self.time_to_mode_switch_dict = self.compute_all_time_until_mode_switch(self.P_s_s_given_g_dict)  # PNAS_paper.py only -- retire with it

        # Lazily-extendable normalized SPK \rho(s_f|s,t_f) per action on space S_k. Index spk[0,t] (in)
        # / spk[1,t] (out); if t exceeds the current horizon it extends on demand (one matmul/step).
        # COMMENTED OUT 2026-08-24 (see the LEGACY note below): 99% of construction time, and both
        # are optional -- spk_for rebuilds an SPK on demand, and SPK_tensor has one legacy reader.
        # self.spk_alpha_list = [
        #     SPK(csr(self.P_s_s_given_g_dict[i]), self.constraint_vec, max_time=self.max_time) for i in
        #     range(self.na)]
        # self.SPK_tensor = self.compute_SPK_tensor(max_time)
        self.spk_alpha_list = None
        self.SPK_tensor = None
        self.region_functions = dict()   # region-key -> {eta (STEF), kappa (CEF), sf (SF), xi (TED), nu (hazard)}

        # ---- LEGACY eager blocks, COMMENTED OUT 2026-08-24 ---------------------------------------
        # Profiling StateSpace_1D.__init__ (1.75s for 3 constructions) showed where the time actually
        # went -- NOT the block the "STEF compute time" print was labelling:
        #     compute_SPK_tensor   1.22s  (70%)   <- and it is O(max_time^2): `reduce(range(t))`
        #                                            recomputes each matrix power from scratch
        #     spk_alpha_list       0.51s  (29%)
        #     the zeta STEF itself:  negligible
        # All three are EAGER on every construction, and nothing in the live pipeline reads them:
        #   * live STEFs come from AffordanceSet.region_functions_for -> stok.stef_from_hr, per region
        #   * spk_for (line ~269) treats spk_alpha_list as an OPTIMISATION and rebuilds if absent
        # The only remaining readers are legacy: TGMDP_methods.dfs_sparse_tensor* (eta_event_0, its
        # okbe_main call site is commented out) and OKBE_code/PNAS_paper.py (eta_event_0 / SPK_tensor /
        # kappa_compliment_0). Those will raise on None at the point of use, which is the intent: fail
        # loudly rather than silently hand back a zero array.
        #
        # NB the normalization below was ALSO the flagged bug -- forcing Sum=1 per start state drives
        # kappa_bar to 0 wherever the true event probability is < 1, and nu divides by kappa_bar. Do not
        # revive it as-is; stok.event_functions_from_stef deliberately does not normalize.
        #
        # starttime = timeit.default_timer()
        # self.eta_event_0 = self.compute_STEF(0, 0)
        # self.eta_event_0[self.eta_event_0 < zero_thresh] = 0
        # normalizer = self.eta_event_0.sum(axis=(1, 2), keepdims=True)
        # normalizer[normalizer == 0] = 1
        # self.eta_event_0 = self.eta_event_0 / normalizer
        # cumsum_eta = np.cumsum(self.eta_event_0, axis=2)
        # self.kappa_event_0 = cumsum_eta.sum(axis=1)
        # self.kappa_compliment_0 = 1 - self.kappa_event_0
        # print("STEF compute time: ", timeit.default_timer() - starttime)
        self.eta_event_0 = self.kappa_event_0 = self.kappa_compliment_0 = None

        default_P = self.P_a_s_s[0, :, :]
        # Guard: build TCV only for the ONE shape compute_tcv_cdf actually supports -- exactly one
        # constraint state and >1 non-constraint state. Its len(constraint_inds)>1 branch has never
        # worked (P_NT is then 2-D [n_non, n_con]: line ~855 assigns it into a 1-D slice, and ~859
        # sums the wrong axis then indexes a length-n_con vector with non-constraint indices), so
        # any space with 2+ exit states crashed here during __init__ -- e.g. space C in
        # test_stok_sampler.py (c=[0,1,1,1,0]), which blocked the flat ground-truth validation.
        # TCV_CDF_mat feeds ONLY PNAS_paper.py (always single-constraint), so None is safe for
        # everything else; this is behaviour-preserving for every case that worked before.
        if len(self.constraint_inds) == 1 and len(self.non_constraint_inds) > 1:
            self.TCV_CDF_mat = self.compute_tcv_cdf(default_P, self.non_constraint_inds, self.constraint_inds, self.ns*3)  # PNAS_paper.py only -- retire with it
        else:
            self.TCV_CDF_mat = None   # TCV degenerate/unsupported (no constraint, 2+ constraints, or <2 non-constraint states e.g. a bit)


        if compute_tcv_cdf:
            pass

    # comptute_TEF (xi/TED) removed: subsumed by OKBEKernelMixin.event_functions_from_stef.

    # compute_SPK_tensor and compute_STEF now provided by OKBEKernelMixin (unified copies).


    # compute_STEF_from_hr now provided by OKBEKernelMixin (unified dense/sparse version).

    def compute_tcv_cdf(self, P_default, non_constraint_inds, constraint_inds, max_time):
        # D_constraint = np.diag(constraint_vec)
        # Returns TCV_CDF_mat which is a T x S matrix which gives you the probability that you'll violate
        # the constraint after t time steps when starting in s_i.
        P_NN = np.squeeze(P_default[non_constraint_inds, :][:, non_constraint_inds])
        P_NT = np.squeeze(P_default[non_constraint_inds, :][:, constraint_inds])
        init_mat = np.linalg.matrix_power(P_NN, 0)
        P_prod_list = [init_mat]
        for t in range(1, max_time + 1):
            P_prod_list.append(np.dot(P_prod_list[t - 1], P_NN))

        P_sum_list = [init_mat]
        for t in range(1, max_time + 1):
            P_sum_list.append(P_sum_list[t - 1] + P_prod_list[t])

        TCV_CDF_w_f_state = [P_NT]
        for t in range(1, max_time + 1):
            TCV_CDF_w_f_state.append(P_sum_list[t - 1].dot(P_NT))

        if len(self.constraint_inds) > 1:
            final_vec = np.ones(self.ns)
            final_vec[self.non_constraint_inds] = P_NT
            TCV_CDF = [final_vec]
            for t in range(1, max_time + 1):
                non_constraint_vec = P_sum_list[t - 1].dot(P_NT).sum(axis=0)
                final_vec[self.non_constraint_inds] = non_constraint_vec[self.non_constraint_inds]
                TCV_CDF.append(final_vec)
        else:
            final_vec = np.ones(self.ns)
            final_vec[self.non_constraint_inds] = P_NT
            TCV_CDF = [final_vec]
            for t in range(1, max_time + 1):
                final_vec = np.ones(self.ns)
                non_constraint_vec = P_sum_list[t - 1].dot(P_NT)
                final_vec[self.non_constraint_inds] = non_constraint_vec
                TCV_CDF.append(final_vec)

        TCV_CDF_mat = np.array(TCV_CDF)
        return TCV_CDF_mat

    def get_goal_ind(self, task_ind):
        val = self.task_ind_2_goal.get(task_ind, 0)
        if task_ind in self.task_ind_2_goal.keys():
            return self.task_ind_2_goal[task_ind]
        else:
            return 0

    def compute_omega_prediction(self, null_mat, max_time, with_dense=False):
        prev_mat = csr(null_mat**0)
        mat_dict = dict()
        mat_dict[0] = prev_mat
        omega_tensor = None
        if with_dense == True:
            ns = null_mat.shape[0]
            omega_tensor = np.zeros([max_time, ns, ns])
            omega_tensor[0, :, :] = prev_mat.todense()
            for t in range(1, max_time):
                prev_mat = prev_mat.dot(null_mat)
                mat_dict[t] = prev_mat
                omega_tensor[t, :, :] = prev_mat.todense()
        else:
            for t in range(1, max_time):
                prev_mat = prev_mat.dot(null_mat)
                mat_dict[t] = prev_mat
        return mat_dict, omega_tensor

    def compute_cond_spk_normalized(self, null_mat, h, max_time, with_dense=False):
        # Legacy eager shape {0: in_stack, 1: out_stack}, now backed by the lazy SPK (single source
        # of truth for the normalization). Prefer building an SPK directly if you may extend in time.
        return SPK(null_mat, h, max_time=max_time).as_dict(dense=with_dense)

    def forcast_null_forward_loop(self, vec, mat, time):
        for t in time:
            vec = vec.dot(mat)

        return vec

    def forcast_null_mat_power(self, vec, mat, time):
        return vec.dot(mat**time)

    def compute_all_time_until_mode_switch(self, P_v_v_given_g_dict):  # PNAS_paper.py only -- retire with it
        ttg_list = []
        for mode in self.mode_to_states_dict.keys():
            print(mode)
            mode_states = self.mode_to_states_dict[mode]
            ns = P_v_v_given_g_dict[0].shape[0]
            full_inds = np.arange(ns)
            time_vec = np.zeros(ns)
            # mode_state_inds = [list(mode_states).index(i) for i in mode_states]
            # mode_states = np.delete(full_inds, mode_states)
            # mode_states
            null_mat = csr(P_v_v_given_g_dict[0][mode_states, :][:, mode_states])
            # if not np.linalg.det(np.eye(len(mode_states), len(mode_states)) - null_mat) == 0:  # If matrix is singular, then the ttg must be one for the states.
            # warnings.filterwarnings("error")
            if null_mat.size > 1:
                try:
                    mat = csr(sp.sparse.eye(len(mode_states), len(mode_states), format='csr') - null_mat)
                    ttg = sp.sparse.linalg.spsolve(mat, csr(np.ones(len(mode_states))).transpose())
                    time_vec[mode_states] = ttg
                    ttg_list.append(time_vec)
                except:
                    time_vec = np.zeros(ns)
                    for ms in mode_states:
                        time_vec = np.insert(time_vec, ms, np.inf)
                    # warnings.resetwarnings()
                    ttg_list.append(time_vec)
            else:
                time_vec = np.zeros(ns)
                for ms in mode_states:
                    time_vec = np.insert(time_vec, ms, np.inf)
                # warnings.resetwarnings()
                ttg_list.append(time_vec)
        return dict(zip(self.mode_to_states_dict.keys(), ttg_list))

class BitSpace(StateSpace_1D):
    """A single boolean state s in {0,1} as a minimal Internal_StateSpace (ns=2), for a factored
    (per-bit) task representation instead of one 2^n boolean hypercube. Bit-to-bit routing is done
    with affordances whose parent is another bit.

    The 4 deterministic memoryless bit-maps are the actions; TOGGLE (s'=1-s) is the ONLY non-static
    (period-2) one -- the sole recurrent pattern a memoryless single bit can have. Default action =
    HOLD (identity) = region drift. Pass constraint_vec to mark a bit value as a violation (default:
    neither), and defective_states=[] by default so bit=0 is NOT treated as a 'death' state."""
    HOLD, SET, CLEAR, TOGGLE = 0, 1, 2, 3

    def __init__(self, name, max_time, action_type_ind, constraint_vec=None,
                 initial_state=0, defective_states=None, **kw):
        if defective_states is None:
            defective_states = []                      # benign task bit: no death state (unlike depletable spaces)
        if constraint_vec is None:
            constraint_vec = np.ones(2)                # neither value is a violation
        P = np.stack([self._bit_kernel(a) for a in (self.HOLD, self.SET, self.CLEAR, self.TOGGLE)])  # [na=4, s, s']
        super().__init__(2, name, max_time, action_type_ind, np.asarray(constraint_vec, float),
                         defective_states=defective_states, transition_kernel=P,
                         initial_state=initial_state, default_action_ind=self.HOLD, **kw)
        self.current_state = 0 if initial_state is None else int(initial_state)   # base only sets it when None

    @staticmethod
    def _bit_kernel(a):
        # P[s, s'] = P(s' | s); rows = current, cols = next (matches P_a_s_s[alpha, s, s'])
        return {
            BitSpace.HOLD:   np.array([[1., 0.], [0., 1.]]),   # s' = s        (identity)
            BitSpace.SET:    np.array([[0., 1.], [0., 1.]]),   # s' = 1        (absorbing at 1)
            BitSpace.CLEAR:  np.array([[1., 0.], [1., 0.]]),   # s' = 0        (absorbing at 0)
            BitSpace.TOGGLE: np.array([[0., 1.], [1., 0.]]),   # s' = 1 - s    (the only oscillator)
        }[a]

class WealthSpace(StateSpace_1D):
    """A resource counter s in {0..size-1} (money, charge, inventory, ...) as a thin Internal_StateSpace,
    so it inherits ALL the OKBE machinery (SPK / SF / STEF via compute_STEF_from_hr, etc.). Unlike a
    satiation space it has NO drift: the actions are NULL (hold), DEPOSIT (+1, capped), WITHDRAW (-1,
    floored), SPEND (-price, floored) -- SPEND is the priced-exchange half (e.g. pay to open a door).
    Default action = NULL (region drift = 'hold', not decay). By default 0 is NOT a death state
    (defective_states=[]); pass constraint_vec/defective_states to mark levels as violation/death."""
    NULL, DEPOSIT, WITHDRAW, SPEND = 0, 1, 2, 3

    def __init__(self, size, name, max_time, action_type_ind, constraint_vec=None,
                 initial_state=0, defective_states=None, spend_amount=1, **kw):
        if defective_states is None:
            defective_states = []                      # a resource at 0 is benign by default
        if constraint_vec is None:
            constraint_vec = np.ones(size)             # no level is a violation
        self.spend_amount = int(spend_amount)          # the SPEND action subtracts this many units
        P = np.zeros((4, size, size))
        for s in range(size):
            P[self.NULL, s, s] = 1.0                    # hold
            P[self.DEPOSIT, s, min(s + 1, size - 1)] = 1.0        # +1, capped at top
            P[self.WITHDRAW, s, max(s - 1, 0)] = 1.0             # -1, floored at 0
            P[self.SPEND, s, max(s - self.spend_amount, 0)] = 1.0  # pay the price, floored at 0
        super().__init__(size, name, max_time, action_type_ind, np.asarray(constraint_vec, float),
                         defective_states=defective_states, transition_kernel=P,
                         initial_state=initial_state, default_action_ind=self.NULL, **kw)
        self.current_state = 0 if initial_state is None else int(initial_state)   # base only sets it when None

class Physio_StateSpace(StateSpace_1D):
    """Physiological / satiation counter (hunger, hydration, temperature, ...): a specific instance of
    StateSpace_1D whose dynamics are the decrement-by-one drift + jump-up refill (the 1D default).
    Sibling of BitSpace / WealthSpace, not their parent."""
    pass

# Backward-compat alias: the satiation space used to be called Internal_StateSpace. Existing callers
# (okbe_main, PNAS, goals_and_tasks, ...) keep working; new code should use Physio_StateSpace / StateSpace_1D.
Internal_StateSpace = Physio_StateSpace


class LinearStateSpace:
    def __init__(self, size, actions=[0, 10], max_drift_time = 100, create_absorb_state = True, name=None, buildTM=True, keep_dense=False):
        # max_drift_time sets the upper limit on the number of matrix powers taken for the drift_mat e.g. drift_mat = D: D^0, D^1, D^2, ... , D^(max_drft_time).
        # absorbing_terminal index is assumed to be 0
        self.name = name
        self.create_absorb_state = create_absorb_state
        self.ns = size  # number of states
        # self.actionsNames = ['S', 'N', 'U', 'D', 'L', 'R']  # 'S' is the 'special' goal action
        self.s_range = [i for i in range(self.ns)]
        self.actions = actions # Actions for a 1D space just jump the state n-spaces
        self.na = len(self.actions)
        self.action_nums = list(range(self.na))
        self.act_nums_2_actions = dict(zip(self.action_nums, self.actions))
        self.max_drift_time = max_drift_time
        self.T_s_s_a, self.T_sa_s = self.build_sa_s_tm(passive_drift=-1)
        self.drift_mat = csr(self.T_s_s_a[:,:,0])
        self.drift_mat_pow_list = self.create_drift_mat_power_list(self.drift_mat, self.max_drift_time)

    def build_sa_s_tm(self, passive_drift):
        T_s_s_a = np.zeros([self.ns, self.ns, self.na])
        T_sa_s = np.zeros([self.ns * self.na, self.ns])
        for i in range(self.ns * self.na):
            s_ind = i // self.na
            a_ind = i % self.na
            s_prime = max(min(s_ind + self.actions[a_ind] + passive_drift, self.ns - 1), 0)
            if self.create_absorb_state and s_ind == 0:
                T_sa_s[i, s_ind] = 1
                T_s_s_a[s_ind, s_ind, a_ind] = 1
            else:
                T_sa_s[i, s_prime] = 1
                T_s_s_a[s_ind, s_prime, a_ind] = 1

        return T_s_s_a, T_sa_s

    def create_drift_mat_power_list(self, drift_mat, max_drift_time):
        drift_pow_list = [drift_mat.power(0)]
        for t in range(max_drift_time):
            drift_pow_list.append(drift_pow_list[-1] * drift_mat)

        return drift_pow_list

    def average_drift_matrix_power(self, time_dist):
        if time_dist.sum() != 1:
            warnings.warn("Time Distribution doesn't sum to 1")
        average_power_mat = csr(np.zeros(list(self.drift_mat_pow_list[-1].shape)))
        for t in range(len(time_dist)):
            if time_dist[t]:
                average_power_mat = average_power_mat + self.drift_mat_pow_list[t] * time_dist[t]
        return average_power_mat


class StateSpace_env:
    # This is a 1D state-space for environment indices
    def __init__(self, size, actions=[0, 10], max_drift_time = 100, create_absorb_state = True, name=None, buildTM=True, keep_dense=False):
        # max_drift_time sets the upper limit on the number of matrix powers taken for the drift_mat e.g. drift_mat = D: D^0, D^1, D^2, ... , D^(max_drft_time).
        # absorbing_terminal index is assumed to be 0
        self.name = name
        self.create_absorb_state = create_absorb_state
        self.ns = size  # number of states
        # self.actionsNames = ['S', 'N', 'U', 'D', 'L', 'R']  # 'S' is the 'special' goal action
        self.s_range = [i for i in range(self.ns)]
        self.actions = actions # Actions for a 1D space just jump the state n-spaces
        self.na = len(self.actions)
        self.action_nums = list(range(self.na))
        self.act_nums_2_actions = dict(zip(self.action_nums, self.actions))
        self.max_drift_time = max_drift_time
        self.T_s_s_a, self.T_sa_s = self.build_sa_s_tm(passive_drift=-1)
        self.drift_mat = csr(self.T_s_s_a[:,:,0])
        self.drift_mat_pow_list = self.create_drift_mat_power_list(self.drift_mat, self.max_drift_time)

    def build_sa_s_tm(self, passive_drift):
        T_s_s_a = np.zeros([self.ns, self.ns, self.na])
        T_sa_s = np.zeros([self.ns * self.na, self.ns])
        for i in range(self.ns * self.na):
            s_ind = i // self.na
            a_ind = i % self.na
            s_prime = max(min(s_ind + self.actions[a_ind] + passive_drift, self.ns - 1), 0)
            if self.create_absorb_state and s_ind == 0:
                T_sa_s[i, s_ind] = 1
                T_s_s_a[s_ind, s_ind, a_ind] = 1
            else:
                T_sa_s[i, s_prime] = 1
                T_s_s_a[s_ind, s_prime, a_ind] = 1

        return T_s_s_a, T_sa_s

    def create_drift_mat_power_list(self, drift_mat, max_drift_time):
        drift_pow_list = [drift_mat.power(0)]
        for t in range(max_drift_time):
            drift_pow_list.append(drift_pow_list[-1] * drift_mat)

        return drift_pow_list

    def average_drift_matrix_power(self, time_dist):
        if time_dist.sum() != 1:
            warnings.warn("Time Distribution doesn't sum to 1")
        average_power_mat = csr(np.zeros(list(self.drift_mat_pow_list[-1].shape)))
        for t in range(len(time_dist)):
            if time_dist[t]:
                average_power_mat = average_power_mat + self.drift_mat_pow_list[t] * time_dist[t]
        return average_power_mat


class StateSpace_2D:
    def __init__(self, ny, nx, wallmat, wallmatlist, env_duration_list, name='BASE', buildTM=True, actions=None, keep_dense=False, restricted_states='all', sparse_only = False, build_saLMDP_passive = False, p_main = 1, initial_state=0, elevmat=None, max_step_up=1):
        self.restricted_states = restricted_states
        if restricted_states != 'all':
            self.ns = self.restricted_states
        else:
            self.ns = ny*nx
        self.current_state = initial_state
        self.name = name
        self.nx = nx
        self.ny = ny
        self.t = 0
        self.p_main = p_main  # probability that an agent goes in the "correct direction", i.e. a_right x -> x_right w/ prob p_main
        self.env_duration_list = env_duration_list
        self.env_final_times = np.cumsum(([0] + env_duration_list)) - 1  # minus 1 is for the final time
        self.env_start_times = np.cumsum(([0] + env_duration_list))
        self.wallmatlist = wallmatlist
        self.wallMatrix_primary = wallmat
        self.wallmat = np.array(wallmat)
        self.wallinds = np.where(self.wallmat.flatten())[0]
        # per-cell integer elevation (row-major, same indexing as coordDict). Default flat -> no gating.
        self.max_step_up = max_step_up
        self.elev = (np.zeros((ny, nx), dtype=int) if elevmat is None else np.asarray(elevmat, dtype=int)).flatten()
        self.actionsNames = ['S', 'N', 'U', 'R', 'D', 'L'] # 'S' is the 'special' goal action
        self.special_action = 0
        self.xRange = [i for i in range(nx)]
        self.yRange = [i for i in range(ny)]
        self.actions = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])  #[y,x]
        self.action_name_to_action = dict(zip(self.actionsNames, self.actions))
        self.na = len(self.actions)
        self.actionNums = [i for i in range(len(self.actionsNames))]
        self.actNums2actions = dict(zip(self.actionNums, self.actions))
        self.coordSet = list(itertools.product(self.yRange, self.xRange))
        self.coordDict = dict(zip(self.coordSet, [i for i in range(len(self.coordSet))]))
        self.liToCoord = dict(zip([i for i in range(len(self.coordSet))], self.coordSet))
        self.na = len(self.actionNums)
        self.wall_list = np.where(self.wallMatrix_primary.flatten()==1)[0]
        self.wall_list_set = [np.where(wm.flatten()==1)[0] for wm in self.wallmat]

        self.ss_to_task_to_action_ind = {}  # Dictionary that maps tasks from another state-space to an action ind
        self.action_ind_to_state_space_to_task = {}  # Dictionary that action inds to tasks in another state-space
        self.action_ind_to_goal_string = {}  # Dictionary that action inds to tasks in another state-space

        self.allLinearInds = range(len(self.coordSet))
        # self.validList = [i for (i, x) in enumerate(self.wallMatrix.flatten()) if x == 0]
        # self.validCoords = [self.liToCoord[i] for i in self.validList]
        # self.nv = len(self.validList)

        yset = set([self.coordDict[(y, x)] for x in range(nx) for y in [0, ny - 1]])
        xset = set([self.coordDict[(y, x)] for x in [0, nx - 1] for y in range(ny)])
        self.perimLI = list(set.union(xset, yset))
        self.perimCoords = [self.liToCoord[i] for i in self.perimLI]
        self.pxx = None
        self.P_dense = None
        self.pxa = None
        self.P_ax_x = None

        if buildTM:
            # self.build_lmdp_p(nx, ny, keep_dense)

            # self.build_salmdp_p_w_obst(nx, ny, wallmat)
            self.sparse_op_list, self.dense_op_list = self.build_all_transition_ops(self.wallmatlist, self.p_main)

        if build_saLMDP_passive:
            self.build_salmdp_p(nx, ny)

    @staticmethod
    def get_wall_inds(wall_matrix):
        wall_list = [i for (i, x) in enumerate(wall_matrix.flatten()) if x == 1]
        return wall_list

    def get_p_ax_x_at_time_t(self, t, mat_type='dense'):
        # This function implicitly represents an operator p_axt_x by only representing the stationary operators p_ax_x in self.env_start_times and returning the one that is active at time t
        idx = bisect_left(self.env_final_times, t)
        if mat_type == 'sparse':
            return self.sparse_op_list[idx]
        else:
            return self.dense_op_list[idx]

    def build_time_indexed_stationary_op(self, p_ax_t, max_time):
        return np.repeat(p_ax_t[:, :, np.newaxis, :], max_time, axis=2)

    def build_all_transition_ops(self, wall_mat_set, p_main):
        sparse_op_list = []
        dense_op_list = []
        if p_main == 1:
            for wm in wall_mat_set:
                sparse_op, dense_op = self.build_stationary_transition_op(self.nx, self.ny, wm, p_main)
                sparse_op_list.append(sparse_op)
                dense_op_list.append(dense_op)
        else:
            for wm in wall_mat_set:
                sparse_op, dense_op = self.build_stoc_stationary_transition_op(self.nx, self.ny, wm, p_main)
                sparse_op_list.append(sparse_op)
                dense_op_list.append(dense_op)

        return sparse_op_list, dense_op_list

    def build_non_stationary_operator(self, wall_times):
        self.sparse_op_list

    def get_xy_coord_from_sa_vec(self, state_vec):
        if state_vec.size != self.ns*self.na:
            raise ValueError("Not a state-action vector")
        sa_ind = sp.sparse.find(state_vec == 1)[1][0]
        state = sa_ind//self.na
        return self.liToCoord[state]

    def deterministic_state_update_under_policy(self, pol, state_vec):
        # state_vec must be sparse
        next_state_vec = sp.dot(state_vec, pol)
        return

    def deterministic_state_update_under_action(self, action_ind, state_LI, wall_mat):
        # This method is the "true" state update.  The action is dictated by the policy, but the update could be
        # different if there is disagreement between the policy and the world.
        cur_vec = onehot(state_LI*self.na+action_ind, self.na*self.ns)
        next_state_LI = csr.dot(cur_vec, self.T_x_xa)
        ind = np.argwhere(next_state_LI == 1)
        if wall_mat.flat[ind]==1:
            # If the next state is a wall, set next state to previous state.
            return state_LI

        next_state_coord = self.get_xy_coord_from_sa_vec(next_state_LI)
        return next_state_LI, next_state_coord


    @staticmethod
    def get_next_sa_salmdp_jointpol(policy, ns, na, current_sa):
        next_sa = np.argmax(csr(current_sa).dot(policy))
        next_s = next_sa // na
        next_a = next_sa % na
        return next_s, next_a

    def build_salmdp_p(self, nx, ny):
        xRange = [i for i in range(nx)]
        yRange = [i for i in range(ny)]
        coordSet = list(itertools.product(yRange, xRange))
        coordDict = dict(zip(coordSet, [i for i in range(len(coordSet))]))
        DyDx = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])  # Action Directions

        # Coordinates for this matrix will be organized with the actions as the
        # inner variable and state as the outer variables [x1a1,x1a2,x1a3,...,xnam]

        # Here we compute two different transition matricies p_x (SA x S'), p_xa (SA x S'A') for deterministic transitions only
        row_xa_2_xa = []  # Conditions on (x,a)
        row_xa_2_x = []  # Conditions on (x,a)

        col_xa = []  # for XA x X'A'
        data_xa = []  # for XA x X'A'

        col_x = []  # for XA x X'
        data_x = []  # for XA x X'

        col_xax = []  # for XA x XAX'
        data_xax = []  # for XA x XAX'

        row_xax = []  # for XAX' x A'
        col_a = []  # for XAX' x A'
        data_a = []  # for XAX' x A'

        for yx in self.coordSet:
            for actionidx, dydx in enumerate(self.actions):
                oldX = coordDict[(yx[0], yx[1])]
                y = yx[0]
                x = yx[1]
                delta_y = dydx[0]
                delta_x = dydx[1]

                newxcoord = max(0, min(x + delta_x, nx - 1))
                newycoord = max(0, min(y + delta_y, ny - 1))

                newX = coordDict[(newycoord, newxcoord)]
                # P[oldS, newS] = 1

                oldS_xa = [oldX * self.na + actionidx] * self.na
                oldS_xa_2_x = oldX * self.na + actionidx

                newX_xa = newX * self.na
                newS_range = list(range(newX_xa, newX_xa+self.na))

                # r_xax = [(newX * self.na * self.ns) + (oldX * self.na) + actionidx] * self.na

                row_xa_2_xa.extend(oldS_xa)
                row_xa_2_x.append(oldS_xa_2_x)
                col_xa.extend(newS_range)
                col_x.extend([newX])
                data_xa.extend([1/self.na]*self.na)
                data_x.extend([1])

        self.pxa = csr((data_xa, (row_xa_2_xa, col_xa)), shape=(self.ns * self.na, self.ns * self.na))
        # self.p_x_xa = csr((data_xa, (row_xa_2_x, col_xa)), shape=(self.ns * self.na, self.ns))
        # self.P_x_xa_dense = self.p_x_xa.todense()
        # self.P_x_tensor = self.p_x_xa.reshape([self.na, self.ns, self.ns])
        self.P_x_xa_sparse = csr((data_x, (row_xa_2_x, col_x)), shape=(self.ns * self.na, self.ns))  # Transition matrix T(x'|x,a)
        self.P_ax_x = self.P_x_xa_sparse.toarray().reshape([self.ns, self.na, self.ns])  # order is [ns,na,ns']
        self.P_ax_x = np.swapaxes(self.P_ax_x, 0, 1)  # Now axis order is [na,ns,ns']
        self.pa = None


    def build_stationary_transition_op(self, nx, ny, wallmat, return_dense=True):
        # Builds Sparse and Dense Stationary Transition Operator
        wallinds = np.where(wallmat.flatten())[0]
        xRange = [i for i in range(nx)]
        yRange = [i for i in range(ny)]
        coordSet = list(itertools.product(yRange, xRange))
        coordDict = dict(zip(coordSet, [i for i in range(len(coordSet))]))
        DyDx = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])  # Action Directions

        # Coordinates for this matrix will be organized with the actions as the
        # inner variable and state as the outer variables [x1a1,x1a2,x1a3,...,xnam]

        # Here we compute two different transition matricies p_x (SA x S'), p_xa (SA x S'A') for deterministic transitions only
        row_xa_2_xa = []  # Conditions on (x,a)
        row_xa_2_x = []  # Conditions on (x,a)

        col_xa = []  # for XA x X'A'
        data_xa = []  # for XA x X'A'

        col_x = []  # for XA x X'
        data_x = []  # for XA x X'

        col_xax = []  # for XA x XAX'
        data_xax = []  # for XA x XAX'

        col_a = []  # for XAX' x A'
        data_a = []  # for XAX' x A'

        _nx = self.nx
        _na = self.na
        if self.restricted_states != 'all':
            num_states = self.restricted_states
        else:
            num_states = self.ns
        for ind in range(num_states):
            yx = self.coordSet[ind]
            for actionidx, dydx in enumerate(self.actions):
                oldXY_ind = coordDict[(yx[0], yx[1])]
                y = yx[0]
                x = yx[1]
                delta_y = dydx[0]
                delta_x = dydx[1]

                newxcoord = max(0, min(x + delta_x, nx - 1))
                newycoord = max(0, min(y + delta_y, ny - 1))

                newXY_ind = np.min([coordDict[(newycoord, newxcoord)], self.ns-1])
                if newXY_ind in wallinds:
                    newXY_ind = oldXY_ind
                if oldXY_ind in wallinds:
                    newXY_ind = oldXY_ind
                # elevation gate: higher elev VALUE = lower/sunken ground. Can't climb UP out of a
                # sunken cell by more than max_step_up (drop in is fine; cliffs are one-way down).
                if self.elev[oldXY_ind] - self.elev[newXY_ind] > self.max_step_up:
                    newXY_ind = oldXY_ind
                # P[oldS, newS] = 1
                # if newXY_ind in wallinds:
                #     newXY_ind = oldYX_ind

                oldS_xa = [oldXY_ind * _na + actionidx] * _na
                oldS_xa_2_x = oldXY_ind * _na + actionidx

                newX_xa = newXY_ind * _na
                newS_range = list(range(newX_xa, newX_xa + _na))


                # r_xax = [(newX * self.na * self.ns) + (oldX * self.na) + actionidx] * self.na

                # row_xa_2_xa.extend(oldS_xa)
                row_xa_2_x.append(oldS_xa_2_x)
                # col_xa.extend(newS_range)
                col_x.extend([newXY_ind])
                data_xa.extend([1 / _na] * _na)
                data_x.extend([1])

        p_ax_x_sparse = csr((data_x, (row_xa_2_x, col_x)), shape=(num_states * _na, num_states))  # Transition matrix P(x'|x,a)
        # p_x_xa_sp_tensor
        p_ax_x = None
        if return_dense:
            p_ax_x = p_ax_x_sparse.toarray().reshape([num_states, _na, num_states])  # order is [ns,na,ns']
            p_ax_x = np.swapaxes(p_ax_x, 0, 1)  # Now axis order is [na,ns,ns']

        return p_ax_x_sparse, p_ax_x


    def build_stoc_stationary_transition_op(self, nx, ny, wallmat, p_main, return_dense=True):
        # Builds Sparse and Dense Stationary Transition Operator
        wallinds = np.where(wallmat.flatten())[0]
        xRange = [i for i in range(nx)]
        yRange = [i for i in range(ny)]
        coordSet = list(itertools.product(yRange, xRange))
        coordDict = dict(zip(coordSet, [i for i in range(len(coordSet))]))
        DyDx = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])  # Action Directions

        # Coordinates for this matrix will be organized with the actions as the
        # inner variable and state as the outer variables [x1a1,x1a2,x1a3,...,xnam]

        _nx = self.nx
        _na = self.na

        # Here we compute two different transition matricies p_x (SA x S'), p_xa (SA x S'A') for deterministic transitions only
        row_xa_2_xa = []  # Conditions on (x,a)
        row_xa_2_x = []  # Conditions on (x,a)

        col_x = []  # for XA x X'
        data_x = []  # for XA x X'

        residual = np.round(1 - p_main, 10)
        # rh = np.round(residual * 0.8 / 2, 10)
        div = np.round(residual / 6, 10)
        rh = div * 2
        rl = div
        # rl = np.round(residual - 2 * rh, 10)
        r_uni = residual / 4

        # c1 = np.array([1, 0, 0, 0, 0, 0])
        # c2 = np.array([0, 1, 0, 0, 0, 0])
        # up = np.array([0, rl, p_main, rh, rl, rh])
        # right = np.array([0, rl, rh, p_main, rh, rl])
        # down = np.array([0, rl, rl, rh, p_main, rh])
        # left = np.array([0, rl, rh, rl, rh, p_main])
        # dists = np.array([c1, c2, up, right, down, left])

        c1 = np.array([1, 0, 0, 0, 0, 0])
        c2 = np.array([0, 1, 0, 0, 0, 0])
        up = np.array([0, r_uni, p_main, r_uni, r_uni, r_uni])
        right = np.array([0, r_uni, r_uni, p_main, r_uni, r_uni])
        down = np.array([0, r_uni, r_uni, r_uni, p_main, r_uni])
        left = np.array([0, r_uni, r_uni, r_uni, r_uni, p_main])
        dists = np.array([c1, c2, up, right, down, left])

        dist_dict = dict(zip(np.arange(len(dists)), dists))

        if self.restricted_states != 'all':
            num_states = self.restricted_states
        else:
            num_states = self.ns
        for ind in range(num_states):
            yx = self.coordSet[ind]
            for actionidx, dydx in enumerate(self.actions):
                oldYX_ind = coordDict[(yx[0], yx[1])]


                y = yx[0]
                x = yx[1]

                if oldYX_ind in wallinds:
                    all_new_coords_minmax = [(y, x)] * _na  # Set to identity transition if inside wall.
                else:
                    all_new_coords_minmax = [(max(0, min(y + dy, ny - 1)), max(0, min(x + dx, nx - 1))) for (dy, dx) in self.actions]

                # residual_prob = np.round(residual / 4, 10)
                # all_data = np.ones(_na) * residual_prob
                # all_data[actionidx] = p_main

                all_data = dist_dict[actionidx]

                # These if statements, just make sure that probability mass only gets put on one of the identity actions
                # if actionidx == 0:
                #     all_data = np.ones(_na) * residual_prob
                #     all_data[actionidx] = p_main
                #     all_data[1] = 0
                #
                # if actionidx > 0:
                #     all_data = np.ones(_na) * residual_prob
                #     all_data[actionidx] = p_main
                #     all_data[0] = 0

                if np.round(np.sum(all_data), 4) != 1:
                    warnings.warn("yikes")

                allYX_inds = [np.min([coordDict[(newycoord, newxcoord)], self.ns - 1]) for (newycoord, newxcoord) in all_new_coords_minmax]

                for ind, newYX_ind in enumerate(allYX_inds):
                    # blocked by a wall, OR an up-climb steeper than max_step_up -> redirect mass to staying
                    if newYX_ind in wallinds or (self.elev[oldYX_ind] - self.elev[newYX_ind] > self.max_step_up):
                        allYX_inds[ind] = oldYX_ind

                if oldYX_ind in wallinds:
                    allYX_inds = [oldYX_ind]*_na

                oldS_xa_2_x = oldYX_ind * _na + actionidx

                row_xa_2_x.extend([oldS_xa_2_x] * _na)
                # col_xa.extend(newS_range)
                col_x.extend(allYX_inds)
                data_x.extend(all_data)

        p_ax_x_sparse = csr((data_x, (row_xa_2_x, col_x)), shape=(num_states * _na, num_states))  # Transition matrix P(x'|x,a)
        # p_x_xa_sp_tensor
        p_ax_x = None
        if return_dense:
            p_ax_x = p_ax_x_sparse.toarray().reshape([num_states, _na, num_states])  # order is [ns,na,ns']
            p_ax_x = np.swapaxes(p_ax_x, 0, 1)  # Now axis order is [na,ns,ns']

        return p_ax_x_sparse, p_ax_x



    def build_salmdp_p_w_obst(self, nx, ny, wallmatset):
        xRange = [i for i in range(nx)]
        yRange = [i for i in range(ny)]
        coordSet = list(itertools.product(yRange, xRange))
        coordDict = dict(zip(coordSet, [i for i in range(len(coordSet))]))
        DyDx = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])  # Action Directions

        # Coordinates for this matrix will be organized with the actions as the
        # inner variable and state as the outer variables [x1a1,x1a2,x1a3,...,xnam]

        # Here we compute two different transition matricies p_x (SA x S'), p_xa (SA x S'A') for deterministic transitions only
        row_xa_2_xa = []  # Conditions on (x,a)
        row_xa_2_x = []  # Conditions on (x,a)

        col_xa = []  # for XA x X'A'
        data_xa = []  # for XA x X'A'

        col_x = []  # for XA x X'
        data_x = []  # for XA x X'

        col_xax = []  # for XA x XAX'
        data_xax = []  # for XA x XAX'

        row_xax = []  # for XAX' x A'
        col_a = []  # for XAX' x A'
        data_a = []  # for XAX' x A'

        for yx in self.coordSet:
            for actionidx, dydx in enumerate(self.actions):
                oldXY_ind = coordDict[(yx[0], yx[1])]
                y = yx[0]
                x = yx[1]
                delta_y = dydx[0]
                delta_x = dydx[1]

                newxcoord = max(0, min(x + delta_x, nx - 1))
                newycoord = max(0, min(y + delta_y, ny - 1))

                newXY_ind = coordDict[(newycoord, newxcoord)]
                if newXY_ind in self.wallinds:
                    newXY_ind = oldXY_ind
                # P[oldS, newS] = 1

                oldS_xa = [oldXY_ind * self.na + actionidx] * self.na
                oldS_xa_2_x = oldXY_ind * self.na + actionidx

                newX_xa = newXY_ind * self.na
                newS_range = list(range(newX_xa, newX_xa+self.na))

                # r_xax = [(newX * self.na * self.ns) + (oldX * self.na) + actionidx] * self.na

                row_xa_2_xa.extend(oldS_xa)
                row_xa_2_x.append(oldS_xa_2_x)
                col_xa.extend(newS_range)
                col_x.extend([newXY_ind])
                data_xa.extend([1/self.na]*self.na)
                data_x.extend([1])

        self.pxa = csr((data_xa, (row_xa_2_xa, col_xa)), shape=(self.ns * self.na, self.ns * self.na))
        # self.p_x_xa = csr((data_xa, (row_xa_2_x, col_xa)), shape=(self.ns * self.na, self.ns))
        # self.P_x_xa_dense = self.p_x_xa.todense()
        # self.P_x_tensor = self.p_x_xa.reshape([self.na, self.ns, self.ns])
        P_x_xa_sparse = csr((data_x, (row_xa_2_x, col_x)), shape=(self.ns * self.na, self.ns))  # Transition matrix T(x'|x,a)
        P_ax_x = P_x_xa_sparse.toarray().reshape([self.ns, self.na, self.ns])  # order is [ns,na,ns']
        self.P_ax_x = np.swapaxes(P_ax_x, 0, 1)  # Now axis order is [na,ns,ns']
        self.pa = None

    @staticmethod
    def get_salmdp_costfunction(targetState, targetAction, wall_list, numberOfStates, numberOfActions, baseCost):
        na = numberOfActions
        ns = numberOfStates
        cfunction = np.ones(ns*na) * np.exp(-baseCost)
        targetLI = targetState*na + targetAction
        wallLIs = np.array([list(range(i*na, (i*na)+na)) for i in wall_list]).flatten() # Walls cover all state-action pairs that for all wall-states
        cfunction[targetLI] = 1
        if wallLIs.size:
            cfunction[wallLIs] = 0  # Order matters here, we want to overwrite goal states that are in walls.
        return sp.sparse.diags(cfunction)

    def build_lmdp_p(self, nx, ny, keep_dense):
        import torch
        import torch.nn.functional as f
        P = torch.zeros([self.ns, self.ns], dtype=torch.double)
        xRange = [i for i in range(nx)]
        yRange = [i for i in range(ny)]
        coordSet = list(itertools.product(yRange, xRange))
        coordDict = dict(zip(coordSet, [i for i in range(len(coordSet))]))
        DyDx = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])  # Action Directions
        for yx in self.coordSet:
            for idx, dydx in enumerate(DyDx):
                oldS = coordDict[(yx[0], yx[1])]
                y = yx[0]
                x = yx[1]
                delta_y = dydx[0]
                delta_x = dydx[1]

                newx = max(0, min(x + delta_x, nx - 1))
                newy = max(0, min(y + delta_y, ny - 1))

                # agent.move_no_obst()
                newS = coordDict[(newy, newx)]
                P[oldS, newS] = 1

        P = f.normalize(P, p=1, dim=1)
        # self.sP = to_sparse(self.P)
        self.pxx = csr(P.numpy(), dtype=np.double)
        if keep_dense:
            self.P = P

    def buildLocalTransitionMatrix(self, keep_dense):
        import torch
        import torch.nn.functional as f
        agent = agent.Simple_Agent(0, 0, localssflag=0)
        P = torch.zeros([self.ns, self.ns], dtype=torch.double)
        for yx in self.coordSet:
            for idx, dydx in enumerate(agent.DyDx):
                oldS = agent.coordDict[(yx[0], yx[1])]
                agent.y = yx[0]
                agent.x = yx[1]
                agent.delta_y = dydx[0]
                agent.delta_x = dydx[1]
                agent.move_no_obst()
                newS = agent.coordDict[(agent.y, agent.x)]
                P[oldS, newS] = 1

        P = f.normalize(P, p=1, dim=1)
        # self.sP = to_sparse(self.P)
        self.pxx = csr(P.numpy(), dtype=np.double)
        if keep_dense:
            self.localP = P


def get_p_doom():
    return np.random.rand()





# ============================================================================================
# OKBE Task logic-space (sigma) kernel helpers.
# Forked out of the tLMDP-era sigma_space.py on 2026-07-13 so the OKBE path owns its Task machinery
# (sigma_space.py + its own Task still exist for the tLMDP mains: ms_lmdp*.py, task_lmdp.py). Pure
# numpy/scipy. Kept here alongside the other state-spaces; the `Task` class below consumes them.
# ============================================================================================

class Goal:
    def __init__(self, idx, xg, state_space, types, kind='primary', dlmean=10000, dlvar=10000, grounded_action_ind=0):
        self.state_space = state_space  # state-space this goal operates on
        self.idx = idx
        self.xg = xg
        self.ag = grounded_action_ind
        self.na = 1 if not state_space else state_space.na
        self.xgag = xg * self.na + self.ag  # joint state goal (goal encoded in SA-space)
        self.kind = kind
        self.types = [idx] + types
        self.dlmean = dlmean
        self.dlvar = dlvar
        self.trueDL = dlmean
        self.order_graph_children = []  # this Goal must precede all goals in this list

    def getPMFfromPDF(self, pdf, addHeavyTail, heavytailconst=0.00005):
        # deadline PMF from a deadline PDF; addHeavyTail keeps a small gradient in the value function.
        from config import MAX_TIME
        deadlinePMFList = [pdf.cdf(0.5)] + \
                          [pdf.cdf(x) - pdf.cdf(x - 1) for x in np.arange(1.5, MAX_TIME - 0.5)] + \
                          [pdf.cdf(10000) - pdf.cdf(MAX_TIME - 0.5)]
        deadlinePMF = np.array(deadlinePMFList).transpose()
        if addHeavyTail:
            deadlinePMF = deadlinePMF + np.ones(deadlinePMF.size) * heavytailconst
            deadlinePMF = deadlinePMF / deadlinePMF.sum()
        return deadlinePMF


def create_sigma_space_objects(goal_inds, goal_pair_dict, goal_to_binary_constraint):
    # Create the sigma-space objects. goal_inds are bit-flips only (no null-action).
    ng = len(goal_inds)
    sigma_list = np.arange(0, (2 ** (ng)))
    T_sigma_constraints = create_sigma_trans_mat_w_constraints(sigma_list, goal_inds, goal_pair_dict, goal_to_binary_constraint)
    T_sigma_free = create_sigma_trans_mat(sigma_list, goal_inds)
    P_sigma_g = create_sigma_goal_passive_dynamics(sigma_list, goal_inds)
    return T_sigma_free, T_sigma_constraints, P_sigma_g


def create_sigma_trans_mat(sigma_list, goal_inds):
    # Unrestricted transition operator for sigma.
    ng = len(goal_inds)
    n_sig = len(sigma_list)
    sigma_lst_repeated = sigma_list.repeat(ng)
    g_lst_tiled = np.tile(goal_inds, (1, n_sig))[0]
    new_sigma_lst = (sigma_lst_repeated ^ (1 << g_lst_tiled))
    from_joint_state_lst = (ng * sigma_lst_repeated) + g_lst_tiled
    t_sigma_data = np.ones(len(new_sigma_lst))
    return csr((t_sigma_data, (from_joint_state_lst, new_sigma_lst)), shape=(len(sigma_list) * len(goal_inds), len(sigma_list)))


def create_sigma_trans_mat_w_null(sigma_list, goal_inds):
    # Unrestricted transition operator for sigma WITH a null action in the 0 spot.
    # Row m -> bitvector m // ng_w_null, action m % ng_w_null. Reshapeable to (ng_w_null, ns, ns).
    expanded_goal_inds = [0] + goal_inds
    ng = len(expanded_goal_inds)
    ng_w_null = len(expanded_goal_inds)
    n_sig = len(sigma_list)
    identity_vect = np.zeros(n_sig * ng_w_null)
    identity_indices = list(range(0, n_sig * ng, ng))
    identity_vect[identity_indices] = 1
    sigma_lst_tiled = np.tile(sigma_list, ng_w_null)
    g_lst_repeated = np.array(expanded_goal_inds).repeat(n_sig)
    bit_shift_rep = 1 << g_lst_repeated
    bit_shift_rep[np.arange(n_sig)] = 0
    new_sigma_lst_alt = (sigma_lst_tiled ^ bit_shift_rep)
    from_joint_state_action_lst = np.arange(ng_w_null * n_sig)
    t_sigma_data = np.ones(len(new_sigma_lst_alt))
    return csr((t_sigma_data, (from_joint_state_action_lst, new_sigma_lst_alt)), shape=(ng_w_null * n_sig, n_sig))


def create_sigma_trans_mat_w_constraints(sigma_list, goal_inds, goal_order_dict, goal_to_binary_constraint):
    # NOTE: unfinished stub -- body == the free create_sigma_trans_mat (ignores the constraint args).
    # Precedence is currently enforced via the ordering-Q cost path, not this kernel. See memory
    # okbe-unimplemented / codebase-gotchas.
    ng = len(goal_inds)
    n_sig = len(sigma_list)
    sigma_lst_repeated = sigma_list.repeat(ng)
    g_lst_tiled = np.tile(goal_inds, (1, n_sig))[0]
    new_sigma_lst = (sigma_lst_repeated ^ (1 << g_lst_tiled))
    from_joint_state_lst = (ng * sigma_lst_repeated) + g_lst_tiled
    t_sigma_data = np.ones(len(new_sigma_lst))
    return csr((t_sigma_data, (from_joint_state_lst, new_sigma_lst)), shape=(len(sigma_list) * len(goal_inds), len(sigma_list)))


def check_no_order_violation(sigma, goal_binary_constraint):
    # False if there is a violation, True if not. (sigma & constraint) must be 0 for no violation.
    return int(not((sigma & goal_binary_constraint) > 0))


def check_constraint_violation_alt_2(sigma, goal_idx, conditional_idx, bit_relation_1, bit_relation_2):
    # Violation iff flip-bit is in state bit_relation_1 AND conditional-bit is in state bit_relation_2.
    sigma_flip_state = int((sigma & (1 << goal_idx)) > 0)
    sigma_conditional_state = int((sigma & (1 << conditional_idx)) > 0)
    return (sigma_flip_state == bit_relation_1) and (sigma_conditional_state == bit_relation_2)


def compute_goal_orders(goal_vars, type_order_dict):
    goal_order_dict = {}
    goal_to_binary_constraint = {}   # constraint as a binary number; 1 = active ordering constraint
    goal_to_binary_constraint_bin_dict = {}
    for bit_relation in type_order_dict:
        constraint_list = []
        for g_i in goal_vars:
            bin_constraint = 0
            for g_j in goal_vars:
                for type_order_entry in type_order_dict[bit_relation]:
                    if type_order_entry[0] in g_i.types and type_order_entry[0] in g_j.types:
                        constraint_list.append((g_i.idx, g_j.idx))
                        bin_constraint = bin_constraint | (1 << g_j.idx)
                        break
            goal_to_binary_constraint[g_i.idx] = bin_constraint
        if len(constraint_list) > 0:
            goal_order_dict[bit_relation] = constraint_list
            goal_to_binary_constraint_bin_dict[bit_relation] = goal_to_binary_constraint
    return goal_order_dict, goal_to_binary_constraint_bin_dict


def create_sigma_goal_passive_dynamics(sigma_list, goal_inds, final_state_dummy_action=False):
    # Unrestricted (sigma,g) -> (sigma',g') passive transition operator. Used in saLMDP desirability.
    ng = len(goal_inds)
    n_sig = len(sigma_list)
    sigma_lst_repeated = sigma_list.repeat(ng)
    g_lst_tiled = np.tile(goal_inds, (1, n_sig))[0]
    new_sigma_lst = (sigma_lst_repeated ^ (1 << g_lst_tiled))
    new_sigma_g_lst = new_sigma_lst * ng
    from_joint_state_lst = np.arange(ng * n_sig)
    row_inds = from_joint_state_lst.repeat(ng)
    ng_range = np.arange(ng)
    col_inds = (np.tile(new_sigma_g_lst, (ng, 1)) + ng_range[:, np.newaxis]).transpose().flatten()
    t_sigma_data = np.ones(len(row_inds)) / ng
    return csr((t_sigma_data, (row_inds, col_inds)), shape=(ng * n_sig, ng * n_sig))


def create_ordering_Q_mat_alt_2(sigma_ind_list, num_of_goals, cost, goal_order_dict, goal_to_binary_constraint):
    cost_vector = np.ones(len(sigma_ind_list) * num_of_goals) * np.exp(-cost)
    for bit_relation in goal_order_dict.keys():
        rel_bit_1 = int(bit_relation[0])
        rel_bit_2 = int(bit_relation[1])
        for sigma_ind in sigma_ind_list:
            for bit_index_pair in goal_order_dict[bit_relation]:
                goal_idx = bit_index_pair[0]
                conditional_idx = bit_index_pair[1]
                violation = check_constraint_violation_alt_2(sigma_ind, goal_idx, conditional_idx, rel_bit_1, rel_bit_2)
                cost_vector[sigma_ind * num_of_goals + goal_idx] = (not violation) * np.exp(-cost)
    return sp.sparse.diags(cost_vector), cost_vector


def add_ids_to_type_order_dict(type_order_dict, goal_var_list, bit_relation):
    lst = []
    for g in goal_var_list:
        lst = lst + [(g.idx, g.idx)]
    type_order_dict[bit_relation] = lst
    return type_order_dict


# === Task: the OKBE boolean logic/goal (sigma) space (uses the helpers above). ===
# The OKBE canonical Task -- self-contained, no dependency on sigma_space.
class Task(OKBEKernelMixin):
    def __init__(self, goal_var_list, type_order_dicts, base_cost, max_time=100, initial_state=0):
        # Deferred imports (used only in __init__) so state_spaces gains NO module-load
        # dependency on salmdp/utilities/TGMDP_methods -> no import cycle. The sigma-kernel helpers
        # (compute_goal_orders, create_sigma_space_objects, create_sigma_trans_mat_w_null,
        # create_ordering_Q_mat_alt_2) are module-level in THIS file, above the class.
        from utilities import onehot
        from salmdp import SALMDP
        import TGMDP_methods
        self.name = 'BOG_task'
        self.color = BLACK
        self.current_state = initial_state
        self.cost = base_cost
        self.ng = len(goal_var_list) # number of goals
        self.na = self.ng + 1  # number of actions is number of goals plus 1 null action
        self.ns = 2 ** self.ng # number of sigmas
        self.max_time = max_time          # used by compute_STEF_from_hr (was missing -> dormant bug)
        self.region_functions = dict()     # region-key -> {eta (STEF), kappa (CEF), sf (SF), xi (TED), nu (hazard)}, like Internal_StateSpace
        self.default_action_ind = 0       # null action (no bit flip) = region default dynamics
        self.goal_var_list = goal_var_list
        self.goal_inds = [g.idx for g in goal_var_list]
        self.number_of_goals = len(self.goal_inds)
        self.type_order_dicts = type_order_dicts
        self.grounded_goal_inds = [g.xg for g in goal_var_list]
        self.grounded_state_action_inds = [g.xg + g.ag for g in goal_var_list]
        self.grounded_goal_ind_dict = dict(zip(self.goal_inds, self.grounded_goal_inds))
        self.sigma_list = np.arange(0, (2 ** (self.ng)))
        # self.goal_pair_dict, self.goal_to_binary_constraint = compute_goal_orders(self.goal_var_list, self.type_order_dicts)
        self.zeta = np.ones(self.ns)

        self.goal_pair_dict, self.goal_to_binary_constraint = compute_goal_orders(self.goal_var_list, self.type_order_dicts)
        self.T_sigma_free, self.T_sigma_constrained, self.sigma_g_passive = create_sigma_space_objects(self.goal_inds,
                                                                                                      self.goal_pair_dict,
                                                                                                      self.goal_to_binary_constraint)

        sigma_list = np.arange(0, (2 ** (len(self.goal_inds))))
        self.T_sigma_free_w_null = create_sigma_trans_mat_w_null(sigma_list, self.goal_inds)

        self.P_a_s_s_dict = {}
        # Assume T_sigma_free_w_null is (na * ns, ns), where rows are grouped by action
        for a in range(self.na):
            # Extract rows corresponding to action a
            start_idx = a * self.ns
            end_idx = (a + 1) * self.ns
            self.P_a_s_s_dict[a] = self.T_sigma_free_w_null[start_idx:end_idx, :].tocsr()

        # create sparse P_a_s_s
        data = []
        coords = [[], [], []]  # To store indices in the (a, s, s') format
        for a in range(self.na):
            P_sparse = self.P_a_s_s_dict[a]
            row, col = P_sparse.nonzero()
            values = P_sparse.data
            # Append values and indices, ensuring 'a' is the first index
            coords[0].extend([a] * len(values))  # Action index
            coords[1].extend(row)  # State index
            coords[2].extend(col)  # Next-state index
            data.extend(values)

        # Convert to a single sparse 3D array
        self.P_a_s_s = sparse.COO(coords, data, shape=(self.na, self.ns, self.ns))

        # Create sparse tensor version:
        # row, col = self.T_sigma_free_w_null.nonzero()
        # data = self.T_sigma_free_w_null.data
        # a, s = divmod(row, self.ns)  # Adjust based on your indexing
        # tensor_shape = (self.na, self.ns, self.ns)
        # self.P_a_s_s = sparse.COO((a, s, col), data, shape=tensor_shape)

        self.null_omega = csr(np.identity(self.ns))
        self.SPK_lst = [sparse.COO(np.identity(self.ns, dtype=np.float32)) for i in range(self.na)]  # State-prediction Kernel (often called omega)
        self.SPK_tensor = self.compute_SPK_tensor(max_time=max_time)


        # self.goal_order_dict_00, self.goal_to_binary_constraint_00 = compute_goal_orders(self.goal_var_list, self.type_order_dicts['00'])
        # self.goal_order_dict_01, self.goal_to_binary_constraint_01 = compute_goal_orders(self.goal_var_list, self.type_order_dicts['01'])
        # self.goal_order_dict_10, self.goal_to_binary_constraint_10 = compute_goal_orders(self.goal_var_list, self.type_order_dicts['10'])
        # self.goal_order_dict_11, self.goal_to_binary_constraint_11 = compute_goal_orders(self.goal_var_list, self.type_order_dicts['11'])
        # self.goal_order_dicts = {'00': self.goal_order_dict_00, '01': self.goal_order_dict_01, '10': self.goal_order_dict_10, '11': self.goal_order_dict_11}

        # HOSEIN: This is the function that creates the sigma_transition_matrix
        # The operator is a |Sigma|*|G| x |Sigma| sparse matrix where the goal variable is the inner varibles, meaning
        # the state-action pair (sigma, g) is encoded as the index sigma * ng + goal

        # self.T_sigma_free, self.T_sigma_constrained, self.sigma_g_passive = create_sigma_space_objects(self.goal_inds, self.goal_pair_dict)
        # self.ordering_Q = create_ordering_Q_mat()
        # Here we need a function that will compute a passive matrix for (sigma,x,pi)->(sigma,x,pi)'
        # self.ordering_Q_mat, self.ordering_Q_vec = create_ordering_Q_mat(self.sigma_list, self.goal_to_binary_constraint, self.number_of_goals, self.cost, self.goal_order_dicts) # Elements in ordering_Q_vec are in (sigma,g).  If expanded to (sigma,xa,pi), you must use np.repeat(vec, num_of_grounded_xa)
        self.ordering_Q_mat, self.ordering_Q_vec = create_ordering_Q_mat_alt_2(self.sigma_list,
                                                                               self.number_of_goals, self.cost,
                                                                               self.goal_pair_dict, self.goal_to_binary_constraint)  # Elements in ordering_Q_vec are in (sigma,g).  If expanded to (sigma,xa,pi), you must use np.repeat(vec, num_of_grounded_xa)
        self.terminal_inds = np.arange(self.ns * self.ng - self.ng, self.ns * self.ng)
        self.task_desirability, _ = SALMDP.compute_desirability_poweriter_v2(self.ordering_Q_mat, self.sigma_g_passive, self.terminal_inds, max_iter=self.ng + 2)
        self.task_policy = SALMDP.rescalePassiveWithZ(self.task_desirability, self.sigma_g_passive)

        init_vec = onehot(np.arange(0, self.ng), self.task_policy.shape[0]) * self.task_desirability

        # self.eta_event_0 = self.compute_STEF(0, 0)  # Not needed, boolean state-spaces are static.
        self.eta_event_0 = np.zeros([self.ns, self.ns, max_time])
        cumsum_eta = np.cumsum(self.eta_event_0, axis=2)
        self.kappa_event_0 = cumsum_eta.sum(axis=1)
        self.kappa_compliment_0 = 1 - self.kappa_event_0  # kappa(x,t_f) = 1 - sum_{x_f,tau_f}eta(x_f,tau_f|x). Used for computing the temporal event function over all spaces.

        cost_func = np.ones([self.ng, self.ns])
        terminal_inds = [(1 << self.ng) - 1]
        P_sig = np.array(self.T_sigma_free.todense())
        P_sg_s = P_sig.reshape(self.ns, self.ng, self.ns)
        P_gs_s = np.swapaxes(P_sg_s, 0, 1)
        self.value_function, self.pi_sp = TGMDP_methods.value_iteration_min_cost(P_gs_s, cost_func, terminal_inds)

        # self.goal_trajectories = compute_plans(self.task_policy, self.ng, init_vec)
        print('done')


    # compute_STEF_from_hr now provided by OKBEKernelMixin (Task's sparse-densify was folded in).

    # compute_STEF and compute_SPK_tensor now provided by OKBEKernelMixin (Task's copies removed).

