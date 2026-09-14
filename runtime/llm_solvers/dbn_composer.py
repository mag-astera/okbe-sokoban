"""Validated tutorial bridge to the original affordance and sparse DBN classes.

State parents refer to time t; all local kernels produce time t+1. This first
composer deliberately has no alpha-parent or next-state-parent interpretation.
"""
import itertools
import math
from types import SimpleNamespace
import numpy as np
import gates
import state_spaces as ss
from dbn_product import dbn_to_joint_operator

MAX_H_ENTRIES = 500_000
# Sized for the tutorial's 48 GB development machine, including
# 81*40*40*2*2 item states and two additional door bits. These are admission
# limits, not peak-memory guarantees: construction and solving need workspace.
MAX_STATES = 2_500_000
MAX_NNZ = 80_000_000


def integer(value, label, size):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < size:
        raise ValueError(f"{label} must be an integer in [0, {size - 1}]")
    return int(value)


def gate(expr, dims, features, trail=(), depth=0):
    if depth > 40:
        raise ValueError("Feature expression is too deep")
    if not isinstance(expr, dict):
        raise ValueError("A condition must be an object")
    if set(expr) == {"feature"}:
        name = expr["feature"]
        if name not in features or name in trail:
            raise ValueError(f"Unknown or cyclic feature: {name}")
        return gate(features[name], dims, features, trail + (name,), depth + 1)
    for op, cls in (("and", gates.And), ("or", gates.Or)):
        if set(expr) == {op}:
            parts = expr[op]
            if not isinstance(parts, list) or not parts:
                raise ValueError(f"{op} requires a nonempty list")
            return cls(*(gate(p, dims, features, trail, depth + 1) for p in parts))
    if set(expr) == {"not"}:
        return gates.Not(gate(expr["not"], dims, features, trail, depth + 1))
    comparisons = {"gt": lambda x,y: x > y, "gte": lambda x,y: x >= y,
                   "lt": lambda x,y: x < y, "lte": lambda x,y: x <= y}
    if "var" in expr and expr["var"] in dims and set(expr) - {"var"} and set(expr) - {"var"} <= comparisons.keys():
        name = expr['var']
        bounds = [(op, value) for op, value in expr.items() if op != 'var']
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for _, value in bounds):
            raise ValueError('Comparison bounds must be finite numbers')
        return gates.Cond([name], lambda x: all(comparisons[op](x, value) for op, value in bounds), str(expr))
    if set(expr) != {"var", "in"} or expr["var"] not in dims:
        raise ValueError('Use {"var": name, "in": [state indices]}, and/or/not, or a feature reference')
    name = expr["var"]
    if not isinstance(expr["in"], list):
        raise ValueError("in must be a list of state indices")
    values = frozenset(integer(v, name, dims[name]) for v in expr["in"])
    return gates.Cond([name], lambda x: x in values, f"{name} in {sorted(values)}")


def dense(value):
    return np.asarray(value.todense() if hasattr(value, "todense") else value, dtype=float)


def distribution(a, axis, label):
    if not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(axis=axis), 1, atol=1e-10, rtol=0):
        raise ValueError(f"{label} must be finite, nonnegative and sum to 1 along axis {axis}")


def compile_model(available, spec):
    if not isinstance(spec, dict) or spec.get("version") != 1:
        raise ValueError("DBN specification requires version: 1")
    unknown = set(spec) - {"version", "spaces", "custom_spaces", "features", "affordances", "x_identity_when", "sokoban"}
    if unknown:
        raise ValueError(f"Unknown specification fields: {sorted(unknown)}")
    available = dict(available)
    for raw in spec.get("custom_spaces", []):
        name = raw["name"]
        if not isinstance(name, str) or not name or name in available or name == "action":
            raise ValueError(f"Duplicate or invalid custom space: {name}")
        p = np.asarray(raw["P"], dtype=float)
        if p.size > MAX_H_ENTRIES or p.ndim != 3 or p.shape[1] != p.shape[2] or min(p.shape) == 0:
            raise ValueError(f"{name}: P must have shape [alpha, state, next_state] within the size budget")
        distribution(p, 2, f"{name}.P")
        default = integer(raw.get("default_alpha", 0), f"{name} default", p.shape[0])
        available[name] = SimpleNamespace(name=name, ns=p.shape[1], na=p.shape[0], P_a_s_s=p, default_action_ind=default)
    names = spec.get("spaces", list(available))
    if not isinstance(names, list) or not names or names[0] != "X" or len(set(names)) != len(names):
        raise ValueError("spaces must be a unique ordered list beginning with X (the free-action root)")
    if any(n not in available for n in names):
        raise ValueError("Unknown space; reload the current world's catalog or define it in custom_spaces")
    spaces = [available[n] for n in names]
    dims = {s.name: int(s.ns) for s in spaces}
    dims["action"] = int(spaces[0].na)
    features = spec.get("features", {})
    for expr in features.values():
        gate(expr, dims, features)
    affs, reports, seen = [], [], set()
    transition = {}
    budget = 0
    # A coordinated workshop exchange touches several counters. Keep a finite
    # materialization budget (~64 MB of float64 entries) for these small worlds.
    h_budget = 8_000_000 if any(a.get('workshop_managed') for a in spec.get('affordances', [])) else MAX_H_ENTRIES
    for raw in spec.get("affordances", []):
        unknown = set(raw) - {"target", "merge", "default_alpha", "rules", "parents", "has_action", "H_psi", "H_alpha", "placement_managed", "h_form", "workshop_managed", "source_binding"}
        if unknown:
            raise ValueError(f"Unknown affordance fields: {sorted(unknown)}; choose a supported H form")
        target = raw["target"]
        if target == "X" or target not in names or target in seen:
            raise ValueError(f"Invalid or duplicate target {target}; merge its rules into one affordance")
        seen.add(target)
        target_sp = available[target]
        form = raw.get('h_form', 'current')
        if form not in ('current','transition'):
            raise ValueError('h_form must be current or transition')
        if form == 'transition':
            from transition_affordance import TransitionAffordance
            transition[target] = TransitionAffordance(raw, dims, features, target_sp.na)
            reports.append(transition[target].report)
            continue
        mode = raw.get("merge", "reject")
        if mode not in ("reject", "priority"):
            raise ValueError("merge must be reject or priority")
        rules = [(gate(r["when"], dims, features), integer(r["alpha"], target, target_sp.na)) for r in raw.get("rules", [])]
        direct = "H_psi" in raw or "H_alpha" in raw
        if direct and rules:
            raise ValueError("Choose rules or explicit H tensors for each target")
        variables = set().union(*(g.vars() for g, _ in rules)) if rules else set()
        parents = raw.get("parents", [n for n in names if n in variables])
        if not isinstance(parents, list) or len(set(parents)) != len(parents) or any(p not in names for p in parents):
            raise ValueError(f"{target}: parents must be unique selected state-space names")
        if variables - set(parents) - {"action"}:
            raise ValueError(f"{target}: missing rule parents")
        has_action = raw.get("has_action", "action" in variables)
        if not isinstance(has_action, bool) or ("action" in variables and not has_action):
            raise ValueError(f"{target}: has_action must include the free-action axis used by rules")
        axes = (["action"] if has_action else []) + parents
        shape = tuple(dims[p] for p in axes)
        nf = len(raw.get("H_psi", [])) if direct else len(rules) + 1
        budget += math.prod(shape) * (nf + target_sp.na) + nf * target_sp.na
        if budget > h_budget:
            raise ValueError(f"H tensors exceed tutorial budget ({h_budget:,} entries); reduce parents or dimensions")
        collisions = 0
        witness = None
        if direct:
            hp, ha = np.asarray(raw["H_psi"], float), np.asarray(raw["H_alpha"], float)
            if nf == 0 or hp.shape != (nf,) + shape or ha.shape != (target_sp.na, nf):
                raise ValueError(f"{target}: H_psi expected [features, {', '.join(axes)}]; H_alpha expected [{target_sp.na}, features]")
            distribution(hp, 0, "H_psi")
            distribution(ha, 0, "H_alpha")
            aff = ss.Affordance(target, parents, hp, ha)
        else:
            default = integer(raw.get("default_alpha", getattr(target_sp, "default_action_ind", 0)), target, target_sp.na)
            for combo in itertools.product(*(range(d) for d in shape)):
                sd = dict(zip(axes, combo))
                active = [(i, alpha) for i, (g, alpha) in enumerate(rules) if g.eval(sd, sd.get("action"))]
                if len({a for _, a in active}) > 1:
                    collisions += 1
                    if witness is None:
                        witness = {"assignment": sd, "rules": active}
            if collisions and mode == "reject":
                raise ValueError(f"{target}: conflicting actions on {collisions} assignments; example {witness}. Use disjoint rules or explicitly select merge: priority.")
            aff = gates.make_affordance(target, parents, rules, default, target_sp.na, dims, has_action=has_action)
        distribution(aff.tensor, 0, f"F_{target}")
        if not np.all(np.isclose(aff.tensor.max(axis=0), 1, atol=1e-10, rtol=0)):
            raise ValueError(f"{target}: F must select one alpha per assignment; stochastic alpha selection is not supported by this product builder")
        affs.append(aff)
        def tensor_info(a):
            return {"shape": list(a.shape), "values": a.tolist() if a.size <= 2000 else None}
        reports.append({"target": target, "parents": parents, "axes": ["feature"] + axes,
                        "merge": mode, "conflicting_assignments": collisions, "witness": witness,
                        "H_psi": tensor_info(aff.H_psi), "H_alpha": tensor_info(aff.H_alpha),
                        "F": tensor_info(aff.tensor)})
    compiled = ss.AffordanceSet(affs, spaces=spaces)
    compiled.transition = transition
    compiled.x_mode = None
    compiled.kernel_modes = []
    definitions = list(getattr(spaces[0], 'mode_definitions', []))
    if spec.get('x_identity_when') is not None:
        definitions.append(dict(name='identity', kind='identity', when=spec['x_identity_when'], labels=['normal movement','identity']))
    for definition in definitions:
        condition = gate(definition['when'], dims, features)
        variables = condition.vars()
        parents = [name for name in names if name in variables]
        has_action = 'action' in variables
        entries = math.prod([dims[p] for p in parents]) * (dims['action'] if has_action else 1)
        budget += 4*entries + 4
        if budget > h_budget:
            raise ValueError('X mode H tensors exceed tutorial budget')
        mode = gates.make_affordance('X', parents, [(condition, 1)], 0, 2, dims, has_action=has_action)
        compiled.kernel_modes.append(mode)
        if definition['kind'] == 'identity':
            compiled.x_mode = mode
        reports.append({'target':'X mode: ' + definition['name'], 'parents':parents, 'axes':['feature'] + (['action'] if has_action else []) + parents,
                        'merge':'reject', 'conflicting_assignments':0, 'witness':None,
                        'mode_labels':definition['labels'],
                        **{name:{'shape':list(value.shape), 'values':value.tolist() if value.size<=2000 else None}
                           for name,value in [('H_psi',mode.H_psi),('H_alpha',mode.H_alpha),('F',mode.tensor)]}})
    from tutorial_mode_kernel import ModeKernel
    compiled.root_kernel = ModeKernel(spaces[0].P_a_s_s, definitions)
    if spec.get('sokoban'):
        from tutorial_boxes import install
        install(spaces,compiled,reports,spec)
    return spaces, compiled, reports


def x_identity(affs, state, action):
    return any(e['kind']=='identity' and f.induced_alpha_at(state,action)==1
               for e,f in zip(affs.root_kernel.effects,affs.kernel_modes))


def local_row(sp, affs, state, action, alpha):
    if sp.name == 'X':
        modes = [f.induced_alpha_at(state, action) for f in affs.kernel_modes]
        return affs.root_kernel.row(state['X'], action, modes)
    return dense(sp.P_a_s_s[alpha, state[sp.name]])


def validate_joint(spaces, affs, joint):
    """Check every row's support and every stored probability against the factors.

    Exact support counts plus matching positive entries also check omitted zeros,
    without allocating the dense N*N tensor.
    """
    sizes = [s.ns for s in spaces]
    n = math.prod(sizes)
    states = np.unravel_index(np.arange(n), sizes)
    by_name = dict(zip([s.name for s in spaces], states))
    a_co, s_co, next_co = joint.coords
    next_states = np.unravel_index(next_co, sizes)
    expected = np.ones(joint.nnz)
    support = np.ones((spaces[0].na, n), dtype=np.int64)
    for i, sp in enumerate(spaces):
        p = dense(sp.P_a_s_s)
        actions = np.empty((spaces[0].na, n), dtype=int)
        identity = np.zeros((spaces[0].na, n), dtype=bool)
        door_mass = np.zeros((spaces[0].na, n))
        mode_values = {}
        for action in range(spaces[0].na):
            if i == 0:
                actions[action] = action
            elif sp.name not in affs.by_driven:
                actions[action] = getattr(sp, 'default_action_ind', 0)
            else:
                aff = affs.by_driven[sp.name]
                index = ((action,) if aff.has_action else ()) + tuple(by_name[name] for name in aff.parents)
                actions[action] = np.argmax(aff.tensor, axis=0)[index]
            if i == 0 and getattr(affs, 'x_mode', None) is not None:
                mode = affs.x_mode
                index = ((action,) if mode.has_action else ()) + tuple(by_name[name] for name in mode.parents)
                identity[action] = np.argmax(mode.tensor, axis=0)[index] == 1
            row_count = np.count_nonzero(p, axis=2)[actions[action], states[i]]
            if i == 0:
                for mode_ix, effect in enumerate(affs.root_kernel.effects):
                    if effect['kind'] != 'blocked_destination':
                        continue
                    mode = affs.kernel_modes[mode_ix]
                    index = ((action,) if mode.has_action else ()) + tuple(by_name[name] for name in mode.parents)
                    selected = np.broadcast_to(np.argmax(mode.tensor, axis=0)[index], (n,))
                    mode_values[action, mode_ix] = selected
                    door = effect
                    closed = (selected == 1) & (states[0] != door['cell'])
                    blocked = p[action, states[0], door['cell']] * closed
                    door_mass[action] += blocked
                    row_count = row_count - (blocked > 0)
                row_count = row_count + ((door_mass[action] > 0) & (p[action, states[0], states[0]] == 0))
            support[action] *= np.where(identity[action], 1, row_count)
        probabilities = p[actions[a_co, s_co], states[i][s_co], next_states[i]]
        if i == 0:
            for mode_ix, door in enumerate(affs.root_kernel.effects):
                if door['kind'] != 'blocked_destination':
                    continue
                selected = np.stack([mode_values[a, mode_ix] for a in range(spaces[0].na)])
                blocked = (selected[a_co, s_co] == 1) & (next_states[0] == door['cell']) & (states[0][s_co] != door['cell'])
                probabilities = np.where(blocked, 0, probabilities)
            probabilities = probabilities + (states[0][s_co] == next_states[0]) * door_mass[a_co, s_co]
        expected *= np.where(identity[a_co, s_co], states[i][s_co] == next_states[i], probabilities)
    counts = np.bincount(a_co*n+s_co, minlength=spaces[0].na*n).reshape(support.shape)
    if not np.array_equal(counts, support) or np.any(expected <= 0):
        raise ValueError('Full product support differs from the factorization')
    error = float(np.max(np.abs(joint.data - expected), initial=0))
    if error > 1e-12:
        raise ValueError(f'Full product differs from factorization: {error}')
    return {"rows_checked": int(support.size), "entries_checked": int(joint.nnz),
            "support_matches": True, "max_probability_error": error, "tolerance": 1e-12}


def step_model(model, state, action, mode, rng=None):
    """One synchronous transition; all gates read the same pre-step joint state."""
    if getattr(model['affs'], 'transition', None):
        from transition_affordance import step_model as transition_step
        return transition_step(model, state, action, mode, rng)
    rng = rng or np.random.default_rng()
    spaces, affs = model['spaces'], model['affs']
    action = integer(action, 'action', spaces[0].na)
    sd = {s.name: integer(state.get(s.name, 0), s.name, s.ns) for s in spaces}
    alphas, rows = {}, []
    for i, sp in enumerate(spaces):
        a = action if i == 0 else (affs.by_driven[sp.name].induced_alpha_at(sd, action)
             if sp.name in affs.by_driven else getattr(sp, 'default_action_ind', 0))
        alphas[sp.name] = int(a)
        rows.append(local_row(sp, affs, sd, action, a))
    if mode == 'factorized':
        nxt = [int(rng.choice(len(row), p=row / row.sum())) for row in rows]
    elif mode == 'tensor':
        if 'csr' not in model:
            raise ValueError('Build sparse product for this world and draft before using full-tensor controls')
        sizes = [s.ns for s in spaces]
        flat = np.ravel_multi_index(tuple(sd[s.name] for s in spaces), sizes)
        row = model['csr'][action].getrow(flat)
        choice = rng.choice(row.indices, p=row.data / row.data.sum())
        nxt = list(map(int, np.unravel_index(choice, sizes)))
    else:
        raise ValueError('Execution mode must be factorized or tensor')
    return {'before': sd, 'state': dict(zip([s.name for s in spaces], nxt)),
            'alphas': alphas, 'action': action, 'mode': mode,
            'environment_alphas': {d['name']:int(f.induced_alpha_at(sd, action)) for d,f in zip(affs.root_kernel.effects, affs.kernel_modes)},
            'x_dynamics': 'identity' if x_identity(affs, sd, action) else 'normal'}


def evaluate(available, spec, state=None, action=0, build=False, retain=None):
    spaces, affs, reports = compile_model(available, spec)
    if getattr(affs,'compact',False):
        from compact_product import evaluate as compact_evaluate
        return compact_evaluate(spaces,affs,reports,state,action,build,retain)
    if affs.transition:
        from transition_affordance import evaluate as transition_evaluate
        return transition_evaluate(available, spec, spaces, affs, reports, state, action, build, retain)
    action = integer(action, "action", spaces[0].na)
    state = state or {}
    sd = {s.name: integer(state.get(s.name, 0), s.name, s.ns) for s in spaces}
    local, alphas, max_fan = {}, {}, 1
    for s in spaces:
        p = s.P_a_s_s
        if hasattr(p, "coords") and getattr(p, "fill_value", 0) == 0:
            if not np.isfinite(p.data).all() or (p.data < 0).any() or not np.allclose(dense(p.sum(axis=2)), 1, atol=1e-10, rtol=0):
                raise ValueError(f"{s.name}.P must have normalized, nonnegative finite rows")
            occupied = p.data != 0
            fan = int(np.bincount(p.coords[0, occupied] * s.ns + p.coords[1, occupied], minlength=s.na*s.ns).max())
        else:
            p = dense(p)
            distribution(p, 2, f"{s.name}.P")
            fan = int(np.count_nonzero(p, axis=2).max())
        a = action if s.name == "X" else (affs.by_driven[s.name].induced_alpha_at(sd, action)
                if s.name in affs.by_driven else int(getattr(s, "default_action_ind", 0)))
        integer(a, f"{s.name} induced alpha", s.na)
        alphas[s.name] = a
        row = local_row(s, affs, sd, action, a)
        local[s.name] = [[int(i), float(row[i])] for i in np.flatnonzero(row)]
        max_fan *= fan
    sizes = [int(s.ns) for s in spaces]
    n = math.prod(sizes)
    bound = int(spaces[0].na) * n * max_fan
    result = {"spaces": [{"name": s.name, "ns": int(s.ns), "na": int(s.na)} for s in spaces],
              "affordances": reports, "state": sd, "action": action, "alphas": alphas,
              "local_next": local, "joint_states": n, "nnz_upper_bound": bound,
              "x_dynamics": 'identity' if x_identity(affs, sd, action) else 'normal',
              "can_build": n <= MAX_STATES and bound <= MAX_NNZ,
              "limits": {"states": MAX_STATES, "nnz": MAX_NNZ},
              "semantics": "All parents read current states. Local next states are conditionally independent given the current state and free action. X reads the free action, with optional identity-mode gating."}
    if build:
        if not result["can_build"]:
            raise ValueError(f"Product exceeds tutorial limits: {n:,} states, at most {bound:,} nonzeros; limits are {MAX_STATES:,} and {MAX_NNZ:,}")
        if any(math.prod(s.P_a_s_s.shape) > 20_000_000 for s in spaces):
            raise ValueError("Local kernel exceeds the original product builder's tutorial densification budget")
        row_fns = {'X': lambda sd, a: local_row(spaces[0], affs, sd, a, a)} if affs.kernel_modes else None
        joint = dbn_to_joint_operator(spaces, affs, root="X", na=spaces[0].na, next_row_fns=row_fns,
                                     max_states=MAX_STATES, max_nnz=MAX_NNZ, bytes_budget=32*MAX_NNZ, verbose=False)
        verification = validate_joint(spaces, affs, joint)
        sums = dense(joint.sum(axis=2))
        error = float(np.max(np.abs(sums - 1)))
        if error > 1e-10:
            raise ValueError(f"Product failed row normalization: {error}")
        flat = int(np.ravel_multi_index(tuple(sd[s.name] for s in spaces), sizes))
        row = dense(joint[action, flat])
        expected = np.array([1.])
        for s in spaces:
            r = np.zeros(s.ns)
            for i, p in local[s.name]: r[i] = p
            expected = np.kron(expected, r)
        row_error = float(np.max(np.abs(row - expected)))
        if row_error > 1e-10:
            raise ValueError(f"Product differs from factorized row: {row_error}")
        nz = np.flatnonzero(row)
        result["product"] = {"shape": list(joint.shape), "nnz": int(joint.nnz),
                              "verification": verification, "retained_for_play": retain is not None,
                              "max_row_sum_error": error, "selected_row_error": row_error,
                              "flat_state": flat, "row_nnz": int(len(nz)),
                              "next": [{"state": dict(zip([s.name for s in spaces], map(int, np.unravel_index(i, sizes)))),
                                        "flat": int(i), "probability": float(row[i])} for i in nz[:200]],
                              "truncated": len(nz) > 200}
        if retain is not None:
            retain.update(spaces=spaces, affs=affs,
                          csr=[joint[a].to_scipy_sparse().tocsr() for a in range(spaces[0].na)])
    return result
