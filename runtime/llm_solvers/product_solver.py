"""Tutorial product-state policies and conditional slices (Codex).

CSR actions and full state-time termination kernels for feasibility policies.
Eta rows index starting joint states; columns index time and terminal joint state.
"""
from time import perf_counter
import operator
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order


# Stored eta entries; sparse multiplication/conversion also needs working memory.
MAX_ETA_NNZ = 80_000_000


def coordinates(model):
    dims = tuple(s.ns for s in model['spaces'])
    return dims, np.array(np.unravel_index(np.arange(np.prod(dims)), dims))


def objective(model, world, options, clauses=None, death_names=()):
    dims, coords = coordinates(model)
    names = [s.name for s in model['spaces']]
    na, n = len(model['csr']), int(np.prod(dims))
    g = np.zeros((na, n))
    if options.get('objective', 'grid') == 'task':
        if not clauses:
            raise ValueError('Author an objective in the Task tab first.')
        ops = {'==': operator.eq, '!=': operator.ne, '>': operator.gt,
               '<': operator.lt, '>=': operator.ge, '<=': operator.le, 'in': np.isin}
        goal = np.zeros(n, dtype=bool)
        for clause in clauses:
            if not clause:
                raise ValueError('Empty task clause is not an objective.')
            match = np.ones(n, dtype=bool)
            for name, op, value in clause:
                if name not in names or op not in ops:
                    raise ValueError(f'Invalid product task literal: {name} {op}')
                match &= ops[op](coords[names.index(name)], value)
            goal |= match
        g[:, goal] = 1
    else:
        group = next((x for x in world['goals'] if str(x['id']) == str(options.get('group'))), None)
        if not group or not group['cells']:
            raise ValueError('Select a nonempty grid goal group.')
        for cell in group['cells']:
            mask = coords[0] == cell['li']
            if cell['a'] is None:
                g[:, mask] = 1
            else:
                g[int(cell['a']), mask] = 1
    bad = np.isin(coords[0], world['constraints'])
    if options.get('physio_constraints', True):
        for name in death_names:
            if name in names:
                bad |= coords[names.index(name)] == 0
    c = np.broadcast_to((~bad).astype(float), (na, n)).copy()
    g *= c  # A constraint takes precedence over success.
    if not g.any():
        raise ValueError('The selected objective has no admissible joint goal states.')
    return g, c


def almost_sure(P, goal):
    """Qualitative reachability: remove traps, then actions that can leave the set.

    Nested fixed point for a finite MDP, using graph support rather than a
    numerical probability-near-one cutoff (which misclassifies rare failures).
    """
    n = P[0].shape[0]
    support = [a.astype(bool).astype(np.int32) for a in P]
    for a in support:
        a.eliminate_zeros()
    keep = np.ones(n, dtype=bool)
    terminal = goal.any(axis=0)
    while True:
        allowed = np.array([(a @ (~keep).astype(np.int32)) == 0 for a in support]) | goal
        graph = sum((sparse.diags((allowed[i] & ~goal[i]).astype(np.int32)) @ a for i, a in enumerate(support)), sparse.csr_matrix((n, n)))
        reverse = sparse.bmat([[graph.T, None], [sparse.csr_matrix(terminal[None, :]), sparse.csr_matrix((1, 1))]], format='csr')
        found = breadth_first_order(reverse, n, directed=True, return_predecessors=False)
        new = np.zeros(n + 1, dtype=bool)
        new[found] = True
        new = new[:n] & keep
        if np.array_equal(new, keep):
            return keep, allowed
        keep = new


def value_iteration(P, g, c, algorithm, params):
    n = P[0].shape[0]
    gamma = float(params.get('gamma', .95)) if algorithm == 'ihdc' else 1.
    cost = float(params.get('step_cost', 1))
    penalty = float(params.get('constraint_penalty', 10))
    reward = float(params.get('goal_reward', 0)) if algorithm == 'ihdc' else 0.
    tol = float(params.get('epsilon', 1e-8))
    if not (0 < gamma <= 1 and (algorithm != 'ihdc' or gamma < 1)) or not cost > 0 or penalty < 0 or not np.isfinite([gamma, cost, penalty, reward, tol]).all() or not 0 < tol <= .01:
        raise ValueError('Use 0 < discount < 1, positive step cost and tolerance, and nonnegative penalty.')
    r = np.where(g > 0, reward, -cost - penalty * (1 - c))
    finite, allowed = (almost_sure(P, g > 0) if algorithm == 'ssp'
                       else (np.ones(n, dtype=bool), np.ones_like(g, dtype=bool)))
    v = np.zeros(n)
    residual = 0.
    for iteration in range(1, 20001):
        q = r + gamma * np.array([a @ v for a in P])
        if algorithm == 'ssp':
            q[g > 0] = 0
            q[~allowed] = -np.inf
        new = q.max(axis=0)
        new[~finite] = 0  # Divergent states are excluded, never used by allowed actions.
        residual = float(np.max(np.abs(new[finite] - v[finite]), initial=0))
        v = new
        if residual <= tol * (1 - gamma if gamma < 1 else 1):
            break
    else:
        raise ValueError(f'Value iteration did not converge after 20,000 backups (residual {residual:.3g}). No solution was published.')
    q = r + gamma * np.array([a @ v for a in P])
    if algorithm == 'ssp':
        q[g > 0] = 0
        q[~allowed] = -np.inf
    pi = q.argmax(axis=0)
    v[~finite] = -np.inf
    pi[~finite] = -1
    return v, pi, {'iterations': iteration, 'residual': residual,
                   'infinite_cost_states': int((~finite).sum()),
                   'error_bound': residual / (1 - gamma) if gamma < 1 else None}


def solve(model, g, c, algorithm, params):
    P = model['csr']
    start = perf_counter()
    phases = {}
    notes = []
    termination = {}
    if algorithm in ('ssp', 'ihdc'):
        v, pi, phases = value_iteration(P, g, c, algorithm, params)
        fields = {'value': v}
        notes.append('First exit ends at the objective; nonterminating states have value −∞.' if algorithm == 'ssp' else 'Discounted continuing rewards; the objective is not absorbing.')
        notes.append('Constraints incur the selected extra cost; they cannot count as success.')
    else:
        import feas_two_phase_sparse as fs
        import feas_thresh_sparse as ft
        known = {'feas', 'feas2', 'feas2s', 'feas_thresh', 'feas_thresh_s',
                 'feas_eta_fast', 'feas_eta_fast_thresh', 'feas_eta_combined', 'feas_eta_combined_thresh'}
        if algorithm not in known:
            raise ValueError('Unknown product algorithm')
        rd = int(params.get('round_decimals', 0))
        rd = rd if 0 < rd < 15 else None
        theta = float(params.get('risk_thresh', .5))
        ke, ee = float(params.get('kappa_epsilon', 1e-9)), float(params.get('ET_epsilon', 1e-5))
        kt = float(params.get('kappa_tol', 0))
        if not np.isfinite([theta, ke, ee, kt]).all() or not 0 <= theta <= 1 or not 0 <= ke <= .5 or not 0 <= ee <= 100 or not 0 <= kt <= 1:
            raise ValueError('Invalid risk threshold or convergence tolerance')
        kw = dict(round_decimals=rd, kappa_epsilon=ke, ET_epsilon=ee, timing=phases)
        if algorithm in ('feas', 'feas2', 'feas_thresh'):
            if P[0].shape[0] > 128:
                raise ValueError('The dense reference algorithms are limited to 128 joint states. Select a sparse algorithm for this product.')
            dense = np.array([a.toarray() for a in P])
            import TGMDP_methods as tg
            if algorithm == 'feas2':
                from feas_two_phase import feas_iter_two_phase
                result = feas_iter_two_phase(dense, g, c, max_time=100, kappa_tol=kt, **kw)
                notes.append('Dense reference eta horizon: 100 steps (policy iteration is independent).')
            else:
                result = tg.feas_iter_stationary_flat_dense(dense, g, c, use_ET=True)
                if algorithm == 'feas_thresh':
                    result = tg.feas_iter_stationary_flat_dense_thresh(dense, dense, g, c, result[3].sum(axis=(1, 2)), risk_thresh=theta, use_ET=True)
                notes.append('Original dense reference retains its diameter-based iteration cap; convergence is not guaranteed.')
        else:
            thresholded = algorithm.endswith('thresh') or algorithm == 'feas_thresh_s'
            calls = 0
            def builder(*args):
                nonlocal calls
                calls += 1
                # Threshold pass 1 always uses the original negative recurrence.
                combined = 'combined' in algorithm and (not thresholded or calls > 1)
                from eta_fast import build_eta
                from eta_combined import build_combined_eta
                compute = build_combined_eta if combined else build_eta
                return compute(*args, mode=1, max_nnz=MAX_ETA_NNZ)
            solver = ft.feas_iter_two_phase_sparse_thresh if thresholded else fs.feas_iter_two_phase_sparse
            extra = {'risk_thresh': theta} if thresholded else {'kappa_tol': kt}
            result = solver(P, g, c, compute_eta=builder, **kw, **extra)
            notes.append('Full state-time eta retained in wide sparse storage: start joint state × (time, terminal joint state). Product solves use wide storage for all sparse variants.')
        kappa, pi = result[:2]
        fields = {'kappa': kappa}
        termination = {'eta_pos': result[2], 'eta_neg': result[3]}
        if len(result) > 4:
            termination['beta'] = result[4]
        notes.append('Eta uses the solver’s finite horizon and tail tolerance; it is not an infinite-time completion guarantee.')
    return {'fields': fields, 'pi': pi, 'algorithm': algorithm, 'g': g, 'c': c,
            'seconds': perf_counter() - start, 'phases': phases, 'notes': notes, **termination}


def termination_row(solution, start, outcome='positive', offset=0, limit=2000):
    """Page a state-time row without allocating a dense joint-state × time array."""
    from eta_combined import OutcomeView
    if outcome not in ('positive', 'negative'):
        raise ValueError('Eta outcome must be positive or negative')
    eta = solution.get('eta_pos' if outcome == 'positive' else 'eta_neg')
    if eta is None:
        raise ValueError('This solution does not contain a state-time termination kernel')
    n, _, horizon = eta.shape
    if isinstance(start, bool) or int(start) != start or not 0 <= start < n:
        raise ValueError('Eta start must be a valid joint-state index')
    if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise ValueError('Eta paging requires offset >= 0 and limit in [1, 10000]')
    source = eta.combined if isinstance(eta, OutcomeView) else eta
    if hasattr(source, 'matrix'):
        row = source.matrix.getrow(int(start))
        terminals, times, mass = row.indices % n, row.indices // n, row.data
        if isinstance(eta, OutcomeView):
            mass = mass * eta.weights[terminals]
        keep = mass != 0
        terminals, times, mass = terminals[keep], times[keep], mass[keep]
    else:
        # Dense reference solvers are limited to 128 joint states.
        dense = eta[int(start)]
        terminals, times = np.nonzero(dense)
        mass = dense[terminals, times]
    page = slice(offset, offset + limit)
    return {'start': int(start), 'outcome': outcome, 'shape': [n, n, horizon],
            'axes': ['start_joint_state', 'terminal_joint_state', 'time'],
            'entries': [{'terminal': int(j), 'time': int(t), 'probability': float(p)}
                        for j, t, p in zip(terminals[page], times[page], mass[page])],
            'total_entries': len(mass), 'total_mass': float(mass.sum()),
            'next_offset': offset + limit if offset + limit < len(mass) else None}


def slice_solution(model, solution, fixed, cell, rows, cols, walls=()):
    spaces = model['spaces']
    dims = tuple(s.ns for s in spaces)
    if dims[0] != rows * cols:
        raise ValueError('Base-space size does not match the grid')
    state = {}
    for s in spaces[1:]:
        value = fixed.get(s.name, s.ns - 1)
        if isinstance(value, bool) or int(value) != value or not 0 <= value < s.ns:
            raise ValueError(f'{s.name} must be an integer in [0, {s.ns - 1}]')
        state[s.name] = int(value)
    if isinstance(cell, bool) or int(cell) != cell or not 0 <= cell < dims[0]:
        raise ValueError('Selected cell is outside the grid')
    ids = np.ravel_multi_index((np.arange(dims[0]), *[state[s.name] for s in spaces[1:]]), dims)
    field = next(iter(solution['fields']))
    full = solution['fields'][field]
    values = full[ids].copy()
    visible = np.ones(dims[0], dtype=bool)
    visible[list(walls)] = False
    values[~visible] = np.nan
    gradient = []
    for i in range(dims[0]):
        r, col = divmod(i, cols)
        def derivative(lo, hi):
            if not np.isfinite(values[i]):
                return None
            low = lo is not None and np.isfinite(values[lo])
            high = hi is not None and np.isfinite(values[hi])
            if low and high:
                return float((values[hi] - values[lo]) / 2)
            if high:
                return float(values[hi] - values[i])
            if low:
                return float(values[i] - values[lo])
            return None
        gradient.append([derivative(i - 1 if col else None, i + 1 if col < cols - 1 else None),
                         derivative(i - cols if r else None, i + cols if r < rows - 1 else None)])
    selected = int(ids[int(cell)])
    def number(v):
        return float(v) if np.isfinite(v) else None
    actions = []
    for a, p in enumerate(model['csr']):
        row = p.getrow(selected)
        expected = row.data @ full[row.indices]
        actions.append({'action': a, 'expected_next': number(expected),
                        'negative_infinity': bool(np.isneginf(expected)),
                        'delta': number(expected - full[selected]) if np.isfinite(full[selected]) else None})
    return {'field': field, 'values': [number(v) for v in values],
            'negative_infinity': np.isneginf(values).tolist(), 'gradient': gradient,
            'policy': solution['pi'][ids].tolist(), 'fixed': state, 'cell': int(cell),
            'value': number(full[selected]), 'selected_policy': int(solution['pi'][selected]),
            'actions': actions, 'rows': rows, 'cols': cols,
            'selected_goal_actions': np.flatnonzero(solution['g'][:, selected]).tolist()}
