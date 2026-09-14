"""Solve the saved Sokoban DBN and audit its policy and full state-time eta."""
import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tutorial_server as server
import dbn_composer as dbn
import product_solver as product


def main():
    world, _, tutorial = server.load_level('sokoban/sokoban_01_two_boxes', with_tutorial=True)
    started = perf_counter()
    response = server.dbn_panel_data(world, dict(operation='solve', spec=tutorial['spec'],
        algorithm='feas_eta_fast_thresh', options={'objective': 'task'}))
    elapsed = perf_counter() - started
    model, solution = server._DBN_MODEL_CACHE['model'], server._DBN_MODEL_CACHE['solution']
    spaces = model['spaces']; names = [s.name for s in spaces]; dims = tuple(s.ns for s in spaces)
    n = int(np.prod(dims)); pi = solution['pi']; beta = solution['beta']
    flatten = lambda state: int(np.ravel_multi_index(tuple(state[name] for name in names), dims))
    decode = lambda i: dict(zip(names, map(int, np.unravel_index(i, dims))))
    labels = ['interact', 'stay', 'down', 'right', 'up', 'left']
    state = server.inspect_x0(world, {s.name: s for s in spaces})
    trajectory = [dict(time=0, state=state, flat=flatten(state))]
    seen = set()
    while not server.play_task_status(world, state)['complete']:
        flat = flatten(state)
        assert flat not in seen, 'Computed policy loops before reaching the goal'
        seen.add(flat)
        action = int(pi[flat])
        factor = dbn.step_model(model, state, action, 'factorized')
        tensor = dbn.step_model(model, state, action, 'tensor')
        assert factor['state'] == tensor['state']
        state = tensor['state']
        assert not server.play_task_status(world, state)['violations']
        trajectory[-1].update(action=action, action_label=labels[action])
        trajectory.append(dict(time=len(trajectory), state=state, flat=flatten(state)))
    # Independently follow every deterministic policy row until its stopping boundary.
    # Compare terminal joint state AND time, rather than merely checking total mass.
    next_state = np.array([model['csr'][int(pi[i])].indices[model['csr'][int(pi[i])].indptr[i]] for i in range(n)])
    assert all(np.all(np.diff(p.indptr) == 1) and np.all(p.data == 1) for p in model['csr'])
    assert np.all((beta == 0) | (beta == 1))
    for start in range(n):
        i = start; t = 0; visited = set()
        while beta[i] == 0:
            assert i not in visited, ('Unterminated policy loop', start)
            visited.add(i); i = int(next_state[i]); t += 1
        success = bool(solution['g'][int(pi[i]), i] and solution['c'][int(pi[i]), i])
        for outcome in ('positive', 'negative'):
            row = product.termination_row(solution, start, outcome)
            expected = [dict(terminal=i, time=t, probability=1.0)] if success == (outcome == 'positive') else []
            assert row['entries'] == expected, (start, outcome, row, expected)
    out = ROOT/'tutorial'/'artifacts'/'sokoban_01_product'; out.mkdir(parents=True, exist_ok=True)
    np.save(out/'policy.npy', pi)
    np.save(out/'kappa.npy', solution['fields']['kappa'])
    for outcome, key in [('positive', 'eta_pos'), ('negative', 'eta_neg')]:
        save_npz(out/f'eta_{outcome}_wide.npz', solution[key].matrix)
    start = trajectory[0]['flat']
    eta_pos = product.termination_row(solution, start, 'positive')
    eta_neg = product.termination_row(solution, start, 'negative')
    for entry in eta_pos['entries'] + eta_neg['entries']:
        entry['state'] = decode(entry['terminal'])
    report = dict(algorithm=solution['algorithm'], parameters={p['id']:p['default'] for p in server.ALGO_BY_ID[solution['algorithm']]['params']},
        states=n, actions=len(model['csr']), spaces=names, dimensions=dims,
        solve_seconds=solution['seconds'], build_and_solve_seconds=elapsed,
        start_kappa=float(solution['fields']['kappa'][start]), policy_steps=len(trajectory)-1,
        success=True, trajectory=trajectory, eta_positive=eta_pos, eta_negative=eta_neg,
        all_eta_rows_checked=n, eta_storage='wide CSR: row=start, column=time * number_of_joint_states + terminal',
        eta_shape=list(solution['eta_pos'].shape), eta_positive_nnz=solution['eta_pos'].matrix.nnz,
        eta_negative_nnz=solution['eta_neg'].matrix.nnz,
        audit='All start states: exact terminal state, stopping time, outcome and probability match direct deterministic policy propagation. Both execution modes agree along the successful rollout.')
    (out/'policy_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
