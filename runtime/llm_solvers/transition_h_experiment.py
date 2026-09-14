"""Opt-in experiment: a child's Hψ reads the realized root transition.

No production imports this module. Existing current-state affordances are reused
unchanged unless explicitly overridden here. Only X' -> child-alpha dependencies
are supported; arbitrary next-state dependencies/cycles are intentionally absent.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dbn_composer as dbn


@dataclass(frozen=True)
class TransitionH:
    target: str
    phi: object
    action_from_features: object

    def evaluate(self, x_next, x, action, na):
        features = self.phi(x_next, x, action)
        alpha = dbn.integer(self.action_from_features(features), self.target + ' alpha', na)
        return features, alpha


class Experiment:
    def __init__(self, model, overrides=()):
        self.spaces, self.affs = model['spaces'], model['affs']
        self.names = [s.name for s in self.spaces]
        self.sizes = [s.ns for s in self.spaces]
        self.n = int(np.prod(self.sizes))
        self.overrides = {h.target:h for h in overrides}
        if len(self.overrides) != len(overrides) or any(n not in self.names[1:] for n in self.overrides):
            raise ValueError('Transition H targets must be unique non-root factors')

    def state(self, flat):
        return dict(zip(self.names, map(int, np.unravel_index(flat, self.sizes))))

    def validate(self, state, action):
        sd = {s.name:dbn.integer(state[s.name], s.name, s.ns) for s in self.spaces}
        return sd, dbn.integer(action, 'action', self.spaces[0].na)

    def conditional_row(self, sp, state, action, x_next):
        if sp.name in self.overrides:
            features, alpha = self.overrides[sp.name].evaluate(x_next, state['X'], action, sp.na)
        else:
            features = None
            aff = self.affs.by_driven.get(sp.name)
            alpha = aff.induced_alpha_at(state, action) if aff else getattr(sp, 'default_action_ind', 0)
        return dbn.local_row(sp, self.affs, state, action, alpha), features, int(alpha)

    def row(self, state, action):
        """Enumerate only supported X transitions and supported child outcomes."""
        sd, action = self.validate(state, action)
        px = dbn.local_row(self.spaces[0], self.affs, sd, action, action)
        entries = {}
        for xn in np.flatnonzero(px):
            branches = [((int(xn),), float(px[xn]))]
            for sp in self.spaces[1:]:
                row, _, _ = self.conditional_row(sp, sd, action, int(xn))
                branches = [(coord+(int(sn),), mass*float(row[sn]))
                            for coord,mass in branches for sn in np.flatnonzero(row)]
            for coord,mass in branches:
                flat = int(np.ravel_multi_index(coord, self.sizes))
                entries[flat] = entries.get(flat, 0.) + mass
        return entries

    def build(self, max_states=10000, max_nnz=1000000):
        if self.n > max_states:
            raise ValueError('Experiment exceeds its small-world state limit')
        kernels = []
        total = 0
        for action in range(self.spaces[0].na):
            indices, data, indptr = [], [], [0]
            for flat in range(self.n):
                entries = self.row(self.state(flat), action)
                if not np.isclose(sum(entries.values()), 1., atol=1e-12, rtol=0):
                    raise ValueError('Non-normalized experimental row')
                total += len(entries)
                if total > max_nnz:
                    raise ValueError('Experiment exceeds its nonzero limit')
                for dest,mass in sorted(entries.items()):
                    indices.append(dest);data.append(mass)
                indptr.append(len(indices))
            kernels.append(csr_matrix((data,indices,indptr),shape=(self.n,self.n)))
        return kernels

    def sample(self, state, action, rng):
        """Sample X' once, then use that same outcome in every child Hψ."""
        sd, action = self.validate(state, action)
        px = dbn.local_row(self.spaces[0], self.affs, sd, action, action)
        xn = int(rng.choice(self.spaces[0].ns, p=px))
        next_state, trace = {'X':xn}, {}
        for sp in self.spaces[1:]:
            row, features, alpha = self.conditional_row(sp, sd, action, xn)
            next_state[sp.name] = int(rng.choice(sp.ns, p=row))
            trace[sp.name] = {'psi':features,'alpha':alpha}
        return next_state, trace


def fixture(success=.75):
    """Three cells: free cell 0, button 1, door 2. α=3 toggles on departure."""
    p = np.zeros((4,3,3))  # interact, wait, left, right
    for a,x in product(range(4),range(3)):
        xn = max(0,x-1) if a==2 else min(2,x+1) if a==3 else x
        p[a,x,x] += 1-success
        p[a,x,xn] += success
    dp = np.array([np.eye(2), [[0,1],[0,1]], [[1,0],[1,0]], [[0,1],[1,0]]],float)
    x = SimpleNamespace(name='X',ns=3,na=4,P_a_s_s=p,default_action_ind=1,
        mode_definitions=[dict(name='door_passage',kind='blocked_destination',cell=2,
                               when={'var':'door','in':[0]},labels=['open','blocked'])])
    d = SimpleNamespace(name='door',ns=2,na=4,P_a_s_s=dp,default_action_ind=0)
    spec = dict(version=1,spaces=['X','door'],features={},affordances=[])
    spaces, affs, _ = dbn.compile_model({'X':x,'door':d},spec)
    h = TransitionH('door', lambda xn,x,a:(int(x==1),int(xn!=x)), lambda psi:3 if all(psi) else 0)
    return Experiment(dict(spaces=spaces,affs=affs),[h])


def demo_data():
    models = []
    for success in (1., .75):
        exp = fixture(success); kernels=exp.build()
        models.append(dict(success=success, base=exp.spaces[0].P_a_s_s.tolist(),
            door=exp.spaces[1].P_a_s_s.tolist(), joint=[k.toarray().tolist() for k in kernels],
            nnz=sum(k.nnz for k in kernels), bytes=sum(k.data.nbytes+k.indices.nbytes+k.indptr.nbytes for k in kernels)))
    return dict(models=models, button=1, door_cell=2, h_alpha=[0,0,0,3])


if __name__=='__main__':
    data=demo_data()
    if len(sys.argv)==3 and sys.argv[1]=='--export':
        Path(sys.argv[2]).write_text(json.dumps(data))
    for model in data['models']:
        print(f"success={model['success']}: 6 joint states, 4 actions, {model['nnz']} nonzeros, {model['bytes']} CSR bytes")
    exp=fixture(1.);state={'X':1,'door':0};rng=np.random.default_rng(4)
    for action in [0,1,2,3,0,3]:
        nxt,trace=exp.sample(state,action,rng)
        print(f"{state} -- {['interact','wait','left','right'][action]} --> {nxt}; H: {trace['door']}")
        state=nxt
