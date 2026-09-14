"""Local P_X(x' | x, a, alpha_e), evaluated without enumerating mode vectors.

The kernel receives only X's current state, the free action, and induced mode
alphas. Reading other state spaces is exclusively the affordance functions' job.
"""
import numpy as np


class ModeKernel:
    def __init__(self, base, effects):
        self.base = base
        self.effects = effects

    def row(self, x, action, alphas):
        value = self.base[action, x]
        row = np.asarray(value.todense() if hasattr(value, 'todense') else value, dtype=float).copy()
        for effect, alpha in zip(self.effects, alphas):
            if not alpha:
                continue
            if effect['kind'] == 'identity':
                row[:] = 0
                row[x] = 1
                return row
            if effect['kind'] == 'blocked_destination':
                cell = effect['cell']
                if cell != x:
                    row[x] += row[cell]
                    row[cell] = 0
        return row
