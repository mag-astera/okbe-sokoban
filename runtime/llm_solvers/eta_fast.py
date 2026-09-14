"""Codex eta builders: wide CSR output or compact propagation with sparse boundaries.

Both preserve the original horizon, per-outcome tail stopping rule, and distinct
success/failure continuation gates. No nonzero probabilities are sparsified away.
"""
import numpy as np
from scipy import sparse as sp


class WideEta:
    """One CSR matrix: row=start, column=time*nx+terminal. No per-time dictionary."""
    def __init__(self, matrix, nx, T):
        self.matrix = matrix.tocsr()
        self.nx, self.T = nx, T
        self.shape = (nx, nx, T)

    def to_dense(self):
        return self.matrix.toarray().reshape(self.nx, self.T, self.nx).transpose(0, 2, 1)

    def __getitem__(self, x):
        return self.matrix.getrow(x).toarray().reshape(self.T, self.nx).T

    def sum(self, axis=None):
        if axis is None:
            return float(self.matrix.sum())
        if axis == (1, 2) or axis == [1, 2]:
            return np.asarray(self.matrix.sum(axis=1)).ravel()
        coo = self.matrix.tocoo()
        if axis == 2:
            return sp.coo_matrix((coo.data, (coo.row, coo.col % self.nx)),
                                 shape=(self.nx, self.nx)).toarray()
        if axis == (0, 1) or axis == [0, 1]:
            return np.bincount(coo.col // self.nx, weights=coo.data, minlength=self.T)
        raise NotImplementedError(axis)

    def to_sparse_stok(self, tol=0.0):
        from stok_chain import SparseSTOK
        coo = self.matrix.tocoo()
        keep = abs(coo.data) > tol
        pairs, inverse = np.unique(coo.row[keep] * self.nx + coo.col[keep] % self.nx,
                                   return_inverse=True)
        values = np.zeros((len(pairs), self.T))
        values[inverse, coo.col[keep] // self.nx] = coo.data[keep]
        out = SparseSTOK(pairs // self.nx, pairs % self.nx, values, self.nx)
        out.T = self.T
        return out

    def nbytes(self):
        m = self.matrix
        return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes

    def dense_nbytes(self):
        return self.nx ** 2 * self.T * 8


class CompactEta:
    """Sparse diagonal at t=0; only terminal columns reachable at t>=1 are propagated.

    Infeasible/wall states often account for most terminal columns but have no
    incoming live transitions. They cost one number here, not nx*T numbers.
    """
    def __init__(self, seed, terms, tail, T):
        self.seed, self.terms, self.tail = seed, terms, tail  # tail: time,start,terminal
        self.nx, self.T = len(seed), T
        self.shape = (self.nx, self.nx, T)

    def to_dense(self):
        out = np.zeros(self.shape)
        x = np.arange(self.nx)
        out[x, x, 0] = self.seed
        out[:, self.terms, 1:1+len(self.tail)] = self.tail.transpose(1, 2, 0)
        return out

    def __getitem__(self, x):
        out = np.zeros((self.nx, self.T))
        out[x, 0] = self.seed[x]
        out[self.terms, 1:1+len(self.tail)] = self.tail[:, x, :].T
        return out

    def sum(self, axis=None):
        if axis is None:
            return float(self.seed.sum() + self.tail.sum())
        if axis == (1, 2) or axis == [1, 2]:
            return self.seed + self.tail.sum(axis=(0, 2))
        if axis == 2:
            out = np.diag(self.seed)
            out[:, self.terms] += self.tail.sum(axis=0)
            return out
        if axis == (0, 1) or axis == [0, 1]:
            out = np.zeros(self.T)
            out[0] = self.seed.sum()
            out[1:1+len(self.tail)] = self.tail.sum(axis=(1, 2))
            return out
        raise NotImplementedError(axis)

    def to_sparse_stok(self, tol=0.0):
        from stok_chain import SparseSTOK
        active = np.any(abs(self.tail) > tol, axis=0)
        rows, cols = np.nonzero(active)
        terminal = self.terms[cols]
        diag = np.flatnonzero(abs(self.seed) > tol)
        pairs = np.union1d(rows*self.nx+terminal, diag*self.nx+diag)
        values = np.zeros((len(pairs), self.T))
        if len(rows):
            values[np.searchsorted(pairs, rows*self.nx+terminal), 1:1+len(self.tail)] = self.tail[:,rows,cols].T
        values[np.searchsorted(pairs, diag*self.nx+diag), 0] = self.seed[diag]
        out = SparseSTOK(pairs//self.nx, pairs%self.nx, values, self.nx)
        out.T = self.T
        return out

    def nbytes(self):
        return self.seed.nbytes + self.terms.nbytes + self.tail.nbytes

    def dense_nbytes(self):
        return self.nx**2*self.T*8


def check_eta_size(nnz, max_nnz):
    if max_nnz is not None and nnz > max_nnz:
        raise ValueError(f'State-time eta exceeds the {max_nnz:,}-nonzero storage limit. '
                         'No outcome-total approximation was substituted. Reduce the product size.')


def build_eta(P, seeds, gates, T, floor, tolerance, mode=0, max_nnz=None):
    """0=automatic, 1=wide sparse, 2=compact dense tails.

    Wide mode uses one block-diagonal sparse propagation operator for both outcomes.
    Compact mode folds gates into sparse operators and prunes structurally empty
    columns before propagation. Both keep each outcome's independent stopping rule.
    """
    n = P.shape[0]
    if max_nnz is not None and mode != 1:
        raise ValueError('An eta nonzero limit requires wide sparse mode (1).')
    Qs = [(sp.diags(gate) @ P).tocsr() for gate in gates]
    for Q in Qs:
        Q.eliminate_zeros()
    deterministic = max(np.diff(Q.indptr).max(initial=0) for Q in Qs) <= 1
    if mode == 0:
        # Deterministic chains do not spread probability over many destinations.
        mode = 1 if deterministic else 2
    if mode == 1 and deterministic:
        return _deterministic_eta(Qs, seeds, T, floor, tolerance, max_nnz)
    if mode == 1:
        Q = sp.block_diag(Qs, format='csr')
        current = sp.vstack([sp.diags(seed, format='csr') for seed in seeds], format='csr')
        blocks = [current]
        stored = current.nnz
        check_eta_size(stored, max_nnz)
        below, stopped = [0, 0], [False, False]
        for t in range(1, T):
            current = Q @ current
            halves = [current[:n].tocsr(), current[n:].tocsr()]
            for i in range(2):
                if stopped[i]:
                    halves[i] = sp.csr_matrix((n, n))
                elif t >= floor:
                    below[i] = below[i]+1 if halves[i].sum() < tolerance else 0
                    stopped[i] = below[i] >= 5
            current = sp.vstack(halves, format='csr')
            current.eliminate_zeros()
            stored += current.nnz
            check_eta_size(stored, max_nnz)
            blocks.append(current)
            if all(stopped):
                break
        wide = sp.hstack(blocks, format='csr')
        wide.resize((2*n, n*T))
        return _split_wide(wide, n, T)
    if mode != 2:
        raise ValueError('eta mode must be 0, 1, or 2')
    results = []
    for Q, seed in zip(Qs, seeds):
        terms = np.flatnonzero(seed)
        first = Q[:, terms].multiply(seed[terms]).tocsr()
        first.eliminate_zeros()
        active = np.unique(first.indices)
        terms = terms[active]
        # Grow geometrically so early stopping does not allocate the whole horizon.
        capacity = min(32, max(0, T-1))
        tail = np.empty((capacity, n, len(terms)))
        used, below = 0, 0
        if T > 1 and len(terms):
            current = first[:, active].toarray()
            for t in range(1, T):
                if used == len(tail):
                    expanded = np.empty((min(T-1, max(1, 2*used)), n, len(terms)))
                    expanded[:used] = tail
                    tail = expanded
                tail[used] = current
                used += 1
                if t >= floor:
                    below = below+1 if current.sum() < tolerance else 0
                    if below >= 5:
                        break
                if t+1 < T:
                    current = Q @ current
        # Copy only when spare capacity would retain a substantial unused allocation.
        tail = tail[:used].copy() if used < len(tail) else tail
        results.append(CompactEta(seed, terms, tail, T))
    return tuple(results)


def _deterministic_eta(Qs, seeds, T, floor, tolerance, max_nnz=None):
    """Unique-successor paths: vectorized pointer following, no matrix products.

    Also supports sub-stochastic weights and soft terminal seeds. This shortcut is
    used only when each continuation row has at most one nonzero destination.
    """
    n = len(seeds[0])
    rows, cols, data = [], [], []
    stored = 0
    for outcome, (Q, seed) in enumerate(zip(Qs, seeds)):
        successor = np.arange(n)
        weights = np.zeros(n)
        nonempty = np.flatnonzero(np.diff(Q.indptr))
        successor[nonempty] = Q.indices[Q.indptr[nonempty]]
        weights[nonempty] = Q.data[Q.indptr[nonempty]]
        state = np.arange(n)
        survival = np.ones(n)
        below = 0
        for t in range(T):
            mass = survival * seed[state]
            used = np.flatnonzero(mass)
            stored += len(used)
            check_eta_size(stored, max_nnz)
            rows.append(used + outcome*n)
            cols.append(t*n+state[used])
            data.append(mass[used])
            if t >= max(1, floor):
                below = below+1 if mass.sum() < tolerance else 0
                if below >= 5:
                    break
            survival *= weights[state]
            state = successor[state]
    wide = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(2*n,n*T)).tocsr()
    return _split_wide(wide, n, T)


class _RiskMass:
    def __init__(self, total):
        self.total = total

    def sum(self, axis):
        if axis != (1, 2):
            raise NotImplementedError(axis)
        return self.total


def build_risk_mass(P, seeds, gates, T, floor, tolerance):
    """Threshold pass 1 only needs sum_{terminal,time} eta^-.

    Linearity allows summing terminal columns BEFORE propagation. Preserve the
    finite horizon and tail rule (unlike an infinite-horizon absorption solve,
    which could change the risk gate). No success tensor or terminal axis needed.
    """
    Q = (sp.diags(gates[1]) @ P).tocsr()
    current = seeds[1].copy()
    total = current.copy()
    below = 0
    for t in range(1, T):
        current = Q @ current
        total += current
        if t >= floor:
            below = below+1 if current.sum() < tolerance else 0
            if below >= 5:
                break
    return None, _RiskMass(total)


def policy_kernel(action_kernels, policy):
    """Select policy rows in sparse batches, avoiding one Python/getrow call per state."""
    n = len(policy)
    result = sp.csr_matrix((n,n))
    for a, kernel in enumerate(action_kernels):
        mask = (policy == a).astype(float)
        if mask.any():
            result += sp.diags(mask) @ kernel
    result.eliminate_zeros()
    return result


def _split_wide(wide, n, T):
    """Two outcome views backed by ONE wide CSR allocation of values and indices."""
    outputs = []
    for offset in (0, n):
        begin, end = wide.indptr[offset], wide.indptr[offset+n]
        view = sp.csr_matrix((wide.data[begin:end], wide.indices[begin:end],
                              wide.indptr[offset:offset+n+1]-begin),
                             shape=(n,n*T), copy=False)
        # SciPy may copy a small slice during CSR construction to release a large
        # parent allocation. Here sharing that parent is intentional.
        view.data = wide.data[begin:end]
        view.indices = wide.indices[begin:end]
        out = WideEta(view, n, T)
        out.storage = wide
        outputs.append(out)
    return tuple(outputs)
