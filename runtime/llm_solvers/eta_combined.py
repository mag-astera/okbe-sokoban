"""One STOK recurrence, one combined termination boundary (Codex).

E[0]=diag(b_plus+b_minus); E[t]=diag(feasible*h_pi) P_pi E[t-1].
All termination events, including infeasibility, stop continuation. Unlike the
reference positive recurrence, this cannot propagate success from an infeasible
state when kappa rounding has classified a weakly reachable state as infeasible.
"""
from functools import partial
import numpy as np
from scipy import sparse as sp
from eta_fast import CompactEta, WideEta, build_eta, build_risk_mass, check_eta_size
from feas_two_phase_sparse import feas_iter_two_phase_sparse
from feas_thresh_sparse import feas_iter_two_phase_sparse_thresh


class OutcomeView:
    """Terminal-weighted view of a single STOK. No independent eta propagation/storage."""
    def __init__(self, combined, weights):
        self.combined, self.weights = combined, weights
        self.nx, self.T, self.shape = combined.nx, combined.T, combined.shape

    def to_dense(self):
        return self.combined.to_dense() * self.weights[None,:,None]

    def __getitem__(self,x):
        return self.combined[x] * self.weights[:,None]

    def sum(self,axis=None):
        e, w = self.combined, self.weights
        if axis == 2:
            return e.sum(2) * w[None,:]
        if axis is None:
            return float(self.sum((1,2)).sum())
        if isinstance(e,CompactEta):
            if axis == (1,2) or axis == [1,2]:
                return e.seed*w + np.einsum('txj,j->x',e.tail,w[e.terms])
            if axis == (0,1) or axis == [0,1]:
                out=np.zeros(self.T)
                out[0]=e.seed@w
                out[1:1+len(e.tail)]=np.einsum('txj,j->t',e.tail,w[e.terms])
                return out
        else:
            coo=e.matrix.tocoo()
            data=coo.data*w[coo.col%self.nx]
            if axis == (1,2) or axis == [1,2]:
                return np.bincount(coo.row,weights=data,minlength=self.nx)
            if axis == (0,1) or axis == [0,1]:
                return np.bincount(coo.col//self.nx,weights=data,minlength=self.T)
        raise NotImplementedError(axis)

    def to_sparse_stok(self,tol=0.0):
        from stok_chain import SparseSTOK
        source=self.combined.to_sparse_stok()
        values=source.tmat*self.weights[source.j,None]
        keep=np.any(abs(values)>tol,axis=1)
        out=SparseSTOK(source.i[keep],source.j[keep],values[keep],self.nx)
        out.T=self.T
        return out


def combined_stok(P,seed,gate,T,floor,tolerance,mode=0,max_nnz=None):
    """Compute E once. Auto uses wide CSR for unique-successor continuations."""
    if max_nnz is not None and mode != 1:
        raise ValueError('An eta nonzero limit requires wide sparse mode (1).')
    Q=(sp.diags(gate)@P).tocsr()
    Q.eliminate_zeros()
    n=len(seed)
    deterministic=np.diff(Q.indptr).max(initial=0)<=1
    if mode==0:
        mode=1 if deterministic else 2
    if mode==2:
        # The compact builder accepts any number of outcomes; here it gets exactly ONE.
        return build_eta(P,(seed,),(gate,),T,floor,tolerance,mode=2)[0]
    if mode!=1:
        raise ValueError('eta mode must be 0, 1, or 2')
    if deterministic:
        successor=np.arange(n)
        weights=np.zeros(n)
        live=np.flatnonzero(np.diff(Q.indptr))
        successor[live]=Q.indices[Q.indptr[live]]
        weights[live]=Q.data[Q.indptr[live]]
        state=np.arange(n)
        survival=np.ones(n)
        rows,cols,data=[],[],[]
        stored=0
        below=0
        for t in range(T):
            mass=survival*seed[state]
            used=np.flatnonzero(mass)
            stored+=len(used)
            check_eta_size(stored,max_nnz)
            rows.append(used);cols.append(t*n+state[used]);data.append(mass[used])
            if t>=max(1,floor):
                below=below+1 if mass.sum()<tolerance else 0
                if below>=5:
                    break
            survival*=weights[state]
            state=successor[state]
        wide=sp.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n*T)).tocsr()
    else:
        current=sp.diags(seed,format='csr')
        blocks=[current]
        stored=current.nnz
        check_eta_size(stored,max_nnz)
        below=0
        for t in range(1,T):
            current=Q@current
            current.eliminate_zeros()
            stored+=current.nnz
            check_eta_size(stored,max_nnz)
            blocks.append(current)
            if t>=floor:
                below=below+1 if current.sum()<tolerance else 0
                if below>=5:
                    break
        wide=sp.hstack(blocks,format='csr')
        wide.resize((n,n*T))
    return WideEta(wide,n,T)


def build_combined_eta(P,seeds,gates,T,floor,tolerance,mode=0,max_nnz=None):
    seed=seeds[0]+seeds[1]
    combined=combined_stok(P,seed,gates[1],T,floor,tolerance,mode,max_nnz)
    success=np.divide(seeds[0],seed,out=np.zeros_like(seed),where=seed!=0)
    failure=np.divide(seeds[1],seed,out=np.zeros_like(seed),where=seed!=0)
    return OutcomeView(combined,success),OutcomeView(combined,failure)


def feas_iter_combined_eta(*args,eta_mode=0,**kwargs):
    return feas_iter_two_phase_sparse(*args,compute_eta=partial(build_combined_eta,mode=eta_mode),**kwargs)


def feas_iter_combined_eta_thresh(*args,eta_mode=0,**kwargs):
    first_pass=True
    def builder(*builder_args):
        nonlocal first_pass
        if first_pass:
            first_pass=False
            return build_risk_mass(*builder_args)
        return build_combined_eta(*builder_args,mode=eta_mode)
    return feas_iter_two_phase_sparse_thresh(*args,compute_eta=builder,**kwargs)
