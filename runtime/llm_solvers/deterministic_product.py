"""Vectorized Cartesian product for deterministic current-state H models.

Uses compiled F tensors and local kernels; rejects modes it cannot represent.
The ordinary DBN validator checks the resulting operator independently.
"""
import numpy as np
import sparse
from scipy.sparse import csr_matrix
import dbn_composer as d


def build(spaces,spec):
    ordered,affs,_=d.compile_model(spaces,spec)
    if affs.transition or any(e['kind']!='identity' for e in affs.root_kernel.effects):
        raise ValueError('Fast deterministic builder supports current-state H and identity modes only')
    sizes=[s.ns for s in ordered];n=int(np.prod(sizes));na=ordered[0].na
    if n>d.MAX_STATES or n*na>d.MAX_NNZ:raise ValueError('Product exceeds tutorial limits')
    rows=np.arange(n);coords=np.unravel_index(rows,sizes);state=dict(zip([s.name for s in ordered],coords))
    tables=[]
    for sp in ordered:
        p=d.dense(sp.P_a_s_s)
        if not np.all(np.count_nonzero(p,axis=-1)==1) or not np.all(p.max(axis=-1)==1):raise ValueError('Non-deterministic local kernel')
        tables.append(p.argmax(axis=-1))
    outputs=[];csr=[]
    for action in range(na):
        nxt=[]
        for i,sp in enumerate(ordered):
            if i==0:alpha=action
            elif sp.name not in affs.by_driven:alpha=getattr(sp,'default_action_ind',0)
            else:
                f=affs.by_driven[sp.name]
                ix=((action,) if f.has_action else ())+tuple(state[p] for p in f.parents)
                alpha=f.tensor.argmax(axis=0)[ix]
            dest=tables[i][alpha,coords[i]]
            if i==0 and affs.x_mode is not None:
                f=affs.x_mode;ix=((action,) if f.has_action else ())+tuple(state[p] for p in f.parents)
                dest=np.where(f.tensor.argmax(axis=0)[ix]==1,coords[i],dest)
            nxt.append(dest)
        col=np.ravel_multi_index(nxt,sizes);outputs.append(col)
        csr.append(csr_matrix((np.ones(n),col,np.arange(n+1)),shape=(n,n)))
    joint=sparse.COO(np.vstack([np.repeat(np.arange(na),n),np.tile(rows,na),np.concatenate(outputs)]),np.ones(na*n),shape=(na,n,n),has_duplicates=False,sorted=True)
    print('Checking all Cartesian rows against the DBN factors…',flush=True)
    verification=d.validate_joint(ordered,affs,joint)
    return {'spaces':ordered,'affs':affs,'csr':csr},verification
