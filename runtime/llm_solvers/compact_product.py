"""Product preview/build for predicate affordances, without dense H/F tables."""
import math
import numpy as np
import dbn_composer as base
from transition_affordance import TransitionModel


def evaluate(spaces,affs,reports,state,action,build,retain):
    tm=TransitionModel(spaces,affs)
    a=base.integer(action,'action',spaces[0].na)
    sd={s.name:base.integer((state or {}).get(s.name,0),s.name,s.ns) for s in spaces}
    alphas={}; local={}; fan=1
    for sp in spaces:
        alpha=a if sp.name=='X' else (affs.by_driven[sp.name].induced_alpha_at(sd,a) if sp.name in affs.by_driven else getattr(sp,'default_action_ind',0))
        if sp.name in affs.transition:continue
        row=base.local_row(sp,affs,sd,a,alpha);alphas[sp.name]=alpha
        local[sp.name]=[[int(i),float(row[i])] for i in np.flatnonzero(row)]
    # Bound local support only; no joint configurations are visited for preview.
    for sp in spaces:
        p=sp.P_a_s_s
        if hasattr(p,'coords'):
            count=np.bincount(p.coords[0]*sp.ns+p.coords[1],minlength=sp.na*sp.ns)
            if not np.isfinite(p.data).all() or (p.data<0).any() or not np.allclose(base.dense(p.sum(axis=2)),1):raise ValueError('Invalid local kernel')
            fan*=int(count.max())
        else:
            p=base.dense(p);base.distribution(p,2,sp.name);fan*=int(np.count_nonzero(p,axis=2).max())
    bound=spaces[0].na*tm.n*fan
    result=dict(spaces=[dict(name=s.name,ns=s.ns,na=s.na) for s in spaces],affordances=reports,state=sd,
        action=a,alphas=alphas,local_next=local,joint_states=tm.n,nnz_upper_bound=bound,
        can_build=tm.n<=base.MAX_STATES and bound<=base.MAX_NNZ,limits=dict(states=base.MAX_STATES,nnz=base.MAX_NNZ),
        semantics='Compact current-state spatial H: shared push event selects box direction and agent move/hold. No box-configuration H tables.',
        x_dynamics='identity' if base.x_identity(affs,sd,a) else 'normal')
    if affs.transition:
        conditional=[]
        for xn,p,children in tm.branches(sd,a):
            conditional.append(dict(X_next=xn,probability=p,alphas={sp.name:alpha for sp,(_,alpha,_) in zip(spaces[1:],children)}))
        result['conditional_next']=conditional
    if build:
        if not result['can_build']:raise ValueError('Product exceeds tutorial state/nonzero limits; compact factorized play is still available')
        kernels,verification=tm.build();flat=int(np.ravel_multi_index(tuple(sd[n] for n in tm.names),tm.sizes));row=kernels[a].getrow(flat)
        result['product']=dict(shape=[len(kernels),tm.n,tm.n],nnz=sum(k.nnz for k in kernels),verification=verification,
            retained_for_play=retain is not None,max_row_sum_error=verification['max_probability_error'],selected_row_error=0.,
            flat_state=flat,row_nnz=row.nnz,next=[dict(state=tm.state(int(i)),flat=int(i),probability=float(p)) for i,p in zip(row.indices[:200],row.data[:200])],truncated=row.nnz>200)
        if retain is not None:retain.update(spaces=spaces,affs=affs,csr=kernels)
    return result
