"""Opt-in transition H implementation, separate from the current-state composer.

Hψ evaluates predicates on (state, action, realized X'); Hα chooses one alpha.
Only X' -> non-root action edges are supported. No dense X×X×A H is stored.
"""
import itertools
import math
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
import dbn_composer as base


def rewrite(expr):
    if isinstance(expr, dict):
        if set(expr)=={'transition'}:
            if expr['transition']!='moved':
                raise ValueError('Unknown transition feature; supported: moved')
            return {'var':'__moved','in':[1]}
        return {k:rewrite(v) for k,v in expr.items()}
    if isinstance(expr,list):
        return [rewrite(v) for v in expr]
    return expr


class TransitionAffordance:
    def __init__(self, raw, dims, features, na):
        if 'H_psi' in raw or 'H_alpha' in raw or 'parents' in raw or 'has_action' in raw:
            raise ValueError('Transition H uses predicate rules; parent axes are inferred')
        self.target=raw['target'];self.mode=raw.get('merge','reject')
        if self.mode not in ('reject','priority'):raise ValueError('merge must be reject or priority')
        self.default=base.integer(raw.get('default_alpha',0),self.target,na)
        if 'X_next' in dims or '__moved' in dims:raise ValueError('X_next and __moved are reserved in transition H')
        if (len(raw.get('rules',[]))+1)*na>base.MAX_H_ENTRIES:raise ValueError('Transition H action-map budget exceeded')
        tdims={**dims,'X_next':dims['X'],'__moved':2}
        self.rules=[(base.gate(rewrite(r['when']),tdims,features),base.integer(r['alpha'],self.target,na)) for r in raw.get('rules',[])]
        variables=set().union(*(g.vars() for g,_ in self.rules)) if self.rules else set()
        self.axes=sorted(variables)
        # Validate logical feature assignments, without storing an H table.
        count=math.prod(tdims[n] for n in self.axes)
        if count>base.MAX_H_ENTRIES:raise ValueError('Transition H conflict-check budget exceeded; simplify its parent conditions')
        collisions=0;witness=None
        for values in itertools.product(*(range(tdims[n]) for n in self.axes)):
            sd=dict(zip(self.axes,values))
            if {'X','X_next','__moved'}<=variables and sd['__moved']!=int(sd['X']!=sd['X_next']):continue
            active=[(i,a) for i,(g,a) in enumerate(self.rules) if g.eval(sd,sd.get('action'))]
            if len({a for _,a in active})>1:
                collisions+=1;witness=witness or sd
        if collisions and self.mode=='reject':raise ValueError(f'{self.target}: conflicting transition actions; example {witness}')
        ha=np.eye(na)[:,[a for _,a in self.rules]+[self.default]]
        self.report=dict(target=self.target,parents=[n for n in self.axes if n not in ('action','__moved')],
            axes=self.axes,merge=self.mode,conflicting_assignments=collisions,witness=witness,h_form='transition',
            H_psi=dict(shape=[len(self.rules)+1,'predicate'],values=None,representation='predicate',rules=raw.get('rules',[]),
                       moved='X_next != X',selection='first matching event, otherwise default'),
            H_alpha=dict(shape=list(ha.shape),values=ha.tolist()))

    def select(self,state,action,x_next):
        sd={**state,'action':action,'X_next':x_next,'__moved':int(x_next!=state['X'])}
        active=[(i,a) for i,(g,a) in enumerate(self.rules) if g.eval(sd,action)]
        if self.mode=='reject' and len({a for _,a in active})>1:raise ValueError(f'{self.target}: conflicting transition actions')
        event,alpha=active[0] if active else (len(self.rules),self.default)
        return int(alpha),dict(event=event,moved=bool(sd['__moved']),X=state['X'],X_next=x_next)


class TransitionModel:
    def __init__(self,spaces,affs):
        self.spaces,self.affs=spaces,affs
        self.sizes=[s.ns for s in spaces];self.names=[s.name for s in spaces];self.n=math.prod(self.sizes)

    def state(self,flat):return dict(zip(self.names,map(int,np.unravel_index(flat,self.sizes))))

    def child(self,sp,sd,a,xn):
        if sp.name in self.affs.transition:
            alpha,psi=self.affs.transition[sp.name].select(sd,a,xn)
        else:
            f=self.affs.by_driven.get(sp.name)
            alpha=int(f.induced_alpha_at(sd,a) if f else getattr(sp,'default_action_ind',0));psi=None
        return base.local_row(sp,self.affs,sd,a,alpha),alpha,psi

    def branches(self,sd,a):
        px=base.local_row(self.spaces[0],self.affs,sd,a,a)
        for xn in np.flatnonzero(px):
            xn=int(xn)
            yield xn,float(px[xn]),[self.child(sp,sd,a,xn) for sp in self.spaces[1:]]

    def row(self,sd,a,limit=base.MAX_NNZ):
        entries={}
        for xn,px,children in self.branches(sd,a):
            support=[np.flatnonzero(row) for row,_,_ in children]
            if math.prod(len(s) for s in support)+len(entries)>limit:raise ValueError('Transition product exceeds nonzero limit')
            for tail in itertools.product(*support):
                prob=px*math.prod(float(children[i][0][s]) for i,s in enumerate(tail))
                flat=int(np.ravel_multi_index((xn,)+tail,self.sizes))
                entries[flat]=entries.get(flat,0.)+prob
        return entries

    def build(self):
        if self.n>base.MAX_STATES:raise ValueError('Product exceeds tutorial state limit')
        kernels=[];total=0;error=0.;checked=0
        for a in range(self.spaces[0].na):
            indices=[];data=[];indptr=[0]
            for flat in range(self.n):
                sd=self.state(flat);entries=self.row(sd,a,base.MAX_NNZ-total)
                error=max(error,abs(sum(entries.values())-1))
                for dest,p in sorted(entries.items()):indices.append(dest);data.append(p)
                total+=len(entries);indptr.append(len(indices))
            kernel=csr_matrix((data,indices,indptr),shape=(self.n,self.n))
            # Independently compare stored entries and support to conditional local rows.
            for flat in range(self.n):
                sd=self.state(flat);row=kernel.getrow(flat);expected_count=0
                for xn,px,children in self.branches(sd,a):
                    expected_count+=math.prod(np.count_nonzero(p) for p,_,_ in children)
                if row.nnz!=expected_count:raise ValueError('Transition product support mismatch')
                for dest,p in zip(row.indices,row.data):
                    nxt=self.state(int(dest));xn=nxt['X']
                    expected=float(base.local_row(self.spaces[0],self.affs,sd,a,a)[xn])
                    for sp in self.spaces[1:]:expected*=float(self.child(sp,sd,a,xn)[0][nxt[sp.name]])
                    error=max(error,abs(expected-p));checked+=1
            kernels.append(kernel)
        if error>1e-12:raise ValueError(f'Transition product verification error: {error}')
        return kernels,dict(rows_checked=self.n*self.spaces[0].na,entries_checked=checked,support_matches=True,max_probability_error=error,tolerance=1e-12)


def step_model(model,state,action,mode,rng=None):
    rng=rng or np.random.default_rng();spaces,affs=model['spaces'],model['affs'];tm=TransitionModel(spaces,affs)
    a=base.integer(action,'action',spaces[0].na)
    sd={s.name:base.integer(state.get(s.name,0),s.name,s.ns) for s in spaces}
    if mode=='tensor':
        if 'csr' not in model:raise ValueError('Build sparse product before using full-tensor controls')
        flat=np.ravel_multi_index(tuple(sd[n] for n in tm.names),tm.sizes);row=model['csr'][a].getrow(flat)
        nxt=tm.state(int(rng.choice(row.indices,p=row.data)));xn=nxt['X']
    elif mode=='factorized':
        px=base.local_row(spaces[0],affs,sd,a,a);xn=int(rng.choice(spaces[0].ns,p=px));nxt={'X':xn}
    else:raise ValueError('Execution mode must be factorized or tensor')
    alphas={'X':a};trace={}
    for sp in spaces[1:]:
        row,alpha,psi=tm.child(sp,sd,a,xn);alphas[sp.name]=alpha
        if psi is not None:trace[sp.name]=psi
        if mode=='factorized':nxt[sp.name]=int(rng.choice(sp.ns,p=row))
    return dict(before=sd,state=nxt,alphas=alphas,action=a,mode=mode,transition_features=trace,
        environment_alphas={d['name']:int(f.induced_alpha_at(sd,a)) for d,f in zip(affs.root_kernel.effects,affs.kernel_modes)},
        x_dynamics='identity' if base.x_identity(affs,sd,a) else 'normal')


def evaluate(available,spec,spaces,affs,reports,state,action,build,retain):
    # Reuse original validation and size estimates, without pretending transition
    # targets have a single pre-transition alpha or independent next-state row.
    plain=deepcopy(spec);plain['affordances']=[a for a in plain.get('affordances',[]) if a.get('h_form')!='transition']
    result=base.evaluate(available,plain,state,action,False)
    tm=TransitionModel(spaces,affs);sd=result['state'];a=result['action']
    result['affordances']=reports
    result['semantics']='Sample X_next, evaluate transition H, then sample child factors conditional on that same realized X_next.'
    conditional=[];marginals={s.name:np.zeros(s.ns) for s in spaces[1:]}
    for xn,p,children in tm.branches(sd,a):
        alphas={};features={};local={}
        for sp,(row,alpha,psi) in zip(spaces[1:],children):
            alphas[sp.name]=alpha;features[sp.name]=psi
            local[sp.name]=[[int(i),float(row[i])] for i in np.flatnonzero(row)]
            marginals[sp.name]+=p*row
        conditional.append(dict(X_next=xn,probability=p,alphas=alphas,features=features,local_next=local))
    result['conditional_next']=conditional
    for name,h in affs.transition.items():
        values={b['alphas'][name] for b in conditional}
        result['alphas'][name]=next(iter(values)) if len(values)==1 else None
    for name,row in marginals.items():result['local_next'][name]=[[int(i),float(row[i])] for i in np.flatnonzero(row)]
    result['local_next_kind']='Marginals only; conditional_next preserves dependence on realized X_next.'
    if build:
        if not result['can_build']:
            raise ValueError(f"Product exceeds tutorial limits: {result['joint_states']:,} states, "
                             f"at most {result['nnz_upper_bound']:,} nonzeros; limits are "
                             f"{result['limits']['states']:,} states and {result['limits']['nnz']:,} nonzeros")
        kernels,verification=tm.build();flat=int(np.ravel_multi_index(tuple(sd[n] for n in tm.names),tm.sizes));row=kernels[a].getrow(flat)
        result['product']=dict(shape=[len(kernels),tm.n,tm.n],nnz=sum(k.nnz for k in kernels),verification=verification,
            retained_for_play=retain is not None,max_row_sum_error=verification['max_probability_error'],selected_row_error=0.,
            flat_state=flat,row_nnz=row.nnz,next=[dict(state=tm.state(int(i)),flat=int(i),probability=float(p)) for i,p in zip(row.indices[:200],row.data[:200])],truncated=row.nnz>200)
        if retain is not None:retain.update(spaces=spaces,affs=affs,csr=kernels)
    return result
