"""Interleaved warm benchmarks of original, separate, and combined STOK solvers.
Run from main_files: tutorial/.venv/bin/python llm_solvers/benchmark_combined_eta.py
"""
import gc
import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import tutorial_server as TS
from feas_two_phase_sparse import feas_iter_two_phase_sparse
from feas_thresh_sparse import feas_iter_two_phase_sparse_thresh
from feas_eta_fast import feas_iter_fast_eta,feas_iter_fast_eta_thresh
from eta_combined import feas_iter_combined_eta,feas_iter_combined_eta_thresh


def run(size=40,repeats=7):
    records=[]
    for label,noise,wall_rate,risk in [('deterministic',1,.10,None),('noisy',.85,.10,None),('open noisy',.85,0,None),('risk',.85,.10,.3)]:
        rng=np.random.default_rng(24)
        n=size*size
        walls=np.flatnonzero(rng.random(n)<wall_rate)
        walls=walls[(walls!=0)&(walls!=n-1)]
        constraints=np.setdiff1d(np.flatnonzero(rng.random(n)<.04),np.r_[walls,0,n-1])
        _,P=TS.get_kernel(size,size,walls.tolist(),noise)
        g=np.zeros((6,n));g[:,n-1]=1
        c=np.ones((6,n));c[:,constraints]=0
        original=feas_iter_two_phase_sparse if risk is None else feas_iter_two_phase_sparse_thresh
        separate=feas_iter_fast_eta if risk is None else feas_iter_fast_eta_thresh
        combined=feas_iter_combined_eta if risk is None else feas_iter_combined_eta_thresh
        methods=[('original',original,{}),('separate-auto',separate,{'eta_mode':0}),
                 ('combined-auto',combined,{'eta_mode':0}),('combined-wide',combined,{'eta_mode':1}),
                 ('combined-compact',combined,{'eta_mode':2})]
        kw={'round_decimals':3}
        if risk is not None:
            kw['risk_thresh']=risk
        samples={name:[] for name,_,_ in methods}
        baseline=None
        for repeat in range(repeats+1):
            order=np.arange(len(methods)) if repeat==0 else rng.permutation(len(methods))
            for index in order:
                name,solver,extra=methods[index]
                gc.collect()
                timing={}
                started=perf_counter()
                result=solver(P,g,c,timing=timing,**kw,**extra)
                total=perf_counter()-started
                kappa,pi,ep,en,beta=result
                started=perf_counter()
                pos=ep.sum((1,2));neg=en.sum((1,2))
                reduction=perf_counter()-started
                if baseline is None:
                    baseline=(kappa.copy(),pi.copy(),pos.copy(),neg.copy())
                assert np.array_equal(pi,baseline[1]), (label,name,'policy changed')
                assert np.array_equal(kappa,baseline[0]), (label,name,'kappa changed')
                if hasattr(ep,'combined'):
                    memory=ep.combined.nbytes()+ep.weights.nbytes+en.weights.nbytes
                else:
                    memory=ep.nbytes()+en.nbytes()
                sample={'eta_ms':timing['stok']*1000,'policy_ms':timing['policy']*1000,
                        'solver_ms':total*1000,'reduction_ms':reduction*1000,'eta_bytes':int(memory),
                        'positive_mass_max_difference':float(np.max(abs(pos-baseline[2]))),
                        'negative_mass_max_difference':float(np.max(abs(neg-baseline[3])))}
                if repeat:
                    samples[name].append(sample)
                del result,ep,en
        for name,_,_ in methods:
            rows=samples[name]
            record={'world':label,'size':size,'method':name,'repeats':repeats,
                    'median':{key:float(np.median([r[key] for r in rows])) for key in rows[0]},
                    'eta_ms_min':min(r['eta_ms'] for r in rows),'eta_ms_max':max(r['eta_ms'] for r in rows),
                    'samples':rows}
            print(json.dumps(record),flush=True)
            records.append(record)
    return records


if __name__=='__main__':
    run(int(sys.argv[1]) if len(sys.argv)>1 else 40,int(sys.argv[2]) if len(sys.argv)>2 else 7)
