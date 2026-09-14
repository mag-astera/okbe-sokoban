"""Reproducible eta benchmark; run from main_files with tutorial/.venv/bin/python."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from functools import partial
from time import perf_counter
import json
import tutorial_server as TS
from feas_two_phase_sparse import feas_iter_two_phase_sparse
from feas_thresh_sparse import feas_iter_two_phase_sparse_thresh
from eta_fast import build_eta
from feas_eta_fast import feas_iter_fast_eta_thresh


def run(size=40, repeats=3):
    results=[]
    for label, noise, wall_rate, risk in [('deterministic',1,.10,None),('noisy',.85,.10,None),('open noisy',.85,0,None),('risk',.85,.10,.3)]:
        rng=np.random.default_rng(24)
        n=size*size
        walls=np.flatnonzero(rng.random(n)<wall_rate)
        walls=walls[(walls!=0)&(walls!=n-1)]
        constraints=np.setdiff1d(np.flatnonzero(rng.random(n)<.04),np.r_[walls,0,n-1])
        _,P=TS.get_kernel(size,size,walls.tolist(),noise)
        g=np.zeros((6,n));g[:,n-1]=1
        c=np.ones((6,n));c[:,constraints]=0
        solver=feas_iter_two_phase_sparse if risk is None else feas_iter_two_phase_sparse_thresh
        kw={} if risk is None else {'risk_thresh':risk}
        for mode in [-1,1,2]:
            times=[]
            for _ in range(repeats):
                timing={}
                if risk is not None and mode != -1:
                    result=feas_iter_fast_eta_thresh(P,g,c,timing=timing,eta_mode=mode,**kw)
                else:
                    result=solver(P,g,c,timing=timing,**kw,**({} if mode==-1 else {'compute_eta':partial(build_eta,mode=mode)}))
                times.append(timing['stok'])
            record=dict(world=label,size=size,method={-1:'original',1:'wide',2:'compact'}[mode],eta_seconds=float(np.median(times)),eta_bytes=sum(x.nbytes() for x in result[2:4]))
            print(json.dumps(record),flush=True);results.append(record)
    return results

if __name__=='__main__':
    run(int(sys.argv[1]) if len(sys.argv)>1 else 40, int(sys.argv[2]) if len(sys.argv)>2 else 3)
