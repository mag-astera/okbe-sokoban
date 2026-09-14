import numpy as np
import pytest
from functools import partial
from eta_fast import build_eta
from feas_eta_fast import feas_iter_fast_eta_thresh
from feas_two_phase_sparse import feas_iter_two_phase_sparse
from feas_thresh_sparse import feas_iter_two_phase_sparse_thresh
from test_feas_sparse import build, WORLDS


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('rd', [None, 3])
@pytest.mark.parametrize('case', sorted(WORLDS))
def test_eta_matches_reference(mode, rd, case):
    P,g,c = build(**WORLDS[case])
    ref = feas_iter_two_phase_sparse(P,g,c,round_decimals=rd)
    new = feas_iter_two_phase_sparse(P,g,c,round_decimals=rd,compute_eta=partial(build_eta,mode=mode))
    for idx in [0,1,4]:
        np.testing.assert_array_equal(ref[idx],new[idx])
    for old, actual in zip(ref[2:4],new[2:4]):
        np.testing.assert_allclose(actual.to_dense(),old.to_dense(),atol=1e-12,rtol=1e-12)
        dense=actual.to_dense()
        for axis in [None,2,(1,2),(0,1)]:
            np.testing.assert_allclose(actual.sum(axis),dense.sum(axis=axis),atol=1e-12)
        np.testing.assert_allclose(actual[10],dense[10],atol=0)
        np.testing.assert_allclose(actual.to_sparse_stok().to_dense(),dense,atol=0)


@pytest.mark.parametrize('mode',[0,1,2])
@pytest.mark.parametrize('theta',[0,.2,.6,1])
@pytest.mark.parametrize('rd',[None,3])
def test_threshold(mode,theta,rd):
    P,g,c=build(p_main=.7, walls=(20,21,22), constraints=(30,40))
    ref=feas_iter_two_phase_sparse_thresh(P,g,c,risk_thresh=theta,round_decimals=rd)
    new=feas_iter_fast_eta_thresh(P,g,c,risk_thresh=theta,round_decimals=rd,eta_mode=mode)
    for idx in [0,1,4]:
        np.testing.assert_array_equal(ref[idx],new[idx])
    for old,actual in zip(ref[2:4],new[2:4]):
        np.testing.assert_allclose(actual.to_dense(),old.to_dense(),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('mode',[0,1,2])
@pytest.mark.parametrize('T',[1,23])
@pytest.mark.parametrize('deterministic',[False,True])
def test_against_matrix_powers(mode,T,deterministic):
    from scipy import sparse
    rng=np.random.default_rng(83)
    n=7
    P=np.eye(n)[np.roll(np.arange(n),1)] if deterministic else rng.random((n,n))
    P=P/P.sum(axis=1,keepdims=True)
    seeds=(np.array([.3,0,0,.7,0,0,0]),np.array([0,0,.4,0,0,.8,0]))
    gates=(np.array([.5,1,0,.8,1,0,1]),np.array([.5,1,0,0,1,0,1]))
    actual=build_eta(sparse.csr_matrix(P),seeds,gates,T,T,1e-12,mode)
    if mode == 1 or (mode == 0 and deterministic):
        assert actual[0].storage is actual[1].storage
        for item in actual:
            if item.matrix.nnz:
                assert np.shares_memory(item.matrix.data, item.storage.data)
    for out,seed,gate in zip(actual,seeds,gates):
        Q=np.diag(gate)@P
        expected=np.stack([np.linalg.matrix_power(Q,t)@np.diag(seed) for t in range(T)],axis=2)
        np.testing.assert_allclose(out.to_dense(),expected,rtol=1e-12,atol=1e-14)


@pytest.mark.parametrize('algo',['feas_eta_fast','feas_eta_fast_thresh'])
@pytest.mark.parametrize('mode',[0,1,2])
def test_tutorial_end_to_end(algo,mode):
    import tutorial_server as ts
    world=ts.normalize_world({'rows':4,'cols':4,'walls':[5],'constraints':[7],
        'goals':[{'id':1,'cells':[{'li':15,'a':None}]}],'start':0,'p_main':.85})
    params={'round_decimals':3,'eta_mode':mode}
    out=ts.solve_world(world,algo,params,want_stok=True)
    assert out['timing']['stok']>0
    for eta in ts.get_eta_sparse(world,algo,params,1):
        assert eta.nx==16
    ref=ts.solve_world(world,'feas2s' if algo=='feas_eta_fast' else 'feas_thresh_s',params,want_stok=True)
    for key in ['pi','kappa','stok','stok_neg','kappa_viol','kappa_none']:
        np.testing.assert_allclose(out['surfaces'][0][key],ref['surfaces'][0][key],atol=1e-5)
