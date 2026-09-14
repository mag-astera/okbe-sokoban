import numpy as np
import pytest
from scipy import sparse
from eta_combined import build_combined_eta,feas_iter_combined_eta,feas_iter_combined_eta_thresh
from feas_two_phase_sparse import feas_iter_two_phase_sparse
from feas_thresh_sparse import feas_iter_two_phase_sparse_thresh
from test_feas_sparse import build


@pytest.mark.parametrize('mode',[0,1,2])
@pytest.mark.parametrize('deterministic',[False,True])
def test_one_recurrence_and_views(mode,deterministic):
    rng=np.random.default_rng(6)
    P=np.eye(5)[[1,2,3,4,4]] if deterministic else rng.random((5,5))
    P/=P.sum(axis=1,keepdims=True)
    # Overlapping soft outcomes test weighted labels, not just binary masks.
    positive=np.array([.2,0,.3,0,1.])
    negative=np.array([.1,.3,0,.5,0])
    gate=1-positive-negative
    ep,en=build_combined_eta(sparse.csr_matrix(P),(positive,negative),(gate,gate),40,40,1e-12,mode)
    assert ep.combined is en.combined
    E=ep.combined.to_dense()
    Q=np.diag(gate)@P
    np.testing.assert_allclose(E[:,:,0],np.diag(positive+negative))
    for t in range(1,40):
        np.testing.assert_allclose(E[:,:,t],Q@E[:,:,t-1],atol=1e-14)
    np.testing.assert_allclose(ep.to_dense()+en.to_dense(),E,atol=1e-14)
    for view,seed in [(ep,positive),(en,negative)]:
        expected=np.stack([np.linalg.matrix_power(Q,t)@np.diag(seed) for t in range(40)],axis=2)
        np.testing.assert_allclose(view.to_dense(),expected,atol=1e-14)
        for axis in [None,2,(1,2),(0,1)]:
            np.testing.assert_allclose(view.sum(axis),expected.sum(axis=axis),atol=1e-13)
        np.testing.assert_allclose(view[2],expected[2],atol=1e-14)
        np.testing.assert_allclose(view.to_sparse_stok().to_dense(),expected,atol=1e-14)


@pytest.mark.parametrize('mode',[0,1,2])
def test_infeasibility_stops_all_outcomes(mode):
    P=sparse.csr_matrix([[.99,.01],[0,1.]])
    ep,en=build_combined_eta(P,(np.array([0.,1]),np.array([1.,0])),
                            (np.array([1.,0]),np.array([0.,0])),20,20,1e-12,mode)
    np.testing.assert_array_equal(ep.combined.sum((1,2)),[1,1])
    assert ep[0].sum()==0
    assert en[0][0,0]==1


@pytest.mark.parametrize('mode',[0,1,2])
@pytest.mark.parametrize('threshold',[None,0,.5,1])
def test_fixed_horizon_matches_reference(mode,threshold):
    P,g,c=build(p_main=.8,walls=(30,31),constraints=(20,))
    kw={'max_time':80,'round_decimals':None}
    if threshold is None:
        old=feas_iter_two_phase_sparse(P,g,c,**kw)
        new=feas_iter_combined_eta(P,g,c,eta_mode=mode,**kw)
    else:
        old=feas_iter_two_phase_sparse_thresh(P,g,c,risk_thresh=threshold,**kw)
        new=feas_iter_combined_eta_thresh(P,g,c,risk_thresh=threshold,eta_mode=mode,**kw)
    for idx in [0,1,4]:
        np.testing.assert_array_equal(old[idx],new[idx])
    for a,b in zip(old[2:4],new[2:4]):
        np.testing.assert_allclose(a.to_dense(),b.to_dense(),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('algo',['feas_eta_combined','feas_eta_combined_thresh'])
@pytest.mark.parametrize('mode',[0,1,2])
def test_tutorial_integration(algo,mode):
    import tutorial_server as ts
    w=ts.normalize_world({'rows':4,'cols':4,'walls':[5],'constraints':[7],
       'goals':[{'id':1,'cells':[{'li':15,'a':None}]}],'start':0,'p_main':.85})
    params={'eta_mode':mode,'round_decimals':3}
    result=ts.solve_world(w,algo,params,want_stok=True)
    assert result['timing']['stok']>0
    ep,en=ts.get_eta(w,algo,params,1)
    assert ep.combined is en.combined
    for eta in ts.get_eta_sparse(w,algo,params,1):
        assert eta.nx==16
