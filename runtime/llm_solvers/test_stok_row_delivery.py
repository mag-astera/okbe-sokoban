"""Single-start delivery preserves full cached eta and avoids all-start payloads."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import tutorial_server as ts


def world():
    return ts.normalize_world({'rows':5,'cols':5,'walls':[6],'constraints':[8],
       'goals':[{'id':1,'cells':[{'li':24,'a':None}]},{'id':2,'cells':[{'li':0,'a':3}]}],
       'start':4,'p_main':.85})


@pytest.mark.parametrize('algo',['feas2s','feas_thresh_s','feas_eta_fast','feas_eta_combined','feas_eta_combined_thresh'])
def test_row_matches_full_and_moves_without_solving(algo,monkeypatch):
    w=world();params={'round_decimals':3}
    full=ts.solve_world(w,algo,params,True)
    row=ts.solve_world(w,algo,params,True,stok_mode='single_x0',x0=4)
    for a,b in zip(full['surfaces'],row['surfaces']):
        assert b['stok_cached'] and b['stok_mode']=='single_x0'
        assert b['x0']==4 and np.shape(b['stok'])==(25,)
        np.testing.assert_allclose(b['stok'],a['stok'][4],atol=1e-5)
        np.testing.assert_allclose(b['stok_neg'],a['stok_neg'][4],atol=1e-5)
    def no_solve(*args,**kwargs):
        raise AssertionError('changing the start must reuse cached eta')
    monkeypatch.setattr(ts,'solve_world',no_solve)
    for group in [1,2]:
        for x0 in [0,4,17,24]:
            r=ts.stok_time_slice(w,algo,params,group,x0)
            assert not r['resolved']
            ref=full['surfaces'][group-1]
            np.testing.assert_allclose(r['stok'],ref['stok'][x0],atol=1e-5)
            np.testing.assert_allclose(r['stok_neg'],ref['stok_neg'][x0],atol=1e-5)


def test_cache_miss_requests_no_full_matrices(monkeypatch):
    w=world();params={'round_decimals':4,'eta_mode':2}
    with ts._eta_lock:
        ts._ETA.clear()
    original=ts.solve_world
    calls=[]
    def checked(*args,**kwargs):
        calls.append(kwargs['stok_mode'])
        result=original(*args,**kwargs)
        assert all('stok' not in s for s in result['surfaces'])
        return result
    monkeypatch.setattr(ts,'solve_world',checked)
    r=ts.stok_time_slice(w,'feas_eta_combined',params,1,4)
    assert r['resolved'] and calls==['cache_only']


def test_default_start_and_validation():
    w=world()
    r=ts.solve_world(w,'feas_eta_combined',{},True,stok_mode='single_x0')
    assert r['surfaces'][0]['x0']==w['start']
    with pytest.raises(ValueError,match='x0'):
        ts.solve_world(w,'feas_eta_combined',{},True,stok_mode='single_x0',x0=25)


@pytest.mark.parametrize('algo',['feas2s','feas_eta_combined','feas_eta_combined_thresh'])
def test_packed_rows_preserve_every_start(algo):
    w=world();params={'round_decimals':3}
    full=ts.solve_world(w,algo,params,True)
    packed=ts.solve_world(w,algo,params,True,stok_mode='packed_rows')
    from scipy.sparse import csr_matrix
    for a,b in zip(full['surfaces'],packed['surfaces']):
        assert 'stok' not in b
        for name in ['stok','stok_neg']:
            p=b[name+'_rows']
            matrix=csr_matrix((p['data'],p['indices'],p['indptr']),shape=(25,25)).toarray()
            np.testing.assert_allclose(matrix,a[name],atol=1e-5)
