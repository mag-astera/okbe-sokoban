import sys
from pathlib import Path
from copy import deepcopy
from itertools import product
import numpy as np
import pytest
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent.parent)]
import tutorial_server as server
import dbn_composer as dbn


def world():
    return server.normalize_world(dict(rows=2,cols=3,nis=3,p_main=.7,start=1,
        internal_goalstates_2_type_ind={'0':0},
        doors=[dict(id=1,cell=2,color='purple',open=False),dict(id=2,cell=5,color='purple',open=True)],
        buttons=[dict(id=1,cell=1,color='purple',transition_h=True)]))


def test_tutorial_joint_every_entry_and_same_realized_movement():
    w=world();spec=server.dbn_panel_data(w,{'operation':'catalog'})['spec']
    report=server.dbn_panel_data(w,dict(operation='build',spec=spec))
    assert report['product']['verification']['rows_checked']==6*6*3*4
    assert 'conditional_next' in report
    model=server._DBN_MODEL_CACHE['model'];spaces={s.name:s for s in model['spaces']}
    sizes=[s.ns for s in model['spaces']];names=[s.name for s in model['spaces']]
    for a,flat in product(range(6),range(np.prod(sizes))):
        sd=dict(zip(names,np.unravel_index(flat,sizes)))
        px=spaces['X'].P_a_s_s[a,sd['X']].copy()
        for name,cell in [('door_1',2),('door_2',5)]:
            if not sd[name] and sd['X']!=cell:px[sd['X']]+=px[cell];px[cell]=0
        expected=np.zeros(np.prod(sizes))
        for xn in np.flatnonzero(px):
            moved=sd['X']==1 and xn!=sd['X']
            hp=spaces['hunger'].P_a_s_s[1 if sd['X']==0 and a==0 else 0,sd['hunger']]
            for hn in np.flatnonzero(hp):
                ns={'X':xn,'hunger':hn,'door_1':1-sd['door_1'] if moved else sd['door_1'],'door_2':1-sd['door_2'] if moved else sd['door_2']}
                expected[np.ravel_multi_index(tuple(ns[n] for n in names),sizes)]+=px[xn]*hp[hn]
        np.testing.assert_allclose(model['csr'][a].getrow(flat).toarray()[0],expected,rtol=0,atol=1e-14)
    for mode in ['factorized','tensor']:
        rng=np.random.default_rng(15)
        for _ in range(150):
            r=dbn.step_model(model,{'X':1,'hunger':2,'door_1':0,'door_2':1},5,mode,rng)
            moved=r['state']['X']!=1
            assert (r['state']['door_1'],r['state']['door_2'])==((1,0) if moved else (0,1))
            assert r['transition_features']['door_1']['moved']==moved
        for a in [0,1]:
            r=dbn.step_model(model,{'X':1,'hunger':2,'door_1':0,'door_2':1},a,mode)
            assert r['state']['door_1']==0 and r['state']['door_2']==1


def test_default_dispatch_and_toggle_invalidates_old_tensor():
    w=world();w['buttons'][0]['transition_h']=False
    spec=server.dbn_panel_data(w,{'operation':'catalog'})['spec']
    server.dbn_panel_data(w,dict(operation='build',spec=spec))
    assert not server._DBN_MODEL_CACHE['model']['affs'].transition
    w['buttons'][0]['transition_h']=True
    with pytest.raises(ValueError,match='Build sparse product'):
        server.dbn_panel_data(w,dict(operation='step',mode='tensor',spec=spec,action=0,state={'X':1}))
    r=server.dbn_panel_data(w,dict(operation='step',mode='factorized',spec=spec,action=0,state={'X':1}))
    assert r['alphas']['door_1']==0
    assert server.normalize_world(w)['buttons'][0]['transition_h'] is True


def test_custom_transition_rule_conflict_and_action_condition():
    w=world();spec=server.dbn_panel_data(w,{'operation':'catalog'})['spec']
    aff=next(a for a in spec['affordances'] if a['target']=='door_1');aff['placement_managed']=False
    aff['rules']=[dict(when={'and':[{'transition':'moved'},{'var':'action','in':[5]}]},alpha=1)]
    spaces=server.build_inspect_spaces(w)
    ordered,compiled,_=dbn.compile_model(spaces,spec)
    assert compiled.transition['door_1'].select({'X':1},5,0)[0]==1
    assert compiled.transition['door_1'].select({'X':1},3,0)[0]==0
    aff['rules'].append(dict(when={'transition':'moved'},alpha=2))
    with pytest.raises(ValueError,match='conflicting transition'):dbn.compile_model(spaces,spec)
    aff['merge']='priority';dbn.compile_model(spaces,spec)
    aff['h_form']='current'
    with pytest.raises(ValueError):dbn.compile_model(spaces,spec)


def test_product_solver_uses_transition_kernel():
    w=server.normalize_world(dict(rows=2,cols=3,p_main=1.,start=1,walls=[3,4,5],
        goals=[{'id':1,'cells':[{'li':2,'a':None}]}],
        doors=[dict(id=1,cell=2,color='purple',open=False)],
        buttons=[dict(id=1,cell=1,color='purple',transition_h=True)]))
    spec=server.dbn_panel_data(w,{'operation':'catalog'})['spec']
    result=server.dbn_panel_data(w,dict(operation='solve',spec=spec,algorithm='ssp',options={'group':1}))
    assert result['selected_policy']==5  # depart left, return, then cross the open door
    assert result['value']==-3
