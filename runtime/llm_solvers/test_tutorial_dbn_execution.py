"""Regression: Test uses the whole composed DBN, including stale placement drafts."""
import sys
from pathlib import Path
from copy import deepcopy
import itertools
import numpy as np
import pytest
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
import tutorial_server as server
import dbn_composer as dbn


def world():
    return server.normalize_world(dict(rows=2,cols=3,nis=4,p_main=1.,start=1,
        internal_goalstates_2_type_ind={'1':0},internal_state_values={0:2},
        doors=[dict(id=1,cell=5,color='purple',open=False)],
        buttons=[dict(id=1,cell=2,color='purple')]))


def test_stale_draft_updates_tree_location_and_adds_missing_factors():
    w=world()
    stale={'version':1,'spaces':['X','hunger'],'affordances':[dict(target='hunger',default_alpha=0,merge='reject',
        rules=[{'when':{'and':[{'var':'action','in':[0]},{'var':'X','in':[0]}]},'alpha':1}])], 'features':{}}
    result=server.dbn_panel_data(w,dict(operation='step',mode='factorized',spec=stale,state={'X':0,'hunger':0},use_world_state=True,action=0))
    assert result['before']=={'X':1,'hunger':2,'door_1':0}
    assert result['state']['hunger']==3
    assert result['alphas']['hunger']==1
    assert result['spec']['spaces']==['X','hunger','door_1']
    w['internal_goalstates_2_type_ind']={3:0};w['start']=3
    moved=server.dbn_panel_data(w,dict(operation='step',mode='factorized',spec=result['spec'],state=result['state'],use_world_state=True,action=0))
    assert moved['state']['hunger']==3 and moved['alphas']['hunger']==1


def test_custom_tree_rule_survives_placement_synchronization():
    w=world();spec=server.dbn_panel_data(w,dict(operation='catalog'))['spec']
    aff=next(a for a in spec['affordances'] if a['target']=='hunger')
    aff.update(placement_managed=False,rules=[])
    result=server.dbn_panel_data(w,dict(operation='step',mode='factorized',spec=spec,use_world_state=True,action=0))
    assert result['state']['hunger']==1 and result['alphas']['hunger']==0


def test_every_joint_row_tree_action_button_state_and_movement_modes():
    w=world();spec=server.dbn_panel_data(w,dict(operation='catalog'))['spec']
    available=server.build_inspect_spaces(w);model={}
    report=dbn.evaluate(available,spec,build=True,retain=model)
    assert any(a['target']=='X mode: door_1_passage' for a in report['affordances'])
    assert next(a for a in report['affordances'] if a['target']=='door_1')['parents']==['X']
    for action,x,hunger,door in itertools.product(range(6),range(6),range(4),range(2)):
        xrow=np.array(available['X'].P_a_s_s[action,x],copy=True)
        if not door and x!=5:xrow[x]+=xrow[5];xrow[5]=0
        hrow=np.asarray(available['hunger'].P_a_s_s[1 if action==0 and x==1 else 0,hunger])
        drow=np.eye(2)[1-door if x==2 else door]
        expected=np.kron(np.kron(xrow,hrow),drow)
        np.testing.assert_allclose(model['csr'][action].getrow((x*4+hunger)*2+door).toarray()[0],expected,atol=1e-14)


@pytest.mark.parametrize('mode',['factorized','tensor'])
def test_button_is_current_state_conditioned_not_action_conditioned(mode):
    w=world();spec=server.dbn_panel_data(w,dict(operation='catalog'))['spec']
    server.dbn_panel_data(w,dict(operation='build',spec=spec))
    for action in range(6):
        result=server.dbn_panel_data(w,dict(operation='step',mode=mode,spec=spec,state={'X':2,'hunger':2},action=action))
        assert result['alphas']['door_1']==3 and result['state']['door_1']==1
    # Arriving on the button does not secretly run a second transition.
    r=server.dbn_panel_data(w,dict(operation='step',mode=mode,spec=spec,state={'X':1,'hunger':2},action=3))
    assert r['state']['X']==2 and r['state']['door_1']==0
    r=server.dbn_panel_data(w,dict(operation='step',mode=mode,spec=spec,state=r['state'],action=1))
    assert r['state']['door_1']==1
