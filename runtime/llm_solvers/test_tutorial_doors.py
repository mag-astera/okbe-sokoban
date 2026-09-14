import sys
from pathlib import Path
import itertools
import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
import tutorial_server as server
import tutorial_doors as doors
import dbn_composer as dbn


def fixture(noise=1.):
    return server.normalize_world(dict(rows=2, cols=3, start=0, p_main=noise,
        goals=[{'id':1,'cells':[{'li':2,'a':None}]}],
        doors=[dict(id=1,cell=1,color='purple',open=False),dict(id=2,cell=2,color='purple',open=True)],
        buttons=[dict(id=1,cell=0,color='purple')]))


@pytest.mark.parametrize('noise', [1., .7])
@pytest.mark.parametrize('identity', [False, True])
def test_all_rows_against_independent_door_formula(noise, identity):
    world=fixture(noise)
    catalog=server.dbn_panel_data(world, {'operation':'catalog'})
    spec=catalog['spec']
    if identity:
        spec['x_identity_when']={'var':'door_2','in':[0]}
    retained={}
    dbn.evaluate(server.build_inspect_spaces(world),spec,build=True,retain=retained)
    _,base=server.get_world_kernel(world,product=True)
    for a,x,b,c in itertools.product(range(6),range(6),range(2),range(2)):
        row=base[a,x].copy()
        if identity and c==0:
            row[:]=0;row[x]=1
        else:
            for cell,bit in [(1,b),(2,c)]:
                if bit==0 and cell!=x:
                    row[x]+=row[cell];row[cell]=0
        nb,nc=(1-b,1-c) if x==0 else (b,c)
        expected=np.zeros(24)
        expected[np.arange(6)*4+nb*2+nc]=row
        np.testing.assert_allclose(retained['csr'][a].getrow(x*4+b*2+c).toarray()[0],expected,atol=1e-14)


def test_opposite_doors_toggle_twice_and_tensor_cache_survives_state_change():
    world=fixture()
    catalog=server.dbn_panel_data(world,{'operation':'catalog'})
    spec=catalog['spec']
    server.dbn_panel_data(world,{'operation':'build','spec':spec})
    for mode in ['factorized','tensor']:
        w=dict(world)
        for expected in [(1,0),(0,1)]:
            result=server.dbn_panel_data(w,dict(operation='step',mode=mode,action=0,spec=spec,state={'X':0}))
            assert tuple(result['state'][n] for n in ['door_1','door_2'])==expected
            assert result['alphas']['door_1']==result['alphas']['door_2']==3
            w.update(result['world_state'])


def test_explicit_links_override_colors_and_support_multiple_buttons():
    w=fixture()
    w['buttons'][0]['targets']=[2]
    w['buttons'].append(dict(id=2,cell=3,color='red',targets=[1,2]))
    spec=server.dbn_panel_data(w,{'operation':'catalog'})['spec']
    for cell,expected in [(0,(0,0)),(3,(1,0)),(4,(0,1))]:
        result=server.dbn_panel_data(w,dict(operation='step',mode='factorized',action=0,spec=spec,state={'X':cell}))
        assert tuple(result['state'][n] for n in ['door_1','door_2'])==expected


def test_flat_snapshot_matches_conditioned_product_movement_and_no_mutation():
    w=fixture(.7)
    _,base=server.get_world_kernel(w,product=True)
    saved=base.copy()
    _,snapshot=server.get_world_kernel(w)
    for a,x in itertools.product(range(6),range(6)):
        expected=doors.block_row(base[a,x],x,[dict(d,name=doors.name(d)) for d in w['doors']],doors.current(w))
        np.testing.assert_allclose(snapshot[a,x],expected)
    np.testing.assert_array_equal(base,saved)
    # Closed door blocks entry, but an agent already inside can leave.
    assert snapshot[3,0,1]==0
    assert snapshot[2,1,4]>0


def test_serialization_and_stable_names_and_invalid_links():
    w=fixture()
    assert server.normalize_world(w)==w
    assert [s['name'] for s in doors.catalog(w)]==['door_1','door_2']
    w['doors'][0]['color']='red'; w['doors'][0]['cell']=4
    assert doors.catalog(w)[0]['name']=='door_1'
    w['buttons'][0]['targets']=[99]
    with pytest.raises(ValueError,match='does not exist'):
        server.normalize_world(w)


def test_product_solver_can_plan_to_press_a_button():
    w=server.normalize_world(dict(rows=2,cols=3,start=0,walls=[3,4,5],
        goals=[{'id':1,'cells':[{'li':2,'a':None}]}],
        doors=[dict(id=1,cell=1,color='purple',open=False)],
        buttons=[dict(id=1,cell=0,color='purple')]))
    spec=server.dbn_panel_data(w,{'operation':'catalog'})['spec']
    result=server.dbn_panel_data(w,dict(operation='solve',spec=spec,algorithm='ssp',options={'group':1}))
    assert result['fixed']['door_1']==0
    assert result['selected_policy'] in range(6)  # any action on the button opens the route
    assert result['value']==-3
    sliced=server.dbn_panel_data(w,dict(operation='slice',spec=spec,solution_id=result['solution_id'],fixed={'door_1':1},cell=0))
    assert sliced['selected_policy']==3  # open door: move right
    assert sliced['value']==-2

@pytest.mark.parametrize('alpha,expected', [(0,(0,1)),(1,(1,1)),(2,(0,0)),(3,(1,0))])
def test_button_effects_factorized_and_full_product(alpha, expected):
    world=fixture()
    world['buttons'][0].update(alpha=alpha, effects={})
    world=server.normalize_world(world)
    spec=server.dbn_panel_data(world, {'operation':'catalog'})['spec']
    server.dbn_panel_data(world, dict(operation='build',spec=spec))
    for mode in ('factorized','tensor'):
        result=server.dbn_panel_data(world,dict(operation='step',spec=spec,mode=mode,state={'X':0},action=0))
        assert tuple(result['state'][n] for n in ('door_1','door_2'))==expected


def test_per_door_effects_and_custom_H_override():
    world=fixture()
    world['buttons'][0].update(alpha=2,effects={'1':1})
    world=server.normalize_world(world)
    spec=server.dbn_panel_data(world, {'operation':'catalog'})['spec']
    result=server.dbn_panel_data(world,dict(operation='step',spec=spec,mode='factorized',state={'X':0},action=0))
    assert result['state']['door_1']==1 and result['state']['door_2']==0
    spec['features']={'open_other':{'var':'door_2','in':[1]}}
    aff=next(a for a in spec['affordances'] if a['target']=='door_1')
    aff.update(placement_managed=False,rules=[{'when':{'and':[{'feature':'open_other'},{'var':'action','in':[1]}]},'alpha':1}])
    updated=doors.augment_spec(spec,world)
    assert next(a for a in updated['affordances'] if a['target']=='door_1')==aff
    server.dbn_panel_data(world,dict(operation='build',spec=spec))
    for mode in ('factorized','tensor'):
        result=server.dbn_panel_data(world,dict(operation='step',spec=spec,mode=mode,state={'X':0},action=1))
        assert result['state']['door_1']==1
        result=server.dbn_panel_data(world,dict(operation='step',spec=spec,mode=mode,state={'X':0},action=0))
        assert result['state']['door_1']==0

@pytest.mark.parametrize('change',[{'alpha':4},{'alpha':True},{'effects':{'1':-1}},{'effects':[]}])
def test_invalid_button_effects_rejected(change):
    world=fixture();world['buttons'][0].update(change)
    with pytest.raises(ValueError):server.normalize_world(world)

@pytest.mark.parametrize('required',[None,[0],[1,3],[]])
def test_button_action_requirement_shared_by_both_execution_modes(required):
    world=fixture();world['buttons'][0]['required_actions']=required
    world=server.normalize_world(world)
    assert server.normalize_world(world)==world
    spec=server.dbn_panel_data(world,{'operation':'catalog'})['spec']
    report=server.dbn_panel_data(world,dict(operation='build',spec=spec))
    aff=next(a for a in report['affordances'] if a['target']=='door_1')
    assert ('action' in aff['axes'])==(required is not None)
    for mode,action,x in itertools.product(('factorized','tensor'),range(6),(0,4)):
        r=server.dbn_panel_data(world,dict(operation='step',mode=mode,spec=spec,state={'X':x},action=action))
        active=x==0 and (required is None or action in required)
        assert r['alphas']['door_1']==(3 if active else 0)
        assert (r['state']['door_1'],r['state']['door_2'])==((1,0) if active else (0,1))
    world['buttons'][0]['required_actions']=[5] if required!=[5] else None
    with pytest.raises(ValueError,match='Build sparse product'):
        server.dbn_panel_data(world,dict(operation='step',mode='tensor',spec=spec,state={'X':0},action=0))

@pytest.mark.parametrize('required',[0,True,'interact',[6],[-1],[True],[0.5]])
def test_bad_action_requirements_rejected(required):
    world=fixture();world['buttons'][0]['required_actions']=required
    with pytest.raises(ValueError,match='required_actions'):server.normalize_world(world)
