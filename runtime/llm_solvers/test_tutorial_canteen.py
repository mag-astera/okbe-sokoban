"""Canteen exchanges use the same H factors in manual play and the joint kernel."""
from test_tutorial_workshop import server, ta, d
import tutorial_workshop as workshop
import numpy as np


def canteen_world():
    return server.normalize_world(dict(rows=2,cols=3,start=0,p_main=1,nis=7,
        internal_goalstates_2_type_ind={2:1},internal_state_values={1:6},
        workshop={'objects':[dict(id=1,cell=0,kind='wood',quantity=1),
                             dict(id=2,cell=1,kind='canteen_bench',quantity=1)],'values':{}}))


def test_canteen_shared_events_and_tensor():
    server.warm_solver();w=canteen_world();spaces=server.build_inspect_spaces(w)
    spec=ta.prepare(w,{});assert ta.prepare(w,spec)==spec
    model={};result=d.evaluate(spaces,spec,build=True,retain=model)
    state=server.inspect_x0(w,spaces)
    for action in [0,3,0,3,0,7,5,6]:
        a=d.step_model(model,state,action,'factorized')['state']
        b=d.step_model(model,state,action,'tensor')['state'];assert a==b
        state=a
    assert state['canteen']==1 and state['canteen_water']==1 and state['hydration']==6
    assert state['wood']==state['resource_1']==0
    # A failed drink still takes a time step, but cannot refill or consume anything.
    for changes in [dict(canteen=0),dict(canteen_water=0)]:
        s=dict(state,hydration=4,**changes)
        out=d.step_model(model,s,6,'factorized')['state']
        assert out['hydration']==3 and out['canteen_water']==s['canteen_water']
    dead=d.step_model(model,dict(state,hydration=0),6,'factorized')['state']
    assert dead['hydration']==0 and dead['canteen_water']==state['canteen_water']
    # Ordinary interact away from a lake never drinks; fill needs a held canteen.
    out=d.step_model(model,dict(state,X=0,hydration=4),0,'factorized')['state']
    assert out['hydration']==3 and out['canteen_water']==1
    out=d.step_model(model,dict(state,X=2,canteen=0,canteen_water=0),0,'factorized')['state']
    assert out['canteen_water']==0
    # No wood / already held: bench must not consume resources or duplicate the canteen.
    for held,wood in [(0,0),(1,2)]:
        out=d.step_model(model,dict(state,X=1,canteen=held,wood=wood),0,'factorized')['state']
        assert out['wood']==wood and out['canteen']==held
    w['workshop']['values']={c['name']:state[c['name']] for c in workshop.catalog(w)}
    assert workshop.current(server.normalize_world(w))['canteen_water']==1
    assert server.task_catalog(w)[0]['na']==8


def test_regular_world_retains_six_actions():
    w=canteen_world();w['workshop']['objects']=[w['workshop']['objects'][0]]
    assert server.build_inspect_spaces(w)['X'].na==6
    assert server.task_catalog(w)[0]['na']==6


def test_chopping_and_forge_recipe():
    w=canteen_world()
    w['internal_goalstates_2_type_ind'][3]=0
    w['boolean_states_2_type_ind']={4:5}
    w['workshop']['chop_trees']=True
    w['workshop']['wood_capacity']=1
    spaces=server.build_inspect_spaces(w);spec=ta.prepare(w,{})
    assert ta.prepare(w,spec)==spec
    model={};d.evaluate(spaces,spec,build=True,retain=model)
    state=server.inspect_x0(w,spaces)
    state.update(X=3,axe=1,wood=0,hunger=4)
    for mode in ['factorized','tensor']:
        chopped=d.step_model(model,state,8,mode)['state']
        assert chopped['tree_3']==0 and chopped['wood']==1 and chopped['axe']==1
        after=d.step_model(model,chopped,0,mode)['state']
        assert after['hunger']==chopped['hunger']-1
        again=d.step_model(model,dict(chopped,wood=0),8,mode)['state']
        assert again['wood']==0
        noaxe=d.step_model(model,dict(state,axe=0),8,mode)['state']
        assert noaxe['tree_3']==1 and noaxe['wood']==0
        full=d.step_model(model,dict(state,wood=1),8,mode)['state']
        assert full['tree_3']==1
    assert w['workshop']['objects'][1]['kind']=='forge'
    assert w['workshop']['objects'][1]['recipe']=='canteen'


def test_vectorized_product_matches_original_builder():
    from deterministic_product import build
    w=canteen_world();spaces=server.build_inspect_spaces(w);spec=ta.prepare(w,{})
    original={};d.evaluate(spaces,spec,build=True,retain=original)
    fast,check=build(spaces,spec)
    assert check['max_probability_error']==0
    for a,b in zip(original['csr'],fast['csr']):assert (a!=b).nnz==0


def test_fill_is_distinct_from_drinking_at_lake():
    w=canteen_world();spaces=server.build_inspect_spaces(w);spec=ta.prepare(w,{})
    ordered,affs,_=d.compile_model(spaces,spec);model={'spaces':ordered,'affs':affs}
    state=server.inspect_x0(w,spaces);state.update(X=2,canteen=1,canteen_water=0,hydration=4)
    drink_lake=d.step_model(model,state,0,'factorized')['state']
    fill=d.step_model(model,state,7,'factorized')['state']
    assert drink_lake['hydration']==6 and drink_lake['canteen_water']==0
    assert fill['hydration']==3 and fill['canteen_water']==2
    away=d.step_model(model,dict(state,X=0),7,'factorized')['state']
    assert away['canteen_water']==0
