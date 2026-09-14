import numpy as np
from test_dbn_composer import setup
import tutorial_server as server
import tutorial_affordances as ta
import tutorial_workshop as workshop
import dbn_composer as d


def world(quantity=1):
    return server.normalize_world(dict(rows=3,cols=4,walls=[],constraints=[],start=0,goals=[],p_main=1,
        workshop={'objects':[{'id':i+1,'cell':i,'kind':k,'quantity':quantity} for i,k in enumerate(['wood','iron','forge','store'])], 'values':{}}))

def model(w,build=False):
    server.warm_solver();spaces=server.build_inspect_spaces(w);spec=ta.prepare(w,{})
    assert ta.prepare(w,spec)==spec
    ordered,affs,_=d.compile_model(spaces,spec);m={'spaces':ordered,'affs':affs}
    if build:d.evaluate(spaces,spec,build=True,retain=m)
    return m,server.inspect_x0(w,spaces)

def test_exchange_and_full_tensor():
    w=world();m,state=model(w,True)
    for a in [0,3,0,3,0,3,0]:
        before=dict(state)
        factor=d.step_model(m,state,a,'factorized')['state']
        tensor=d.step_model(m,state,a,'tensor')['state']
        assert factor==tensor
        state=factor
    assert state['money']==1 and state['axe']==0
    assert state['wood']==state['iron']==state['resource_1']==state['resource_2']==0
    assert d.step_model(m,dict(state,X=0),0,'factorized')['state']['wood']==0
    # No ingredients: forge must not grant an axe.
    assert d.step_model(m,dict(state,X=2),0,'factorized')['state']['axe']==0
    # Full wallet: selling must not consume the axe.
    full=d.step_model(m,dict(state,X=3,money=5,axe=1),0,'factorized')['state']
    assert full['money']==5 and full['axe']==1
    # Full inventory: collecting must not consume source stock.
    full=d.step_model(m,dict(state,X=0,wood=5,resource_1=1),0,'factorized')['state']
    assert full['wood']==5 and full['resource_1']==1
    # Missing iron must not consume wood.
    missing=d.step_model(m,dict(state,X=2,wood=1,iron=0,axe=0),0,'factorized')['state']
    assert missing['wood']==1 and missing['axe']==0

def test_five_sales_and_state_roundtrip():
    w=world(5);m,state=model(w)
    for _ in range(5):
        for cell in range(4):
            state=d.step_model(m,dict(state,X=cell),0,'factorized')['state']
    assert state['money']==5 and state['resource_1']==state['resource_2']==0
    w['workshop']['values']={c['name']:state[c['name']] for c in workshop.catalog(w)}
    w=server.normalize_world(w)
    assert workshop.current(w)==w['workshop']['values']
    # Runtime values do not invalidate the transition-kernel cache.
    import tutorial_doors
    empty=world(5)
    assert tutorial_doors.dynamics(w)==tutorial_doors.dynamics(empty)

def test_existing_money_and_axe_are_reused():
    w=world();w['internal_goalstates_2_type_ind']={8:3};w['nis_by_type']={3:8};w['internal_state_values']={3:4}
    w['boolean_states_2_type_ind']={9:5};w['item_held']={9:1}
    spaces=server.build_inspect_spaces(w);state=server.inspect_x0(w,spaces)
    assert spaces['money'].ns==8 and state['money']==4 and state['axe']==1
    names=[c['name'] for c in server.task_catalog(w)];assert len(names)==len(set(names))
