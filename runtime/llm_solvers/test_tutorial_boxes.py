import itertools
import numpy as np
import pytest
from test_tutorial_workshop import server,ta,d
import tutorial_boxes as boxes


def world(**extra):
    return server.normalize_world(dict(rows=3,cols=3,start=3,p_main=1,
        boxes=[dict(id=1,cell=4),dict(id=2,cell=7)],**extra))


def compile_world(w,build=False):
    spaces=server.build_inspect_spaces(w);spec=ta.prepare(w,{})
    model={}; preview=d.evaluate(spaces,spec,server.inspect_x0(w,spaces),3,build,model)
    if not build:
        ordered,affs,_=d.compile_model(spaces,spec);model.update(spaces=ordered,affs=affs)
    return model,preview,spec


def test_push_and_collisions():
    w=world();m,preview,_=compile_world(w)
    assert preview['alphas']['box_1']==3 and preview['alphas']['box_2']==0
    before={'X':3,'box_1':4,'box_2':7}
    out=d.step_model(m,before,3,'factorized')['state']
    assert out=={'X':4,'box_1':5,'box_2':7}
    assert d.step_model(m,out,3,'factorized')['state']==out # boundary
    chain={'X':3,'box_1':4,'box_2':5}
    assert d.step_model(m,chain,3,'factorized')['state']==chain
    assert d.step_model(m,before,0,'factorized')['state']==before
    away=d.step_model(m,before,4,'factorized')['state']
    assert away==dict(before,X=0)
    m,_,_=compile_world(world(walls=[5]))
    assert d.step_model(m,before,3,'factorized')['state']==before


def test_doors_and_identity_gate_stop_both_agent_and_box():
    w=world(doors=[dict(id=1,cell=5,open=False)])
    m,_,spec=compile_world(w)
    s={'X':3,'box_1':4,'box_2':7,'door_1':0}
    assert d.step_model(m,s,3,'factorized')['state']==s
    opened=dict(s,door_1=1)
    assert d.step_model(m,opened,3,'factorized')['state']==dict(opened,X=4,box_1=5)
    spec['x_identity_when']={'var':'door_1','in':[1]}
    ordered,affs,_=d.compile_model(server.build_inspect_spaces(w),spec)
    assert d.step_model({'spaces':ordered,'affs':affs},opened,3,'factorized')['state']==opened


def oracle(x,b,c,a,rows=3,cols=3,walls=frozenset()):
    if len({x,b,c})<3 or any(v in walls for v in (x,b,c)):return (x,b,c)
    dr,dc=boxes.MOVES.get(a,(0,0)); r,col=divmod(x,cols)
    nr,nc=r+dr,col+dc
    if not(0<=nr<rows and 0<=nc<cols):return (x,b,c)
    ahead=nr*cols+nc
    if ahead in walls:return (x,b,c)
    if ahead in (b,c):
        nr+=dr;nc+=dc
        if not(0<=nr<rows and 0<=nc<cols) or nr*cols+nc in (b,c) or nr*cols+nc in walls:return(x,b,c)
        return(ahead,nr*cols+nc if ahead==b else b,nr*cols+nc if ahead==c else c)
    return(ahead,b,c)


def test_all_product_rows_match_independent_sokoban_and_factorized_play():
    m,preview,_=compile_world(world(walls=[2]),True)
    assert preview['product']['verification']['max_probability_error']==0
    for a,(x,b,c) in itertools.product(range(6),itertools.product(range(9),repeat=3)):
        expected=oracle(x,b,c,a,walls={2});flat=np.ravel_multi_index((x,b,c),(9,9,9))
        row=m['csr'][a].getrow(flat)
        assert row.nnz==1 and row.data[0]==1
        assert row.indices[0]==np.ravel_multi_index(expected,(9,9,9))
        state=dict(X=x,box_1=b,box_2=c)
        for mode in ('factorized','tensor'):
            result=d.step_model(m,state,a,mode)['state']
            assert tuple(result[n] for n in ('X','box_1','box_2'))==expected
        if len({x,b,c})==3 and not {x,b,c}&{2}:
            assert len(set(expected))==3 and not set(expected)&{2}


def test_no_box_configuration_enumeration_for_compile_or_preview(monkeypatch):
    w=server.normalize_world(dict(rows=4,cols=4,start=0,p_main=1,
        boxes=[dict(id=i,cell=i) for i in range(1,9)]))
    # 16**9 conceptual states. Parent-combination enumeration is forbidden even
    # once; sparse local position kernels and compact H must still compile.
    def forbidden(*args,**kwargs):raise AssertionError('Enumerated parent configurations')
    monkeypatch.setattr(d.itertools,'product',forbidden)
    m,preview,_=compile_world(w)
    assert preview['joint_states']==16**9 and not preview['can_build']
    assert all(not hasattr(m['affs'].by_driven[n],'tensor') for n in boxes.current(w))
    assert all(m['affs'].by_driven[n].H_alpha.size==30 for n in boxes.current(w))


def test_stochastic_movement_rejected_and_box_state_cache_stable():
    w=world();w['p_main']=.8
    with pytest.raises(ValueError,match='deterministic'):compile_world(w)
    from tutorial_doors import dynamics
    w=world();moved=world();moved['boxes'][0]['cell']=5
    assert dynamics(w)==dynamics(moved)
    assert ta.prepare(w,{})==ta.prepare(moved,{})


def test_demo_level_solve_play_save_and_box_only_objective():
    w,_,tutorial=server.load_level('sokoban/sokoban_01_two_boxes',with_tutorial=True)
    spec=tutorial['spec']
    result=server.dbn_panel_data(w,dict(operation='solve',spec=spec,algorithm='feas_eta_fast',options={'objective':'task'}))
    assert result['joint_states']==12**3
    # Follow the independently specified six-action route with factorization and
    # with the cached full kernel. Moves must preserve the cache identity.
    for a in [4,3,2,4,3,2]:
        factor=server.dbn_panel_data(w,dict(operation='step',spec=spec,action=a,use_world_state=True,mode='factorized'))
        tensor=server.dbn_panel_data(w,dict(operation='step',spec=spec,action=a,use_world_state=True,mode='tensor'))
        assert factor['state']==tensor['state']
        w=server.normalize_world({**w,**factor['world_state']})
    assert w['boxes']==[{'id':1,'cell':9},{'id':2,'cell':10}]
    assert server.hl_panel_data(w)['box_state']=={'box_1':9,'box_2':10}
    # Use current positions to request a goal-state slice of the original solve.
    goal=server.dbn_panel_data(w,dict(operation='slice',spec=spec,solution_id=result['solution_id'],fixed={'box_1':9,'box_2':10},cell=w['start']))
    assert goal['value']==1


def test_physiological_identity_cannot_leave_box_moving_alone():
    w=world(internal_goalstates_2_type_ind={0:0},nis=3)
    spaces=server.build_inspect_spaces(w);spec=ta.prepare(w,{})
    spec['x_identity_when']={'var':'hunger','in':[0]}
    ordered,affs,reports=d.compile_model(spaces,spec)
    assert 'hunger' in affs.by_driven['box_1'].parents
    m=dict(spaces=ordered,affs=affs)
    state=dict(X=3,box_1=4,box_2=7,hunger=0)
    assert d.step_model(m,state,3,'factorized')['state']==state
    living=dict(state,hunger=2)
    assert d.step_model(m,living,3,'factorized')['state']==dict(X=4,box_1=5,box_2=7,hunger=1)
