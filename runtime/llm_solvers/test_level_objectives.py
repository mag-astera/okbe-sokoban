"""Saved level tasks must express the stated joint goal, not just the agent cell."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_tutorial_workshop import server,ta,d
import task_gui

ROOT=Path(__file__).resolve().parents[1]/'tutorial'/'levels'
FILES=sorted((ROOT/'progressive_levels').glob('*.json'))+sorted((ROOT/'sokoban').glob('*.json'))
# Independent, human-specified winning states and ways they must fail.
EXPECTED={
 'progressive_01':({'X':44},{'X':43}),
 'progressive_02':({'X':44,'flower':1,'honey':1},{'X':43,'flower':0,'honey':0}),
 'progressive_03':({'X':8,'hunger':1,'hydration':1},{'X':7,'hunger':0,'hydration':0}),
 'progressive_04':({'X':8,'flower':1,'honey':1,'hunger':1,'hydration':1},{'X':7,'flower':0,'honey':0,'hunger':0,'hydration':0}),
 'progressive_05':({'X':44,'key':1,'door_1':1},{'X':43,'key':0,'door_1':0}),
 'progressive_06':({'X':8,'key':1,'door_1':1,'hunger':1,'hydration':1},{'X':7,'key':0,'door_1':0,'hunger':0,'hydration':0}),
 'progressive_07':({'axe':1},{'axe':0}),
 'progressive_08':({'money':5},{'money':4}),
 'progressive_09':({'X':16,'hunger':1,'hydration':1,'canteen':1,'tree_48':1},{'X':15,'hunger':0,'hydration':0,'canteen':0,'tree_48':0}),
 'sokoban_01':({'box_1':9,'box_2':10},{'box_1':5,'box_2':6}),
 'progressive_10':({'box_1':24,'box_2':56,'box_3':72},{'box_1':19,'box_2':30,'box_3':51}),
 'progressive_11':({'box_1':24,'box_2':56,'box_3':72,'hunger':1,'hydration':1},{'box_1':19,'box_2':30,'box_3':51,'hunger':0,'hydration':0}),
 'progressive_12':({'box_1':24,'box_2':56,'box_3':72},{'box_1':19,'box_2':30,'box_3':51}),
 'progressive_13':({'box_1':24,'box_2':56,'box_3':72,'hunger':1,'hydration':1},{'box_1':19,'box_2':30,'box_3':51,'hunger':0,'hydration':0}),
}


@pytest.mark.parametrize('path',FILES,ids=lambda p:p.stem)
def test_saved_task_is_valid_nonempty_and_matches_objective(path):
    record=json.loads(path.read_text());w=server.normalize_world(record['world'])
    clauses=w['task_clauses'];catalog=server.task_catalog(w)
    assert clauses and all(clauses)
    assert record['tutorial']['objective']=='task'
    assert not task_gui.validate(clauses,{s['name']:SimpleNamespace(ns=s['ns']) for s in catalog})
    goal=task_gui.task_gate(clauses)
    prefix='_'.join(path.stem.split('_')[:2]);good,bad=EXPECTED[prefix]
    state={s['name']:0 for s in catalog};state.update(good)
    assert goal.eval(state,0)
    for name,value in bad.items():
        assert not goal.eval(dict(state,**{name:value}),0),(path.name,name)
    # These are joint goals, independent of the BL action selected at termination.
    assert all(goal.eval(state,a) for a in range(next(s['na'] for s in catalog if s['name']=='X')))


@pytest.mark.parametrize('path',FILES,ids=lambda p:p.stem)
def test_saved_level_has_successful_factorized_route_to_its_task(path):
    record=json.loads(path.read_text());w=server.normalize_world(record['world'])
    spaces=server.build_inspect_spaces(w);spec=ta.prepare(w,record['tutorial']['spec'])
    ordered,affs,_=d.compile_model(spaces,spec);model=dict(spaces=ordered,affs=affs)
    state=server.inspect_x0(w,spaces);goal=task_gui.task_gate(w['task_clauses'])
    assert not goal.eval(state,0)
    assert not server.play_task_status(w,state)['complete']
    if path.stem.startswith('sokoban'):
        actions=[4,3,2,4,3,2]
    elif path.stem.startswith(('progressive_10','progressive_11','progressive_12','progressive_13')):
        records=json.loads((path.parent/'sokoban_verification.txt').read_text())
        actions=next(r['steps'] for r in records if r['level']==path.stem)
    elif path.stem.startswith('progressive_09'):
        actions=json.loads((path.parent/'progressive_09_canteen_journey_verification.txt').read_text())['actions']
    elif path.stem.startswith(('progressive_07','progressive_08')):
        actions=[]
        # 3x5 workshop: collect at 0 and 4; craft at 7; sell at 12.
        from collections import deque
        def route(start,end):
            queue=deque([(start,[])]);seen={start}
            while queue:
                x,steps=queue.popleft()
                if x==end:return steps
                r,c=divmod(x,w['cols'])
                for a,dr,dc in [(2,1,0),(3,0,1),(4,-1,0),(5,0,-1)]:
                    rr,cc=r+dr,c+dc;y=rr*w['cols']+cc
                    if 0<=rr<w['rows'] and 0<=cc<w['cols'] and y not in seen and y not in w['walls']:
                        seen.add(y);queue.append((y,steps+[a]))
            raise AssertionError('No workshop route')
        x=state['X']
        for _ in range(5 if path.stem.startswith('progressive_08') else 1):
            for cell in ([0,4,7,12] if path.stem.startswith('progressive_08') else [0,4,7]):
                actions+=route(x,cell)+[0];x=cell
    else:
        records=json.loads((path.parent/'progressive_verification.txt').read_text())
        actions=next(r['steps'] for r in records if r['level']==path.stem)
    for action in actions:
        state=d.step_model(model,state,action,'factorized')['state']
        assert state['X'] not in w['constraints']
        for name in ('hunger','hydration'):
            if name in state:assert state[name]>0
    assert goal.eval(state,0),(path.name,state)
    assert server.play_task_status(w,state)['complete']


def test_play_outcome_constraints_override_goal_and_empty_tasks_do_not_win():
    w=server.normalize_world(dict(rows=3,cols=3,start=0,constraints=[1],
        task_clauses=[[['X','==',1]]],internal_goalstates_2_type_ind={2:0,3:1,4:3}))
    safe={'X':0,'hunger':2,'hydration':2,'money':0}
    assert not server.play_task_status(w,safe)['violations']  # Empty wallet isn't death.
    result=server.play_task_status(w,dict(safe,X=1))
    assert not result['complete'] and result['violations']
    for name in ('hunger','hydration'):
        assert server.play_task_status(w,dict(safe,**{name:0}))['violations']
    for clauses in ([],[[]],[[['X','==',99]]],[[['missing','==',0]]]):
        result=server.play_task_status(w,safe,clauses)
        assert not result['ready'] and not result['complete']
    w['start']=None
    assert not server.play_task_status(w)['complete']


def test_step_reports_entry_into_constraint_and_departure():
    w=server.normalize_world(dict(rows=3,cols=3,start=0,p_main=1,constraints=[1],
                                 task_clauses=[[['X','==',1]]]))
    result=server.dbn_panel_data(w,dict(operation='step',spec={},action=3,use_world_state=True))
    assert not result['play_status_before']['violations']
    assert result['play_status']['violations'] and not result['play_status']['complete']
    w.update(result['world_state'])
    result=server.dbn_panel_data(w,dict(operation='step',spec=result['spec'],action=3,use_world_state=True))
    assert result['play_status_before']['violations']
    assert not result['play_status']['violations']  # Browser retains the episode's prior loss.
