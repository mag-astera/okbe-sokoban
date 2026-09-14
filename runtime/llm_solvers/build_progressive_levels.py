"""Build and validate the six currently supported PDF scenarios (no existing saves overwritten)."""
import json
import sys
from pathlib import Path
from collections import deque
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'llm_solvers')]
import tutorial_server as server
import tutorial_affordances as affordances
import dbn_composer as dbn
import task_gui

OUT=ROOT/'tutorial/levels/progressive_levels'

def generate(index, title, phys=False, items=False, door=False):
    start,goal=(72,8) if phys else (36,44)
    w=dict(rows=9,cols=9,walls=[r*9+4 for r in range(9) if r not in (3,4,5)],constraints=[31,49],
           start=start,green_badgers=[goal],p_main=1.0,kernel={},goals=[{'id':1,'cells':[{'li':goal,'a':None}]}],nextGoalId=2,
           doors=[{'id':1,'cell':40,'color':'orange','open':False}] if door else [],buttons=[],
           internal_goalstates_2_type_ind={'38':0,'42':1} if phys else {},boolean_states_2_type_ind={},
           nis=17,nis_by_type={'0':17,'1':17} if phys else {},internal_state_values={'0':16,'1':16} if phys else {},
           item_held={},task_clauses=[[['X','==',goal]]],task_clause_ix=0)
    if items:
        w['boolean_states_2_type_ind']={'64':2,'16':3}
        w['task_clauses'][0]+=[['flower','==',1],['honey','==',1]]
    if phys:w['task_clauses'][0]+=[['hunger','>',0],['hydration','>',0]]
    if door:
        w['boolean_states_2_type_ind']={str(64 if phys else 10):1}
        w['task_clauses'][0]+=[['key','==',1],['door_1','==',1]]
    world=server.normalize_world(w)
    spec=affordances.prepare(world,{})
    if phys:spec['x_identity_when']={'or':[{'var':'hunger','in':[0]},{'var':'hydration','in':[0]}]}
    if door:
        for a in spec['affordances']:
            if a['target']=='door_1':
                a.update(placement_managed=False,rules=[{'when':{'and':[{'var':'key','in':[1]},{'var':'X','in':[39,41]},{'var':'action','in':[0]}]},'alpha':1}])
    description=('Goal: reach the green badger'+(' while holding both flowers and honey' if items else '')+'. Avoid fire.'
       +(' Keep hunger and hydration above zero. Interact at the tree to eat and at the lake to drink. Both bars start at 16; every step, including interaction, advances time.' if phys else '')
       +(' Collect the key, then interact beside the closed orange door (cell 39 or 41) to unlock it. The key is retained and the door stays open.' if door else '')
       +' Use Test: arrow keys move; Space interacts or collects an item. For planning use DBN Solve & plot with Task tab objective; the flat grid solver does not plan inventory or physiology.'
       +(' Delivery means arriving while holding the items; giving or dropping them is not required.' if items else ''))
    name=f'progressive_{index:02d}_{title}'
    record=server.make_level(world,name=name,saved=1789240000-index,strings={},summary=server.describe_world(world),api_version=server.API_VERSION,
        tutorial={'spec':spec,'objective':'task','description':description,'source':'progressively-harder-levels.pdf','deterministic':True})
    return record

def path(world,start,end,closed):
    queue=deque([(start,[])]);seen={start};blocked=set(world['walls'])|set(world['constraints'])|set(closed)
    while queue:
        x,actions=queue.popleft()
        if x==end:return actions
        r,c=divmod(x,9)
        for a,dr,dc in [(2,1,0),(3,0,1),(4,-1,0),(5,0,-1)]:
            rr,cc=r+dr,c+dc;y=rr*9+cc
            if 0<=rr<9 and 0<=cc<9 and y not in seen and y not in blocked:
                seen.add(y);queue.append((y,actions+[a]))
    raise AssertionError(('no route',start,end))

def verify(record,phys,items,door):
    world=server.normalize_world(record['world']);spec=record['tutorial']['spec']
    spaces=server.build_inspect_spaces(world);ordered,affs,_=dbn.compile_model(spaces,spec)
    model={'spaces':ordered,'affs':affs}
    state=server.inspect_x0(world,spaces)
    goal=task_gui.task_gate(world['task_clauses'])
    assert not goal.eval(state,0)
    steps=[]
    def step(a):
        nonlocal state
        state=dbn.step_model(model,state,a,'factorized',np.random.default_rng(0))['state'];steps.append(a)
        assert state['X'] not in world['constraints']
        if phys:assert state['hunger']>0 and state['hydration']>0,(record['name'],len(steps),state)
    def visit(cell,interact=False):
        closed=[40] if door and state['door_1']==0 else []
        for a in path(world,state['X'],cell,closed):step(a)
        if interact:step(0)
    if door:
        # Without the key, interacting beside the door cannot unlock it.
        probe=dict(state,X=39,key=0)
        assert dbn.step_model(model,probe,0,'factorized')['state']['door_1']==0
        assert dbn.step_model(model,probe,3,'factorized')['state']['X']==39
        visit(64 if phys else 10,True);assert state['key']==1
    if items:visit(64,True);assert state['flower']==1
    if phys:visit(38,True);assert state['hunger']==16
    if door:visit(39,True);assert state['door_1']==1
    if phys:visit(42,True);assert state['hydration']==16
    if items:visit(16,True);assert state['honey']==1
    visit(world['green_badgers'][0]);assert goal.eval(state,0)
    if items:
        assert not goal.eval(dict(state,honey=0),0)
        assert not goal.eval(dict(state,flower=0),0)
    if phys:
        dead=dict(state,hunger=0)
        assert dbn.step_model(model,dead,5,'factorized')['state']['X']==dead['X']
    return {'level':record['name'],'steps':steps,'final_state':state,'joint_states':int(np.prod([s.ns for s in ordered])),'verified':'Successful factorized rollout; objective and constraint checks'}

if __name__=='__main__':
    server.warm_solver();assert server.SOLVER['loaded'],server.SOLVER
    cases=[('reach_avoid',False,False,False),('flowers_and_honey',False,True,False),('physiological_survival',True,False,False),('items_and_survival',True,True,False),('key_and_door',False,False,True),('key_door_and_survival',True,False,True)]
    results=[];records=[]
    for i,(title,phys,items,door) in enumerate(cases,1):
        record=generate(i,title,phys,items,door)
        report=verify(record,phys,items,door);print(json.dumps(report),flush=True)
        records.append(record);results.append(report)
    if '--write' in sys.argv:
        for record in records:
            p=OUT/(record['name']+'.json')
            if p.exists():raise FileExistsError(p)
        for record in records:(OUT/(record['name']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        (OUT/'progressive_verification.txt').write_text(json.dumps(results,indent=2)+'\n')
