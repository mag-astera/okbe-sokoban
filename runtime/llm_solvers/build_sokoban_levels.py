"""Build PDF Sokoban levels with compact relational H and verified factorized play.

Search visits reachable push configurations, never the full Cartesian product.
The image-only maps on pages 22–24 specify keys, doors and physiology.
"""
import json
import sys
import math
from pathlib import Path
from collections import deque
from heapq import heappush, heappop
from itertools import permutations
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'llm_solvers')]
import tutorial_server as server
import tutorial_affordances as ta
import tutorial_boxes as boxes
import dbn_composer as dbn
import task_gui

OUT=ROOT/'tutorial/levels/progressive_levels'
WALLS={10,11,12,13,16,22,23,32,33,35,46,47,48,57,58,59,60,62,64,76}
TARGETS=[24,56,72]
MOVES=[2,3,4,5]
OPPOSITE={2:4,3:5,4:2,5:3}
TOP=40

def neighbor(x,a):return boxes.neighbor(x,a,9,9)
def reachable(x,positions,blocked):
    queue=deque([x]);paths={x:[]}
    while queue:
        x=queue.popleft()
        for a in MOVES:
            y=neighbor(x,a)
            if y is not None and y not in blocked and y not in positions and y not in paths:
                paths[y]=paths[x]+[a];queue.append(y)
    return paths


def push_plan(start,positions):
    """A* over reachable pushes; reverse-pull distances prune static dead squares."""
    distances=[]
    for target in TARGETS:
        ds={target:0};queue=deque([target])
        while queue:
            x=queue.popleft()
            for a in MOVES:
                b=neighbor(x,a);p=neighbor(b,a) if b is not None else None
                if b is not None and p is not None and b not in WALLS and p not in WALLS and b not in ds:
                    ds[b]=ds[x]+1;queue.append(b)
        distances.append(ds)
    def heuristic(bs):return min(sum(ds.get(b,999) for ds,b in zip(distances,order)) for order in permutations(bs))
    queue=[(heuristic(positions),0,start,tuple(positions),[])];best={}
    while queue:
        _,cost,x,bs,pushes=heappop(queue);paths=reachable(x,bs,WALLS);key=(min(paths),tuple(sorted(bs)))
        if best.get(key,math.inf)<=cost:continue
        best[key]=cost
        if set(bs)==set(TARGETS):return pushes,len(best)
        for i,b in enumerate(bs):
            for a in MOVES:
                behind=neighbor(b,OPPOSITE[a]);dest=neighbor(b,a)
                if behind not in paths or dest is None or dest in WALLS or dest in bs:continue
                nxt=list(bs);nxt[i]=dest;nxt=tuple(nxt);h=heuristic(nxt)
                if h<999:heappush(queue,(cost+1+h,cost+1,b,nxt,pushes+[(i,a)]))
    raise AssertionError('No solution to PDF box layout')


def generate(index,title,phys=False,doors=False):
    # Exact key and door placements from PDF pages 22 and 24.
    dd=[dict(id=i+1,cell=cell,color=color,open=False) for i,(cell,color) in enumerate([(9,'blue'),(34,'purple'),(45,'red'),(61,'orange')])] if doors else []
    clauses=[[['box_'+str(i),'in',TARGETS] for i in (1,2,3)]]
    if phys:clauses[0]+=[['hunger','>',0],['hydration','>',0]]
    raw=dict(rows=9,cols=9,start=36,walls=sorted(WALLS),constraints=[],p_main=1,
        boxes=[dict(id=i+1,cell=c) for i,c in enumerate([29,30,51])],doors=dd,
        goals=[dict(id=1,cells=[dict(li=c,a=None) for c in TARGETS])],task_clauses=clauses,
        internal_goalstates_2_type_ind={38:0,41:1} if phys else {},
        boolean_states_2_type_ind={8:1,21:1,74:1,80:1} if doors else {},
        nis=TOP+1,internal_state_values={0:12,1:12} if phys else {})
    if phys:raw['workshop']=dict(objects=[],values={},movement_metabolism=True)
    if index==11:
        raw.update(start=36,nis=16,internal_state_values={0:15,1:15},
            internal_goalstates_2_type_ind={**{c:0 for c in [1,37,44,78]},**{c:1 for c in [5,20,49,75]}})
    if index==13:
        raw.update(nis=16,internal_state_values={0:15,1:15},
            internal_goalstates_2_type_ind={**{c:0 for c in [1,37,44,78]},**{c:1 for c in [6,20,49,75]}})
    w=server.normalize_world(raw);spec=ta.prepare(w,{})
    if doors:
        for aff in spec['affordances']:
            if aff['target'].startswith('door_'):
                door=dd[int(aff['target'].split('_')[1])-1]
                adjacent=[neighbor(door['cell'],a) for a in MOVES]
                aff.update(placement_managed=False,rules=[dict(alpha=1,when={'and':[
                    {'var':{'red':'key_1','orange':'key_2','blue':'key_3','purple':'key_4'}[door['color']],'in':[1]},
                    {'var':'X','in':[c for c in adjacent if c is not None and c not in WALLS]},
                    {'var':'action','in':[0]}]})])
    if phys:
        spec['x_identity_when']={'or':[{'var':'hunger','in':[0]},{'var':'hydration','in':[0]}]}
        for aff in spec['affordances']:
            if aff['target'] in ('hunger','hydration'):
                cells=[c for c,t in w['internal_goalstates_2_type_ind'].items() if t==(0 if aff['target']=='hunger' else 1)]
                refill={'and':[{'var':'action','in':[0]},{'var':'X','in':cells}]}
                aff.update(placement_managed=False,source_binding="physiological_placement",rules=[dict(when=refill,alpha=1),
                    dict(when={'and':[{'var':'action','in':[0,1]},{'not':refill}]},alpha=2)])
    objective='Push all three boxes onto the tan targets at cells 24, 56 and 72; any box may fill any target.'
    if phys:objective+=' Keep hunger and hydration above 0.'
    controls='Arrows: move / push · Space: interact · W: wait'
    description=objective+' The walls, boxes, targets and starting position follow the PDF page 20–21 map. No chain pushes.'
    if index==11:
        description=objective+' Based on PDF page 23, with the top lake moved two cells right: agent starts at cell 36 (aligned with the other two challenges); trees at 1, 37, 44, 78; lakes at 5, 20, 49, 75. Hunger and hydration start at 15 and refill to 15 (states 0–15, with 0 death). Space eats or drinks on a source; arrow actions deplete both by one; other actions hold unless refilling. No chain pushes.'
    if doors:
        description=objective+' Matches the key/door map on PDF page '+('24' if phys else '22')+': agent starts at 36; red key 8 → red door 45; orange key 21 → orange door 61; blue key 74 → blue door 9; purple key 80 → purple door 34. Space collects a key on its cell or opens a matching door from an adjacent cell. Keys are retained; doors stay open. No chain pushes.'
        if phys:description+=' Trees: 1, 37, 44, 78. Lakes: 6, 20, 49, 75; the top lake is moved two cells right from its PDF position (4 → 6), as requested. Hunger and hydration start at 15 and refill to 15; arrow actions deplete each by one; Space/W hold unless refilling.'
    description+=' Use factorized Test. The full Cartesian product of this 9x9 three-box map exceeds the tutorial build limit; compact H does not enumerate it.'
    name=f'progressive_{index:02d}_{title}'
    return server.make_level(w,name=name,saved=1789250000+index,strings={},summary='9x9 · 3 boxes'+(' · keys / doors' if doors else '')+(' · survival' if phys else ''),api_version=server.API_VERSION,
        tutorial=dict(spec=spec,objective='task',objective_text=objective,controls_text=controls,description=description,
        source='progressively-harder-levels.pdf',source_pages=[18 if index==10 else 23 if index==11 else 22 if index==12 else 24],deterministic=True))


def verify(record, key_route=None):
    w=server.normalize_world(record['world']);spec=ta.prepare(w,record['tutorial']['spec'])
    spaces=server.build_inspect_spaces(w);ordered,affs,_=dbn.compile_model(spaces,spec);model=dict(spaces=ordered,affs=affs)
    state=server.inspect_x0(w,spaces);steps=[];trace=[dict(state)];phys='hunger' in state
    food={c for c,t in w['internal_goalstates_2_type_ind'].items() if t==0}
    water={c for c,t in w['internal_goalstates_2_type_ind'].items() if t==1}
    top=spaces['hunger'].ns-1 if phys else TOP
    assert not task_gui.validate(w['task_clauses'],spaces)
    def act(a):
        nonlocal state
        state=dbn.step_model(model,state,a,'factorized')['state'];steps.append(a);trace.append(dict(state))
        assert not server.play_task_status(w,state)['violations'],(record['name'],len(steps),state)
    def walk(targets,reserve=None):
        reserve=(7 if top==15 else 18) if reserve is None else reserve
        blocked=WALLS|{d['cell'] for d in w['doors'] if not state['door_'+str(d['id'])]}
        bs={state['box_'+str(i)] for i in (1,2,3)}
        if not phys:
            paths=reachable(state['X'],bs,blocked);c=min((c for c in targets if c in paths),key=lambda c:len(paths[c]))
            for a in paths[c]:act(a)
            return
        # Search only the walking subproblem at fixed box/door states.
        start=(state['X'],state['hunger'],state['hydration']);queue=deque([start]);prev={start:None};end=None
        while queue:
            node=queue.popleft();x,h,y=node
            if x in targets and min(h,y)>=reserve:end=node;break
            choices=[]
            if x in food and h<top:choices.append((0,(x,top,y)))
            if x in water and y<top:choices.append((0,(x,h,top)))
            if min(h,y)>1:
                for a in MOVES:
                    dest=neighbor(x,a)
                    if dest is not None and dest not in blocked and dest not in bs:choices.append((a,(dest,h-1,y-1)))
            for a,nxt in choices:
                if nxt not in prev:prev[nxt]=(node,a);queue.append(nxt)
        assert end is not None,('No safe walk',state,targets)
        route=[]
        while prev[end] is not None:end,a=prev[end];route.append(a)
        for a in reversed(route):act(a)
    if w['doors']:
        for cell,ident in (key_route if key_route is not None else [(21,4),(74,1),(8,3),(80,2)]):
            walk({cell},reserve=7);act(0)
            door=next(d for d in w['doors'] if d['id']==ident)
            walk({neighbor(door['cell'],a) for a in MOVES}-{None}-WALLS,reserve=7);act(0)
            assert state['door_'+str(ident)]==1
        assert all(state['key_'+str(i)]==1 for i in range(1,5))
    pushes,expanded=push_plan(state['X'],[state['box_'+str(i)] for i in (1,2,3)])
    if phys and top==15:
        # Plan refills over the entire push sequence; greedy per-push reserves can
        # strand the agent when a box temporarily blocks a route to food/water.
        configurations=[tuple(state['box_'+str(i)] for i in (1,2,3))]
        for i,a in pushes:
            nxt=list(configurations[-1]);nxt[i]=neighbor(nxt[i],a);configurations.append(tuple(nxt))
        start=(0,state['X'],state['hunger'],state['hydration'])
        queue=deque([start]);prev={start:None};end=None
        while queue:
            node=queue.popleft();k,x,h,y=node
            if k==len(pushes):end=node;break
            occupied=set(configurations[k]);choices=[]
            if x in food and h<top:choices.append((0,(k,x,top,y)))
            if x in water and y<top:choices.append((0,(k,x,h,top)))
            if min(h,y)>1:
                for a in MOVES:
                    dest=neighbor(x,a)
                    if dest is not None and dest not in WALLS and dest not in occupied:
                        choices.append((a,(k,dest,h-1,y-1)))
                i,a=pushes[k];b=configurations[k][i]
                if x==neighbor(b,OPPOSITE[a]):choices.append((a,(k+1,b,h-1,y-1)))
            for a,nxt in choices:
                if nxt not in prev:prev[nxt]=(node,a);queue.append(nxt)
        assert end is not None,'No physiological route for this push sequence'
        route=[]
        while prev[end] is not None:end,a=prev[end];route.append(a)
        for a in reversed(route):act(a)
    else:
        for i,a in pushes:
            b=state['box_'+str(i+1)];walk({neighbor(b,OPPOSITE[a])})
            expected=neighbor(b,a);act(a);assert state['box_'+str(i+1)]==expected
    assert server.play_task_status(w,state)['complete']
    if phys:
        for field in ('hunger','hydration'):
            assert any(b[field]>a[field] for a,b in zip(trace,trace[1:])),field
    return dict(level=record['name'],steps=steps,final_state=state,pushes=len(pushes),push_search_states=expanded,
        joint_states=math.prod(s.ns for s in ordered),verified='Every action replayed through compiled DBN H/F and local kernels; goal satisfied without constraints; physiology refills used when present.')


def main():
    cases=[(10,'sokoban',False,False),(11,'sokoban_survival',True,False),
           (12,'sokoban_keys_and_doors',False,True),(13,'sokoban_keys_doors_survival',True,True)]
    records=[generate(*case) for case in cases];reports=[]
    for record in records:
        report=verify(record);reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k!='steps'}),flush=True)
    if '--write' in sys.argv:
        for record in records:
            path=OUT/(record['name']+'.json')
            if path.exists():raise FileExistsError(path)
        for record in records:(OUT/(record['name']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        (OUT/'sokoban_verification.txt').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
