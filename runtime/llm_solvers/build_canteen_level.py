"""Create level 09 and verify full-kernel equivalence and the necessity of drinking."""
import json
import sys
from pathlib import Path
from collections import deque
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import tutorial_server as server
import tutorial_affordances as ta
import dbn_composer as d


def build():
    server.warm_solver()
    world=server.normalize_world(dict(rows=9,cols=9,start=64,green_badgers=[16],p_main=1,nis=11,
        nis_by_type={0:11,1:12},internal_state_values={0:10,1:10},
        internal_goalstates_2_type_ind={45:0,48:0,75:0,72:1},boolean_states_2_type_ind={66:5},
        workshop={'objects':[dict(id=1,kind='forge',recipe='canteen',cell=57,quantity=1)],
                  'values':{},'chop_trees':True,'wood_capacity':1,'water_capacity':1,'movement_metabolism':True},
        task_clauses=[[['X','==',16],['hunger','>',0],['hydration','>',0],['canteen','==',1],['tree_48','==',1]]]))
    spec=ta.prepare(world,{})
    spec['x_identity_when']={'or':[{'var':n,'in':[0]} for n in ('hunger','hydration')]}
    spaces=server.build_inspect_spaces(world)
    from deterministic_product import build as build_product
    model,verification=build_product(spaces,spec)
    report={'product':{'verification':verification,'states':int(np.prod([s.ns for s in model['spaces']]))}}
    print('Searching for a successful route…',flush=True)
    names=[s.name for s in model['spaces']];sizes=[s.ns for s in model['spaces']]
    initial=server.inspect_x0(world,spaces)
    coords=np.unravel_index(np.arange(int(np.prod(sizes))),sizes)
    safe=(coords[names.index('hydration')]>0)&(coords[names.index('hunger')]>0)
    goal=safe&(coords[0]==16)&(coords[names.index('canteen')]==1)&(coords[names.index('tree_48')]==1)
    def search(actions, overrides=None, goal_override=None):
        target=goal if goal_override is None else goal_override
        start_state={**initial,**(overrides or {})}
        start=int(np.ravel_multi_index(tuple(start_state[n] for n in names),sizes))
        parent=np.full(len(safe),-1,dtype=np.int32);used=np.zeros(len(safe),dtype=np.int8)
        parent[start]=start;q=deque([start])
        while q:
            row=q.popleft()
            if target[row]:
                path=[]
                while row!=start:path.append(int(used[row]));row=int(parent[row])
                return path[::-1]
            for action in actions:
                nxt=int(model['csr'][action].indices[row])
                if safe[nxt] and parent[nxt]<0:
                    parent[nxt]=row;used[nxt]=action;q.append(nxt)
        return None
    route=search(range(9));print('Candidate route:',route,flush=True);assert route and 6 in route
    assert search([0,1,2,3,4,5,7,8]) is None,'Journey can be completed without drinking'
    assert search(range(9),{'tree_48':0},safe&(coords[0]==16)&(coords[names.index('canteen')]==1)) is None,'Central tree is not necessary for survival'
    # Replay successful policy path through both execution modes.
    state=initial
    for a in route:
        factor=d.step_model(model,state,a,'factorized')['state']
        tensor=d.step_model(model,state,a,'tensor')['state'];assert factor==tensor
        state=factor
    name='progressive_09_canteen_journey'
    description=('PDF page 9 layout: collect the axe at cell 66. Use Chop / Q on a tree to obtain '
        'one wood. Preserve the central tree (48): it is needed for food on the journey. '
        'Space at the forge (57) crafts a canteen. Fill canteen / L at the lake (72) fills it; Space drinks directly from the lake; '
        'Drink / R restores hydration using one serving. Reach the green badger at 16 alive. '
        'Trees are at 45, 48, 75; the agent starts at 64. Hunger has 10 live levels plus death; '
        'hydration has 11 live levels plus death, adjusted for four-direction movement and '
        'metabolism on movement only: interactions still transition the DBN but hold physiological levels unless refilling. Wood capacity is one unit. The forge recipe is canteen.')
    record=server.make_level(world,name=name,saved=1789240000-9,summary=server.describe_world(world),
        strings={},api_version=server.API_VERSION,
        tutorial={'spec':spec,'objective':'task','description':description})
    folder=Path(server.LEVELDIR)/'progressive_levels';folder.mkdir(exist_ok=True)
    path=folder/(name+'.json')
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('world',{}).get('rows')!=2 and '--replace' not in sys.argv:raise FileExistsError('Use --replace to replace the generated level')
        backup=folder/(name+'_corridor_backup.txt')
        if not backup.exists():backup.write_text(path.read_text())
    path.write_text(json.dumps(record,indent=2)+'\n')
    verification={'actions':route,'final_state':state,'states':int(np.prod(sizes)),
        'actions_count':9,'no_drink_route':False,'product':report.get('product')}
    (folder/(name+'_verification.txt')).write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':build()
