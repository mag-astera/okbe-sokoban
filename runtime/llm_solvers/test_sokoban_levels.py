"""PDF layout, permutation-invariant goals, keys/doors and physiological H checks."""
import itertools
import json
from pathlib import Path
import pytest
from test_tutorial_workshop import server,ta,d
from build_sokoban_levels import WALLS,TARGETS

ROOT=Path(__file__).resolve().parents[1]/'tutorial/levels/progressive_levels'
FILES=sorted(p for p in ROOT.glob('*.json') if p.stem.startswith(('progressive_10','progressive_11','progressive_12','progressive_13')))

@pytest.mark.parametrize('path',FILES,ids=lambda p:p.stem)
def test_pdf_layout_any_box_goal_and_compact_preview(path):
    record=json.loads(path.read_text());w=server.normalize_world(record['world'])
    assert set(w['walls'])==WALLS
    assert w['start']==(25 if path.stem.startswith('progressive_11') else 28 if w['doors'] else 36)
    if path.stem.startswith('progressive_11'):
        assert w['internal_goalstates_2_type_ind']=={1:0,37:0,44:0,78:0,5:1,20:1,49:1,75:1}
        assert w['nis']==16 and w['internal_state_values']=={0:15,1:15}
    assert [b['cell'] for b in w['boxes']]==[19,30,51]
    spec=ta.prepare(w,record['tutorial']['spec']);assert spec==ta.prepare(w,spec)
    spaces=server.build_inspect_spaces(w);state=server.inspect_x0(w,spaces)
    preview=d.evaluate(spaces,spec,state)
    assert preview['joint_states']>=81**4 and not preview['can_build']
    for perm in itertools.permutations(TARGETS):
        goal={**state,**{'box_'+str(i+1):c for i,c in enumerate(perm)}}
        assert server.play_task_status(w,goal)['complete']
    ordered,affs,_=d.compile_model(spaces,spec);model=dict(spaces=ordered,affs=affs)
    if w['doors']:
        assert [(x['cell'],x['color'],x['open']) for x in w['doors']]==[(9,'blue',False),(34,'purple',False),(45,'red',False),(61,'orange',False)]
        assert w['boolean_states_2_type_ind']=={8:1,21:1,74:1,80:1}
        if path.stem.startswith('progressive_13'):
            assert w['internal_goalstates_2_type_ind']=={1:0,37:0,44:0,78:0,6:1,20:1,49:1,75:1}
            assert w['nis']==16 and w['internal_state_values']=={0:15,1:15}
        # Matching keys only; opening is explicit interaction, not occupancy.
        for door in w['doors']:
            cell=next(c for c in [door['cell']-9,door['cell']+9,door['cell']-1,door['cell']+1] if 0<=c<81 and c not in WALLS and c not in [19,30,51])
            right={'red':'key_1','orange':'key_2','blue':'key_3','purple':'key_4'}[door['color']];name='door_'+str(door['id'])
            probe=dict(state,X=cell)
            for wrong in set(['key_1','key_2','key_3','key_4'])-{right}:
                incorrect=dict(probe,**{wrong:1})
                assert d.step_model(model,incorrect,0,'factorized')['state'][name]==0
            probe[right]=1
            assert d.step_model(model,probe,1,'factorized')['state'][name]==0
            assert d.step_model(model,probe,0,'factorized')['state'][name]==1
        probe=dict(state,X=43,box_1=52)
        assert d.step_model(model,probe,2,'factorized')['state']['box_1']==52
        probe['door_4']=1
        assert d.step_model(model,probe,2,'factorized')['state']['box_1']==61
    if 'hunger' in state:
        for cell,t in w['internal_goalstates_2_type_ind'].items():
            name='hunger' if t==0 else 'hydration'
            probe=dict(state,X=cell,**{name:2})
            assert d.step_model(model,probe,0,'factorized')['state'][name]==spaces[name].ns-1
            assert d.step_model(model,probe,1,'factorized')['state'][name]==probe[name]
            assert d.step_model(model,probe,4,'factorized')['state'][name]==probe[name]-1
            dead=dict(state,**{name:0})
            after=d.step_model(model,dead,3,'factorized')['state']
            assert all(after[n]==dead[n] for n in ['X','box_1','box_2','box_3'])
