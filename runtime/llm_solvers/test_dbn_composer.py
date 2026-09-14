import copy
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbn_composer as d
from dbn_product import dbn_to_joint_operator


def setup():
    x = SimpleNamespace(name="X", ns=2, na=2, P_a_s_s=np.array([[[.3,.7],[.6,.4]],np.eye(2)]))
    bit = lambda name: SimpleNamespace(name=name, ns=2, na=2, default_action_ind=0,
                                       P_a_s_s=np.array([np.eye(2), [[0,1],[0,1]]]))
    spaces = {"X": x, "A": bit("A"), "B": bit("B")}
    spec = {"version":1, "spaces":list(spaces), "features":{"trigger":{"or":[{"var":"X","in":[1]}, {"var":"action","in":[0]}]}},
            "affordances":[{"target":n,"rules":[{"when":{"feature":"trigger"},"alpha":1}]} for n in ["A","B"]]}
    return spaces, spec


def test_every_product_row_against_independent_formula():
    spaces, spec = setup()
    ordered, aff, _ = d.compile_model(spaces,spec)
    joint = d.dense(dbn_to_joint_operator(ordered,aff,na=2,verbose=False))
    expected = np.zeros((2,8,8))
    for a,x,b,c in itertools.product(range(2),repeat=4):
        for nx in range(2):
            triggered = x == 1 or a == 0
            dest = np.ravel_multi_index((nx,1 if triggered else b,1 if triggered else c),(2,2,2))
            expected[a,4*x+2*b+c,dest] = spaces['X'].P_a_s_s[a,x,nx]
    np.testing.assert_allclose(joint,expected)
    out = d.evaluate(spaces,spec,{"X":1},0,True)
    assert out['product']['max_row_sum_error'] < 1e-12
    assert out['product']['selected_row_error'] < 1e-12


def test_conflict_rejected_with_witness_and_explicit_priority():
    spaces,spec = setup()
    spec['affordances'][0]['rules'].append({'when':{'var':'X','in':[1]},'alpha':0})
    with pytest.raises(ValueError,match='conflicting actions.*example'):
        d.compile_model(spaces,spec)
    spec['affordances'][0]['merge']='priority'
    result=d.evaluate(spaces,spec,{'X':1})
    assert result['alphas']['A']==1
    assert result['affordances'][0]['conflicting_assignments']==2


def test_same_action_overlap_and_default():
    spaces,spec=setup()
    spec['affordances'][0]['rules'].append({'when':{'var':'X','in':[1]},'alpha':1})
    assert d.evaluate(spaces,spec,{'X':0},1)['alphas']['A']==0
    assert d.evaluate(spaces,spec,{'X':1},1)['alphas']['A']==1


def test_explicit_h_roundtrip_and_reject_stochastic_alpha():
    spaces,spec=setup()
    _,_,reports=d.compile_model(spaces,spec)
    raw=reports[0]
    spec['affordances'][0]={'target':'A','parents':raw['parents'],'has_action':True,
                            'H_psi':raw['H_psi']['values'],'H_alpha':raw['H_alpha']['values']}
    assert d.evaluate(spaces,spec,{'X':1})['alphas']['A']==1
    spec['affordances'][0]['H_alpha']=[[.5,.5],[.5,.5]]
    with pytest.raises(ValueError,match='select one alpha'):
        d.compile_model(spaces,spec)


@pytest.mark.parametrize('change,match',[
    (lambda s:s['affordances'].append(copy.deepcopy(s['affordances'][0])),'duplicate target'),
    (lambda s:s['affordances'][0].update(alpha_parents=['B']),'Unknown affordance fields'),
    (lambda s:s['features'].update(trigger={'feature':'trigger'}),'cyclic feature'),
    (lambda s:s['affordances'][0]['rules'][0].update(alpha=-1),'integer'),
    (lambda s:s['affordances'][0].update(parents=[]),'missing rule parents'),
])
def test_invalid_specs(change,match):
    spaces,spec=setup();change(spec)
    with pytest.raises(ValueError,match=match): d.compile_model(spaces,spec)


def test_state_feedback_is_valid_and_and_not_work():
    spaces,spec=setup()
    spec['affordances']=[{'target':n,'rules':[{'when':{'and':[{'var':other,'in':[1]}, {'not':{'var':'X','in':[1]}}]},'alpha':1}]} for n,other in [('A','B'),('B','A')]]
    out=d.evaluate(spaces,spec,{'A':1,'B':0,'X':0})
    assert out['alphas']=={'X':0,'A':0,'B':1}


def test_custom_factor_and_limits(monkeypatch):
    spaces,spec=setup()
    spec['custom_spaces']=[{'name':'C','P':[np.eye(2).tolist()]}]
    spec['spaces'].append('C')
    assert d.evaluate(spaces,spec,build=True)['joint_states']==16
    monkeypatch.setattr(d,'MAX_STATES',5)
    assert not d.evaluate(spaces,spec)['can_build']
    with pytest.raises(ValueError,match='exceeds tutorial limits'): d.evaluate(spaces,spec,build=True)
    monkeypatch.setattr(d,'MAX_H_ENTRIES',2)
    with pytest.raises(ValueError,match='budget'): d.compile_model(spaces,spec)


def test_tutorial_catalog_and_product():
    import tutorial_server as t
    world=t.normalize_world({'rows':5,'cols':5,'walls':[6], 'p_main':.8,
                             'internal_goalstates_2_type_ind':{'8':0},'boolean_states_2_type_ind':{'9':1}})
    cat=t.dbn_panel_data(world,{'operation':'catalog'})
    out=t.dbn_panel_data(world,{'operation':'build','spec':cat['spec'],'state':{'X':9,'key':0,'hunger':4}})
    assert out['alphas']['key']==1
    assert out['product']['max_row_sum_error']<1e-12


def test_full_validation_detects_wrong_probabilities():
    spaces,spec=setup()
    ordered,aff,_=d.compile_model(spaces,spec)
    joint=dbn_to_joint_operator(ordered,aff,na=2,verbose=False)
    assert d.validate_joint(ordered,aff,joint)['rows_checked']==16
    corrupted=joint.copy()
    corrupted.data[0] += .01
    with pytest.raises(ValueError,match='differs from factorization'):
        d.validate_joint(ordered,aff,corrupted)


def test_both_execution_modes_eat_decay_and_invalidate_cache():
    import tutorial_server as t
    world=t.normalize_world({'rows':7,'cols':11,'p_main':1.,'nis':30,
        'start':10,'internal_goalstates_2_type_ind':{'10':0,'66':1}})
    cat=t.dbn_panel_data(world,{'operation':'catalog'})
    spec=cat['spec']; state={'X':10,'hunger':2,'hydration':20}
    result=t.dbn_panel_data(world,{'operation':'build','spec':spec,'state':state})
    assert result['product']['verification']['rows_checked']==6*69300
    assert result['product']['retained_for_play']
    for mode in ['factorized','tensor']:
        r=t.dbn_panel_data(world,{'operation':'step','mode':mode,'spec':spec,'state':state,'action':0})
        assert r['state']=={'X':10,'hunger':29,'hydration':19}
        assert r['world_state']['internal_state_values']=={0:29,1:19}
        r=t.dbn_panel_data(world,{'operation':'step','mode':mode,'spec':spec,'state':state,'action':3})
        assert r['state']=={'X':10,'hunger':1,'hydration':19}  # right edge: stays put
    changed=copy.deepcopy(spec);changed['affordances'][0].update(rules=[],placement_managed=False)
    with pytest.raises(ValueError,match='Build sparse product'):
        t.dbn_panel_data(world,{'operation':'step','mode':'tensor','spec':changed,'state':state,'action':0})


def test_noisy_modes_match_transition_probabilities_statistically():
    spaces,spec=setup();model={}
    d.evaluate(spaces,spec,build=True,retain=model)
    for mode in ['factorized','tensor']:
        rng=np.random.default_rng(81)
        samples=[d.step_model(model,{'X':0,'A':0,'B':0},0,mode,rng)['state'] for _ in range(2000)]
        assert all(s['A']==s['B']==1 for s in samples)
        assert abs(np.mean([s['X']==1 for s in samples])-.7)<.04


def test_identity_mode_or_rule_all_states_both_executors():
    spaces,spec=setup()
    spec['x_identity_when']={'or':[{'var':'A','in':[0]},{'var':'B','in':[0]}]}
    model={}
    out=d.evaluate(spaces,spec,build=True,retain=model)
    assert out['product']['verification']['max_probability_error']==0
    assert out['product']['verification']['support_matches']
    for a,b,x,action in itertools.product(range(2),repeat=4):
        state={'X':x,'A':a,'B':b}
        for mode in ['factorized','tensor']:
            result=d.step_model(model,state,action,mode,np.random.default_rng(2))
            assert result['x_dynamics']==('identity' if a==0 or b==0 else 'normal')
            if a==0 or b==0:
                assert result['state']['X']==x
        preview=d.evaluate(spaces,spec,state,action)
        if a==0 or b==0:
            assert preview['local_next']['X']==[[x,1.]]
        else:
            row=np.zeros(2)
            for j,p in preview['local_next']['X']: row[j]=p
            np.testing.assert_allclose(row,spaces['X'].P_a_s_s[action,x])


def test_identity_mode_timing_and_recovery():
    import tutorial_server as t
    world=t.normalize_world({'rows':5,'cols':5,'p_main':1.,'nis':4,'internal_goalstates_2_type_ind':{'12':0}})
    spec=t.dbn_panel_data(world,{'operation':'catalog'})['spec']
    spec['x_identity_when']={'var':'hunger','in':[0]}
    model={};d.evaluate(t.build_inspect_spaces(world),spec,build=True,retain=model)
    for mode in ['factorized','tensor']:
        r=d.step_model(model,{'X':11,'hunger':1},3,mode)
        assert r['state']=={'X':12,'hunger':0}  # current-state gate: this move completes
        r=d.step_model(model,r['state'],3,mode)
        assert r['state']=={'X':12,'hunger':0}  # next move is frozen
        r=d.step_model(model,r['state'],0,mode)
        assert r['state']=={'X':12,'hunger':3}  # physiology has its own refill rule
        r=d.step_model(model,r['state'],3,mode)
        assert r['state']=={'X':13,'hunger':2}
