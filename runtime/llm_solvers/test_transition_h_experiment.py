from itertools import product
from collections import Counter
import numpy as np
import pytest
from transition_h_experiment import fixture, Experiment
import dbn_composer as dbn


@pytest.mark.parametrize('success',[1.,.75,0.])
def test_every_entry_against_independent_joint_formula(success):
    exp=fixture(success);kernels=exp.build()
    for a,x,d in product(range(4),range(3),range(2)):
        intended=max(x-1,0) if a==2 else min(x+1,2) if a==3 else x
        if intended==2 and not d and x!=2:intended=x
        expected=np.zeros(6)
        for xn,mass in [(x,1-success),(intended,success)]:
            dn=1-d if x==1 and xn!=x else d
            expected[xn*2+dn]+=mass
        np.testing.assert_allclose(kernels[a].getrow(x*2+d).toarray()[0],expected,atol=1e-14,rtol=0)


def test_self_transitions_and_failed_moves_never_toggle():
    exp=fixture(.75)
    for action in (0,1,3):  # interact, wait, right into a closed door
        assert exp.row({'X':1,'door':0},action)=={2:1.}
    assert exp.row({'X':1,'door':0},2)=={1:.75,2:.25}
    # Product of separate marginals would wrongly invent these outcomes.
    assert 0 not in exp.row({'X':1,'door':0},2)
    assert 3 not in exp.row({'X':1,'door':0},2)


def test_sampling_preserves_realized_movement_correlation():
    exp=fixture(.75);rng=np.random.default_rng(818);counts=Counter()
    for _ in range(4000):
        state,trace=exp.sample({'X':1,'door':0},2,rng)
        assert state['door']==int(state['X']!=1)
        assert trace['door']['alpha']==(3 if state['X']!=1 else 0)
        counts[state['X']]+=1
    assert abs(counts[0]/4000-.75)<.03


def test_no_override_reproduces_existing_current_state_factorization():
    exp=fixture(.75)
    normal=Experiment(dict(spaces=exp.spaces,affs=exp.affs))
    for action,flat in product(range(4),range(6)):
        state=normal.state(flat)
        xrow=dbn.local_row(exp.spaces[0],exp.affs,state,action,action)
        expected=np.kron(xrow,np.eye(2)[state['door']])
        row=np.zeros(6)
        for n,p in normal.row(state,action).items():row[n]=p
        np.testing.assert_allclose(row,expected,atol=1e-14)


def test_limits_and_invalid_target():
    exp=fixture()
    with pytest.raises(ValueError):exp.build(max_states=5)
    with pytest.raises(ValueError):exp.build(max_nnz=1)
    with pytest.raises(ValueError):exp.row({'X':3,'door':0},0)
