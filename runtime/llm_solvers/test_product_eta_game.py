"""Full eta through a real tutorial DBN, checked without using its transition rows.

The oracle below spells out game rules directly. It neither evaluates H nor reads
local/joint kernels. Forward trajectories are merged only when state AND time
agree, preserving terminal joint states and exact arrival-time probabilities.
"""
from collections import defaultdict
import itertools
import numpy as np
import pytest
from test_tutorial_workshop import server, ta, d
import product_solver as ps

MOVES = {2:(1,0), 3:(0,1), 4:(-1,0), 5:(0,-1)}
NAMES = ('X','hunger','hydration','key','door_1','wood','canteen','canteen_water','resource_1')
SIZES = (4,4,4,2,2,2,2,2,2)


def next_cell(x,a):
    r,c=divmod(x,2); dr,dc=MOVES[a]
    return (r+dr)*2+c+dc if 0<=r+dr<2 and 0<=c+dc<2 else x


def independent_step(state,a):
    """Literal rules for this test level, using only the old joint state."""
    s=dict(zip(NAMES,state)); out=dict(s); x=s['X']
    if a in MOVES:
        out['hunger']=max(0,s['hunger']-1)
        out['hydration']=max(0,s['hydration']-1)
    if a==0 and x==0:
        out['hunger']=3
        out['key']=1
        if s['resource_1'] and not s['wood']:
            out['resource_1']=0; out['wood']=1
    if a==0 and x==1:
        out['hydration']=3
        if s['key']: out['door_1']=1
        if s['wood'] and not s['canteen']:
            out['wood']=0; out['canteen']=1
    if a==7 and x==1 and s['canteen'] and not s['canteen_water'] and s['hydration']>0:
        out['canteen_water']=1
    if a==6 and s['canteen'] and s['canteen_water'] and s['hydration']>0:
        out['canteen_water']=0; out['hydration']=3
    destinations={x:1.}
    if a in MOVES and s['hunger']>0 and s['hydration']>0:
        target=next_cell(x,a)
        if target==3 and x!=3 and not s['door_1']: target=x
        if target!=x: destinations={x:.25,target:.75}
    return {tuple(dict(out,X=j)[n] for n in NAMES):p for j,p in destinations.items()}


def is_goal(s):
    v=dict(zip(NAMES,s))
    return (v['X']==3 and v['hunger']>0 and v['hydration']==3 and
            v['key']==v['door_1']==v['canteen']==1 and v['canteen_water']==0)


@pytest.fixture(scope='module')
def game():
    # Tree/key/wood at 0; lake/canteen forge at 1; locked destination at 3.
    kernel={}
    for x,a in itertools.product(range(4),MOVES):
        target=next_cell(x,a)
        kernel.setdefault(x,{})[a]={x:1.} if x==target else {x:.25,target:.75}
    w=server.normalize_world(dict(rows=2,cols=2,start=0,p_main=1,nis=4,kernel=kernel,
        internal_goalstates_2_type_ind={0:0,1:1},internal_state_values={0:3,1:3},
        boolean_states_2_type_ind={0:1},doors=[dict(id=1,cell=3,open=False)],
        workshop=dict(objects=[dict(id=1,kind='wood',cell=0,quantity=1),
                               dict(id=2,kind='forge',recipe='canteen',cell=1)],
                      wood_capacity=1,water_capacity=1,movement_metabolism=True)))
    spec=ta.prepare(w,{})
    door=next(a for a in spec['affordances'] if a['target']=='door_1')
    door.update(placement_managed=False,rules=[dict(alpha=1,when={'and':[
        {'var':'X','in':[1]},{'var':'action','in':[0]},{'var':'key','in':[1]}]})])
    spec['x_identity_when']={'or':[{'var':'hunger','in':[0]},{'var':'hydration','in':[0]}]}
    # Pin ordering to make the oracle's independent tuple indexing transparent.
    spec['spaces']=list(NAMES)
    server.warm_solver(); model={}
    d.evaluate(server.build_inspect_spaces(w),spec,build=True,retain=model)
    assert tuple(s.ns for s in model['spaces'])==SIZES
    states=list(itertools.product(*(range(n) for n in SIZES)))
    index={s:i for i,s in enumerate(states)}
    g=np.array([[float(is_goal(s)) for s in states]]*8)
    c=np.array([[float(s[1]>0 and s[2]>0) for s in states]]*8)
    return model,states,index,g,c


def test_every_game_transition_matches_independent_rules(game):
    model,states,index,_,_=game
    for a,P in enumerate(model['csr']):
        for i,s in enumerate(states):
            expected={index[t]:p for t,p in independent_step(s,a).items()}
            row=P.getrow(i)
            actual=dict(zip(row.indices,row.data))
            assert actual.keys()==expected.keys(),(a,s,actual,expected)
            for j,p in expected.items(): assert actual[j]==pytest.approx(p,abs=1e-14)


def enumerate_termination(start,pi,states,index,seeds,gates,horizon):
    outcomes=[]; visited_actions=set()
    for seed,gate in zip(seeds,gates):
        frontier={start:1.}; terminal=defaultdict(float)
        for time in range(horizon):
            following=defaultdict(float)
            for s,mass in frontier.items():
                i=index[s]
                if seed[i]: terminal[(i,time)]+=mass*seed[i]
                if not gate[i]: continue
                a=int(pi[i]); visited_actions.add(a)
                for nxt,p in independent_step(s,a).items():
                    following[nxt]+=mass*gate[i]*p
            frontier=following
            if not frontier: break
        outcomes.append(dict(terminal))
    return outcomes,visited_actions


def stored_row(solution,start,outcome):
    offset=0; result={}
    while True:
        page=ps.termination_row(solution,start,outcome,offset,limit=23)
        result.update({(e['terminal'],e['time']):e['probability'] for e in page['entries']})
        if page['next_offset'] is None:return result
        offset=page['next_offset']


def assert_rows(actual,expected):
    for key in actual.keys()|expected.keys():
        assert actual.get(key,0)==pytest.approx(expected.get(key,0),abs=2e-11),key


@pytest.mark.parametrize('algorithm,theta',
    [('feas_eta_fast',None),('feas_eta_combined',None)] +
    [(a,t) for a in ('feas_eta_fast_thresh','feas_eta_combined_thresh') for t in (.1,.3,.9)])
def test_full_game_eta_against_forward_trajectories(game,algorithm,theta):
    model,states,index,g,c=game
    solution=ps.solve(model,g,c,algorithm,{'risk_thresh':theta} if theta is not None else {})
    k=solution['fields']['kappa']; pi=solution['pi']; live=(k>0).astype(float)
    if algorithm.endswith('thresh'):
        # Establish the risk gate independently by enumerating the preliminary
        # policy's failure trajectories, not by reading its stored eta totals.
        preliminary=ps.solve(model,g,c,'feas_eta_fast',{})
        live1=(preliminary['fields']['kappa']>0).astype(float)
        horizon=preliminary['eta_neg'].shape[2]
        failure=np.zeros(len(states))
        seed1=(1-live1)+live1*(1-c[0])
        gate1=live1*(1-g[0])*c[0]
        for i,s in enumerate(states):
            result,_=enumerate_termination(s,preliminary['pi'],states,index,
                                          (seed1,),(gate1,),horizon)
            failure[i]=sum(result[0].values())
        safe=(failure<=theta).astype(float)
        f=g[0]*safe; h=(1-g[0])*safe; constraint=np.maximum(c[0],safe)
    else:
        f=g[0]*c[0]; h=(1-g[0])*c[0]; constraint=c[0]
    seeds=(live*f,live*(1-constraint)+(1-live))
    gates=(live*h if 'combined' in algorithm else h,live*h)
    starts=[(0,3,3,0,0,0,0,0,1), # collect, craft, unlock, fill, travel, drink
            (1,2,3,1,0,1,0,0,0), # ingredients but not yet crafted/unlocked
            (1,2,3,1,1,0,1,1,0), # ready to travel; slipping costs physiology
            (3,1,2,1,1,0,1,1,0), # drink from carried water at destination
            (3,1,3,1,1,0,1,0,0), # immediate success at t=0
            (1,0,2,1,1,0,1,1,0), # immediate physiological failure
            (0,3,3,0,0,0,0,0,0)] # resources exhausted: unreachable objective
    actions=set()
    for s in starts:
        expected,used=enumerate_termination(s,pi,states,index,seeds,gates,solution['eta_pos'].shape[2])
        actions|=used
        for outcome,entries in zip(('positive','negative'),expected):
            assert_rows(stored_row(solution,index[s],outcome),entries)
    # Check that this is actually testing the crafting/fill/drink policy path.
    if theta != .1: assert {0,2,3,6,7} <= actions
    success=stored_row(solution,index[starts[0]],'positive')
    failure=stored_row(solution,index[starts[0]],'negative')
    assert failure
    if theta == .1:
        assert not success  # The initial state exceeds this strict risk budget.
    else:
        assert success
    if theta is None:
        assert len({time for _,time in success})>1
    elif theta == .3:
        assert {time for _,time in success}=={6}
    assert all(is_goal(states[j]) for j,_ in success)


def test_successful_game_trajectory_exercises_all_exchanges(game):
    model,states,index,_,_=game
    state=(0,3,3,0,0,0,0,0,1)
    expected=[(0,3,3,1,0,1,0,0,0), # collect wood/key and eat
              (1,2,2,1,0,1,0,0,0), # move to forge/lake
              (1,2,3,1,1,0,1,0,0), # craft, unlock, drink directly
              (1,2,3,1,1,0,1,1,0), # fill canteen
              (3,1,2,1,1,0,1,1,0), # pass through open door
              (3,1,3,1,1,0,1,0,0)] # drink carried water at destination
    for a,nxt in zip([0,3,0,7,2,6],expected):
        probability=independent_step(state,a)[nxt]
        assert model['csr'][a][index[state],index[nxt]]==probability
        state=nxt
    assert is_goal(state)


def independently_score_game_policy(game, solution):
    """Enumerate episode outcomes, not eta seeds/gates or stored transition rows.

    The option terminates on success, physiological death, or zero feasibility
    (the solver's unsuccessful-stop boundary). Require no unaccounted tail mass.
    """
    _,states,index,_,_=game
    start=(0,3,3,0,0,0,0,0,1)
    frontier={start:1.}; won=defaultdict(float); lost=defaultdict(float)
    for time in range(solution['eta_pos'].shape[2]):
        following=defaultdict(float)
        for s,mass in frontier.items():
            i=index[s]
            if is_goal(s):
                won[i,time]+=mass
            elif s[1]==0 or s[2]==0 or solution['fields']['kappa'][i]==0:
                lost[i,time]+=mass
            else:
                for nxt,p in independent_step(s,int(solution['pi'][i])).items():
                    following[nxt]+=mass*p
        frontier=following
        if not frontier:break
    assert sum(frontier.values())<1e-12, 'Unresolved trajectories cannot be counted as wins'
    assert sum(won.values())+sum(lost.values())==pytest.approx(1,abs=1e-12)
    return won,lost


@pytest.mark.parametrize('algorithm',['feas_eta_fast_thresh','feas_eta_combined_thresh'])
def test_agent_wins_within_final_policy_risk_budget(game,algorithm):
    model,states,index,g,c=game
    theta=.5
    solution=ps.solve(model,g,c,algorithm,{'risk_thresh':theta})
    wins,failures=independently_score_game_policy(game,solution)
    win_probability=sum(wins.values()); failure_probability=sum(failures.values())
    assert win_probability==pytest.approx(45/64,abs=1e-12)
    assert failure_probability==pytest.approx(19/64,abs=1e-12)
    assert failure_probability<=theta
    assert win_probability>=1-theta
    start=index[(0,3,3,0,0,0,0,0,1)]
    assert_rows(stored_row(solution,start,'positive'),wins)
    assert_rows(stored_row(solution,start,'negative'),failures)


@pytest.mark.xfail(strict=True, reason='Preliminary-policy risk gate does not bound final-policy failure: 0.4375 > 0.3')
@pytest.mark.parametrize('algorithm',['feas_eta_fast_thresh','feas_eta_combined_thresh'])
def test_final_policy_respects_tighter_risk_budget_known_counterexample(game,algorithm):
    model,_,_,g,c=game
    theta=.3
    solution=ps.solve(model,g,c,algorithm,{'risk_thresh':theta})
    wins,failures=independently_score_game_policy(game,solution)
    assert sum(wins.values())>0
    assert sum(failures.values())<=theta
