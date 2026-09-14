import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
import product_solver as ps


def test_values_and_policy_on_chain():
    P = [sparse.csr_matrix([[0,1,0], [0,0,1], [0,0,1]]), sparse.eye(3, format='csr')]
    g = np.array([[0,0,1], [0,0,1.]])
    for algorithm, expected in [('ssp', [-2,-1,0]), ('ihdc', [-1.95,-1,0])]:
        value, pi, info = ps.value_iteration(P, g, np.ones_like(g), algorithm, {})
        np.testing.assert_allclose(value, expected)
        assert pi[:2].tolist() == [0,0]


def test_first_exit_traps_rare_failure_and_escape_action():
    # State 0 can reach the goal but a tiny probability of a permanent trap
    # makes expected first-exit cost infinite. No numerical near-one shortcut.
    p = sparse.csr_matrix([[0,1-1e-14,1e-14], [0,1,0], [0,0,1]])
    g = np.array([[0,1,0.]])
    v, pi, _ = ps.value_iteration([p], g, np.ones_like(g), 'ssp', {})
    assert np.isneginf(v[[0,2]]).all() and pi[0] == -1
    # An alternative action safely escapes 0, but not 2.
    escape = sparse.csr_matrix([[0,1,0],[0,1,0],[0,0,1]])
    v, pi, _ = ps.value_iteration([p,escape], np.repeat(g,2,axis=0), np.ones((2,3)), 'ssp', {})
    assert v[0] == -1 and pi[0] == 1 and np.isneginf(v[2])


def test_first_exit_geometric_loop_and_action_goal():
    P = [sparse.csr_matrix([[.5,.5],[1,0]]), sparse.eye(2, format='csr')]
    g = np.array([[0,0],[0,1.]])
    v, pi, _ = ps.value_iteration(P, g, np.ones_like(g), 'ssp', {})
    np.testing.assert_allclose(v, [-2,0], atol=1e-7)
    assert pi.tolist() == [0,1]


@pytest.mark.parametrize('algorithm', ['feas2s', 'feas_eta_fast', 'feas_eta_combined'])
def test_product_retains_terminal_states_and_arrival_times(algorithm):
    # Success reaches state 3 after two steps; failure reaches state 2 after one.
    P = [sparse.csr_matrix([[0,.6,.4,0], [0,0,0,1], [0,0,1,0], [0,0,0,1]])]
    g = np.array([[0,0,0,1.]])
    c = np.array([[1,1,0,1.]])
    result = ps.solve({'csr':P}, g, c, algorithm, {})
    positive = np.zeros(result['eta_pos'].shape)
    negative = np.zeros(result['eta_neg'].shape)
    positive[0,3,2]=.6; positive[1,3,1]=1; positive[3,3,0]=1
    negative[0,2,1]=.4; negative[2,2,0]=1
    np.testing.assert_allclose(result['eta_pos'].to_dense(),positive,atol=1e-13)
    np.testing.assert_allclose(result['eta_neg'].to_dense(),negative,atol=1e-13)
    row=ps.termination_row(result,0)
    assert row['entries']==[{'terminal':3,'time':2,'probability':.6}]
    assert ps.termination_row(result,0,'negative')['entries']==[{'terminal':2,'time':1,'probability':.4}]


@pytest.mark.parametrize('algorithm', ['feas_eta_fast', 'feas_eta_combined'])
def test_product_eta_limit_errors_instead_of_reducing_to_totals(monkeypatch,algorithm):
    monkeypatch.setattr(ps,'MAX_ETA_NNZ',1)
    with pytest.raises(ValueError,match='No outcome-total approximation'):
        ps.solve({'csr':[sparse.eye(3,format='csr')]},np.ones((1,3)),np.ones((1,3)),algorithm,{})


@pytest.mark.parametrize('theta', [.1,.5,.9])
def test_threshold_policy_matches_existing_fast_solver(theta):
    from feas_eta_fast import feas_iter_fast_eta_thresh
    P = [sparse.csr_matrix([[0,.8,.2],[0,1,0],[0,0,1]]), sparse.eye(3,format='csr')]
    g = np.array([[0,1,0],[0,1,0.]])
    c = np.array([[1,1,0],[1,1,0.]])
    reference = feas_iter_fast_eta_thresh(P,g,c,risk_thresh=theta,round_decimals=3,ET_epsilon=1e-5)
    result = ps.solve({'csr':P},g,c,'feas_eta_fast_thresh',{'risk_thresh':theta,'round_decimals':3})
    for key,index in [('eta_pos',2),('eta_neg',3)]:
        np.testing.assert_allclose(result[key].to_dense(),reference[index].to_dense(),atol=1e-13)
    np.testing.assert_array_equal(result['pi'],reference[1])
    np.testing.assert_allclose(result['fields']['kappa'],reference[0],atol=1e-13)


def test_product_slice_order_gradient_and_joint_action_changes():
    spaces = [SimpleNamespace(name='X',ns=6),SimpleNamespace(name='hunger',ns=3),SimpleNamespace(name='key',ns=2)]
    dims=(6,3,2)
    coords=np.array(np.unravel_index(np.arange(36),dims))
    v = coords[0]%3 + 10*(coords[0]//3) + 100*coords[1] + 1000*coords[2]
    # Action changes the key, not the grid. The conditional spatial slope
    # must not confuse this 1000-unit change with its 1-unit x gradient.
    dest=np.ravel_multi_index((coords[0],coords[1],1-coords[2]),dims)
    P=sparse.csr_matrix((np.ones(36),(np.arange(36),dest)),shape=(36,36))
    model={'spaces':spaces,'csr':[P]}
    solution={'fields':{'value':v.astype(float)},'pi':np.zeros(36,dtype=int),'g':np.zeros((1,36))}
    result=ps.slice_solution(model,solution,{'hunger':2,'key':0},1,2,3)
    assert result['values']==[200,201,202,210,211,212]
    assert result['gradient'][1]==[1,10]
    assert result['actions'][0]['delta']==1000
    with pytest.raises(ValueError,match='hunger'):
        ps.slice_solution(model,solution,{'hunger':3},0,2,3)


def test_product_task_death_constraint_and_action_specific_grid_goal():
    model={'spaces':[SimpleNamespace(name='X',ns=2),SimpleNamespace(name='hunger',ns=3)],'csr':[sparse.eye(6)]*2}
    world={'goals':[{'id':1,'cells':[{'li':1,'a':1}]}],'constraints':[]}
    g,c=ps.objective(model,world,{'group':1},death_names=['hunger'])
    assert np.flatnonzero(g).tolist()==[10,11]
    g,c=ps.objective(model,world,{'objective':'task'},[[['hunger','>=',2],['X','==',1]]],['hunger'])
    assert np.flatnonzero(g).tolist()==[5,11]


def test_server_build_solve_slice_and_stale_cache():
    import tutorial_server as server
    world=server.normalize_world({'rows':3,'cols':3,'start':0,'goals':[{'id':1,'cells':[{'li':8,'a':None}]}]})
    catalog=server.dbn_panel_data(world,{'operation':'catalog'})
    spec=catalog['spec']
    request={'operation':'solve','spec':spec,'algorithm':'ssp','options':{'group':1}}
    result=server.dbn_panel_data(world,request)
    assert result['values'][0]==-4
    assert result['values'][8]==0
    slice_request={'operation':'slice','spec':spec,'solution_id':result['solution_id'],'fixed':{},'cell':1}
    assert server.dbn_panel_data(world,slice_request)['value']==-3
    changed=dict(world,p_main=.8)
    with pytest.raises(ValueError,match='changed'):
        server.dbn_panel_data(changed,slice_request)


def test_server_product_eta_query_and_stale_solution():
    import tutorial_server as server
    world=server.normalize_world({'rows':2,'cols':2,'start':0,'p_main':1,
                                  'goals':[{'id':1,'cells':[{'li':3,'a':None}]}]})
    spec=server.dbn_panel_data(world,{'operation':'catalog'})['spec']
    result=server.dbn_panel_data(world,{'operation':'solve','spec':spec,
        'algorithm':'feas_eta_fast','options':{'group':1}})
    query={'operation':'eta','spec':spec,'solution_id':result['solution_id'],'start':0}
    row=server.dbn_panel_data(world,query)
    assert row['entries']==[{'terminal':3,'time':2,'probability':1.0}]
    assert row['spaces']==[{'name':'X','ns':4}]
    with pytest.raises(ValueError,match='no longer cached'):
        server.dbn_panel_data(world,dict(query,solution_id='outdated'))


def test_eta_row_paging_keeps_arrival_times():
    from eta_fast import build_eta
    P=sparse.csr_matrix([[.5,.5],[0,1.]])
    eta=build_eta(P,(np.array([0.,1.]),np.zeros(2)),
                  (np.array([1.,0.]),np.array([1.,0.])),8,8,0,mode=1)[0]
    solution={'eta_pos':eta}
    first=ps.termination_row(solution,0,limit=2)
    second=ps.termination_row(solution,0,offset=first['next_offset'],limit=2)
    assert [e['time'] for e in first['entries']+second['entries']]==[1,2,3,4]
    assert [e['probability'] for e in first['entries']]==[.5,.25]
    assert first['total_entries']==7


@pytest.mark.parametrize('combined',[False,True])
def test_stochastic_full_eta_storage_limit(combined):
    from eta_fast import build_eta
    from eta_combined import build_combined_eta
    build=build_combined_eta if combined else build_eta
    P=sparse.csr_matrix([[.5,.5],[0,1.]])
    with pytest.raises(ValueError,match='No outcome-total approximation'):
        build(P,(np.array([0.,1.]),np.zeros(2)),
              (np.array([1.,0.]),np.array([1.,0.])),8,8,0,mode=1,max_nnz=2)
