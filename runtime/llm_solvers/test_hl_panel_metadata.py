"""Placement needs metadata only, even while a numerical solve holds the lock."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tutorial_server as ts
import world_spaces as ws


def test_metadata_matches_real_spaces_without_constructing_on_refresh(monkeypatch):
    world = ts.normalize_world({'rows':9, 'cols':9,
        'internal_goalstates_2_type_ind':{str(t):t for t in range(6)},
        'nis_by_type':{'0':8,'3':12}, 'internal_state_values':{'0':3},
        'boolean_states_2_type_ind':{'20':1}, 'item_held':{'20':1}})
    types, spaces = ws.internal_spaces_from_placement(world['internal_goalstates_2_type_ind'],world['nis'],world['nis_by_type'])
    def forbidden(*a, **kw):
        raise AssertionError('Placement must not construct numerical spaces')
    monkeypatch.setattr(ws, 'internal_spaces_from_placement', forbidden)
    result = ts.hl_panel_data(world)
    for row, t, space in zip(result['features'], types, spaces):
        assert (row['name'],row['ns'],row['na']) == (space.name,space.ns,space.na)
        assert row['value'] == (3 if t==0 else space.ns-1)
        assert row['has_death'] == (t not in ws.NO_DEATH_TYPES)
    assert result['items'][0]['held']


def test_metadata_can_finish_while_solve_holds_lock():
    for path in ['/api/hl_panel','/api/task_panel']:
        done = threading.Event()
        request = SimpleNamespace(path=path, _post_serial=done.set)
        with ts._compute_lock:
            worker=threading.Thread(target=ts.Handler.do_POST,args=(request,),daemon=True)
            worker.start()
            completed = done.wait(1)
        worker.join(1)
        assert completed, f'{path} queued behind a solve'
