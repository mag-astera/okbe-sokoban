"""Moved sources update both refill and complementary hold H predicates."""
import json
from pathlib import Path
import pytest
from test_tutorial_workshop import server, ta, d


@pytest.mark.parametrize('level', ['progressive_11', 'progressive_13'])
def test_moved_lake_and_tree_follow_placements(level, tmp_path, monkeypatch):
    path = next((Path(__file__).resolve().parents[1] / 'tutorial/levels/progressive_levels').glob(level+'*.json'))
    record = json.loads(path.read_text())
    w = server.normalize_world(record['world'])
    old_lake = min(c for c,t in w['internal_goalstates_2_type_ind'].items() if t == 1)
    old_tree = 1
    for old, new in [(old_lake, old_lake-1), (old_tree, 0)]:
        w['internal_goalstates_2_type_ind'][new] = w['internal_goalstates_2_type_ind'].pop(old)
    # Save must repair the old specification without regenerating/resetting the world.
    monkeypatch.setattr(server, 'LEVELDIR', str(tmp_path))
    saved = server.save_level(w, name='moved', tutorial=record['tutorial'])
    spec = saved['tutorial']['spec']
    assert spec == ta.prepare(w, spec)
    spaces = server.build_inspect_spaces(w)
    ordered, affs, _ = d.compile_model(spaces, spec)
    model = dict(spaces=ordered, affs=affs)
    state = server.inspect_x0(w, spaces)
    for name, old, new in [('hydration',old_lake,old_lake-1), ('hunger',old_tree,0)]:
        for cell, expected in [(old,2), (new,15)]:
            before = dict(state, X=cell, **{name:2})
            assert d.step_model(model,before,0,'factorized')['state'][name] == expected
            assert d.step_model(model,before,1,'factorized')['state'][name] == 2
            assert d.step_model(model,before,3,'factorized')['state'][name] == 1
