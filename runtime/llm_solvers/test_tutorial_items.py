"""Moving item placements must preserve the state identities used by H and tasks."""
import json
from pathlib import Path
from test_tutorial_workshop import server, ta, d
import tutorial_items


def test_moved_key_keeps_pickup_and_door_association():
    path = next((Path(__file__).resolve().parents[1] / 'tutorial/levels/progressive_levels').glob('progressive_12*.json'))
    record = json.loads(path.read_text())
    world = server.normalize_world(record['world'])
    # Move red past the other keys in spatial ordering, then swap with purple.
    world['boolean_states_2_type_ind'] = {21: 1, 74: 1, 79: 1, 80: 1}
    world['item_names'] = {21: 'key_2', 74: 'key_3', 79: 'key_4', 80: 'key_1'}
    world = server.normalize_world(json.loads(json.dumps(world)))
    assert {s['cell']: s['name'] for s in tutorial_items.item_specs(world)} == world['item_names']
    spaces = server.build_inspect_spaces(world)
    spec = ta.prepare(world, record['tutorial']['spec'])
    ordered, affs, _ = d.compile_model(spaces, spec)
    model = dict(spaces=ordered, affs=affs)
    state = server.inspect_x0(world, spaces)
    picked = d.step_model(model, dict(state, X=80), 0, 'factorized')['state']
    assert picked['key_1'] == 1
    assert picked['key_4'] == 0
    opened = d.step_model(model, dict(picked, X=36), 0, 'factorized')['state']
    assert opened['door_3'] == 1  # red door
    wrong = d.step_model(model, dict(state, X=36, key_4=1), 0, 'factorized')['state']
    assert wrong['door_3'] == 0
    world['item_held'] = {80: 1}
    assert server.inspect_x0(world, server.build_inspect_spaces(world))['key_1'] == 1
