"""Tutorial placement adapter: one independent bit per door, linked toggle actions.

Uses the original BitSpace HOLD/SET/CLEAR/TOGGLE convention. Button locations
are features of X, not additional state spaces. Door IDs survive moving/recoloring.
"""
from copy import deepcopy
import numpy as np

COLORS = ('white', 'red', 'blue', 'orange', 'purple')


def name(door):
    return f"door_{door['id']}"


def normalize(raw, n, walls):
    out = {}
    for kind in ('doors', 'buttons'):
        records, ids, cells = [], set(), set()
        for item in raw.get(kind, []):
            ident, cell = item.get('id'), item.get('cell')
            if type(ident) is not int or ident < 1 or ident in ids:
                raise ValueError(f'{kind}: IDs must be unique positive integers')
            if type(cell) is not int or not 0 <= cell < n or cell in cells or cell in walls:
                raise ValueError(f'{kind}: cells must be unique, in bounds, and outside fixed walls')
            color = item.get('color', 'purple')
            if color not in COLORS:
                raise ValueError(f'{kind}: unknown color {color}')
            row = dict(id=ident, cell=cell, color=color)
            if kind == 'doors':
                value = item.get('open', False)
                if type(value) not in (bool, int) or value not in (0, 1):
                    raise ValueError('Door open state must be a bit')
                row['open'] = bool(value)
            else:
                targets = item.get('targets')
                if targets is not None and (not isinstance(targets, list) or any(type(x) is not int for x in targets)):
                    raise ValueError('Button targets must be door IDs, or null for matching colors')
                row['targets'] = sorted(set(targets)) if targets is not None else None
                alpha = item.get('alpha', 3)
                effects = item.get('effects', {})
                if type(alpha) is not int or alpha not in range(4):
                    raise ValueError('Button alpha must be 0 hold, 1 open, 2 close, or 3 toggle')
                if not isinstance(effects, dict) or any(not str(k).isdigit() or type(v) is not int or v not in range(4) for k,v in effects.items()):
                    raise ValueError('Button effects must map door IDs to actions 0 through 3')
                transition_h = item.get('transition_h', False)
                if type(transition_h) is not bool:raise ValueError('Button transition_h must be a boolean')
                row['transition_h'] = transition_h
                required = item.get('required_actions')
                if required is not None and (not isinstance(required, list) or any(type(a) is not int or a not in range(6) for a in required)):
                    raise ValueError('Button required_actions must be null or a list of free actions 0 through 5')
                row.update(alpha=alpha, effects={str(k):v for k,v in effects.items()},
                           required_actions=sorted(set(required)) if required is not None else None)
            records.append(row); ids.add(ident); cells.add(cell)
        out[kind] = sorted(records, key=lambda r: r['id'])
    ids = {d['id'] for d in out['doors']}
    for b in out['buttons']:
        b['effects'] = {k:v for k,v in b['effects'].items() if int(k) in ids}
        if b['targets'] is not None and set(b['targets']) - ids:
            raise ValueError('A button links to a door that does not exist')
    if {d['cell'] for d in out['doors']} & {b['cell'] for b in out['buttons']}:
        raise ValueError('A button and door cannot occupy the same cell')
    return out


def targets(button, world):
    return [d['id'] for d in world.get('doors', [])
            if (d['color'] == button['color'] if button['targets'] is None else d['id'] in button['targets'])]


def catalog(world):
    return [dict(name=name(d), ns=2, na=4, owner='Doors',
                 state_labels=['closed', 'open'], action_labels=['hold', 'open', 'close', 'toggle'])
            for d in world.get('doors', [])]


def attach(spaces, world):
    from state_spaces import BitSpace
    doors = world.get('doors', [])
    for d in doors:
        spaces[name(d)] = BitSpace(name(d), 60, 400 + d['id'])
    spaces['X'].mode_definitions = [dict(name=f"door_{d['id']}_passage", kind='blocked_destination', cell=d['cell'],
        when={'var':name(d),'in':[0]}, labels=['passable','blocked']) for d in doors]


def augment_spec(spec, world):
    """Door affordances are managed by the HL linking controls; other rules survive."""
    spec = deepcopy(spec)
    if spec.get('version') != 1:
        return spec
    def managed(n):
        return isinstance(n, str) and n.startswith('door_') and n[5:].isdigit()
    spec['spaces'] = [n for n in spec.get('spaces', []) if not managed(n)]
    overrides = {r['target'] for r in spec.get('affordances', []) if managed(r.get('target')) and r.get('placement_managed') is False and r['target'] in {name(d) for d in world.get('doors', [])}}
    spec['affordances'] = [r for r in spec.get('affordances', []) if not managed(r.get('target')) or r.get('target') in overrides]
    for d in world.get('doors', []):
        spec['spaces'].append(name(d))
        if name(d) in overrides:
            continue
        def condition(button):
            location = {'var':'X', 'in':[button['cell']]}
            required = button.get('required_actions')
            parts = [location]
            if required is not None:parts.append({'var':'action','in':required})
            if button.get('transition_h', False):parts.append({'transition':'moved'})
            return parts[0] if len(parts)==1 else {'and':parts}
        rules = [{'when': condition(b),
                  'alpha':b.get('effects', {}).get(str(d['id']), b.get('alpha', 3))}
                 for b in world.get('buttons', []) if d['id'] in targets(b, world)]
        affordance = dict(target=name(d), default_alpha=0, merge='reject', rules=rules, placement_managed=True)
        if any(b.get('transition_h',False) for b in world.get('buttons',[]) if d['id'] in targets(b,world)):
            affordance['h_form']='transition'
        spec['affordances'].append(affordance)
    return spec


def current(world):
    return {name(d): int(d['open']) for d in world.get('doors', [])}


def dynamics(world):
    out = {k:v for k,v in world.items() if k not in ('start','internal_state_values','item_held','green_badgers')}
    if 'workshop' in out: out['workshop'] = {'objects':out['workshop'].get('objects',[])}
    out['doors'] = [{k:v for k,v in d.items() if k != 'open'} for d in world.get('doors', [])]
    out['boxes'] = [{'id':b['id']} for b in world.get('boxes',[])]
    return out


def block_row(row, x, doors, state):
    """A closed destination bounces to x. Closing around x still allows exit.

    All bits read time t; a button's toggles become visible on the next step.
    This also respects hand-edited/stochastic rows without enumerating modes.
    """
    row = np.array(row, dtype=float, copy=True)
    for door in doors:
        if not state[door['name']] and door['cell'] != x:
            row[x] += row[door['cell']]
            row[door['cell']] = 0
    return row
