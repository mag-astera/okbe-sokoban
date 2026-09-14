"""Translate tutorial placements into an editable DBN specification, never state updates."""
from copy import deepcopy
import world_spaces as ws
import tutorial_items
import tutorial_doors as doors


def legacy_placement_rule(aff):
    """Recognize the original tutorial's generated interact/refill/collect rule."""
    if 'placement_managed' in aff:
        return aff['placement_managed'] is True
    if aff.get('h_form') == 'transition':return False
    if aff.get('default_alpha', 0) != 0 or aff.get('merge', 'reject') != 'reject':
        return False
    rules = aff.get('rules', [])
    if len(rules) != 1 or rules[0].get('alpha') != 1:
        return False
    parts = rules[0].get('when', {}).get('and', [])
    return (len(parts) == 2 and {'var':'action','in':[0]} in parts
            and any(set(p) == {'var','in'} and p['var'] == 'X' for p in parts)
            and not any(k in aff for k in ('H_psi','H_alpha')))


def bind_physiological_sources(world, spec):
    """Refresh placement-bound refill predicates, including old Sokoban saves."""
    for aff in spec.get('affordances', []):
        name = aff.get('target')
        if name not in ('hunger', 'hydration'):
            continue
        rules = aff.get('rules', [])
        # Narrow migration of the generated two-rule movement-metabolism form.
        # Arbitrary authored H conditions are not spatially rewritten.
        legacy = False
        if world.get('workshop', {}).get('movement_metabolism') and len(rules) == 2:
            refill = rules[0].get('when', {})
            parts = refill.get('and', [])
            legacy = (aff.get('placement_managed') is False
                and aff.get('default_alpha', 0) == 0
                and aff.get('merge', 'reject') == 'reject'
                and len(parts) == 2 and parts[0] == {'var':'action','in':[0]}
                and set(parts[1]) == {'var','in'} and parts[1]['var'] == 'X'
                and rules[0].get('alpha') == 1
                and rules[1] == {'when':{'and':[{'var':'action','in':[0,1]}, {'not':refill}]}, 'alpha':2}
                and not any(k in aff for k in ('H_psi','H_alpha','h_form')))
        if aff.get('source_binding') != 'physiological_placement' and not legacy:
            continue
        aff['source_binding'] = 'physiological_placement'
        source_type = 0 if name == 'hunger' else 1
        cells = sorted(int(c) for c,t in world['internal_goalstates_2_type_ind'].items() if t == source_type)
        refill = {'and':[{'var':'action','in':[0]}, {'var':'X','in':cells}]}
        aff['rules'] = [{'when':refill, 'alpha':1},
            {'when':{'and':[{'var':'action','in':[0,1]}, {'not':deepcopy(refill)}]}, 'alpha':2}]
    return spec


def prepare(world, spec):
    spec = deepcopy(spec) if spec else {'version':1, 'spaces':['X'], 'features':{}, 'affordances':[], 'custom_spaces':[]}
    if spec.get('version') != 1:
        return spec
    bind_physiological_sources(world, spec)
    generated = {}
    def reach(name, cells):
        generated[name] = dict(target=name, default_alpha=0, merge='reject', placement_managed=True,
            rules=[{'when':{'and':[{'var':'action','in':[0]}, {'var':'X','in':sorted(cells)}]}, 'alpha':1}])
    for t in sorted(set(world['internal_goalstates_2_type_ind'].values())):
        reach(ws.TYPE_NAMES[t], [int(c) for c,v in world['internal_goalstates_2_type_ind'].items() if v == t])
    for item in tutorial_items.item_specs(world):
        reach(item['name'], [int(item['cell'])])
    existing = spec.get('affordances', [])
    # Only generated rules follow placement changes; explicitly authored rules survive.
    automatic = {a['target'] for a in existing if a.get('placement_managed') is True or
                 (a.get('target') in set(ws.TYPE_NAMES.values()) | set(generated) and legacy_placement_rule(a))}
    spec['affordances'] = [a for a in existing if a['target'] not in automatic]
    spec['spaces'] = [n for n in spec.get('spaces', ['X']) if n not in automatic or n in generated or n.startswith('door_')]
    for name, rule in generated.items():
        if name not in spec['spaces']:
            spec['spaces'].append(name)
        if not any(a['target'] == name for a in spec['affordances']):
            spec['affordances'].append(rule)
    import tutorial_workshop
    import tutorial_boxes
    return tutorial_boxes.prepare(world,tutorial_workshop.augment_spec(doors.augment_spec(spec, world), world))
