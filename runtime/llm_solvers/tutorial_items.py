"""Stable tutorial item identities, independent of placement order."""
import re
import world_spaces as ws


def item_names(world):
    specs=ws.item_specs(world.get('boolean_states_2_type_ind',{}))
    supplied={int(k):v for k,v in (world.get('item_names') or {}).items()}
    names={s['cell']:supplied[s['cell']] for s in specs if s['cell'] in supplied}
    for s in specs:
        if s['cell'] in names:
            value=names[s['cell']];base=ws.BOOL_TYPE_NAMES[s['type']]
            if not isinstance(value,str) or not re.fullmatch(re.escape(base)+r'(?:_[1-9][0-9]*)?',value):
                raise ValueError('Invalid stable item name')
    if len(set(names.values()))!=len(names):raise ValueError('Duplicate stable item names')
    used=set(names.values())
    for s in specs:
        if s['cell'] in names:continue
        name=s['name'];base=ws.BOOL_TYPE_NAMES[s['type']];i=1
        while name in used:name=f'{base}_{i}';i+=1
        names[s['cell']]=name;used.add(name)
    return names


def item_specs(world):
    names=item_names(world)
    return [dict(s,name=names[s['cell']]) for s in ws.item_specs(world['boolean_states_2_type_ind'])]


def rename_catalog(world,catalog):
    mapping={s['name']:item_names(world)[s['cell']] for s in ws.item_specs(world['boolean_states_2_type_ind'])}
    return [dict(s,name=mapping.get(s['name'],s['name'])) for s in catalog]


def rename_spaces(world,spaces):
    mapping={s['name']:item_names(world)[s['cell']] for s in ws.item_specs(world['boolean_states_2_type_ind'])}
    result={}
    for name,sp in spaces.items():
        new=mapping.get(name,name);sp.name=new;result[new]=sp
    return result
