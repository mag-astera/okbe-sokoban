"""Finite resources, crafting and trading compiled entirely into shared H features."""
from copy import deepcopy
from state_spaces import WealthSpace, BitSpace, StateSpace_1D
import numpy as np

KINDS=('wood','iron','forge','store','canteen_bench')
DRINK = 6
CHOP = 8
FILL = 7

def tree_cells(world):
    return sorted(int(c) for c,t in world.get("internal_goalstates_2_type_ind",{}).items() if t==0) if world.get("workshop",{}).get("chop_trees") else []

def wood_capacity(world):
    return world.get("workshop",{}).get("wood_capacity",5)
WATER_CAPACITY = 2

def canteen_enabled(world):
    return any(is_canteen_forge(o) for o in objects(world))

def is_canteen_forge(o):
    return o['kind']=='canteen_bench' or (o['kind']=='forge' and o.get('recipe')=='canteen')

def special_actions(world):
    actions=[]
    if canteen_enabled(world):
        actions.extend([dict(name='drink',label='Drink from canteen',key='R'),
                        dict(name='fill',label='Fill canteen at lake',key='L')])
    if tree_cells(world):actions.append(dict(name='chop',label='Chop tree with axe',key='Q'))
    return [dict(a,id=i+6) for i,a in enumerate(actions)]

def action_id(world,name):
    return next(a['id'] for a in special_actions(world) if a['name']==name)

class CanteenWaterSpace(StateSpace_1D):
    """Local water kernel: hold, fill to capacity, consume one serving."""
    def __init__(self,capacity=WATER_CAPACITY):
        n=capacity+1
        P=np.zeros((3,n,n))
        for i in range(n):
            P[0,i,i]=1; P[1,i,n-1]=1; P[2,i,max(0,i-1)]=1
        super().__init__(n,'canteen_water',40,802,np.ones(n),defective_states=[],
                         transition_kernel=P,initial_state=0,default_action_ind=0)

def normalize(raw, n, walls):
    data=raw.get('workshop') or {}; objects=[]; seen=set(); cells=set()
    for obj in data.get('objects',[]):
        ident,cell,kind=obj.get('id'),obj.get('cell'),obj.get('kind')
        quantity=obj.get('quantity',1)
        if type(ident) is not int or ident<1 or ident in seen: raise ValueError('Workshop IDs must be unique positive integers')
        if type(cell) is not int or not 0<=cell<n or cell in cells or cell in walls: raise ValueError('Workshop objects need distinct cells outside walls')
        if kind not in KINDS or type(quantity) is not int or not 1<=quantity<=5: raise ValueError('Workshop: select wood, iron, forge, store or canteen bench, with 1–5 units')
        recipe=obj.get('recipe','canteen' if kind=='canteen_bench' else 'axe')
        if kind in ('forge','canteen_bench') and recipe not in ('axe','canteen'):raise ValueError('Forge recipe must be axe or canteen')
        row=dict(id=ident,cell=cell,kind='forge' if kind=='canteen_bench' else kind,quantity=quantity)
        if kind in ('forge','canteen_bench'):row['recipe']=recipe
        objects.append(row);seen.add(ident);cells.add(cell)
    if len(objects)>20: raise ValueError('At most 20 workshop objects')
    values=data.get('values',{})
    if not isinstance(values,dict) or any(type(v) is not int or v<0 for v in values.values()):raise ValueError('Workshop state values must be nonnegative integers')
    extra={}
    if data.get('movement_metabolism'):extra['movement_metabolism']=True
    if data.get('chop_trees'):
        extra['chop_trees']=True
    if 'wood_capacity' in data:
        if type(data['wood_capacity']) is not int or not 1<=data['wood_capacity']<=5:raise ValueError('Wood capacity must be 1–5')
        extra['wood_capacity']=data['wood_capacity']
    if 'water_capacity' in data:
        if type(data['water_capacity']) is not int or not 1<=data['water_capacity']<=5:raise ValueError('Water capacity must be 1–5')
        extra['water_capacity']=data['water_capacity']
    return {'objects':sorted(objects,key=lambda o:o['id']),'values':dict(values),**extra}

def objects(world):return world.get('workshop',{}).get('objects',[])
def stock(o):return 'resource_'+str(o['id'])
def catalog(world):
    if not objects(world):return []
    money_n=world.get('nis_by_type',{}).get(3,world.get('nis',6)) if 3 in world.get('internal_goalstates_2_type_ind',{}).values() else 6
    rows=[dict(name=n,ns=money_n if n=='money' else 6,na=4,owner='Workshop',action_labels=['hold','add 1','remove 1','spend 1']) for n in ('wood','iron','money')]
    rows.append(dict(name='axe',ns=2,na=4,owner='Workshop',state_labels=['absent','held'],action_labels=['hold','acquire','consume','toggle']))
    if canteen_enabled(world):
        if not any(o['kind'] in ('iron','forge','store') and not is_canteen_forge(o) for o in objects(world)):
            rows=[c for c in rows if c['name']=='wood']
        rows.extend([dict(name='canteen',ns=2,na=4,owner='Workshop',state_labels=['absent','held'],action_labels=['hold','acquire','consume','toggle']),
                     dict(name='canteen_water',ns=world.get('workshop',{}).get('water_capacity',WATER_CAPACITY)+1,na=3,owner='Workshop',action_labels=['hold','fill','drink one serving'])])
    rows += [dict(name=stock(o),ns=o['quantity']+1,na=4,owner='Workshop',action_labels=['hold','add 1','take 1','spend 1']) for o in objects(world) if o['kind'] in ('wood','iron')]
    if tree_cells(world):
        if not any(c['name']=='axe' for c in rows):rows.append(dict(name='axe',ns=2,na=4,owner='Workshop',action_labels=['hold','acquire','consume','toggle']))
        rows += [dict(name='tree_'+str(c),ns=2,na=4,owner='Workshop',state_labels=['chopped','standing'],action_labels=['hold','grow','chop','toggle']) for c in tree_cells(world)]
    for c in rows:
        if c['name']=='wood':c['ns']=wood_capacity(world)+1
    return rows

def attach(spaces,world):
    if world.get('workshop',{}).get('movement_metabolism'):
        for name in ('hunger','hydration'):
            if name in spaces:
                old=spaces[name];P=np.concatenate([np.asarray(old.P_a_s_s),np.eye(old.ns)[None]],axis=0)
                spaces[name]=StateSpace_1D(old.ns,name,40,old.action_type_ind,np.asarray(old.constraint_vec),transition_kernel=P,defective_states=[0],initial_state=old.current_state)
    for c in catalog(world):
        if c['name'] not in spaces:
            spaces[c['name']]=(CanteenWaterSpace(c['ns']-1) if c['name']=='canteen_water' else BitSpace(c['name'],40,800) if c['name'] in ('axe','canteen') or c['name'].startswith('tree_') else WealthSpace(c['ns'],c['name'],40,801))

def current(world,existing=None):
    existing=existing or {}; values=world.get('workshop',{}).get('values',{})
    defaults={stock(o):o['quantity'] for o in objects(world) if o['kind'] in ('wood','iron')}
    defaults.update({'tree_'+str(c):1 for c in tree_cells(world)})
    return {c['name']:min(c['ns']-1,existing.get(c['name'],values.get(c['name'],defaults.get(c['name'],0)))) for c in catalog(world)}

def augment_spec(spec,world):
    spec=deepcopy(spec); cs=catalog(world); names={c['name'] for c in cs}
    hydration = canteen_enabled(world) and 1 in world.get('internal_goalstates_2_type_ind',{}).values()
    if hydration:names.add('hydration')
    if tree_cells(world):names.add('hunger')
    if world.get('workshop',{}).get('movement_metabolism'):
        names.update(n for n,t in [('hunger',0),('hydration',1)] if t in world.get('internal_goalstates_2_type_ind',{}).values())
    old={a['target'] for a in spec.get('affordances',[]) if a.get('workshop_managed')}
    spec['affordances']=[a for a in spec.get('affordances',[]) if not a.get('workshop_managed')]
    spec['spaces']=[n for n in spec.get('spaces',[]) if n not in old or n in names]
    features=spec.setdefault('features',{})
    for name in list(features):
        if name.startswith('workshop_'):del features[name]
    if not cs:return spec
    # Preserve explicitly customized rules; generated placement rules are replaced for money
    # so a money marker cannot supply free deposits alongside the store exchange.
    overrides={a['target'] for a in spec['affordances'] if a.get('placement_managed') is False}
    hydration = canteen_enabled(world) and 1 in world.get('internal_goalstates_2_type_ind',{}).values()
    if hydration:names.add('hydration')
    if tree_cells(world):names.add('hunger')
    if world.get('workshop',{}).get('movement_metabolism'):
        names.update(n for n,t in [('hunger',0),('hydration',1)] if t in world.get('internal_goalstates_2_type_ind',{}).values())
    rules={n:[] for n in names}
    for a in spec['affordances']:
        if (a['target'] in ('axe','canteen','hydration') or (a['target']=='hunger' and not tree_cells(world))) and a['target'] in rules and a.get('placement_managed') is not False:rules[a['target']].extend(a.get('rules',[]))
    spec['affordances']=[a for a in spec['affordances'] if a['target'] not in names or a['target'] in overrides]
    def event(name,parts,effects):
        # Do not silently split a coordinated exchange when only one target was customized.
        if set(effects)&overrides:raise ValueError('Workshop exchange overlaps customized rules for '+', '.join(sorted(set(effects)&overrides))+'. Restore managed workshop rules before using the recipe.')
        features[name]={'and':parts}
        for target,alpha in effects.items():rules[target].append({'when':{'feature':name},'alpha':alpha})
    money_max=next((c['ns']-1 for c in cs if c['name']=='money'),5)
    for o in objects(world):
        parts=[{'var':'X','in':[o['cell']]},{'var':'action','in':[0]}];kind=o['kind'];name='workshop_'+str(o['id'])
        if kind in ('wood','iron'):
            event(name,parts+[{'var':stock(o),'gte':1},{'var':kind,'lt':wood_capacity(world) if kind=='wood' else 5}],{stock(o):2,kind:1})
        elif kind=='forge' and not is_canteen_forge(o):
            event(name,parts+[{'var':'wood','gte':1},{'var':'iron','gte':1},{'var':'axe','in':[0]}],{'wood':2,'iron':2,'axe':1})
        elif is_canteen_forge(o):
            event(name,parts+[{'var':'wood','gte':1},{'var':'canteen','in':[0]}],{'wood':2,'canteen':1})
        else:event(name,parts+[{'var':'axe','in':[1]},{'var':'money','lt':money_max}],{'axe':2,'money':1})
    for cell in tree_cells(world):
        name='tree_'+str(cell)
        event('workshop_chop_'+str(cell),[{'var':'X','in':[cell]},{'var':'action','in':[action_id(world,'chop')]},
              {'var':'axe','in':[1]},{'var':name,'in':[1]},{'var':'wood','lt':wood_capacity(world)}],
              {name:2,'wood':1})
        event('workshop_eat_'+str(cell),[{'var':'X','in':[cell]},{'var':'action','in':[0]},
              {'var':name,'in':[1]}],{'hunger':1})
    if canteen_enabled(world):
        # One current-state event drives both local transformations. No imperative transfer.
        alive=[{'var':'hydration','gt':0}] if hydration else []
        lakes=[int(c) for c,t in world.get('internal_goalstates_2_type_ind',{}).items() if t==1]
        event('workshop_fill_canteen',[{'var':'X','in':lakes},{'var':'action','in':[FILL]},
              {'var':'canteen','in':[1]},{'var':'canteen_water','lt':world.get('workshop',{}).get('water_capacity',WATER_CAPACITY)}]+alive,{'canteen_water':1})
        if hydration:
            event('workshop_drink_canteen',[{'var':'action','in':[DRINK]},
                  {'var':'canteen','in':[1]},{'var':'canteen_water','gte':1}]+alive,
                  {'canteen_water':2,'hydration':1})
    if world.get('workshop',{}).get('movement_metabolism'):
        for name in ('hunger','hydration'):
            if name in rules:
                refill={'or':[r['when'] for r in rules[name]]}
                event('workshop_rest_'+name,[{'not':{'var':'action','in':[2,3,4,5]}},{'not':refill}],{name:2})
    for n in sorted(names):
        if n not in spec['spaces']:spec['spaces'].append(n)
        if n not in overrides:spec['affordances'].append(dict(target=n,default_alpha=0,merge='reject',placement_managed=True,workshop_managed=True,rules=rules[n]))
    return spec
