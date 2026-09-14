"""Deterministic Sokoban as compact current-state H and local position kernels.

No table over box configurations is constructed. A categorical push event drives
one box alpha and the root's hold/move mode from the same pre-transition state.
"""
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import sparse

MOVES={2:(1,0),3:(0,1),4:(-1,0),5:(0,-1)}
LABELS=['hold','hold','down','right','up','left']

def name(box): return 'box_'+str(box['id'])

def normalize(world,n,walls):
    boxes=[]; ids=set(); cells=set()
    for b in world.get('boxes',[]):
        ident,cell=b.get('id'),b.get('cell')
        if type(ident) is not int or ident<1 or ident in ids: raise ValueError('Box IDs must be unique positive integers')
        if type(cell) is not int or not 0<=cell<n or cell in walls or cell in cells: raise ValueError('Boxes need distinct cells outside walls')
        if cell==world.get('start'): raise ValueError('Agent and box cannot occupy the same cell')
        boxes.append(dict(id=ident,cell=cell));ids.add(ident);cells.add(cell)
    return sorted(boxes,key=lambda b:b['id'])

def catalog(world):
    return [dict(name=name(b),ns=world['rows']*world['cols'],na=6,owner='Boxes',action_labels=LABELS) for b in world.get('boxes',[])]

def current(world): return {name(b):b['cell'] for b in world.get('boxes',[])}

def neighbor(cell,action,rows,cols):
    if action not in MOVES:return cell
    r,c=divmod(cell,cols); dr,dc=MOVES[action]
    return (r+dr)*cols+c+dc if 0<=r+dr<rows and 0<=c+dc<cols else None

def attach(spaces,world):
    boxes=world.get('boxes',[])
    if not boxes:return
    n=world['rows']*world['cols']; walls=set(world['walls'])
    # O(6*N) shared local kernel; never N raised to the number of boxes.
    coords=[]
    for a in range(6):
        for x in range(n):
            j=neighbor(x,a,world['rows'],world['cols'])
            if j is None or j in walls or x in walls:j=x
            coords.append((a,x,j))
    P=sparse.COO(np.array(coords).T,np.ones(len(coords)),shape=(6,n,n))
    for b in boxes:spaces[name(b)]=SimpleNamespace(name=name(b),ns=n,na=6,default_action_ind=0,P_a_s_s=P)

def prepare(world,spec):
    spec=deepcopy(spec); old=set(spec.pop('sokoban',{}).get('boxes',[])); names=[name(b) for b in world.get('boxes',[])]
    spec['spaces']=[n for n in spec['spaces'] if n not in old or n in names]
    if not names:return spec
    for n in names:
        if n not in spec['spaces']:spec['spaces'].append(n)
    spec['sokoban']=dict(rows=world['rows'],cols=world['cols'],walls=world['walls'],boxes=names,
        doors=[dict(name='door_'+str(b['id']),cell=b['cell']) for b in world.get('doors',[])])
    return spec


class PushEvents:
    """Relational predicates evaluated on demand; O(boxes + doors) per event."""
    def __init__(self,config,base_modes,base_effects):
        self.config=config;self.names=tuple(config['boxes']);self.walls=set(config['walls'])
        self.base_modes=tuple(zip(base_modes,base_effects))

    def evaluate(self,s,a):
        c=self.config; x=s['X']; positions=[s[n] for n in self.names]
        # Invalid Cartesian configurations remain closed under X/box identity.
        if x in self.walls or x in positions or len(set(positions))!=len(positions) or any(p in self.walls for p in positions):
            return True,None
        if any(e['kind']=='identity' and f.induced_alpha_at(s,a) for f,e in self.base_modes):return True,None
        if a not in MOVES:return False,None
        blocked=self.walls|{b['cell'] for b in c['doors'] if not s[b['name']]}
        ahead=neighbor(x,a,c['rows'],c['cols'])
        if ahead is None or ahead in blocked:return True,None
        if ahead not in positions:return False,None
        target=neighbor(ahead,a,c['rows'],c['cols'])
        if target is None or target in blocked or target in positions:return True,None
        return False,self.names[positions.index(ahead)]


class PushAffordance:
    def __init__(self,events,target):
        self.events=events;self.target=target
        self.parents=list(dict.fromkeys(['X',*events.names,*[b['name'] for b in events.config['doors']],
            *[n for f,e in events.base_modes for n in f.parents]]))
        self.has_action=True
        self.H_alpha=np.eye(2) if target=='X' else np.eye(6)[:,[0,2,3,4,5]]

    def induced_alpha_at(self,state,action):
        blocked,box=self.events.evaluate(state,action)
        return int(blocked) if self.target=='X' else int(action if box==self.target else 0)

    def report(self):
        return dict(target='X mode: box collision' if self.target=='X' else self.target,
            parents=self.parents,merge='reject',conflicting_assignments=0,h_form='compact spatial',
            H_psi=dict(shape=[self.H_alpha.shape[1],'predicate'],representation='relational predicates',values=None,
                definition='Adjacent box AND free cell beyond (no wall, closed door, or other box); shared current-state push event',
                selection='One free-action direction and at most one adjacent box; invalid overlaps hold'),
            H_alpha=dict(shape=list(self.H_alpha.shape),values=self.H_alpha.tolist()),
            F=dict(representation='predicate composition',values=None))


def install(spaces,affs,reports,spec):
    c=spec['sokoban']; names=c['boxes']; byname={s.name:s for s in spaces}; root=spaces[0]
    if c['rows']*c['cols']!=root.ns or len(set(names))!=len(names) or any(n not in byname or byname[n].ns!=root.ns for n in names):
        raise ValueError('Invalid Sokoban position spaces')
    if any(n in affs.by_driven or n in affs.transition for n in names):raise ValueError('Box push rules cannot be overridden by another affordance')
    # Validate the local BL operator only: stochastic/teleport movement must not
    # desynchronize agent and box. No joint assignments are visited here.
    import dbn_composer as d
    walls=set(c['walls'])
    for a in range(root.na):
        for x in range(root.ns):
            j=neighbor(x,a,c['rows'],c['cols'])
            if j is None or j in walls or x in walls:j=x
            row=d.dense(root.P_a_s_s[a,x])
            if np.count_nonzero(row)!=1 or row[j]!=1:
                raise ValueError('Boxes currently require deterministic ordinary grid movement (p_main = 1, no movement overrides)')
    events=PushEvents(c,affs.kernel_modes,affs.root_kernel.effects)
    # Explicit custom spatial functions may not smuggle in unaccounted root modes.
    for effect in affs.root_kernel.effects:
        if effect['kind'] not in ('identity','blocked_destination'):raise ValueError('Unsupported root mode with boxes')
    for n in names:
        f=PushAffordance(events,n);affs.by_driven[n]=f;reports.append(f.report())
    mode=PushAffordance(events,'X');affs.kernel_modes.append(mode)
    affs.root_kernel.effects.append(dict(name='box_collision',kind='identity'))
    reports.append(mode.report());affs.compact=True
