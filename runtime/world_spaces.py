"""
world_spaces.py -- SHARED placement -> HL spaces, for BOTH the trigger panel catalog (what you author
against) and the solver (the real space objects), so the two can't drift.

FACTORED / per-bit design (Tom's call): each placed boolean-goal ITEM becomes its OWN BitSpace (key_red,
honey_1, ...), not a single 2^n task hypercube -- so precedence is an AND of per-bit guards, and you can
see which bit is which item. Colored WALL cells become door MODE spaces (BitSpace X reads). Money is a
WealthSpace with a SPEND action. All are drivable in the panel (guards + effects); the root X is guarded
via grid-click / the free action.

  catalog_from_placement(...) -> {"spaces":[{name,ns,na,state_labels,action_labels,is_bl?}...], "na_free"}
      construction-FREE (instant panel open); sizes are known from the classes.
  build_spaces_from_placement(...) -> {name: real space obj}   (for the solver; names MATCH the catalog)
  item_specs(...) / door_specs(...)  -> the shared name<->cell mapping (used to auto-gen reach rules)
"""
import numpy as np
import state_spaces

WEALTH_TYPES   = {3}
NO_DEATH_TYPES = {3, 4, 5}
TYPE_NAMES     = {0: 'hunger', 1: 'hydration', 2: 'temperature', 3: 'money', 4: 'potion', 5: 'battery'}
_PHYSIO_ACTIONS = ['no-op', 'refill']                                       # StateSpace_1D 2-action chain
_WEALTH_ACTIONS = ['hold', 'deposit +1', 'withdraw -1', 'spend']            # WealthSpace na=4 (incl. SPEND)
# BitSpace actions (HOLD/SET/CLEAR/TOGGLE) relabelled per role:
_ITEM_ACTIONS   = ['hold', 'collect', 'drop', 'toggle']                     # item bit: SET=collect
_ITEM_STATES    = ['0 (not held)', '1 (held)']
_DOOR_ACTIONS   = ['hold', 'open', 'close', 'toggle']                       # door mode: SET=open
_DOOR_STATES    = ['closed', 'open']
_RECV_ACTIONS   = ['hold', 'give', 'take back', 'toggle']                   # NPC receiver bit: SET=give
_RECV_STATES    = ['0 (not given)', '1 (given)']
BOOL_TYPE_NAMES = {1: 'key', 2: 'flower', 3: 'honey', 4: 'badger', 5: 'axe', 6: 'canteen'}
COLOR_NAMES     = {0: 'white', 1: 'red', 2: 'blue', 3: 'orange', 4: 'purple'}
# Root grid space X (StateSpace_2D): fixed 6 actions, special_action=0. Guard-only (states = cells).
X_ACTION_LABELS = ['S (interact)', 'N (stay)', 'U (up)', 'R (right)', 'D (down)', 'L (left)']


# ---- shared name<->placement mappings (used by catalog, builder, AND reach-rule generation) ---------
def item_specs(boolean_states_2_type_ind, buttonColorIdx=None):
    """One spec per placed boolean-goal ITEM -> a factored bit. Stable, de-duplicated names.
    Returns [{name, cell, type, color}] sorted by cell (type 0 = legacy red-dot placeholder is skipped)."""
    bci = {int(k): int(v) for k, v in (buttonColorIdx or {}).items()}
    raw = []
    for cell in sorted(int(c) for c in boolean_states_2_type_ind):
        t = int(boolean_states_2_type_ind[cell])
        if t == 0:
            continue
        color = bci.get(cell, 0)
        base = f"key_{COLOR_NAMES.get(color, color)}" if (t == 1 and color) else BOOL_TYPE_NAMES.get(t, f"item{t}")
        raw.append((cell, t, color, base))
    counts = {}
    for _, _, _, base in raw:
        counts[base] = counts.get(base, 0) + 1
    seen, specs = {}, []
    for cell, t, color, base in raw:
        if counts[base] > 1:
            seen[base] = seen.get(base, 0) + 1
            name = f"{base}_{seen[base]}"
        else:
            name = base
        specs.append({"name": name, "cell": cell, "type": t, "color": color})
    return specs


def door_specs(buttonColorIdx=None, walls=None):
    """One spec per colored WALL cell -> a door mode space. Returns [{name, cell, color}]."""
    bci = {int(k): int(v) for k, v in (buttonColorIdx or {}).items()}
    wallset = set(int(w) for w in (walls or []))
    specs = []
    for cell, color in sorted(bci.items()):
        if cell in wallset and color >= 1:                                  # colored wall = door
            specs.append({"name": f"door_{COLOR_NAMES.get(color, color)}", "cell": cell, "color": color})
    return specs


def npc_specs(npc_states):
    """One spec per placed NPC badger (other agent). Returns [{name, cell}]."""
    cells = sorted(int(c) for c in (npc_states or []))
    if len(cells) == 1:
        return [{"name": "npc", "cell": cells[0]}]
    return [{"name": f"npc_{i + 1}", "cell": c} for i, c in enumerate(cells)]


def npc_receiver_specs(npc_states, boolean_states_2_type_ind):
    """RECEIVER bits: for each NPC and each ITEM TYPE present, a bit '<npc>_has_<type>' (0/1). The NPC's
    own state -> you author a transfer affordance (at NPC + holding item -> npc_has_item:give, item:drop)
    and set the GOAL to the NPC holding the items. Returns [{name, npc, npc_cell, type}]."""
    types = sorted(set(int(v) for v in boolean_states_2_type_ind.values()) - {0})
    out = []
    for npc in npc_specs(npc_states):
        for t in types:
            tn = BOOL_TYPE_NAMES.get(t, f"item{t}")
            out.append({"name": f"{npc['name']}_has_{tn}", "npc": npc["name"], "npc_cell": npc["cell"], "type": t})
    return out


# ---- catalog (construction-free) --------------------------------------------------------------------
def _internal_catalog_entry(t, ns):
    name = TYPE_NAMES.get(t, f'phys_space_{t}')
    labels = _WEALTH_ACTIONS if t in WEALTH_TYPES else _PHYSIO_ACTIONS
    return {"name": name, "ns": int(ns), "na": len(labels),
            "state_labels": [str(s) for s in range(int(ns))], "action_labels": list(labels)}


def _bit_entry(name, states, actions):
    return {"name": name, "ns": 2, "na": len(actions),
            "state_labels": list(states), "action_labels": list(actions)}


def catalog_from_placement(internal_goalstates_2_type_ind, boolean_states_2_type_ind,
                           nis, na_free=None, nis_by_type=None, T_f=60, x_ns=None,
                           buttonColorIdx=None, walls=None, npc_states=None, active_int_types=None):
    """Panel catalog. Order: [X] + internal spaces + per-item bits + door modes + NPC receiver bits.
    Internal types = placed goals UNION active_int_types (chain spaces created via the +/- button)."""
    nis_by_type = {int(k): int(v) for k, v in (nis_by_type or {}).items()}
    nis_of = lambda t: int(nis_by_type.get(int(t), nis))
    spaces = []

    def _tag(entry, owner):
        entry["owner"] = owner
        return entry

    if x_ns:
        spaces.append(_tag({"name": "X", "ns": int(x_ns), "na": len(X_ACTION_LABELS), "is_bl": True,
                            "state_labels": [str(s) for s in range(int(x_ns))],
                            "action_labels": list(X_ACTION_LABELS)}, "You"))
        na_free = len(X_ACTION_LABELS)
    _int_types = (set(int(v) for v in internal_goalstates_2_type_ind.values())
                  | set(int(t) for t in (active_int_types or ())))
    for t in sorted(_int_types):                                            # money -> Resources; physio -> You
        spaces.append(_tag(_internal_catalog_entry(t, nis_of(t)), "Resources" if t in WEALTH_TYPES else "You"))
    for spec in item_specs(boolean_states_2_type_ind, buttonColorIdx):      # one bit PER item
        spaces.append(_tag(_bit_entry(spec["name"], _ITEM_STATES, _ITEM_ACTIONS), "Items"))
    for spec in door_specs(buttonColorIdx, walls):                          # door mode spaces
        spaces.append(_tag(_bit_entry(spec["name"], _DOOR_STATES, _DOOR_ACTIONS), "Environment"))
    for spec in npc_receiver_specs(npc_states, boolean_states_2_type_ind):  # NPC receiver bits (grouped UNDER the NPC)
        spaces.append(_tag(_bit_entry(spec["name"], _RECV_STATES, _RECV_ACTIONS), spec["npc"]))
    return {"spaces": spaces, "na_free": int(na_free if na_free is not None else len(X_ACTION_LABELS))}


# ---- real space objects (for the solver; names MATCH the catalog) -----------------------------------
def internal_spaces_from_placement(internal_goalstates_2_type_ind, nis, nis_by_type=None, T_f=60,
                                   active_int_types=None):
    """Build the real internal space objects. Types = placed goals UNION `active_int_types` (chain spaces
    turned on via the +/- button with no BL object). Returns (int_types, int_ss_list)."""
    nis_by_type = {int(k): int(v) for k, v in (nis_by_type or {}).items()}
    nis_of = lambda t: int(nis_by_type.get(int(t), nis))
    int_types = sorted(set(int(v) for v in internal_goalstates_2_type_ind.values())
                       | set(int(t) for t in (active_int_types or ())))
    _kw = {}
    lst = []
    for i, t in enumerate(int_types):
        nm, ns_t = TYPE_NAMES.get(t, f'phys_space_{i+1}'), nis_of(t)
        if t in WEALTH_TYPES:
            lst.append(state_spaces.WealthSpace(ns_t, nm, T_f, t, constraint_vec=np.ones(ns_t), **_kw))
        elif t in NO_DEATH_TYPES:
            lst.append(state_spaces.Internal_StateSpace(ns_t, nm, T_f, t, np.ones(ns_t), defective_states=[], **_kw))
        else:
            cv = np.ones(ns_t); cv[0] = 0
            lst.append(state_spaces.Internal_StateSpace(ns_t, nm, T_f, t, cv, defective_states=[0], **_kw))
    return int_types, lst


def build_spaces_from_placement(internal_goalstates_2_type_ind, boolean_states_2_type_ind,
                                nis, nis_by_type=None, T_f=60, buttonColorIdx=None, walls=None,
                                price=1, ymax=None, npc_states=None, active_int_types=None,
):
    """Build the real HL space objects (NO X -- the solver builds that from the grid). Names MATCH the
    catalog: per-item BitSpaces, door BitSpaces, NPC receiver BitSpaces, internal/wealth (wealth=SPEND)."""
    spaces, meta = {}, {"items": item_specs(boolean_states_2_type_ind, buttonColorIdx),
                        "doors": door_specs(buttonColorIdx, walls),
                        "npcs": npc_specs(npc_states),
                        "receivers": npc_receiver_specs(npc_states, boolean_states_2_type_ind)}
    _kw = {}
    int_types, int_ss = internal_spaces_from_placement(internal_goalstates_2_type_ind, nis, nis_by_type, T_f,
                                                       active_int_types=active_int_types, **_kw)
    for t, sp in zip(int_types, int_ss):
        if t in WEALTH_TYPES:                                               # rebuild wealth WITH spend
            _ns = sp.ns
            sp = state_spaces.WealthSpace(_ns, sp.name, T_f, t, constraint_vec=np.ones(_ns),
                                          spend_amount=int(price), **_kw)
        spaces[sp.name] = sp
    for j, spec in enumerate(meta["items"]):
        spaces[spec["name"]] = state_spaces.BitSpace(spec["name"], T_f, 300 + j, **_kw)
    for j, spec in enumerate(meta["doors"]):
        spaces[spec["name"]] = state_spaces.BitSpace(spec["name"], T_f, 400 + j, **_kw)
    for j, spec in enumerate(meta["receivers"]):
        spaces[spec["name"]] = state_spaces.BitSpace(spec["name"], T_f, 500 + j, **_kw)
    return spaces, meta
