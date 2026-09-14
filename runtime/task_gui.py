"""
task_gui.py -- the in-editor TASK panel (T key): author a level's objective as a Boolean condition
over the PRODUCT space.

WHAT THIS IS NOT. The I panel (trigger_gui) authors AFFORDANCE rules -- they compile to F via
gates.make_affordance, which turns each rule's GUARD into a row of H_psi and its EFFECT into a column
of H_alpha. Those are DYNAMICS: what the world does. This panel authors the OBJECTIVE: which joint
states count as success. Same Boolean vocabulary, opposite side of the problem.

WHY DNF IS THE ENTRY FORM
A DNF term IS a region. The region formalism is R_r = prod_k R_r^k -- a box, i.e. one condition per
space AND-ed together. So an OR-of-ANDs is a union of boxes, which is exactly what build_h_r consumes
and what the STOK factorization needs per option. Authoring in DNF means every clause you write IS a
region the solver uses; there is no conversion step where what you wrote stops resembling what runs.
It also makes the cost visible: `dnf_terms()` counts the clauses live, so the panel can say "3
regions" as you type. That matters because Gate.dnf() warns it "may blow up exponentially" -- a
CNF-shaped spec explodes on conversion, and you want to see that before committing, not after.

STORAGE IS THE GATE TREE, NOT THE DNF. gates.Gate already gives .dnf() / .cnf() / .dnf_terms() /
.cnf_clauses(), so the stored form stays general and either normal form is one call away. Nothing
here forecloses a CNF entry mode (planned) or a tree editor later.

A task is a list of CLAUSES; each clause is a list of LITERALS; each literal is
    (space_name, op, value)      op in gates: ==, !=, in, >=, <=, >, <
The whole task is  OR over clauses  of  (AND over that clause's literals).
"""
import gates

# op token -> gates constructor. Kept as data so the panel can offer them as chips and so a saved
# task is plain JSON-able tuples rather than pickled lambdas.
OPS = {
    '==': gates.Eq,
    '!=': gates.Ne,
    'in': gates.In,
    '>=': gates.Ge,
    '<=': gates.Le,
    '>':  gates.Gt,
    '<':  gates.Lt,
}
OP_ORDER = ['==', '!=', '>=', '<=', '>', '<', 'in']


def fmt_val(v):
    """Render a literal's value: a symbol as @name, a set as {a,b,c}, anything else plainly."""
    if is_symbolic(v):
        return sym_label(v)
    if isinstance(v, (list, tuple, set, frozenset)):
        return "{" + ",".join(str(x) for x in sorted(v)) + "}"
    return str(v)


def literal_gate(lit):
    """(space, op, value) -> a gates.Cond.

    'in' takes a SET of values -- it is the general per-space region primitive: a DNF term is
    prod_k R^k_r where each R^k_r is a SET of states, so '==' is the singleton case and 'in' the
    general one. X in {40,41,42} stays ONE clause (one region); the same thing via '==' would need
    three. A scalar is coerced to a 1-element set rather than crashing, since the value field hands
    over a plain int until you type a comma."""
    space, op, value = lit
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {OP_ORDER}")
    if op == 'in':
        value = tuple(value) if isinstance(value, (list, tuple, set, frozenset)) else (value,)
    return OPS[op](space, value)


def clause_gate(clause):
    """[lit, ...] -> AND over the clause. An empty clause is vacuously TRUE and is rejected, because
    a true clause makes the whole task trivially satisfied and that is never what you meant."""
    if not clause:
        raise ValueError("empty clause (would be vacuously true)")
    g = literal_gate(clause[0])
    for lit in clause[1:]:
        g = g & literal_gate(lit)
    return g


def task_gate(clauses):
    """[[lit, ...], ...] -> OR over clauses of AND over literals. Returns None for an empty task
    (no objective authored yet) rather than a vacuously-false Gate, so callers can tell the two apart."""
    clauses = [c for c in clauses if c]
    if not clauses:
        return None
    g = clause_gate(clauses[0])
    for c in clauses[1:]:
        g = g | clause_gate(c)
    return g


def validate(clauses, spaces, host=None):
    """Human-readable problems with an authored task (empty list = OK). Checked against the REAL space
    objects so a stale space name or an out-of-range value is caught here, with a message, instead of
    surfacing as a numpy error deep in a compose. Mirrors trigger_rules.validate_rules."""
    errs = []
    clauses = resolve(clauses, host) if host is not None else clauses
    for ci, clause in enumerate(clauses):
        if not clause:
            errs.append(f"clause[{ci}]: empty (vacuously true)")
            continue
        seen = {}
        for li, lit in enumerate(clause):
            tag = f"clause[{ci}].literal[{li}]"
            try:
                space, op, value = lit
            except (TypeError, ValueError):
                errs.append(f"{tag}: malformed, expected (space, op, value)"); continue
            if space not in spaces:
                errs.append(f"{tag}: unknown space {space!r} (have {sorted(spaces)})"); continue
            if op not in OPS:
                errs.append(f"{tag}: unknown op {op!r} (have {OP_ORDER})"); continue
            ns = getattr(spaces[space], 'ns', None)
            vals = value if op == 'in' else [value]
            try:
                bad = [v for v in vals if ns is not None and not (0 <= int(v) < ns)]
            except (TypeError, ValueError):
                errs.append(f"{tag}: value {value!r} is not an int (or set of ints)"); continue
            if bad:
                errs.append(f"{tag}: value(s) {bad} outside {space} range 0..{ns - 1}")
            # Two literals on the SAME space within one clause is legal (a>=2 AND a<=5 is a band) but
            # an == pair is unsatisfiable, which makes the clause dead weight. Worth saying out loud.
            if op == '==' and seen.get(space) == '==':
                errs.append(f"{tag}: second '==' on {space} in the same clause -- unsatisfiable")
            seen[space] = op
    return errs


def summarize(clauses, host=None):
    """Counts for the panel header: clauses authored, and what the DNF actually costs.

    n_terms is what the solver would see. It equals len(clauses) for a task authored directly in DNF
    (that is the point of authoring in DNF), so a divergence means something non-DNF crept in."""
    clauses = resolve(clauses, host) if host is not None else clauses
    g = task_gate(clauses)
    if g is None:
        return {'clauses': 0, 'n_terms': 0, 'spaces': [], 'ok': True, 'text': '(no objective)'}
    terms = g.dnf_terms()
    return {
        'clauses': len([c for c in clauses if c]),
        'n_terms': len(terms),                 # == regions the objective decomposes into
        'spaces': sorted(g.vars()),
        'ok': len(terms) == len([c for c in clauses if c]),
        'text': repr(g),
    }


def satisfied(clauses, state, host=None):
    """Evaluate the task at a joint state dict {space_name: index}. The objective is a CONDITION over
    the product space, so this is the natural check for a planner/tree-search terminal test -- it does
    NOT go through g[na, nx], which is X-only and cannot see the other spaces."""
    g = task_gate(resolve(clauses, host) if host is not None else clauses)
    return True if g is None else bool(g.eval(state))


def describe(clauses):
    """One line per clause, for the panel body and for a level's manifest entry."""
    out = []
    for ci, clause in enumerate(clauses):
        if not clause:
            continue
        lits = " AND ".join(f"{s} {op} {fmt_val(v)}" for (s, op, v) in clause)
        out.append(f"[{ci}] {lits}")
    return out or ["(no objective)"]


# =================================================================================================
# PANEL  (T key)  -- author the DNF: pick the active AND-clause, add clauses with +, G-click the grid
# =================================================================================================
# One colour per clause, cycled. The SAME colour marks that clause's cells on the grid, so the
# disjunction is visible where you author it: red cells are one way to satisfy the task, blue another.
CLAUSE_COLORS = [
    (232, 92, 84), (86, 170, 240), (240, 190, 70), (120, 210, 120),
    (200, 130, 235), (90, 215, 210), (240, 150, 90), (170, 175, 190),
]
def clause_color(ix):
    return CLAUSE_COLORS[ix % len(CLAUSE_COLORS)]


ROW_H, PAD = 26, 10
DEL_W = 18           # delete-button width
GAP = 6              # gap between the clause row and its delete button
INSET = 8            # text inset from its own box's left edge


def _box(x0, y_center, x1, h):
    """Rect helper: build an element from EXPLICIT left/right edges rather than a centre plus a
    guessed width. The original panel did the latter and the arithmetic silently went wrong -- the
    clause rect ran 6px under the delete button, and label text was placed 11px outside its own box."""
    return dict(cx=(x0 + x1) / 2.0, cy=y_center, w=x1 - x0, h=h, x0=x0, x1=x1)


def panel_layout(clauses, active_ix, W, H, catalog=None, host_ref=None):
    """Pure layout -> element dicts (cx, cy, w, h, label, action). No arcade calls.

    Every element carries x0/x1 as well as cx/w so panel_layout.rect_overlaps can check the result
    arithmetically; test_task_panel.py asserts no two boxes overlap and every label sits inside the
    box it labels.

    actions: ('sel', i) pick the active clause | ('del', i) drop it | ('add', None) new clause."""
    els = []
    y = H - 26 - 18                                     # below the I|T|H mode bar
    s = summarize(clauses, host_ref[0] if host_ref else None)

    def text(lbl, cy, size=11, kind='label', col=None):
        b = _box(PAD, cy, W - PAD, 18)
        els.append(dict(kind=kind, label=lbl, action=None, size=size, filled=False, col=col,
                        x=PAD, **b))

    text("TASK RULES   ·   T = save & close", y, size=14, kind='title'); y -= 22
    text("objective = OR over AND-clauses (DNF)", y, size=10); y -= 15
    text("G + click a grid cell -> adds to the ACTIVE clause", y, size=10); y -= 20

    row_x1 = W - PAD - DEL_W - GAP                      # clause row STOPS before the delete button
    for i, cl in enumerate(clauses):
        lits = " AND ".join(f"{a} {o} {v}" for (a, o, v) in cl) if cl else "(empty -- add cells with G)"
        b = _box(PAD, y, row_x1, ROW_H - 4)
        els.append(dict(kind='clause', label=f"[{i}] {lits}", action=('sel', i), size=11,
                        filled=(i == active_ix), col=clause_color(i), x=PAD + INSET, **b))
        d = _box(W - PAD - DEL_W, y, W - PAD, ROW_H - 4)
        els.append(dict(kind='del', label="x", action=('del', i), size=11, filled=False,
                        col=(150, 90, 90), x=W - PAD - DEL_W + 6, **d))
        y -= ROW_H

    a = _box(PAD, y, W - PAD, ROW_H - 6)
    els.append(dict(kind='add', label="+  new AND clause", action=('add', None), size=11,
                    filled=False, col=(90, 150, 110), x=PAD + INSET, **a))
    y -= ROW_H + 6

    # The DNF cost, live. clauses == terms when authored directly in DNF; a divergence means
    # something non-DNF got in and the solver would see more regions than you wrote.
    text(f"{s['clauses']} clause(s)  ->  {s['n_terms']} region(s)", y, size=11,
         col=(150, 200, 150) if s['ok'] else (235, 170, 90)); y -= 15
    text(f"spaces: {', '.join(s['spaces']) or '-'}", y, size=10); y -= 22

    cat = catalog or []
    if cat:
        e, y = custom_layout(host_ref[0] if host_ref else None, cat, W, y)
        els += e
    return els


def _pick(host):
    """The custom-literal picker's state: {space, op, value}. Lazily created."""
    p = getattr(host, 'task_pick', None)
    if p is None:
        p = {'space': None, 'op': '==', 'value': 0}
        host.task_pick = p
    return p


def _wrap(items, x0, y, maxx, w, h=17, gap=4):
    """Lay chips left-to-right, wrapping. items = [(label, selected, action)]. Returns (els, next_y)."""
    els, x = [], x0
    for label, sel, act in items:
        if x + w > maxx:
            x, y = x0, y - (h + gap)
        els.append(dict(kind='chip', label=str(label), action=act, size=9, filled=sel,
                        col=(90, 150, 190), x=x + 4, cx=x + w / 2.0, cy=y, w=w, h=h,
                        x0=x, x1=x + w))
        x += w + gap
    return els, y - (h + gap)


def custom_layout(host, catalog, W, y):
    """The CUSTOM LITERAL section: pick any space by name, any op, any value.

    The click-a-widget paths (grid cell, chain-bar segment, inventory slot) only reach spaces that
    HAVE a widget. Doors, wealth counters, task bits and NPC receiver bits have none, and a space with
    a large ns is impractical to click through anyway. This is the general fallback: every space in the
    catalog is listed by name, so nothing is unreachable just because it is hard to draw."""
    p = _pick(host)
    els = []
    x0, maxx = PAD, W - PAD

    def text(lbl, cy, size=10, col=None):
        els.append(dict(kind='label', label=lbl, action=None, size=size, filled=False, col=col,
                        x=PAD, cx=W / 2.0, cy=cy, w=W - 2 * PAD, h=16, x0=PAD, x1=W - PAD))

    text("CUSTOM LITERAL  --  any space, any op, any value", y, size=11, col=(150, 190, 220)); y -= 18
    names = [c["name"] for c in catalog]
    e, y = _wrap([(n[:9], n == p['space'], ('pspace', n)) for n in names], x0, y, maxx, w=62)
    els += e

    e, y = _wrap([(o, o == p['op'], ('pop', o)) for o in OP_ORDER], x0, y, maxx, w=30)
    els += e

    ns = next((int(c["ns"]) for c in catalog if c["name"] == p['space']), None)
    # Numeric FIELD, not a +/- stepper: X has ns=81, so stepping to a cell index is unusable, and a
    # typed value is the only sane input once ns gets large. Click it to focus, type digits, Enter to
    # commit (clamped to the space's ns), Esc to cancel. Modal while focused -- see value_key.
    typing = bool(getattr(host, 'task_val_active', False)) if host is not None else False
    shown = (getattr(host, 'task_val_buf', '') if typing else str(p['value']))
    fw = 70
    els.append(dict(kind='field', label=shown, action=('pfocus', None), size=12, filled=typing,
                    col=(120, 230, 255) if typing else (110, 116, 132),
                    x=PAD + 8, cx=PAD + fw / 2.0, cy=y, w=fw, h=20, x0=PAD, x1=PAD + fw))
    hint = (f"type a value, Enter to set   (0..{ns - 1})" if typing else
            (f"click to type a value   (0..{ns - 1})" if ns else "pick a space first"))
    text(hint, y, size=9)
    els[-1].update(x=PAD + fw + 10, x0=PAD + fw + 6, x1=W - PAD)
    y -= 24

    ok = p['space'] is not None
    els.append(dict(kind='add', label="add literal to active clause" if ok else "(pick a space first)",
                    action=('addlit', None) if ok else None, size=10, filled=False,
                    col=(90, 150, 110) if ok else (70, 74, 86),
                    x=PAD + INSET, cx=W / 2.0, cy=y, w=W - 2 * PAD, h=ROW_H - 6,
                    x0=PAD, x1=W - PAD))
    y -= ROW_H + 6
    e, y = derived_layout(host, W, y)
    return els + e, y


def custom_apply(host, action, catalog):
    """Apply a custom-picker action. Returns True if consumed."""
    if action is None:
        return False
    kind, v = action
    p = _pick(host)
    if kind == 'pspace':
        p['space'] = v
        ns = next((int(c["ns"]) for c in catalog if c["name"] == v), 1)
        # switching space re-clamps the pending value; it may be a SET (op 'in') or a scalar
        cur = p['value']
        if isinstance(cur, (list, tuple, set, frozenset)):
            p['value'] = tuple(sorted({min(int(q), ns - 1) for q in cur})) or (0,)
        else:
            p['value'] = min(int(cur), ns - 1)
        print(f"[task] custom: space {v} (0..{ns - 1})")
    elif kind == 'pop':
        p['op'] = v
    elif kind == 'pfocus':
        if p['space'] is None:
            print("[task] pick a space before entering a value")
            return True
        host.task_val_active = True
        host.task_val_buf = ''
        print(f"[task] value for {p['space']}: type digits, Enter to set, Esc to cancel")
    elif kind == 'pval':
        ns = next((int(c["ns"]) for c in catalog if c["name"] == p['space']), None)
        hi = (ns - 1) if ns else 999
        p['value'] = max(0, min(hi, p['value'] + v))
    elif kind == 'addlit':
        v = p['value']
        v = tuple(v) if isinstance(v, (list, tuple, set, frozenset)) else int(v)
        add_literal(host, p['space'], p['op'], v)
    else:
        return False
    return True


def panel_height(clauses):
    return 26 + 18 + 22 + 15 + 20 + (len(clauses) + 1) * ROW_H + 6 + 34 + PAD


def panel_hit(els, x, y):
    for el in els:
        if el["action"] is None:
            continue
        if abs(x - el["cx"]) <= el["w"] / 2 and abs(y - el["cy"]) <= el["h"] / 2:
            return el["action"]
    return None


def panel_apply(host, action):
    """Apply a panel action to the host's task state. Returns True if the click was consumed."""
    if action is None:
        return False
    kind, i = action
    if kind == 'sel':
        host.task_clause_ix = i
        print(f"[task] active clause -> [{i}]")
    elif kind == 'add':
        host.task_clauses.append([])
        host.task_clause_ix = len(host.task_clauses) - 1
        print(f"[task] new AND clause [{host.task_clause_ix}]")
    elif kind == 'del':
        if len(host.task_clauses) <= 1:
            host.task_clauses = [[]]                    # always keep one empty clause to author into
            host.task_clause_ix = 0
            print("[task] cleared the only clause (kept one empty)")
        else:
            host.task_clauses.pop(i)
            host.task_clause_ix = min(host.task_clause_ix, len(host.task_clauses) - 1)
            print(f"[task] deleted clause [{i}]")
    return True


def cell_clause_map(clauses):
    """cell -> [clause indices that contain (X == cell)], for drawing goal cells on the grid.
    A cell can belong to several clauses, so callers get the list and can show the first / ring them."""
    out = {}
    for i, cl in enumerate(clauses):
        for (space, op, val) in cl:
            if space == 'X' and op == '==':
                out.setdefault(int(val), []).append(i)
    return out


def draw_panel(host):
    """Draw the T panel in the left tool strip. arcade calls confined here, mirroring trigger_gui.draw
    (rects first, then text, per-element text failures swallowed so one bad label cannot kill a frame).
    Text writes are change-guarded: assigning .text/.font_size/.color re-lays out a pyglet document,
    and doing that every frame for static content is what made the level panels cost ~40 ms/frame."""
    import arcade
    W = _panel_w(host)
    H = _panel_h(host)
    arcade.draw_rectangle_filled(W / 2, H / 2, W, H, (18, 21, 30, 255))
    els = panel_layout(host.task_clauses, host.task_clause_ix, W, H, host_catalog(host), [host])
    for el in els:
        k = el["kind"]
        if k in ('clause', 'add', 'del', 'chip', 'field'):
            base = el.get("col") or (60, 66, 80)
            fill = base if el["filled"] else (42, 47, 60)
            arcade.draw_rectangle_filled(el["cx"], el["cy"], el["w"], el["h"], fill)
            arcade.draw_rectangle_outline(el["cx"], el["cy"], el["w"], el["h"], base, 1)
    for i, el in enumerate(els):
        try:
            t = _text(host, i)
            col = (arcade.color.WHITE if el["kind"] in ('title', 'clause', 'add', 'del')
                   else (el.get("col") or arcade.color.LIGHT_GRAY))
            _set(t, str(el["label"]), el.get("x", el["cx"]), el["cy"] - 6, el.get("size", 11), col)
            t.draw()
        except Exception:
            pass


def draw_goal_cells(host, grid_x0, sq_w, sq_h, spacing, li_to_coord):
    """Ring every G-tagged cell on the grid in its CLAUSE's colour.

    Without this, G-tagging is invisible: P places a sprite you can see, G only mutated a list. The
    colour is the disjunction made visible -- red cells are one way to satisfy the task, blue another.
    A cell in several clauses gets concentric rings, one per clause."""
    import arcade
    cmap = cell_clause_map(host.task_clauses)
    if not cmap:
        return
    for cell, idxs in cmap.items():
        try:
            row, col = li_to_coord[cell]
        except (KeyError, IndexError):
            continue                                   # stale tag (grid resized) -- skip, do not crash
        cx = spacing * col + sq_w / 2 + grid_x0
        cy = spacing * row + sq_h / 2
        for d, ci in enumerate(idxs):
            inset = 3 + 4 * d
            arcade.draw_rectangle_outline(cx, cy, sq_w - inset * 2, sq_h - inset * 2,
                                          clause_color(ci), 3)


def _panel_w(host):
    return getattr(host, 'grid_x0', 300)


def _panel_h(host):
    return getattr(host, 'content_h', getattr(host, 'height', 700))


def _text(host, i, pool='_task_text_pool'):
    import arcade
    p = getattr(host, pool, None)
    if p is None:
        p = []; setattr(host, pool, p)
    while len(p) <= i:
        p.append(arcade.Text("", 0, 0, arcade.color.WHITE, 11))
    return p[i]


def _set(t, label, x, y, size, color):
    """Assign only what changed -- see levels_gui._set_text for why this matters."""
    if t.text != label: t.text = label
    if t.x != x: t.x = x
    if t.y != y: t.y = y
    if t.font_size != size: t.font_size = size
    if tuple(t.color)[:3] != tuple(color)[:3]: t.color = color


def click(host, x, y):
    """Route a click in the T panel. Returns True if consumed."""
    W, H = _panel_w(host), _panel_h(host)
    if x > W:
        return False                                   # on the grid, not the panel
    cat = host_catalog(host)
    a = panel_hit(panel_layout(host.task_clauses, host.task_clause_ix, W, H, cat, [host]), x, y)
    return derived_apply(host, a) or custom_apply(host, a, cat) or panel_apply(host, a)


def host_catalog(host):
    """Every space in the world, from world_spaces.catalog_from_placement -- the SAME catalog the
    affordance panel uses, so the task picker cannot offer a space the solver will not build."""
    try:
        import world_spaces as ws
        return ws.catalog_from_placement(
            host.internal_goalstates_2_type_ind, host.boolean_states_2_type_ind,
            nis=getattr(host, 'nis', 15), nis_by_type=getattr(host, 'nis_by_type', {}),
            x_ns=int(getattr(host, 'grid', [[0]]).size) or None,
            buttonColorIdx=getattr(host, 'buttonColorIdx', {}),
            walls=getattr(host, 'buttonToWall', {}),
            npc_states=getattr(host, 'npc_states', set()),
            active_int_types=getattr(host, 'active_int_types', set()))['spaces']
    except Exception as e:
        print(f"[task] catalog unavailable ({e})")
        return []


def add_literal(host, space, op, value, label=None):
    """Toggle a literal in the ACTIVE clause. Shared by every G-click target -- grid cell, internal
    chain-bar state, inventory item -- so they all land in one place with one set of semantics.

    Returns the action taken ('added' / 'removed') for the caller's console echo."""
    cl = host.task_clauses[host.task_clause_ix]
    lit = (space, op, value)
    if lit in cl:
        cl.remove(lit)
        act = 'removed'
    else:
        cl.append(lit)
        act = 'added'
    shown = label or f"{space} {op} {fmt_val(value)}"
    body = " AND ".join(f"{a} {o} {fmt_val(v)}" for a, o, v in cl) or "(empty)"
    print(f"[task] clause {host.task_clause_ix}: {act} {shown}   ->  {body}")
    return act


def tagged_clauses(clauses, space, op, value):
    """Indices of the clauses containing exactly this literal. Lets the renderer ask 'is hunger==7 in
    the task, and in which clause(s)?' so a bar segment / inventory slot can be ringed in that
    clause's colour -- the same visual language as the grid's goal rings."""
    lit = (space, op, value)
    return [i for i, cl in enumerate(clauses) if lit in cl]


def draw_state_rings(host, cx, cy, w, h, idxs, thickness=3):
    """Concentric clause-coloured rings around one widget (a bar segment or an inventory slot).
    Shared with the grid rings so every G-taggable thing highlights identically."""
    import arcade
    for d, ci in enumerate(idxs):
        inset = 2 + 4 * d
        arcade.draw_rectangle_outline(cx, cy, max(2.0, w - inset * 2), max(2.0, h - inset * 2),
                                      clause_color(ci), thickness)


def cycle_binary(host, space, label=None):
    """Click a 2-state (bit) space to cycle its literal in the active clause:

        untagged  ->  space == 1 (held)  ->  space == 0 (NOT held)  ->  untagged

    A bit space needs this because, unlike a chain bar, the widget is the ITEM, not a state -- clicking
    it cannot say WHICH value you mean. Without the cycle there was no way to author 'not holding the
    key' at all, and no way to tell 1 from 0 once authored."""
    cl = host.task_clauses[host.task_clause_ix]
    one, zero = (space, '==', 1), (space, '==', 0)
    if one in cl:
        cl.remove(one); cl.append(zero); act, val = 'requires NOT', 0
    elif zero in cl:
        cl.remove(zero); act, val = 'cleared', None
    else:
        cl.append(one); act, val = 'requires', 1
    shown = label or space
    body = " AND ".join(f"{a} {o} {v}" for a, o, v in cl) or "(empty)"
    print(f"[task] clause {host.task_clause_ix}: {act} {shown}"
          + (f" == {val}" if val is not None else "") + f"   ->  {body}")
    return val


def binary_tags(clauses, space):
    """[(clause_index, required_value), ...] for a bit space -- what the renderer badges on the slot."""
    out = []
    for i, cl in enumerate(clauses):
        for (sp, op, v) in cl:
            if sp == space and op == '==' and v in (0, 1):
                out.append((i, v))
    return out


def draw_binary_badge(host, cx, cy, w, h, tags):
    """Ring a bit-space widget per clause AND stamp the required value, so 1 (held) and 0 (not held)
    are distinguishable at a glance. A ring alone cannot show which -- that was the ambiguity."""
    import arcade
    draw_state_rings(host, cx, cy, w, h, [i for i, _ in tags])
    if not tags:
        return
    txt = "/".join(str(v) for _, v in tags)
    bx, by = cx + w / 2 - 6, cy + h / 2 - 6
    col = clause_color(tags[0][0])
    arcade.draw_rectangle_filled(bx, by, 15, 13, (18, 21, 30))
    arcade.draw_rectangle_outline(bx, by, 15, 13, col, 1)
    if not hasattr(host, '_task_badge_pool'):
        host._task_badge_pool = {}
    t = host._task_badge_pool.get((cx, cy))
    if t is None:
        t = arcade.Text("", 0, 0, arcade.color.WHITE, 9, bold=True)
        host._task_badge_pool[(cx, cy)] = t
    _set(t, txt, bx - 4 * len(txt) / 1.6, by - 4, 9, col)
    t.draw()


# -------------------------------------------------------------------------------------------------
# Numeric value entry. Mirrors levels_gui's save-as prompt: MODAL while focused, so typing digits
# cannot also fire the editor's single-key modes (w = wall, g = goal...). Setup.on_key_press must
# call value_key FIRST and return when it consumes.
# -------------------------------------------------------------------------------------------------
def value_active(host):
    return bool(getattr(host, 'task_val_active', False))


def _commit_value(host, catalog):
    p = _pick(host)
    buf = getattr(host, 'task_val_buf', '')
    host.task_val_active = False
    if not buf:
        print("[task] value unchanged")
        return
    ns = next((int(c["ns"]) for c in catalog if c["name"] == p['space']), None)
    lo, hi = 0, (ns - 1) if ns is not None else 10 ** 6
    parts = [q for q in buf.split(',') if q != '']
    if not parts:
        print("[task] value unchanged")
        return
    vals, clamped = [], False
    for q in parts:
        v = int(q)
        if not (lo <= v <= hi):
            v = max(lo, min(hi, v)); clamped = True
        if v not in vals:
            vals.append(v)
    if clamped:
        print(f"[task] value(s) outside {p['space']} {lo}..{hi} -- clamped")
    if p['op'] == 'in':
        p['value'] = tuple(vals)                 # tuple: hashable-ish and compares by value
    else:
        if len(vals) > 1:
            print(f"[task] {p['op']!r} takes ONE value -- using {vals[0]} (use 'in' for a set)")
        p['value'] = vals[0]
    print(f"[task] value = {fmt_val(p['value'])}")


def value_key(host, key, modifiers, catalog=None):
    """Handle a key while the value field is focused. Returns True if consumed (caller must return)."""
    import arcade
    if not value_active(host):
        return False
    if key == arcade.key.ESCAPE:
        host.task_val_active = False
        print("[task] value entry cancelled")
    elif key == arcade.key.BACKSPACE:
        host.task_val_buf = getattr(host, 'task_val_buf', '')[:-1]
    elif key in (arcade.key.ENTER, arcade.key.RETURN, arcade.key.NUM_ENTER):
        _commit_value(host, catalog if catalog is not None else host_catalog(host))
    elif arcade.key.KEY_0 <= key <= arcade.key.KEY_9:        # fallback when Setup.TEXT_INPUT is off
        host.task_val_buf = (getattr(host, 'task_val_buf', '') + chr(ord('0') + key - arcade.key.KEY_0))[:4]
    elif arcade.key.NUM_0 <= key <= arcade.key.NUM_9:
        host.task_val_buf = (getattr(host, 'task_val_buf', '') + chr(ord('0') + key - arcade.key.NUM_0))[:4]
    return True                                              # swallow everything else while focused


def value_text(host, text):
    """Feed a typed character in. Digits only -- the field is numeric."""
    if not value_active(host):
        return False
    # commas allowed so 'in' can take a SET: type 40,41,42
    if text and (text.isdigit() or text == ','):
        host.task_val_buf = (getattr(host, 'task_val_buf', '') + text)[:24]
    return True


# =================================================================================================
# DERIVED SETS -- goals as predicates over sets of states, not hand-enumerated cells.
#
# "thirst > 7" is already a predicate: it names a SET of states by a rule rather than by listing them.
# X had no such vocabulary -- only == (one cell) and in (cells you typed). But "be on the left side"
# and "be at a tree" are the same idea: `X in <computed set>`. So these need no new machinery, just
# constructors that fill the `in` value from the world's own structure.
#
# "at a tree" is precisely a FEATURE psi over X -- a function of state whose level-sets are a state
# abstraction. The affordance side already builds these as rows of H_psi; the same predicate used as
# an OBJECTIVE instead of as a guard is what a derived-set goal is. type_2_internal_goalstates[t] IS
# that level-set, already computed.
# =================================================================================================
def derived_sets(host):
    """[(label, space, frozenset(values)), ...] -- named predicates over the current world.

    Geometric ones come from the grid shape; feature ones from placement, so 'at a tree' tracks the
    trees you actually placed and needs no re-authoring when you move them."""
    import numpy as _np
    out = []
    try:
        grid = _np.asarray(host.grid)
        nrow, ncol = grid.shape
        cells = lambda pred: frozenset(int(r * ncol + c) for r in range(nrow) for c in range(ncol) if pred(r, c))
        out += [
            ("left half",   'X', cells(lambda r, c: c < ncol / 2.0)),
            ("right half",  'X', cells(lambda r, c: c >= ncol / 2.0)),
            ("top half",    'X', cells(lambda r, c: r < nrow / 2.0)),
            ("bottom half", 'X', cells(lambda r, c: r >= nrow / 2.0)),
            ("edge",        'X', cells(lambda r, c: r in (0, nrow - 1) or c in (0, ncol - 1))),
            ("interior",    'X', cells(lambda r, c: 0 < r < nrow - 1 and 0 < c < ncol - 1)),
        ]
    except Exception:
        pass
    try:
        import world_spaces as ws
        for t, cs in sorted(getattr(host, 'type_2_internal_goalstates', {}).items()):
            if cs:
                out.append((f"at {ws.TYPE_NAMES.get(int(t), f'type{t}')}", 'X',
                            frozenset(int(c) for c in cs)))
        for t, cs in sorted(getattr(host, 'type_2_boolean_states', {}).items()):
            if cs and int(t) != 4:                       # 4 = badger (an NPC, not an item site)
                out.append((f"at {ws.BOOL_TYPE_NAMES.get(int(t), f'item{t}')}", 'X',
                            frozenset(int(c) for c in cs)))
    except Exception:
        pass
    return [(lbl, sp, vs) for (lbl, sp, vs) in out if vs]


def derived_layout(host, W, y):
    """Chips for the derived sets. Clicking one adds `space in <that set>` to the active clause."""
    els = []
    if host is None:
        return els, y
    sets_ = derived_sets(host)
    if not sets_:
        return els, y
    els.append(dict(kind='label', label="DERIVED SETS  --  goals as predicates, not typed cells",
                    action=None, size=11, filled=False, col=(150, 190, 220),
                    x=PAD, cx=W / 2.0, cy=y, w=W - 2 * PAD, h=16, x0=PAD, x1=W - PAD))
    y -= 18
    chips = [(f"{lbl} ({len(vs)})", False, ('dset', i)) for i, (lbl, sp, vs) in enumerate(sets_)]
    e, y = _wrap(chips, PAD, y, W - PAD, w=88)
    return els + e, y


def derived_apply(host, action):
    """Apply a derived-set chip click. Returns True if consumed."""
    if action is None or action[0] != 'dset':
        return False
    sets_ = derived_sets(host)
    i = action[1]
    if 0 <= i < len(sets_):
        lbl, sp, vs = sets_[i]
        sym = _sym_for(host, lbl)
        if sym is not None:
            # feature/item sets are stored SYMBOLICALLY -> the goal follows the entity when you move it
            add_literal(host, sp, 'in', sym, label=f"{lbl} (tracks placement)")
        else:
            add_literal(host, sp, 'in', tuple(sorted(vs)), label=f"{lbl} [{len(vs)} states]")
    return True


def _sym_for(host, label):
    """('@feature'|'@item', type) for an 'at X' derived-set label, else None (geometric sets)."""
    import world_spaces as ws
    if not label.startswith("at "):
        return None
    nm = label[3:]
    for t, n in ws.TYPE_NAMES.items():
        if n == nm:
            return (SYM_FEATURE, int(t))
    for t, n in ws.BOOL_TYPE_NAMES.items():
        if n == nm:
            return (SYM_ITEM, int(t))
    return None


# =================================================================================================
# SYMBOLIC VALUES -- "at the key", not "at cell 40".
#
# Clicking a key with G stores X == <the cell the key happens to occupy>. Move the key and the goal
# stays behind, because it was never about the key: it was a coordinate that had a key on it. Holding
# the key (key_red == 1) is a different statement again. Neither can say "meet me at the yellow car".
#
# Fix without restructuring: keep the literal SYMBOLIC and resolve it at USE time.
#     ('X', 'in', ('@feature', 0))  ->  cells of feature type 0, looked up when evaluated
#     ('X', 'in', ('@item', 1))     ->  cells holding items of boolean type 1
# Storage stays a plain tuple, so saved levels stay JSON-able; only the resolution point moves. Every
# consumer (validate / satisfied / summarize / playtest) calls resolve() first, so a symbolic literal
# is concrete by the time any gate is built.
#
# LIMIT: this tracks an entity's PLACEMENT, so the goal follows the key when you move it in the editor.
# It cannot track an entity whose position changes during PLAY -- that is a relation between two state
# spaces (X == car_position), which needs a relational literal. trigger_rules already has `same` guards
# for exactly that shape on the affordance side, so it is the natural extension when a moving object
# exists to point at.
# =================================================================================================
SYM_FEATURE, SYM_ITEM = '@feature', '@item'


def is_symbolic(value):
    return isinstance(value, tuple) and len(value) == 2 and value[0] in (SYM_FEATURE, SYM_ITEM)


def sym_label(value):
    """Human name for a symbolic value, e.g. '@ tree' -- what the clause row shows."""
    import world_spaces as ws
    kind, t = value
    if kind == SYM_FEATURE:
        return f"@{ws.TYPE_NAMES.get(int(t), f'type{t}')}"
    return f"@{ws.BOOL_TYPE_NAMES.get(int(t), f'item{t}')}"


def resolve_value(host, value):
    """Expand a symbolic value against the CURRENT placement; pass anything else through."""
    if not is_symbolic(value) or host is None:
        return value
    kind, t = value
    src = (getattr(host, 'type_2_internal_goalstates', {}) if kind == SYM_FEATURE
           else getattr(host, 'type_2_boolean_states', {}))
    cells = src.get(int(t), src.get(t, []))
    return tuple(sorted(int(c) for c in cells))


def resolve(clauses, host):
    """Clauses with every symbolic value expanded. Call before building gates or evaluating."""
    return [[(sp, op, resolve_value(host, v)) for (sp, op, v) in cl] for cl in clauses]


# =================================================================================================
# HUMAN DESCRIPTION -- render a spec as prose for the browser page / a level blurb.
#
# DERIVED, never stored. The description is a function of task_clauses + placement, so caching it in
# the .obj would create a second copy free to go stale the moment you move a tree or edit a clause.
# It is generated at export time from the same clauses the game evaluates, so it cannot disagree with
# what actually wins. (`notes` in the manifest stays for AUTHORIAL prose -- why the level exists --
# which is not derivable from anything.)
#
# Humanize BEFORE resolve(): a symbolic value renders as "where the key is" rather than as the cell
# it happens to sit on today.
# =================================================================================================
def _cellname(v, ncol=None):
    return f"cell {v}" + (f" (r{v // ncol}, c{v % ncol})" if ncol else "")


def humanize_literal(lit, ncol=None, spaces=None):
    """One literal as a phrase."""
    sp, op, v = lit
    ns = (spaces or {}).get(sp)
    is_bit = (ns == 2)
    if is_symbolic(v):
        return f"be where the {sym_label(v).lstrip('@')} is" if sp == 'X' else f"{sp} in {sym_label(v)}"
    if sp == 'X':
        if op == '==':
            return f"be at {_cellname(v, ncol)}"
        if op == 'in':
            vals = list(v) if isinstance(v, (list, tuple, set, frozenset)) else [v]
            if len(vals) == 1:
                return f"be at {_cellname(vals[0], ncol)}"
            return f"be at one of {len(vals)} cells"
        return f"X {op} {fmt_val(v)}"
    if is_bit and op == '==':
        return f"{'be holding' if v == 1 else 'NOT be holding'} {sp}"
    words = {'>=': 'at least', '<=': 'at most', '>': 'above', '<': 'below',
             '==': 'exactly', '!=': 'not'}
    if op == 'in':
        return f"{sp} in {fmt_val(v)}"
    return f"{sp} {words.get(op, op)} {fmt_val(v)}"


def humanize(clauses, ncol=None, spaces=None):
    """The whole objective as one or more sentences. Empty -> a clear 'no objective'."""
    cl = [c for c in clauses if c]
    if not cl:
        return "No objective set -- this level cannot be won."
    parts = [" and ".join(humanize_literal(l, ncol, spaces) for l in c) for c in cl]
    if len(parts) == 1:
        return "Win by: " + parts[0] + "."
    return "Win by EITHER: " + "  ...OR: ".join(parts) + "."


def humanize_constraints(constraint_cells, spaces, ncol=None):
    """The lose conditions, in the same voice as the objective."""
    out = []
    n = len(constraint_cells or [])
    if n:
        out.append(f"stepping on {'a hazard cell' if n == 1 else f'any of {n} hazard cells'}")
    for nm, info in sorted((spaces or {}).items()):
        if info.get('lethal'):
            out.append(f"letting {nm} reach 0")
    if not out:
        return "Nothing can kill you in this level."
    return "You LOSE by: " + ", or ".join(out) + "."
