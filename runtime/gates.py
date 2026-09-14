"""
gates.py -- ergonomic boolean gates for authoring affordance features.

Author a feature as a boolean expression over parent variables (and the free 'action'), using & | ~ ;
predicates can be anything that returns True/False (equality, thresholds, is_even, relations, ...).
Then:
  * compile(...) evaluates the gate over the domain -> the 0/1 H_psi indicator the solver consumes,
  * the gate is KEPT (Affordance.spec), so describe() prints exactly what you wrote (intent, not a
    reverse-engineered cover),
  * eval(state) tests it at a single concrete state WITHOUT materializing the dense array (factored).

More general than DNF: arbitrary AND/OR/NOT nesting. (A single conjunction / a flat OR-of-ANDs are the
DNF special cases.)
"""
import itertools
import numpy as np


class Gate:
    """Base boolean node. Combine with & (AND), | (OR), ~ (NOT)."""
    def __and__(self, o):  return And(self, o)
    def __or__(self, o):   return Or(self, o)
    def __invert__(self):  return Not(self)
    # subclasses provide: eval(state_dict, action) -> bool ; vars() -> set ; nnf() -> Gate ; __repr__

    # ---- normal forms (structural; each leaf Cond / its negation is treated as an atomic literal) ----
    def to_dnf(self):
        """Return an equivalent DNF Gate (OR of ANDs of literals). May blow up exponentially."""
        terms = _dedup_terms([_dedup_lits(t) for t in _dnf_lol(self.nnf())])
        return _mk_or([_mk_and(t) for t in terms])
    def to_cnf(self):
        """Return an equivalent CNF Gate (AND of ORs of literals). May blow up exponentially."""
        clauses = _dedup_terms([_dedup_lits(c) for c in _cnf_lol(self.nnf())])
        return _mk_and([_mk_or(c) for c in clauses])
    def dnf_terms(self):
        """The DNF as a list of terms, each a list of literal reprs (for counting / inspection)."""
        return [[repr(l) for l in t] for t in _dedup_terms([_dedup_lits(t) for t in _dnf_lol(self.nnf())])]
    def cnf_clauses(self):
        """The CNF as a list of clauses, each a list of literal reprs (for counting / inspection)."""
        return [[repr(l) for l in c] for c in _dedup_terms([_dedup_lits(c) for c in _cnf_lol(self.nnf())])]

class Cond(Gate):
    """A leaf predicate over one or more variables. `test(*values)` returns True/False; `label` is what
    describe() prints. Use 'action' as a variable name to read the free BL action."""
    def __init__(self, variables, test, label):
        self.variables = tuple(variables); self.test = test; self.label = label
    def eval(self, sd, action=None):
        return bool(self.test(*[(action if v == "action" else sd[v]) for v in self.variables]))
    def vars(self): return set(self.variables)
    def nnf(self): return self
    def __repr__(self): return self.label

class And(Gate):
    def __init__(self, *parts): self.parts = parts
    def eval(self, sd, action=None): return all(p.eval(sd, action) for p in self.parts)
    def vars(self): return set().union(*[p.vars() for p in self.parts])
    def nnf(self): return And(*[p.nnf() for p in self.parts])
    def __repr__(self): return "(" + " AND ".join(map(repr, self.parts)) + ")"

class Or(Gate):
    def __init__(self, *parts): self.parts = parts
    def eval(self, sd, action=None): return any(p.eval(sd, action) for p in self.parts)
    def vars(self): return set().union(*[p.vars() for p in self.parts])
    def nnf(self): return Or(*[p.nnf() for p in self.parts])
    def __repr__(self): return "(" + " OR ".join(map(repr, self.parts)) + ")"

class Not(Gate):
    def __init__(self, part): self.part = part
    def eval(self, sd, action=None): return not self.part.eval(sd, action)
    def vars(self): return self.part.vars()
    def nnf(self):
        p = self.part
        if isinstance(p, Cond):  return self                                    # negated literal: stop here
        if isinstance(p, Not):   return p.part.nnf()                            # double negation
        if isinstance(p, And):   return Or(*[Not(x).nnf() for x in p.parts])    # De Morgan
        if isinstance(p, Or):    return And(*[Not(x).nnf() for x in p.parts])   # De Morgan
        return self
    def __repr__(self): return "NOT " + repr(self.part)


# ----- normal-form machinery: distribute over an NNF tree into a list-of-lists -----------------------
# A "literal" is a Cond or a Not(Cond). DNF list-of-lists = OR of terms, each an AND-list of literals;
# CNF list-of-lists = AND of clauses, each an OR-list of literals. The two builders are exact duals:
# for the "same" connective you take a UNION of child lists; for the "opposite" one, a cartesian product.
def _is_literal(node):  return isinstance(node, Cond) or (isinstance(node, Not) and isinstance(node.part, Cond))

def _dnf_lol(node):     # node is in NNF
    if _is_literal(node): return [[node]]
    if isinstance(node, Or):                                   # OR -> union of term-lists
        out = []
        for p in node.parts: out += _dnf_lol(p)
        return out
    if isinstance(node, And):                                  # AND -> cartesian product of terms
        result = [[]]
        for p in node.parts:
            pt = _dnf_lol(p); result = [r + t for r in result for t in pt]
        return result
    raise TypeError(f"non-NNF node in _dnf_lol: {node!r}")

def _cnf_lol(node):     # dual of _dnf_lol
    if _is_literal(node): return [[node]]
    if isinstance(node, And):                                  # AND -> union of clause-lists
        out = []
        for p in node.parts: out += _cnf_lol(p)
        return out
    if isinstance(node, Or):                                   # OR -> cartesian product of clauses
        result = [[]]
        for p in node.parts:
            pc = _cnf_lol(p); result = [r + c for r in result for c in pc]
        return result
    raise TypeError(f"non-NNF node in _cnf_lol: {node!r}")

def _dedup_lits(group):                                        # drop repeated literals within a term/clause
    seen = set(); out = []
    for lit in group:
        k = repr(lit)
        if k not in seen: seen.add(k); out.append(lit)
    return out

def _dedup_terms(groups):                                      # drop duplicate terms/clauses (order-insensitive)
    seen = set(); out = []
    for g in groups:
        key = frozenset(repr(l) for l in g)
        if key not in seen: seen.add(key); out.append(g)
    return out

def _mk_and(lits):   return lits[0] if len(lits) == 1 else And(*lits)
def _mk_or(terms):   return terms[0] if len(terms) == 1 else Or(*terms)

# ----- leaf constructors (single-variable predicates; the last two are the general escape hatches) -----
def Eq(v, val):   return Cond((v,), lambda x: x == val,  f"{v}=={val!r}")
def Ne(v, val):   return Cond((v,), lambda x: x != val,  f"{v}!={val!r}")
def In(v, vals):  S = set(vals); return Cond((v,), lambda x: x in S, f"{v} in {sorted(S)}")
def Ge(v, t):     return Cond((v,), lambda x: x >= t,    f"{v}>={t}")
def Le(v, t):     return Cond((v,), lambda x: x <= t,    f"{v}<={t}")
def Gt(v, t):     return Cond((v,), lambda x: x > t,     f"{v}>{t}")
def Lt(v, t):     return Cond((v,), lambda x: x < t,     f"{v}<{t}")
def P(v, fn, label=None):        return Cond((v,), fn, label or f"{getattr(fn,'__name__','pred')}({v})")   # ANY predicate
def Rel(variables, fn, label):   return Cond(tuple(variables), fn, label)                                   # relational (multi-var)
def OR(*gs):  return Or(*gs)
def AND(*gs): return And(*gs)


def compile_indicator(gate, axes, sizes):
    """Evaluate a gate over the whole domain -> 0/1 array of shape `sizes`; axes[i] names axis i
    (use 'action' for the free-action axis). This is the 'translate the formula into 1s and 0s' step."""
    arr = np.zeros(sizes, dtype=float)
    for combo in itertools.product(*[range(s) for s in sizes]):
        sd = {axes[i]: combo[i] for i in range(len(axes))}
        arr[combo] = 1.0 if gate.eval(sd, sd.get("action")) else 0.0
    return arr


def make_affordance(driven, parents, rules, default_alpha, na_driven, dims, has_action=True):
    """Build a state_spaces.Affordance from authored gates and KEEP the spec for describe().
      rules   : ordered list of (gate, alpha) -- earlier rules win on overlap (first-match priority).
      dims    : {var_name: n_values} for every parent (and 'action' if has_action).
      returns : an Affordance whose H_psi/H_alpha are compiled from the gates, with `.spec` attached.
    """
    import state_spaces as ss
    axes = (["action"] if has_action else []) + list(parents)
    sizes = [dims[a] for a in axes]
    n_feat = len(rules) + 1                                        # one feature per rule + a 'none'
    H_psi = np.zeros([n_feat] + sizes)
    covered = np.zeros(sizes)
    for i, (gate, _a) in enumerate(rules):
        fire = np.clip(compile_indicator(gate, axes, sizes) - covered, 0, 1)   # first-match priority
        H_psi[i] = fire
        covered = np.clip(covered + fire, 0, 1)
    H_psi[-1] = 1.0 - covered                                     # 'none' = no rule fired
    H_alpha = np.zeros((na_driven, n_feat))
    for i, (_g, alpha) in enumerate(rules):
        H_alpha[alpha, i] = 1.0
    H_alpha[default_alpha, -1] = 1.0
    spec = [(alpha, gate) for (gate, alpha) in rules]             # describe() prints these
    aff = ss.Affordance(driven, parents, H_psi, H_alpha, spec=spec)   # spec is now a declared __init__ arg
    aff.default_alpha = default_alpha
    return aff


if __name__ == "__main__":
    # demo: author the priced-door gate, compile it, and describe from the spec
    SPECIAL, CLOSED, PRICE = 0, 0, 2
    pay = In("X", {2}) & Ge("Y", PRICE) & Eq("Door", CLOSED) & Eq("action", SPECIAL)
    aff = make_affordance("Door", ["X", "Y", "Door"], [(pay, 1)], default_alpha=0,
                          na_driven=2, dims={"action": 6, "X": 8, "Y": 3, "Door": 2})
    print("authored gate:", pay)
    print(aff.describe())
    # a predicate-based + disjunctive example
    g2 = (In("X", {2, 4}) & Ge("Y", PRICE)) | (P("X", lambda x: x % 2 == 0, "is_even(X)") & Eq("Y", 0))
    aff2 = make_affordance("Door", ["X", "Y"], [(g2, 1)], 0, 2, {"X": 8, "Y": 3}, has_action=False)
    print("\nauthored gate:", g2)
    print(aff2.describe())

    # --- to_dnf / to_cnf : which shape is compact, which blows up --------------
    print("\n--- normal forms ---")
    A, Bb, Cc, D, E, Fp = (Eq("a", 1), Eq("b", 1), Eq("c", 1), Eq("d", 1), Eq("e", 1), Eq("f", 1))
    cases = {
        "flat AND  a&b&c":            A & Bb & Cc,
        "CNF-shaped (a|b)&(c|d)&(e|f)": (A | Bb) & (Cc | D) & (E | Fp),
        "DNF-shaped (a&b)|(c&d)|(e&f)": (A & Bb) | (Cc & D) | (E & Fp),
        "mixed a&(b|c)&~d":           A & (Bb | Cc) & ~D,
        "neither (a|b)&(c|(d&e))":    (A | Bb) & (Cc | (D & E)),
    }
    for name, g in cases.items():
        print(f"\n{name}")
        print(f"   tree: {g}")
        print(f"   DNF ({len(g.dnf_terms())} terms):   {g.to_dnf()}")
        print(f"   CNF ({len(g.cnf_clauses())} clauses): {g.to_cnf()}")

    # correctness self-check: DNF and CNF agree with the tree on every assignment
    import itertools as _it
    ok = True
    for g in cases.values():
        vs = sorted(g.vars()); dnf, cnf = g.to_dnf(), g.to_cnf()
        for combo in _it.product([0, 1], repeat=len(vs)):
            sd = dict(zip(vs, combo))
            if not (g.eval(sd) == dnf.eval(sd) == cnf.eval(sd)): ok = False
    print(f"\nto_dnf / to_cnf match the tree on all assignments: {ok}")
