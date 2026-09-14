"""
tutorial_server.py -- local web server for the OKBE tutorial.

WHY A SERVER (rather than a self-contained page like levels/export/play.html):
play.html ships PRECOMPUTED tables and only replays them. The tutorial has to SOLVE -- feasibility
iteration and value iteration -- while you edit. Those live in TGMDP_methods / utilities and are the
implementations every other result in this repo was produced with. Re-implementing them in JS would
mean two copies of the OKBE numerics to keep in agreement, so instead the browser is the UI and
Python stays the single source of truth.

Importing the solver stack costs ~38s cold / ~1.4s warm (matplotlib + scipy dominate), which is far
too slow per request but irrelevant once at startup -- hence a long-lived process rather than a CLI.

Stdlib only: no Flask, no npm, no CDN. Run it and open the printed URL.

    python3 tutorial_server.py [--port 8765] [--no-browser]
"""
import argparse
import http.server
import json
import os
import re
import socket
import socketserver
import subprocess
import sys
import threading
import time
from shared_level import make_level, read_level

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "tutorial", "static")
# feas_two_phase / feas_two_phase_sparse / feas_thresh_sparse live in llm_solvers/, grouped there
# since they're the custom two-phase solver stack (not TGMDP_methods.py, Tom's originals) --
# needed on sys.path explicitly since they're no longer siblings of this file.
sys.path.insert(0, os.path.join(HERE, "llm_solvers"))
IMAGES = os.path.join(HERE, "OKBE_code", "Images")
LEVELDIR = os.path.join(HERE, "tutorial", "levels")
LOGDIR = os.path.join(HERE, "tutorial", "logs")
LOGFILE = os.path.join(LOGDIR, "session.log")
_loglock = threading.Lock()
# Serialize API work, including response encoding: another panel's SR/JSON work must
# not overlap a timed solve. GETs remain responsive; queue time is outside solve timing.
_compute_lock = threading.Lock()


def log(kind, **fields):
    """Append one JSON line to tutorial/logs/session.log.

    Everything goes in one file, browser and server together, so the ordering of a failure is
    visible: a "failed to fetch" in the page is only diagnosable next to whether the server saw
    the request at all. Never raises -- logging must not be able to break a request."""
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        rec = {"t": round(time.time(), 3), "kind": kind}
        rec.update(fields)
        with _loglock:
            with open(LOGFILE, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass

# The toolbar palette. `img` names a file in OKBE_code/Images and is served from /img/ -- the SAME
# art the arcade editor uses, so the two editors look alike. Unlike levels/export/play.html there is
# no base64 inlining here: a server can just serve the file. `glyph` is for tools with no artwork.
# `key` is the keyboard shortcut that selects the tool. Unlike the arcade editor these are
# SELECT, not hold-to-place: press once, then click cells. Keys are chosen to match the thing
# rather than the field name -- M for mountain, F for fire -- since that is what is on screen.
# `move` is first and is the resting tool: with no key held, dragging rearranges the world. The
# others are HOLD-to-place -- press and hold the key, click/drag, release to fall back to move.
TOOLS = [
    {"id": "move",       "label": "move",  "key": "V", "glyph": "\u271b"},
    {"id": "wall",       "label": "wall",  "key": "M", "img": "mountain.png"},
    {"id": "start",      "label": "start", "key": "S", "img": "Stoffel_pic_right.png"},
    # Goal has no art of its own -- a vector check (drawCheck in editor.js) reads as "success"
    # regardless of what a world's internal spaces are named, and nothing in FEATURE_TYPES/
    # ITEM_TYPES below is a plausible stand-in for it. Constraint's DEFAULT look is fire1.png,
    # but only when the cell has no feature/item of its own -- when it does, that art becomes
    # the cell's main content and the constraint shrinks to a small red-X corner badge instead
    # (see editor.js's draw()), so it never actually competes with tree.png/lake.png/... for the
    # centre the way it would have if drawn there unconditionally.
    {"id": "goal",       "label": "goal",       "key": "G", "glyph": "✓", "color": "#3ea55a"},
    {"id": "constraint", "label": "constraint", "key": "F", "img": "fire1.png"},
    {"id": "erase",      "label": "erase", "key": "E", "glyph": "\u2715"},
    # The conditioned views (STOK, successor, string) plot FROM the start state, so there is no
    # separate "plot state" to place -- one marker, one meaning. Targets are still their own mode,
    # since they are final states rather than a starting one.
    {"id": "pick_xf",    "label": "targets", "key": "T", "glyph": "\u25c9"},
    # Pick a cell whose transition rows you want to edit by hand. Its own mode, because clicking a
    # cell to INSPECT it must not also place something on it.
    {"id": "pick_kernel", "label": "P(x'|x,a)", "key": "K", "glyph": "\u21dd"},
    # Place an internal/HL-space anchor (hunger, hydration, ... money) or an inventory item. WHICH
    # type gets placed is chosen from the type-chip rows in the aux panel (drawFeatureTypes /
    # drawItemTypes in editor.js), not from the tool itself -- same split as 'goal' + activeGoal.
    {"id": "feature",    "label": "feature", "key": "H", "glyph": "\u25c6"},
    {"id": "item",       "label": "item",    "key": "B", "glyph": "\u2726"},
    {"id": "green_badger", "label": "green badger", "key": "A", "img": "green_badger.png"},
    # Tag a cell/bar-segment/inventory-slot into the ACTIVE DNF task clause (Task tab). Its own
    # key rather than G: G already means "place a flat goal" (the group/kappa-solve system), a
    # different and older objective representation that this does not replace.
    {"id": "task",       "label": "clause",  "key": "C", "glyph": "\u00a7", "color": "#c98fe0"},
]
_dup = [t["key"] for t in TOOLS]
assert len(set(_dup)) == len(_dup), f"duplicate tool key bindings: {_dup}"

# Internal/HL-space anchor types placed on X (mirrors pygame_windows.py's `internal_goalstates_2_
# type_ind` scheme and world_spaces.py's TYPE_NAMES/NO_DEATH_TYPES, which stay the single source of
# truth for NAMES; `img`/`bar_img` are art paths only the two GUIs need, so they live here, same
# split as world_spaces.py keeping names separate from pygame_windows.py's own image dict).
FEATURE_TYPES = [
    {"type": 0, "label": "hunger",      "img": "tree.png",               "bar_img": "apple.png"},
    {"type": 1, "label": "hydration",   "img": "lake.png",               "bar_img": "water_drops.png"},
    {"type": 2, "label": "temperature", "img": "house.png",              "bar_img": "thermometer.png"},
    {"type": 3, "label": "money",       "img": "gold_coin.png",          "bar_img": "moneybag.png"},
    {"type": 4, "label": "potion",      "img": "small_potion_bottle.png","bar_img": "large_potion_bottle.png"},
    {"type": 5, "label": "battery",     "img": "battery_cyan.png",       "bar_img": "battery_cyan.png"},
]
# Inventory (boolean pickup) item types. Type 4 (badger) is an NPC in the arcade editor, not an
# inventory item, so it is left out of the placeable set here.
# Internal-space state counts go far higher than a grid dimension (a health/hydration space with
# ns=200 is exactly the case this exists for) so they get their OWN cap, not DIMS-style reuse.
# 500 keeps StateSpace_1D's dense P_a_s_s build (state_spaces.py: ns*ns*max_time < 5e7 at the
# default T_f=60) comfortably in its fast path: 500*500*60 = 1.5e7.
NIS_MIN, NIS_MAX = 2, 500

ITEM_TYPES = [
    {"type": 1, "label": "key",     "img": "key.png"},
    {"type": 2, "label": "flower",  "img": "flowers.png"},
    {"type": 3, "label": "honey",   "img": "honey.png"},
    {"type": 5, "label": "axe",     "img": "axe.png"},
    {"type": 6, "label": "canteen", "img": "wooden-canteen.png"},
]

# The solver stack is imported ONCE at startup, on the MAIN THREAD. It must be the main thread:
# something under TGMDP_methods installs a signal handler, and Python only permits that from the
# main thread -- importing it from a worker fails with "signal only works in main thread". So the
# HTTP server runs on a background thread instead and the import keeps the main one.
SOLVER = {"loaded": False, "error": None, "seconds": None}

# Bumped whenever the API surface changes (a new endpoint, a new request field). The page checks it
# and says so plainly -- otherwise a stale server shows up as an unrelated error three panels away
# ("Unexpected token '<'" is what a 404 HTML page looks like to JSON.parse).
API_VERSION = "26.transition-h.1"

# The root grid space has six actions; a goal's terminal action indexes into them.
# Named by the displacement they actually produce ON SCREEN, where the grid is drawn row 0 at the
# TOP and li = row * cols + col. The state space's own vertical convention is the opposite: its
# action 2 raises the row INDEX, which moves DOWN the screen, and its action 4 lowers it. Naming
# them by the state space's convention put a "up" label on a move that visibly goes down.
# test_action_labels.py pins every one of these against the kernel, so the two cannot drift apart.
ACTION_LABELS = ["interact", "stay", "down", "right", "up", "left"]
# (dr, dc) per action index, same order. None = no displacement.
ACTION_DELTA = [(0, 0), (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
N_ACTIONS = len(ACTION_LABELS)

# The algorithm menu, DECLARED rather than hard-coded in the page: the left panel builds its inputs
# from `params` below, so adding an algorithm here is all it takes for the UI to offer it.
#
# `params` are the algorithm's OWN knobs only. Anything that defines the WORLD -- the transition
# kernel, its slip probability, the walls, where the constraints are -- is a property of the world
# and lives in the editor panel, not here.
#
# Shared by feas2 and feas2s: two independent knobs for the same problem -- a route that is
# ~99.99% likely to succeed can lose the argmax to another route that is fractionally (1e-5 or
# less) safer, even though it is visibly longer, because kappa has no notion of path length.
#   kappa_tol      widens A*_x to "within epsilon of the max" so expected time (ET) gets to
#                  choose among near-ties, not just exact ones.
#   round_decimals rounds kappa to N decimals before A*_x is read off -- the SAME mechanism the
#                  plain 'feas' solver hardcodes at 4 -- so two routes that display as the same
#                  number (e.g. both 0.9999) are also treated as tied for the tie-break, without
#                  needing any separate epsilon at all.
# kappa_tol defaults to 0 (OFF, exact argmax-kappa policy). round_decimals defaults to 3, per
# explicit request -- matching the ORIGINAL 'feas' solver's own hardcoded precision, meant to fix
# a route that is a hair (sub-1e-5) safer routinely beating one that is visibly shorter (see
# feas_two_phase's own module docstring). THIS WAS TRIED ONCE BEFORE (alongside kappa_tol=1e-4)
# and reverted: widening the tied-action set made it possible for a STATE-ACTION goal's policy to
# pick a NON-TERMINATING action that merely ties the terminal one within tolerance, which does not
# just look wrong the way the up/right-arrow case did -- it makes eta_pos sum to EXACTLY 0, i.e.
# the option can be shown as never succeeding at all. Measured then: a 5-leg composed string with
# two state-action handoffs went from correctly succeeding (mass 0.85) to reporting zero success,
# from that default. kappa_tol stays OFF this time -- only round_decimals is defaulted on -- but
# round_decimals alone widens ties the same way (see its own CAUTION below) and was part of what
# triggered that failure originally, so the risk is not eliminated, only narrowed. If a world's
# option ever reports zero success where it visibly should not, check this slider first.
#   round_decimals's slider TOPS OUT at ROUND_DECIMALS_MAX rather than running away arbitrarily:
#   float64 carries about 15-17 significant decimal digits, so rounding a [0,1] probability beyond
#   that is not a finer distinction, it is rounding to noise already below the kernel's own
#   numerical precision. OFF at both ends -- 0 (the original sentinel) and the max (new) -- so
#   "turn it off" does not require remembering which end is special.
ROUND_DECIMALS_MAX = 15

# Shared by every two-phase solver (feas2, feas2s, feas_thresh_s): kappa and ET converge
# SEPARATELY (that is the whole point of "two-phase"), so they get separate convergence
# thresholds rather than one shared `epsilon` -- kappa is a bounded [0,1] probability that feeds
# the tied-action set directly (too loose here can flip which actions tie, the same failure mode
# round_decimals/kappa_tol already warn about); ET is an unbounded expected step count, where the
# same absolute number means a looser fraction of the answer the farther a state is from the
# goal. Both default to 1e-9, matching this codebase's original single shared epsilon, so nothing
# changes until one is actually moved. Both are genuinely worth loosening on a LARGE grid: the
# undiscounted kappa/ET recursions have no geometric contraction rate the way a discounted value
# function does (see value_world's own ihdc, which stops at delta < 0.1 and converges in a
# roughly grid-size-independent number of iterations for exactly that reason) -- convergence here
# instead costs iterations proportional to the grid's diameter, which 1e-9 pays for in full every
# time. Measured: a 40x40 grid's own solve time is dominated by this, not by kernel construction
# or by the eta rollout that follows it.
_CONVERGENCE_PARAMS = [
    {"id": "kappa_epsilon", "label": "kappa convergence  ε", "type": "range", "log": True,
     "min": 0.0, "max": 0.5, "logMin": 1e-9, "default": 1e-9,
     "help": "how tightly kappa must stop changing between iterations before phase 1 calls it "
             "converged. 0 = tightest (run until two iterations are floating-point identical, or "
             "the iteration cap). Looser trades exactness for speed on a large/slow-converging "
             "grid -- see round_decimals/kappa_tol for the SAME tie-set-widening effect this can "
             "also have if pushed too loose, including their CAUTION about state-action goals"},
    {"id": "ET_epsilon", "label": "expected-time convergence  ε", "type": "range", "log": True,
     "min": 0.0, "max": 100.0, "logMin": 1e-9, "default": 1e-9,
     "help": "how tightly expected time (ET) must stop changing before phase 2 calls it "
             "converged. 0 = tightest. ET is an expected STEP COUNT, not a probability, so this "
             "is an absolute number of steps, not a fraction -- loosen it once round_decimals/"
             "kappa_tol/kappa_epsilon are not the bottleneck and phase 2 itself is slow"},
]
_TWO_PHASE_PARAMS = [
    {"id": "kappa_tol", "label": "near-tie tolerance  ε", "type": "range", "log": True,
     "min": 0.0, "max": 1.0, "logMin": 1e-6, "default": 0.0,
     "help": "widens the tied-action set beyond EXACT kappa ties: an action counts as optimal "
             "if it is within epsilon of the max, so expected time -- not an arbitrary sub-1e-5 "
             "kappa difference -- decides between two routes that are both practically certain. "
             "0 = exact argmax kappa. CAUTION: nonzero can make a state-action goal's policy "
             "pick a non-terminating action tied within tolerance, so the option never actually "
             "completes -- checked to be safe on this world before raising it above 0"},
    {"id": "round_decimals", "label": "round kappa to N decimals", "type": "range",
     "min": 0, "max": ROUND_DECIMALS_MAX, "step": 1, "default": 3,
     "help": f"OFF (exact kappa, no rounding) at either end of the slider -- 0, or maxed out at "
             f"{ROUND_DECIMALS_MAX} decimals, past which float64 has nothing left to round. "
             "In between, rounds kappa to this many decimals before reading off the tied-action "
             "set -- the same effect as the near-tie tolerance above, including the same CAUTION "
             "about state-action goals, but expressed as matching precision instead of an "
             "absolute epsilon. Defaults to 3 (matching the original 'feas' solver's own "
             "hardcoded precision) -- move to 0 or the max to go back to exact kappa if an "
             "option ever reports zero success where it visibly should not"},
] + _CONVERGENCE_PARAMS
ALGORITHMS = [
    {
        "id": "feas", "label": "Feasibility iteration  (kappa)",
        "returns": "kappa",
        "about": "Probability of reaching the goal without violating a constraint. "
                 "TGMDP_methods.feas_iter_stationary_flat_dense",
        "params": [],
    },
    {
        "id": "feas2", "label": "Feasibility iteration, two-phase  (kappa)",
        "returns": "kappa",
        "about": "Same fixed point as the plain solver, but kappa, ET and eta are converged "
                 "SEPARATELY instead of one backup each per iteration. The interleaved version "
                 "stops on kappa's delta or a 4*sqrt(nx) cap, which (a) truncates kappa at high "
                 "noise -- 36 iterations where 170 are needed -- and (b) leaves ET flat where "
                 "kappa converges first, so the argmin tie-break falls to action 0, a self-loop, "
                 "and the policy parks. Here kappa converges, then minimum expected time is "
                 "solved over the fixed A*_x, then eta is built once under the final pi -- so the "
                 "eta returned is the eta OF that pi, which the original's is not. "
                 "feas_two_phase.feas_iter_two_phase (written by Claude, derived from Tom's).",
        "params": _TWO_PHASE_PARAMS,
    },
    {
        "id": "feas2s", "label": "Feasibility iteration, two-phase SPARSE  (kappa)",
        "returns": "kappa",
        "about": "The same numbers as the two-phase solver -- verified elementwise to 1e-12 -- on "
                 "a sparse representation. The kernel becomes one CSR matrix per action (a grid "
                 "row has 5 destinations, not nx), and eta is indexed by TERMINAL STATE rather "
                 "than by state: x_f is preserved by the propagation, so the only final states "
                 "that can carry mass are the ones the boundary conditions seed -- 2 columns of "
                 "81 on a one-goal one-fire world. 284x faster and 225x smaller at 15x15, and "
                 "both ratios grow with the grid. "
                 "feas_two_phase_sparse.feas_iter_two_phase_sparse (written by Claude).",
        "params": _TWO_PHASE_PARAMS,
    },
    {
        "id": "feas_thresh", "label": "Risk-thresholded feasibility  (kappa_theta)",
        "returns": "kappa",
        "about": "Two passes: solve, take kappa_c = P(ending in violation) per state, admit only "
        "states with kappa_c <= theta, then re-solve restricted to those. Answers 'is "
        "there a route within the risk budget', so it reads closer to 0/1 than the plain "
        "kappa, which is a survival probability. "
        "TGMDP_methods.feas_iter_stationary_flat_dense_thresh",
        "params": [
            {"id": "risk_thresh", "label": "risk threshold  theta", "type": "range",
             "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5,
             "help": "max tolerated probability of constraint violation. "
                     "0 = refuse all risk, 1 = accept anything"},
        ],
    },
    {
        "id": "feas_thresh_s", "label": "Risk-thresholded feasibility, sparse two-phase  (kappa_theta)",
        "returns": "kappa",
        "about": "Same two passes as the dense risk-thresholded solver above, but pass 1 (which "
        "supplies kappa_c) uses feas2s's own two-phase solver instead of the ORIGINAL "
        "interleaved 'feas' one the dense version calls for it -- so kappa_c does not "
        "inherit the truncation/frozen-ET defects documented on plain 'feas' -- and both "
        "passes run on the sparse representation. theta = 1 is NOT the same as the plain "
        "solve in general: c only re-enters at the eta- boundary here, never in the "
        "achievement/continuation objective, so theta=1 matches feas2s's own answer only "
        "on a world with no reachable constraints. "
        "feas_thresh_sparse.feas_iter_two_phase_sparse_thresh (written by Claude).",
        "params": [
            {"id": "risk_thresh", "label": "risk threshold  theta", "type": "range",
             "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5,
             "help": "max tolerated probability of constraint violation. 0 = refuse all risk, "
                     "1 = accept anything -- NOT the same as switching constraints off: a "
                     "constraint still blocks whatever it blocks, this just stops it from "
                     "gating OTHER states as too risky to pass through"},
            {"id": "round_decimals", "label": "round kappa to N decimals", "type": "range",
             "min": 0, "max": ROUND_DECIMALS_MAX, "step": 1, "default": 3,
             "help": f"same mechanism, and same CAUTION, as feas2/feas2s's own round_decimals: "
                     f"BOTH passes here reuse feas2s's tied-action-set logic internally, so a "
                     "nonzero value can equally make a state-action goal's policy pick a "
                     "non-terminating action tied within rounding, and OFF at either end of "
                     f"the slider -- 0, or maxed out at {ROUND_DECIMALS_MAX} -- for the same "
                     "reason. Defaults to 3, matching feas2/feas2s's own default; move to 0 or "
                     "the max for exact kappa if an option ever reports zero success where it "
                     "visibly should not"},
        ] + _CONVERGENCE_PARAMS,
    },
]
# Additional eta implementations; keep the reference algorithms available.
for _base, _id, _label in [
    ("feas2s", "feas_eta_fast", "Feasibility, fast eta (Codex)"),
    ("feas_thresh_s", "feas_eta_fast_thresh", "Risk-thresholded feasibility, fast eta (Codex)"),
]:
    _source = next(a for a in ALGORITHMS if a["id"] == _base)
    ALGORITHMS.append({
        "id": _id, "label": _label, "returns": "kappa",
        "default": _id == "feas_eta_fast_thresh",
        "about": "Same policy optimization, with a new full state-time eta computation. "
                 "Automatic mode uses wide sparse storage for deterministic continuations and "
                 "compact terminal propagation for stochastic ones. Time-zero-only terminal "
                 "events are stored separately. The risk variant computes pass-1 risk directly "
                 "without building pass-1 eta. Horizon and tail tolerance match the reference.",
        "params": list(_source["params"]) + [{
            "id": "eta_mode", "label": "eta computation: 0 auto (recommended) / 1 wide / 2 compact",
            "type": "range", "min": 0, "max": 2, "step": 1, "default": 0,
            "help": "Leave this at 0 for automatic selection. This changes speed and memory use, "
                    "not your risk threshold or accuracy settings. 1 (wide) stores probabilities "
                    "for ending states and arrival times in a broad sparse matrix. 2 (compact) "
                    "works with only the ending states that can receive probability after time zero. "
                    "Both compute the same eta distribution with the same stopping tolerance; "
                    "which is faster depends on the world.",
        }],
    })
for _base, _id, _label in [
    ("feas_eta_fast", "feas_eta_combined", "Feasibility, combined STOK (Codex)"),
    ("feas_eta_fast_thresh", "feas_eta_combined_thresh", "Risk-thresholded feasibility, combined STOK (Codex)"),
]:
    _source = next(a for a in ALGORITHMS if a["id"] == _base)
    ALGORITHMS.append({
        "id": _id, "label": _label, "returns": "kappa", "params": list(_source["params"]),
        "about": "One STOK with the combined success/failure time-zero boundary and one "
                 "continuation operator. Infeasible states terminate for every outcome. "
                 "Success/failure plots are terminal-weighted views of the same stored kernel, "
                 "not separately propagated eta tensors. One tail stopping test uses total "
                 "termination mass. Policy optimization remains unchanged.",
    })
# Interactive risk preset: keep probability convergence tight and automatic eta
# storage, but avoid spending iterations on nanostep expected-time differences.
# Copy parameter dictionaries so the reference solvers retain their defaults.
for _algo in ALGORITHMS:
    if _algo["id"] == "feas_eta_fast_thresh":
        _algo["params"] = [dict(p, default=1e-5) if p["id"] == "ET_epsilon" else dict(p)
                           for p in _algo["params"]]
ALGO_BY_ID = {a["id"]: a for a in ALGORITHMS}

# Fraction of arrival mass the time axis must cover. The remaining tail is real but negligible, and
# letting it set the scale makes every distribution unreadable -- see stok_time_slice.
TAIL_KEEP = 0.999

# Right panel. Same declarative shape as ALGORITHMS: the page builds its controls from `params`.
#
# Both are cost minimisations expressed as reward maximisation (rewards are negative), so V is
# negative everywhere and 0 at the goal. `constraint_penalty` is what makes the value function
# notice the fire cells at all -- without it V is just distance-to-goal and the two panels would
# be showing unrelated things.
VALUE_ALGORITHMS = [
    {
        "id": "ihdc", "label": "Infinite-horizon discounted  (V)",
        "about": "Discounted value iteration on the flat kernel. V = -(1-gamma^d)/(1-gamma) for a "
        "deterministic world d steps from the goal. utilities.value_iteration_ihdc_outer_action",
        "params": [
            {"id": "gamma", "label": "discount  gamma", "type": "range",
             "min": 0.50, "max": 0.99, "step": 0.01, "default": 0.95,
             "help": "how far ahead it looks; near 1 approaches shortest-path"},
            {"id": "step_cost", "label": "step cost", "type": "range",
             "min": 0.1, "max": 5.0, "step": 0.1, "default": 1.0,
             "help": "charged on every move"},
            {"id": "constraint_penalty", "label": "constraint penalty", "type": "range",
             "min": 0.0, "max": 50.0, "step": 1.0, "default": 10.0,
             "help": "extra cost for occupying a fire cell"},
            {"id": "goal_reward", "label": "goal reward", "type": "range",
             "min": 0.0, "max": 50.0, "step": 1.0, "default": 0.0,
             "help": "collected on EVERY step spent on a goal cell (the goal is not absorbing "
                     "here), so V at the goal tends to reward/(1-gamma)"},
        ],
    },
    {
        "id": "ssp", "label": "Min-cost shortest path  (first-exit)",
        "about": "Undiscounted first-exit: the goal is made ABSORBING and free, so V converges to "
        "minus the expected cost-to-go. With unit step cost and no slip that is exactly "
        "minus the shortest-path length. No goal reward here: with gamma = 1 an absorbing "
        "state with a non-zero reward accumulates it every iteration and never converges "
        "(V would just be reward x iteration count). The value IS the cost-to-go.",
        "params": [
            {"id": "step_cost", "label": "step cost", "type": "range",
             "min": 0.1, "max": 5.0, "step": 0.1, "default": 1.0,
             "help": "charged on every move"},
            {"id": "constraint_penalty", "label": "constraint penalty", "type": "range",
             "min": 0.0, "max": 50.0, "step": 1.0, "default": 10.0,
             "help": "extra cost for occupying a fire cell"},
        ],
    },
]
VALUE_BY_ID = {a["id"]: a for a in VALUE_ALGORITHMS}


def _stub_solver_plots():
    """feas_iter_stationary_flat_dense_thresh calls utilities.plotgrid2() UNCONDITIONALLY at the end
    (TGMDP_methods.py:2067-2068). On a server that would try to open matplotlib windows, and it
    crashes outright on a flat kappa vector ("Invalid shape (49,) for image data").

    Stubbed HERE, in this process, rather than edited out of TGMDP_methods.py -- that file is shared
    by okbe_main and a dozen scripts, and silently changing a solver other code depends on is not
    something to do as a side effect of building a tutorial."""
    import matplotlib
    matplotlib.use("Agg")                       # belt and braces: never try to open a window
    import utilities
    utilities.plotgrid2 = lambda *a, **k: None
    utilities.polquiver = lambda *a, **k: None


def warm_solver():
    """Import the solver stack once, in the background. Stage 3 will call into it."""
    t0 = time.time()
    try:
        sys.path.insert(0, HERE)
        import TGMDP_methods  # noqa: F401
        import utilities      # noqa: F401
        import world_spaces   # noqa: F401  -- pre-warms state_spaces for /api/hl_panel too
        import task_gui       # noqa: F401  -- pre-warms gates.py for /api/task_panel too
        _stub_solver_plots()
        SOLVER["loaded"] = True
    except Exception as e:                       # a missing dep must not take the page down
        SOLVER["error"] = f"{type(e).__name__}: {e}"
    SOLVER["seconds"] = round(time.time() - t0, 2)


# The transition kernel depends ONLY on the grid shape, the walls and the noise -- not on goals,
# constraints, or any algorithm parameter. Rebuilding StateSpace_2D per request cost ~26ms on a 7x7
# and ~120ms on a 15x15, which dwarfed value iteration itself (~0.6ms) and was paid again on every
# nudge of a slider. Cache it.
_KERNELS = {}
_KERNEL_ORDER = []
_KERNEL_MAX = 24
_kernel_lock = threading.Lock()


def _kernel_key_overrides(overrides):
    """Canonical, hashable form of the hand-edited rows -- part of the cache key."""
    if not overrides:
        return ()
    return tuple(sorted(
        (int(li), int(a), tuple(sorted((int(x2), round(float(p), 9)) for x2, p in dist.items())))
        for li, by_a in overrides.items() for a, dist in by_a.items()))


def _kernel_P_fast(rows, cols, walls, p_main):
    """Vectorized equivalent of state_spaces.StateSpace_2D's build_stoc_stationary_transition_op,
    for the tutorial's own use only: no elevation (StateSpace_2D's own elev is uniformly 0 for a
    plain 2D grid call, so its max_step_up redirect can never fire and is omitted here), single
    wall layer, actions [interact, stay, down, right, up, left] matching ACTION_DELTA exactly.

    WHY THIS EXISTS: StateSpace_2D builds this same array through a per-(state, action) PYTHON
    loop -- na*nx iterations, each doing several small numpy reduction calls -- which is fine at
    9x9 (486 iterations) but ~7s of pure kernel construction at 40x40 (9600 iterations), BEFORE
    any solving even starts. Verified byte-identical to StateSpace_2D's own dense_op_list[0]
    across 8 cases (empty/scattered/all walls, p_main in {0.05, 1.0} inclusive) -- see this
    function's own test coverage. ~50x faster at 40x40 (7s -> 0.13s) purely from vectorizing what
    was already an O(na*nx) computation; it is not a different algorithm.
    """
    import numpy as np
    nx = rows * cols
    residual = round(1 - p_main, 10)
    r_uni = residual / 4
    # dists[intended, realized] = P(action `intended` actually executes as `realized`).
    # interact/stay (0/1) are deterministic; a movement action spreads its residual over the
    # other 4 non-interact outcomes (never realizes as interact -- see StateSpace_2D's own c1/c2).
    dists = np.array([
        [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
        [0, r_uni, p_main, r_uni, r_uni, r_uni],
        [0, r_uni, r_uni, p_main, r_uni, r_uni],
        [0, r_uni, r_uni, r_uni, p_main, r_uni],
        [0, r_uni, r_uni, r_uni, r_uni, p_main],
    ])
    is_wall = np.zeros(nx, dtype=bool)
    if len(walls):
        is_wall[np.asarray(list(walls), dtype=int)] = True
    rr, cc = np.divmod(np.arange(nx), cols)
    # landing[realized_action] = state landed at, for every state, under that realized action --
    # clamped to the grid edge, then redirected back to self if it would land in a wall.
    landing = np.empty((6, nx), dtype=int)
    for ra, (dy, dx) in enumerate(ACTION_DELTA):
        ny = np.clip(rr + dy, 0, rows - 1)
        ncx = np.clip(cc + dx, 0, cols - 1)
        li = ny * cols + ncx
        li = np.where(is_wall[li], np.arange(nx), li)
        landing[ra] = li
    P = np.zeros((6, nx, nx))
    idx = np.arange(nx)
    for ia in range(6):
        for ra in range(6):
            p = dists[ia, ra]
            if p == 0:
                continue
            P[ia, idx, landing[ra]] += p
    # A WALL as the SOURCE state overrides everything above: every action self-loops there,
    # matching StateSpace_2D's own "if oldYX_ind in wallinds: allYX_inds = [oldYX_ind]*_na".
    wall_idx = np.where(is_wall)[0]
    if wall_idx.size:
        P[:, wall_idx, :] = 0
        P[:, wall_idx, wall_idx] = 1
    return P


def get_world_kernel(world, product=False):
    sp, P = get_kernel(world['rows'], world['cols'], world['walls'], world.get('p_main', 1.), world.get('kernel'))
    closed = [d['cell'] for d in world.get('doors', []) if not d['open']]
    if not product and closed:
        import numpy as np
        P = P.copy()
        for cell in closed:
            sources = np.arange(P.shape[1])
            sources = sources[sources != cell]
            for a in range(P.shape[0]):
                P[a, sources, sources] += P[a, sources, cell]
                P[a, sources, cell] = 0
    if product:
        import tutorial_workshop
        if tutorial_workshop.canteen_enabled(world) or tutorial_workshop.tree_cells(world):
            import numpy as np
            count=len(tutorial_workshop.special_actions(world))
            P=np.concatenate([P,np.repeat(np.eye(P.shape[1])[None],count,axis=0)],axis=0)
    return sp, P


def get_kernel(rows, cols, walls, p_main, overrides=None):
    key = (rows, cols, tuple(sorted(int(w) for w in walls)), round(float(p_main), 6),
           _kernel_key_overrides(overrides))
    with _kernel_lock:
        hit = _KERNELS.get(key)
    if hit is not None:
        log("kernel", hit=True, rows=rows, cols=cols)
        return hit
    import numpy as np
    t0 = time.time()
    # No state_spaces.StateSpace_2D() here on purpose -- see _kernel_P_fast's own docstring. `sp`
    # is None for every caller now; the one algorithm that still needs a real xspace object
    # (dense "feas"/"feas_thresh" in solve_world's own dispatch) builds it there itself, only when
    # that algorithm is actually selected, so choosing a sparse algorithm never pays for it.
    sp = None
    P = _kernel_P_fast(rows, cols, walls, float(p_main))
    # HAND-EDITED ROWS, laid over the p_main kernel. Sparse by design: a world with none is
    # byte-for-byte what it was, so nothing about the existing behaviour depends on this.
    # Each row is renormalised here as well as in the browser -- the kernel has to be stochastic
    # whatever reaches the endpoint, including a hand-written request or an old saved level.
    for li, by_a in (overrides or {}).items():
        for a, dist in by_a.items():
            li_i, a_i = int(li), int(a)
            if not (0 <= li_i < P.shape[1] and 0 <= a_i < P.shape[0]):
                raise ValueError(f"transition override out of range: state {li_i}, action {a_i}")
            row = np.zeros(P.shape[2])
            for x2, prob in dist.items():
                x2_i = int(x2)
                if not (0 <= x2_i < P.shape[2]):
                    raise ValueError(f"transition override destination out of range: {x2_i}")
                row[x2_i] = max(0.0, float(prob))
            total = row.sum()
            if total <= 0:
                raise ValueError(f"transition override for state {li_i}, action {a_i} is all zero")
            P[a_i, li_i, :] = row / total
    val = (sp, P)
    with _kernel_lock:
        _KERNELS[key] = val
        _KERNEL_ORDER.append(key)
        while len(_KERNEL_ORDER) > _KERNEL_MAX:          # plain FIFO; these are a few MB each
            _KERNELS.pop(_KERNEL_ORDER.pop(0), None)
    log("kernel", hit=False, rows=rows, cols=cols, ms=round((time.time() - t0) * 1000))
    return val


# The TIME-RESOLVED eta, eta(t_f, x_f | x), is nx*nx*T floats -- 1.9 MB at 9x9, 8 MB at 12x12 --
# far too big to ship to the browser whole. Keep the last few here instead and serve the single
# x0 row (nx x T) on request, which is small.
_ETA = {}
_ETA_ORDER = []
# One entry PER GOAL GROUP, so a world with several groups needs several slots. Composition walks
# a chain of options and needs every one of them live at once.
_ETA_MAX = 12
_eta_lock = threading.Lock()


def _eta_key(world, algo_id, params, group_id):
    """Cache key for eta.

    Deliberately EXCLUDES the start state. eta(x_f, t | x) is computed for every x at once, so it
    does not depend on which x you happen to be conditioning on -- but the start state is part of
    the world dict, so keying on the whole thing meant dragging the start invalidated the cache
    and re-solved on every cell crossed. That was the lag on the arrival-time plot.
    """
    key_world = {k: v for k, v in world.items() if k not in ("start", "nextGoalId", "green_badgers")}
    return json.dumps([key_world, algo_id, params, group_id], sort_keys=True, default=str)


def stok_time_slice(world, algo_id, params, group_id, x0):
    """eta(t_f, x_f | x0) for one initial state: the temporal distribution over each final state,
    plus its marginal over final states. Served from the cache left by the last solve; if that has
    been evicted, the solve is repeated."""
    import numpy as np
    key = _eta_key(world, algo_id, params, group_id)
    with _eta_lock:
        pair = _ETA.get(key)
    resolved = pair is None
    if pair is None:
        # group_id may name a composed STRING ("chain:2,3") rather than a single policy
        if isinstance(group_id, str) and group_id.startswith("chain:"):
            seq = [int(g) for g in group_id[len("chain:"):].split(",") if g != ""]
            # compose_fast's 'all_x0' mode -- same content as compose_sequence (verified
            # byte-identical), ~10x faster to build. Matters here specifically because this path
            # runs on EVERY view of the chain/STOK tab that needs an arrival-time slice, whether
            # or not /api/compose's own call used the fast single-x0 mode -- falling back to the
            # dense path here would silently undo that speedup for this one piece of the UI.
            pos, neg = compose_fast(world, algo_id, params, seq, 0, True, "all_x0")
            with _eta_lock:
                _ETA[key] = (pos, neg)
                if key in _ETA_ORDER:
                    _ETA_ORDER.remove(key)
                _ETA_ORDER.append(key)
                while len(_ETA_ORDER) > _ETA_MAX:
                    _ETA.pop(_ETA_ORDER.pop(0), None)
            pair = (pos, neg)
        else:
            solve_world(world, algo_id, params, want_stok=True, group_id=group_id, stok_mode="cache_only")
            with _eta_lock:
                pair = _ETA.get(key)
        if pair is None:
            raise ValueError("could not obtain eta for this world")
    ep, en = pair
    nx = ep.shape[0]
    if not (0 <= int(x0) < nx):
        raise ValueError(f"x0 out of range: {x0}")
    sp_ = ep[int(x0)]                             # [x_f, t_f], successful
    sn_ = en[int(x0)]                             # [x_f, t_f], violated
    mp, mn = sp_.sum(axis=0), sn_.sum(axis=0)     # over final STATES -> time distributions
    # trim the shared all-zero tail so the browser is not sent a long run of zeros
    nz = np.nonzero((mp + mn) > 1e-9)[0]
    tmax = int(nz[-1]) + 1 if len(nz) else 1
    # THE AXIS IS NOT tmax. tmax is this x0's own last arrival, so it shrinks for a start beside the
    # goal and grows for a distant one -- and the client stretches whatever it is given across the
    # full card width. Two starts then get the same width for different spans of time, and nothing
    # can be compared between them.
    #
    # t_axis is the last arrival ANY start state has under this solve, so it is one number for the
    # whole eta and every card drawn from it shares a ruler. The per-x0 trim above still applies to
    # what is SENT -- the client pads out to t_axis -- so the axis is stable without shipping a run
    # of zeros for every near-goal start.
    # The cutoff is a PROBABILITY threshold, not a bare epsilon. These are stochastic dynamics, so
    # there is always a vanishing chance of taking arbitrarily long; keying the axis to the last
    # nonzero bin would let a tail carrying 0.01% of the mass set the scale and squash everything
    # that matters into the first tenth of the card. Instead: the earliest t by which
    # TAIL_KEEP of the mass has arrived.
    # summed separately rather than as (ep + en): the packed TerminalEta from the sparse solver
    # is not an ndarray and does not add, and there is no reason to materialise the sum anyway
    tot = ep.sum(axis=(0, 1)) + en.sum(axis=(0, 1))      # over start AND final states
    total = float(tot.sum())
    if total > 0:
        cum = np.cumsum(tot) / total
        reached = np.nonzero(cum >= TAIL_KEEP)[0]
        t_axis = int(reached[0]) + 1 if len(reached) else int(tot.shape[0])
    else:
        t_axis = 1
    t_axis = max(t_axis, 2)
    cut = lambda m: [[round(float(v), 6) for v in row[:tmax]] for row in m]
    vec = lambda m: [round(float(v), 6) for v in m[:tmax]]
    return {
        "x0": int(x0), "T": tmax, "T_axis": t_axis, "resolved": resolved,
        # Marginalize BEFORE display-tail trimming, so the surface preserves all eta mass.
        "stok": np.round(sp_.sum(axis=1), 5).tolist(),
        "stok_neg": np.round(sn_.sum(axis=1), 5).tolist(),
        "by_state": cut(sp_), "marginal": vec(mp), "mass": round(float(mp.sum()), 6),
        "by_state_neg": cut(sn_), "marginal_neg": vec(mn), "mass_neg": round(float(mn.sum()), 6),
    }


# ---- saved levels -------------------------------------------------------------------------------
_NAME_OK = re.compile(r"[^A-Za-z0-9_\-]+")


def sanitize_name(name):
    """Filenames come from the browser, so they are untrusted. Strip to a safe character set and
    take the basename -- a name is a LABEL, never a path."""
    name = os.path.basename(str(name or "")).strip()
    name = _NAME_OK.sub("_", name).strip("_")
    return name[:60]


def next_level_name(folder=""):
    """level_01, level_02, ... -- the first number not already taken."""
    used = set()
    directory = os.path.dirname(level_path(folder + "/placeholder")) if folder else LEVELDIR
    for f in os.listdir(directory) if os.path.isdir(directory) else []:
        m = re.fullmatch(r"level_(\d+)\.json", f)
        if m:
            used.add(int(m.group(1)))
    i = 1
    while i in used:
        i += 1
    return (folder + "/" if folder else "") + f"level_{i:02d}"


def level_path(name):
    parts = str(name or '').split('/')
    if len(parts) > 2 or any(part in ('', '.', '..') or '\\' in part for part in parts):
        raise ValueError('Choose a level name or folder/level name')
    clean = [sanitize_name(part) for part in parts]
    if not all(clean): raise ValueError('Invalid level name')
    path = os.path.join(LEVELDIR, *clean) + '.json'
    # Old browser snapshots may still refer to the original ungrouped pack names.
    if len(clean) == 1 and clean[0].startswith('progressive_') and not os.path.exists(path):
        grouped = os.path.join(LEVELDIR, 'progressive_levels', clean[0] + '.json')
        if os.path.isfile(grouped): path = grouped
    root = os.path.realpath(LEVELDIR)
    if os.path.commonpath([root, os.path.realpath(path)]) != root:
        raise ValueError('Level path must stay inside the levels folder')
    return path


def describe_world(w):
    """A one-line summary for the picker, so a list of generic names is still navigable."""
    ngoal = sum(len(g["cells"]) for g in w.get("goals", []))
    bits = [f"{w['rows']}x{w['cols']}"]
    if w.get("walls"): bits.append(f"{len(w['walls'])} walls")
    if ngoal: bits.append(f"{ngoal} goal{'s' if ngoal != 1 else ''} in "
    f"{len([g for g in w['goals'] if g['cells']])} group(s)")
    if w.get("constraints"): bits.append(f"{len(w['constraints'])} constraints")
    if w.get("start") is None: bits.append("no start")
    nz = round(1 - float(w.get("p_main", 1.0)), 2)
    if nz: bits.append(f"noise {nz}")
    return " · ".join(bits)


def save_level(world, name=None, strings=None, tutorial=None, folder=""):
    """`strings` are the saved policy strings for THIS level. They are sequences of goal-group ids,
    which only mean anything inside the level that defined those groups -- kept globally they
    survive into the next level and name policies that do not exist there."""
    os.makedirs(LEVELDIR, exist_ok=True)
    path = level_path(name or next_level_name(folder))
    nm = os.path.relpath(path, LEVELDIR)[:-5].replace(os.sep, '/')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = make_level(world, name=nm, saved=time.time(),
           strings={str(k): [int(g) for g in v] for k, v in (strings or {}).items()},
           summary=describe_world(world), api_version=API_VERSION)
    if tutorial is not None:
        if not isinstance(tutorial, dict): raise ValueError("tutorial settings must be an object")
        import tutorial_affordances
        tutorial = dict(tutorial)
        if "spec" in tutorial:
            tutorial["spec"] = tutorial_affordances.prepare(world, tutorial["spec"])
        rec["tutorial"] = tutorial
    with open(level_path(nm), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def list_levels():
    if not os.path.isdir(LEVELDIR):
        return []
    out = []
    for folder in ['', *sorted(f for f in os.listdir(LEVELDIR) if os.path.isdir(os.path.join(LEVELDIR, f)) and not os.path.islink(os.path.join(LEVELDIR, f)))]:
        directory = os.path.join(LEVELDIR, folder)
        for f in sorted(os.listdir(directory)):
            if not f.endswith('.json') or os.path.islink(os.path.join(directory, f)): continue
            ident = (folder + '/' if folder else '') + f[:-5]
            try:
                with open(os.path.join(directory, f)) as fh: rec = json.load(fh)
                out.append({'name':ident, 'label':f[:-5], 'folder':folder,
                            'saved':rec.get('saved',0), 'summary':rec.get('summary','')})
            except Exception as e:
                out.append({'name':ident,'label':f[:-5],'folder':folder,'saved':0,'summary':f'unreadable: {e}'})
    out.sort(key=lambda r: -r["saved"])           # newest first
    return out


def load_level(name, with_tutorial=False):
    p = level_path(name)
    if not os.path.exists(p):
        raise ValueError(f"no saved level called {sanitize_name(name)!r}")
    with open(p) as fh:
        rec = json.load(fh)
    rec = read_level(rec)
    result = (normalize_world(rec["world"]), rec.get("strings", {}))
    return result + (rec.get("tutorial"),) if with_tutorial else result


def delete_level(name):
    p = level_path(name)
    if not os.path.exists(p):
        raise ValueError(f"no saved level called {sanitize_name(name)!r}")
    os.remove(p)


# Policies, cached like the kernels. A rollout solves once per leg, so sampling a two-leg string
# 400 times was 800 identical solves. Keyed without the start state, since pi does not depend on it.
_POLICY = {}
_POLICY_ORDER = []
_POLICY_MAX = 64
_pol_lock = threading.Lock()


def get_policy(world, source, algo_id, params, group_id):
    import numpy as np
    key = _eta_key(world, (source, algo_id), params, group_id)
    with _pol_lock:
        hit = _POLICY.get(key)
    if hit is not None:
        return hit
    one = dict(world)
    one["goals"] = [g for g in world["goals"] if g["id"] == group_id]
    if not one["goals"]:
        raise ValueError(f"policy {group_id} does not exist")
    res = value_world(one, algo_id, params) if source == "value" \
        else solve_world(one, algo_id, params)
    pi = np.asarray(res["surfaces"][0]["pi"], dtype=int)
    with _pol_lock:
        _POLICY[key] = pi
        if key in _POLICY_ORDER:
            _POLICY_ORDER.remove(key)
        _POLICY_ORDER.append(key)
        while len(_POLICY_ORDER) > _POLICY_MAX:
            _POLICY.pop(_POLICY_ORDER.pop(0), None)
    return pi


# kappa*(x) for every state, one goal group at a time -- separate cache from _POLICY (same
# pattern get_eta uses alongside it), since a caller may want one without the other.
_KAPPA = {}
_KAPPA_ORDER = []
_KAPPA_MAX = 64
_kap_lock = threading.Lock()


# compose_sequence's running (pos, neg) after each PREFIX of a policy string -- keyed on the
# world/algo/params/handover and the exact prefix, one entry per prefix length. The chain-builder
# panel calls compose_sequence again on every click, always either extending or shortening
# whatever string is already on screen, so without this it recomposed the ENTIRE string from the
# first policy on every single click -- the string got progressively slower to build, not just
# progressively more expensive per click, because each click redid all the previous clicks' work
# too. A separate cache from _ETA (rather than reusing it under a different key prefix) because
# an N-policy string leaves N of these alive at once on top of _ETA's own per-policy entries, and
# evicting the one that made the NEXT click fast would defeat the point.
_COMPOSE_PREFIX = {}
_COMPOSE_PREFIX_ORDER = []
_COMPOSE_PREFIX_MAX = 48
_compose_prefix_lock = threading.Lock()


# SparseSTOK versions of each policy's (eta_pos, eta_neg) -- for compose_fast (single-x0 and
# all-x0-sparse STOK modes), which need a SparseSTOK per leg, not the dense [nx,nx,T] arrays
# get_eta returns. Converting costs an O(nx^2*T) scan (stok_chain.SparseSTOK.from_dense), which
# would otherwise happen on every request; cached the same way get_eta's own dense result is.
_SPARSE_ETA = {}
_SPARSE_ETA_ORDER = []
_SPARSE_ETA_MAX = 12
_sparse_eta_lock = threading.Lock()


def get_eta_sparse(world, algo_id, params, group_id):
    """(SparseSTOK, SparseSTOK) for one policy's (eta_pos, eta_neg) -- see _SPARSE_ETA above."""
    import stok_chain as sc
    key = _eta_key(world, algo_id, params, group_id)
    with _sparse_eta_lock:
        hit = _SPARSE_ETA.get(key)
    if hit is not None:
        return hit
    ep, en = get_eta(world, algo_id, params, group_id)
    if hasattr(ep, "to_sparse_stok"):
        pair = (ep.to_sparse_stok(), en.to_sparse_stok())
    else:
        if hasattr(ep, "to_dense"):
            ep, en = ep.to_dense(), en.to_dense()
        pair = (sc.SparseSTOK.from_dense(ep), sc.SparseSTOK.from_dense(en))
    with _sparse_eta_lock:
        _SPARSE_ETA[key] = pair
        if key in _SPARSE_ETA_ORDER:
            _SPARSE_ETA_ORDER.remove(key)
        _SPARSE_ETA_ORDER.append(key)
        while len(_SPARSE_ETA_ORDER) > _SPARSE_ETA_MAX:
            _SPARSE_ETA.pop(_SPARSE_ETA_ORDER.pop(0), None)
    return pair


def get_kappa(world, source, algo_id, params, group_id):
    import numpy as np
    key = _eta_key(world, (source, algo_id), params, group_id)
    with _kap_lock:
        hit = _KAPPA.get(key)
    if hit is not None:
        return hit
    one = dict(world)
    one["goals"] = [g for g in world["goals"] if g["id"] == group_id]
    if not one["goals"]:
        raise ValueError(f"policy {group_id} does not exist")
    res = value_world(one, algo_id, params) if source == "value" \
        else solve_world(one, algo_id, params)
    kappa = np.asarray(res["surfaces"][0]["kappa"], dtype=float)
    with _kap_lock:
        _KAPPA[key] = kappa
        if key in _KAPPA_ORDER:
            _KAPPA_ORDER.remove(key)
        _KAPPA_ORDER.append(key)
        while len(_KAPPA_ORDER) > _KAPPA_MAX:
            _KAPPA.pop(_KAPPA_ORDER.pop(0), None)
    return kappa


def get_eta(world, algo_id, params, group_id):
    """(eta_pos, eta_neg) for one goal group, from the solve cache or by solving."""
    key = _eta_key(world, algo_id, params, group_id)
    with _eta_lock:
        pair = _ETA.get(key)
    if pair is None:
        solve_world(world, algo_id, params, want_stok=True, group_id=group_id, stok_mode="cache_only")
        with _eta_lock:
            pair = _ETA.get(key)
    if pair is None:
        raise ValueError(f"could not obtain eta for policy {group_id}")
    return pair


def compose_stok_pair(eta1, eta2):
    """Compose two space-time option kernels into ONE:

        eta_c(x_f, t | x)  =  sum_m  ( eta_1(m, . | x)  *  eta_2(x_f, . | m) )        [* = in time]

    Run policy 1 from x, land at some intermediate m at some time, then run policy 2 from m.
    The result has the same shape, so composition is CLOSED: a string of policies is itself a
    policy and can be composed again. No initial state is involved.

    Row x of the composite is exactly what TGMDP_methods.chapman_kolmo_forward returns when a
    delta at x is propagated through eta1 and then eta2, which is how okbe_main chains two
    options; this just does it for every x so the result is a kernel rather than one conditioned
    distribution.

    ONE convolution pass, not two. Propagating a ONE-HOT delta at x through eta1 is eta1's own
    row x, exactly -- convolving with a unit impulse [1.0] is the identity, so
    chapman_kolmo_forward(delta_x, eta1) == eta1[x] for every x, verified to be exactly 0.0
    difference for a random eta1. The first of the two passes this function used to run was
    therefore computing eta1 through eta1's OWN convolution machinery and getting eta1 back,
    at the cost of nx separate Python/scipy calls for nothing. Only the second pass (propagating
    eta1's own, genuinely non-delta content through eta2) does real work.
    """
    import numpy as np
    import TGMDP_methods as tg

    nx = eta1.shape[0]
    # The same eta2 is used for every start state, so find its nonzero (i, j) pairs ONCE rather
    # than rescanning inside each of the nx calls -- that scan is the dominant cost once the empty
    # convolutions are skipped.
    nz2 = np.any(eta2 != 0, axis=2)
    out = None
    for x in range(nx):
        # method='fft': see chapman_kolmo_forward's own comment. Direct convolution (scipy's own
        # 'auto' choice at these array sizes) is what made composing a long policy string get
        # progressively slower per additional step even with the prefix cache working correctly --
        # each fold's own convolution cost grows with the string's own time axis, and direct
        # convolution scales quadratically in that while FFT scales quasi-linearly.
        r = tg.chapman_kolmo_forward(eta1[x], eta2, nz2, method='fft')
        if out is None:
            out = np.zeros((nx, r.shape[0], r.shape[1]))
        out[x] = r
    return out


def handover_kernel(group, P):
    """The one-step kernel D applied when an option HANDS OVER to the next.

    An option terminates on entering its goal region. If that goal is a STATE-ACTION goal, the
    terminal action has not happened yet -- reaching the tree is not drinking from it -- so the
    handover executes it, which costs a step and applies whatever transition it induces. A
    STATE-ONLY goal has no action to execute and hands over immediately.

    Both live in one kernel, so a group mixing the two needs no special case:

        D[x, :, 1] = P[a_g(x), x, :]     cells whose goal names a terminal action
        D[x, x, 0] = 1                   every other cell -- identity, no delay

    Composition then reads eta_1 * D_1 * eta_2 * D_2 ... using the same convolution as everything
    else. Note `interact` is a deterministic self-loop even under kernel noise, so a "drink here"
    goal dwells in place for exactly one step; a movement terminal action would move the agent,
    and D already reflects that.

    LIMITATION: the handover step is not itself checked against constraints. Stepping into a
    constraint while executing a terminal action is not counted as a violation.
    """
    import numpy as np
    nx = P.shape[1]
    D = np.zeros((nx, nx, 2))
    acts = {e["li"]: e["a"] for e in group["cells"] if e["a"] is not None}
    for x in range(nx):
        a = acts.get(x)
        if a is None:
            D[x, x, 0] = 1.0                      # nothing to execute: pass straight through
        else:
            D[x, :, 1] = P[int(a), x, :]          # execute it: one step, and it may move you
    return D


def group_has_terminal_action(group):
    return any(e["a"] is not None for e in group["cells"])


def _pad_t(a, T):
    """Pad a [nx, nx, t] kernel out to T time steps."""
    import numpy as np
    if a.shape[2] >= T:
        return a
    return np.pad(a, ((0, 0), (0, 0), (0, T - a.shape[2])), "constant")


def _flatten_chain(sequence):
    """A policy-string element is either a plain policy id (int), or a BLOCK -- a list of ids,
    typically a saved sub-string appended as one step (see chain.js). This recovers the flat,
    ID-only sequence a block-unaware caller (or a cache key) needs."""
    out = []
    for el in sequence:
        out.extend(_flatten_chain(el)) if isinstance(el, (list, tuple)) else out.append(el)
    return out


def compose_sequence(world, algo_id, params, sequence, handover=True):
    """Fold a sequence of policies into a single composite STOK -- BOTH halves.

    Success composes straightforwardly:      P_c = P_1 D_1 P_2 D_2 ...
    Failure has to accumulate, because a string can fail during ANY leg:

        N_c  =  N_1  +  (P_1 D_1) N_2  +  (P_1 D_1 P_2 D_2) N_3  +  ...

    i.e. fail immediately, or complete the first leg and then fail the second, and so on. Zeroing
    this claimed a composite could never fail, which is wrong whenever the world has constraints.

    D_k is the handover kernel (see handover_kernel): it executes a leg's terminal ACTION where it
    has one, costing a step, and is the identity otherwise. It is applied after EVERY leg
    including the last -- an option is not complete until its terminal action is taken, so a
    one-policy string with a "drink here" goal is walk-then-drink, not just walk.

    `handover=False` drops D entirely, reproducing the earlier behaviour for comparison.

    INCREMENTAL: the loop's own running (pos, neg) IS the state after each PREFIX of `sequence` --
    P_1 D_1 ... P_k D_k is exactly what a k-policy string composes to on its own. So the state at
    every prefix length is cached (_COMPOSE_PREFIX) as it is produced, and on entry this looks for
    the LONGEST already-cached prefix of the REQUESTED sequence and resumes the loop from there,
    instead of always starting at policy 1. The chain-builder panel only ever extends or shortens
    whatever string is already on screen, so in ordinary use this turns "recompose the whole
    string on every click" into "compose the one new leg."

    BLOCKS: an element of `sequence` may be a plain policy id (int), or a LIST of ids -- an
    already-known BLOCK, which in practice is a SAVED sub-string appended as one step (see
    chain.js). A block is composed as ONE leg, via its own (recursively cached) composite,
    instead of being expanded back into one compose_stok_pair call per policy inside it. Without
    this, using a saved N-policy string a second time later in a longer string re-derived all N
    of its steps from scratch, one at a time, with the time axis growing throughout -- the
    prefix cache above cannot help there, since positions 9..16 of "string A, string A again"
    have never been seen at those FLAT positions before, even though their CONTENT is a repeat.
    A block's own composite already has ITS OWN last leg's handover baked in (being a complete,
    closed option, composed with handover=True when it was first built), so no further handover
    kernel is applied for a block leg here -- only for a plain single-policy leg.
    """
    import numpy as np
    by_id = {g["id"]: g for g in world["goals"]}
    _sp, P = get_world_kernel(world)

    flat_seq = _flatten_chain(sequence)
    # flat_at[i] = how many underlying policies sequence[0..i] covers -- the cache is always keyed
    # on FLAT position, so a prefix computed via a block is indistinguishable from (and reusable
    # by) the same prefix built one policy at a time, and vice versa.
    flat_at, _n = [], 0
    for el in sequence:
        _n += len(_flatten_chain(el)) if isinstance(el, (list, tuple)) else 1
        flat_at.append(_n)

    def prefix_key(flat_n):
        pfx = ",".join(str(g) for g in flat_seq[:flat_n])
        return _eta_key(world, algo_id, params, f"chainprefix:{handover}:{pfx}")

    pos = neg = None
    start = 0
    for i in range(len(sequence), 0, -1):
        with _compose_prefix_lock:
            hit = _COMPOSE_PREFIX.get(prefix_key(flat_at[i - 1]))
        if hit is not None:
            pos, neg = hit
            start = i
            break

    for i in range(start, len(sequence)):
        leg = sequence[i]
        is_block = isinstance(leg, (list, tuple))
        if is_block:
            ep, en = compose_sequence(world, algo_id, params, list(leg), handover)
        else:
            ep, en = get_eta(world, algo_id, params, leg)
            # chapman_kolmo_forward convolves a full [nx, nx, T] kernel, so a packed TerminalEta
            # from the sparse solver is reinflated HERE and nowhere else. Composition is the one
            # place that genuinely needs every column; the solve, the surfaces and the arrival
            # times all work off the packed form.
            if hasattr(ep, "to_dense"):
                ep, en = ep.to_dense(), en.to_dense()
        if pos is None:
            pos, neg = ep, en
        else:
            leg_fail = compose_stok_pair(pos, en)  # reached this leg, then violated during it
            pos = compose_stok_pair(pos, ep)
            T = max(neg.shape[2], leg_fail.shape[2], pos.shape[2])
            neg = _pad_t(neg, T) + _pad_t(leg_fail, T)
            pos = _pad_t(pos, T)
        if handover and not is_block and group_has_terminal_action(by_id[leg]):
            # only the SUCCESSFUL half hands over: a leg that ended in violation never gets to
            # take its terminal action
            pos = compose_stok_pair(pos, handover_kernel(by_id[leg], P))
            T = max(pos.shape[2], neg.shape[2])
            pos, neg = _pad_t(pos, T), _pad_t(neg, T)
        with _compose_prefix_lock:
            k = prefix_key(flat_at[i])
            _COMPOSE_PREFIX[k] = (pos, neg)
            if k in _COMPOSE_PREFIX_ORDER:
                _COMPOSE_PREFIX_ORDER.remove(k)
            _COMPOSE_PREFIX_ORDER.append(k)
            while len(_COMPOSE_PREFIX_ORDER) > _COMPOSE_PREFIX_MAX:
                _COMPOSE_PREFIX.pop(_COMPOSE_PREFIX_ORDER.pop(0), None)
    return pos, neg


def compose_fast(world, algo_id, params, sequence, x0, handover=True, stok_mode="single_x0"):
    """Fast alternative to compose_sequence, via stok_chain's sparse per-x0 convolution instead of
    one dense [nx,nx,T] chapman_kolmo_forward call per LEG PER STATE.

    stok_mode:
      'single_x0'  Only x0's own (pos, neg) is computed -- stok_chain.ck_chain_with_failure_sparse
                   run once. Cheapest by far (measured ~3ms vs ~300ms for 'all_x0' vs ~3.2s for the
                   dense compose_sequence path, on a 25-policy string), but the result answers only
                   THIS x0: querying a different one means calling this again, not re-reading a row
                   of an already-built kernel.
      'all_x0'     The full (pos, neg) kernel, via stok_chain.compose_all_x0_sparse -- same shape
                   and content as compose_sequence's own return (verified byte-identical), built by
                   running the single-x0 case once per starting state instead of nx dense
                   chapman_kolmo_forward calls per leg. ~10x faster than compose_sequence at 25
                   legs and the gap widens with string length, since the dense path's cost is
                   dominated by [nx,nx,T] array allocation/padding, which this never does.

    BLOCKS (saved sub-strings, see compose_sequence's own docstring) are simply flattened here --
    this path does not (yet) reuse a block's own composite the way compose_sequence's prefix cache
    does, since a single-x0 pass is already so cheap that re-deriving it is rarely the bottleneck.

    Returns (pos, neg): for 'single_x0', each [nx, T]; for 'all_x0', each [nx, nx, T] (row x0 of
    which equals the 'single_x0' result, verified).
    """
    import stok_chain as sc

    rows, cols = world["rows"], world["cols"]
    nx = rows * cols
    by_id = {g["id"]: g for g in world["goals"]}
    flat_sequence = _flatten_chain(sequence)
    _sp, P = get_world_kernel(world)

    legs = []
    for gid in flat_sequence:
        ep_s, en_s = get_eta_sparse(world, algo_id, params, gid)
        D_s = None
        if handover and group_has_terminal_action(by_id[gid]):
            D_s = sc.SparseSTOK.from_dense(handover_kernel(by_id[gid], P))
        legs.append((ep_s, en_s, D_s))

    if stok_mode == "all_x0":
        return sc.compose_all_x0_sparse(legs, nx)
    return sc.ck_chain_with_failure_sparse(sc.delta(nx, int(x0)), legs)


def compose_options(world, algo_id, params, sequence, x0, handover=True, stok_mode="single_x0"):
    """Chain options by CONVOLVING their space-time option kernels.

    This is TGMDP_methods.chapman_kolmo_forward -- your own function -- applied once per option,
    exactly as okbe_main does it:

        xt = delta(x0) x delta(t=0)
        xt = chapman_kolmo_forward(xt, eta_1)
        xt = chapman_kolmo_forward(xt, eta_2)   ...

    Each application convolves the running (state, time) distribution with that option's eta and
    extends the time axis, so the result is the joint distribution over WHERE you end up and WHEN,
    after running the whole string.

    eta_POS only: chaining presupposes the previous option succeeded. The mass lost at each step is
    reported, so the drop-off along the chain is visible rather than silently normalised away.

    Composition is associative, so an n-string followed by an m-string is just the concatenated
    sequence -- the client can therefore build chains out of chains with no extra machinery here.

    stok_mode -- see compose_fast's own docstring for what each does and its measured cost:
      'single_x0' (default) Only x0's own answer is computed. Fastest by far, but "kappa"/"kappa_c"
                  (the string's success/violation probability from EVERY state, not just x0) have
                  no single-x0 equivalent and are OMITTED from the response -- the cumulative-kappa
                  tab has nothing to show for a string viewed this way. "stok"/"stok_neg" are [nx]
                  vectors (x0's own row) rather than [nx,nx] matrices, and re-conditioning on a
                  different x0 means calling this again, not re-reading a row client-side.
      'all_x0'    The full kernel, ~10x faster to build than the original dense compose_sequence
                  path (verified byte-identical) and functionally a drop-in replacement for it:
                  same response shape as before this mode existed, "kappa"/"kappa_c" included,
                  re-conditioning on any x0 stays instant client-side.
    """
    import numpy as np

    if not sequence:
        raise ValueError("empty policy string: append at least one policy")
    if x0 is None:
        raise ValueError("no start state placed -- place one (S) before composing a string")
    rows, cols = world["rows"], world["cols"]
    nx = rows * cols
    if not (0 <= int(x0) < nx):
        raise ValueError(f"x0 out of range: {x0}")
    x0 = int(x0)

    by_id = {g["id"]: g for g in world["goals"]}
    flat_sequence = _flatten_chain(sequence)
    for gid in flat_sequence:
        if gid not in by_id or not by_id[gid]["cells"]:
            raise ValueError(f"policy {gid} does not exist or has no goal cells")

    # per-step progress, for the readout: mass surviving after each policy (or BLOCK -- a saved
    # sub-string used as one step, see compose_sequence's own docstring) in turn, from x0 only --
    # exactly what compose_fast's single-x0 path computes, so this reuses it rather than running
    # its own separate (and, before this, much more expensive) dense pass.
    import stok_chain as sc
    steps, xt_running = [], sc.delta(nx, x0)
    for leg in sequence:
        is_block = isinstance(leg, (list, tuple))
        leg_ids = _flatten_chain(leg) if is_block else [leg]
        cells = [] if is_block else [dict(c) for c in by_id[leg]["cells"]]
        for gid in leg_ids:
            ep_s, _en_s = get_eta_sparse(world, algo_id, params, gid)
            xt_running = sc.ck_convolve_sparse(xt_running, ep_s)
        rowm = float(xt_running.sum())
        tm = xt_running.sum(axis=0)
        steps.append({
            "group": leg_ids if is_block else leg,
            "mass": round(rowm, 6),
            "mean_time": round(float((tm * np.arange(len(tm))).sum() / rowm), 3)
                         if rowm > 1e-12 else None,
            "cells": cells,
        })

    if stok_mode == "all_x0":
        eta_c, eta_c_neg = compose_fast(world, algo_id, params, sequence, x0, handover, "all_x0")
        # Keyed on the FLATTENED sequence: the "chain:1,2,3" format is a flat, comma-joined id
        # list everywhere else it is used (stok_time_slice's own parser splits on "," expecting
        # plain ints), and the composite is mathematically identical however it was built -- via
        # blocks or one policy at a time -- so a lookup by either path finds the same entry.
        key = _eta_key(world, algo_id, params, "chain:" + ",".join(str(g) for g in flat_sequence))
        with _eta_lock:
            _ETA[key] = (eta_c, eta_c_neg)
            if key in _ETA_ORDER:
                _ETA_ORDER.remove(key)
            _ETA_ORDER.append(key)
            while len(_ETA_ORDER) > _ETA_MAX:
                _ETA.pop(_ETA_ORDER.pop(0), None)

        xt = eta_c[x0]                             # [x_f, t] for the chosen start
        tmarg = xt.sum(axis=0)
        nzt = np.nonzero(tmarg > 1e-9)[0]
        tmax = int(nzt[-1]) + 1 if len(nzt) else 1
        total = float(xt.sum())
        return {
            "rows": rows, "cols": cols, "nx": nx, "x0": x0, "stok_mode": "all_x0",
            "sequence": flat_sequence, "steps": steps,
            # the COMPOSITE kernel, time-marginalised: row x is where you end up starting from x.
            # Shipping the whole thing means changing the start state costs nothing.
            "stok": [[round(float(v), 5) for v in row] for row in eta_c.sum(axis=2)],
            "stok_neg": [[round(float(v), 5) for v in row] for row in eta_c_neg.sum(axis=2)],
            # summing a row of the composite over final states AND time is the probability that
            # the whole string completes from that state -- kappa for the string, so it can be
            # viewed in the cumulative-feasibility tab exactly like a single policy
            "kappa": [round(float(v), 6) for v in eta_c.sum(axis=(1, 2))],
            "kappa_c": [round(float(v), 6) for v in eta_c_neg.sum(axis=(1, 2))],
            "chain_key": "chain:" + ",".join(str(g) for g in flat_sequence),
            "state": [round(float(v), 6) for v in xt.sum(axis=1)],
            "time": [round(float(v), 6) for v in tmarg[:tmax]],
            "mass": round(total, 6),
            "mean_time": round(float((tmarg * np.arange(len(tmarg))).sum() / total), 3)
                         if total > 1e-12 else None,
        }

    # 'single_x0': only x0's own (pos, neg) row, computed directly -- no [nx,nx,T] kernel, no
    # per-sequence cache entry (there is no full kernel to cache). See this function's own
    # docstring for exactly what that costs the response (no kappa/kappa_c, stok is one row).
    xt, xt_neg = compose_fast(world, algo_id, params, sequence, x0, handover, "single_x0")
    tmarg = xt.sum(axis=0)
    nzt = np.nonzero(tmarg > 1e-9)[0]
    tmax = int(nzt[-1]) + 1 if len(nzt) else 1
    total = float(xt.sum())
    return {
        "rows": rows, "cols": cols, "nx": nx, "x0": x0, "stok_mode": "single_x0",
        "sequence": flat_sequence, "steps": steps,
        "stok": [round(float(v), 5) for v in xt.sum(axis=1)],
        "stok_neg": [round(float(v), 5) for v in xt_neg.sum(axis=1)],
        "chain_key": "chain:" + ",".join(str(g) for g in flat_sequence),
        "state": [round(float(v), 6) for v in xt.sum(axis=1)],
        "time": [round(float(v), 6) for v in tmarg[:tmax]],
        "mass": round(total, 6),
        "mean_time": round(float((tmarg * np.arange(len(tmarg))).sum() / total), 3)
                     if total > 1e-12 else None,
    }


def inspect_world(world, spec):
    """Every quantity for one policy (or string), in one response, so the inspector does not depend
    on which tab happens to be open or on what a panel has cached.

    Each entry is {label, kind, values} where `values` is one number per cell, so the client can lay
    any of them out over the grid without knowing what it is looking at.
    """
    import numpy as np

    group = spec.get("group")
    seq = spec.get("sequence")
    x0 = spec.get("x0")
    algo = spec.get("algorithm", "feas")
    vparams = spec.get("value_params") or {}
    fparams = spec.get("params") or {}
    nx = world["rows"] * world["cols"]
    if x0 is None or not (0 <= int(x0) < nx):
        x0 = world["start"] if world["start"] is not None else 0
    x0 = int(x0)

    out = []
    def add(key, label, kind, vals, note=""):
        out.append({"key": key, "label": label, "kind": kind, "note": note,
                    "values": [round(float(v), 6) for v in vals]})

    # ---- feasibility side ----
    if seq:
        res = compose_options(world, algo, fparams, [int(g) for g in seq], x0)
        surf = {"kappa": res["kappa"], "kappa_c": res["kappa_c"],
                "stok": res["stok"], "stok_neg": res["stok_neg"]}
        who = "string " + "\u03c0" + "".join(str(g) for g in seq)
    else:
        one = dict(world); one["goals"] = [g for g in world["goals"] if g["id"] == group]
        if not one["goals"] or not one["goals"][0]["cells"]:
            raise ValueError(f"policy {group} does not exist or has no goal cells")
        res = solve_world(one, algo, fparams, want_stok=True)
        surf = res["surfaces"][0]
        who = "\u03c0" + str(group)

    add("kappa", "kappa  (feasibility)", "prob", surf["kappa"],
        "probability of reaching the goal without violating a constraint, from each state")
    add("kappa_c", "kappa_c  (violation)", "prob", surf["kappa_c"],
        "probability of ending in a constraint, from each state")
    pos = np.array(surf["stok"])[x0]
    neg = np.array(surf["stok_neg"])[x0]
    add("eta_pos", f"eta+ (\u00b7|x0={x0})", "prob", pos,
        "where the option terminates SUCCESSFULLY, given it started at x0")
    add("eta_neg", f"eta- (\u00b7|x0={x0})", "prob", neg,
        "where it terminates in VIOLATION, given it started at x0")
    add("eta_all", f"eta total (\u00b7|x0={x0})", "prob", pos + neg, "eta+ plus eta-")
    if "pi" in surf:
        add("pi_feas", "pi  (feasibility policy)", "action", surf["pi"],
            "action index chosen in each state: " + ", ".join(
                f"{i}={l}" for i, l in enumerate(ACTION_LABELS)))

    # ---- xi: the arrival-TIME distribution, eta marginalised over final states ------------
    # Everything above is indexed by STATE (one number per grid cell). This is the other axis:
    # eta(t | x0), summed over every final state, is exactly the paper's xi (TED, time-to-event
    # distribution) -- state_spaces.py's own bundle['xi'] is this same quantity for a space's
    # region SPK, so it gets that name here too rather than "eta ... marginal", which named it by
    # how it is COMPUTED (an eta with an axis summed out) instead of what it IS. Answers "how long
    # until this terminates" directly off the server -- the same computation /api/stok_time serves
    # the plot from, read here without going through the panel's fetch/cache path at all. `values`
    # is indexed by TIME, not by cell, so the client must not try to lay it out on the grid.
    time_group = ("chain:" + ",".join(str(g) for g in seq)) if seq else group
    tt = stok_time_slice(world, algo, fparams, time_group, x0)
    add("eta_time_pos", f"xi+ (·|x0={x0})", "time", tt["marginal"],
        "P(terminate SUCCESSFULLY at exactly time t | x0), summed over every final state")
    add("eta_time_neg", f"xi- (·|x0={x0})", "time", tt["marginal_neg"],
        "P(terminate in VIOLATION at exactly time t | x0), summed over every final state")

    # ---- value side ----
    if not seq and group is not None:
        one = dict(world); one["goals"] = [g for g in world["goals"] if g["id"] == group]
        valgo = spec.get("value_algorithm", "ihdc")
        vres = value_world(one, valgo, vparams)
        vs = vres["surfaces"][0]
        add("V", f"V  ({valgo})", "value", vs["v"], "expected discounted return")
        add("pi_value", "pi  (value policy)", "action", vs["pi"], "action index in each state")
        gamma = float(spec.get("sr_gamma", 0.9))
        sr = sr_world(one, "value", valgo, vparams, gamma, group)
        M = np.array(sr["surfaces"][0]["sr"])
        add("sr", f"M(x0={x0}, \u00b7)  gamma={gamma}", "value", M[x0],
            "discounted expected occupancy of each state, starting from x0")

    return {"rows": world["rows"], "cols": world["cols"], "x0": x0, "of": who,
            "actions": ACTION_LABELS, "quantities": out}


def normalize_world(w):
    """Validate + canonicalise a world posted by the browser. Mirrors model.js normalize()."""
    rows, cols = int(w["rows"]), int(w["cols"])
    if not (2 <= rows <= 40 and 2 <= cols <= 40):
        raise ValueError(f"dimensions out of range: {rows}x{cols}")
    n = rows * cols
    out = {"rows": rows, "cols": cols}
    for k in ("walls", "constraints", "green_badgers"):
        vals = sorted({int(v) for v in w.get(k, [])})
        bad = [v for v in vals if not (0 <= v < n)]
        if bad:
            raise ValueError(f"{k}: index out of range for {rows}x{cols}: {bad[:5]}")
        out[k] = vals
    st = w.get("start")
    if st is not None:
        st = int(st)
        if not (0 <= st < n):
            raise ValueError(f"start out of range for {rows}x{cols}: {st}")
    out["start"] = st
    import tutorial_doors
    out.update(tutorial_doors.normalize(w, n, out["walls"]))
    import tutorial_boxes
    out['boxes'] = tutorial_boxes.normalize(w,n,out['walls'])
    import tutorial_workshop
    if w.get("workshop"): out["workshop"] = tutorial_workshop.normalize(w,n,out["walls"])

    p_main = float(w.get("p_main", 1.0))
    if not (0.0 < p_main <= 1.0):
        raise ValueError(f"p_main must be in (0, 1], got {p_main}")
    out["p_main"] = p_main

    # HAND-EDITED TRANSITION ROWS, sparse: {li: {a: {x2: p}}}. Absent means "use the p_main kernel",
    # which is what every world had before this existed. Validated here rather than trusted, since
    # a bad row would otherwise surface as a shape error deep inside the solver.
    kern = w.get("kernel") or {}
    if not isinstance(kern, dict):
        raise ValueError("kernel overrides must be an object keyed by state")
    out_kern = {}
    for li, by_a in kern.items():
        li_i = int(li)
        if not (0 <= li_i < n):
            raise ValueError(f"transition override for state {li_i}, which is outside the grid")
        row = {}
        for a, dist in (by_a or {}).items():
            a_i = int(a)
            if not (0 <= a_i < N_ACTIONS):
                raise ValueError(f"transition override for action {a_i}; there are {N_ACTIONS}")
            clean = {}
            for x2, prob in (dist or {}).items():
                x2_i, pv = int(x2), float(prob)
                if not (0 <= x2_i < n):
                    raise ValueError(f"transition override to state {x2_i}, outside the grid")
                if pv < 0:
                    raise ValueError(f"negative transition probability: {pv}")
                if pv > 0:
                    clean[x2_i] = pv
            if not clean:
                raise ValueError(
                    f"transition override for state {li_i}, action {a_i} has no positive outcome")
            row[a_i] = clean
        if row:
            out_kern[li_i] = row
    out["kernel"] = out_kern

    # GOAL GROUPS. Each group is one goal function g in feas_iter_stationary_flat_dense(P, g, c):
    # its cells are OR'd (reach any), and separate groups are separate SOLVES, not one solve with
    # more targets. Ids must be unique and a cell may belong to at most one group, or "which goal
    # function is this state a target of" has no answer.
    # A goal cell is an ENTRY {"li": cell, "a": action | null}. a = null means the goal is a
    # function of STATE ONLY (any action there terminates); an integer makes it a STATE-ACTION
    # goal, which the solver has always supported since its g is [na, nx].
    # A bare integer is accepted and read as {"li": n, "a": null}, so levels saved before this
    # change still load and behave identically.
    groups, seen_ids, seen_cells = [], set(), set()
    for grp in w.get("goals", []):
        gid = int(grp["id"])
        if gid in seen_ids:
            raise ValueError(f"duplicate goal group id: {gid}")
        seen_ids.add(gid)
        entries = {}
        for c in grp.get("cells", []):
            if isinstance(c, dict):
                li, a = int(c["li"]), c.get("a")
            else:
                li, a = int(c), None
            if not (0 <= li < n):
                raise ValueError(f"goal group {gid}: index out of range for {rows}x{cols}: {li}")
            if a is not None:
                a = int(a)
                if not (0 <= a < N_ACTIONS):
                    raise ValueError(f"goal group {gid}, cell {li}: action {a} out of range "
                                     f"(0..{N_ACTIONS - 1}, or null for a state-only goal)")
            entries[li] = a
        dup = seen_cells & set(entries)
        if dup:
            raise ValueError(f"cell(s) {sorted(dup)[:5]} are in more than one goal group")
        seen_cells |= set(entries)
        groups.append({"id": gid, "cells": [{"li": li, "a": entries[li]}
                                            for li in sorted(entries)]})
    out["goals"] = groups
    out["nextGoalId"] = int(w.get("nextGoalId", max(seen_ids, default=0) + 1))

    # Internal/HL-space anchors and inventory items, keyed by cell -> type index. Same field names
    # as pygame_windows.py / world_spaces.py so a world round-trips through either GUI unchanged.
    # Cells are NOT cross-checked against walls/goals/etc. here (the client already keeps content
    # exclusive per cell via clearContent); the server just validates shape and range.
    def _cell_type_map(key, valid_types):
        raw = w.get(key) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{key} must be an object keyed by cell")
        m = {}
        for li, t in raw.items():
            li_i, t_i = int(li), int(t)
            if not (0 <= li_i < n):
                raise ValueError(f"{key}: cell {li_i} out of range for {rows}x{cols}")
            if t_i not in valid_types:
                raise ValueError(f"{key}: unknown type {t_i} for cell {li_i}")
            m[li_i] = t_i
        return m

    out["internal_goalstates_2_type_ind"] = _cell_type_map(
        "internal_goalstates_2_type_ind", {t["type"] for t in FEATURE_TYPES})
    out["boolean_states_2_type_ind"] = _cell_type_map(
        "boolean_states_2_type_ind", {t["type"] for t in ITEM_TYPES})

    import tutorial_items
    out["item_names"] = tutorial_items.item_names({**out, "item_names":w.get("item_names", {})})

    nis = int(w.get("nis", 5))
    if not (NIS_MIN <= nis <= NIS_MAX):
        raise ValueError(f"nis out of range [{NIS_MIN},{NIS_MAX}]: {nis}")
    out["nis"] = nis
    nbt = w.get("nis_by_type") or {}
    if not isinstance(nbt, dict):
        raise ValueError("nis_by_type must be an object keyed by feature type")
    out_nbt = {}
    valid_ftypes = {t["type"] for t in FEATURE_TYPES}
    for t, v in nbt.items():
        t_i, v_i = int(t), int(v)
        if t_i not in valid_ftypes:
            raise ValueError(f"nis_by_type: unknown feature type {t_i}")
        if not (NIS_MIN <= v_i <= NIS_MAX):
            raise ValueError(f"nis_by_type[{t_i}] out of range [{NIS_MIN},{NIS_MAX}]: {v_i}")
        out_nbt[t_i] = v_i
    out["nis_by_type"] = out_nbt

    # x0 for the HL spaces. internal_state_values is one CURRENT value per placed TYPE (clamped to
    # that type's ns, computed the same way hl_panel_data does); item_held is one bit per placed
    # inventory INSTANCE (by cell), matching the real per-instance BitSpace model in world_spaces.py.
    def _ns_of(t_i):
        return out_nbt.get(t_i, nis)

    isv = w.get("internal_state_values") or {}
    if not isinstance(isv, dict):
        raise ValueError("internal_state_values must be an object keyed by feature type")
    out_isv = {}
    for t, v in isv.items():
        t_i, v_i = int(t), int(v)
        if t_i not in valid_ftypes:
            raise ValueError(f"internal_state_values: unknown feature type {t_i}")
        ns_t = _ns_of(t_i)
        if not (0 <= v_i < ns_t):
            raise ValueError(f"internal_state_values[{t_i}] out of range for ns={ns_t}: {v_i}")
        out_isv[t_i] = v_i
    out["internal_state_values"] = out_isv

    held = w.get("item_held") or {}
    if not isinstance(held, dict):
        raise ValueError("item_held must be an object keyed by cell")
    out_held = {}
    for li, v in held.items():
        li_i = int(li)
        if not (0 <= li_i < n):
            raise ValueError(f"item_held: cell {li_i} out of range for {rows}x{cols}")
        if v:
            out_held[li_i] = 1
    out["item_held"] = out_held

    # The DNF task objective: a list of CLAUSES (OR of them), each a list of LITERALS [space, op,
    # value] (AND of them) -- task_gui.py's model verbatim, so a browser-authored task and a
    # pygame-authored one mean exactly the same thing. Only shape-checked here (list of lists of
    # 3-element literals); semantic checks (space exists, value in range, ...) need the space
    # catalog and run in task_panel_data via task_gui.validate, not on every save/solve request.
    tc = w.get("task_clauses") or []
    if not isinstance(tc, list):
        raise ValueError("task_clauses must be a list of clauses")
    out_tc = []
    for ci, cl in enumerate(tc):
        if not isinstance(cl, list):
            raise ValueError(f"task_clauses[{ci}] must be a list of literals")
        out_cl = []
        for li_, lit in enumerate(cl):
            if not (isinstance(lit, (list, tuple)) and len(lit) == 3):
                raise ValueError(f"task_clauses[{ci}][{li_}] must be [space, op, value]")
            sp, op, val = lit
            out_cl.append([str(sp), str(op), val])
        out_tc.append(out_cl)
    out["task_clauses"] = out_tc
    out["task_clause_ix"] = max(0, min(len(out_tc) - 1, int(w.get("task_clause_ix", 0)))) if out_tc else 0
    return out


def world_to_setup(w):
    """The world in the shape the Python side uses -- a wall MATRIX plus linear-index sets, matching
    pygame_windows / levels.py. li = row*cols + col, row-major (verified against StateSpace_2D)."""
    rows, cols = w["rows"], w["cols"]
    grid = [[0] * cols for _ in range(rows)]
    for li in w["walls"]:
        grid[li // cols][li % cols] = 1
    return {
        "grid": grid,
        "start_state": w["start"],
        "constraint_states": list(w["constraints"]),
        # one entry per goal FUNCTION -- stage 3 runs feasibility iteration once per group
        "goal_groups": [{"id": g["id"], "cells": [dict(c) for c in g["cells"]]}
                        for g in w["goals"]],
        "rows": rows, "cols": cols,
    }


def hl_panel_data(world):
    """Placement metadata from the shared catalog, without constructing transition kernels."""
    import world_spaces
    catalog = world_spaces.catalog_from_placement(
        world["internal_goalstates_2_type_ind"], {}, world["nis"], nis_by_type=world["nis_by_type"])["spaces"]
    by_cell = world["internal_goalstates_2_type_ind"]
    values = world["internal_state_values"]
    features = []
    for t, entry in zip(sorted(set(by_cell.values())), catalog):
        cells = sorted(li for li, tt in by_cell.items() if tt == t)
        ns = int(entry["ns"])
        # A type with no explicit x0 yet reads as TOPPED UP (ns - 1), not 0 -- 0 is a death/broke
        # state for most types, and a freshly placed bar silently starting there would be a trap.
        value = max(0, min(ns - 1, int(values.get(t, ns - 1))))
        features.append({
            "type": t, "name": entry["name"], "ns": ns, "na": int(entry["na"]), "value": value,
            "has_death": t not in world_spaces.NO_DEATH_TYPES, "cells": cells,
        })
    held = world["item_held"]
    items = [{**spec, "held": bool(held.get(spec["cell"], 0))}
             for spec in __import__("tutorial_items").item_specs(world)]
    import tutorial_doors
    import tutorial_workshop
    import tutorial_boxes
    return {"features": features, "items": items, "special_actions":tutorial_workshop.special_actions(world),
            "boxes":world.get('boxes',[]), "box_state":tutorial_boxes.current(world),
            "workshop_state":tutorial_workshop.current(world,{**{f["name"]:f["value"] for f in features},**{i["name"]:int(i["held"]) for i in items}}),
            "workshop_catalog":tutorial_workshop.catalog(world),
            "doors": [dict(d, name=tutorial_doors.name(d)) for d in world.get('doors', [])]}


def _task_host(world):
    """A minimal stand-in for the arcade Setup object that task_gui.py's derived_sets/host_catalog
    were written against -- built from the tutorial's plain world dict instead of pygame state.
    Only the attributes those two functions actually read (defensively, via getattr with a
    fallback) are supplied; there is no colour/door/NPC concept in the tutorial yet."""
    import types
    rows, cols = world["rows"], world["cols"]
    grid = [[0] * cols for _ in range(rows)]
    for li in world["walls"]:
        grid[li // cols][li % cols] = 1
    t2i, t2b = {}, {}
    for li, t in world["internal_goalstates_2_type_ind"].items():
        t2i.setdefault(t, []).append(li)
    for li, t in world["boolean_states_2_type_ind"].items():
        t2b.setdefault(t, []).append(li)
    return types.SimpleNamespace(
        grid=grid,
        internal_goalstates_2_type_ind=world["internal_goalstates_2_type_ind"],
        boolean_states_2_type_ind=world["boolean_states_2_type_ind"],
        nis=world["nis"], nis_by_type=world["nis_by_type"],
        type_2_internal_goalstates=t2i, type_2_boolean_states=t2b,
    )


def task_catalog(world):
    """Every space a task literal can name: X plus every placed internal space and item instance.
    world_spaces.catalog_from_placement is construction-free (no state_spaces objects built), so
    this is cheap to call on every panel refresh -- the SAME catalog world_spaces would hand the
    real solver, so the picker can never offer a space the solver would not build."""
    import world_spaces as ws
    import tutorial_doors
    result = ws.catalog_from_placement(
        world["internal_goalstates_2_type_ind"], world["boolean_states_2_type_ind"],
        nis=world["nis"], nis_by_type=world["nis_by_type"],
        x_ns=world["rows"] * world["cols"])["spaces"] + tutorial_doors.catalog(world)
    import tutorial_workshop
    if world.get('workshop',{}).get('movement_metabolism'):
        for c in result:
            if c['name'] in ('hunger','hydration'):c['na']=3;c['action_labels']=['deplete','refill','hold']
    if tutorial_workshop.canteen_enabled(world) or tutorial_workshop.tree_cells(world):
        for c in result:
            if c['name']=='X':
                c['na']=6+len(tutorial_workshop.special_actions(world)); c['action_labels']=['interact','stay','down','right','up','left']+[a['label'] for a in tutorial_workshop.special_actions(world)]
    import tutorial_boxes
    import tutorial_items
    result = tutorial_items.rename_catalog(world, result)
    return result + [c for c in tutorial_workshop.catalog(world) if c["name"] not in {r["name"] for r in result}] + tutorial_boxes.catalog(world)


def play_task_status(world, state=None, clauses=None):
    """Evaluate manual play using the authored joint goal and the world's constraints."""
    from types import SimpleNamespace
    import task_gui as tg
    import world_spaces as ws
    spaces = {c['name']: SimpleNamespace(ns=c['ns']) for c in task_catalog(world)}
    if state is None:
        if world.get('start') is None:
            return dict(complete=False, violations=[], ready=False)
        state = inspect_x0(world, spaces)
    clauses = world.get('task_clauses', []) if clauses is None else clauses
    host = _task_host(world)
    ready = bool(clauses) and not tg.validate(clauses, spaces, host)
    violations = []
    if state.get('X') in world['constraints']:
        violations.append('Entered a grid constraint')
    for t in set(world['internal_goalstates_2_type_ind'].values()):
        name = ws.TYPE_NAMES.get(t)
        if t not in ws.NO_DEATH_TYPES and state.get(name) == 0:
            violations.append(f'{name} reached its death state (0)')
    complete = ready and not violations and tg.satisfied(clauses, state, host)
    return dict(complete=bool(complete), violations=violations, ready=bool(ready))


def task_panel_data(world, clauses):
    """Everything the Task panel needs in one round trip -- summary, validation, a human-readable
    description, derived-set chips, and the space catalog for the custom-literal picker -- all from
    task_gui.py, the SAME module the arcade editor's T panel uses. Porting the browser panel onto
    these functions (rather than re-deriving the DNF semantics) means a browser-authored task and a
    pygame-authored one are validated, counted, and worded identically."""
    import task_gui as tg
    host = _task_host(world)
    catalog = task_catalog(world)
    spaces_by_name = {c["name"]: c for c in catalog}
    ns_by_name = {c["name"]: int(c["ns"]) for c in catalog}
    return {
        "play_status": play_task_status(world, clauses=clauses),
        "catalog": catalog,
        "summary": tg.summarize(clauses, host),
        "errors": tg.validate(clauses, spaces_by_name, host),
        "describe": tg.describe(clauses),
        "human": tg.humanize(clauses, ncol=world["cols"], spaces=ns_by_name),
        "human_constraints": tg.humanize_constraints(world["constraints"], {}, ncol=world["cols"]),
        "derived_sets": [{"label": lbl, "space": sp, "values": sorted(int(v) for v in vs)}
                         for (lbl, sp, vs) in tg.derived_sets(host)],
    }


def build_inspect_spaces(world, t_f=40):
    """Every space this world's placement can name (X + every placed internal/item space) as REAL
    OKBEKernelMixin-capable objects, keyed by name -- the SAME names task_gui.py's DNF literals and
    world_spaces.py's catalog already use. X is wrapped in a minimal OKBEKernelMixin host built
    from the SAME kernel the flat solver uses (get_kernel), so the two can never disagree; its
    constraint_vec comes from the flat 'constraint' tool's cells, giving it a real c_k the way a
    bare HL space would not have. (state_spaces.StateSpace_2D predates OKBEKernelMixin and has no
    constraint_vec/region_functions of its own, which is why it needs this wrapper and the HL/item
    spaces below -- already OKBEKernelMixin subclasses -- do not.)"""
    import numpy as np
    import state_spaces as ss
    import world_spaces as ws
    rows, cols = world["rows"], world["cols"]
    sp, P = get_world_kernel(world, product=True)
    x = ss.OKBEKernelMixin()
    x.name, x.ns, x.na = "X", int(P.shape[1]), int(P.shape[0])
    x.max_time = t_f
    x.P_a_s_s = P
    cvec = np.ones(x.ns)
    for li in world["constraints"]:
        cvec[int(li)] = 0.0
    x.constraint_vec = cvec
    x.default_action_ind = 1                      # 'stay' -- ACTION_LABELS[1]; 0 is 'interact'
    x.region_functions = {}
    spaces = {"X": x}
    hl_spaces, _meta = ws.build_spaces_from_placement(
        world["internal_goalstates_2_type_ind"], world["boolean_states_2_type_ind"],
        world["nis"], nis_by_type=world["nis_by_type"], T_f=t_f)
    import tutorial_items
    spaces.update(tutorial_items.rename_spaces(world, hl_spaces))
    import tutorial_doors
    tutorial_doors.attach(spaces, world)
    import tutorial_workshop
    tutorial_workshop.attach(spaces, world)
    import tutorial_boxes
    tutorial_boxes.attach(spaces,world)
    return spaces


_DBN_MODEL_CACHE = {}  # One model only; bounded by the product builder's limits.
_DBN_PROGRESS = {}
_DBN_PROGRESS_LOCK = threading.Lock()


def dbn_progress(body, stage):
    request_id = body.get("progress_id")
    if isinstance(request_id, str) and len(request_id) <= 100:
        with _DBN_PROGRESS_LOCK:
            if request_id in _DBN_PROGRESS:
                _DBN_PROGRESS[request_id] = stage


def dbn_panel_data(world, body):
    """Independent product-space workbench; it does not change the BL solver."""
    import dbn_composer
    import world_spaces as ws
    import tutorial_doors as doors
    import tutorial_affordances
    spec = tutorial_affordances.prepare(world, body.get("spec", {}))
    dynamics = doors.dynamics(world)
    key = json.dumps([dynamics, spec], sort_keys=True)
    operation = body.get("operation")
    if operation in ("solve", "slice", "eta"):
        import product_solver as ps
        options = body.get("options", {})
        if operation == "solve":
            dbn_progress(body, 'Building and checking sparse kernel')
            build_seconds = 0.0
            if _DBN_MODEL_CACHE.get("key") != key or "csr" not in _DBN_MODEL_CACHE.get("model", {}):
                started = time.perf_counter()
                retained = {}
                dbn_composer.evaluate(build_inspect_spaces(world), spec, build=True, retain=retained)
                _DBN_MODEL_CACHE.clear()
                _DBN_MODEL_CACHE.update(key=key, model=retained)
                build_seconds = time.perf_counter() - started
            model = _DBN_MODEL_CACHE["model"]
            dbn_progress(body, 'Preparing objective')
            clauses = None
            if options.get("objective") == "task":
                import task_gui
                clauses = task_gui.resolve(world.get("task_clauses", []), _task_host(world))
                errors = task_gui.validate(clauses, {s.name: s for s in model["spaces"]})
                if errors:
                    raise ValueError("; ".join(errors))
            metadata = hl_panel_data(world)
            death_names = [f["name"] for f in metadata["features"] if f["has_death"]]
            g, c = ps.objective(model, world, options, clauses, death_names)
            algorithm = body.get("algorithm", "feas_eta_fast_thresh")
            if algorithm not in ALGO_BY_ID and algorithm not in VALUE_BY_ID:
                raise ValueError("Unknown product algorithm")
            definition = ALGO_BY_ID.get(algorithm, VALUE_BY_ID.get(algorithm))
            params = {p["id"]: p["default"] for p in definition["params"]}
            params.update(body.get("params", {}))
            dbn_progress(body, 'Solving policy')
            solution = ps.solve(model, g, c, algorithm, params)
            dbn_progress(body, 'Preparing plots')
            solution["id"] = str(time.time_ns())
            _DBN_MODEL_CACHE["solution"] = solution
            fixed = {s.name: (0 if s.name in {x["name"] for x in metadata["items"]} else s.ns - 1)
                     for s in model["spaces"][1:]}
            fixed.update(doors.current(world))
        else:
            if _DBN_MODEL_CACHE.get("key") != key:
                raise ValueError("World or draft changed. Solve the product again.")
            model = _DBN_MODEL_CACHE["model"]
            solution = _DBN_MODEL_CACHE.get("solution")
            if not solution or body.get("solution_id") != solution["id"]:
                raise ValueError("This product solution is no longer cached. Solve again.")
            fixed = body.get("fixed", {})
        if operation == 'eta':
            result = ps.termination_row(solution, body.get('start', 0), body.get('outcome', 'positive'),
                                        body.get('offset', 0), body.get('limit', 2000))
            result['spaces'] = [{'name': s.name, 'ns': s.ns} for s in model['spaces']]
            result['solution_id'] = solution['id']
            return result
        result = ps.slice_solution(model, solution, fixed, body.get("cell", world.get("start") or 0),
                                   world["rows"], world["cols"], world["walls"])
        result.update(solution_id=solution["id"], algorithm=solution["algorithm"],
                      spaces=[{"name": s.name, "ns": int(s.ns)} for s in model["spaces"]],
                      seconds=solution["seconds"], phases=solution["phases"], notes=solution["notes"])
        if operation == "solve":
            result.update(build_seconds=build_seconds, joint_states=int(model["csr"][0].shape[0]))
        return result
    if body.get("operation") == "step":
        from types import SimpleNamespace
        spaces_for_state = {c['name']:SimpleNamespace(ns=c['ns']) for c in task_catalog(world)} if body.get('use_world_state') else None
        if spaces_for_state is not None:
            step_state = {**(body.get('state') or {}), **inspect_x0(world, spaces_for_state)}
        else:
            step_state = {**(body.get("state") or {}), **doors.current(world)}
        if _DBN_MODEL_CACHE.get("key") != key:
            if body.get("mode") == "tensor":
                raise ValueError("Build sparse product for the current world and draft first")
            spaces = build_inspect_spaces(world)
            dbn_composer.evaluate(spaces, spec, step_state, body.get("action", 0))
            ordered, affs, _ = dbn_composer.compile_model(spaces, spec)
            _DBN_MODEL_CACHE.clear()
            _DBN_MODEL_CACHE.update(key=key, model={"spaces": ordered, "affs": affs})
        result = dbn_composer.step_model(_DBN_MODEL_CACHE["model"], step_state,
                                        body.get("action", 0), body.get("mode", "factorized"))
        state = result["state"]
        result['play_status_before'] = play_task_status(world, step_state)
        result['play_status'] = play_task_status(world, state)
        values = dict(world["internal_state_values"])
        for t in set(world["internal_goalstates_2_type_ind"].values()):
            if ws.TYPE_NAMES.get(t) in state:
                values[t] = state[ws.TYPE_NAMES[t]]
        held = dict(world["item_held"])
        for item in __import__("tutorial_items").item_specs(world):
            if item["name"] in state:
                held[item["cell"]] = state[item["name"]]
        result["world_state"] = {"start": state["X"], "internal_state_values": values, "item_held": held}
        if world.get('workshop'):
            import tutorial_workshop
            result['world_state']['workshop'] = {**world['workshop'], 'values':{c['name']:state[c['name']] for c in tutorial_workshop.catalog(world)}}
        result['spec'] = spec
        result['world_state']['doors'] = [dict(d, open=bool(state[doors.name(d)])) for d in world.get('doors', [])]
        result['world_state']['boxes'] = [dict(b,cell=state['box_'+str(b['id'])]) for b in world.get('boxes',[])]
        return result
    dbn_progress(body, 'Building and checking sparse kernel' if operation == 'build' else 'Preparing factors')
    spaces = build_inspect_spaces(world)
    if body.get("operation") == "catalog":
        catalog = ws.catalog_from_placement(world["internal_goalstates_2_type_ind"],
                    world["boolean_states_2_type_ind"], world["nis"],
                    nis_by_type=world["nis_by_type"], x_ns=spaces["X"].ns, na_free=spaces["X"].na)
        for row in catalog["spaces"]:
            if row["name"] == "X":
                row["action_labels"] = ["interact", "stay", "down", "right", "up", "left"]
        catalog['spaces'] = task_catalog(world)
        return {"catalog": catalog, "state": inspect_x0(world, spaces),
                "spec": spec}
    state = body.get("state")
    if state is None:
        state = inspect_x0(world, spaces)
    else:
        state = {**doors.current(world), **state}
    retained = {} if body.get("operation") == "build" else None
    result = dbn_composer.evaluate(spaces, spec, state, body.get("action", 0),
                                  build=retained is not None, retain=retained)
    if retained is not None:
        _DBN_MODEL_CACHE.clear()
        _DBN_MODEL_CACHE.update(key=key, model=retained)
    return result


def inspect_x0(world, spaces):
    """This world's x0 in the {space_name: state_index} shape AffordanceSet wants -- the SAME
    default-to-topped-up / held-bit rules hl_panel_data uses, so what this panel shows always
    matches the HL panel's bars/inventory."""
    import world_spaces as ws
    out = {}
    if "X" in spaces:
        out["X"] = int(world["start"]) if world["start"] is not None else 0
    values = world["internal_state_values"]
    for t in sorted(set(world["internal_goalstates_2_type_ind"].values())):
        nm = ws.TYPE_NAMES.get(t)
        if nm in spaces:
            ns = spaces[nm].ns
            out[nm] = max(0, min(ns - 1, int(values.get(t, ns - 1))))
    held = world["item_held"]
    for spec in __import__("tutorial_items").item_specs(world):
        if spec["name"] in spaces:
            out[spec["name"]] = 1 if held.get(spec["cell"], 0) else 0
    import tutorial_doors
    out.update(tutorial_doors.current(world))
    import tutorial_workshop
    out.update(tutorial_workshop.current(world,out))
    import tutorial_boxes
    out.update({n:v for n,v in tutorial_boxes.current(world).items() if n in spaces})
    return out


def compute_phi(aff, spaces, x0, t_f):
    """SED phi_r(sigma | x0, t_f) = prod_k nu_k^sigma_k (1-nu_k)^(1-sigma_k) / (1 - prod_k(1-nu_k)),
    over every PLACED space (paper sec_event_dist). Direct enumeration of the 2^n sigma vectors --
    cheap, since a world realistically has a handful of placed spaces, not hundreds of them."""
    import itertools
    names = sorted(x0)
    nus = []
    for nm in names:
        sp = spaces[nm]
        region_key = aff.region_of(nm)
        bundle = aff.region_functions_for(sp, region_key, bl_action=0)
        s_i = max(0, min(sp.ns - 1, x0[nm]))
        t_i = min(t_f, bundle["nu"].shape[1] - 1)
        nus.append(float(bundle["nu"][s_i, t_i]))
    p_none = 1.0
    for nu in nus:
        p_none *= (1.0 - nu)
    denom = 1.0 - p_none
    sigmas = []
    if denom > 1e-12:
        for bits in itertools.product([0, 1], repeat=len(names)):
            if not any(bits):
                continue
            p = 1.0
            for b, nu in zip(bits, nus):
                p *= (nu if b else (1.0 - nu))
            sigmas.append({"sigma": dict(zip(names, bits)), "p": p / denom})
        sigmas.sort(key=lambda r: -r["p"])
    return {"nu": dict(zip(names, nus)), "p_none": p_none, "sigma": sigmas[:16]}


def inspect_panel_data(world, space_name, s_i, t_max, t_inspect):
    """kappa / kappa_bar (paper's SF, code name 'sf') / xi / nu as series over t=0..t_max-1, plus
    eta / rho / phi at one t_inspect -- from AffordanceSet.region_functions_for / spk_for
    (state_spaces.py), the SAME machinery okbe_main / exchange_world.py use for this. No
    affordances are registered yet (the tutorial has no affordance-authoring UI wired up), so every
    space's region is just its own default action with g_k=0 -- an honest survival/hazard analysis
    under drift, not yet gated by any goal or cross-space trigger. That is what g_k=0 means in
    AffordanceSet.build_h_r, not a simplification specific to this endpoint; it is the accurate
    answer for "no affordance authored on this space yet", and will sharpen once one is."""
    import numpy as np
    import state_spaces as ss
    t_f = max(int(t_max), int(t_inspect) + 1, 8)
    spaces = build_inspect_spaces(world, t_f=t_f + 8)
    if space_name not in spaces:
        raise ValueError(f"unknown space {space_name!r} (have {sorted(spaces)})")
    x0 = inspect_x0(world, spaces)
    aff = ss.AffordanceSet(spaces=list(spaces.values()))
    sp = spaces[space_name]
    s_i = max(0, min(sp.ns - 1, int(s_i)))
    region_key = aff.region_of(space_name)
    bundle = aff.region_functions_for(sp, region_key, bl_action=0)
    T = min(int(t_max), bundle["kappa"].shape[1])
    t_i = min(int(t_inspect), bundle["kappa"].shape[1] - 1)

    def top_states(vec, k=8):
        idx = np.argsort(vec)[::-1]
        return [{"state": int(i), "p": float(vec[i])} for i in idx[:k] if vec[i] > 1e-6]

    eta_row = bundle["eta"][s_i, :, t_i]
    # SPK convention (state_spaces.SPK): index [1]=in (event fired within this space, row-
    # normalized) / [0]=out (survived in-region); row s_i of the [ns,ns] matrix at time t_i is
    # the conditional final-state distribution from s_i.
    spk = aff.spk_for(sp, region_key, int(sp.default_action_ind), bl_action=0)
    rho_in = np.asarray(spk.in_(t_i).todense())[s_i]
    rho_out = np.asarray(spk.out(t_i).todense())[s_i]

    return {
        "space": space_name, "ns": int(sp.ns), "s_i": s_i, "t_max": T, "t_inspect": t_i,
        "default_action": int(sp.default_action_ind),
        "series": {
            "kappa": bundle["kappa"][s_i, :T].tolist(),
            "kappa_bar": bundle["sf"][s_i, :T].tolist(),
            "xi": bundle["xi"][s_i, :T].tolist(),
            "nu": bundle["nu"][s_i, :T].tolist(),
        },
        "eta": top_states(eta_row),
        "rho_in": top_states(rho_in), "rho_out": top_states(rho_out),
        "phi": compute_phi(aff, spaces, x0, t_i),
        "x0": x0,
    }


def value_world(world, algo_id, params):
    """Value iteration over the same flat world, once per goal group."""
    import numpy as np
    import scipy.sparse as sps
    import state_spaces as ss
    import utilities

    if algo_id not in VALUE_BY_ID:
        raise ValueError(f"unknown value algorithm: {algo_id}")

    rows, cols = world["rows"], world["cols"]
    # Timed in PARTS, as in solve_world: the endpoint clock also covers the kernel and the JSON.
    _t_kernel = time.time()
    sp, P = get_world_kernel(world)
    timing = {"kernel": round(time.time() - _t_kernel, 4), "solve": 0.0, "groups": 0}
    na, nx = P.shape[0], P.shape[1]

    step = float(params.get("step_cost", 1.0))
    pen = float(params.get("constraint_penalty", 10.0))
    goal_r = float(params.get("goal_reward", 0.0))

    out = []
    for grp in world["goals"]:
        if not grp["cells"]:
            out.append({"id": grp["id"], "empty": True})
            continue
        goals = [e["li"] for e in grp["cells"]]
        r = np.full((na, nx), -step)
        for li in world["constraints"]:
            r[:, li] -= pen
        for li in goals:
            # the reward is on the CELL, not the terminal action: V answers "how good is it to be
            # here", and making it action-specific would mean V no longer matched its own policy
            r[:, li] = goal_r          # 0 = the goal is merely free; > 0 = actively rewarding

        if algo_id == "ssp" and goal_r:
            # Guard rather than quietly drop it: silently ignoring a slider the user is dragging is
            # worse than saying why it cannot apply.
            raise ValueError("goal reward does not apply to min-cost shortest path: with gamma=1 "
            "an absorbing goal accumulates the reward every iteration, so V never "
            "converges. Use the discounted algorithm for a goal reward.")

        Pk = P.copy()
        if algo_id == "ssp":
            # First-exit needs an ABSORBING, free goal: without it, undiscounted iteration keeps
            # paying the step cost forever and V runs to minus infinity.
            for li in goals:
                Pk[:, li, :] = 0.0
                Pk[:, li, li] = 1.0
            gamma = 1.0
        else:
            gamma = float(params.get("gamma", 0.95))

        # action-OUTER flattening: row index is (a, s), matching value_iteration_ihdc_outer_action
        T = sps.csr_matrix(Pk.reshape(na * nx, nx))
        _t_solve = time.time()
        v, _pi_lib = utilities.value_iteration_ihdc_outer_action(T, r, gamma, nx, na)
        timing["solve"] += time.time() - _t_solve
        timing["groups"] += 1

        # The library returns pi via `v_sxa_new.reshape(ns, na).argmax(axis=1)` while its own
        # iteration uses reshape(na, ns) -- the two disagree, so the returned policy is indexed
        # wrongly for an action-outer layout. Recompute it here rather than trust it.
        q = r + gamma * (Pk.reshape(na * nx, nx).dot(v)).reshape(na, nx)
        pi = q.argmax(axis=0)

        v = np.asarray(v, dtype=float)
        out.append({
            "id": grp["id"], "empty": False, "cells": [dict(c) for c in grp["cells"]],
            "v": [round(float(x), 6) for x in v],
            "pi": [int(x) for x in pi],
            "vmin": round(float(v.min()), 6), "vmax": round(float(v.max()), 6),
            "start": None if world["start"] is None else round(float(v[world["start"]]), 6),
        })
    timing["solve"] = round(timing["solve"], 4)
    return {"algorithm": algo_id, "params": params, "rows": rows, "cols": cols,
            "na": int(na), "nx": int(nx), "surfaces": out, "timing": timing}


def sr_world(world, source, algo_id, params, gamma=0.95, group_id=None, policy=None,
             sequence=None):
    """Successor representation under a solved policy: M = (I - gamma * P_pi)^-1.

    NOTE ON PROVENANCE: there is no successor-representation code in this repo, so this is computed
    here. It is a DEFINITION rather than an algorithm choice -- M is the discounted expected state
    occupancy under a fixed policy, and the closed form is a single linear solve. The POLICY still
    comes entirely from your solvers (feas / feas_thresh / ihdc / ssp); nothing here optimises
    anything. MarkovChainFunctions has related fundamental-matrix code, but it deletes terminal
    rows and columns, which renumbers the states and so cannot be plotted over the grid.
    """
    import numpy as np
    from successor_sparse import successor_matrix

    started = time.perf_counter()
    timing = {"sr": 0.0}

    def build_sr(kernel):
        t0 = time.perf_counter()
        result = successor_matrix(kernel, gamma)
        timing["sr"] += time.perf_counter() - t0
        return result

    rows, cols = world["rows"], world["cols"]
    groups = [g for g in world["goals"] if g["cells"]]
    if not groups:
        raise ValueError("no goal cells placed")

    sp, P = get_world_kernel(world)
    na, nx = P.shape[0], P.shape[1]
    g = float(gamma)
    if not (0.0 < g < 1.0):
        raise ValueError(f"successor representation needs 0 < gamma < 1, got {g}")

    # A STRING of policies: compose their successor representations by MATRIX PRODUCT,
    #     M_c = M_1 M_2 ... M_k
    # Note this is NOT the successor representation of the composed policy -- that would require
    # the chain induced by running them in sequence, and a product of resolvents is not a
    # resolvent. It is a two-stage occupancy: how much time pi_2 spends in s' having been handed
    # the discounted occupancy pi_1 produced. Offered because it is worth LOOKING at, not because
    # it means what the single-policy SR means.
    if sequence:
        seq = [int(g) for g in sequence]
        by_id = {g["id"]: g for g in world["goals"]}
        for gid in seq:
            if gid not in by_id or not by_id[gid]["cells"]:
                raise ValueError(f"policy {gid} does not exist or has no goal cells")
        M = None
        pis = []
        for gid in seq:
            one = dict(world)
            one["goals"] = [by_id[gid]]
            res = value_world(one, algo_id, params) if source == "value" \
                else solve_world(one, algo_id, params)
            pi = np.asarray(res["surfaces"][0]["pi"], dtype=int)
            pis.append(pi)
            Mi = build_sr(P[pi, np.arange(nx), :])
            M = Mi if M is None else M.dot(Mi)
        return {"timing": timing,
                "rows": rows, "cols": cols, "nx": int(nx), "gamma": g,
                "source": source, "algorithm": algo_id, "sequence": seq, "composed": True,
                # the flag rides on the SURFACE, not just the result: the view reads one surface
                # at a time and cannot see the envelope around it
                "surfaces": [{"id": "\u2217", "empty": False, "composed": True, "sequence": seq,
                              "sr": [[round(float(v), 5) for v in row] for row in M],
                              "pi": [int(v) for v in pis[-1]]}],
                "seconds": time.perf_counter() - started}

    # One SR per goal GROUP, exactly like every other view. Each group has its own policy, so it
    # has its own induced chain and its own occupancy -- returning only one collapsed the panel to
    # a single group the moment you switched to this tab.
    out = []
    for grp in world["goals"]:
        if not grp["cells"]:
            out.append({"id": grp["id"], "empty": True})
            continue
        if policy is not None and len(policy) == nx and grp["id"] == (group_id or groups[0]["id"]):
            pi = np.asarray(policy, dtype=int)
        else:
            one = dict(world)
            one["goals"] = [grp]
            res = value_world(one, algo_id, params) if source == "value" \
                else solve_world(one, algo_id, params)
            pi = np.asarray(res["surfaces"][0]["pi"], dtype=int)
        # P_pi[s, s'] -- the chain this policy induces. Nothing here knows or cares which solver
        # produced pi, so any policy works.
        P_pi = P[pi, np.arange(nx), :]
        M = build_sr(P_pi)
        out.append({"id": grp["id"], "empty": False, "cells": [dict(c) for c in grp["cells"]],
                    "sr": [[round(float(v), 5) for v in row] for row in M],
                    "pi": [int(v) for v in pi]})
    return {"timing": timing,
                "rows": rows, "cols": cols, "nx": int(nx), "gamma": g,
            "source": source, "algorithm": algo_id, "surfaces": out,
            "seconds": time.perf_counter() - started}


def fundamental_world(world, algo_id, params, group_id=None, policy=None, sequence=None):
    """The UNDISCOUNTED fundamental matrix under the solved feasibility policy: N = (I - Q)^-1,
    where Q is P_pi restricted to the TRANSIENT states (everything but walls/goals/constraints).
    N[i, j] is the expected number of visits to transient state j starting from transient state i,
    before absorption -- the un-discounted twin of sr_world's SR above, with terminal states
    actually removed from the linear system rather than merely down-weighted by gamma.

    Uses utilities.compute_fundamental_matrix_with_absorption, which embeds N back into a FULL
    [nx, nx] matrix (walls/goals/constraints as all-zero rows) rather than the renumbered
    transient-only form most fundamental-matrix code returns -- renumbering would need translating
    back before this could be indexed by cell or plotted over the grid at all.

    Goal/constraint COLUMNS hold something different in kind from the rest of the row: not a visit
    count but B = N @ R, the probability of being absorbed there -- so a row mixes an expected
    count in most columns with a probability in a few. Left as-is (not split into two arrays)
    because that mixing is exactly what the source function returns, and the client documents it
    rather than hiding it.

    HOPELESS STATES are folded into the wall list, not left transient. compute_fundamental_matrix_
    with_absorption inverts (I - Q) over every non-wall/goal/constraint cell, which assumes that
    block is genuinely transient -- every state in it eventually leaves, with probability 1, to a
    goal or a constraint. That is NOT true of a state where kappa (success) AND kappa_viol
    (cumulative violation) are both 0: nothing ever absorbs it, so pi there is arbitrary (every
    action ties at 0 in the OBE), and if that arbitrary tie-break happens to close a loop -- one
    hopeless state's action pointing at another, or at itself -- Q's spectral radius there hits 1
    and (I - Q) is exactly singular. This is not rare: feas_thresh_s gates OUT every state whose
    unconstrained kappa_c exceeds theta (f_thresh = h_thresh = 0 there, so kappa_ax is uniformly 0
    and pi is whatever np.argmax(all-zeros) picks), so a low theta can hopeless-gate a large block
    of the grid at once -- reported as "LinAlgError: Singular matrix" on this exact tab before this
    fix, reproducing at theta=0.4 and not at theta=0.75 on the SAME world, since fewer states are
    gated at the higher threshold. A hopeless state's own fundamental-matrix row is correctly all
    zero either way (it has no finite expected-visits count worth reporting, precisely because it
    never leaves) -- exactly what excluding it like a wall produces.

    A STRING composes as a WEIGHTED SUM of each leg's own N, not a flat matrix sum: leg i+1 does
    not start fresh from x0, it starts from wherever leg i's process handed off, which is a
    DISTRIBUTION (leg i's own kappa-worth of mass, spread over its goal cells, one more step under
    handover_kernel if a cell's goal is state-action). So each leg's contribution is `reach @ N_i`,
    where `reach` is that arrival distribution -- itself just leg (i-1)'s goal-column mass pushed
    through ITS handover -- propagated leg to leg starting from reach_1 = I (leg 1 starts exactly
    at x0, certainly).

    Whether a non-final leg's OWN goal columns are zeroed out of its contribution to the total
    depends on the handoff TYPE, and getting this wrong silently drops or double-counts exactly
    one visit -- both were caught by test_fundamental_string.py's Monte Carlo/exact-chain checks
    before the fix below, so trust those over intuition if this is ever touched again:
      STATE-ONLY : zeroed. Leg i+1 starts AT that SAME cell, on the SAME timestep leg i arrives --
                   its own identity term ((I-Q)^-1 always counts the start state once) already
                   counts that one visit, so counting it here too would double it.
      STATE-ACTION: kept. Reaching the cell and EXECUTING the terminal action are two genuinely
                   different timesteps (handover_kernel's own D[x,:,1] = P[a,x,:] is a real
                   kernel step, landing leg i+1 somewhere else entirely in general) -- the visit
                   to the goal cell itself is real and is not counted by anything else in the sum.
    Constraint columns are simpler: a violation truly ends the whole string then and there, at
    WHICHEVER leg it happens in, so every leg's constraint columns accumulate into the total
    untouched, at every leg, always. NOTE this inherits handover_kernel's own documented
    limitation -- "the handover step is not itself checked against constraints" -- so a terminal
    action whose OWN noise slips onto a constraint cell is not caught as a violation there; it is
    treated as an ordinary landing and handed to leg i+1's dynamics from that cell instead. Shared
    with STOK/eta composition (compose_sequence), not something specific to this function.
    """
    import numpy as np
    from utilities import compute_fundamental_matrix_with_absorption

    rows, cols = world["rows"], world["cols"]
    groups = [g for g in world["goals"] if g["cells"]]
    if not groups:
        raise ValueError("no goal cells placed")

    sp, P = get_world_kernel(world)
    na, nx = P.shape[0], P.shape[1]

    def leg_fundamental(grp, fixed_policy=None):
        """(N, pi) for ONE goal group's own solved policy, hopeless states folded in with walls
        (see this function's own docstring above)."""
        if fixed_policy is not None and len(fixed_policy) == nx:
            pi = np.asarray(fixed_policy, dtype=int)
            walls = world["walls"]
        else:
            one = dict(world); one["goals"] = [grp]
            res = solve_world(one, algo_id, params)
            surf = res["surfaces"][0]
            pi = np.asarray(surf["pi"], dtype=int)
            total_absorb = np.asarray(surf["kappa"]) + np.asarray(surf["kappa_viol"])
            hopeless = np.nonzero(total_absorb <= 1e-9)[0]
            walls = sorted(set(world["walls"]) | set(int(x) for x in hopeless))
        P_pi = P[pi, np.arange(nx), :]
        goalstates = [c["li"] for c in grp["cells"]]
        N = compute_fundamental_matrix_with_absorption(P_pi, goalstates, world["constraints"], walls)
        return N, pi, goalstates

    if sequence:
        seq = [int(g) for g in sequence]
        by_id = {g["id"]: g for g in world["goals"]}
        for gid in seq:
            if gid not in by_id or not by_id[gid]["cells"]:
                raise ValueError(f"policy {gid} does not exist or has no goal cells")
        N_total = np.zeros((nx, nx))
        reach = np.eye(nx)               # reach[x0, h] = P(about to start THIS leg at h | x0)
        pi_last = None
        for i, gid in enumerate(seq):
            grp = by_id[gid]
            is_last = (i == len(seq) - 1)
            N_i, pi_i, goalstates = leg_fundamental(grp)
            pi_last = pi_i
            contribution = N_i if is_last else N_i.copy()
            if not is_last:
                # A STATE-ONLY goal hands off in the SAME instant it is reached -- leg i+1 starts
                # AT that cell, at that same timestep, so ITS OWN identity term ((I-Q)^-1 always
                # counts the start state once) already counts this visit; adding leg i's own
                # column here too would count it twice. Verified by Monte Carlo simulation.
                #
                # A STATE-ACTION goal is different: reaching it and EXECUTING the terminal action
                # are two separate timesteps (handover_kernel's own D[x,:,1] = P[a,x,:] is a real
                # kernel step, not identity), landing leg i+1 somewhere else entirely in general --
                # so the visit to the goal cell itself is real and distinct from whatever leg i+1
                # goes on to count, and must stay in the total. Also Monte Carlo-verified: zeroing
                # it here undercounted the goal cell by exactly 1.0, the one visit no one else in
                # the sum was accounting for.
                state_only = [c["li"] for c in grp["cells"] if c["a"] is None]
                contribution[:, state_only] = 0.0
            N_total += reach @ contribution
            if not is_last:
                D = handover_kernel(grp, P)
                D2 = D[:, :, 0] + D[:, :, 1]           # collapse the (unneeded here) delay axis
                arrived = np.zeros((nx, nx))
                arrived[:, goalstates] = (reach @ N_i)[:, goalstates]
                reach = arrived @ D2
        return {"rows": rows, "cols": cols, "nx": int(nx), "algorithm": algo_id,
                "sequence": seq, "composed": True,
                "surfaces": [{"id": "∗", "empty": False, "composed": True, "sequence": seq,
                              "fundamental": [[round(float(v), 5) for v in row] for row in N_total],
                              "pi": [int(v) for v in pi_last]}]}

    out = []
    for grp in world["goals"]:
        if not grp["cells"]:
            out.append({"id": grp["id"], "empty": True})
            continue
        fixed = policy if (policy is not None and grp["id"] == (group_id or groups[0]["id"])) else None
        N, pi, _goalstates = leg_fundamental(grp, fixed)
        out.append({"id": grp["id"], "empty": False, "cells": [dict(c) for c in grp["cells"]],
                    "fundamental": [[round(float(v), 5) for v in row] for row in N],
                    "pi": [int(v) for v in pi]})
    return {"rows": rows, "cols": cols, "nx": int(nx), "algorithm": algo_id, "surfaces": out}


def rollout_string(world, source, algo_id, params, sequence, max_steps=600, seed=None,
                   handover=True):
    """Sample a trajectory that runs a STRING of policies back to back.

    A string is SEMI-MARKOV: something has to say when one option hands over to the next. The rule
    here is the one the composition already assumes -- an option runs until it TERMINATES, and in
    OKBE an option terminates on entering its goal region. eta(x_f, t_f | x) is precisely the
    first-passage distribution to that event, so convolving eta_1 * eta_2 integrates over exactly
    the intermediate state and time at which leg 1 finished. Switching on goal-region entry makes
    this rollout the sampling counterpart of the composed kernel, which test_tutorial_compose.py
    checks by comparing the empirical (final state, total time) against it.

    Entering a constraint ends the whole string, matching how eta- accumulates across legs: fail
    during leg 1, or complete leg 1 and fail during leg 2, and so on.
    """
    import numpy as np

    if world["start"] is None:
        raise ValueError("no start state placed")
    if not sequence:
        raise ValueError("empty policy string")
    rows, cols = world["rows"], world["cols"]
    by_id = {g["id"]: g for g in world["goals"]}
    for gid in sequence:
        if gid not in by_id or not by_id[gid]["cells"]:
            raise ValueError(f"policy {gid} does not exist or has no goal cells")

    sp, P = get_world_kernel(world)
    na, nx = P.shape[0], P.shape[1]
    rng = np.random.RandomState(None if seed is None else int(seed))
    con = set(world["constraints"])

    traj = [int(world["start"])]
    legs, outcome = [], "timeout"
    steps_left = int(max_steps)

    for gid in sequence:
        pi = get_policy(world, source, algo_id, params, gid)
        goal_set = {e["li"] for e in by_id[gid]["cells"]}
        t_leg, done = 0, False

        if traj[-1] in goal_set:              # already there: the leg terminates immediately
            done = True
        while not done and steps_left > 0:
            s0 = traj[-1]
            p = P[int(pi[s0]), s0, :].astype(float)
            tot = p.sum()
            if tot <= 0:
                outcome = "stuck"
                break
            nxt = int(rng.choice(nx, p=p / tot))
            traj.append(nxt); t_leg += 1; steps_left -= 1
            if nxt in con:
                outcome = "violated"
                break
            if nxt in goal_set:
                done = True
        # HANDOVER: if this leg's goal names a terminal action, executing it is part of finishing
        # the option -- reaching the tree is not drinking from it. That costs a step and applies
        # whatever the action does (interact is a self-loop, so it dwells in place). This is the
        # sampling counterpart of the D kernel in compose_sequence, so the two agree.
        acted = None
        if done and handover:
            acts = {e["li"]: e["a"] for e in by_id[gid]["cells"] if e["a"] is not None}
            a_g = acts.get(traj[-1])
            if a_g is not None and steps_left > 0:
                p = P[int(a_g), traj[-1], :].astype(float)
                tot = p.sum()
                if tot > 0:
                    traj.append(int(rng.choice(nx, p=p / tot)))
                    t_leg += 1; steps_left -= 1
                    acted = int(a_g)

        legs.append({"group": gid, "steps": t_leg, "ended": len(traj) - 1,
                     "done": bool(done), "terminal_action": acted})
        if not done:
            break                              # violated, stuck, or out of steps: the string ends
        if gid == sequence[-1]:
            outcome = "goal"

    return {"trajectory": traj, "outcome": outcome, "steps": len(traj) - 1,
            "legs": legs, "sequence": list(sequence),
            "source": source, "algorithm": algo_id, "resolved": True}


def rollout_world(world, source, algo_id, params, group_id=None, max_steps=300, seed=None,
                  policy=None):
    """Sample ONE trajectory under a policy, from the same kernel the solver used.

    `policy` is the pi the browser already holds from its last solve -- passing it makes this a
    pure forward sample with NO solving. The client only sends it when its displayed result still
    matches the current world; otherwise it is omitted and we solve here to get one.

    Actions are executed against P by SAMPLING, not by taking the most likely successor, so with
    kernel noise > 0 repeated runs differ and can slip into constraints the policy avoids.
    """
    import numpy as np

    rows, cols = world["rows"], world["cols"]
    if world["start"] is None:
        raise ValueError("no start state placed")
    groups = [g for g in world["goals"] if g["cells"]]
    if not groups:
        raise ValueError("no goal cells placed")
    grp = next((g for g in groups if g["id"] == group_id), groups[0])

    nx_expected = rows * cols
    kappa_at_start, solved = None, False
    if policy is not None and len(policy) == nx_expected:
        pi = np.asarray(policy, dtype=int)      # reuse what the panel already computed
    else:
        pi = get_policy(world, source, algo_id, params, grp["id"])
        solved = True

    sp, P = get_world_kernel(world)
    nx = P.shape[1]

    rng = np.random.RandomState(None if seed is None else int(seed))
    goal_set = {e["li"] for e in grp["cells"]}
    con_set = set(world["constraints"])

    s0 = int(world["start"])
    traj, acts = [s0], []
    outcome = "timeout"
    if s0 in goal_set:
        outcome = "goal"
    else:
        for _ in range(int(max_steps)):
            a = int(pi[traj[-1]])
            p = P[a, traj[-1], :].astype(float)
            tot = p.sum()
            if tot <= 0:                       # a dead state with no outgoing mass
                outcome = "stuck"
                break
            nxt = int(rng.choice(nx, p=p / tot))
            acts.append(a)
            traj.append(nxt)
            if nxt in goal_set:
                outcome = "goal"
                break
            if nxt in con_set:
                outcome = "violated"           # entering a constraint ends the run: that IS failure
                break
    return {"trajectory": traj, "actions": acts, "outcome": outcome,
            "steps": len(traj) - 1, "group": grp["id"],
            "source": source, "algorithm": algo_id,
            "resolved": solved,                 # False = pure forward sample, no solve
            "kappa_at_start": kappa_at_start}


def unroll_sequence(world, algo_id, params, groups, x0, max_steps=None):
    """Tick-by-tick forward simulation of an N-policy STRING, kept in ONE continuous spatial
    distribution over the whole grid, plus the success/failure mass absorbed at each tick.

    This is the SAME identity feas_two_phase_sparse already uses, run once per leg and stitched
    together:
        eta_pos(t|x) as a RECURRENCE: split off (goal, constraint, neither) at t, propagate
        "neither" through P_pi, repeat.
    Leg k runs this against its OWN goal (the handoff cell(s) into leg k+1, or the chain's final
    goal on the last leg); leg k+1 runs the SAME recurrence but SOURCE-FED, by whatever leg k
    hands it, tick by tick, instead of a single x0. Each handoff is exactly the D-kernel logic
    feas_two_phase/compose_sequence already use: a state-only goal cell hands its mass over at the
    SAME tick; a state-action goal cell's mass takes ONE MORE step under the terminal action
    (which, per kernel_row, can be stochastic -- spread over several destinations) before it is
    available, i.e. one tick later. N=2 reproduces the original two-leg version exactly (leg 0's
    goal is a pure handoff, leg 1's goal is the chain's only success condition).

    Returns, per tick: the FULL occupancy vector over every state (still-alive mass, summed across
    every leg -- the thing to animate on the grid) and the newly-absorbed success/violation mass
    (the thing to bar-chart). Also returns the server's own composed marginal for the same string,
    as a ground-truth overlay -- test_unroll_sequence.py checks the two agree exactly.
    """
    import numpy as np

    rows, cols = world["rows"], world["cols"]
    nx = rows * cols
    if x0 is None:
        raise ValueError("no start state placed -- place one (S) before unrolling a string")
    x0 = int(x0)
    if not groups or any(g is None for g in groups):
        raise ValueError("no policy selected -- pick one (or a saved string) before unrolling")
    groups = [int(g) for g in groups]
    group_map = {g["id"]: g for g in world["goals"] if g["cells"]}
    missing = [g for g in groups if g not in group_map]
    if missing:
        raise ValueError(f"policy {missing[0]} does not exist or has no goal cells")
    gdefs = [group_map[g] for g in groups]
    N = len(gdefs)

    import scipy.sparse as sps

    pis = [get_policy(world, "feas", algo_id, params, g) for g in groups]
    _sp, P = get_world_kernel(world)
    x_inds = np.arange(nx)
    P_pis = [sps.csr_matrix(P[pi, x_inds, :]) for pi in pis]     # [nx, nx] each, row = from, col = to

    cons_mask = np.zeros(nx); cons_mask[list(world["constraints"])] = 1.0
    goal_masks = []
    for gdef in gdefs:
        m = np.zeros(nx)
        for e in gdef["cells"]:
            m[int(e["li"])] = 1.0
        goal_masks.append(m)

    # A THIRD way an option can end, per the paper's own eta^- boundary condition (Eq 11):
    # entering a state kappa*(x)=0 -- feasible only in the past tense, no path anywhere left to the
    # goal -- terminates immediately, same as a constraint hit but without ever touching one. Left
    # out, that mass would just wander the kappa=0 pocket forever under "neither", silently
    # inflating occupancy at the far end of the horizon instead of being counted as failure the
    # moment it becomes hopeless.
    kappas = [get_kappa(world, "feas", algo_id, params, g) for g in groups]
    dead_masks = [((kappas[k] < 1e-6) & (goal_masks[k] == 0) & (cons_mask == 0)).astype(float)
                  for k in range(N)]
    neithers = [1.0 - goal_masks[k] - cons_mask - dead_masks[k] for k in range(N)]

    # Per handoff cell, whether it hands over at the SAME tick (a is None) or ONE tick later,
    # through the terminal action's own (possibly stochastic) transition row. Only legs 0..N-2 hand
    # off (into the next leg); the last leg's goal is the chain's own success condition, not a
    # handoff, so it gets no entry here.
    same_tick_masks = [np.zeros(nx) for _ in range(N - 1)]
    delayed_rows_list = [[] for _ in range(N - 1)]
    for k in range(N - 1):
        for e in gdefs[k]["cells"]:
            li_, a = int(e["li"]), e.get("a")
            if a is None:
                same_tick_masks[k][li_] = 1.0
            else:
                delayed_rows_list[k].append((li_, P[int(a), li_, :]))

    chain_key = "chain:" + ",".join(str(g) for g in groups)
    # This same call also IS the ground-truth ("truth_pos"/"truth_neg") overlay returned below --
    # stok_time_slice does not cache itself, so calling it a second time with the identical
    # (world, algo_id, params, chain_key, x0) at the end recomputed the whole N-leg composition
    # from scratch for no reason. Measured at N=12 legs: ~130ms of a ~370ms request, i.e. over a
    # third of the total, for work already sitting in `truth` below. Kept lazy (only called when
    # actually needed) so a caller-supplied max_steps -- which skips the horizon-sizing use -- does
    # not pay for a call it never asked for.
    truth = None
    if max_steps is None:
        # Size the horizon from the composed string's OWN, so a slow string is not truncated.
        truth = stok_time_slice(world, algo_id, params, chain_key, x0)
        max_steps = max(30, int(truth.get("T_axis", 30)) + 20)
    T = int(max_steps)

    # ---- ONE BLOCK-SPARSE OPERATOR for the whole string ----------------------------------------
    # An earlier version of this function kept N separate nx-length vectors and moved mass between
    # them with a hand-written per-leg Python loop every tick (masks, a `pending` buffer for the
    # delayed-handoff case). That is exactly the same LINEAR MAP as one (N*nx, N*nx) block-sparse
    # matrix Q applied once per tick -- the loop was just that operator spelled out by hand, one
    # small matvec and a few small elementwise multiplies per leg instead of one big one:
    #   diagonal block (k,k)     = diag(neither_k) @ P_pi_k
    #       leg k's own still-alive mass, continuing under its own policy. neither_k already
    #       excludes every one of leg k's goal/constraint/dead cells, so absorbed mass never
    #       re-enters this block -- it leaves entirely through a super-diagonal block below, or
    #       (on the LAST leg) simply stops being tracked, which is what "the chain succeeded" is.
    #   super-diagonal block (k,k+1), SAME-TICK handoff cells:
    #       diag(same_tick_mask_k) @ P_pi_(k+1)
    #       the mass is relabelled into leg k+1 AND takes its first step under leg k+1's policy in
    #       this SAME matvec, matching "hands over at the same tick".
    #   super-diagonal block (k,k+1), DELAYED (state-action) handoff cells:
    #       just the raw terminal-action kernel row, P[a, li, :] -- no leg-(k+1) policy applied
    #       yet. That mass lands as leg (k+1)'s RESIDENT mass after this matvec, and only picks up
    #       P_pi_(k+1) on the NEXT application of Q (via block (k+1,k+1)) -- reproducing "one tick
    #       later" with no special-casing at all, because a single matvec step already IS one tick.
    # Every other block is zero: a leg can only hand off to the very next one. Built ONCE, outside
    # the loop; the tick loop below then does exactly one sparse matvec per tick, on the WHOLE
    # (N*nx)-length stacked state, instead of N small ones plus manual handoff bookkeeping.
    blocks = [[None] * N for _ in range(N)]
    for k in range(N):
        blocks[k][k] = sps.diags(neithers[k]) @ P_pis[k]
    for k in range(N - 1):
        same = sps.diags(same_tick_masks[k]) @ P_pis[k + 1]
        if delayed_rows_list[k]:
            rows_idx = [li_ for li_, _row in delayed_rows_list[k]]
            data_rows = np.array([row for _li_, row in delayed_rows_list[k]])
            delayed = sps.lil_matrix((nx, nx))
            delayed[rows_idx, :] = data_rows
            same = same + delayed.tocsr()
        blocks[k][k + 1] = same
    Q = sps.bmat(blocks, format="csr")

    v = np.zeros(N * nx)
    v[x0] = 1.0                                    # leg 0's block starts at offset 0
    cons_stacked = np.tile(cons_mask, N)
    dead_stacked = np.concatenate(dead_masks)
    succ_stacked = np.zeros(N * nx)
    succ_stacked[(N - 1) * nx:N * nx] = goal_masks[N - 1]   # only the LAST leg's goal is success

    # Per-tick, PER-CELL breakdown of newly-absorbed mass, one vector per outcome class -- the
    # surface plot's classified bars (green/red/yellow) are just this, accumulated over ticks
    # played so far. `success`/`failure` stay their existing GLOBAL scalars (summed here), so
    # test_unroll_sequence.py's exact match to the server's own composed marginal is untouched.
    occupancy, success, failure = [], [], []
    absorbed_goal, absorbed_violation, absorbed_dead = [], [], []
    for _t in range(T):
        # Reported occupancy/absorption are PER GRID CELL, not per leg -- multiple legs' blocks can
        # (and for a constraint, always do, since cons_stacked repeats the same mask in every
        # block) carry mass over the same physical cell, so this collapses the N stacked copies
        # back down to one nx-length reading, same contract as before.
        occupancy.append(v.reshape(N, nx).sum(axis=0))

        viol_t = v * cons_stacked
        dead_t = v * dead_stacked
        succ_t = v * succ_stacked

        success.append(float(succ_t.sum()))
        # failure is the FULL eta^-, not just constraint hits -- entering a kappa=0 pocket is also
        # a failure event by the paper's own boundary condition, just one without a constraint.
        failure.append(float(viol_t.sum() + dead_t.sum()))
        # RAW arrays only -- no rounding here. Rounding each of nx cells with Python's round() on
        # every one of T ticks (here, times 4 arrays: this and occupancy above) was the actual
        # bottleneck in the old per-leg loop -- rounding is a JSON-transport concern, not a
        # per-tick one; done ONCE below as a single vectorized np.round over the whole [T, nx]
        # array, not nx*T individual Python-level round() calls.
        absorbed_goal.append(succ_t.reshape(N, nx).sum(axis=0))
        absorbed_violation.append(viol_t.reshape(N, nx).sum(axis=0))
        absorbed_dead.append(dead_t.reshape(N, nx).sum(axis=0))
        v = v @ Q

    if truth is None:
        truth = stok_time_slice(world, algo_id, params, chain_key, x0)
    return {
        "rows": rows, "cols": cols, "nx": nx, "x0": x0, "T": T,
        "occupancy": np.round(np.asarray(occupancy), 6).tolist(),
        "success": [round(v, 6) for v in success],
        "failure": [round(v, 6) for v in failure],
        "absorbed_goal": np.round(np.asarray(absorbed_goal), 6).tolist(),
        "absorbed_violation": np.round(np.asarray(absorbed_violation), 6).tolist(),
        "absorbed_dead": np.round(np.asarray(absorbed_dead), 6).tolist(),
        "truth_pos": truth["marginal"], "truth_neg": truth["marginal_neg"],
    }


def kernel_row(world, x, a):
    """P(x'|x,a) for one state and action -- a pure slice of the already-instantiated kernel, no
    new math. Exists for the sliding-convolution prototype: a terminal action's one-step evolution
    (the handover kernel D uses this exact row) can be stochastic, spreading over several
    destinations, so a caller needs the real distribution rather than assuming a single landing
    cell."""
    import numpy as np
    rows, cols = world["rows"], world["cols"]
    nx = rows * cols
    x, a = int(x), int(a)
    _sp, P = get_world_kernel(world)
    if not (0 <= x < nx and 0 <= a < P.shape[0]):
        raise ValueError(f"state {x} or action {a} out of range")
    row = P[a, x, :]
    nz = np.nonzero(row)[0]
    return {"x": x, "a": a, "to": [int(v) for v in nz], "p": [round(float(row[v]), 6) for v in nz]}


def tree_children(world, state_vec):
    """One child per BL action, from a full {space_name: value} state vector -- for the Inspect
    panel's tree view / IDDFS. X evolves by the MOST-LIKELY successor under P(x'|x,a) (argmax of
    kernel_row's distribution): a deliberate simplification so the tree has one child per action
    rather than one per possible stochastic outcome, with the taken probability reported alongside
    so a low-probability "most likely" move is never silently read as certain. Every other space's
    value carries through UNCHANGED -- no affordance wires a BL action to an HL transition yet,
    same caveat as inspect_panel_data (this tree does not contradict that endpoint, it shares it)."""
    import numpy as np
    rows, cols = world["rows"], world["cols"]
    nx = rows * cols
    x = int(state_vec.get("X", 0))
    if not (0 <= x < nx):
        raise ValueError(f"state vector's X={x} out of range for {rows}x{cols}")
    _sp, P = get_world_kernel(world)
    constraints = {int(c) for c in world["constraints"]}
    goal_cells = {int(c["li"] if isinstance(c, dict) else c)
                  for g in world["goals"] for c in g["cells"]}
    out = []
    for a in range(P.shape[0]):
        row = P[a, x, :]
        x2 = int(np.argmax(row))
        p = float(row[x2])
        child = dict(state_vec)
        child["X"] = x2
        out.append({
            "action": a, "label": ACTION_LABELS[a] if a < len(ACTION_LABELS) else str(a),
            "state": child, "p": round(p, 4),
            "is_goal": x2 in goal_cells, "is_constraint": x2 in constraints,
        })
    return out


def empirical_time_slice(world, algo_id, params, group_id, x0, samples=300, max_steps=300):
    """Sample the SAME policy `samples` times from x0 and bucket outcomes by elapsed step, so a
    theoretical eta(t|x0) can be checked against what actually happens when you run it.

    Reuses get_policy/get_kernel -- the solve happens ONCE, not once per sample -- and matches
    rollout_world's own outcome semantics exactly (goal / violated / stuck / timeout), so this is
    counting the same events a single rollout reports, just many of them at once.

    Three buckets, matching the plot's existing eta+ / eta- palette plus one more:
        pos         -- reached the goal                                   (green)
        neg         -- entered a constraint                               (red)
        infeasible  -- "stuck" (no outgoing mass: a genuinely dead state) or "timeout" (still
                       going at max_steps). Both mean "neither succeeded nor violated," matching
                       kappa_none's definition -- this is its empirical counterpart. Timeouts are
                       stacked at bin max_steps rather than dropped, so the empirical mass still
                       sums to 1 over `samples` even when a slow tail gets cut off by the cap.
    """
    import numpy as np

    rows, cols = world["rows"], world["cols"]
    if world["start"] is None:
        raise ValueError("no start state placed")
    groups = [g for g in world["goals"] if g["cells"]]
    grp = next((g for g in groups if g["id"] == group_id), None)
    if grp is None:
        raise ValueError(f"policy {group_id} does not exist or has no goal cells")

    pi = get_policy(world, "feas", algo_id, params, grp["id"])
    sp, P = get_world_kernel(world)
    nx = P.shape[1]
    goal_set = {e["li"] for e in grp["cells"]}
    con_set = set(world["constraints"])
    x0 = int(x0)

    rng = np.random.RandomState()
    max_steps = int(max_steps)
    pos = np.zeros(max_steps + 1)
    neg = np.zeros(max_steps + 1)
    infeasible = np.zeros(max_steps + 1)
    counts = {"goal": 0, "violated": 0, "stuck": 0, "timeout": 0}

    for _ in range(int(samples)):
        x = x0
        if x in goal_set:
            pos[0] += 1; counts["goal"] += 1
            continue
        t = 0
        outcome = "timeout"
        for t in range(1, max_steps + 1):
            a = int(pi[x])
            p = P[a, x, :].astype(float)
            tot = p.sum()
            if tot <= 0:
                outcome = "stuck"
                break
            x = int(rng.choice(nx, p=p / tot))
            if x in goal_set:
                outcome = "goal"
                break
            if x in con_set:
                outcome = "violated"
                break
        counts[outcome] += 1
        if outcome == "goal":
            pos[t] += 1
        elif outcome == "violated":
            neg[t] += 1
        else:                                  # stuck or timeout: neither success nor violation
            infeasible[max_steps if outcome == "timeout" else t] += 1

    # Trim the shared all-zero tail, same as the theoretical stok_time_slice, so the browser is
    # not sent a long run of zeros.
    nz = np.nonzero((pos + neg + infeasible) > 0)[0]
    tmax = int(nz[-1]) + 1 if len(nz) else 1
    n = float(samples)
    return {
        "samples": int(samples), "max_steps": max_steps, "x0": x0,
        "pos": [round(float(v) / n, 6) for v in pos[:tmax]],
        "neg": [round(float(v) / n, 6) for v in neg[:tmax]],
        "infeasible": [round(float(v) / n, 6) for v in infeasible[:tmax]],
        "mass_pos": round(counts["goal"] / n, 6),
        "mass_neg": round(counts["violated"] / n, 6),
        "mass_infeasible": round((counts["stuck"] + counts["timeout"]) / n, 6),
        "counts": counts,
    }


def solve_world(world, algo_id, params, want_stok=False, group_id=None,
                stok_mode="full", x0=None):
    """Run one algorithm over a world, once PER GOAL GROUP. Each group is its own goal function g
    (its cells OR'd), so N groups means N solves and N kappa surfaces -- not one solve with N
    targets. Returns a surface per group."""
    import numpy as np
    import TGMDP_methods as tg
    import feas_two_phase as f2p
    import feas_two_phase_sparse as f2ps
    import feas_thresh_sparse as fts
    import feas_eta_fast as fef
    import eta_combined as fec
    import state_spaces as ss
    import warnings

    algo = ALGO_BY_ID.get(algo_id)
    if algo is None:
        raise ValueError(f"unknown algorithm: {algo_id}")

    rows, cols = world["rows"], world["cols"]
    nx_cells = rows * cols
    if stok_mode not in ("full", "single_x0", "packed_rows", "cache_only"):
        raise ValueError("unknown STOK response mode")
    if want_stok and stok_mode == "single_x0":
        x0 = world.get("start") if x0 is None else x0
        x0 = 0 if x0 is None else int(x0)
        if not 0 <= x0 < nx_cells:
            raise ValueError("x0 out of range")
    # Timed in PARTS. The endpoint's wall clock covers the kernel build, every group's solve and
    # the JSON assembly together, so one number cannot answer "how long did the policy take" --
    # which is the only one of the three anybody wants to know.
    _t_kernel = time.perf_counter()
    sp, P = get_world_kernel(world)
    # get_kernel's own `sp` is always None now (see its docstring) -- only the DENSE solvers below
    # (feas / feas_thresh) read xspace at all, and only they pay for building a real one, so
    # picking a sparse algorithm (the default, and the only one that scales past a small grid)
    # never pays this cost. Rare enough, and already dwarfed by the dense solve itself (measured
    # ~14s at just 15x15 for feas_thresh's own interleaved loop), that it is not cached.
    if algo_id in ("feas", "feas_thresh"):
        import state_spaces as ss
        wall = np.zeros((rows, cols))
        for li in world["walls"]:
            wall[li // cols, li % cols] = 1
        sp = ss.StateSpace_2D(rows, cols, wall, [wall], [1], name="X",
                              p_main=float(world.get("p_main", 1.0)))
    timing = {"kernel": round(time.perf_counter() - _t_kernel, 4), "solve": 0.0, "groups": 0}
    na, nx = P.shape[0], P.shape[1]
    if nx != nx_cells:
        raise ValueError(f"kernel has {nx} states for a {rows}x{cols} grid")
    # slope_for's own kernel-weighted expectation, sparse: P has ~5 nonzero entries per row (one
    # per action's destination), but is stored dense (na*nx*nx floats) because the solvers need
    # it that way -- np.einsum('axy,y->ax', P, v) then costs na*nx*nx multiplies to do work that
    # only needs na*nx*5. Immaterial at 9x9; ~10.5s (3 calls, one per kappa/kappa_viol/kappa_none)
    # at 40x40, which is most of what made the whole solve feel slow there. Built once per solve
    # and reused across every group AND every slope_for call within a group.
    P_sparse = f2ps.kernel_to_sparse(P)

    # Constraints are hard: entering one is a violation. With p_main < 1 the agent can SLIP into a
    # constraint it did not aim for, which is what makes kappa graded rather than 0/1 -- and what
    # gives the risk threshold something to trade off.
    c = np.ones([na, nx])
    for li in world["constraints"]:
        c[:, li] = 0.0

    out = []
    for grp in world["goals"]:
        if not grp["cells"]:
            out.append({"id": grp["id"], "cells": [], "empty": True})
            continue
        g = np.zeros([na, nx])
        for e in grp["cells"]:
            if e["a"] is None:
                g[:, e["li"]] = 1.0         # state-only: any action there terminates
            else:
                g[e["a"], e["li"]] = 1.0    # state-ACTION: only that action terminates
        phases = {}
        _t_solve = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if algo_id == "feas":
                kappa, pi, ep, en, beta = tg.feas_iter_stationary_flat_dense(
                    P, g, c, use_ET=True, xspace=sp)
                kappa_c = en.sum(axis=(1, 2))
            elif algo_id == "feas2":
                # round_decimals/kappa_tol default to 0 (OFF, exact fixed point) -- see
                # _TWO_PHASE_PARAMS for why a nonzero default was tried and reverted: it can make a
                # state-action goal's policy pick a non-terminating action, so the option never
                # completes at all. Both stay reachable as sliders for a world checked to be safe.
                kappa_tol = float(params.get("kappa_tol", 0.0))
                rd = int(params.get("round_decimals", 0))
                rd = None if rd <= 0 or rd >= ROUND_DECIMALS_MAX else rd
                kappa_eps = float(params.get("kappa_epsilon", 1e-9))
                ET_eps = float(params.get("ET_epsilon", 1e-9))
                kappa, pi, ep, en, beta = f2p.feas_iter_two_phase(
                    P, g, c, round_decimals=rd, kappa_tol=kappa_tol,
                    kappa_epsilon=kappa_eps, ET_epsilon=ET_eps, timing=phases)
                kappa_c = en.sum(axis=(1, 2))
            elif algo_id in ("feas2s", "feas_eta_fast", "feas_eta_combined"):
                # ep/en come back as TerminalEta, which implements the sum(axis=...) and [x0]
                # reductions used below and by stok_time_slice -- so nothing downstream has to
                # densify. /api/compose is the one consumer that needs a real array, and it asks
                # for one there.
                kappa_tol = float(params.get("kappa_tol", 0.0))
                rd = int(params.get("round_decimals", 0))
                rd = None if rd <= 0 or rd >= ROUND_DECIMALS_MAX else rd
                kappa_eps = float(params.get("kappa_epsilon", 1e-9))
                ET_eps = float(params.get("ET_epsilon", 1e-9))
                solver = {"feas2s": f2ps.feas_iter_two_phase_sparse,
                          "feas_eta_fast": fef.feas_iter_fast_eta,
                          "feas_eta_combined": fec.feas_iter_combined_eta}[algo_id]
                extra = {"eta_mode": int(params.get("eta_mode", 0))} if algo_id != "feas2s" else {}
                kappa, pi, ep, en, beta = solver(
                    P, g, c, **extra, round_decimals=rd, kappa_tol=kappa_tol,
                    kappa_epsilon=kappa_eps, ET_epsilon=ET_eps, timing=phases)
                kappa_c = en.sum(axis=(1, 2))
            elif algo_id in ("feas_thresh_s", "feas_eta_fast_thresh", "feas_eta_combined_thresh"):
                # Sparse, two-phase counterpart to "feas_thresh" below -- one call does both
                # passes (see feas_thresh_sparse's own module docstring for why pass 1 uses the
                # FIXED two-phase solver, not the dense original's interleaved one, to compute
                # kappa_c). ep/en come back as TerminalEta, same as feas2s -- no separate
                # densifying path needed anywhere downstream.
                rd = int(params.get("round_decimals", 0))
                rd = None if rd <= 0 or rd >= ROUND_DECIMALS_MAX else rd
                theta = float(params.get("risk_thresh", 0.5))
                kappa_eps = float(params.get("kappa_epsilon", 1e-9))
                ET_eps = float(params.get("ET_epsilon", 1e-9))
                solver = {"feas_thresh_s": fts.feas_iter_two_phase_sparse_thresh,
                          "feas_eta_fast_thresh": fef.feas_iter_fast_eta_thresh,
                          "feas_eta_combined_thresh": fec.feas_iter_combined_eta_thresh}[algo_id]
                extra = {"eta_mode": int(params.get("eta_mode", 0))} if algo_id != "feas_thresh_s" else {}
                kappa, pi, ep, en, beta = solver(
                    P, g, c, **extra, risk_thresh=theta, round_decimals=rd,
                    kappa_epsilon=kappa_eps, ET_epsilon=ET_eps, timing=phases)
                kappa_c = en.sum(axis=(1, 2))
            else:
                # pass 1 supplies kappa_c, pass 2 applies the threshold
                _k, _pi, _ep, en0, _b = tg.feas_iter_stationary_flat_dense(
                    P, g, c, use_ET=True, xspace=sp)
                kappa_c = en0.sum(axis=(1, 2))
                theta = float(params.get("risk_thresh", 0.5))
                # NB use_ET=True is REQUIRED: the use_ET=False branch of the thresholded solver is
                # warnings.warn("Not implemented") and silently returns nothing useful.
                # NB it returns FOUR values, not five -- no beta, unlike the plain solver.
                kappa, pi, ep, en = tg.feas_iter_stationary_flat_dense_thresh(
                    P, P, g, c, kappa_c, risk_thresh=theta, use_ET=True, xspace=sp)
        timing["solve"] += time.perf_counter() - _t_solve
        if phases:
            for phase, seconds in phases.items():
                timing[phase] = timing.get(phase, 0.0) + seconds
        timing["groups"] += 1
        # THE THREE CUMULATIVE OUTCOMES, as a partition of the mass leaving each state.
        #
        #   kappa      = kappa^+, cumulative SUCCESS   (the solver's own kappa)
        #   kappa_viol = cumulative CONSTRAINT VIOLATION
        #                  = sum_{x_f,t_f} eta^-(x_f,t_f|x) * (1 - c(x_f, pi(x_f)))
        #   kappa_none = 1 - kappa_viol - kappa^+, i.e. everything that neither succeeded nor
        #                violated
        #
        # kappa_viol is NOT eta^-.sum(): eta^- carries two kinds of failure. Its boundary condition
        # is  feas(kappa)*(1 - c[pi,x]) + infeas(kappa), so it also holds mass absorbed in states
        # that are simply infeasible, with no constraint involved. Weighting each final state by
        # (1 - c) keeps only the terminations that were actually constraint violations, which is
        # what the raw sum in `kappa_c` (kept below, and fed to the thresholded solver, which
        # expects that definition) conflates.
        #
        # c is [na, nx]. Conditioning on pi(x_f) is what the state-ACTION case needs; when c is a
        # function of state alone every action gives the same number, so this is also correct for
        # state-only constraints.
        viol_w = 1.0 - c[pi, np.arange(nx)]                     # [nx], per FINAL state
        kappa_viol = (en.sum(axis=2) * viol_w[np.newaxis, :]).sum(axis=1)
        kappa_none = 1.0 - kappa_viol - kappa
        # kappa_none goes very slightly negative -- order 5e-5 -- in worlds where kappa saturates
        # at 1. That is the solver's own quantization: it does np.round(kappa_w_actions_ax, 4)
        # every iteration, so kappa carries error at the 1e-4 level and the subtraction inherits
        # it. Clamp, but only at that scale: anything bigger is a real inconsistency and should
        # surface as a log line rather than be silently flattened to zero.
        worst = float(kappa_none.min())
        if worst < -1e-3:
            log("kappa_partition_negative", worst=round(worst, 6), group=grp["id"])
        kappa_none = np.clip(kappa_none, 0.0, 1.0)

        # SLOPE, redefined as an EXPECTATION under the real kernel rather than a spatial finite
        # difference. The old version differenced a cumulative surface across GRID neighbours
        # (north/south/east/west cells), which silently included neighbours the agent cannot
        # actually reach from x -- a wall or mountain one cell over still has a kappa value (0),
        # and the difference used it as if stepping there were possible. That distorted the plot
        # near any obstacle: it eyballed a "downhill" wherever a blocked cell happened to be
        # lower, whether or not any action leads there.
        #
        # Correct one-step reading: Q_a(x) = sum_x' P(x'|x,a) * V(x') is the actual expected value
        # of taking action a from x, weighted by the kernel -- an inaccessible cell simply gets
        # ~0 weight, exactly as it should. slope(x) = Q_{pi(x)}(x) - mean_a Q_a(x): the optimal
        # action's one-step expectation minus a RANDOM policy's (uniform over actions), so a
        # positive slope means pi does meaningfully better than picking blindly, zero means pi is
        # no better than random (e.g. every action leads to the same place, as at a self-loop or a
        # cell boxed in on all reachable sides), and it can never be negative since pi is optimal.
        x_inds = np.arange(nx)

        def slope_for(v):
            v = np.asarray(v, dtype=float)
            # Same value as np.einsum('axy,y->ax', P, v) -- one sparse matvec per action instead
            # of a dense contraction over the whole [na,nx,nx] array (see P_sparse's own comment).
            q_all = np.stack([P_sparse[a] @ v for a in range(na)])   # [na, nx]
            q_pi = q_all[pi, x_inds]
            q_rand = q_all.mean(axis=0)
            return [round(float(s), 6) for s in (q_pi - q_rand)]

        entry = {
            "id": grp["id"], "cells": [dict(c) for c in grp["cells"]], "empty": False,
            "kappa": [round(float(v), 6) for v in kappa],
            "kappa_c": [round(float(v), 6) for v in kappa_c],
            "kappa_viol": [round(float(v), 6) for v in kappa_viol],
            "kappa_none": [round(float(v), 6) for v in kappa_none],
            "slope_pos": slope_for(kappa),
            "slope_viol": slope_for(kappa_viol),
            "slope_none": slope_for(kappa_none),
            "pi": [int(v) for v in pi],
            "start": None if world["start"] is None else round(float(kappa[world["start"]]), 6),
        }
        if want_stok:
            # BOTH halves. eta_pos is successful termination, eta_neg is termination in constraint
            # violation; eta = eta_pos + eta_neg is the full first-passage distribution. Shipping
            # only eta_pos hides every unsuccessful outcome, which is most of what a risk-sensitive
            # reading of a world is about.
            k = _eta_key(world, algo_id, params, grp["id"])
            with _eta_lock:
                _ETA[k] = (ep, en)
                # De-duplicate before appending: re-solving the same world used to append the key
                # again, so evicting "the oldest" threw away entries that had just been REFRESHED.
                # With N groups and a cap of N, a second solve emptied the cache completely.
                if k in _ETA_ORDER:
                    _ETA_ORDER.remove(k)
                _ETA_ORDER.append(k)
                while len(_ETA_ORDER) > _ETA_MAX:
                    _ETA.pop(_ETA_ORDER.pop(0), None)
            # eta_pos is indexed [x, x_f, t_f]. Summing out FINAL TIME leaves eta(x_f | x): where
            # the option lands, having started at x. Row x0 of this matrix is the STOK surface for
            # a chosen initial state -- shipped whole so the browser can change x0 without a
            # round trip. Summing the row again recovers kappa[x0], which the code's own
            # consistency check relies on.
            if stok_mode == "packed_rows":
                # Send all time-marginalized rows without the millions of zero entries.
                # The browser can then change x0 immediately, with no network round trip.
                from scipy.sparse import csr_matrix
                def pack_rows(eta):
                    matrix = csr_matrix(np.round(eta.sum(axis=2), 5))
                    matrix.eliminate_zeros()
                    return {"indptr": matrix.indptr.tolist(), "indices": matrix.indices.tolist(),
                            "data": matrix.data.tolist(), "nx": nx}
                entry.update(stok_rows=pack_rows(ep), stok_neg_rows=pack_rows(en))
            elif stok_mode == "single_x0":
                entry.update(stok_mode="single_x0", stok_cached=True, x0=x0,
                             stok=np.round(ep[x0].sum(axis=1), 5).tolist(),
                             stok_neg=np.round(en[x0].sum(axis=1), 5).tolist())
            elif stok_mode == "full":
                entry["stok"] = [[round(float(v), 5) for v in row] for row in ep.sum(axis=2)]
                entry["stok_neg"] = [[round(float(v), 5) for v in row] for row in en.sum(axis=2)]
        out.append(entry)
    timing["solve"] = round(timing["solve"], 4)
    return {"algorithm": algo_id, "params": params, "rows": rows, "cols": cols,
            "na": int(na), "nx": int(nx), "surfaces": out, "timing": timing}


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files out of tutorial/static, plus a small JSON API under /api/."""

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=STATIC, **kw)

    def _timed(self, label, fn):
        """Run a handler, logging how long it took and anything it raised."""
        t0 = time.time()
        try:
            out = fn()
            log("api", path=label, ms=round((time.time() - t0) * 1000))
            return out
        except Exception as e:
            import traceback
            log("api_error", path=label, ms=round((time.time() - t0) * 1000),
                error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
            raise

    def _api_404(self, path):
        """A 404 under /api/ must be JSON. The default handler returns an HTML error page, which
        the browser then tries to JSON.parse -- surfacing as "Unexpected token '<'" with no clue
        that the real problem is an endpoint this server does not have."""
        return self._json({"ok": False, "error": f"unknown endpoint {path}",
                           "api_version": API_VERSION,
                           "hint": "the server may be running older code than the page; restart it"},
                          404)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The page is regenerated constantly while developing; caching only causes confusion.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            return self.send_error(404, "File not found")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Every static file this server sends is a JS/CSS/HTML source file that changes constantly
        # during development. Without this, a PLAIN reload (not a hard reload) could silently keep
        # serving a browser-cached copy from before the last edit -- so "reload and try again"
        # sometimes tested old code without anyone knowing it. no-store makes an ordinary reload
        # always current; a hard reload was never actually required for this server's own files.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def handle_one_request(self):
        """Wrap the per-request handler so an unexpected exception is LOGGED and the connection is
        closed cleanly. Previously a bug in a handler (e.g. the log_message TypeError) surfaced in
        the browser as an EMPTY REPLY / "failed to fetch", with nothing recorded anywhere."""
        try:
            super().handle_one_request()
        except Exception as e:
            import traceback
            log("server_crash", path=getattr(self, "path", "?"),
                error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
            try:
                self.close_connection = True
            except Exception:
                pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/dbn/progress":
            request_id = self.path.partition('?')[2]
            with _DBN_PROGRESS_LOCK:
                stage = _DBN_PROGRESS.get(request_id)
            return self._json({"stage": stage})

        if path == "/api/logs":
            # tail the log so it can be read back without leaving the browser
            try:
                with open(LOGFILE) as fh:
                    lines = fh.readlines()[-500:]
                return self._json({"ok": True, "lines": [l.strip() for l in lines]})
            except FileNotFoundError:
                return self._json({"ok": True, "lines": []})

        if path == "/api/algorithms":
            return self._json({"algorithms": ALGORITHMS,
                               "value_algorithms": VALUE_ALGORITHMS})

        if path == "/api/icons":
            return self._json({"tools": TOOLS, "feature_types": FEATURE_TYPES, "item_types": ITEM_TYPES})

        if path == "/api/levels":
            return self._json({"ok": True, "levels": list_levels()})

        if path.startswith("/img/"):
            # Serve editor art. basename() alone contains the path: no directory component can
            # survive it, so ../.. cannot climb out of IMAGES.
            name = os.path.basename(path[len("/img/"):])
            if not name.lower().endswith(".png"):
                return self.send_error(404, "File not found")
            return self._send_file(os.path.join(IMAGES, name), "image/png")

        if self.path.split("?")[0] == "/api/health":
            return self._json({
                "ok": True,
                "api_version": API_VERSION,
                "endpoints": sorted(["/api/health", "/api/algorithms", "/api/icons", "/api/logs",
                                     "/api/world", "/api/solve", "/api/value", "/api/rollout",
                                     "/api/sr", "/api/fundamental", "/api/stok_time", "/api/empirical_time",
                                     "/api/compose", "/api/log", "/api/inspect", "/api/hl_panel",
                                     "/api/task_panel", "/api/inspect_panel", "/api/tree_children",
                                     "/api/levels", "/api/dbn",
                                     "/api/levels/save", "/api/levels/load",
                                     "/api/levels/delete"]),
                "stage": 6,
                "solver": SOLVER,
                "python": sys.version.split()[0],
            })
        # Anything still under /api/ is an endpoint this server does not have. Answer in JSON:
        # the default HTML error page is what makes a stale server look like "Unexpected token '<'".
        if path.startswith("/api/"):
            return self._api_404(path)

        return super().do_GET()

    def do_POST(self):
        # These read-only panels only build metadata. They must not queue behind
        # policy/STOK/SR computations when the user places their first feature.
        if self.path.split("?")[0] in ("/api/hl_panel", "/api/task_panel", "/api/h-preview"):
            return self._post_serial()
        with _compute_lock:
            return self._post_serial()

    def _post_serial(self):
        path = self.path.split("?")[0]
        if path == "/api/log":
            # the browser reporting its own errors; accept and store, never argue with it
            try:
                nbytes = int(self.headers.get("Content-Length", 0))
                rec = json.loads(self.rfile.read(nbytes) or b"{}")
            except Exception as e:
                rec = {"parse_error": str(e)}
            log("client", **rec)
            return self._json({"ok": True})

        if path not in ("/api/world", "/api/solve", "/api/value", "/api/rollout", "/api/sr",
                        "/api/fundamental",
                        "/api/stok_time", "/api/empirical_time", "/api/kernel_row", "/api/unroll_sequence",
                        "/api/compose", "/api/hl_panel", "/api/task_panel", "/api/inspect_panel", "/api/tree_children",
                        "/api/inspect", "/api/dbn", "/api/h-preview", "/api/levels/save", "/api/levels/load", "/api/levels/delete"):
            return self._api_404(path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json({"ok": False, "error": f"bad JSON: {e}"}, 400)

        if path == "/api/h-preview":
            try:
                from local_h_preview import propose
                return self._json({"ok": True, "result": propose(body)})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)

        if path == "/api/dbn":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            request_id = body.get('progress_id')
            if not isinstance(request_id, str) or len(request_id) > 100:
                request_id = None
            if request_id is not None:
                with _DBN_PROGRESS_LOCK:
                    _DBN_PROGRESS[request_id] = 'Preparing factors'
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed(path, lambda: dbn_panel_data(w, body))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}")
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)
            finally:
                with _DBN_PROGRESS_LOCK:
                    _DBN_PROGRESS.pop(request_id, None)

        if path.startswith("/api/levels/"):
            try:
                if path.endswith("/save"):
                    w = normalize_world(body.get("world", {}))
                    rec = save_level(w, body.get("name"), body.get("strings"), body.get("tutorial"), body.get("folder", ""))
                    log("level_saved", name=rec["name"])
                    return self._json({"ok": True, "name": rec["name"],
                                       "summary": rec["summary"], "levels": list_levels()})
                if path.endswith("/load"):
                    w, strings, tutorial = load_level(body.get("name"), with_tutorial=True)
                    return self._json({"ok": True, "world": w, "strings": strings, "tutorial": tutorial})
                delete_level(body.get("name"))
                return self._json({"ok": True, "levels": list_levels()})
            except Exception as e:
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}")
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/inspect":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed("/api/inspect", lambda: inspect_world(w, body))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/hl_panel":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed(path, lambda: hl_panel_data(w))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/task_panel":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = task_panel_data(w, w["task_clauses"])
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/inspect_panel":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed("/api/inspect_panel", lambda: inspect_panel_data(
                    w, str(body.get("space", "X")), int(body.get("s_i", 0)),
                    int(body.get("t_max", 20)), int(body.get("t_inspect", 5))))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/tree_children":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = tree_children(w, dict(body.get("state") or {}))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/compose":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                # sequence elements are policy ids OR blocks (a saved sub-string, sent as a
                # nested array -- see okbeChainBlocks in chain.js); coerce leaves to int while
                # preserving that nesting, since compose_sequence/compose_options tell a block
                # apart from a plain id by isinstance(..., (list, tuple)).
                def coerce_chain(seq):
                    return [coerce_chain(g) if isinstance(g, list) else int(g) for g in seq]
                res = self._timed("/api/compose", lambda: compose_options(
                    w, body.get("algorithm", "feas"), body.get("params") or {},
                    coerce_chain(body.get("sequence") or []), body.get("x0", 0),
                    bool(body.get("handover", True)),
                    body.get("stok_mode", "single_x0")))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/stok_time":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed("/api/stok_time", lambda: stok_time_slice(
                    w, body.get("algorithm", "feas"), body.get("params") or {},
                    body.get("group"), body.get("x0", 0)))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/kernel_row":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed("/api/kernel_row", lambda: kernel_row(
                    w, body.get("x", 0), body.get("a", 0)))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/unroll_sequence":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                # "groups" is the current, N-length interface; "group1"/"group2" is the original
                # two-leg one, kept accepted (not just tolerated -- actively supported) since
                # test_unroll_sequence.py's whole suite still calls it that way and there is no
                # reason to force a rewrite of tests that exercise exactly the N=2 case correctly.
                groups = body.get("groups")
                if groups is None:
                    groups = [g for g in (body.get("group1"), body.get("group2")) if g is not None]
                res = self._timed("/api/unroll_sequence", lambda: unroll_sequence(
                    w, body.get("algorithm", "feas"), body.get("params") or {},
                    groups, body.get("x0", 0),
                    max_steps=body.get("max_steps")))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/empirical_time":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed("/api/empirical_time", lambda: empirical_time_slice(
                    w, body.get("algorithm", "feas"), body.get("params") or {},
                    body.get("group"), body.get("x0", 0),
                    samples=int(body.get("samples", 300)),
                    max_steps=int(body.get("max_steps", 300))))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/sr":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed("/api/sr", lambda: sr_world(
                    w, body.get("source", "value"), body.get("algorithm", "ihdc"),
                    body.get("params") or {}, body.get("gamma", 0.95),
                    body.get("group"), body.get("policy"), body.get("sequence")))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/fundamental":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                res = self._timed("/api/fundamental", lambda: fundamental_world(
                    w, body.get("algorithm", "feas2s"), body.get("params") or {},
                    body.get("group"), body.get("policy"), body.get("sequence")))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/rollout":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                if body.get("sequence"):
                    res = self._timed("/api/rollout", lambda: rollout_string(
                        w, body.get("source", "feas"), body.get("algorithm", "feas"),
                        body.get("params") or {}, [int(g) for g in body["sequence"]],
                        body.get("max_steps", 600), body.get("seed"),
                        bool(body.get("handover", True))))
                    return self._json({"ok": True, "result": res})
                res = self._timed("/api/rollout", lambda: rollout_world(
                    w, body.get("source", "feas"), body.get("algorithm", "feas"),
                    body.get("params") or {}, body.get("group"),
                    body.get("max_steps", 300), body.get("seed"), body.get("policy")))
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/value":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded"}, 503)
            try:
                w = normalize_world(body.get("world", {}))
                t0 = time.time()
                res = self._timed("/api/value", lambda: value_world(w, body.get("algorithm", "ihdc"), body.get("params") or {}))
                res["seconds"] = round(time.time() - t0, 3)
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        if path == "/api/solve":
            if not SOLVER["loaded"]:
                return self._json({"ok": False, "error": "solver not loaded: "
                + str(SOLVER["error"])}, 503)
            try:
                world = normalize_world(body.get("world", {}))
                t0 = time.time()
                res = self._timed("/api/solve", lambda: solve_world(
                    world, body.get("algorithm", "feas"), body.get("params") or {},
                    want_stok=bool(body.get("include_stok")),
                    stok_mode=body.get("stok_mode", "full"), x0=body.get("x0")))
                res["seconds"] = round(time.time() - t0, 3)
                return self._json({"ok": True, "result": res})
            except Exception as e:
                import traceback
                log("api_rejected", path=path, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

        world = body
        try:
            norm = normalize_world(world)
        except Exception as e:
            return self._json({"ok": False, "error": str(e)}, 400)
        # Echo the CANONICAL form plus the Python-side view, so a test can assert that the browser
        # and Python agree on what the world is before any solving happens (stage 3).
        return self._json({"ok": True, "world": norm, "setup": world_to_setup(norm)})

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):           # quieter than the default per-request noise
        # NB: log_error() passes an int status code as args[0], so this must not assume a string --
        # doing so raised TypeError inside the handler and the client saw an EMPTY REPLY, not a 404.
        try:
            line = fmt % args
        except Exception:
            line = str(fmt)
        if "/api/" in line:
            return
        sys.stderr.write("  %s\n" % line)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True                   # restart immediately after Ctrl-C
    daemon_threads = True


def port_free(p):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", p))
            return True
        except OSError:
            return False


def free_port(preferred):
    """Return `preferred` if bindable, else an OS-assigned free port. Re-running the server during
    development should not fail merely because the old one is still shutting down."""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-warm", action="store_true", help="skip the solver import")
    # A busy port is now a hard ERROR by default. Silently binding elsewhere was actively harmful:
    # the browser stays pointed at the old server, which keeps serving STALE CODE, and every
    # symptom then looks like a bug in the app rather than "you are talking to yesterday's server".
    # Three servers accumulated that way before it was noticed.
    ap.add_argument("--any-port", action="store_true",
                    help="if the port is busy, use any free one (old behaviour; not recommended)")
    ap.add_argument("--strict-port", action="store_true", help=argparse.SUPPRESS)  # back-compat
    args = ap.parse_args()

    if not os.path.isdir(STATIC):
        sys.exit(f"missing static dir: {STATIC}")

    port = free_port(args.port) if args.any_port else args.port
    url = f"http://127.0.0.1:{port}/"
    if not args.any_port and not port_free(port):
        who = ""
        try:
            out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True, timeout=5).stdout.splitlines()
            pids = sorted({l.split()[1] for l in out[1:] if len(l.split()) > 1})
            if pids:
                who = ("\n  Already serving on that port: pid " + ", ".join(pids) +
                "\n  Stop it first:  kill " + " ".join(pids))
        except Exception:
            pass
        sys.exit(f"port {port} is already in use." + who +
        "\n  (or run with --any-port to bind somewhere else -- but then the browser at "
        f"http://127.0.0.1:{port}/ would still be talking to the OLD server)")
    with Server(("127.0.0.1", port), Handler) as httpd:
        # Serve FIRST, import second: the page is up immediately and reports "loading solver…"
        # rather than the browser failing to connect for however long the import takes.
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"OKBE tutorial  ->  {url}")
        if port != args.port:
            print(f"  (port {args.port} was busy)")
        print("  Ctrl-C to stop")
        if not args.no_browser:
            import webbrowser
            threading.Timer(0.4, webbrowser.open, args=(url,)).start()
        if not args.no_warm:
            warm_solver()                        # main thread -- see the note by SOLVER
            print(f"  solver {'ready' if SOLVER['loaded'] else 'FAILED: ' + str(SOLVER['error'])}"
            f" ({SOLVER['seconds']}s)")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nstopped")
            httpd.shutdown()


if __name__ == "__main__":
    main()
