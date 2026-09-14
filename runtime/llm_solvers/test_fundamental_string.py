"""Correctness checks for fundamental_world's `sequence` support (the "occupancy" tab's fundamental
matrix, composed across a policy string).

The composed matrix is NOT a flat sum of each leg's own fundamental matrix -- leg i+1 does not
start fresh from x0, it starts from wherever leg i's process handed off, a DISTRIBUTION over leg
i's own goal cells (one more real step under handover_kernel if a cell's goal is state-action).
See fundamental_world's own module docstring for the exact "reach @ N_i, propagated leg to leg"
derivation this file checks.

GROUND TRUTH here is an EXPANDED (leg, cell) product-state Markov chain, built independently by
hand below rather than reusing any of fundamental_world's own machinery: leg i's own policy governs
transitions within (i, ·); a state-only goal cell moves the LEG index without moving the CELL
(same timestep); a state-action goal cell executes the terminal action -- one real kernel step --
before the leg index advances; a constraint is absorbing regardless of which leg it is hit in; the
FINAL leg's own goal cells are the only ones that stay absorbing rather than triggering a handoff.
Its own fundamental matrix (via the same utilities.compute_fundamental_matrix_with_absorption,
which is legitimate to reuse -- it is the thing being composed, not the composition logic itself)
is computed once over the WHOLE expanded chain, in one linear solve, with no leg-to-leg propagation
of any kind -- an entirely different computation path from fundamental_world's -- then marginalised
by summing over legs to compare against fundamental_world's own composed answer, cell for cell,
exactly (not sampled).
"""
import os
import sys

import numpy as np
import pytest

# tutorial_server.py and utilities.py live one level up (main_files/), not alongside this file
# (main_files/llm_solvers/).
_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

import tutorial_server as TS
from utilities import compute_fundamental_matrix_with_absorption

ROWS = COLS = 6
NX = ROWS * COLS


def build_world(goals, constraints=(20,), p_main=0.85):
    return TS.normalize_world({
        "rows": ROWS, "cols": COLS, "walls": [], "constraints": list(constraints),
        "goals": goals, "start": 0, "p_main": p_main,
    })


def ground_truth(world, seq, x0):
    """Fundamental matrix of the composed string, via one big (leg, cell) product chain -- see
    module docstring. Returns the [nx] row for x0, marginalised back down to plain cells."""
    _sp, P = TS.get_kernel(ROWS, COLS, world["walls"], world["p_main"], world.get("kernel"))
    by_id = {g["id"]: g for g in world["goals"]}
    k = len(seq)
    pis = []
    for gid in seq:
        one = dict(world); one["goals"] = [by_id[gid]]
        res = TS.solve_world(one, "feas2s", {})
        pis.append(np.asarray(res["surfaces"][0]["pi"], dtype=int))

    N = NX * k
    idx = lambda leg, x: leg * NX + x   # noqa: E731 -- leg is 0-indexed here
    P_big = np.zeros((N, N))
    goalcells = []      # per leg: {cell: a_or_None}
    for leg, gid in enumerate(seq):
        grp = by_id[gid]
        goalcells.append({c["li"]: c["a"] for c in grp["cells"]})

    # A STATE-ONLY handoff cell must never be its own separately-VISITED node: leg i+1 begins
    # governing at the SAME timestep leg i arrives there, so (leg, x) and (leg+1, x) would both
    # independently claim the identity-term visit for that one real instant if both existed as
    # ordinary nodes -- exactly the double-count fundamental_world's own leg_fundamental has to
    # avoid too (see its comment). Modelled here by REDIRECTING at the SOURCE: any transition
    # that would land on a state-only goal cell of the CURRENT leg is redirected one leg forward
    # instead, so the (leg, that-cell) node is simply never reached and never contributes a visit.
    # A STATE-ACTION handoff cell has no such problem -- reaching it and executing the terminal
    # action are two genuinely different timesteps -- so it stays an ordinary, reachable node.
    def route(leg, y, p, out):
        gc = goalcells[leg]
        if leg < k - 1 and y in gc and gc[y] is None:
            out[idx(leg + 1, y)] += p
        else:
            out[idx(leg, y)] += p

    for leg in range(k):
        gc = goalcells[leg]
        pi = pis[leg]
        for x in range(NX):
            if x in world["constraints"]:
                P_big[idx(leg, x), idx(leg, x)] = 1.0   # absorbing regardless of leg
                continue
            if x in gc:
                if leg == k - 1:
                    P_big[idx(leg, x), idx(leg, x)] = 1.0   # final leg's own goal: absorbing
                    continue
                a = gc[x]
                if a is None:
                    continue   # unreachable by construction (redirected at every source above)
                for y in range(NX):
                    p = P[int(a), x, y]
                    if p > 0:
                        P_big[idx(leg, x), idx(leg + 1, y)] += p
                continue
            a = int(pi[x])
            for y in range(NX):
                p = P[a, x, y]
                if p > 0:
                    route(leg, y, p, P_big[idx(leg, x)])

    # "goal" for the expanded chain is ONLY the final leg's own goal cells; everything else that
    # is absorbing in P_big (constraints at any leg) goes in as a constraint at that (leg,cell).
    final_goals = [idx(k - 1, x) for x in goalcells[k - 1]]
    all_constraints = [idx(leg, x) for leg in range(k) for x in world["constraints"]]
    walls = []   # none in this test world
    N_big = compute_fundamental_matrix_with_absorption(P_big, final_goals, all_constraints, walls)
    row = N_big[idx(0, x0)]
    out = np.zeros(NX)
    for leg in range(k):
        out += row[idx(leg, 0):idx(leg, NX)]
    return out


CASES = {
    "two legs, state-only handoff": [
        {"id": 1, "cells": [14]}, {"id": 2, "cells": [35]},
    ],
    "two legs, state-action handoff": [
        {"id": 1, "cells": [{"li": 14, "a": 3}]}, {"id": 2, "cells": [35]},
    ],
    "three legs, mixed handoff": [
        {"id": 1, "cells": [{"li": 14, "a": 3}]}, {"id": 2, "cells": [28]}, {"id": 3, "cells": [35]},
    ],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_composed_fundamental_matches_the_expanded_chain_exactly(name):
    # A state-action handoff's own constraint stays at cell 20 (adjacent to leg 1's goal, cell
    # 14) for the state-only case, but moves to 33 (nowhere near 14's noise-scattered neighbours)
    # for the two cases with a state-action leg -- handover_kernel's own documented LIMITATION
    # ("the handover step is not itself checked against constraints") means the terminal action's
    # noise can, in principle, slip onto a constraint cell without being caught as a violation.
    # That is an existing, separately-documented property of handover_kernel shared with STOK/eta
    # composition, not something this test (or fundamental_world's own composition logic) is
    # responsible for -- so the geometry here is chosen to not exercise it, keeping this test
    # about the leg-to-leg composition itself.
    constraints = (33,) if ("state-action" in name or "mixed" in name) else (20,)
    world = build_world(CASES[name], constraints=constraints)
    seq = [g["id"] for g in CASES[name]]
    x0 = 0
    res = TS.fundamental_world(world, "feas2s", {}, None, None, seq)
    formula = np.asarray(res["surfaces"][0]["fundamental"])[x0]
    truth = ground_truth(world, seq, x0)
    # atol above the JSON response's own 5-decimal rounding (fundamental_world rounds every entry
    # to round(v, 5) before it ever reaches this test), not a looseness in the check itself.
    assert np.allclose(formula, truth, rtol=0, atol=2e-5), \
        f"{name}: max diff {np.abs(formula - truth).max()} at cell {np.abs(formula - truth).argmax()}"


def test_single_leg_sequence_matches_the_plain_single_policy_answer():
    """A length-1 sequence has no handoff at all to get wrong -- degenerate check that the
    sequence path and the plain group_id path agree."""
    world = build_world([{"id": 1, "cells": [14]}])
    plain = TS.fundamental_world(world, "feas2s", {}, None, None, None)
    composed = TS.fundamental_world(world, "feas2s", {}, None, None, [1])
    plain_N = np.asarray(plain["surfaces"][0]["fundamental"])
    composed_N = np.asarray(composed["surfaces"][0]["fundamental"])
    assert np.allclose(plain_N, composed_N, rtol=0, atol=1e-9)


def test_sequence_naming_a_missing_policy_raises_a_clean_error():
    world = build_world([{"id": 1, "cells": [14]}])
    with pytest.raises(ValueError):
        TS.fundamental_world(world, "feas2s", {}, None, None, [1, 99])
