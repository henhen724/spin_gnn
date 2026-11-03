"""Correctness checks for the `fall=True` graph ensemble (finding 43).

A fall colouring is one where every vertex is adjacent to a vertex of every
other colour (Dunbar et al. 2000). It matters because DD can only ever deduce
a node whose neighbourhood shows all q-1 other colours: a node missing one is
a PERMANENT hint, and under the unconstrained stub split their density is
fixed at 0.3827 (q=4, d=5), which is the whole of the old rho_min ~ 0.475
floor. Forcing the planted colouring to be fall drives that to exactly 0 and
drops the floor below rho_c ~ 0.46, which is what opens the slow phase.

Every result in the project since 11-02 is measured on this ensemble, so a
silent regression here would put the whole scan back in the fast phase while
still producing plausible numbers.

These checks were written as `$SCRATCH/fall_verify.py` and ran once before the
flag was used. $SCRATCH purges at 90 days, so they are ported here. ONE check
is deliberately not ported: the original compared `fall=False` against
`git show HEAD:spin_gnn/sudoku_dd.py` to prove the new flag left the default
rng stream untouched. That was a migration guard -- HEAD now contains the
flag, so the comparison is vacuous and keeping it would be a test that cannot
fail. Its surviving half is test_default_ensemble_still_has_uncolourful_nodes,
which pins the behaviour the default path is supposed to have.

No pytest import: the only pytest on Sherlock is py39 (`ml spider py-pytest`)
and this project runs on the py312 stack, so the suite cannot be collected
here at all. Plain asserts and a __main__ runner work under the real module
stack, and pytest still collects `test_*` functions from a `test_*.py` file
that never imports it. Run it with:

    python3 tests/test_sudoku_fall.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spin_gnn.sudoku_dd import dd_times, planted_regular_graph, removal_path

Q = 4


def colourful_mask(adj, colors, q):
    """Per node: does its closed neighbourhood show all q colours?

    `seen[v, c]` is True when v or one of its neighbours has colour c. A node
    is deducible by DD only if it is colourful, so the mean of ~mask is the
    permanently-undeducible fraction that sets the rho_min floor.
    """
    n = len(colors)
    seen = np.zeros((n, q), dtype=bool)
    seen[np.repeat(np.arange(n), adj.shape[1]), colors[adj].ravel()] = True
    seen[np.arange(n), colors] = True
    return seen.all(axis=1)


def build_fall(N, d, q, seed, attempts=20):
    """A fall graph, retrying the rng. The construction can fail by chance.

    It reserves one mandatory stub per ordered colour pair before splitting
    the rest at random, and the random remainder can still dead-end into a
    multi-edge or self-loop, so a RuntimeError is a failed draw and not a bug.
    """
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        try:
            return planted_regular_graph(N, d, q, rng, fall=True)
        except RuntimeError:
            continue
    raise AssertionError(f"no fall graph for N={N} d={d} in {attempts} attempts")


def assert_valid_planted_graph(adj, colors, N, d, q):
    assert adj.shape == (N, d), f"shape {adj.shape} != {(N, d)}"
    assert np.all(np.bincount(colors, minlength=q) == N // q), "colouring not balanced"
    assert (colors[adj] != colors[:, None]).all(), "colouring not proper"
    for v in range(N):
        assert v not in adj[v], f"self-loop at {v}"
        assert len(set(adj[v].tolist())) == d, f"repeated edge at {v}"
    back = np.zeros((N, N), dtype=bool)
    back[np.repeat(np.arange(N), d), adj.ravel()] = True
    assert (back == back.T).all(), "adjacency not symmetric"


def test_fall_graphs_are_valid_and_every_node_is_colourful():
    """The fall constraint must be added WITHOUT breaking the base ensemble.

    d=3 at q=4 is the load-bearing case: d = q-1 is the minimum degree at
    which a fall colouring can exist at all, since each vertex needs q-1
    distinct foreign colours among exactly d neighbours, so every neighbour's
    colour is forced to be distinct. If the construction is going to fail
    anywhere it fails there.
    """
    for N, d in ((48, 5), (96, 5), (252, 7), (48, 3)):
        adj, colors = build_fall(N, d, Q, seed=0)
        assert_valid_planted_graph(adj, colors, N, d, Q)
        mask = colourful_mask(adj, colors, Q)
        assert mask.all(), f"N={N} d={d}: {(~mask).sum()} non-colourful nodes under fall=True"


def test_default_ensemble_still_has_uncolourful_nodes():
    """The other half of the regression guard: fall must be OPT-IN.

    0.3827 at q=4,d=5 is the inclusion-exclusion prediction for the
    unconstrained split, measured at 0.371-0.383 over N in {996, 4140, 17196}.
    A default path that came back colourful would mean `fall` had leaked into
    it, and every pre-11-02 checkpoint would no longer match its own ensemble.
    """
    N, d = 996, 5
    fracs = []
    for seed in range(3):
        adj, colors = planted_regular_graph(N, d, Q, np.random.default_rng(seed))
        assert_valid_planted_graph(adj, colors, N, d, Q)
        fracs.append(float((~colourful_mask(adj, colors, Q)).mean()))
    mean = float(np.mean(fracs))
    assert 0.34 < mean < 0.42, f"uncolourful fraction {mean:.4f} outside the measured 0.371-0.383"


def test_incremental_removal_agrees_with_from_scratch_dd_under_fall():
    """`remove_hint`'s maintained times must equal `dd_times` on the ensemble
    we now train on, not just on the default one the existing suite covers."""
    for N, d in ((48, 5), (96, 5)):
        adj, colors = build_fall(N, d, Q, seed=1)
        path = removal_path(adj, colors, Q, np.random.default_rng(2))
        ref = dd_times(adj, colors, path["final_is_hint"], Q)
        assert np.array_equal(ref, path["final_t"]), f"N={N} d={d}: incremental != from-scratch"


def test_fall_lowers_the_deduction_floor():
    """The claim the whole 11-02 line of work rests on, at a training size.

    Finding 43 measured N=48 rho_min 0.469 +- 0.049 -> 0.332 +- 0.024 over 200
    seeds, with T_end 7.9 -> 13.8. 24 seeds is enough to separate means that
    far apart and keeps this runnable in seconds; the bar is set at 0.40,
    between the two measured means and ~3 sd below the unconstrained one, so
    seed noise cannot trip it but a genuine regression to the old ensemble
    would. Asserted as a gap rather than an absolute so it tracks the
    generator if the ensemble is ever retuned.
    """
    N, d, n_seeds = 48, 5, 24
    floors, depths = {}, {}
    for name, fall in (("random", False), ("fall", True)):
        rmin, tend = [], []
        for seed in range(n_seeds):
            if fall:
                adj, colors = build_fall(N, d, Q, seed=9_000 + seed)
            else:
                adj, colors = planted_regular_graph(N, d, Q, np.random.default_rng(9_000 + seed))
            path = removal_path(adj, colors, Q, np.random.default_rng(seed))
            rmin.append(float(path["rho_min"]))
            tend.append(float(path["T"][-1]))
        floors[name] = float(np.mean(rmin))
        depths[name] = float(np.mean(tend))

    assert floors["fall"] < floors["random"] - 0.08, (
        f"fall floor {floors['fall']:.4f} not meaningfully below "
        f"random {floors['random']:.4f} -- the slow phase has closed")
    assert floors["fall"] < 0.40, (
        f"fall floor {floors['fall']:.4f} is not below rho_c ~ 0.46 with margin")
    assert depths["fall"] > depths["random"], (
        f"fall T_end {depths['fall']:.1f} <= random {depths['random']:.1f} -- "
        "a lower floor that does not buy reasoning depth means the sampler changed")
    print(f"    rho_min  random {floors['random']:.4f} -> fall {floors['fall']:.4f}")
    print(f"    T_end    random {depths['random']:.1f} -> fall {depths['fall']:.1f}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
