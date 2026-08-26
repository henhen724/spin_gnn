"""Correctness checks for spin_gnn.ksat's DPLL unit-propagation solver.

There's no closed-form ground truth here the way test_exact_recovery.py has
for SpinGNN, so the real correctness guarantee is the brute-force cross-check
below: for small enough n to enumerate every assignment, solve_with_trace's
satisfiable/unsatisfiable verdict must agree with exhaustive search. The
hand-crafted instances pin down the exact trace-encoding conventions (reduced
clauses, token dims, event kinds) that the brute-force check can't see.
"""
import itertools

import numpy as np

from spin_gnn.ksat import initial_prompt, sample_random_ksat, solve_with_trace
from spin_gnn.config import SATConfig


def _brute_force_satisfiable(clauses):
    n = clauses.shape[1]
    for bits in itertools.product([-1, 1], repeat=n):
        assignment = np.array(bits, dtype=np.int8)
        if all(np.any(clauses[c] * assignment == 1) for c in range(clauses.shape[0])):
            return True
    return False


def test_chain_of_unit_propagations_is_satisfiable_with_reduced_clause_encoding():
    # clause0 = (x0 or x1), clause1 = (not x0) -- a unit clause that forces
    # x0=False, which then falsifies clause0's x0 literal and leaves x1 as
    # its sole live literal, forcing x1=True. No decisions needed.
    clauses = np.array([[1, 1], [-1, 0]], dtype=np.int8)

    trace = solve_with_trace(clauses, n_vars=2)

    assert trace["satisfiable"] is True
    np.testing.assert_array_equal(trace["assignment"], [-1, 1])

    assert [e["kind"] for e in trace["events"]] == ["unit_prop", "unit_prop"]
    assert trace["events"][0] == {"kind": "unit_prop", "var": 0, "value": -1}
    assert trace["events"][1] == {"kind": "unit_prop", "var": 1, "value": 1}

    assert trace["step_lengths"] == [2, 1]
    tokens = trace["tokens"]
    assert tokens.shape == (3, 3)
    # step 0: assignment token, then clause0's reduced encoding (x0's slot
    # zeroed since it was just falsified, x1's slot still live at +1).
    np.testing.assert_array_equal(tokens[0], [-1.0, 0.0, -1.0])
    np.testing.assert_array_equal(tokens[1], [0.0, 1.0, 1.0])
    # step 1: assignment token only -- both clauses satisfied, no active clauses left.
    np.testing.assert_array_equal(tokens[2], [-1.0, 1.0, -1.0])


def test_immediate_contradiction_is_unsatisfiable_with_no_decisions():
    # x0 and not x0: unit-propagates x0=True, then "not x0" becomes an empty
    # (fully falsified) clause before any decision is ever made, so
    # backtracking has no decision to undo -- UNSAT with zero decisions.
    clauses = np.array([[1], [-1]], dtype=np.int8)

    trace = solve_with_trace(clauses, n_vars=1)

    assert trace["satisfiable"] is False
    assert trace["assignment"] is None
    assert all(e["kind"] != "decision" for e in trace["events"])
    assert all(e["kind"] != "backtrack" for e in trace["events"])


def test_solver_verdict_matches_brute_force_on_random_small_instances():
    rng = np.random.default_rng(0)
    saw_sat, saw_unsat = False, False
    for alpha in (2.0, 4.2, 7.0):
        cfg = SATConfig(n_vars=9, k=3, alpha=alpha, seed=0)
        for seed in range(15):
            clauses = sample_random_ksat(cfg, np.random.default_rng(seed))
            trace = solve_with_trace(clauses, cfg.n_vars)
            expected = _brute_force_satisfiable(clauses)
            assert trace["satisfiable"] == expected, (alpha, seed)
            if trace["satisfiable"]:
                np.testing.assert_equal(
                    all(np.any(clauses[c] * trace["assignment"] == 1) for c in range(clauses.shape[0])),
                    True,
                )
                saw_sat = True
            else:
                saw_unsat = True
    # sanity: the alpha sweep above should actually exercise both outcomes,
    # not vacuously pass because everything landed on one side.
    assert saw_sat and saw_unsat


def test_trace_encoding_invariants_on_a_larger_random_instance():
    cfg = SATConfig(n_vars=20, k=3, alpha=4.2, seed=1)
    rng = np.random.default_rng(1)
    clauses = sample_random_ksat(cfg, rng)
    trace = solve_with_trace(clauses, cfg.n_vars)

    tokens = trace["tokens"]
    # Regression check: np.concatenate silently upcasts float32 to float64 if
    # any operand isn't explicitly float32 (e.g. a bare Python-float list) --
    # caught this exact bug via a downstream crash in sat_transformer.py's
    # nn.Linear, not via value assertions, since values were still correct.
    assert tokens.dtype == np.float32
    assert set(np.unique(tokens)).issubset({-1.0, 0.0, 1.0})

    offset = 0
    for length in trace["step_lengths"]:
        block = tokens[offset : offset + length]
        assert block[0, -1] == -1.0  # variable-values token
        if length > 1:
            assert np.all(block[1:, -1] == 1.0)  # clause tokens
        offset += length
    assert offset == tokens.shape[0]


def test_initial_prompt_encodes_all_unassigned_plus_every_clause():
    clauses = np.array([[1, 1], [-1, 0]], dtype=np.int8)
    prompt = initial_prompt(clauses, n_vars=2)

    assert prompt.shape == (3, 3)
    assert prompt.dtype == np.float32
    np.testing.assert_array_equal(prompt[0], [0.0, 0.0, -1.0])  # all-unassigned assignment token
    np.testing.assert_array_equal(prompt[1], [1.0, 1.0, 1.0])  # clause 0, unreduced (nothing assigned yet)
    np.testing.assert_array_equal(prompt[2], [-1.0, 0.0, 1.0])  # clause 1


def test_backtrack_and_decision_counts_are_consistent():
    # High clause/variable ratio -> likely UNSAT and likely to need some
    # backtracking, but the invariant checked here holds regardless of outcome.
    cfg = SATConfig(n_vars=15, k=3, alpha=6.0, seed=2)
    rng = np.random.default_rng(2)
    clauses = sample_random_ksat(cfg, rng)
    trace = solve_with_trace(clauses, cfg.n_vars)

    kinds = [e["kind"] for e in trace["events"]]
    num_decisions = kinds.count("decision")
    num_backtracks = kinds.count("backtrack")
    assert num_backtracks >= 0
    assert num_backtracks <= num_decisions
    if trace["satisfiable"] is False:
        # UNSAT can only be reached by exhausting (backtracking past) every
        # decision that was ever made, including the very first.
        assert num_backtracks == num_decisions
