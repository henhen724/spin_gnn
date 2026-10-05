"""Correctness tests for spin_gnn.ksat2's event-based DPLL trace -- the
event stream must be replayable into exactly the same state transitions it
claims, every unit-prop must actually be unit at the time it's claimed, and
the final verdict/assignment must agree with the original (well-tested)
ksat.solve_with_trace on the same instance."""
import numpy as np
import pytest

from spin_gnn.config import SATConfig
from spin_gnn.ksat import sample_random_ksat, solve_with_trace
from spin_gnn.ksat2 import sample_hints, solve_with_reasoning_trace
from spin_gnn.ksat2_tokens import check_satisfies


def _random_instances(n=30, n_vars=10, k=3, alpha=4.2, seed_start=0):
    cfg = SATConfig(n_vars=n_vars, k=k, alpha=alpha)
    for i in range(n):
        rng = np.random.default_rng(seed_start + i)
        yield sample_random_ksat(cfg, rng), cfg.n_vars


def test_agrees_with_original_solver_on_satisfiability():
    for clauses, n_vars in _random_instances():
        old = solve_with_trace(clauses, n_vars)
        new = solve_with_reasoning_trace(clauses, n_vars)
        assert new["satisfiable"] == old["satisfiable"]
        if new["satisfiable"]:
            assert check_satisfies(
                {v: int(new["assignment"][v]) for v in range(n_vars) if new["assignment"][v] != 0},
                clauses, n_vars,
            )


def test_event_stream_replays_to_a_consistent_assignment_history():
    """Every event's precondition/postcondition on a from-scratch replayed
    assignment must hold: decision/flip/unit_prop only ever (re)set an
    UNASSIGNED variable (never overwrite a live value without an
    intervening unassign -- this is exactly the invariant a duplicate
    "unassign" bug would preserve for state but that a duplicate/missing
    event for a *decision/flip/unit_prop* would violate), and unassign only
    ever clears a variable that is currently assigned."""
    for clauses, n_vars in _random_instances(n=40):
        trace = solve_with_reasoning_trace(clauses, n_vars)
        sim = np.zeros(n_vars, dtype=np.int8)
        for var, val in trace["hints"].items():
            sim[var] = val
        for event in trace["events"]:
            var = event["var"]
            if event["kind"] == "unassign":
                assert sim[var] != 0, f"unassign on already-unassigned var {var}"
                sim[var] = 0
            else:
                assert sim[var] == 0, f"{event['kind']} on already-assigned var {var}"
                sim[var] = event["value"]
        if trace["satisfiable"]:
            np.testing.assert_array_equal(sim, trace["assignment"])


def test_no_duplicate_consecutive_unassign_for_same_var():
    """Regression test for a specific bug: popping an exhausted decision
    level must not re-emit an unassign for a variable the trail-unwind loop
    already unassigned (that variable's own unassign event, with no state
    change in between)."""
    for clauses, n_vars in _random_instances(n=60, n_vars=8, alpha=5.5):
        trace = solve_with_reasoning_trace(clauses, n_vars)
        unassign_vars = [e["var"] for e in trace["events"] if e["kind"] == "unassign"]
        for i in range(1, len(unassign_vars)):
            assert unassign_vars[i] != unassign_vars[i - 1], (
                f"duplicate consecutive unassign of var {unassign_vars[i]}"
            )


def test_unit_prop_events_are_actually_unit_at_the_time():
    for clauses, n_vars in _random_instances(n=40):
        trace = solve_with_reasoning_trace(clauses, n_vars)
        sim = np.zeros(n_vars, dtype=np.int8)
        for var, val in trace["hints"].items():
            sim[var] = val
        for event in trace["events"]:
            if event["kind"] == "unit_prop":
                clause = clauses[event["clause"]]
                literal_values = clause * sim
                assert not np.any(literal_values == 1), "clause already satisfied, can't be the unit that fired"
                live = np.flatnonzero((clause != 0) & (literal_values != -1))
                assert live.size == 1 and live[0] == event["var"], (
                    f"clause {event['clause']} isn't unit on var {event['var']} at this point"
                )
                assert clause[event["var"]] == event["value"]
            if event["kind"] == "unassign":
                sim[event["var"]] = 0
            else:
                sim[event["var"]] = event["value"]


def test_hints_are_never_touched_by_backtracking():
    cfg = SATConfig(n_vars=10, k=3, alpha=4.2)
    for i in range(30):
        rng = np.random.default_rng(1000 + i)
        clauses = sample_random_ksat(cfg, rng)
        hint_rng = np.random.default_rng(2000 + i)
        hints = sample_hints(cfg.n_vars, n_hints=2, rng=hint_rng)
        trace = solve_with_reasoning_trace(clauses, cfg.n_vars, hints=hints)
        for event in trace["events"]:
            assert event["var"] not in hints, "a hinted variable must never appear as a later event"
        if trace["satisfiable"]:
            for var, val in hints.items():
                assert trace["assignment"][var] == val


def test_sample_hints_rejects_too_many():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_hints"):
        sample_hints(n_vars=5, n_hints=6, rng=rng)
