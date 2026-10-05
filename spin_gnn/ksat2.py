"""DPLL solver instrumented for the "reasoning trace" token encoding (v2),
as opposed to ksat.py's original per-step full-state-snapshot encoding.

The difference from ksat.py's solve_with_trace: instead of re-emitting the
*entire* current clause database at every step (extremely redundant -- see
the copy_vs_delta_breakdown investigation in train_sat_transformer.py),
this only emits *events*: which single variable got set/unset and why.
Two things ksat.py's solve_with_trace didn't need to track, that this does:

- Which clause triggered each unit propagation (ksat.py's propagate() found
  the forced literal but discarded the clause index it came from).
- Explicit "unassign" events for every variable un-set during a backtrack
  unwind, not just the final "give up on this decision level entirely" pop
  -- ksat.py's encoding didn't need these because it recomputed the whole
  clause state from scratch every step, so an unassignment was invisible
  bookkeeping; an event-based encoding has to say it out loud.

Same textbook DPLL as ksat.py: chronological backtracking, no clause
learning, no watched literals, most-constrained-variable (MCV) branching.
`sample_random_ksat` is reused directly from ksat.py.

Event dicts, one per element of the returned "events" list:
  {"kind": "decision" | "flip" | "unit_prop" | "unassign",
   "var": int, "value": int in {-1,1} or None (unassign), "clause": int or
   None (only set for unit_prop, the clause index that forced it),
   "satisfied": list[int] (clause indices newly satisfied by this
   assignment, empty for "unassign"), "reactivated": list[int] (clause
   indices that go back to unsatisfied because this unassign removed the
   value that had satisfied them, empty for assign events)}

"satisfied"/"reactivated" exist because telling a transformer which clause
is currently unit, or whether every clause is now satisfied, otherwise
requires it to silently re-derive clause status from the entire history by
attention alone -- a pure state-tracking burden transformers are known to
be bad at, as opposed to genuine reasoning. Tagging each clause's
satisfied/unsatisfied transition explicitly, right where it happens, turns
both into a locally-checkable fact in the token stream instead of
something to infer. Clauses already satisfied by `hints` before the first
propagate() call are NOT tagged (there's no event to attach the tag to --
hints are listed before BEGIN_STATEMENT even names any clause) -- a known
gap that's inconsequential while every caller uses n_hints=0.

Hints: a partial assignment fixed *before* any decision is ever made --
given facts, not choices, so they're pushed onto the trail ahead of the
first propagate() call and are never touched by backtracking (every
decision's own trail_len_before is computed after the hint prefix, so the
backtrack unwind's `while len(trail) > trail_len_before` can never unwind
past it).
"""
import numpy as np

from .ksat import _active_clauses, sample_random_ksat  # noqa: F401  (re-exported)


def sample_hints(n_vars, n_hints, rng):
    """n_hints distinct variables, each given a uniform-random +-1 value,
    independent of any solution -- a hint is a *given* constraint on the
    problem (like an extra unit clause), not a leaked answer."""
    if n_hints > n_vars:
        raise ValueError(f"n_hints={n_hints} must be <= n_vars={n_vars}")
    variables = rng.choice(n_vars, size=n_hints, replace=False)
    values = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_hints)
    return {int(v): int(val) for v, val in zip(variables, values)}


def solve_with_reasoning_trace(clauses, n_vars, hints=None, max_steps=200_000):
    """DPLL with unit propagation + MCV branching, starting from `hints`
    (dict var -> +-1, or None/{} for no hints).

    Returns a dict:
      events:      list of event dicts, see module docstring.
      hints:       the hints dict actually used (echoed back).
      satisfiable: bool.
      assignment:  final (n_vars,) int8 array if satisfiable else None.
    """
    hints = hints or {}
    assignment = np.zeros(n_vars, dtype=np.int8)
    trail = []
    decision_points = []  # dicts: var, trail_len_before, tried_other, first_value
    events = []

    def check_step_budget():
        if len(events) > max_steps:
            raise RuntimeError(
                f"solve_with_reasoning_trace exceeded max_steps={max_steps} -- correct "
                "DPLL always terminates on a finite formula, so this most likely means "
                "a bug rather than a genuinely huge search; raising rather than "
                "silently truncating the trace."
            )

    for var, value in hints.items():
        assignment[var] = value
        trail.append(var)

    def active_clause_indices():
        return {idx for idx, _ in _active_clauses(clauses, assignment)}

    def propagate():
        while True:
            unit = None  # (var, value, clause_idx)
            for idx, reduced in _active_clauses(clauses, assignment):
                nz = np.flatnonzero(reduced)
                if nz.size == 0:
                    return "conflict"
                if nz.size == 1 and unit is None:
                    unit = (int(nz[0]), int(reduced[nz[0]]), idx)
            if unit is None:
                return "ok"
            var, value, clause_idx = unit
            before = active_clause_indices()
            assignment[var] = value
            trail.append(var)
            satisfied = sorted(before - active_clause_indices())
            events.append({
                "kind": "unit_prop", "var": var, "value": value, "clause": clause_idx,
                "satisfied": satisfied, "reactivated": [],
            })
            check_step_budget()

    def backtrack():
        """Undo the trail through decision points until either a not-yet-tried
        branch is flipped (True) or the stack is exhausted (False = UNSAT).
        Every undone variable gets its own "unassign" event, in the same
        LIFO order the trail unwinds in."""
        while decision_points:
            dp = decision_points[-1]
            while len(trail) > dp["trail_len_before"]:
                before = active_clause_indices()
                var = trail.pop()
                assignment[var] = 0
                reactivated = sorted(active_clause_indices() - before)
                events.append({
                    "kind": "unassign", "var": var, "value": None, "clause": None,
                    "satisfied": [], "reactivated": reactivated,
                })
                check_step_budget()
            if not dp["tried_other"]:
                dp["tried_other"] = True
                other_value = -dp["first_value"]
                before = active_clause_indices()
                assignment[dp["var"]] = other_value
                trail.append(dp["var"])
                satisfied = sorted(before - active_clause_indices())
                events.append({
                    "kind": "flip", "var": dp["var"], "value": other_value, "clause": None,
                    "satisfied": satisfied, "reactivated": [],
                })
                check_step_budget()
                return True
            # dp["var"] was already popped, unassigned, and given its own
            # "unassign" event by the while-loop above (trail_len_before
            # excludes it, so it's the last thing that loop pops) -- this
            # decision level is simply exhausted, no further state change
            # to report, so no new event here.
            decision_points.pop()
        return False

    def result_dict(satisfiable):
        return {
            "events": events,
            "hints": hints,
            "satisfiable": satisfiable,
            "assignment": assignment.copy() if satisfiable else None,
        }

    result = propagate()
    while True:
        if result == "conflict":
            if not backtrack():
                return result_dict(False)
            result = propagate()
            continue

        if not any(True for _ in _active_clauses(clauses, assignment)):
            return result_dict(True)

        var = _most_constrained_variable(clauses, assignment)
        before = active_clause_indices()
        assignment[var] = 1
        trail.append(var)
        decision_points.append({
            "var": var, "trail_len_before": len(trail) - 1,
            "tried_other": False, "first_value": 1,
        })
        satisfied = sorted(before - active_clause_indices())
        events.append({
            "kind": "decision", "var": var, "value": 1, "clause": None,
            "satisfied": satisfied, "reactivated": [],
        })
        check_step_budget()
        result = propagate()


def _most_constrained_variable(clauses, assignment):
    n = assignment.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for _, reduced in _active_clauses(clauses, assignment):
        counts += reduced != 0
    unassigned = np.flatnonzero(assignment == 0)
    return int(unassigned[np.argmax(counts[unassigned])])
