"""Random k-SAT instance sampling and a DPLL unit-propagation solver with a
most-constrained-variable (MCV) branching heuristic, emitting the solver's
execution trace as a sequence of numeric vector tokens.

This is deliberately the simple textbook DPLL algorithm (chronological
backtracking, no clause learning, no watched literals) -- the point of this
module is observing unit-propagation/backtracking *dynamics* at toy problem
sizes for the exploration in sat_stats.py and train_sat_transformer.py, not
building a competitive SAT solver. Clause status/reduction is recomputed from
`clauses` + the live assignment on every scan (O(m*n) per scan) rather than
maintaining incrementally-updated mutable clause state that would need to be
un-done in sync with backtracking -- simplest-correct at this scale, matching
this repo's existing preference for plain loops over cleverness at toy sizes
(see simulate.py's docstring).

Token encoding (shared by both token types below), a vector in R^{n+1}:
  dims 0..n-1: **variable-values token**: current assignment per variable
               (-1=False, 0=unassigned, 1=True).
               **clause token**: -1 if the variable appears negated in this
               clause, +1 if it appears positive, 0 if the variable is absent
               from the clause OR if it appeared but has since been falsified
               by the current assignment (reduced-clause convention --
               falsified literals are zeroed out as soon as they're
               falsified, rather than keeping the clause's original pattern).
  dim n:       -1 for a variable-values token, +1 for a clause token.

One "step" = one point where the assignment changes: a unit-propagation
assignment, a branching decision, a branch flip (retrying a decision's other
value), or a backtrack pop (a decision fully exhausted in both directions).
Each step emits one variable-values token followed by one clause token per
currently non-trivial (not yet satisfied) clause, in fixed clause-index
order. Only backtrack pops count toward `num_backtracks` -- flips are just
completing the current decision level's branch, not backtracking past it.
"""
import numpy as np

from .config import SATConfig


def sample_random_ksat(cfg: SATConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample m = round(alpha * n_vars) random k-clauses over n_vars
    variables. Returns an (m, n_vars) int8 array in {-1,0,1}: row c, column v
    is -1/1 if variable v appears negated/positive in clause c, 0 if it
    doesn't appear. Each clause picks k distinct variables uniformly at
    random with an independent uniform-random sign for each."""
    n, k = cfg.n_vars, cfg.k
    if k > n:
        raise ValueError(f"k={k} must be <= n_vars={n}")
    m = max(1, round(cfg.alpha * n))
    clauses = np.zeros((m, n), dtype=np.int8)
    for c in range(m):
        variables = rng.choice(n, size=k, replace=False)
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=k)
        clauses[c, variables] = signs
    return clauses


def initial_prompt(clauses, n_vars):
    """The step-0 token block for a fresh instance -- never emitted by
    solve_with_trace itself (it only emits a step after the *first*
    assignment change), but a well-defined starting point for prompting a
    trained model to generate a trace "from scratch": the all-unassigned
    variable-values token, followed by every original clause (all of them
    are still non-trivial before any assignment)."""
    assignment_token = np.concatenate([np.zeros(n_vars, dtype=np.float32), np.float32([-1.0])])
    clause_tokens = [
        np.concatenate([clauses[c].astype(np.float32), np.float32([1.0])]) for c in range(clauses.shape[0])
    ]
    return np.stack([assignment_token] + clause_tokens, axis=0)


def _reduced_clause(clause, assignment):
    """Reduced encoding of one clause under the current assignment: falsified
    literals zeroed, live/unassigned literals unchanged, or None if the
    clause is already satisfied (some literal evaluates true)."""
    literal_values = clause * assignment  # +1 true, -1 false, 0 absent/unassigned
    if np.any(literal_values == 1):
        return None
    return np.where(literal_values == -1, 0, clause)


def _active_clauses(clauses, assignment):
    """Yield (index, reduced_encoding) for every not-yet-satisfied clause."""
    for c in range(clauses.shape[0]):
        reduced = _reduced_clause(clauses[c], assignment)
        if reduced is not None:
            yield c, reduced


def _emit_step(clauses, assignment, steps, events, kind, var, value, max_steps):
    # The label must be a float32 array too -- concatenating a float32 array
    # with a bare Python-float list silently upcasts the whole result to
    # float64 (numpy's type promotion), which then crashes deep in
    # sat_transformer.py's nn.Linear (float32 weights) on any tokens array
    # that bypasses sat_dataset.py's dtype-casting padded-batch allocation.
    tokens = [np.concatenate([assignment.astype(np.float32), np.float32([-1.0])])]
    for _, reduced in _active_clauses(clauses, assignment):
        tokens.append(np.concatenate([reduced.astype(np.float32), np.float32([1.0])]))
    steps.append(np.stack(tokens, axis=0))
    events.append({
        "kind": kind,
        "var": int(var) if var is not None else None,
        "value": int(value) if value is not None else None,
    })
    if len(steps) > max_steps:
        raise RuntimeError(
            f"solve_with_trace exceeded max_steps={max_steps} -- correct DPLL "
            "always terminates on a finite formula, so this most likely means "
            "a bug rather than a genuinely huge search; raising rather than "
            "silently truncating the trace."
        )


def _most_constrained_variable(clauses, assignment):
    """Among unassigned variables, the one appearing (either polarity) in the
    most currently-unsatisfied clauses; ties broken by lowest index."""
    n = assignment.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for _, reduced in _active_clauses(clauses, assignment):
        counts += reduced != 0
    unassigned = np.flatnonzero(assignment == 0)
    return int(unassigned[np.argmax(counts[unassigned])])


def _stack_tokens(steps, n_vars):
    if not steps:
        return np.zeros((0, n_vars + 1), dtype=np.float32)
    return np.concatenate(steps, axis=0)


def solve_with_trace(clauses, n_vars, max_steps=200_000):
    """DPLL with unit propagation + most-constrained-variable branching over
    `clauses` ((m, n_vars) array as returned by sample_random_ksat).

    Returns a dict:
      tokens:       (T, n_vars+1) float32 array, the full trace token stream
                    (every step's token block, concatenated in solver order).
      step_lengths: list[int], tokens per step -- step i's tokens are
                    tokens[sum(step_lengths[:i]) : sum(step_lengths[:i+1])].
      events:       list[dict], one per step: {"kind": one of "unit_prop",
                    "decision", "flip", "backtrack", "var": int, "value":
                    int or None (backtrack pops have no associated value)}.
      satisfiable:  bool.
      assignment:   final (n_vars,) int8 array if satisfiable else None.
    """
    assignment = np.zeros(n_vars, dtype=np.int8)
    trail = []  # assigned variable indices, in assignment order
    decision_points = []  # dicts: var, trail_len_before, tried_other, first_value
    steps, events = [], []

    def propagate():
        while True:
            unit = None
            for _, reduced in _active_clauses(clauses, assignment):
                nz = np.flatnonzero(reduced)
                if nz.size == 0:
                    return "conflict"
                if nz.size == 1 and unit is None:
                    unit = (int(nz[0]), int(reduced[nz[0]]))
            if unit is None:
                return "ok"
            var, value = unit
            assignment[var] = value
            trail.append(var)
            _emit_step(clauses, assignment, steps, events, "unit_prop", var, value, max_steps)

    def backtrack():
        """Undo the trail through decision points until either a not-yet-tried
        branch is flipped (True) or the stack is exhausted (False = UNSAT)."""
        while decision_points:
            dp = decision_points[-1]
            while len(trail) > dp["trail_len_before"]:
                assignment[trail.pop()] = 0
            if not dp["tried_other"]:
                dp["tried_other"] = True
                other_value = -dp["first_value"]
                assignment[dp["var"]] = other_value
                trail.append(dp["var"])
                _emit_step(clauses, assignment, steps, events, "flip", dp["var"], other_value, max_steps)
                return True
            decision_points.pop()
            _emit_step(clauses, assignment, steps, events, "backtrack", dp["var"], None, max_steps)
        return False

    def result_dict(satisfiable):
        return {
            "tokens": _stack_tokens(steps, n_vars),
            "step_lengths": [len(s) for s in steps],
            "events": events,
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
        assignment[var] = 1
        trail.append(var)
        decision_points.append({
            "var": var, "trail_len_before": len(trail) - 1,
            "tried_other": False, "first_value": 1,
        })
        _emit_step(clauses, assignment, steps, events, "decision", var, 1, max_steps)
        result = propagate()
