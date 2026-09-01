"""Tests for spin_gnn.ksat_render's per-step (not per-token) rendering:
trace_to_lines (one line per step: assignment + all non-trivial clauses) and
side_by_side_trace (two such line lists as aligned columns)."""
import numpy as np

from spin_gnn.ksat import sample_random_ksat, solve_with_trace
from spin_gnn.ksat_render import infer_step_lengths, side_by_side_trace, trace_to_lines
from spin_gnn.config import SATConfig


def test_trace_to_lines_one_line_per_step_with_assignment_and_clauses():
    n_vars = 4
    # step 0: assignment token (all unassigned) + two clause tokens.
    tokens = np.array(
        [
            [0, 0, 0, 0, -1],  # assignment: all unassigned
            [1, 0, 0, 0, 1],  # clause: (x1)
            [0, -1, 0, 0, 1],  # clause: (~x2)
            [1, 0, 0, 0, -1],  # step 1's assignment: x1=True
        ],
        dtype=np.float32,
    )
    step_lengths = [3, 1]

    lines = trace_to_lines(tokens, step_lengths)

    assert len(lines) == 2
    assert lines[0].startswith("x1=? x2=? x3=? x4=?")
    assert "(x1)" in lines[0]
    assert "(~x2)" in lines[0]
    assert lines[1] == "x1=T x2=? x3=? x4=?"  # no clauses this step -> no trailing junk


def test_trace_to_lines_matches_real_solver_trace():
    """End-to-end: a real solved instance's step count must match
    len(step_lengths), and every line must actually start with an
    assignment string (sanity, not exact-value, since the solver's own
    dynamics aren't being re-tested here -- that's test_ksat_solver.py's job)."""
    cfg = SATConfig(n_vars=6, k=3, alpha=4.2)
    rng = np.random.default_rng(0)
    clauses = sample_random_ksat(cfg, rng)
    trace = solve_with_trace(clauses, cfg.n_vars)

    lines = trace_to_lines(trace["tokens"], trace["step_lengths"])

    assert len(lines) == len(trace["step_lengths"])
    for line in lines:
        assert line.startswith("x1=")


def test_side_by_side_trace_aligns_and_pads_unequal_lengths():
    left = ["short", "a bit longer line"]
    right = ["one", "two", "three (extra step)"]

    out = side_by_side_trace(left, right, left_title="MODEL", right_title="TRUE", col_width=20)
    rows = out.splitlines()

    assert "MODEL" in rows[0] and "TRUE" in rows[0]
    assert len(rows) == 2 + 3  # header + rule + 3 step rows (longer of the two inputs)
    assert "short" in rows[2]
    assert "one" in rows[2]
    assert "three (extra step)" in rows[-1]
    # left column padded/present even though `left` ran out at step 2:
    assert rows[-1].strip().endswith("three (extra step)")


def test_infer_step_lengths_matches_solver_on_a_wellformed_trace():
    """On solve_with_trace's own output, inference must reproduce exactly
    the same step_lengths the solver itself returned -- trace_to_lines'
    fallback path (used for freely model-generated sequences, which have no
    externally-known grouping) must not silently disagree with the ground
    truth when the input actually IS well-formed."""
    cfg = SATConfig(n_vars=6, k=3, alpha=4.2)
    rng = np.random.default_rng(1)
    clauses = sample_random_ksat(cfg, rng)
    trace = solve_with_trace(clauses, cfg.n_vars)

    assert infer_step_lengths(trace["tokens"]) == trace["step_lengths"]


def test_infer_step_lengths_handles_malformed_sequence():
    """A sequence that doesn't follow the expected assignment-then-clauses
    alternation (e.g. two assignment tokens in a row, kind=-1 both) must
    not crash -- it just reports whatever grouping falls out, here two
    length-1 steps rather than one merged step."""
    tokens = np.array([[0, 0, -1], [0, 0, -1]], dtype=np.float32)
    assert infer_step_lengths(tokens) == [1, 1]


def test_side_by_side_trace_truncates_long_lines():
    long_line = "x" * 200
    out = side_by_side_trace([long_line], ["y"], col_width=10)
    data_row = out.splitlines()[-1]
    assert "..." in data_row
    assert long_line not in data_row
