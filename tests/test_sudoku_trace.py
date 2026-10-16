"""Correctness checks for spin_gnn.sudoku_trace: the recorded round-by-round
event trace must agree exactly with sudoku_dd.dd_times's from-scratch
deduction times (the existing, independently-tested reference)."""
import numpy as np
import pytest

from spin_gnn.sudoku_dd import INF, dd_times, planted_regular_graph
from spin_gnn.sudoku_trace import sample_hints, solve_with_dd_trace


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("N,d,q,rho", [(48, 5, 4, 0.4), (60, 4, 3, 0.5), (96, 7, 4, 0.25)])
def test_trace_times_match_dd_times(N, d, q, rho, seed):
    rng = np.random.default_rng(seed)
    adj, colors = planted_regular_graph(N, d, q, rng)
    n_hints = round(rho * N)
    hints = sample_hints(N, n_hints, colors, rng)
    trace = solve_with_dd_trace(adj, colors, hints, q)

    is_hint = np.zeros(N, dtype=bool)
    is_hint[list(hints)] = True
    t_ref = dd_times(adj, colors, is_hint, q)

    trace_time = {node: 0 for node in hints}
    for r, round_events in enumerate(trace["rounds"], start=1):
        for node, c in round_events:
            trace_time[node] = r
            assert c == int(colors[node])

    for node in range(N):
        if t_ref[node] >= INF:
            assert node not in trace_time
        else:
            assert trace_time[node] == t_ref[node]
    assert trace["resolved"] == bool(np.all(t_ref < INF))
    assert set(trace["unresolved_nodes"]) == set(np.flatnonzero(t_ref >= INF).tolist())


def test_sample_hints_shape_and_colors():
    rng = np.random.default_rng(0)
    N, d, q = 48, 5, 4
    adj, colors = planted_regular_graph(N, d, q, rng)
    hints = sample_hints(N, 10, colors, rng)
    assert len(hints) == 10
    for node, c in hints.items():
        assert c == int(colors[node])


def test_round_events_are_sorted_and_nonempty():
    rng = np.random.default_rng(3)
    N, d, q = 60, 5, 4
    adj, colors = planted_regular_graph(N, d, q, rng)
    hints = sample_hints(N, round(0.5 * N), colors, rng)
    trace = solve_with_dd_trace(adj, colors, hints, q)
    assert len(trace["rounds"]) > 0, "rho=0.5 at N=60,d=5,q=4 should take more than one DD round"
    for round_events in trace["rounds"]:
        assert len(round_events) > 0
        assert round_events == sorted(round_events)


def test_too_few_hints_leaves_nodes_unresolved():
    rng = np.random.default_rng(7)
    N, d, q = 48, 5, 4
    adj, colors = planted_regular_graph(N, d, q, rng)
    hints = sample_hints(N, 1, colors, rng)
    trace = solve_with_dd_trace(adj, colors, hints, q)
    assert not trace["resolved"]
    assert len(trace["unresolved_nodes"]) > 0
