"""Correctness checks for spin_gnn.sudoku_dd.

The incremental `remove_hint` update is the thing that has to be right: after
every attempted removal along a random path, the maintained deduction times
must equal a from-scratch DD run (`dd_times`) on the current hint set, and a
removal must be accepted exactly when from-scratch DD still recovers the
whole coloring.
"""
import numpy as np
import pytest

from spin_gnn.sudoku_dd import (
    INF, dd_times, make_workspace, planted_regular_graph, remove_hint, removal_path,
)


@pytest.mark.parametrize("N,d,q", [(48, 3, 4), (96, 5, 4), (120, 7, 4), (60, 4, 3), (90, 6, 3)])
def test_planted_graph_is_simple_regular_and_properly_colored(N, d, q):
    adj, colors = planted_regular_graph(N, d, q, np.random.default_rng(0))
    assert adj.shape == (N, d)
    assert np.all(np.bincount(colors, minlength=q) == N // q)
    for i in range(N):
        assert len(set(adj[i])) == d and i not in adj[i]
        assert np.all(colors[adj[i]] != colors[i])
        for j in adj[i]:
            assert i in adj[j]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("N,d,q", [(96, 5, 4), (120, 7, 4), (60, 4, 3)])
def test_incremental_removal_matches_from_scratch_dd(N, d, q, seed):
    rng = np.random.default_rng(seed)
    adj, colors = planted_regular_graph(N, d, q, rng)
    t = np.zeros(N, dtype=np.int64)
    is_hint = np.ones(N, dtype=bool)
    ws = make_workspace(N, d, q)
    n_accepted = 0
    for v in rng.permutation(N):
        trial = is_hint.copy()
        trial[v] = False
        expect_ok = np.all(dd_times(adj, colors, trial, q) < INF)
        ok, delta, mxnew = remove_hint(v, adj, colors, q, t, is_hint, ws)
        assert ok == expect_ok
        np.testing.assert_array_equal(t, dd_times(adj, colors, is_hint, q))
        n_accepted += ok
    assert n_accepted > 0
    # minimal: no remaining hint is individually removable
    for v in np.flatnonzero(is_hint):
        trial = is_hint.copy()
        trial[v] = False
        assert np.any(dd_times(adj, colors, trial, q) == INF)


def test_removal_path_bookkeeping():
    N, d, q = 1200, 5, 4
    rng = np.random.default_rng(1)
    adj, colors = planted_regular_graph(N, d, q, rng)
    path = removal_path(adj, colors, q, rng, snapshot_rhos=(0.7,))
    t = dd_times(adj, colors, path["final_is_hint"], q)
    np.testing.assert_array_equal(t, path["final_t"])
    assert path["T"][-1] == t.max()
    assert np.isclose(path["m"][-1], t.mean() / t.max())
    assert np.all(np.diff(path["T"]) >= 0)
    assert np.isclose(path["rho"][-1], path["rho_min"])
    assert 0.7 in path["snapshots"]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("N,d,q", [(96, 5, 4), (120, 7, 4)])
def test_mcmc_moves_keep_times_exact(N, d, q, seed):
    from spin_gnn.sudoku_mcmc import mcmc_steps
    rng = np.random.default_rng(seed)
    adj, colors = planted_regular_graph(N, d, q, rng)
    path = removal_path(adj, colors, q, rng)
    t, is_hint = path["final_t"].copy(), path["final_is_hint"].copy()
    ws = make_workspace(N, d, q)
    heap2 = np.zeros(N * d * 8 + 16, dtype=np.int64)
    for mu in (0.0, -0.5, -1.5, 1.0):
        for _ in range(20):
            mcmc_steps(rng.integers(N, size=N // 4), rng.random(N // 4), mu,
                       adj, colors, q, t, is_hint, ws, heap2)
            np.testing.assert_array_equal(t, dd_times(adj, colors, is_hint, q))
