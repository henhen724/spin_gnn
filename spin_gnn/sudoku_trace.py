"""DD ("naked single") solver instrumented for an event-based reasoning-trace
token encoding, mirroring spin_gnn.ksat2's "events, not full snapshots"
philosophy -- see sudoku_dd.py's module docstring for the underlying model
(random d-regular graph, planted q-coloring, hints, deduction time t_i).

Unlike sudoku_dd.dd_times (which only returns final deduction *times*, built
for the large-N phase-transition sweep and numba-jitted for speed at that
scale), solve_with_dd_trace replays the same synchronous frontier algorithm
in plain numpy and records, for every DD round, exactly which non-hint
nodes got deduced and what color -- the "reasoning trace" a transformer is
asked to reproduce. This is deliberately the small-N regime (tens to a few
hundred nodes): the point here is learnability of the *process*, the same
scale ksat2.py's SATReasoningConfig uses for k-SAT, not the large-N
phase-transition statistics sudoku_sweep.py targets.

Round boundaries are recorded explicitly (one BEGIN_ROUND token per round in
sudoku_tokens.encode_instance) rather than left for the model to infer from
token position or count -- the same reason ksat2.py tags
SATISFIED/REACTIVATED explicitly instead of expecting the model to re-derive
clause status by attention alone.

DD has no search/backtracking (monotone forward propagation only), so unlike
ksat2 there is no decision/flip/unassign event kind -- every event is a
forced deduction, and hints are always consistent with the planted coloring
(there is no such thing as a "wrong" hint, unlike ksat2's independently-
sampled +-1 hints). If some nodes never get deduced (hints too sparse),
that's reported as "unresolved" -- the direct analogue of ksat2's UNSAT.
"""
import numpy as np


def sample_hints(n_nodes, n_hints, colors, rng):
    """n_hints distinct nodes, each revealing its true planted color --
    unlike ksat2.sample_hints, a Sudoku hint is a revealed fact about a
    fixed ground truth, not an independently sampled constraint, so there's
    no separate value to draw."""
    if n_hints > n_nodes:
        raise ValueError(f"n_hints={n_hints} must be <= n_nodes={n_nodes}")
    nodes = rng.choice(n_nodes, size=n_hints, replace=False)
    return {int(v): int(colors[v]) for v in nodes}


def solve_with_dd_trace(adj, colors, hints, n_colors):
    """adj: (N, d) int neighbor table. colors: (N,) int8 ground-truth
    planted coloring. hints: dict node -> color (must equal colors[node]).

    Returns a dict:
      rounds:            list of per-round events, each round a sorted list
                          of (node, color) tuples, round 1 first.
      hints:              echoed back.
      resolved:           bool, True iff every node got deduced.
      unresolved_nodes:   sorted list of nodes never deduced (empty if resolved).
    """
    N, d = adj.shape
    is_deduced = np.zeros(N, dtype=bool)
    deduced_color = {}
    for node, c in hints.items():
        if c != int(colors[node]):
            raise ValueError(f"hint at node {node} ({c}) disagrees with planted color {colors[node]}")
        is_deduced[node] = True
        deduced_color[node] = int(c)

    seen = np.zeros(N, dtype=np.int64)  # bitmask of neighbor colors already observed
    nseen = np.zeros(N, dtype=np.int64)
    frontier = list(hints.keys())
    rounds = []
    while frontier:
        nxt = []
        for w in frontier:
            bit = 1 << int(colors[w])
            for u in adj[w]:
                u = int(u)
                if is_deduced[u]:
                    continue
                if not (seen[u] & bit):
                    seen[u] |= bit
                    nseen[u] += 1
                    if nseen[u] == n_colors - 1:
                        is_deduced[u] = True
                        deduced_color[u] = int(colors[u])
                        nxt.append(u)
        if nxt:
            rounds.append(sorted((node, deduced_color[node]) for node in nxt))
        frontier = nxt

    unresolved_nodes = sorted(int(i) for i in np.flatnonzero(~is_deduced))
    return {
        "rounds": rounds,
        "hints": dict(hints),
        "resolved": len(unresolved_nodes) == 0,
        "unresolved_nodes": unresolved_nodes,
    }
