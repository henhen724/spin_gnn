"""Metropolis sampling of solvable hint sets at fixed planted coloring, under
the tilted measure  P(V_hint) ∝ e^{mu |V_hint|} * 1[DD recovers x*]  --
the T = infinity version of the slides' non-planted ensemble (uniform over
DD-solvable hint sets, with a chemical potential mu on the hint count).

Random-order greedy removal (sudoku_dd.removal_path) gets stuck at a
minimal hint set whose density rho_min sits near rho_c for d=5, so it can't
see the slow phase there. Annealing mu downward lets the sampler trade hints
around (remove one, add another elsewhere) and reach lower densities.

Move: pick a node uniformly. If it's a hint, propose removing it (accepted
with prob min(1, e^{-mu}) *and* only if DD still recovers the coloring --
remove_hint checks and reverts). Otherwise propose adding it (accepted with
prob min(1, e^{mu}); always solvable). Symmetric proposal -> detailed
balance w.r.t. the measure above.

`add_hint` is the decreasing counterpart of remove_hint: label-correcting
relaxation from v. It converges to the right (least) fixed point because a
finite-valued fixed point of the DD time equations is unique (each node's
time is justified by strictly smaller neighbors, so induction on time pins
it down), and starting from the old times we only ever move down onto one.
"""
import argparse
import os

import numba as nb
import numpy as np

from .sudoku_dd import (_heap_pop, _heap_push, make_workspace,
                        planted_regular_graph, removal_path, remove_hint)


@nb.njit(cache=True)
def add_hint(v, adj, colors, n_colors, t, is_hint, heap):
    N, d = adj.shape
    q = n_colors
    keymul = np.int64(N)
    is_hint[v] = True
    if t[v] == 0:
        return
    t[v] = 0
    hn = 0
    for j in range(d):
        u = adj[v, j]
        if not is_hint[u]:
            hn = _heap_push(heap, hn, u)  # key = node; value read fresh
    while hn > 0:
        u, hn = _heap_pop(heap, hn)
        u = u % keymul
        if is_hint[u]:
            continue
        cu = colors[u]
        mx = np.int64(0)
        for a in range(q):
            if a == cu:
                continue
            mn = np.int64(1) << 40
            for j in range(d):
                w = adj[u, j]
                if colors[w] == a and t[w] < mn:
                    mn = t[w]
            if mn > mx:
                mx = mn
        nv = mx + 1
        if nv < t[u]:
            t[u] = nv
            for j in range(d):
                y = adj[u, j]
                if not is_hint[y] and t[y] > nv + 1:
                    if hn >= heap.shape[0]:
                        # heap full: compact by restarting is safe (values
                        # only decrease), but never observed in practice
                        raise RuntimeError("add_hint heap overflow")
                    hn = _heap_push(heap, hn, nv * keymul + y)


@nb.njit(cache=True)
def mcmc_steps(nodes, uniforms, log_fug, adj, colors, n_colors, t, is_hint, ws, heap2):
    """Run len(nodes) Metropolis moves. Returns net change in hint count."""
    acc_rm = np.exp(-log_fug) if log_fug > 0 else 1.0
    acc_add = np.exp(log_fug) if log_fug < 0 else 1.0
    dn = 0
    for k in range(nodes.shape[0]):
        v = nodes[k]
        if is_hint[v]:
            if uniforms[k] < acc_rm:
                ok, _, _ = remove_hint(v, adj, colors, n_colors, t, is_hint, ws)
                if ok:
                    dn -= 1
        else:
            if uniforms[k] < acc_add:
                add_hint(v, adj, colors, n_colors, t, is_hint, heap2)
                dn += 1
    return dn


def anneal(adj, colors, n_colors, rng, mus, sweeps_per_mu, start="greedy"):
    """Anneal mu through `mus`, running sweeps_per_mu sweeps (N moves each)
    at every value. Starts from the greedy minimal hint set (start="greedy")
    or the full coloring (start="full"). Returns per-sweep records."""
    N, d = adj.shape
    if start == "greedy":
        p = removal_path(adj, colors, n_colors, rng)
        t, is_hint = p["final_t"].copy(), p["final_is_hint"].copy()
    else:
        t, is_hint = np.zeros(N, dtype=np.int64), np.ones(N, dtype=bool)
    ws = make_workspace(N, d, n_colors)
    heap2 = np.zeros(N * d * 8 + 16, dtype=np.int64)
    rec = {"mu": [], "rho": [], "T": [], "m": []}
    for mu in mus:
        for _ in range(sweeps_per_mu):
            mcmc_steps(rng.integers(N, size=N), rng.random(N), mu,
                       adj, colors, n_colors, t, is_hint, ws, heap2)
            T = t.max()
            rec["mu"].append(mu)
            rec["rho"].append(is_hint.mean())
            rec["T"].append(T)
            rec["m"].append(t.mean() / max(T, 1))
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["final_t"] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--N", type=int, default=4140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu-min", type=float, default=-3.0)
    ap.add_argument("--n-mu", type=int, default=31)
    ap.add_argument("--sweeps-per-mu", type=int, default=40)
    ap.add_argument("--start", default="greedy", choices=["greedy", "full"])
    ap.add_argument("--out-dir", default="sudoku_mcmc_results")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng([a.q, a.d, a.N, a.seed, 7])
    adj, colors = planted_regular_graph(a.N, a.d, a.q, rng)
    mus = np.linspace(0.0, a.mu_min, a.n_mu)
    r = anneal(adj, colors, a.q, rng, mus, a.sweeps_per_mu, a.start)
    path = os.path.join(a.out_dir, f"q{a.q}_d{a.d}_N{a.N}_s{a.seed}_{a.start}.npz")
    np.savez_compressed(path, q=a.q, d=a.d, N=a.N, seed=a.seed, **r)
    print(path)


if __name__ == "__main__":
    main()
