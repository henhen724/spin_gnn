"""Direct-deduction (DD) "Sudoku" on random regular graphs with a planted
q-coloring -- the minimal model of deductive reasoning depth from
"A Transition in Reasoning Depth in Sudoku on Random Graphs".

Model:
  * G = (V, E) a random d-regular graph on N nodes with a planted proper
    q-coloring x* (built together: the coloring is drawn first, balanced,
    and the graph is a uniformly-paired configuration model restricted to
    edges between different color classes -- see planted_regular_graph).
  * A hint set V_hint ⊆ V has its colors revealed.
  * DD ("naked single"): a non-hint node is deduced one round after its
    already-deduced neighbors show all q-1 colors other than its own.
    Hints are deduced at round 0, so the deduction time of node i is the
    least fixed point of

        t_i = 0                                      (i a hint)
        t_i = 1 + max_{a != x_i} min_{j in di, x_j = a} t_j   (otherwise)

    and T = max_i t_i is the number of sequential DD rounds (the "reasoning
    depth"), m = (1/N) sum_i t_i / T the order parameter.

Sampling protocol (slides' numerical methodology): start from the fully
revealed coloring and visit nodes in a uniformly random order, removing each
node's hint iff DD still recovers the full coloring without it. Recording
(rho = |V_hint|/N, T, m) after every accepted removal gives one sample path
from rho = 1 down to a minimal hint set (no single hint removable -- since
the DD closure is monotone in the hint set, a node that can't be removed now
can never be removed later, so one pass suffices).

Re-running DD from scratch after each removal costs O(N) per check and
O(N^2) per path. Instead `remove_hint` updates the deduction times
incrementally (Ramalingam-Reps style for a superior function): removing a
hint can only *increase* times, a node's time is justified only by
neighbors with strictly smaller time (so justification is acyclic), and the
update touches only the nodes whose justification actually broke. This makes
paths at N ~ 10^5 cost seconds. `dd_times` is the from-scratch reference
that tests/test_sudoku_dd.py checks the incremental update against.
"""
import numpy as np
import numba as nb

INF = np.int64(1) << 40


# ---------------------------------------------------------------------------
# Graph + planted coloring
# ---------------------------------------------------------------------------

def planted_regular_graph(n_nodes, degree, n_colors, rng, max_fix_iters=100_000):
    """Random d-regular graph with a planted balanced proper q-coloring.

    The coloring puts exactly N/q nodes in each class. Each class's d*N/q
    edge stubs are split evenly across the other q-1 classes, so every pair
    of classes shares exactly d*N/(q(q-1)) edges, and stubs within a class
    pair are matched uniformly at random. Multi-edges are then removed by
    random endpoint switches within the same class pair (which preserve both
    degrees and the coloring). Requires q(q-1) | d*N and q | N -- e.g. the
    slides' N values for q=4 are all multiples of 12.

    Returns (adj, colors): adj is an (N, d) int32 neighbor table, colors an
    (N,) int8 array, both with node labels shuffled so color is not
    correlated with index.
    """
    N, d, q = n_nodes, degree, n_colors
    if N % q or (d * N) % (q * (q - 1)):
        raise ValueError(f"need q | N and q(q-1) | d*N (N={N}, d={d}, q={q})")
    per_class = N // q
    s = d * N // (q * (q - 1))  # edges per class pair
    colors = np.repeat(np.arange(q, dtype=np.int8), per_class)
    # groups[a][b]: stubs of class a destined for class b
    groups = [[None] * q for _ in range(q)]
    for a in range(q):
        stubs = rng.permutation(np.repeat(np.arange(a * per_class, (a + 1) * per_class), d))
        others = [b for b in range(q) if b != a]
        for i, b in enumerate(others):
            groups[a][b] = stubs[i * s:(i + 1) * s]
    edge_blocks = []
    for a in range(q):
        for b in range(a + 1, q):
            u = groups[a][b].copy()
            v = rng.permutation(groups[b][a])
            _remove_multi_edges(u, v, rng, max_fix_iters)
            edge_blocks.append(np.stack([u, v], axis=1))
    edges = np.concatenate(edge_blocks)

    # relabel nodes so color isn't a function of index
    perm = rng.permutation(N)
    edges = perm[edges]
    colors_new = np.empty_like(colors)
    colors_new[perm] = colors

    adj = np.empty((N, d), dtype=np.int32)
    fill = np.zeros(N, dtype=np.int64)
    for x, y in edges:
        adj[x, fill[x]] = y
        fill[x] += 1
        adj[y, fill[y]] = x
        fill[y] += 1
    assert np.all(fill == d)
    return adj, colors_new


def _remove_multi_edges(u, v, rng, max_iters):
    """In-place: switch v-endpoints between random edges of one bipartite
    class-pair block until no (u, v) pair repeats."""
    n = len(u)
    for _ in range(max_iters):
        keys = u.astype(np.int64) * (1 << 32) + v
        order = np.argsort(keys, kind="stable")
        dup = order[1:][keys[order[1:]] == keys[order[:-1]]]
        if dup.size == 0:
            return
        for i in dup:
            j = rng.integers(n)
            v[i], v[j] = v[j], v[i]
    raise RuntimeError("could not remove multi-edges")


# ---------------------------------------------------------------------------
# DD from scratch (reference)
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def dd_times(adj, colors, is_hint, n_colors):
    """Synchronous DD from scratch. Returns int64 times (INF if never
    deduced). Frontier-based: a deduced node pushes its color bit to its
    undeduced neighbors; a node whose seen-color mask reaches q-1 bits is
    deduced next round."""
    N, d = adj.shape
    t = np.full(N, INF, dtype=np.int64)
    seen = np.zeros(N, dtype=np.int64)
    nseen = np.zeros(N, dtype=np.int64)
    frontier = np.empty(N, dtype=np.int64)
    nxt = np.empty(N, dtype=np.int64)
    nf = 0
    for i in range(N):
        if is_hint[i]:
            t[i] = 0
            frontier[nf] = i
            nf += 1
    r = 0
    while nf > 0:
        nn = 0
        for k in range(nf):
            w = frontier[k]
            bit = np.int64(1) << colors[w]
            for j in range(d):
                u = adj[w, j]
                if t[u] != INF:
                    continue
                if (seen[u] & bit) == 0:
                    seen[u] |= bit
                    nseen[u] += 1
                    if nseen[u] == n_colors - 1:
                        t[u] = r + 1
                        nxt[nn] = u
                        nn += 1
        for k in range(nn):
            frontier[k] = nxt[k]
        nf = nn
        r += 1
    return t


# ---------------------------------------------------------------------------
# Incremental hint removal
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _heap_push(h, n, key):
    h[n] = key
    i = n
    while i > 0:
        p = (i - 1) >> 1
        if h[p] <= h[i]:
            break
        h[p], h[i] = h[i], h[p]
        i = p
    return n + 1


@nb.njit(cache=True)
def _heap_pop(h, n):
    top = h[0]
    n -= 1
    h[0] = h[n]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= n:
            break
        c = l
        if l + 1 < n and h[l + 1] < h[l]:
            c = l + 1
        if h[i] <= h[c]:
            break
        h[i], h[c] = h[c], h[i]
        i = c
    return top, n


@nb.njit(cache=True)
def remove_hint(v, adj, colors, n_colors, t, is_hint, ws):
    """Try to remove hint v. On success, updates t/is_hint in place and
    returns (True, delta_sum_t, max_new_t). On failure (DD would no longer
    recover the coloring) leaves state untouched and returns (False, 0, 0).

    ws is a workspace tuple from make_workspace (reused across calls so
    nothing O(N) is allocated or reset per call).
    """
    aff, pos, aff_list, old_t, heap, f, tent, done = ws
    N, d = adj.shape
    q = n_colors
    keymul = np.int64(N)

    # ---- phase A: find nodes whose current time lost its justification,
    # in increasing order of old time (justifiers have strictly smaller t).
    is_hint[v] = False
    n_aff = 0
    aff[v] = 1
    pos[v] = n_aff
    aff_list[n_aff] = v
    old_t[n_aff] = t[v]
    n_aff += 1
    hn = 0
    for j in range(d):
        u = adj[v, j]
        if not is_hint[u]:
            hn = _heap_push(heap, hn, t[u] * keymul + u)
    while hn > 0:
        key, hn = _heap_pop(heap, hn)
        u = key % keymul
        if aff[u]:
            continue
        tu = t[u]
        mask = np.int64(0)
        cnt = 0
        for j in range(d):
            w = adj[u, j]
            if aff[w] or t[w] >= tu:
                continue
            bit = np.int64(1) << colors[w]
            if (mask & bit) == 0:
                mask |= bit
                cnt += 1
        if cnt >= q - 1:
            continue
        aff[u] = 1
        pos[u] = n_aff
        aff_list[n_aff] = u
        old_t[n_aff] = tu
        n_aff += 1
        for j in range(d):
            y = adj[u, j]
            if not is_hint[y] and not aff[y] and t[y] > tu:
                hn = _heap_push(heap, hn, t[y] * keymul + y)

    # ---- phase B: recompute affected times, Dijkstra-like in increasing
    # new time, seeded from the unaffected neighbors (whose times are final).
    hn = 0
    for k in range(n_aff):
        u = aff_list[k]
        done[k] = 0
        cu = colors[u]
        for a in range(q):
            f[k * q + a] = INF
        for j in range(d):
            w = adj[u, j]
            if aff[w]:
                continue
            a = colors[w]
            if t[w] < f[k * q + a]:
                f[k * q + a] = t[w]
        mx = np.int64(0)
        for a in range(q):
            if a != cu and f[k * q + a] > mx:
                mx = f[k * q + a]
        tent[k] = INF if mx >= INF else mx + 1
        if tent[k] < INF:
            hn = _heap_push(heap, hn, tent[k] * keymul + u)
    n_done = 0
    while hn > 0:
        key, hn = _heap_pop(heap, hn)
        u = key % keymul
        k = pos[u]
        tv = key // keymul
        if done[k] or tv != tent[k]:
            continue
        done[k] = 1
        n_done += 1
        cu = colors[u]
        for j in range(d):
            y = adj[u, j]
            if not aff[y]:
                continue
            ky = pos[y]
            if done[ky]:
                continue
            idx = ky * q + cu
            if tv < f[idx]:
                f[idx] = tv
                cy = colors[y]
                mx = np.int64(0)
                for a in range(q):
                    if a != cy and f[ky * q + a] > mx:
                        mx = f[ky * q + a]
                nt = INF if mx >= INF else mx + 1
                if nt < tent[ky]:
                    tent[ky] = nt
                    hn = _heap_push(heap, hn, nt * keymul + y)

    ok = n_done == n_aff
    delta = np.int64(0)
    mxnew = np.int64(0)
    for k in range(n_aff):
        u = aff_list[k]
        aff[u] = 0
        if ok:
            delta += tent[k] - old_t[k]
            t[u] = tent[k]
            if tent[k] > mxnew:
                mxnew = tent[k]
    if not ok:
        is_hint[v] = True
        return False, np.int64(0), np.int64(0)
    return True, delta, mxnew


def make_workspace(n_nodes, degree, n_colors):
    N = n_nodes
    return (
        np.zeros(N, dtype=np.int8),          # aff
        np.zeros(N, dtype=np.int64),         # pos
        np.zeros(N, dtype=np.int64),         # aff_list
        np.zeros(N, dtype=np.int64),         # old_t
        np.zeros(N * degree + 1, dtype=np.int64),  # heap (each node pushed <= d+1 times)
        np.zeros(N * n_colors, dtype=np.int64),    # f
        np.zeros(N, dtype=np.int64),         # tent
        np.zeros(N, dtype=np.int8),          # done
    )


@nb.njit(cache=True)
def _run_removal(order, start, stop_n_hints, adj, colors, n_colors, t, is_hint,
                 n_hints, sum_t, T, ws, rec_nh, rec_T, rec_sum, n_rec):
    """Walk `order` from index `start`, attempting removals, until the order
    is exhausted or n_hints reaches stop_n_hints. Records (n_hints, T,
    sum_t) after every accepted removal."""
    i = start
    while i < order.shape[0] and n_hints > stop_n_hints:
        v = order[i]
        i += 1
        ok, delta, mxnew = remove_hint(v, adj, colors, n_colors, t, is_hint, ws)
        if ok:
            n_hints -= 1
            sum_t += delta
            if mxnew > T:
                T = mxnew
            rec_nh[n_rec] = n_hints
            rec_T[n_rec] = T
            rec_sum[n_rec] = sum_t
            n_rec += 1
    return i, n_hints, sum_t, T, n_rec


def removal_path(adj, colors, n_colors, rng, snapshot_rhos=()):
    """One greedy random-order hint-removal path from rho=1 to a minimal
    hint set.

    Returns a dict with per-accepted-removal arrays `rho`, `T`, `m`
    (m = mean_i t_i / T), the final minimal density `rho_min`, and
    `snapshots`: {rho: t-array} copies of the full deduction-time vector at
    the first point the path reaches each requested density (for deduction
    time distributions / per-round deduction counts).
    """
    N, d = adj.shape
    t = np.zeros(N, dtype=np.int64)
    is_hint = np.ones(N, dtype=np.bool_)
    ws = make_workspace(N, d, n_colors)
    order = rng.permutation(N).astype(np.int64)
    rec_nh = np.empty(N, dtype=np.int64)
    rec_T = np.empty(N, dtype=np.int64)
    rec_sum = np.empty(N, dtype=np.int64)
    i, n_hints, sum_t, T, n_rec = 0, N, 0, 0, 0
    snapshots = {}
    for rho in sorted(snapshot_rhos, reverse=True):
        stop = int(np.floor(rho * N))
        i, n_hints, sum_t, T, n_rec = _run_removal(
            order, i, stop, adj, colors, n_colors, t, is_hint,
            n_hints, sum_t, T, ws, rec_nh, rec_T, rec_sum, n_rec)
        if n_hints <= stop:
            snapshots[rho] = t.copy()
    i, n_hints, sum_t, T, n_rec = _run_removal(
        order, i, -1, adj, colors, n_colors, t, is_hint,
        n_hints, sum_t, T, ws, rec_nh, rec_T, rec_sum, n_rec)
    nh, Tr, sr = rec_nh[:n_rec], rec_T[:n_rec], rec_sum[:n_rec]
    return {
        "rho": nh / N,
        "T": Tr,
        "m": sr / (N * np.maximum(Tr, 1)),
        "rho_min": n_hints / N,
        "snapshots": snapshots,
        "final_t": t,
        "final_is_hint": is_hint,
    }


def on_grid(path, rho_grid):
    """Sample a removal path at each grid density: state at the first
    accepted removal with rho <= grid value (NaN below the path's rho_min)."""
    rho = path["rho"]  # decreasing
    T = np.full(len(rho_grid), np.nan)
    m = np.full(len(rho_grid), np.nan)
    for k, r in enumerate(rho_grid):
        idx = np.searchsorted(-rho, -r, side="left")
        if idx < len(rho):
            T[k] = path["T"][idx]
            m[k] = path["m"][idx]
    return T, m
