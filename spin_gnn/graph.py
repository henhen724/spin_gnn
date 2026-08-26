"""Graph structure for the disordered kinetic Ising model, and random
sparsification of the (physically fully-connected) coupling graph.

The true model couples every pair of sites through J_k for each of the K
lags -- an O(N^2 K) dense graph. `sparsify_graph` keeps each directed edge
independently with probability `keep_prob`, giving a random Erdos-Renyi-style
support on top of the quenched couplings. This is the object of the
"random sparsification" study: how much of the exact local field / next-spin
posterior (spin_gnn.simulate.local_field) can a message-passing GNN recover
from a sparsified view of the disorder, and does compensating surviving
edges by 1/keep_prob (preserving E[edge weight] under the sparsification
ensemble, the usual dilute-spin-glass / inverse-propensity trick) help.

Edge convention: edge (source=j, target=i) carries edge_attr = J[i, j, :]
(shape (K,)), i.e. message flows from the site holding the past state s_{t-lag}
towards the site whose field it contributes to -- matching PyG's default
source_to_target message flow.

edge_attr's K axis is stored *oldest-lag-first* (edge_attr[e, 0] = J[i,j] for
lag K, edge_attr[e, K-1] = J[i,j] for lag 1) to match the oldest-first
convention of the context window built in dataset.py/simulate.py (history[0]
= s_{t-K}, history[K-1] = s_{t-1}). J's own last axis is lag-1-first (J[...,0]
= lag 1, from simulate.local_field), so it's reversed here -- model.py's
LagCoupledInputConv.message contracts x_j's and edge_attr's K axes together
elementwise and silently misassigns which coupling multiplies which
historical spin if the two orderings don't match.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SpinGraph:
    edge_index: np.ndarray  # (2, E) int64, [source, target]
    edge_attr: np.ndarray  # (E, K) float32
    n_spins: int
    context_length: int
    keep_prob: float


def sparsify_graph(j, keep_prob=1.0, compensate=True, rng=None):
    """j: (N, N, K) coupling tensor. Returns (edge_index, edge_attr).

    Dense O(N^2) mask -- fine up to a few thousand nodes; for much larger N,
    sample edges directly rather than materializing an N x N mask.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = j.shape[0]
    if keep_prob >= 1.0:
        mask = ~np.eye(n, dtype=bool)
    else:
        mask = rng.random((n, n)) < keep_prob
        np.fill_diagonal(mask, False)

    target_idx, source_idx = np.nonzero(mask)  # j[i, j_src, :] -> row=target, col=source
    edge_attr = j[target_idx, source_idx, ::-1].astype(np.float32)  # lag-1-first -> oldest-first
    if compensate and keep_prob < 1.0:
        edge_attr = edge_attr / keep_prob
    edge_index = np.stack([source_idx, target_idx], axis=0).astype(np.int64)
    return edge_index, edge_attr


def build_spin_graph(j, context_length, keep_prob=1.0, compensate=True, seed=None):
    rng = np.random.default_rng(seed)
    edge_index, edge_attr = sparsify_graph(j, keep_prob=keep_prob, compensate=compensate, rng=rng)
    return SpinGraph(
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_spins=j.shape[0],
        context_length=context_length,
        keep_prob=keep_prob,
    )
