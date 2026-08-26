"""Sliding-window next-spin-prediction dataset.

Within one disorder realization the coupling graph is fixed (quenched), so a
minibatch of time-windows drawn from the *same* realization can share one
copy of (edge_index, edge_attr) tiled `batch_size` times -- this avoids
duplicating a potentially large edge list once per window, which is what
naively wrapping every window in its own torch_geometric.data.Data object
would do. Training across several independent realizations is handled by
sampling one realization per step and batching within it (see train.py);
graphs are never mixed within a batch.
"""
import numpy as np
import torch

from .graph import SpinGraph


def spin_to_onehot(s):
    onehot = np.zeros(s.shape + (2,), dtype=np.float32)
    onehot[..., 0] = s < 0
    onehot[..., 1] = s > 0
    return onehot


class SpinTrajectoryWindows:
    """One realization's trajectory, sliced into K-lag context windows."""

    def __init__(self, realization, spin_graph: SpinGraph):
        self.s = realization["S"]
        self.p = realization["P"]
        self.k = spin_graph.context_length
        self.spin_graph = spin_graph
        assert self.s.shape[1] == spin_graph.n_spins
        self.edge_index = torch.from_numpy(spin_graph.edge_index)
        self.edge_attr = torch.from_numpy(spin_graph.edge_attr)

    def __len__(self):
        return self.s.shape[0] - self.k

    def __getitem__(self, idx):
        t = idx + self.k
        history = self.s[t - self.k : t]  # (K, N), oldest first
        x = torch.from_numpy(np.transpose(spin_to_onehot(history), (1, 0, 2)))  # (N, K, 2)
        y = torch.from_numpy((self.s[t] > 0).astype(np.int64))
        p_true = torch.from_numpy(self.p[t].astype(np.float32))
        return x, y, p_true

    def time_split(self, val_fraction):
        n = len(self)
        n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
        split = n - n_val
        return list(range(0, split)), list(range(split, n))


def tile_graph_for_batch(edge_index, edge_attr, n_nodes, batch_size):
    offsets = (torch.arange(batch_size) * n_nodes).repeat_interleave(edge_index.shape[1])
    tiled_index = edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)
    tiled_attr = edge_attr.repeat(batch_size, 1)
    return tiled_index, tiled_attr


def make_batch(dataset: SpinTrajectoryWindows, indices):
    xs, ys, ps = zip(*(dataset[i] for i in indices))
    b = len(indices)
    n = dataset.spin_graph.n_spins
    x = torch.cat(xs, dim=0)  # (B*N, K, 2)
    y = torch.cat(ys, dim=0)  # (B*N,)
    p_true = torch.cat(ps, dim=0)  # (B*N,)
    edge_index, edge_attr = tile_graph_for_batch(dataset.edge_index, dataset.edge_attr, n, b)
    return x, edge_index, edge_attr, y, p_true


def iterate_batches(dataset, indices, batch_size, rng=None, shuffle=True):
    indices = list(indices)
    if shuffle:
        rng = rng or np.random.default_rng()
        rng.shuffle(indices)
    for start in range(0, len(indices) - batch_size + 1, batch_size):
        yield make_batch(dataset, indices[start : start + batch_size])
