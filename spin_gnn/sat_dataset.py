"""Next-token-prediction dataset over spin_gnn.ksat solver traces.

Unlike spin_gnn.dataset's fixed-K-lag windows (all the same length by
construction), each k-SAT instance's solve_with_trace token sequence has a
different length (it depends on how much unit propagation/backtracking that
particular random instance needed) -- so batching here pads to the batch's
max length with an all-zero pad token and carries a boolean mask marking
which target positions are real (used to exclude padding from the loss and
from attention), rather than tiling a shared fixed-shape structure like
dataset.py's graph batching does.
"""
import numpy as np
import torch


class SATTraceDataset:
    """One entry per k-SAT instance: its full solve_with_trace token
    sequence, an (T_i, n_vars+1) float32 array. Instances with fewer than 2
    trace tokens (nothing to predict) are dropped.

    max_tokens (optional) also drops instances whose trace is longer than
    that many tokens. Self-attention is O(L^2) in sequence length, and trace
    length is heavy-tailed (a hard, deep-backtracking UNSAT instance can run
    an order of magnitude longer than a typical SAT one) -- without a cap,
    a large enough instance pool is near-certain to include an outlier long
    enough to blow up batch memory (every batch pads to its longest member).
    This does skew the training distribution away from the hardest instances;
    report the drop count/fraction alongside any result trained with a cap."""

    def __init__(self, traces, n_vars, max_tokens=None):
        self.n_vars = n_vars
        sequences = [t["tokens"] for t in traces if t["tokens"].shape[0] >= 2]
        dropped_short = len(traces) - len(sequences)
        if max_tokens is not None:
            n_before = len(sequences)
            sequences = [s for s in sequences if s.shape[0] <= max_tokens]
            dropped_long = n_before - len(sequences)
        else:
            dropped_long = 0
        self.sequences = sequences
        if dropped_short:
            print(f"SATTraceDataset: dropped {dropped_short}/{len(traces)} instances with < 2 trace tokens")
        if dropped_long:
            print(
                f"SATTraceDataset: dropped {dropped_long}/{len(traces)} instances "
                f"({dropped_long / len(traces):.1%}) with > {max_tokens} trace tokens"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        return seq[:-1], seq[1:]  # (T-1, n+1) input, (T-1, n+1) target


def make_sat_batch(dataset: SATTraceDataset, indices):
    xs, ys = zip(*(dataset[i] for i in indices))
    lengths = [x.shape[0] for x in xs]
    max_len = max(lengths)
    b, dim = len(indices), dataset.n_vars + 1

    x = torch.zeros(b, max_len, dim)
    y = torch.zeros(b, max_len, dim)
    mask = torch.zeros(b, max_len, dtype=torch.bool)  # True = real (non-pad) position
    for i, (xi, yi, length) in enumerate(zip(xs, ys, lengths)):
        x[i, :length] = xi
        y[i, :length] = yi
        mask[i, :length] = True
    return x, y, mask


def iterate_sat_batches(dataset, indices, batch_size, rng=None, shuffle=True):
    indices = list(indices)
    if not shuffle:
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            yield make_sat_batch(dataset, indices[start : start + batch_size])
        return

    # Bucket by length before batching, rather than batching in fully random
    # order: trace length varies wildly (hundreds to tens of thousands of
    # tokens), and random batching routinely pads a short sequence up to
    # whatever the longest one in its batch happens to be, wasting a lot of
    # compute/memory. Sort by length, chunk into batches, then shuffle the
    # *order* of batches so training still sees a different sequence each
    # epoch -- batch composition itself becomes deterministic given the
    # dataset, which is the tradeoff that buys the padding savings.
    rng = rng or np.random.default_rng()
    indices.sort(key=lambda i: dataset.sequences[i].shape[0])
    batches = [indices[start : start + batch_size] for start in range(0, len(indices) - batch_size + 1, batch_size)]
    rng.shuffle(batches)
    for batch in batches:
        yield make_sat_batch(dataset, batch)
