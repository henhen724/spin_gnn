"""Next-token-prediction dataset over spin_gnn.ksat2's event-based token
sequences -- (T,) int64 arrays of categorical token ids, unlike
sat_dataset.py's (T, n_vars+1) real-valued vectors. Padding uses id 0
(BEGIN_HINTS -- never a valid *target*, since nothing ever predicts it, so
reusing it as the pad id can't be confused with a real target) with a
boolean mask marking real (non-pad) positions, same batching strategy as
sat_dataset.py otherwise (bucket-by-length before batching to limit
wasted padding)."""
import numpy as np
import torch


class SATTokenDataset:
    def __init__(self, token_sequences, max_len=None):
        sequences = [s for s in token_sequences if s.shape[0] >= 2]
        dropped_short = len(token_sequences) - len(sequences)
        if max_len:
            n_before = len(sequences)
            sequences = [s for s in sequences if s.shape[0] <= max_len]
            dropped_long = n_before - len(sequences)
        else:
            dropped_long = 0
        self.sequences = sequences
        if dropped_short:
            print(f"SATTokenDataset: dropped {dropped_short}/{len(token_sequences)} instances with < 2 tokens")
        if dropped_long:
            print(
                f"SATTokenDataset: dropped {dropped_long}/{len(token_sequences)} instances "
                f"({dropped_long / len(token_sequences):.1%}) with > {max_len} tokens"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        return seq[:-1], seq[1:]


def make_token_batch(dataset: SATTokenDataset, indices):
    xs, ys = zip(*(dataset[i] for i in indices))
    lengths = [x.shape[0] for x in xs]
    max_len = max(lengths)
    b = len(indices)

    x = torch.zeros(b, max_len, dtype=torch.long)
    y = torch.zeros(b, max_len, dtype=torch.long)
    mask = torch.zeros(b, max_len, dtype=torch.bool)
    for i, (xi, yi, length) in enumerate(zip(xs, ys, lengths)):
        x[i, :length] = xi
        y[i, :length] = yi
        mask[i, :length] = True
    return x, y, mask


def iterate_token_batches(dataset, indices, batch_size, rng=None, shuffle=True):
    indices = list(indices)
    if not shuffle:
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            yield make_token_batch(dataset, indices[start : start + batch_size])
        return

    rng = rng or np.random.default_rng()
    indices.sort(key=lambda i: dataset.sequences[i].shape[0])
    batches = [indices[start : start + batch_size] for start in range(0, len(indices) - batch_size + 1, batch_size)]
    rng.shuffle(batches)
    for batch in batches:
        yield make_token_batch(dataset, batch)
