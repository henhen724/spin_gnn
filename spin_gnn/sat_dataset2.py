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
    def __init__(self, token_sequences, max_len=None, aux=None):
        """aux, when given, is a list parallel to token_sequences of
        (pos, target) pairs -- see spin_gnn.sudoku_determinacy. It is filtered
        in lockstep with the sequences, so a dropped instance never shifts the
        alignment between a sequence and its targets. None reproduces every
        dataset built before stage X exactly."""
        if aux is not None and len(aux) != len(token_sequences):
            raise ValueError(f"aux has {len(aux)} entries for {len(token_sequences)} sequences")
        keep = [i for i, s in enumerate(token_sequences) if s.shape[0] >= 2]
        dropped_short = len(token_sequences) - len(keep)
        if max_len:
            n_before = len(keep)
            keep = [i for i in keep if token_sequences[i].shape[0] <= max_len]
            dropped_long = n_before - len(keep)
        else:
            dropped_long = 0
        sequences = [token_sequences[i] for i in keep]
        self.sequences = sequences
        self.aux = [aux[i] for i in keep] if aux is not None else None
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


def make_token_batch_aux(dataset, indices):
    """make_token_batch plus the determinacy targets for the same instances.

    Returns x, y, mask, aux_pos, aux_target. Instances differ in how many
    deductions their trace holds, so aux_pos is padded with 0 (a position that
    always exists) and aux_target with MASKED, which spin_gnn.sudoku_determinacy
    drops -- a padded row therefore contributes nothing to the loss rather than
    contributing a zero.
    """
    from .sudoku_determinacy import MASKED

    x, y, mask = make_token_batch(dataset, indices)
    entries = [dataset.aux[i] for i in indices]
    n_pos = [p.shape[0] for p, _t in entries]
    max_pos = max(n_pos) if n_pos else 0
    n_nodes = entries[0][1].shape[1]
    b = len(indices)

    aux_pos = torch.zeros(b, max_pos, dtype=torch.long)
    aux_target = torch.full((b, max_pos, n_nodes), MASKED, dtype=torch.int8)
    for i, ((p, t), length) in enumerate(zip(entries, n_pos)):
        aux_pos[i, :length] = torch.from_numpy(p)
        aux_target[i, :length] = torch.from_numpy(t)
    return x, y, mask, aux_pos, aux_target


def iterate_token_batches(dataset, indices, batch_size, rng=None, shuffle=True, collate=make_token_batch):
    """`collate` selects what a batch carries; the bucketing and the shuffle
    above it are untouched by the choice, so a determinacy run sees exactly the
    batch order its plain counterpart would at the same seed."""
    indices = list(indices)
    if not shuffle:
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            yield collate(dataset, indices[start : start + batch_size])
        return

    rng = rng or np.random.default_rng()
    indices.sort(key=lambda i: dataset.sequences[i].shape[0])
    batches = [indices[start : start + batch_size] for start in range(0, len(indices) - batch_size + 1, batch_size)]
    rng.shuffle(batches)
    for batch in batches:
        yield collate(dataset, batch)
