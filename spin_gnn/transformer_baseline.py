"""Standard single-head Transformer baseline for the same self-supervised
next-spin prediction task as SpinGNN (spin_gnn.model), for direct comparison.

The key difference from SpinGNN: this model gets NO access to the true
coupling graph J -- it only ever sees the raw spin trajectory (the K-lag
one-hot history of every spin, jointly, at each window) and must discover any
useful structure purely from data via self-attention. SpinGNN, by contrast, is
handed the (possibly sparsified) coupling graph as edges. Comparing the two
isolates the value of that privileged structural knowledge: does attention
over spin *values* alone recover comparable prediction quality?

Architecture: torch.nn.TransformerEncoder built from the standard
TransformerEncoderLayer (multi-head attention + feedforward, each with
residual + post-LayerNorm -- the original "standard" block) with nhead=1,
applied over the K-length context window. Each of the K "positions" is the
one-hot configuration of ALL N spins at that lag, flattened to a single
2N-dim vector and linearly embedded; a learned positional embedding
distinguishes the K lags (attention alone is permutation-invariant over
them). The final position's representation is read out through a linear head
predicting the next N-spin configuration jointly. There's no causal masking:
with a fixed K-length context predicting a single token *after* it, the last
position already attends to the whole window under a causal mask anyway, so
masking would be a no-op here -- see this module's tests for the (unmasked)
implementation.

Parameter matching: SpinGNN's parameter count depends on context_length (its
readin layer scales with K), so "equal parameters" is tuned per K via
`find_matching_d_model`, which constructs both models and counts actual
parameters rather than using a hand-derived formula.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import SpinGNN
from .train import bayes_optimal_nll, build_windows


class SpinTransformer(nn.Module):
    def __init__(self, context_length, n_spins, d_model, num_layers=2, dim_feedforward=None):
        super().__init__()
        self.context_length = context_length
        self.n_spins = n_spins
        self.input_embed = nn.Linear(2 * n_spins, d_model)
        self.pos_embed = nn.Parameter(0.02 * torch.randn(context_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=1,
            dim_feedforward=dim_feedforward or 4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.readout = nn.Linear(d_model, 2 * n_spins)

    def forward(self, x):
        """x: (B, K, N, 2) one-hot history, oldest-first (same convention as
        SpinGNN's input, just not yet transposed to the graph's node axis).
        Returns logits: (B, N, 2)."""
        b, k, n, _ = x.shape
        seq = x.reshape(b, k, n * 2)
        h = self.input_embed(seq) + self.pos_embed
        h = self.encoder(h)
        return self.readout(h[:, -1, :]).view(b, n, 2)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def gnn_param_count(context_length, hidden_dim=32, num_layers=2):
    return count_params(SpinGNN(context_length, hidden_dim=hidden_dim, num_layers=num_layers))


def find_matching_d_model(target_params, context_length, n_spins, num_layers=2, max_d_model=1024):
    """Linear scan (param count is monotone increasing in d_model) rather
    than a closed-form solve -- cheap (each candidate is just a model
    construction + numel sum) and avoids trusting another hand-derived
    formula for this many moving pieces (nhead=1 attention + FFN + 2 layers)."""
    best_d, best_diff = None, None
    for d in range(4, max_d_model + 1):
        n = count_params(SpinTransformer(context_length, n_spins, d_model=d, num_layers=num_layers))
        diff = abs(n - target_params)
        if best_diff is None or diff < best_diff:
            best_d, best_diff = d, diff
        elif n > target_params and best_diff is not None:
            break  # monotone increasing: once we've overshot and stopped improving, stop
    return best_d


def make_transformer_batch(dataset, indices):
    xs, ys, ps = zip(*(dataset[i] for i in indices))
    x = torch.stack([xi.permute(1, 0, 2) for xi in xs], dim=0)  # (B,K,N,2)
    y = torch.stack(ys, dim=0)  # (B,N)
    p_true = torch.stack(ps, dim=0)  # (B,N)
    return x, y, p_true


def iterate_transformer_batches(dataset, indices, batch_size, rng=None, shuffle=True):
    indices = list(indices)
    if shuffle:
        rng = rng or np.random.default_rng()
        rng.shuffle(indices)
    for start in range(0, len(indices) - batch_size + 1, batch_size):
        yield make_transformer_batch(dataset, indices[start : start + batch_size])


def evaluate_transformer(model, dataset, indices, device, batch_size=8):
    model.eval()
    total_nll, total_bayes_nll, total_correct, total_bayes_correct, total_n = 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for x, y, p_true in iterate_transformer_batches(dataset, indices, batch_size, shuffle=False):
            x, y, p_true = x.to(device), y.to(device), p_true.to(device)
            logits = model(x)
            b, n, _ = logits.shape
            logits_flat, y_flat, p_flat = logits.reshape(b * n, 2), y.reshape(b * n), p_true.reshape(b * n)
            nll = F.cross_entropy(logits_flat, y_flat, reduction="sum").item()
            pred = logits_flat.argmax(dim=-1)
            bayes_pred = (p_flat > 0.5).long()
            total_nll += nll
            total_bayes_nll += bayes_optimal_nll(p_flat, y_flat) * y_flat.numel()
            total_correct += (pred == y_flat).sum().item()
            total_bayes_correct += (bayes_pred == y_flat).sum().item()
            total_n += y_flat.numel()
    return {
        "nll": total_nll / total_n,
        "bayes_nll": total_bayes_nll / total_n,
        "nll_gap": total_nll / total_n - total_bayes_nll / total_n,
        "accuracy": total_correct / total_n,
        "bayes_accuracy": total_bayes_correct / total_n,
    }


def train_transformer(cfg, d_model, num_layers=2, test_windows=None):
    """Mirrors train.train_model's structure/semantics (same held-out-test-set
    support), adapted for SpinTransformer's (B,K,N,2) batching instead of
    SpinGNN's graph-batched (B*N,K,2) + tiled edge_index/edge_attr. Builds
    windows via train.build_windows -- this also constructs the (unused, by
    this model) sparsified graph, which is wasteful but guarantees the
    transformer sees exactly the same disorder realizations/trajectories as
    SpinGNN would for the same cfg, i.e. a paired comparison."""
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    windows = build_windows(cfg)

    if test_windows is None:
        splits = [w.time_split(cfg.train.val_fraction) for w in windows]
        train_indices = [s[0] for s in splits]
        eval_windows, eval_indices = windows, [s[1] for s in splits]
    else:
        train_indices = [list(range(len(w))) for w in windows]
        eval_windows, eval_indices = test_windows, [list(range(len(w))) for w in test_windows]

    model = SpinTransformer(cfg.disorder.context_length, cfg.disorder.n_spins, d_model, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    rng = np.random.default_rng(cfg.disorder.seed)

    history = []
    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        order = rng.permutation(len(windows))
        for w_idx in order:
            dataset = windows[w_idx]
            for x, y, _ in iterate_transformer_batches(
                dataset, train_indices[w_idx], cfg.train.batch_size, rng=rng
            ):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                b, n, _ = logits.shape
                loss = F.cross_entropy(logits.reshape(b * n, 2), y.reshape(b * n))
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * y.numel()
                epoch_n += y.numel()

        val_metrics = [
            evaluate_transformer(model, w, eval_indices[i], device) for i, w in enumerate(eval_windows)
        ]
        avg_val = {k: float(np.mean([m[k] for m in val_metrics])) for k in val_metrics[0]}
        record = {"epoch": epoch, "train_loss": epoch_loss / epoch_n, **avg_val}
        history.append(record)
        print(record)

    return model, history
