"""Causal transformer over spin_gnn.ksat solver-trace vector tokens, trained
via next-token prediction to see whether attention alone can learn to
reproduce unit-propagation/backtracking dynamics.

Unlike spin_gnn.transformer_baseline.SpinTransformer (a fixed-K-window,
non-causal "look at the whole context window, predict the single token after
it" model, where the module docstring notes masking would be a no-op), a
solver trace is a genuinely open-ended autoregressive sequence -- every token
depends on everything before it, and predicting token t+1 must not see token
t+1 or later. So this model applies a real causal attention mask
(nn.Transformer.generate_square_subsequent_mask).

Each token is a continuous vector in {-1,0,1}^{n_vars+1} (see ksat.py's
module docstring for the encoding), linearly embedded (not one-hot -- there's
no small fixed vocabulary the way there is for the 2-symbol spin alphabet).
The readout predicts each of the n_vars+1 dimensions as an independent 3-way
class ({-1,0,1} -> {0,1,2}): the direct generalization of
SpinTransformer/SpinGNN's per-site 2-way cross-entropy to this vector-token
alphabet.

SATTransformer's attention is nn.TransformerEncoder by default, but takes an
optional qk_dim to decouple the Q/K projection rank from nhead/d_model (see
LowRankSelfAttention/LowRankEncoderLayer below) -- V and the output
projection always stay at full d_model.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def to_class_index(x):
    """{-1,0,1}-valued float tensor -> {0,1,2}-valued long tensor."""
    return (x + 1).long()


class LowRankSelfAttention(nn.Module):
    """Causal multi-head self-attention with the Q/K projection size ("rank")
    decoupled from nhead and d_model. nn.MultiheadAttention always ties
    head_dim to exactly d_model/nhead and has no way to shrink (or expand)
    just the Q/K side while leaving V/the output at full d_model, so this
    reimplements attention directly from matmuls to allow that.

    qk_dim is the *total* (pre-head-split) Q/K projection size; each of the
    nhead heads gets qk_dim // nhead of it for scoring. V and the output
    projection stay at the full d_model (split as d_model // nhead per head)
    -- only the key/query rank is being restricted, per the request that
    prompted this ("the rank of the key query matrices", not V's).

    Reimplements nn.MultiheadAttention's dropout placement exactly (a
    dropout on the softmax attention weights before the V matmul) so this is
    an apples-to-apples swap-in for nn.TransformerEncoderLayer's default
    attention, not a confound alongside the actual qk_dim variable."""

    def __init__(self, d_model, nhead, qk_dim, dropout=0.1):
        super().__init__()
        if qk_dim % nhead != 0:
            raise ValueError(f"qk_dim={qk_dim} must be divisible by nhead={nhead}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.nhead = nhead
        self.qk_head_dim = qk_dim // nhead
        self.v_head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, qk_dim)
        self.k_proj = nn.Linear(d_model, qk_dim)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask=None):
        """x: (B, L, d_model). attn_mask: (L, L) bool, True = disallowed
        (causal). key_padding_mask: (B, L) bool, True = disallowed (pad)."""
        b, l, _ = x.shape

        def split_heads(t, head_dim):
            return t.view(b, l, self.nhead, head_dim).transpose(1, 2)  # (B, nhead, L, head_dim)

        q = split_heads(self.q_proj(x), self.qk_head_dim)
        k = split_heads(self.k_proj(x), self.qk_head_dim)
        v = split_heads(self.v_proj(x), self.v_head_dim)

        scores = (q @ k.transpose(-2, -1)) / (self.qk_head_dim ** 0.5)  # (B, nhead, L, L)
        mask = attn_mask.view(1, 1, l, l)
        if key_padding_mask is not None:
            mask = mask | key_padding_mask.view(b, 1, 1, l)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = weights @ v  # (B, nhead, L, v_head_dim)
        out = out.transpose(1, 2).reshape(b, l, self.nhead * self.v_head_dim)
        return self.out_proj(out)


class LowRankEncoderLayer(nn.Module):
    """Post-norm transformer block (attention + FFN, each with residual +
    LayerNorm, dropout in the same three places) built on
    LowRankSelfAttention -- mirrors nn.TransformerEncoderLayer's default
    (norm_first=False, activation=relu, dropout=0.1) layout exactly, so
    results stay comparable with every prior sweep that used the built-in
    layer; qk_dim is the only actual architectural difference."""

    def __init__(self, d_model, nhead, dim_feedforward, qk_dim, dropout=0.1):
        super().__init__()
        self.self_attn = LowRankSelfAttention(d_model, nhead, qk_dim, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, attn_mask, key_padding_mask=None):
        attn_out = self.dropout1(self.self_attn(x, attn_mask, key_padding_mask))
        x = self.norm1(x + attn_out)
        ff_out = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm2(x + ff_out)
        return x


class SATTransformer(nn.Module):
    def __init__(self, n_vars, d_model=64, num_layers=2, nhead=1, max_len=4096,
                 dim_feedforward=None, qk_dim=None):
        super().__init__()
        self.n_vars = n_vars
        self.token_dim = n_vars + 1
        # 0/None both mean "full rank" (= d_model), matching the
        # dim_feedforward-or-default convention elsewhere in this module.
        self.qk_dim = qk_dim or d_model
        self.input_embed = nn.Linear(self.token_dim, d_model)
        self.pos_embed = nn.Parameter(0.02 * torch.randn(max_len, d_model))
        resolved_ff = dim_feedforward or 4 * d_model

        if self.qk_dim == d_model:
            # Full-rank case: route through the exact built-in layer every
            # prior sweep used, rather than this module's own reimplementation
            # -- keeps every existing checkpoint/result 100% reproducible
            # when this new axis isn't actually being exercised.
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=resolved_ff, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self._custom_attn = False
        else:
            self.encoder = nn.ModuleList(
                LowRankEncoderLayer(d_model, nhead, resolved_ff, self.qk_dim) for _ in range(num_layers)
            )
            self._custom_attn = True
        self.readout = nn.Linear(d_model, self.token_dim * 3)

    def forward(self, x, valid_mask=None):
        """x: (B, L, n_vars+1) input tokens.
        valid_mask: (B, L) bool, True = real (non-pad) position, or None if
        the batch has no padding.
        Returns logits: (B, L, n_vars+1, 3)."""
        b, l, _ = x.shape
        if l > self.pos_embed.shape[0]:
            raise ValueError(f"sequence length {l} exceeds max_len={self.pos_embed.shape[0]}")
        h = self.input_embed(x) + self.pos_embed[:l]
        # Bool masks (True = disallowed/pad) on both args, matching dtypes --
        # mixing a float causal mask with a bool padding mask triggers a
        # PyTorch deprecation warning even though it still works.
        causal_mask = torch.triu(torch.ones(l, l, dtype=torch.bool, device=x.device), diagonal=1)
        src_key_padding_mask = ~valid_mask if valid_mask is not None else None
        if self._custom_attn:
            for layer in self.encoder:
                h = layer(h, causal_mask, src_key_padding_mask)
        else:
            h = self.encoder(h, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        return self.readout(h).view(b, l, self.token_dim, 3)


def sat_nll_loss(logits, target, valid_mask):
    """logits: (B, L, D, 3). target: (B, L, D) in {-1,0,1}. valid_mask:
    (B, L) bool, True = real (non-pad) position. Cross-entropy per dimension,
    summed over the D dimensions, averaged over valid positions."""
    b, l, d, _ = logits.shape
    target_idx = to_class_index(target)
    per_position = F.cross_entropy(
        logits.reshape(b * l * d, 3), target_idx.reshape(b * l * d), reduction="none"
    ).view(b, l, d).sum(dim=-1)  # (B, L), summed over dimensions
    return per_position[valid_mask].mean()


def sat_accuracy(logits, target, valid_mask):
    """Fraction of (position, dimension) predictions matching the target,
    over valid (non-pad) positions."""
    pred = logits.argmax(dim=-1)  # (B, L, D)
    target_idx = to_class_index(target)
    correct = (pred == target_idx)[valid_mask]  # (num_valid_positions, D)
    return correct.float().mean().item()


def generate_trace(model, prompt_tokens, max_new_tokens, device="cpu", temperature=0.0, rng=None):
    """Autoregressively continue `prompt_tokens` ((T0, n_vars+1) numpy array,
    e.g. spin_gnn.ksat.initial_prompt's output) for up to max_new_tokens
    steps: at each step, sample every one of the n_vars+1 dimensions' 3-way
    class independently from the model's predicted distribution at the last
    position (temperature=0 -> greedy argmax per dimension). No stopping
    rule beyond max_new_tokens or hitting the model's max_len -- the model
    was never trained to emit an explicit end-of-sequence marker, since
    spin_gnn.ksat's own encoding doesn't have one either (a real trace just
    stops when the solver is done). Recomputes the full forward pass over
    the whole sequence so far at every step (no KV cache) -- fine for the
    exploratory/demo scale this is used at, not for fast generation.

    Returns the full sequence (prompt + generated), (T0+generated, n_vars+1)
    float32 array."""
    model.eval()
    rng = rng or np.random.default_rng()
    max_len = model.pos_embed.shape[0]
    seq = torch.from_numpy(prompt_tokens.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if seq.shape[1] >= max_len:
                break
            logits = model(seq)[0, -1]  # (n_vars+1, 3)
            if temperature <= 0:
                next_idx = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_idx = torch.multinomial(probs, 1).squeeze(-1)
            next_token = (next_idx.float() - 1.0).view(1, 1, -1)  # {0,1,2} -> {-1,0,1}
            seq = torch.cat([seq, next_token], dim=1)
    return seq[0].cpu().numpy()
