"""Causal transformer LM over spin_gnn.ksat2's event-based token sequences.

Unlike sat_transformer.py's SATTransformer (each token a continuous
{-1,0,1}^{n+1} vector, no small fixed vocabulary), every token here is a
single categorical id into a small per-instance vocabulary (spin_gnn.ksat2_tokens.Vocab)
-- a standard decoder-only LM: nn.Embedding input, a single softmax
cross-entropy readout over the vocabulary, no per-dimension decomposition.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .ksat2_tokens import decode


class SATReasoningTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_layers=2, nhead=1, max_len=1024, dim_feedforward=None,
                 norm_first=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.norm_first = norm_first
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(0.02 * torch.randn(max_len, d_model))
        resolved_ff = dim_feedforward or 4 * d_model
        # norm_first=False (post-norm, the default) renormalizes the
        # residual stream itself every block -- x = LayerNorm(x +
        # Sublayer(x)) -- which is known to destabilize training at depth
        # without careful LR warmup, exactly the failure mode seen at
        # num_layers in {12,16} with lr=1e-3 (0% well-formed output, total
        # collapse). norm_first=True (pre-norm, x = x + Sublayer(LayerNorm(x)))
        # keeps the residual stream itself un-normalized end to end, the
        # standard fix every modern deep decoder-only LM uses for exactly
        # this reason.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=resolved_ff, batch_first=True, norm_first=norm_first,
        )
        # Pre-norm stacks conventionally add one final LayerNorm after the
        # last block, since otherwise the (deliberately un-normalized)
        # residual stream feeds straight into the readout -- standard
        # practice in every GPT-2-style decoder, not needed for post-norm
        # (each post-norm block already ends with its own LayerNorm).
        final_norm = nn.LayerNorm(d_model) if norm_first else None
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=final_norm)
        self.readout = nn.Linear(d_model, vocab_size)

    def forward(self, x, valid_mask=None):
        """x: (B, L) long token ids. valid_mask: (B, L) bool, True = real
        (non-pad) position, or None if the batch has no padding.
        Returns logits: (B, L, vocab_size)."""
        b, l = x.shape
        if l > self.pos_embed.shape[0]:
            raise ValueError(f"sequence length {l} exceeds max_len={self.pos_embed.shape[0]}")
        h = self.token_embed(x) + self.pos_embed[:l]
        causal_mask = torch.triu(torch.ones(l, l, dtype=torch.bool, device=x.device), diagonal=1)
        src_key_padding_mask = ~valid_mask if valid_mask is not None else None
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        return self.readout(h)


def sat2_nll_loss(logits, target, valid_mask):
    """logits: (B, L, V). target: (B, L) long token ids. valid_mask: (B, L)
    bool. Standard token-level cross-entropy over valid positions."""
    b, l, v = logits.shape
    per_position = F.cross_entropy(logits.reshape(b * l, v), target.reshape(b * l), reduction="none").view(b, l)
    return per_position[valid_mask].mean()


def sat2_accuracy(logits, target, valid_mask):
    pred = logits.argmax(dim=-1)
    return (pred == target)[valid_mask].float().mean().item()


def generate(model, prompt_tokens, vocab, max_new_tokens, device="cpu", temperature=0.0, rng=None):
    """Autoregressively continue `prompt_tokens` ((T0,) int array of token
    ids -- typically everything through BEGIN_REASONING, i.e. hints +
    statement + the reasoning-section header) for up to max_new_tokens
    steps. Stops early once the sequence is unambiguously complete: right
    after UNSAT, or once SAT has been generated and at least 2*vocab.n_vars
    further tokens have followed (the true SAT assignment listing can never
    exceed one VAR+value pair per variable, so this bounds how long the
    model is allowed to keep babbling after SAT without an explicit
    end-of-sequence token in the grammar).

    Returns the full sequence (prompt + generated), (T,) int64 numpy array.
    """
    model.eval()
    rng = rng or np.random.default_rng()
    max_len = model.pos_embed.shape[0]
    seq = torch.from_numpy(prompt_tokens.astype(np.int64)).unsqueeze(0).to(device)
    sat_seen_at = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if seq.shape[1] >= max_len:
                break
            logits = model(seq)[0, -1]
            if temperature <= 0:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == vocab.UNSAT:
                break
            if next_id == vocab.SAT and sat_seen_at is None:
                sat_seen_at = seq.shape[1]
            if sat_seen_at is not None and seq.shape[1] - sat_seen_at >= 2 * vocab.n_vars:
                break
    return seq[0].cpu().numpy()


def solve_rate(model, instances, vocab, device, max_new_tokens, temperature=0.0, rng=None):
    """instances: list of (clauses, n_vars, hints, trace) tuples (as built by
    train_sat_transformer2.build_instances). For each, prompts the model
    with everything through BEGIN_REASONING (hints + true statement) and
    lets it freely generate the rest -- the actual end-to-end capability:
    can it derive its own correct reasoning trace and final answer, not
    just predict one token given perfect ground-truth history.

    Returns a dict: solve_rate (fraction reaching the CORRECT outcome, and
    if SAT, an assignment that actually satisfies the true clauses),
    outcome_accuracy (fraction whose SAT/UNSAT call is right, regardless of
    whether the listed assignment is actually valid), well_formed_rate
    (fraction that parse without hitting a grammar violation)."""
    from .ksat2_tokens import check_satisfies, encode_instance

    rng = rng or np.random.default_rng()
    n_solved = n_outcome_correct = n_well_formed = 0
    for clauses, n_vars, hints, trace in instances:
        full_tokens = encode_instance(clauses, n_vars, hints, trace, vocab)
        # Prompt = everything through BEGIN_REASONING: find that token's
        # position directly rather than recomputing the hints/statement
        # length by hand.
        reasoning_start = int(np.flatnonzero(full_tokens == vocab.BEGIN_REASONING)[0]) + 1
        prompt = full_tokens[:reasoning_start]
        generated = generate(model, prompt, vocab, max_new_tokens, device=device, temperature=temperature, rng=rng)
        decoded = decode(generated, vocab)

        if decoded["well_formed"]:
            n_well_formed += 1
        true_outcome = "SAT" if trace["satisfiable"] else "UNSAT"
        if decoded["outcome"] == true_outcome:
            n_outcome_correct += 1
            if true_outcome == "SAT" and check_satisfies(decoded["assignment"], clauses, n_vars):
                n_solved += 1
            elif true_outcome == "UNSAT":
                n_solved += 1

    n = len(instances)
    return {
        "solve_rate": n_solved / n if n else None,
        "outcome_accuracy": n_outcome_correct / n if n else None,
        "well_formed_rate": n_well_formed / n if n else None,
        "n_instances": n,
    }
