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

# See SudokuReasoningModelConfig.arch_mode for what each one removes.
ARCH_MODES = ("full", "linear", "rf_mlp", "rf_all", "no_mlp", "tied_no_mlp")


def _identity(x):
    """MLP activation for arch_mode='linear'.

    A plain function rather than nn.Identity so that
    nn.TransformerEncoderLayer's `activation is F.relu` / `is F.gelu` checks
    both fail and it sets activation_relu_or_gelu=0, i.e. takes its ordinary
    Python forward path instead of a fused kernel that would apply a ReLU we
    asked it not to apply.
    """
    return x


class SATReasoningTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_layers=2, nhead=1, max_len=1024, dim_feedforward=None,
                 norm_first=False, arch_mode="full"):
        super().__init__()
        if arch_mode not in ARCH_MODES:
            raise ValueError(f"arch_mode must be one of {ARCH_MODES}, got {arch_mode!r}")
        self.vocab_size = vocab_size
        self.norm_first = norm_first
        self.arch_mode = arch_mode
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
            # F.relu is nn.TransformerEncoderLayer's own default, so passing it
            # explicitly leaves every arch_mode="full" run byte-identical.
            activation=_identity if arch_mode == "linear" else F.relu,
        )
        # Pre-norm stacks conventionally add one final LayerNorm after the
        # last block, since otherwise the (deliberately un-normalized)
        # residual stream feeds straight into the readout -- standard
        # practice in every GPT-2-style decoder, not needed for post-norm
        # (each post-norm block already ends with its own LayerNorm).
        final_norm = nn.LayerNorm(d_model) if norm_first else None
        # arch_mode="tied_no_mlp" shares ONE block across every iteration, so
        # that the network is a single operator applied num_layers times --
        # h_{t+1} = F(h_t) -- rather than a composition of num_layers distinct
        # maps. nn.TransformerEncoder deep-copies `layer`, so a tied stack is
        # built with one copy and iterated in forward() instead.
        #
        # `num_layers` keeps its meaning as the DEPTH, i.e. how many times the
        # block is applied, which is what joins a tied cell to the depth ladder
        # at all; what changes is that the depth no longer costs parameters.
        # save_checkpoint records it off model_cfg and never off
        # len(encoder.layers), so a tied checkpoint round-trips exactly.
        self.tied = arch_mode == "tied_no_mlp"
        self.n_iters = num_layers
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1 if self.tied else num_layers, norm=final_norm)
        self.readout = nn.Linear(d_model, vocab_size)
        self._apply_arch_mode()

    def _apply_arch_mode(self):
        """Freeze or zero whatever `arch_mode` takes away.

        Applied AFTER construction on purpose, so that at a given seed every
        mode starts from exactly the same initialization as "full": the frozen
        weights of a random-feature cell are precisely the weights the trained
        cell started from, which is what makes the comparison a statement about
        training the weights rather than about drawing different ones.
        """
        if self.arch_mode in ("full", "linear"):
            return
        if self.arch_mode == "rf_mlp":
            for blk in self.encoder.layers:
                blk.linear1.weight.requires_grad_(False)
                blk.linear1.bias.requires_grad_(False)
        elif self.arch_mode == "rf_all":
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.readout.parameters():
                p.requires_grad_(True)
        elif self.arch_mode in ("no_mlp", "tied_no_mlp"):
            # Zeroing and freezing the SECOND MLP linear makes the feedforward
            # write identically zero for every input (dropout of zero is zero)
            # without patching nn.TransformerEncoderLayer.forward or deleting
            # modules that other code reads -- save_checkpoint takes
            # dim_feedforward off linear1.out_features, and load_state_dict
            # expects both linears to still be there.
            for blk in self.encoder.layers:
                with torch.no_grad():
                    blk.linear2.weight.zero_()
                    blk.linear2.bias.zero_()
                for p in (blk.linear1.weight, blk.linear1.bias,
                          blk.linear2.weight, blk.linear2.bias):
                    p.requires_grad_(False)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        if self.tied:
            # One shared block, applied n_iters times. Calling the layer
            # directly rather than self.encoder avoids re-applying the final
            # LayerNorm (nn.TransformerEncoder's `norm`) once per iteration --
            # it belongs after the last one only, exactly as in the untied
            # stack.
            blk = self.encoder.layers[0]
            for _ in range(self.n_iters):
                h = blk(h, src_mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
            if self.encoder.norm is not None:
                h = self.encoder.norm(h)
        else:
            h = self.encoder(h, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        return self.readout(h)


def is_tied(model):
    """True when the network is ONE block applied n_iters times."""
    return bool(getattr(model, "tied", False))


def model_depth(model):
    """How many times a block is APPLIED, which is the layer axis every
    per-layer analysis indexes.

    For an untied stack this is len(encoder.layers). For arch_mode
    "tied_no_mlp" it is NOT: nn.TransformerEncoder holds a single copy and
    forward() applies it n_iters times, so len(encoder.layers) is 1 where the
    depth is 16. n_iters is set for every arch_mode, tied or not, so it is the
    right field in both cases; the length is only a fallback for a checkpoint
    predating the field.
    """
    n = getattr(model, "n_iters", None)
    return int(n) if n else len(model.encoder.layers)


def applied_blocks(model):
    """The blocks in the order forward() applies them, one entry per
    APPLICATION -- so a tied model yields the same module model_depth(model)
    times.

    Code that replicates forward by hand (to reach something the module API
    does not expose, as sudoku_attention_distance does for attention weights)
    must iterate this rather than encoder.layers, or it runs a depth-n network
    as depth 1 and returns one map where the caller expects n.
    """
    if is_tied(model):
        return [model.encoder.layers[0]] * model_depth(model)
    return list(model.encoder.layers)


class ApplicationHooks:
    """Forward hooks keyed by a block's APPLICATION index, not by its index in
    encoder.layers.

    On an untied stack the two agree: block i fires exactly once, so its
    position is also its round. On a tied block they do not, and a module-keyed
    hook fails silently in both directions -- a capture keeps only the last
    round, under key 0, and an intervention aimed at round k fires in every
    round instead. Either one still returns a full-looking array, which is why
    this is a class and not a comment.

    fn(i, out) is called with the output of the i-th application (0-based).
    Return None to leave the pass untouched (a capture), or a tensor to replace
    that application's output (an intervention) -- torch's own forward-hook
    contract, so fn can be written once for both cases.

    Use as a context manager. The call counter is reset by a pre-hook on the
    model, so one instance survives any number of model(...) calls; a block
    invoked directly, outside model.forward, is not counted.
    """

    def __init__(self, model, fn):
        self.model = model
        self.fn = fn
        self._handles = []
        self._calls = 0

    def __enter__(self):
        model = self.model
        if is_tied(model):
            def _reset(_m, _i):
                self._calls = 0

            def _dispatch(_m, _i, out):
                i, self._calls = self._calls, self._calls + 1
                return self.fn(i, out)

            self._handles = [
                model.register_forward_pre_hook(_reset),
                model.encoder.layers[0].register_forward_hook(_dispatch),
            ]
        else:
            def mk(i):
                def hook(_m, _i, out):
                    return self.fn(i, out)
                return hook

            self._handles = [
                blk.register_forward_hook(mk(i))
                for i, blk in enumerate(model.encoder.layers)
            ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


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
