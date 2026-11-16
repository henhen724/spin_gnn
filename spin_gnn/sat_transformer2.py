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
                 norm_first=False, arch_mode="full", n_blocks=1,
                 determinacy_channel=False, node_offset=None, n_nodes=None,
                 determinacy_feedback=False):
        super().__init__()
        if arch_mode not in ARCH_MODES:
            raise ValueError(f"arch_mode must be one of {ARCH_MODES}, got {arch_mode!r}")
        # n_blocks = k: how many DISTINCT blocks the tied stack cycles through.
        # k=1 is the weight-tied cell (one operator, num_layers applications);
        # k=num_layers would be the untied stack. In between, blocks fire in the
        # order 0,1,...,k-1,0,1,... so the network is still a single map
        # iterated num_layers/k times -- G = F_{k-1} o ... o F_0 -- which is the
        # whole point: every rung of the k ladder stays a fixed-point iteration,
        # and only the number of primitive maps inside one step changes.
        #
        # The divisibility requirement is not cosmetic: a partial final cycle
        # would make the last few applications a different composition from
        # every earlier one, and "iterated T/k times" would stop being true.
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        if n_blocks > 1:
            if arch_mode != "tied_no_mlp":
                raise ValueError(
                    "n_blocks > 1 cycles a SHARED block and is only defined for "
                    f"arch_mode='tied_no_mlp', got {arch_mode!r}. An untied stack "
                    "already has num_layers distinct blocks.")
            if num_layers % n_blocks:
                raise ValueError(
                    f"n_blocks={n_blocks} does not divide num_layers={num_layers}; "
                    "the cycle would end part-way through and the last "
                    "applications would compose differently from the rest.")
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
        # k distinct blocks for a tied stack, num_layers for an untied one. The
        # k copies come from nn.TransformerEncoder's own _get_clones, which
        # DEEP-COPIES the one layer it is given, so at a fixed seed all k start
        # IDENTICAL to each other and to the k=1 cell -- the same discipline
        # _apply_arch_mode documents below, and it makes the k ladder a
        # statement about what training does with the extra blocks rather than
        # about having drawn different weights.
        self.n_blocks = n_blocks if self.tied else num_layers
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.n_blocks if self.tied else num_layers,
            norm=final_norm)
        self.readout = nn.Linear(d_model, vocab_size)
        # Stage Y. A SECOND score per node token, g_{p,v} = <h[p] * s, E_v>,
        # read off the same hidden state through the token embedding with one
        # learned diagonal s in R^d -- d_model new parameters, 320 at d=320,
        # +0.05%. The emission adds log sigma(g) to the node block only:
        #
        #     z'_{p,v} = z_{p,v} + log sigma(g_{p,v})   for v a node token
        #     z'_{p,v} = z_{p,v}                        otherwise
        #
        # Two properties, and stage Y is a test of both:
        #
        #   It can RE-RANK. Finding 74 measured what the untied guard does when
        #   it fires -- it names a DIFFERENT still-forced node (node token
        #   0.966, still forced 0.903, names v again 0.000 of 350), it does not
        #   stop. A monotone gate z -> z + log sigma(a(z - tau)) is strictly
        #   increasing in z and so cannot produce that at all; it can only
        #   build the stop-shaped guard stage X already built. g is a different
        #   function of h than z is -- token_embed and readout are separate,
        #   never weight-tied -- so log sigma(g) reorders the node block.
        #
        #   It SEPARATES the two objectives. Stage X put the determinacy BCE on
        #   the readout's own node logits and got AUROC 0.947 with zero
        #   counterfactual responsiveness (finding 75): a per-node loss on
        #   absolute logit HEIGHT shifts the whole node block against the
        #   terminal tokens instead of making one node's logit conditional on
        #   that node's determinacy. Here the cross-entropy writes to z and the
        #   BCE writes to g, so they stop competing for the same scalars, and
        #   the only way g can change the emission is through the node block's
        #   internal order -- which is the thing that was missing.
        #
        # Off by default, and when off this class is byte-identical to every
        # run from stages S through X: no parameter is created, forward()
        # returns self.readout(h) through the same expression, and every
        # existing checkpoint loads unchanged.
        self.det_node_offset = node_offset
        self.det_n_nodes = n_nodes
        if determinacy_channel:
            if node_offset is None or n_nodes is None:
                raise ValueError(
                    "determinacy_channel=True needs node_offset and n_nodes: "
                    "the channel scores the node-token block of the vocabulary "
                    "and cannot find it otherwise")
            if node_offset < 0 or node_offset + n_nodes > vocab_size:
                raise ValueError(
                    f"node block [{node_offset}, {node_offset + n_nodes}) does "
                    f"not fit in vocab_size={vocab_size}")
            # ZERO, and this is the one initialization that works. The first
            # submission (52336800) used torch.ones on the reasoning that
            # g = <h, E_v> is the model's own embedding similarity and so
            # log sigma(g) would be "a small common offset". MEASURED, it is
            # neither small nor common: h is LayerNorm'd and E_v ~ N(0, I), so
            # g has std 21 and log sigma(g) has std 10.3 and reaches -59,
            # against node logits whose own spread is 0.61. That is a random
            # PER-NODE penalty seventeen times the signal it perturbs -- the
            # ranking is scrambled before the first step. Seeds 0 and 2
            # recovered from it; seed 1 never left the initial plateau
            # (train_nll 3.67 -> 3.96 in epoch 0, val accuracy 0.032 and det
            # AUROC 0.59 flat for 13 epochs), which costs Y4 and with it the
            # whole stage.
            #
            # At s = 0 the channel is inert: g is identically 0, so log
            # sigma(g) = -log 2 for every node at every position. A constant
            # on the whole node block leaves the ORDER within it exactly the
            # control's, which is the matched start the comment above only
            # claimed. It is not a saddle -- dL/ds_i = sum_{p,v} (dL/dg) h_pi
            # E_vi is O(1) -- and it costs no parameter, so the registered
            # count of +320 is unchanged.
            self.det_scale = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("det_scale", None)

        # STAGE LOOP. The channel above is read off the FINAL hidden state and
        # written to the emission, so g can change what the network SAYS and
        # never what it COMPUTES: finding 78 called the result a predicate that
        # is not causally wired, and 68-74 put it as the shared block computing
        # the ambiguity and then scheduling as if certain. This flag closes the
        # loop -- the gate is recomputed after every application of the shared
        # block and folded back into the residual stream, so iteration t+1 reads
        # iteration t's determinacy estimate:
        #
        #     g_t         = <h_t * s, E_v>                      (as stage Y)
        #     a_t         = log sigma(g_t) + log 2              (0 when g_t = 0)
        #     h_t         <- h_t + c * (a_t @ E_nodes) / n_nodes
        #
        # for t = 0 .. n_iters-2. NOT after the last application: the readout and
        # `determinacy_logits` then see a hidden state reached the same way stage
        # Y's was, so the P1-P4 read-out measures the same object and the loop
        # changes what the BLOCK sees rather than what the readout reads.
        #
        # ONE new parameter, the scalar c, and it is initialized at ZERO so the
        # arm starts at EXACTLY the stage Y body it warms from. That matters more
        # here than it did for `det_scale`: this arm warms from a TRAINED body
        # whose g is not identically zero, so an open loop at step 0 would
        # perturb a solved network by an unmeasured amount -- the `torch.ones`
        # mistake of stage Y, one stage later. c = 0 is not a saddle:
        # dL/dc = sum_t <dL/dh_t, (a_t @ E_nodes)/n_nodes> is O(1) at a trained
        # body, so gradient descent opens the loop if the objective wants it, and
        # |c| at the end is itself the measurement of whether it did.
        #
        # The `+ log 2` centering is what makes c = 0 and g = 0 BOTH inert; the
        # 1/n_nodes makes the delta the size of one embedding row rather than 48,
        # which is the magnitude lesson of findings 76 and 79. Re-using
        # `token_embed`'s node rows is what keeps the parameter count at +1: no
        # new matrix is introduced in either direction.
        #
        # Off by default. When off, no parameter is created and forward() runs
        # the same expressions in the same order, so every checkpoint from stages
        # S through PAIR loads and evaluates unchanged.
        self.determinacy_feedback = bool(determinacy_feedback)
        if self.determinacy_feedback:
            if not determinacy_channel:
                raise ValueError(
                    "determinacy_feedback=True needs determinacy_channel=True: "
                    "there is no g to feed back without the channel")
            if not self.tied:
                raise ValueError(
                    "determinacy_feedback=True is defined for the tied cell "
                    f"only (arch_mode='tied_no_mlp'), got {arch_mode!r}: an "
                    "untied stack has no iteration to feed back through")
            self.loop_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("loop_scale", None)
        # Set by forward() when the loop is on: mean |delta| / mean |h| at the
        # last fed-back iteration, detached. The magnitude of the feedback, not
        # just its coefficient -- c alone says nothing without the scale of a.
        self._fb_rel = None
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

    def determinacy_logits(self, h):
        """The stage-Y channel: (B, L, d_model) hidden -> (B, L, n_nodes).

        Only the node block's embedding rows are touched, so this is an
        (n_nodes x d) matmul and not a second full readout.
        """
        lo, hi = self.det_node_offset, self.det_node_offset + self.det_n_nodes
        return (h * self.det_scale) @ self.token_embed.weight[lo:hi].t()

    def determinacy_feedback_delta(self, h):
        """The stage-LOOP term: (B, L, d_model) -> (B, L, d_model).

        `log sigma(g) + log 2` is zero at g = 0, so this returns exactly zero
        both at `det_scale = 0` (an untrained channel) and at `loop_scale = 0`
        (the warm start), which is what makes the arm start on the body it
        warms from rather than near it.
        """
        lo, hi = self.det_node_offset, self.det_node_offset + self.det_n_nodes
        a = F.logsigmoid(self.determinacy_logits(h)) + 0.6931471805599453
        return (self.loop_scale
                * (a @ self.token_embed.weight[lo:hi]) / self.det_n_nodes)

    def forward(self, x, valid_mask=None, return_determinacy=False):
        """x: (B, L) long token ids. valid_mask: (B, L) bool, True = real
        (non-pad) position, or None if the batch has no padding.
        Returns logits: (B, L, vocab_size), or (logits, g) with g the
        (B, L, n_nodes) determinacy channel when return_determinacy=True.

        return_determinacy on a model built without the channel raises rather
        than returning None: a training loop that asked for g and silently got
        nothing would train the stage-X objective under a stage-Y name.
        """
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
            # t % n_blocks, so k=1 is the single shared block applied n_iters
            # times (byte-identical to before this field existed) and k>1
            # cycles 0,1,...,k-1 for n_iters/k full turns.
            for t in range(self.n_iters):
                blk = self.encoder.layers[t % self.n_blocks]
                h = blk(h, src_mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
                # stage LOOP, and never after the last application -- see the
                # constructor. `is not None` rather than the bool flag so a
                # checkpoint loaded without the parameter cannot take this path.
                if self.loop_scale is not None and t < self.n_iters - 1:
                    delta = self.determinacy_feedback_delta(h)
                    self._fb_rel = (delta.detach().abs().mean()
                                    / h.detach().abs().mean().clamp_min(1e-12))
                    h = h + delta
            if self.encoder.norm is not None:
                h = self.encoder.norm(h)
        else:
            h = self.encoder(h, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        z = self.readout(h)
        if self.det_scale is None:
            if return_determinacy:
                raise ValueError(
                    "return_determinacy=True on a model built with "
                    "determinacy_channel=False: there is no channel to return")
            return z
        g = self.determinacy_logits(h)
        lo, hi = self.det_node_offset, self.det_node_offset + self.det_n_nodes
        # torch.cat rather than an in-place slice assignment on z: z is the
        # output of a Linear and writing into a view of it is an autograd
        # hazard that silently differs between torch versions.
        z = torch.cat([z[..., :lo], z[..., lo:hi] + F.logsigmoid(g),
                       z[..., hi:]], dim=-1)
        return (z, g) if return_determinacy else z


def is_tied(model):
    """True when the network SHARES blocks across applications.

    k distinct blocks cycled n_iters/k times is still tied in the only sense
    this predicate is used for: len(encoder.layers) is not the depth, and
    forward() applies blocks more than once each.
    """
    return bool(getattr(model, "tied", False))


def n_blocks(model):
    """How many DISTINCT blocks the stack holds (k). 1 for the shared-block
    cell, num_layers for an untied stack, k in between."""
    n = getattr(model, "n_blocks", None)
    return int(n) if n else len(model.encoder.layers)


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
        k = n_blocks(model)
        return [model.encoder.layers[t % k] for t in range(model_depth(model))]
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

            # One counter, a hook on EVERY distinct block: with k > 1 the
            # applications still fire in order 0,1,...,k-1,0,..., so a single
            # shared counter is the application index whichever block fired,
            # and hooking only layers[0] would miss k-1 applications in k.
            self._handles = [model.register_forward_pre_hook(_reset)]
            self._handles += [blk.register_forward_hook(_dispatch)
                              for blk in model.encoder.layers]
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
