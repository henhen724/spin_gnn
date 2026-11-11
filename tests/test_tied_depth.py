"""A tied cell has ONE module in encoder.layers and a depth of n_iters, and
every per-layer analysis in this tree used to read the former.

Nothing crashed. A probe sweep came back with two layers, an ablation aimed at
round 8 fired in all sixteen, and the hand-rolled forward in
sudoku_attention_distance scored a depth-1 network as if it were the
checkpoint. Each one returns a full-looking array, which is why the fix needs
tests and not a comment: the failure mode is a plausible number.

The two properties checked here, on every module that indexes depth:
  1. the layer axis is model_depth(model) long on a tied cell, and its entries
     are the successive ROUNDS (checked against a hand-applied loop);
  2. an intervention named for round k lands on round k alone.
Untied models are checked alongside, since the fix must not move them.
"""
import numpy as np
import torch

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sat_transformer2 import (
    ApplicationHooks, SATReasoningTransformer, applied_blocks, is_tied, model_depth,
)
from spin_gnn.sudoku_activations import ResidualCapture
from spin_gnn.sudoku_attention_distance import forward_capture_attention
from spin_gnn.sudoku_circuits import (
    collect_cases, default_sites, run_head_ablation, run_patching, run_sublayer_ablation,
)
from spin_gnn.sudoku_direction_ablate import SubspaceAblation
from spin_gnn.train_sudoku_transformer import build_vocab

CFG = SudokuReasoningConfig(n_nodes=24, degree=5, n_colors=4, rho=0.7, rho_max=0.7, seed=0)
DEPTH = 4
D_MODEL = 16


def _vocab():
    return build_vocab(CFG)


def _model(vocab, arch_mode):
    torch.manual_seed(0)
    return SATReasoningTransformer(
        vocab_size=vocab.size, d_model=D_MODEL, nhead=2, num_layers=DEPTH,
        dim_feedforward=32, max_len=4096, norm_first=True, arch_mode=arch_mode,
    ).eval()


def _tokens(vocab, length=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab.size, (1, length), generator=g)


# --- the depth rule itself -------------------------------------------------

def test_tied_cell_has_one_module_and_depth_n_iters():
    """The premise of every other test here, asserted rather than assumed."""
    vocab = _vocab()
    tied = _model(vocab, "tied_no_mlp")
    assert is_tied(tied)
    assert len(tied.encoder.layers) == 1, "a tied stack must share one block"
    assert tied.n_iters == DEPTH
    assert model_depth(tied) == DEPTH, "the depth is n_iters, not the module count"

    untied = _model(vocab, "no_mlp")
    assert not is_tied(untied)
    assert model_depth(untied) == len(untied.encoder.layers) == DEPTH


def test_applied_blocks_repeats_the_same_module_on_a_tied_cell():
    vocab = _vocab()
    tied = _model(vocab, "tied_no_mlp")
    blocks = applied_blocks(tied)
    assert len(blocks) == DEPTH
    assert all(b is tied.encoder.layers[0] for b in blocks), "must be the SAME object"

    untied = _model(vocab, "no_mlp")
    assert [id(b) for b in applied_blocks(untied)] == [id(b) for b in untied.encoder.layers]


def test_application_hooks_count_rounds_and_reset_between_forwards():
    """The counter has to restart on every model(...) call. If it does not, a
    second instance measured with the same hook object gets no intervention at
    all -- the condition silently becomes the control."""
    vocab = _vocab()
    for arch in ("tied_no_mlp", "no_mlp"):
        model = _model(vocab, arch)
        seen = []
        with ApplicationHooks(model, lambda i, out: seen.append(i)) as _h:
            with torch.no_grad():
                model(_tokens(vocab))
                model(_tokens(vocab, seed=1))
        assert seen == list(range(DEPTH)) * 2, (arch, seen)


# --- sudoku_activations.ResidualCapture ------------------------------------

def test_residual_capture_returns_every_round_on_a_tied_cell():
    vocab = _vocab()
    tied = _model(vocab, "tied_no_mlp")
    with ResidualCapture(tied) as cap:
        acts = cap.run(_tokens(vocab))
    assert acts.shape[0] == DEPTH + 1, "embedding plus one row per round"
    # A hook keyed by module would have written the same key every round; these
    # must be the successive states, so consecutive rows differ.
    for i in range(1, DEPTH + 1):
        assert not torch.allclose(acts[i], acts[i - 1]), f"round {i} equals round {i-1}"


def test_residual_capture_rows_are_the_successive_rounds():
    """Checked against the loop in SATReasoningTransformer.forward, applied by
    hand: row i must be the residual after exactly i applications."""
    vocab = _vocab()
    tied = _model(vocab, "tied_no_mlp")
    x = _tokens(vocab)
    with ResidualCapture(tied) as cap:
        acts = cap.run(x)

    blk = tied.encoder.layers[0]
    l = x.shape[1]
    mask = torch.triu(torch.ones(l, l, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        h = tied.token_embed(x) + tied.pos_embed[:l]
        torch.testing.assert_close(acts[0], h)
        for i in range(DEPTH):
            h = blk(h, src_mask=mask)
            torch.testing.assert_close(acts[i + 1], h, msg=f"round {i+1}")


def test_residual_capture_untied_shape_is_unchanged():
    vocab = _vocab()
    untied = _model(vocab, "no_mlp")
    with ResidualCapture(untied) as cap:
        acts = cap.run(_tokens(vocab))
    assert acts.shape[0] == DEPTH + 1


# --- sudoku_direction_ablate.SubspaceAblation ------------------------------

def _basis():
    """A (1, d) orthonormal row: project out the first coordinate."""
    b = np.zeros((1, D_MODEL), dtype=np.float32)
    b[0, 0] = 1.0
    return b


def _capture_under(model, bases, x):
    """Activations captured INSIDE an ablation context.

    Torch calls forward hooks in registration order and feeds each the previous
    one's return value, so a capture registered after the ablation sees the
    ablated output -- which is what makes "did round k-1 move?" answerable.
    """
    with SubspaceAblation(model, bases):
        with ResidualCapture(model) as cap:
            return cap.run(x)


def test_subspace_ablation_hits_only_the_round_it_names_on_a_tied_cell():
    vocab = _vocab()
    tied = _model(vocab, "tied_no_mlp")
    x = _tokens(vocab)
    k = 3  # 1-based layer index, i.e. the output of round 3 of 4

    clean = _capture_under(tied, {}, x)
    ablated = _capture_under(tied, {k: _basis()}, x)

    for i in range(0, k):
        torch.testing.assert_close(ablated[i], clean[i], msg=f"round {i} moved and should not have")
    assert not torch.allclose(ablated[k], clean[k]), "the named round did not move"
    assert not torch.allclose(ablated[DEPTH], clean[DEPTH]), "the edit did not propagate"


def test_subspace_ablation_rejects_layers_past_the_tied_depth():
    """Before the fix the bound was len(encoder.layers) == 1, so every real
    layer index on a tied cell was rejected; now the bound is the depth."""
    vocab = _vocab()
    tied = _model(vocab, "tied_no_mlp")
    SubspaceAblation(tied, {DEPTH: _basis()})  # the last round is in range
    try:
        SubspaceAblation(tied, {DEPTH + 1: _basis()})
    except ValueError:
        pass
    else:
        raise AssertionError("a layer past the depth must be rejected")


# --- sudoku_attention_distance.forward_capture_attention -------------------

def test_forward_capture_attention_reproduces_the_models_own_logits():
    """This function replicates forward() by hand to reach the attention
    weights, so it is only ever as good as that replication. On a tied cell it
    iterated encoder.layers and therefore scored a DIFFERENT, depth-1 network;
    the logits are the check that catches it."""
    vocab = _vocab()
    for arch in ("tied_no_mlp", "no_mlp"):
        model = _model(vocab, arch)
        x = _tokens(vocab)
        logits, attn = forward_capture_attention(model, x)
        assert len(attn) == DEPTH, (arch, len(attn))
        assert attn[0].shape == (2, x.shape[1], x.shape[1])
        with torch.no_grad():
            torch.testing.assert_close(logits, model(x), atol=1e-5, rtol=1e-4)


# --- sudoku_circuits: the fix from 2e552b3, until now untested -------------

def _cases(vocab, n=8):
    return collect_cases(CFG, vocab, n_instances=n, seed_start=11_000)[:3]


def test_patching_layer_axis_is_the_depth_on_a_tied_cell():
    vocab = _vocab()
    tied = _model(vocab, "tied_no_mlp")
    rows = run_patching(tied, vocab, "cpu", _cases(vocab), default_sites)
    assert rows, "no patching rows"
    assert sorted({r["layer"] for r in rows}) == list(range(DEPTH)), "layer axis is not the depth"
    assert all(np.isfinite(r["recovery"]) for r in rows)


def test_weight_space_ablations_refuse_on_a_tied_cell():
    """A shared weight set cannot be ablated per round: looping over the depth
    would emit DEPTH identical rows that read as a localisation. Refusing is
    the result; this test is what keeps the refusal from being 'fixed'."""
    vocab = _vocab()
    tied = _model(vocab, "tied_no_mlp")
    cases = _cases(vocab)
    for fn in (run_head_ablation, run_sublayer_ablation):
        try:
            fn(tied, vocab, "cpu", cases)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} must refuse on a tied model")
