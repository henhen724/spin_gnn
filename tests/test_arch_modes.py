"""Verify what each arch_mode actually removes, BEFORE spending GPU-hours on it.

The minimal-architecture programme asks how little machinery learning DD
requires, and answers it by taking machinery away: the MLP nonlinearity
(arch_mode="linear"), the training of the MLP's first layer
("rf_mlp"), the training of everything but the readout ("rf_all"), and the MLP
write itself ("no_mlp"). Every one of those is a silent failure mode if it is
wrong -- a "linear" cell that still applies a ReLU trains fine, scores
plausibly and answers a different question, and nothing in the logs says so.
Finding 43 is the precedent: `fall` was not persisted by save_checkpoint, so
a fall-trained model would have been scored on default-ensemble instances with
no error at all.

So each mode is checked three ways: the forward pass does what the mode says,
one optimizer step moves exactly the parameters the mode says are trainable,
and a save/load round trip reproduces the logits bit for bit. The round trip is
the one that matters most -- it is the finding-43 check -- because every
analysis reader in the repo goes through load_checkpoint.

No `import pytest`: the only pytest module on Sherlock is py39 and this project
is py312, so the suite could not be collected here at all (see tests/_pytest_shim
and STATUS Open problems). Plain asserts, run with

    python3 tests/test_arch_modes.py
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningModelConfig  # noqa: E402
from spin_gnn.sat_transformer2 import ARCH_MODES, SATReasoningTransformer  # noqa: E402
from spin_gnn.sudoku_hparam_sweep import ARCH_TAG  # noqa: E402
from spin_gnn.sudoku_tokens import Vocab  # noqa: E402
from spin_gnn.train_sudoku_transformer import load_checkpoint, save_checkpoint  # noqa: E402

D_MODEL, N_LAYERS, N_HEAD, FF, MAX_LEN, VOCAB = 16, 3, 2, 32, 24, 29


def build(mode, seed=0):
    torch.manual_seed(seed)
    return SATReasoningTransformer(
        VOCAB, d_model=D_MODEL, num_layers=N_LAYERS, nhead=N_HEAD, max_len=MAX_LEN,
        dim_feedforward=FF, norm_first=True, arch_mode=mode,
    )


def ff_write(blk, h):
    """The feedforward sublayer's contribution to the residual stream."""
    return blk._ff_block(blk.norm2(h))


def test_same_initialization_across_modes():
    """Every mode starts from the SAME weights at the same seed.

    _apply_arch_mode's docstring claims this and the whole comparison rests on
    it: a random-feature cell's frozen weights have to be exactly the weights
    the trained cell started from, or the contrast is confounded by a different
    draw. It also proves the `activation=` argument does not perturb init.
    """
    ref = build("full").state_dict()
    for mode in ARCH_MODES:
        sd = build(mode).state_dict()
        assert set(sd) == set(ref), f"{mode} changed the state_dict keys"
        for k, v in sd.items():
            if mode == "no_mlp" and "linear2" in k:
                continue  # deliberately zeroed, checked in its own test
            assert torch.equal(v, ref[k]), f"{mode} initializes {k} differently from full"


def test_linear_has_no_nonlinearity_in_the_mlp():
    full, lin = build("full"), build("linear")
    x = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    assert torch.equal(full.encoder.layers[0].activation(x), torch.relu(x))
    assert torch.equal(lin.encoder.layers[0].activation(x), x), "linear still applies a ReLU"
    # The fused-kernel guard: TransformerEncoderLayer flags relu as 1 and gelu
    # as 2, and only a flag of 0 keeps it on the Python path that honours a
    # custom activation.
    assert lin.encoder.layers[0].activation_relu_or_gelu == 0
    assert full.encoder.layers[0].activation_relu_or_gelu == 1

    # The MLP write is exactly one affine map of its input, so it is additive
    # in a way the ReLU one is not. Tested on the real sublayer, in eval mode
    # so dropout is off.
    lin.eval()
    blk = lin.encoder.layers[0]
    torch.manual_seed(1)
    a, b = torch.randn(1, 5, D_MODEL), torch.randn(1, 5, D_MODEL)
    # linear2(linear1(.)) is affine, so f(a+b) - f(a) - f(b) + f(0) == 0.
    z = torch.zeros_like(a)
    resid = (blk._ff_block(a + b) - blk._ff_block(a) - blk._ff_block(b) + blk._ff_block(z))
    assert resid.abs().max().item() < 1e-5, f"linear MLP is not affine, max {resid.abs().max()}"
    full.eval()
    fb = full.encoder.layers[0]
    nonlin = (fb._ff_block(a + b) - fb._ff_block(a) - fb._ff_block(b) + fb._ff_block(z))
    assert nonlin.abs().max().item() > 1e-3, "the full MLP looks affine too -- test is vacuous"

    # And the two models must not agree, or 'linear' would be a no-op.
    tokens = torch.randint(0, VOCAB, (2, 7))
    assert not torch.allclose(full(tokens), lin(tokens), atol=1e-4)


def test_no_mlp_writes_exactly_zero():
    model = build("no_mlp").eval()
    torch.manual_seed(2)
    h = torch.randn(2, 6, D_MODEL)
    for i, blk in enumerate(model.encoder.layers):
        w = ff_write(blk, h)
        assert torch.count_nonzero(w).item() == 0, f"block {i} still writes an MLP output"
    # Attention must still be live, or the model is a bag of embeddings.
    tokens = torch.randint(0, VOCAB, (2, 7))
    before = model(tokens).clone()
    with torch.no_grad():
        model.encoder.layers[0].self_attn.out_proj.weight.mul_(2.0)
    assert not torch.allclose(before, model(tokens), atol=1e-5), "attention is not doing anything"


def one_step_delta(mode):
    """Train one step on a dummy objective; return {param name: did it move}."""
    model = build(mode)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)
    torch.manual_seed(3)
    tokens = torch.randint(0, VOCAB, (4, 8))
    loss = model(tokens).square().mean()
    loss.backward()
    opt.step()
    after = model.state_dict()
    return {k: not torch.equal(before[k], after[k]) for k in before}, model


def test_rf_mlp_trains_only_the_second_mlp_layer():
    moved, model = one_step_delta("rf_mlp")
    frozen = [k for k, m in moved.items() if not m]
    assert all("linear1" in k for k in frozen), f"rf_mlp froze more than linear1: {frozen}"
    for i in range(N_LAYERS):
        assert not moved[f"encoder.layers.{i}.linear1.weight"]
        assert not moved[f"encoder.layers.{i}.linear1.bias"]
        assert moved[f"encoder.layers.{i}.linear2.weight"], f"block {i} linear2 did not train"
        assert moved[f"encoder.layers.{i}.self_attn.out_proj.weight"], "attention did not train"
    assert moved["readout.weight"]
    n_frozen = sum(p.numel() for blk in model.encoder.layers
                   for p in (blk.linear1.weight, blk.linear1.bias))
    total = sum(p.numel() for p in model.parameters())
    assert model.trainable_parameter_count() == total - n_frozen


def test_rf_all_trains_only_the_readout():
    moved, model = one_step_delta("rf_all")
    for k, m in moved.items():
        expect = k.startswith("readout.")
        assert m == expect, f"rf_all: {k} moved={m}, expected {expect}"
    assert model.trainable_parameter_count() == sum(p.numel() for p in model.readout.parameters())
    # The point of this cell is that the trained problem is convex, which needs
    # the features to be a FIXED function of the input. Nothing upstream of the
    # readout may carry grad.
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not model.token_embed.weight.requires_grad and not model.pos_embed.requires_grad


def test_no_mlp_stays_zero_after_a_step():
    moved, _ = one_step_delta("no_mlp")
    for i in range(N_LAYERS):
        for name in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
            assert not moved[f"encoder.layers.{i}.{name}"], f"no_mlp block {i} {name} trained"
        assert moved[f"encoder.layers.{i}.self_attn.out_proj.weight"]


def test_full_and_linear_train_everything():
    for mode in ("full", "linear"):
        moved, model = one_step_delta(mode)
        assert all(moved.values()), f"{mode} left {[k for k, m in moved.items() if not m]} frozen"
        assert model.trainable_parameter_count() == sum(p.numel() for p in model.parameters())


def test_checkpoint_round_trip_preserves_the_mode():
    """The finding-43 check: config.json carries arch_mode and readers use it.

    Without this, load_checkpoint rebuilds a "linear" cell with a ReLU and
    every probe, ablation and rho curve in the repo silently scores a model
    that was never trained.
    """
    sudoku_cfg = SudokuReasoningConfig(n_nodes=12, degree=3, n_colors=3, shuffle_adj=True, fall=True)
    vocab = Vocab(12, 3, 3)
    tokens = torch.randint(0, min(VOCAB, vocab.size), (2, 9))
    for mode in ARCH_MODES:
        model_cfg = SudokuReasoningModelConfig(
            d_model=D_MODEL, num_layers=N_LAYERS, nhead=N_HEAD, dim_feedforward=FF,
            norm_first=True, arch_mode=mode,
        )
        torch.manual_seed(0)
        model = SATReasoningTransformer(
            vocab.size, d_model=D_MODEL, num_layers=N_LAYERS, nhead=N_HEAD, max_len=MAX_LEN,
            dim_feedforward=FF, norm_first=True, arch_mode=mode,
        ).eval()
        want = model(tokens).detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "ckpt")
            save_checkpoint(out, sudoku_cfg, model_cfg, model, vocab, MAX_LEN, 0.1, {}, [])
            back, _vocab, config = load_checkpoint(out)
            assert config["arch_mode"] == mode, f"config.json lost arch_mode for {mode}"
            assert config["n_trainable_params"] == model.trainable_parameter_count()
            assert back.arch_mode == mode
            got = back(tokens).detach()
        assert torch.allclose(want, got, atol=1e-6), f"{mode} does not survive a round trip"


def test_pre_2026_11_04_checkpoints_read_as_full():
    """A config.json with no arch_mode key must rebuild the full ReLU MLP."""
    sudoku_cfg = SudokuReasoningConfig(n_nodes=12, degree=3, n_colors=3)
    vocab = Vocab(12, 3, 3)
    model_cfg = SudokuReasoningModelConfig(d_model=D_MODEL, num_layers=N_LAYERS, nhead=N_HEAD,
                                           dim_feedforward=FF, norm_first=True)
    torch.manual_seed(0)
    model = SATReasoningTransformer(vocab.size, d_model=D_MODEL, num_layers=N_LAYERS, nhead=N_HEAD,
                                    max_len=MAX_LEN, dim_feedforward=FF, norm_first=True).eval()
    import json
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ckpt")
        save_checkpoint(out, sudoku_cfg, model_cfg, model, vocab, MAX_LEN, 0.1, {}, [])
        path = os.path.join(out, "config.json")
        config = json.load(open(path))
        del config["arch_mode"], config["n_trainable_params"]
        json.dump(config, open(path, "w"))
        back, _v, cfg = load_checkpoint(out)
    assert "arch_mode" not in cfg and back.arch_mode == "full"
    assert back.encoder.layers[0].activation_relu_or_gelu == 1


def test_tags_are_distinct_and_full_is_unchanged():
    assert ARCH_TAG["full"] == "", "a full cell's directory name must not change"
    tags = [t for m, t in ARCH_TAG.items() if m != "full"]
    assert len(set(tags)) == len(tags) and all(t.startswith("_") for t in tags)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # not just AssertionError: a crash is a failure too
            failures.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
