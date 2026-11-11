"""k distinct blocks, cycled: the rung between the shared block and the untied stack.

Stage S trained ONE block applied 16 times (rho_50 0.757) against an untied
stack of 16 (0.580), and stage S's S4 showed the whole gap is what the tied
cell's 0.011 lower token accuracy buys -- the relation is unchanged, only the
level. Stage U asks whether that level is set by one block having to serve 16
roles, by giving the stack k distinct blocks cycled num_layers/k times.

Every rung stays a single map iterated: with k blocks the step is
G = F_{k-1} o ... o F_0 and the network is G applied num_layers/k times. That is
the whole reason k is the knob and not "add more layers".

Three things here are silent failures if they are wrong, which is why each has a
test rather than a comment:

  1. k=1 must be the OLD tied path exactly. If it is not, stage S's result stops
     being the control for stage U and nothing in the output says so.
  2. The cycle order must be 0,1,...,k-1,0,... -- the composition, not a
     permutation of it, and not k blocks applied once each (that is the untied
     stack at depth k, a different and much shallower model that would still
     train and still report a number).
  3. k must reach config.json and come back. This is the finding-43 check: a
     k=2 checkpoint reloaded as k=1 has half its blocks, and while
     load_state_dict does raise here, the rung key is the part that fails
     quietly -- a k=2 curve keying as the shared-block rung pools into stage S.

No `import pytest`: the only pytest on Sherlock is py39 and this project is
py312 (see tests/_pytest_shim). Plain asserts, run with

    python3 tests/test_cycled_blocks.py
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.minarch_analysis import rung_key  # noqa: E402
from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningModelConfig  # noqa: E402
from spin_gnn.sat_transformer2 import (  # noqa: E402
    ApplicationHooks, SATReasoningTransformer, applied_blocks, is_tied, model_depth, n_blocks,
)
from spin_gnn.sudoku_tokens import Vocab  # noqa: E402
from spin_gnn.train_sudoku_transformer import load_checkpoint, save_checkpoint  # noqa: E402

D_MODEL, DEPTH, N_HEAD, FF, MAX_LEN, VOCAB = 16, 4, 2, 32, 32, 29


def build(k=1, mode="tied_no_mlp", depth=DEPTH, seed=0):
    torch.manual_seed(seed)
    return SATReasoningTransformer(
        VOCAB, d_model=D_MODEL, num_layers=depth, nhead=N_HEAD, max_len=MAX_LEN,
        dim_feedforward=FF, norm_first=True, arch_mode=mode, n_blocks=k,
    ).eval()


def tokens(n=2, length=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n, length), generator=g)


# --- k=1 is the cell stage S trained ---------------------------------------

def test_k1_is_byte_identical_to_the_shared_block_cell():
    """The control for the whole ladder: passing n_blocks=1 explicitly and not
    passing it at all must be the same network, parameter for parameter and
    logit for logit."""
    torch.manual_seed(0)
    default = SATReasoningTransformer(
        VOCAB, d_model=D_MODEL, num_layers=DEPTH, nhead=N_HEAD, max_len=MAX_LEN,
        dim_feedforward=FF, norm_first=True, arch_mode="tied_no_mlp",
    ).eval()
    explicit = build(k=1)
    assert len(default.encoder.layers) == len(explicit.encoder.layers) == 1
    assert n_blocks(default) == n_blocks(explicit) == 1
    x = tokens()
    with torch.no_grad():
        assert torch.equal(default(x), explicit(x))


def test_untied_stack_reports_its_depth_as_its_block_count():
    untied = build(mode="no_mlp")
    assert not is_tied(untied)
    assert n_blocks(untied) == model_depth(untied) == DEPTH


# --- the cycle --------------------------------------------------------------

def test_k_blocks_exist_and_the_depth_is_still_num_layers():
    for k in (1, 2, 4):
        m = build(k=k)
        assert len(m.encoder.layers) == k, k
        assert model_depth(m) == DEPTH, "k must not change the number of APPLICATIONS"
        assert n_blocks(m) == k


def test_blocks_start_identical_at_a_fixed_seed():
    """nn.TransformerEncoder deep-copies the layer it is given, so the k blocks
    are the SAME weights at init. That is what makes the ladder a statement
    about training rather than about having drawn k different blocks."""
    m = build(k=4)
    first = m.encoder.layers[0].state_dict()
    for i in range(1, 4):
        for key, v in m.encoder.layers[i].state_dict().items():
            assert torch.equal(v, first[key]), (i, key)
    # ... and distinct objects, so a gradient step can separate them.
    assert len({id(b) for b in m.encoder.layers}) == 4


def test_applied_blocks_cycles_in_order():
    m = build(k=2)
    got = [id(b) for b in applied_blocks(m)]
    want = [id(m.encoder.layers[t % 2]) for t in range(DEPTH)]
    assert got == want, "the cycle must be 0,1,0,1 -- not 0,0,1,1 and not 0,1"
    assert len(got) == DEPTH


def test_forward_equals_a_hand_applied_cycle():
    """forward() against the composition written out by hand. This is the test
    that would catch a cycle running k applications instead of num_layers."""
    m = build(k=2)
    x = tokens()
    with torch.no_grad():
        want = m(x)
        h = m.token_embed(x) + m.pos_embed[:x.shape[1]]
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), diagonal=1)
        for t in range(DEPTH):
            h = m.encoder.layers[t % 2](h, src_mask=mask)
        if m.encoder.norm is not None:
            h = m.encoder.norm(h)
        got = m.readout(h)
    assert torch.allclose(want, got, atol=1e-6)


def test_k_equal_depth_matches_the_untied_stack_at_the_same_init():
    """k=num_layers is the untied stack: same blocks, each applied once. The
    two differ only in which forward() branch runs, so they must agree."""
    cyc = build(k=DEPTH)
    unt = build(mode="no_mlp")
    for i in range(DEPTH):
        for key, v in cyc.encoder.layers[i].state_dict().items():
            assert torch.equal(v, unt.encoder.layers[i].state_dict()[key]), (i, key)
    x = tokens()
    with torch.no_grad():
        assert torch.allclose(cyc(x), unt(x), atol=1e-6)


def test_application_hooks_see_every_application_with_k_blocks():
    """A hook on layers[0] alone would see DEPTH/k applications and report them
    as rounds 0..DEPTH/k-1 -- a full-looking array of the wrong length."""
    for k in (1, 2, 4):
        m = build(k=k)
        seen = []
        with ApplicationHooks(m, lambda i, out: seen.append(i)):
            with torch.no_grad():
                m(tokens())
                m(tokens(seed=1))
        assert seen == list(range(DEPTH)) * 2, (k, seen)


def test_an_intervention_lands_on_one_application_only():
    """Round-indexed, not module-indexed: with k=2 an intervention aimed at
    round 1 must not also fire in round 3."""
    m = build(k=2)
    x = tokens()
    fired = []

    def at_round_1(i, out):
        fired.append(i)
        return out * 0.0 if i == 1 else None

    with torch.no_grad():
        base = m(x)
        with ApplicationHooks(m, at_round_1):
            hit = m(x)
    assert fired == list(range(DEPTH))
    assert not torch.allclose(base, hit), "zeroing round 1 must change the output"


# --- the guards -------------------------------------------------------------

def test_k_must_divide_the_depth():
    for k in (3, 5, 6):
        try:
            build(k=k, depth=4)
        except ValueError as exc:
            assert "does not divide" in str(exc), str(exc)
        else:
            raise AssertionError("k=%d does not divide depth 4 and must raise" % k)


def test_k_above_one_is_refused_for_an_untied_mode():
    for mode in ("full", "no_mlp", "linear"):
        try:
            build(k=2, mode=mode)
        except ValueError as exc:
            assert "tied_no_mlp" in str(exc), str(exc)
        else:
            raise AssertionError("n_blocks=2 must raise for arch_mode=%r" % mode)


def test_k_below_one_is_refused():
    for k in (0, -1):
        try:
            build(k=k)
        except ValueError as exc:
            assert "must be >= 1" in str(exc), str(exc)
        else:
            raise AssertionError("n_blocks=%d must raise" % k)


# --- what k costs -----------------------------------------------------------

def test_trainable_count_grows_by_one_block_per_k():
    """The per-block cost is paid k times and the embeddings once, which is the
    arithmetic the stage U registration predicts its cells from."""
    counts = {k: build(k=k).trainable_parameter_count() for k in (1, 2, 4)}
    per_block = counts[2] - counts[1]
    assert per_block > 0
    assert counts[4] - counts[1] == 3 * per_block, counts
    # The MLP write is zeroed and frozen in EVERY block, not just the first.
    for k in (1, 2, 4):
        m = build(k=k)
        for blk in m.encoder.layers:
            assert torch.count_nonzero(blk.linear2.weight) == 0
            assert not blk.linear2.weight.requires_grad
            assert not blk.linear1.weight.requires_grad


# --- the finding-43 check ---------------------------------------------------

def test_checkpoint_round_trip_preserves_k():
    sudoku_cfg = SudokuReasoningConfig(n_nodes=12, degree=3, n_colors=3, shuffle_adj=True)
    vocab = Vocab(12, 3, 3)
    x = torch.randint(0, min(VOCAB, vocab.size), (2, 9))
    for k in (1, 2, 4):
        model_cfg = SudokuReasoningModelConfig(
            d_model=D_MODEL, num_layers=DEPTH, nhead=N_HEAD, dim_feedforward=FF,
            norm_first=True, arch_mode="tied_no_mlp", n_blocks=k,
        )
        torch.manual_seed(0)
        model = SATReasoningTransformer(
            vocab.size, d_model=D_MODEL, num_layers=DEPTH, nhead=N_HEAD, max_len=MAX_LEN,
            dim_feedforward=FF, norm_first=True, arch_mode="tied_no_mlp", n_blocks=k,
        ).eval()
        with torch.no_grad():
            want = model(x).clone()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "ckpt")
            save_checkpoint(out, sudoku_cfg, model_cfg, model, vocab, MAX_LEN, 0.1, {}, [])
            cfg = json.load(open(os.path.join(out, "config.json")))
            assert cfg["n_blocks"] == k, cfg.get("n_blocks")
            back, _v, cfg2 = load_checkpoint(out)
            assert n_blocks(back) == k and model_depth(back) == DEPTH
            with torch.no_grad():
                got = back(x)
        assert torch.allclose(want, got, atol=1e-6), "k=%d does not survive a round trip" % k


def test_pre_2026_11_11_checkpoints_read_as_one_block():
    """A config.json with no n_blocks key is a shared-block cell, which is what
    every tied checkpoint before stage U is."""
    sudoku_cfg = SudokuReasoningConfig(n_nodes=12, degree=3, n_colors=3)
    vocab = Vocab(12, 3, 3)
    model_cfg = SudokuReasoningModelConfig(
        d_model=D_MODEL, num_layers=DEPTH, nhead=N_HEAD, dim_feedforward=FF,
        norm_first=True, arch_mode="tied_no_mlp",
    )
    torch.manual_seed(0)
    model = SATReasoningTransformer(
        vocab.size, d_model=D_MODEL, num_layers=DEPTH, nhead=N_HEAD, max_len=MAX_LEN,
        dim_feedforward=FF, norm_first=True, arch_mode="tied_no_mlp",
    ).eval()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ckpt")
        save_checkpoint(out, sudoku_cfg, model_cfg, model, vocab, MAX_LEN, 0.1, {}, [])
        path = os.path.join(out, "config.json")
        cfg = json.load(open(path))
        del cfg["n_blocks"]
        json.dump(cfg, open(path, "w"))
        back, _v, cfg2 = load_checkpoint(out)
    assert "n_blocks" not in cfg2 and n_blocks(back) == 1


# --- the key that keeps the rungs apart -------------------------------------

def _row(k=1, t=None, L=16):
    return {"num_layers": L, "nhead": 4, "arch_mode": "tied_no_mlp", "d_model": 320,
            "dim_feedforward": 1024, "warmup_epochs": 0, "epochs": 50,
            "n_blocks": k, "n_iters_eval": t if t is not None else L}


def test_rung_key_separates_k_from_the_shared_block_cell():
    assert rung_key(_row(k=1)) != rung_key(_row(k=2))
    assert rung_key(_row(k=2)) != rung_key(_row(k=4))


def test_rung_key_is_unchanged_for_every_pre_stage_u_row():
    """k=1 and a row with no n_blocks at all must key exactly as before."""
    row = _row(k=1)
    bare = {k: v for k, v in row.items() if k != "n_blocks"}
    assert rung_key(row) == rung_key(bare)
    assert rung_key(bare) == (16, 4, "tied_no_mlp", 320, 1024, 0)


def test_k_does_not_collide_with_an_eval_t_of_the_same_number():
    """The reason k is keyed as a string: k=2 at the trained depth and k=1
    evaluated at T=2 are different models and must not share a key."""
    assert rung_key(_row(k=2)) != rung_key(_row(k=1, t=2))
    assert rung_key(_row(k=2, t=24)) != rung_key(_row(k=1, t=24))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("\n%d/%d passed" % (len(tests) - len(failures), len(tests)))
    sys.exit(1 if failures else 0)
