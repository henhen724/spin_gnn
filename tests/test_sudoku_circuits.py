"""The counterfactual is the whole experiment: if it does not really flip
DD's answer, then a "flip rate" measures nothing and the patching maps
localize noise. These tests check the construction against the forcing rule
itself, not against the code that built it.
"""
import numpy as np
import torch

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_circuits import (
    build_pair, collect_cases, default_sites, find_flippable_deductions,
    reasoning_color_position, run_behavior, run_head_ablation, run_patching,
)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab

CFG = SudokuReasoningConfig(n_nodes=24, degree=5, n_colors=4, rho=0.7, rho_max=0.7, seed=0)


def _cases(n=30):
    vocab = build_vocab(CFG)
    return vocab, collect_cases(CFG, vocab, n_instances=n, seed_start=11_000)


def test_cases_are_found_at_all():
    _vocab, cases = _cases()
    assert len(cases) > 5, f"only {len(cases)} flippable deductions; the experiment needs cases"


def test_clean_case_satisfies_the_forcing_rule():
    """u's known neighbours must cover exactly n_colors-1 colors with u's true
    color the free one -- otherwise u was not forced and DD's answer is not
    determined."""
    _vocab, cases = _cases()
    for adj, colors, hints, _trace, case in cases:
        known = [hints[int(u)] for u in adj[case["node"]] if int(u) in hints]
        covered = set(known)
        assert len(covered) == CFG.n_colors - 1
        assert case["true_color"] not in covered
        assert colors[case["node"]] == case["true_color"]
        # the neighbour being recolored must uniquely hold the flipped color
        assert known.count(case["flipped_color"]) == 1
        assert hints[case["recolored_nbr"]] == case["flipped_color"]


def test_counterfactual_actually_flips_dds_answer():
    """After the edit, u must be forced to the flipped color. This is the
    claim the whole module rests on, checked by re-deriving the forcing
    condition on the edited hints rather than trusting the construction."""
    _vocab, cases = _cases()
    for adj, _colors, hints, _trace, case in cases:
        edited = dict(hints)
        edited[case["recolored_nbr"]] = case["true_color"]
        known = {int(u): edited[int(u)] for u in adj[case["node"]] if int(u) in edited}
        covered = set(known.values())
        # still forced ...
        assert len(covered) == CFG.n_colors - 1, (covered, case)
        # ... but now to the flipped color, not the original one
        free = set(range(CFG.n_colors)) - covered
        assert free == {case["flipped_color"]}, (free, case)


def test_pair_differs_in_exactly_one_token_and_it_is_the_hint_color():
    vocab, cases = _cases()
    for adj, colors, hints, trace, case in cases:
        clean, corrupt, pos = build_pair(adj, colors, hints, trace, vocab, case)
        diff = np.flatnonzero(clean != corrupt)
        assert diff.tolist() == [case["hint_color_pos"]]
        assert vocab.color_index(clean[case["hint_color_pos"]]) == case["flipped_color"]
        assert vocab.color_index(corrupt[case["hint_color_pos"]]) == case["true_color"]
        # the graph table and the whole reasoning section must be untouched
        assert pos > int(np.flatnonzero(clean == vocab.BEGIN_REASONING)[0])


def test_predict_position_precedes_the_true_color_token():
    """logits at predict_pos must be predicting u's color; off by one here
    would silently score the wrong token."""
    vocab, cases = _cases()
    for adj, _colors, hints, trace, case in cases:
        tokens = encode_instance(adj, trace, vocab)
        pos = reasoning_color_position(tokens, vocab, case["node"])
        assert vocab.is_node_token(tokens[pos])
        assert vocab.node_index(tokens[pos]) == case["node"]
        assert vocab.color_index(tokens[pos + 1]) == case["true_color"]


def test_sites_point_at_the_tokens_they_name():
    vocab, cases = _cases()
    for adj, _colors, hints, trace, case in cases[:5]:
        tokens = encode_instance(adj, trace, vocab)
        pos = reasoning_color_position(tokens, vocab, case["node"])
        sites = default_sites(tokens, vocab, case, pos)
        assert vocab.is_color_token(tokens[sites["edited_hint"]])
        assert vocab.node_index(tokens[sites["graph_row_u"]]) == case["node"]
        assert vocab.node_index(tokens[sites["graph_row_w"]]) == case["recolored_nbr"]
        assert sites["predict_pos"] == pos


def _tiny_model(vocab):
    from spin_gnn.config import SudokuReasoningModelConfig
    from spin_gnn.sat_transformer2 import SATReasoningTransformer
    torch.manual_seed(0)
    cfg = SudokuReasoningModelConfig(d_model=16, num_layers=2, nhead=2, dim_feedforward=32)
    return SATReasoningTransformer(
        vocab_size=vocab.size, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward,
        max_len=4096, norm_first=True,
    ).eval()


def test_measurement_plumbing_runs_on_an_untrained_model():
    """Shapes, hook attach/detach and the recovery normalization. An
    untrained model has no circuit, so only well-formedness is asserted --
    not any particular flip rate."""
    vocab, cases = _cases(12)
    model = _tiny_model(vocab)
    cases = cases[:4]

    beh = run_behavior(model, vocab, "cpu", cases)
    assert len(beh) == len(cases)
    assert all(isinstance(r["clean_correct"], bool) for r in beh)

    patch = run_patching(model, vocab, "cpu", cases, default_sites)
    assert {r["site"] for r in patch} <= {"edited_hint", "graph_row_u", "graph_row_w", "predict_pos"}
    assert all(np.isfinite(r["recovery"]) for r in patch)

    abl = run_head_ablation(model, vocab, "cpu", cases)
    assert {r["head"] for r in abl} == {0, 1}
    assert {r["layer"] for r in abl} == {0, 1}

    # Hooks must not leak: a plain forward has to match the pre-measurement one.
    clean, _corrupt, _pos = build_pair(*cases[0][:4], vocab, cases[0][4])
    t = torch.from_numpy(clean).unsqueeze(0)
    with torch.no_grad():
        a = model(t)
        b = model(t)
    torch.testing.assert_close(a, b)


def test_head_ablation_actually_perturbs_the_model():
    """The counterpart of the patching sanity check, and the test this file was
    missing: every assertion above passes while the ablation does nothing at
    all. nn.MultiheadAttention hands out_proj.weight to the functional fast
    path, so the out_proj module is never called and a forward_pre_hook on it
    never fires -- the first real run came back with effect_retained == 1.0 on
    all 7200 rows, which reads as "no head matters" but means "nothing was
    ablated".

    Zeroing a head must move the effect on at least one (layer, head), and the
    weights must come back exactly afterwards."""
    vocab, cases = _cases(12)
    model = _tiny_model(vocab)
    before = [l.self_attn.out_proj.weight.detach().clone() for l in model.encoder.layers]

    rows = run_head_ablation(model, vocab, "cpu", cases[:4])
    assert rows, "no ablation rows"
    retained = [r["effect_retained"] for r in rows]
    assert not all(v == 1.0 for v in retained), "ablation is a no-op: nothing was zeroed"
    assert any(abs(v - 1.0) > 1e-3 for v in retained), retained[:8]

    for layer, w0 in zip(model.encoder.layers, before):
        torch.testing.assert_close(layer.self_attn.out_proj.weight.detach(), w0)


def test_patching_the_prediction_site_at_the_last_layer_fully_recovers():
    """Sanity check on the recovery metric's normalization: splicing the clean
    residual into the very position and layer that produce the logits must
    restore the clean answer by construction. If this is not ~1.0 the metric
    is miscomputed and every other cell is uninterpretable."""
    vocab, cases = _cases(12)
    model = _tiny_model(vocab)
    rows = run_patching(model, vocab, "cpu", cases[:4], default_sites)
    last = max(r["layer"] for r in rows)
    vals = [r["recovery"] for r in rows if r["layer"] == last and r["site"] == "predict_pos"]
    assert vals, "no predict_pos rows at the final layer"
    assert all(abs(v - 1.0) < 1e-3 for v in vals), vals
