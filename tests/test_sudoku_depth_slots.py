import numpy as np
import pytest

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_depth_slots import (
    COLORING, SELECTION, label_reasoning_slots, summarize,
)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_vocab


def _toy():
    """A 4-node, degree-2, 3-color instance with two rounds of sizes 2 and 1."""
    cfg = SudokuReasoningConfig(n_nodes=4, degree=2, n_colors=3, rho=0.25, rho_max=0.25)
    vocab = build_vocab(cfg)
    adj = np.array([[1, 2], [0, 3], [0, 3], [1, 2]], dtype=np.int64)
    trace = {
        "hints": {0: 0},
        "rounds": [[(1, 1), (2, 2)], [(3, 0)]],
        "resolved": True,
        "unresolved_nodes": [],
    }
    return vocab, adj, trace


def test_slots_alternate_selection_and_coloring_with_round_indices():
    vocab, adj, trace = _toy()
    tokens = encode_instance(adj, trace, vocab)
    slots = label_reasoning_slots(tokens, vocab)

    assert [s["slot"] for s in slots] == [SELECTION, COLORING] * 3
    assert [s["round_idx"] for s in slots] == [1, 1, 1, 1, 2, 2]
    assert [s["pos_in_reasoning"] for s in slots] == [0, 1, 2, 3, 4, 5]


def test_labeled_targets_are_the_tokens_they_claim_to_be():
    """The labels must point at the real emitted nodes and colors, in order."""
    vocab, adj, trace = _toy()
    tokens = encode_instance(adj, trace, vocab)
    slots = label_reasoning_slots(tokens, vocab)

    emitted = [(node, c) for rnd in trace["rounds"] for node, c in rnd]
    nodes = [tokens[s["target_index"]] for s in slots if s["slot"] is SELECTION]
    colors = [tokens[s["target_index"]] for s in slots if s["slot"] is COLORING]
    assert nodes == [vocab.node_token(n) for n, _ in emitted]
    assert colors == [vocab.color_token(c) for _, c in emitted]


def test_terminator_is_not_scored_as_a_slot():
    vocab, adj, trace = _toy()
    tokens = encode_instance(adj, trace, vocab)
    slots = label_reasoning_slots(tokens, vocab)
    assert all(int(tokens[s["target_index"]]) != vocab.SOLVED for s in slots)
    assert len(slots) == 2 * sum(len(r) for r in trace["rounds"])


def test_scratchpad_grammar_is_rejected_not_mislabeled():
    cfg = SudokuReasoningConfig(n_nodes=4, degree=2, n_colors=3, rho=0.25, rho_max=0.25,
                                scratchpad=True)
    vocab = build_vocab(cfg)
    _, adj, trace = _toy()
    tokens = encode_instance(adj, trace, vocab)
    with pytest.raises(ValueError, match="scratchpad"):
        label_reasoning_slots(tokens, vocab)


def _rows(spec):
    """spec: list of (round_idx, frac, n_correct, n_total) for one slot."""
    rows = []
    for round_idx, frac, n_correct, n_total in spec:
        for i in range(n_total):
            rows.append({
                "slot": COLORING, "round_idx": round_idx, "frac_through_reasoning": frac,
                "pos_in_reasoning": 0, "correct": int(i < n_correct),
            })
    return rows


def test_pure_position_effect_shows_no_round_effect_once_position_is_fixed():
    """The confound this module exists to catch, as a direct assertion.

    Accuracy here depends only on position, never on round index: inside each
    position bin every round scores the same. The marginal slope must be
    clearly negative (round and position advance together) while the
    position-controlled slope must be ~0, because there is no depth effect to
    find. If the control reported the marginal slope, every result from this
    module would be uninterpretable.
    """
    rows = _rows([
        (1, 0.05, 95, 100), (2, 0.05, 95, 100),   # early position, both rounds
        (3, 0.95, 55, 100), (4, 0.95, 55, 100),   # late position, both rounds
    ])
    _, controlled = summarize(rows, n_position_bins=2, min_cell=30)
    c = controlled[COLORING]
    assert c["marginal_slope"] < -0.05
    assert abs(c["round_effect_within_position"]) < 0.01
    assert c["n_position_bins_usable"] == 2


def test_real_round_effect_survives_the_position_control():
    """The complement: a depth effect present at every position must survive."""
    rows = _rows([
        (1, 0.05, 95, 100), (2, 0.05, 85, 100),
        (1, 0.95, 90, 100), (2, 0.95, 80, 100),
    ])
    _, controlled = summarize(rows, n_position_bins=2, min_cell=30)
    c = controlled[COLORING]
    assert c["round_effect_within_position"] == pytest.approx(-0.10, abs=0.01)


def test_thin_cells_are_dropped_rather_than_fitted():
    rows = _rows([(1, 0.5, 50, 60), (2, 0.5, 0, 5)])
    marginal, controlled = summarize(rows, n_position_bins=2, min_cell=30)
    assert [m["round_idx"] for m in marginal] == [1]
    # Only one usable round index remains, so no slope can be fitted at all.
    assert np.isnan(controlled[COLORING]["round_effect_within_position"])
