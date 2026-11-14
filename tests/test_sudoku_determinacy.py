"""The stage-X auxiliary target is DD's own deducibility predicate.

sudoku_determinacy.determinacy_targets is a lean reimplementation of two
columns of sudoku_activations.dd_state_labels, written because the full
function builds eight (P, N) arrays and re-encodes the instance, which at
240,000 training instances is work the training loop does not need. The lean
path is therefore only allowed to exist if it agrees with the tested one
exactly, which is what these tests pin.

They also pin the two properties the training loop relies on but
dd_state_labels does not state: that the label is never positive for a node
already on the page, and that the node the trace emits next is always a
positive.
"""
import numpy as np

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_activations import dd_state_labels, reasoning_positions
from spin_gnn.sudoku_determinacy import MASKED, determinacy_targets, summarize
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab


def _instances(n=6, n_nodes=24, shuffle_adj=True, seed_start=0):
    cfg = SudokuReasoningConfig(
        n_nodes=n_nodes, degree=5, n_colors=4, rho=0.6, rho_max=0.9,
        shuffle_adj=shuffle_adj,
    )
    return cfg, build_vocab(cfg), build_instances(cfg, n, seed_start=seed_start)


def test_matches_dd_state_labels_exactly():
    """The oracle. determined == uncoloured and n_known_colors == q-1, read off
    dd_state_labels at its pre_node rows, must equal the lean target cell for
    cell -- including which cells are masked."""
    cfg, vocab, instances = _instances()
    q = cfg.n_colors
    checked = 0
    for adj, _c, _h, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        pos, target = determinacy_targets(adj, trace, vocab, tokens)

        positions, lab = dd_state_labels(adj, trace, vocab)
        keep = np.flatnonzero(lab["kind_is_node"])
        assert list(pos) == [positions[i]["pos"] for i in keep]

        known = lab["known_mask"][keep].astype(bool)
        oracle = np.where(known, MASKED, (lab["n_known_colors"][keep] == q - 1).astype(np.int8))
        assert np.array_equal(target, oracle.astype(np.int8))
        checked += target.size
    assert checked > 0


def test_positions_are_the_pre_node_ones_and_nothing_else():
    cfg, vocab, instances = _instances()
    for adj, _c, _h, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        pos, target = determinacy_targets(adj, trace, vocab, tokens)
        expected = [p["pos"] for p in reasoning_positions(tokens, vocab) if p["kind"] == "pre_node"]
        assert list(pos) == expected
        assert target.shape == (len(expected), cfg.n_nodes)
        # One pre_node position per deduction the trace ever emits.
        assert len(expected) == sum(len(rd) for rd in trace["rounds"])


def test_the_node_emitted_next_is_always_a_positive():
    """The label has to agree with the trace it is built from: DD emits a node
    only when that node is deducible, so the very node tokens[pos+1] names must
    carry a 1. If this fails the aux head is being trained against the main
    loss."""
    cfg, vocab, instances = _instances()
    for adj, _c, _h, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        pos, target = determinacy_targets(adj, trace, vocab, tokens)
        for p, tok_pos in enumerate(pos):
            node = vocab.node_index(int(tokens[tok_pos + 1]))
            assert target[p, node] == 1


def test_coloured_nodes_are_masked_never_scored():
    cfg, vocab, instances = _instances()
    for adj, _c, _h, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        _pos, target = determinacy_targets(adj, trace, vocab, tokens)
        # Hints are coloured at every position, so they are masked everywhere.
        for u in trace["hints"]:
            assert np.all(target[:, int(u)] == MASKED)
        # Masking only ever grows along the trace: once a node is coloured it
        # stays coloured.
        masked = (target == MASKED)
        assert np.all(masked[:-1] <= masked[1:])


def test_both_classes_are_well_populated():
    """Sanity bound used by the dataset-build printout, and the reason the
    scored set is the OPEN nodes only.

    Measured at N=48 d=5 q=4 on 200 instances: 15.6% of cells are scored at
    all (the rest are already coloured), and of those 65.4% are positive --
    7.5 open nodes at the median position, 4.9 of them deducible. Both
    classes are therefore well populated, which is what the loss needs; the
    bounds below are loose enough to pass at other N and tight enough to
    catch a target that has collapsed to one class."""
    _cfg, vocab, instances = _instances(n=8, n_nodes=48)
    targets = []
    for adj, _c, _h, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        targets.append(determinacy_targets(adj, trace, vocab, tokens)[1])
    stats = summarize(targets)
    assert stats["positives"] > 0 and stats["negatives"] > 0
    assert 0.1 < stats["positive_rate"] < 0.9
    assert 0.0 < stats["scored_fraction"] <= 1.0
