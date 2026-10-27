"""Correctness of the DD state labels and residual capture.

Every interpretability conclusion downstream rests on these labels being
exactly right, so they are checked against the DD invariants directly rather
than against a reimplementation of the same walk.
"""
import numpy as np
import torch

from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from spin_gnn.sat_transformer2 import SATReasoningTransformer
from spin_gnn.sudoku_activations import (
    ResidualCapture, collect_dataset, dd_state_labels, reasoning_positions,
)
from spin_gnn.sudoku_tokens import encode_instance
from spin_gnn.train_sudoku_transformer import build_instances, build_vocab


def _instances(n=6, scratchpad=False, hint_mode="removal", seed_start=0):
    cfg = SudokuReasoningConfig(
        n_nodes=24, degree=5, n_colors=4, rho=0.6, rho_max=0.9,
        scratchpad=scratchpad, hint_mode=hint_mode,
    )
    return cfg, build_vocab(cfg), build_instances(cfg, n, seed_start=seed_start)


def test_positions_point_at_the_tokens_they_claim():
    """pos must satisfy: tokens[pos+1] is the deduced node (pre_node) or its
    color (pre_color), for every deduction in every round."""
    cfg, vocab, instances = _instances()
    checked = 0
    for adj, _c, _h, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        positions = reasoning_positions(tokens, vocab)
        rounds = [[(int(u), int(c)) for u, c in rd] for rd in trace["rounds"]]
        assert len(positions) == 2 * sum(len(rd) for rd in rounds)
        for info in positions:
            nxt = int(tokens[info["pos"] + 1])
            node, color = rounds[info["round"]][info["k"]]
            if info["kind"] == "pre_node":
                assert vocab.is_node_token(nxt) and vocab.node_index(nxt) == node
            else:
                assert vocab.is_color_token(nxt) and vocab.color_index(nxt) == color
            checked += 1
    assert checked > 0


def test_frontier_mask_is_exactly_the_round_being_emitted():
    """The identity the whole module relies on: DD uses only pre-round
    knowledge, so the deducible set at round r IS round r's event list."""
    cfg, vocab, instances = _instances()
    for adj, _c, _h, trace in instances:
        positions, lab = dd_state_labels(adj, trace, vocab)
        for p, info in enumerate(positions):
            expected = {int(u) for u, _ in trace["rounds"][info["round"]]}
            assert set(np.flatnonzero(lab["frontier_mask"][p])) == expected


def test_frontier_nodes_satisfy_the_dd_forcing_condition():
    """A node is deducible exactly when its KNOWN neighbors cover n_colors-1
    distinct colors, leaving one free -- checked against known_mask/node_color
    at the same position, which is what the probes actually see."""
    cfg, vocab, instances = _instances()
    checked = 0
    for adj, _c, _h, trace in instances:
        positions, lab = dd_state_labels(adj, trace, vocab)
        for p, info in enumerate(positions):
            # Only round-start positions see the untouched pre-round state.
            if info["k"] != 0 or info["kind"] != "pre_node":
                continue
            for v in np.flatnonzero(lab["frontier_mask"][p]):
                nbr_colors = {
                    int(lab["node_color"][p, int(u)]) for u in adj[v]
                    if lab["known_mask"][p, int(u)]
                }
                assert len(nbr_colors) == cfg.n_colors - 1, (
                    f"node {v} is in the frontier but its known neighbors cover "
                    f"{len(nbr_colors)} colors, not {cfg.n_colors - 1}"
                )
                assert int(lab["next_color"][p]) not in nbr_colors or int(lab["next_node"][p]) != v
                checked += 1
    assert checked > 0, "no round-start frontier nodes examined; test is vacuous"


def test_known_mask_excludes_same_round_siblings_not_yet_emitted():
    cfg, vocab, instances = _instances()
    checked = 0
    for adj, _c, _h, trace in instances:
        positions, lab = dd_state_labels(adj, trace, vocab)
        for p, info in enumerate(positions):
            rd = [int(u) for u, _ in trace["rounds"][info["round"]]]
            for u in rd[info["k"]:]:
                assert not lab["known_mask"][p, u], (
                    "a deduction not yet written out must not count as known"
                )
                checked += 1
            for u in rd[: info["k"]]:
                assert lab["known_mask"][p, u]
    assert checked > 0


def test_known_mask_and_hints_agree_at_the_first_position():
    cfg, vocab, instances = _instances()
    for adj, _c, hints, trace in instances:
        positions, lab = dd_state_labels(adj, trace, vocab)
        if not positions:
            continue
        assert set(np.flatnonzero(lab["known_mask"][0])) == {int(k) for k in hints}
        for node, color in hints.items():
            assert int(lab["node_color"][0, int(node)]) == int(color)


def test_n_known_nbrs_matches_the_adjacency_table():
    cfg, vocab, instances = _instances(n=3)
    for adj, _c, _h, trace in instances:
        positions, lab = dd_state_labels(adj, trace, vocab)
        for p in range(0, len(positions), 5):
            for v in range(cfg.n_nodes):
                expected = sum(1 for u in adj[v] if lab["known_mask"][p, int(u)])
                assert lab["n_known_nbrs"][p, v] == expected


def test_n_known_colors_is_the_quantity_dd_thresholds():
    """DD deduces v exactly when v's known neighbors show n_colors-1 DISTINCT
    colors. n_known_nbrs can reach that count without the deduction being
    available (two known neighbors sharing a color), so this is the stronger
    label: at every position, the node about to be emitted must have exactly
    n_colors-1 distinct colors on its known incident edges."""
    cfg, vocab, instances = _instances(n=3)
    for adj, _c, _h, trace in instances:
        positions, lab = dd_state_labels(adj, trace, vocab)
        for p in range(len(positions)):
            for v in range(cfg.n_nodes):
                seen = {int(lab["node_color"][p, int(u)]) for u in adj[v]
                        if lab["known_mask"][p, int(u)]}
                assert lab["n_known_colors"][p, v] == len(seen)
                assert lab["n_known_colors"][p, v] <= lab["n_known_nbrs"][p, v]
            nxt = int(lab["next_node"][p])
            assert lab["n_known_colors"][p, nxt] == cfg.n_colors - 1


def test_justifier_mask_is_the_known_neighborhood_of_the_next_node():
    """The edge-level label: exactly the known neighbors of the node about to
    be emitted, i.e. the edges forcing its color. Their colors must be the
    n_colors-1 colors other than the one DD is about to write."""
    cfg, vocab, instances = _instances(n=3)
    for adj, _c, _h, trace in instances:
        positions, lab = dd_state_labels(adj, trace, vocab)
        for p in range(len(positions)):
            nxt = int(lab["next_node"][p])
            nbrs = {int(u) for u in adj[nxt]}
            assert set(np.flatnonzero(lab["nbr_of_next_mask"][p])) == nbrs
            just = set(np.flatnonzero(lab["justifier_mask"][p]))
            assert just == {u for u in nbrs if lab["known_mask"][p, u]}
            assert just <= nbrs
            # The justifying colors are every color except the forced one.
            assert ({int(lab["node_color"][p, u]) for u in just}
                    == set(range(cfg.n_colors)) - {int(lab["next_color"][p])})


def test_residual_capture_layer0_is_the_embedding():
    cfg, vocab, instances = _instances(n=1)
    model = SATReasoningTransformer(vocab.size, d_model=32, num_layers=3, nhead=2, max_len=512, norm_first=True)
    model.eval()
    adj, _c, _h, trace = instances[0]
    tokens = torch.as_tensor(encode_instance(adj, trace, vocab)[None, :], dtype=torch.long)
    with ResidualCapture(model) as cap:
        acts = cap.run(tokens)
    assert acts.shape[0] == 4  # embedding + 3 blocks
    assert acts.shape[1] == 1 and acts.shape[2] == tokens.shape[1] and acts.shape[3] == 32
    expected_emb = model.token_embed(tokens) + model.pos_embed[: tokens.shape[1]]
    torch.testing.assert_close(acts[0], expected_emb.detach())
    # The blocks must actually change the stream, or the hooks are mis-wired.
    assert not torch.allclose(acts[0], acts[-1])


def test_collect_dataset_shapes_and_label_alignment():
    cfg, vocab, instances = _instances(n=4)
    model = SATReasoningTransformer(vocab.size, d_model=32, num_layers=2, nhead=2, max_len=512, norm_first=True)
    model.eval()
    acts, labels = collect_dataset(model, instances, vocab, "cpu", max_positions_per_instance=5)
    assert acts.shape[0] == 3
    assert acts.shape[2] == 32
    p_total = acts.shape[1]
    assert 0 < p_total <= 4 * 5
    for key, arr in labels.items():
        assert arr.shape[0] == p_total, f"label {key} has {arr.shape[0]} rows, expected {p_total}"
    assert labels["frontier_mask"].shape[1] == cfg.n_nodes


def test_labels_work_in_scratchpad_mode():
    """The evidence rows must be skipped, not mistaken for deductions."""
    cfg, vocab, instances = _instances(n=3, scratchpad=True)
    for adj, _c, _h, trace in instances:
        tokens = encode_instance(adj, trace, vocab)
        positions = reasoning_positions(tokens, vocab)
        assert len(positions) == 2 * sum(len(rd) for rd in trace["rounds"])
        rounds = [[(int(u), int(c)) for u, c in rd] for rd in trace["rounds"]]
        for info in positions:
            nxt = int(tokens[info["pos"] + 1])
            node, color = rounds[info["round"]][info["k"]]
            if info["kind"] == "pre_node":
                assert vocab.node_index(nxt) == node
            else:
                assert vocab.color_index(nxt) == color


def test_labels_work_on_unresolved_random_hint_instances():
    """Random-hint instances stop at UNRESOLVED; the walk must terminate
    cleanly there and still label every round it did emit."""
    cfg, vocab, instances = _instances(n=10, hint_mode="random", seed_start=1234)
    saw_unresolved = False
    for adj, _c, _h, trace in instances:
        if not trace["resolved"]:
            saw_unresolved = True
        positions, lab = dd_state_labels(adj, trace, vocab)
        assert len(positions) == 2 * sum(len(rd) for rd in trace["rounds"])
        for p, info in enumerate(positions):
            assert set(np.flatnonzero(lab["frontier_mask"][p])) == {
                int(u) for u, _ in trace["rounds"][info["round"]]
            }
    assert saw_unresolved, "expected at least one unresolved instance in this seed range"
