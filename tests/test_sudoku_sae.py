"""SAE mechanics and the DD-predicate feature scoring.

The scoring is the part that can silently lie (a bug there would invent or
hide evidence of DD structure), so it is checked against planted features
whose correct answer is known by construction.
"""
import numpy as np
import torch

from spin_gnn.sudoku_sae import SparseAutoencoder, dd_predicates, score_features, train_sae


def _fake_labels(P, n_nodes=6, n_colors=4, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "frontier_mask": rng.integers(0, 2, size=(P, n_nodes)).astype(np.int8),
        "remaining_mask": rng.integers(0, 2, size=(P, n_nodes)).astype(np.int8),
        "known_mask": rng.integers(0, 2, size=(P, n_nodes)).astype(np.int8),
        "node_color": rng.integers(0, n_colors + 1, size=(P, n_nodes)).astype(np.int64),
        "n_known_nbrs": rng.integers(0, 4, size=(P, n_nodes)).astype(np.int64),
        "n_known_colors": rng.integers(0, n_colors + 1, size=(P, n_nodes)).astype(np.int64),
        "nbr_of_next_mask": rng.integers(0, 2, size=(P, n_nodes)).astype(np.int8),
        "justifier_mask": rng.integers(0, 2, size=(P, n_nodes)).astype(np.int8),
        "next_node": rng.integers(0, n_nodes, size=P).astype(np.int64),
        "next_color": rng.integers(0, n_colors, size=P).astype(np.int64),
        "kind_is_node": rng.integers(0, 2, size=P).astype(bool),
        "round_index": np.zeros(P, dtype=np.int64),
    }


def test_decoder_columns_stay_unit_norm():
    """The L1 penalty is only meaningful if the decoder cannot grow to cheat."""
    sae = SparseAutoencoder(d_model=8, d_sae=32)
    norms = sae.W_dec.data.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms))
    with torch.no_grad():
        sae.W_dec.data *= 7.5
    sae.normalize_decoder()
    norms = sae.W_dec.data.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_sae_recovers_a_planted_sparse_code():
    """Data built as a sparse mix of 6 fixed directions must be reconstructed
    with high variance explained AND recover the sparsity -- 6 atoms firing at
    15% means a true L0 near 0.9.

    This is also the calibration that set the module defaults. A high FVE on
    its own proves nothing: at l1_coeff=1e-3/400 epochs this same problem
    reconstructs at FVE=0.99 with L0~21 of 48 features, a dense and
    uninterpretable dictionary that looks healthy on reconstruction alone. The
    sparsity term bites slowly, so the assertion here is deliberately on L0."""
    rng = np.random.default_rng(0)
    d_model, n_atoms, P = 12, 6, 4000
    dictionary = rng.normal(size=(n_atoms, d_model))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    codes = (rng.random((P, n_atoms)) < 0.15) * rng.random((P, n_atoms)) * 3.0
    x = torch.as_tensor(codes @ dictionary, dtype=torch.float32)

    _sae, feats, history, _ = train_sae(x, d_sae=48, l1_coeff=0.1, epochs=1500, lr=3e-3, seed=0)
    assert history[-1]["frac_variance_explained"] > 0.95, history[-1]
    assert history[-1]["l0"] < 4.0, history[-1]
    # Only 6 directions are needed out of 48, so most features should die.
    assert history[-1]["n_dead_features"] > 5, history[-1]
    assert feats.shape == (P, 48)


def test_score_features_identifies_a_planted_frontier_feature():
    """A feature that fires exactly when frontier[2] holds must be labelled
    frontier[2] with F1 == 1."""
    P, n_nodes, n_colors = 400, 6, 4
    labels = _fake_labels(P, n_nodes, n_colors)
    feats = torch.zeros(P, 3)
    feats[:, 0] = torch.as_tensor(labels["frontier_mask"][:, 2].astype(np.float32))
    feats[:, 1] = torch.as_tensor((labels["next_node"] == 4).astype(np.float32))
    rows = score_features(feats, labels, n_nodes, n_colors)

    assert rows[0]["best_predicate"] == "frontier[2]"
    assert rows[0]["best_f1"] == 1.0
    assert rows[1]["best_predicate"] == "next_node[4]"
    assert rows[1]["best_f1"] == 1.0
    # A feature that never fires must not be credited with any predicate.
    assert rows[2]["firing_rate"] == 0.0
    assert rows[2]["best_f1"] == 0.0
    assert not rows[2]["is_interpretable"]


def test_score_features_does_not_credit_noise():
    """A randomly firing feature must not reach the interpretability
    threshold -- the guard against reading structure into noise."""
    P, n_nodes, n_colors = 2000, 6, 4
    labels = _fake_labels(P, n_nodes, n_colors, seed=1)
    rng = np.random.default_rng(2)
    feats = torch.as_tensor((rng.random((P, 8)) < 0.1).astype(np.float32))
    rows = score_features(feats, labels, n_nodes, n_colors, min_f1=0.3)
    assert not any(r["is_interpretable"] for r in rows), [r["best_f1"] for r in rows]


def test_predicate_set_covers_the_named_families():
    labels = _fake_labels(50, n_nodes=6, n_colors=4)
    names = [n for n, _ in dd_predicates(labels, 6, 4)]
    for expected in ["frontier[0]", "remaining[0]", "known[5]", "next_node[3]", "color[2]=1",
                     "next_color=0", "kind_is_node",
                     # the edge families: which neighbours are in play right now
                     "justifies_next[0]", "nbr_of_next[5]"]:
        assert expected in names, expected
    assert any(n.startswith("next_node_nbr_count=") for n in names)
    assert any(n.startswith("next_node_distinct_colors=") for n in names)
    # Every predicate must be a (P,)-shaped boolean mask.
    for _name, mask in dd_predicates(labels, 6, 4):
        assert mask.shape == (50,) and mask.dtype == bool


def test_score_features_identifies_a_planted_edge_feature():
    """A feature firing exactly when node 3 is a justifying neighbour of the
    node being emitted must be labelled justifies_next[3] -- the edge-level
    half of "relevant nodes and edges at the relevant times"."""
    P, n_nodes, n_colors = 400, 6, 4
    labels = _fake_labels(P, n_nodes, n_colors)
    feats = torch.zeros(P, 2)
    feats[:, 0] = torch.as_tensor(labels["justifier_mask"][:, 3].astype(np.float32))
    feats[:, 1] = torch.as_tensor(labels["nbr_of_next_mask"][:, 1].astype(np.float32))
    rows = score_features(feats, labels, n_nodes, n_colors)
    assert rows[0]["best_predicate"] == "justifies_next[3]"
    assert rows[0]["best_f1"] == 1.0 and rows[0]["beats_null"]
    assert rows[1]["best_predicate"] == "nbr_of_next[1]"
    assert rows[1]["best_f1"] == 1.0


def test_noise_features_do_not_beat_their_own_null():
    """The argmax over a few hundred predicates buys a substantial F1 for free,
    so best_f1 alone cannot separate a real match from the search.

    Noise features firing at 30% reach best_f1 ~0.40 here -- a number that
    would look like a finding in a table -- and their rolled null reaches the
    same value, which is the whole point: all of it is the argmax. `beats_null`
    is a per-feature randomization test at level 1/(shifts+1), so a few of the
    12 clearing it is the expected false-positive rate and not a bug; what must
    hold is that no noise feature clears it by a MARGIN."""
    P, n_nodes, n_colors = 3000, 6, 4
    labels = _fake_labels(P, n_nodes, n_colors, seed=5)
    rng = np.random.default_rng(6)
    feats = torch.as_tensor((rng.random((P, 12)) < 0.3).astype(np.float32))
    rows = score_features(feats, labels, n_nodes, n_colors, n_null_shifts=19, seed=1)
    assert all(r["best_f1"] > 0.1 for r in rows), "null control is untested if raw F1 is ~0"
    margins = [r["best_f1"] - r["best_f1_null"] for r in rows]
    assert max(margins) < 0.02, sorted(margins)[-3:]
    # And the threshold alone, without the null, would have credited all 12.
    assert all(r["best_f1"] >= 0.3 for r in rows)


def test_aliveness_floor_excludes_single_position_features():
    """firing_rate>0 called all 2048 features alive in the 2026-10-19 run. A
    feature firing once in P positions is not a feature."""
    P, n_nodes, n_colors = 2000, 6, 4
    labels = _fake_labels(P, n_nodes, n_colors, seed=7)
    feats = torch.zeros(P, 3)
    feats[0, 0] = 1.0                                     # fires once: 5e-4
    feats[:, 1] = torch.as_tensor(labels["known_mask"][:, 0].astype(np.float32))
    rows = score_features(feats, labels, n_nodes, n_colors, alive_rate=1e-3)
    assert rows[0]["firing_rate"] > 0 and not rows[0]["is_alive"]
    assert rows[0]["n_fire"] == 1
    assert rows[1]["is_alive"]
    assert not rows[2]["is_alive"]


def test_target_l0_search_rescues_a_dense_dictionary():
    """A fixed l1_coeff does not transport across activation widths: the
    reconstruction term is a SUM over dimensions, so it grows with d_model
    while the L1 term does not. l1_coeff=0.1 (calibrated at d_model=12) left
    the real d_model=128 run at L0=84..278 of 2048 with ZERO dead features --
    dense, hence uninterpretable, while still reporting FVE=0.992.

    Here a wide planted problem reproduces that, and the search must fix it."""
    from spin_gnn.sudoku_sae import train_sae_to_target_l0
    rng = np.random.default_rng(3)
    d_model, n_atoms, P = 96, 10, 3000
    dictionary = rng.normal(size=(n_atoms, d_model))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    codes = (rng.random((P, n_atoms)) < 0.2) * rng.random((P, n_atoms)) * 3.0
    x = torch.as_tensor(codes @ dictionary, dtype=torch.float32)

    # The starting l1 is the one that was too weak; the target is set below
    # what it achieves, so the search has to escalate to succeed.
    _loose, _f, hist_loose, _ = train_sae(x, d_sae=256, l1_coeff=0.1, epochs=300, lr=3e-3, seed=0)
    target = 4.0
    assert hist_loose[-1]["l0"] > target + 2.0, "planted problem too easy to test the search"

    _sae, _feats, hist, _ = train_sae_to_target_l0(
        x, d_sae=256, target_l0=target, l1_start=0.1, epochs=300, lr=3e-3, seed=0,
    )
    assert hist[-1]["l0"] < hist_loose[-1]["l0"], (hist[-1]["l0"], hist_loose[-1]["l0"])
    # Two-sided: landing far BELOW the target is a failure too, see
    # test_target_l0_search_backs_off_when_the_start_is_already_too_strong.
    assert abs(hist[-1]["l0"] - target) <= 2.0, hist[-1]["l0"]
    assert hist[-1]["hit_target_l0"]
    # It had to raise l1 above the start to get there, and the path must be
    # recorded so a dictionary that never got sparse is visible in the output
    # instead of being reported as a clean result.
    assert hist[-1]["l1_coeff"] > 0.1, hist[-1]["l1_coeff"]
    assert len(hist[-1]["l1_search_path"]) > 1


def test_target_l0_search_backs_off_when_the_start_is_already_too_strong():
    """The regression test for job 49702811.

    That run passed l1_start=1.0 against target_l0=64. The first fit of every
    one of its 18 sweeps collapsed to L0=2..16, which satisfied the old
    one-sided acceptance test `l0 <= target + tolerance`, so the search broke
    on iteration 1, reported hit_target_l0=True, and handed back dictionaries
    with ~130 of 4096 features alive. Scored against a ground truth of ~56
    simultaneously-true predicates, those dictionaries reported zero F1 on the
    edge families -- a fact about the dictionary, not about the model.

    So: an l1_start that is already too STRONG must be walked back down."""
    from spin_gnn.sudoku_sae import train_sae_to_target_l0
    rng = np.random.default_rng(5)
    d_model, n_atoms, P = 32, 12, 800
    dictionary = rng.normal(size=(n_atoms, d_model))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    codes = (rng.random((P, n_atoms)) < 0.4) * rng.random((P, n_atoms)) * 3.0
    x = torch.as_tensor(codes @ dictionary, dtype=torch.float32)

    # l1_start=6 drives this problem to L0=0.7 on the first fit, i.e. the
    # collapse regime; the search has to come back DOWN from there.
    target, l1_start = 6.0, 6.0
    _sae, _feats, hist, _ = train_sae_to_target_l0(
        x, d_sae=128, target_l0=target, l1_start=l1_start, max_tries=6,
        epochs=150, lr=3e-3, seed=0,
    )
    path = hist[-1]["l1_search_path"]
    assert path[0]["l0"] < target - hist[-1]["l0_tolerance"], (
        "l1_start must collapse the dictionary for this test to exercise the bug, "
        f"got L0={path[0]['l0']}")
    # The old code could not do either of these: it only multiplied l1 upward,
    # and it returned the SPARSEST fit tried rather than the closest one.
    assert len(path) > 1, "search broke on the first fit, as in 49702811"
    assert hist[-1]["l1_coeff"] < l1_start, hist[-1]["l1_coeff"]
    assert abs(hist[-1]["l0"] - target) <= hist[-1]["l0_tolerance"], hist[-1]["l0"]
    assert hist[-1]["hit_target_l0"]


def test_target_l0_search_reports_failure_instead_of_hiding_it():
    """Given an unreachable target and no room to search, the result must come
    back flagged, since the caller's feature labels depend on it."""
    from spin_gnn.sudoku_sae import train_sae_to_target_l0
    rng = np.random.default_rng(4)
    x = torch.as_tensor(rng.normal(size=(600, 24)), dtype=torch.float32)
    _sae, _feats, hist, _ = train_sae_to_target_l0(
        x, d_sae=64, target_l0=0.001, l1_start=1e-6, max_tries=2, epochs=60, lr=3e-3, seed=0,
    )
    assert hist[-1]["hit_target_l0"] is False
    assert len(hist[-1]["l1_search_path"]) == 2


def test_position_mask_separates_a_free_predicate_from_a_computed_one():
    """The control for SAE pre-registration (a) of job 49729375.

    At `pre_color` positions `pos = i`, so `tokens[pos]` IS the node about to be
    coloured and `next_node[v]` is matched for free by any token-identity
    feature; at `pre_node` positions `pos = i - 1` and it is not. A count of
    layer-0 `next_node` features therefore cannot settle anything unless the two
    kinds of position are scored separately.

    Planted: one feature firing only at pre_color positions where next_node==3
    (the freebie) and one firing whenever next_node==3 (the computed version).
    Scored on all positions the two carry the SAME predicate label, which is the
    bug this control exists for; under the masks the freebie must vanish at
    pre_node while the computed one survives both."""
    P, n_nodes, n_colors = 1200, 6, 4
    labels = _fake_labels(P, n_nodes, n_colors, seed=11)
    is_node = labels["kind_is_node"].astype(bool)
    is_next = labels["next_node"] == 3
    feats = torch.zeros(P, 2)
    feats[:, 0] = torch.as_tensor((is_next & ~is_node).astype(np.float32))
    feats[:, 1] = torch.as_tensor(is_next.astype(np.float32))

    full = score_features(feats, labels, n_nodes, n_colors)
    assert full[1]["best_predicate"] == "next_node[3]" and full[1]["best_f1"] == 1.0
    assert full[0]["best_predicate"] == "next_node[3]", "the two are indistinguishable unmasked"
    assert 0.5 < full[0]["best_f1"] < 1.0, full[0]["best_f1"]

    color = score_features(feats, labels, n_nodes, n_colors, position_mask=~is_node)
    node = score_features(feats, labels, n_nodes, n_colors, position_mask=is_node)
    assert color[0]["best_predicate"] == "next_node[3]" and color[0]["best_f1"] == 1.0
    assert color[1]["best_predicate"] == "next_node[3]" and color[1]["best_f1"] == 1.0
    # The freebie does not fire at all once its kind of position is removed.
    assert node[0]["n_fire"] == 0 and not node[0]["is_alive"]
    assert node[0]["best_f1"] == 0.0
    assert node[1]["best_predicate"] == "next_node[3]" and node[1]["best_f1"] == 1.0
    # Counts are recomputed inside the mask, not carried over from all positions.
    assert node[1]["n_fire"] == int((is_next & is_node).sum())
    assert color[1]["n_fire"] == int((is_next & ~is_node).sum())
    assert node[1]["n_fire"] + color[1]["n_fire"] == full[1]["n_fire"]


def test_position_mask_rejects_a_mismatched_or_empty_mask():
    """A mask of the wrong length would silently score a shifted subset, and an
    empty one would report F1=0 for every feature as if that were a result."""
    import pytest
    P, n_nodes, n_colors = 200, 6, 4
    labels = _fake_labels(P, n_nodes, n_colors, seed=12)
    rng = np.random.default_rng(0)
    feats = torch.as_tensor((rng.random((P, 2)) < 0.2).astype(np.float32))
    with pytest.raises(ValueError, match="entries for"):
        score_features(feats, labels, n_nodes, n_colors,
                       position_mask=np.ones(P - 1, dtype=bool))
    with pytest.raises(ValueError, match="keeps"):
        score_features(feats, labels, n_nodes, n_colors,
                       position_mask=np.zeros(P, dtype=bool))
