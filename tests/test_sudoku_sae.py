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
                     "next_color=0", "kind_is_node"]:
        assert expected in names, expected
    assert any(n.startswith("next_node_nbr_count=") for n in names)
    # Every predicate must be a (P,)-shaped boolean mask.
    for _name, mask in dd_predicates(labels, 6, 4):
        assert mask.shape == (50,) and mask.dtype == bool


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
    assert hist[-1]["l0"] <= target + 2.0, hist[-1]["l0"]
    assert hist[-1]["hit_target_l0"]
    # It had to raise l1 above the start to get there, and the path must be
    # recorded so a dictionary that never got sparse is visible in the output
    # instead of being reported as a clean result.
    assert hist[-1]["l1_coeff"] > 0.1, hist[-1]["l1_coeff"]
    assert len(hist[-1]["l1_search_path"]) > 1


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
