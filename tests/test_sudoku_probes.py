"""Tests for the threshold-free probe metrics.

These exist because the frontier_mask result turned on which metric was
quoted: balanced accuracy thresholds a BCE probe's logits at 0, which on a
~15%-positive target reports a floor rather than what the probe's direction
separates. AUROC and average precision replace it, so they have to be right.
"""
import numpy as np
import pytest

from spin_gnn.sudoku_probes import auroc, average_precision


def _balanced_accuracy(pred, y):
    """The metric fit_linear_probe reports, written out independently."""
    pred, y = pred > 0.5, y > 0.5
    tpr = (pred & y).sum() / max(1, y.sum())
    tnr = (~pred & ~y).sum() / max(1, (~y).sum())
    return 0.5 * (tpr + tnr)


def test_auroc_perfect_and_inverted():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == pytest.approx(1.0)
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == pytest.approx(0.0)


def test_auroc_constant_score_is_chance():
    # All ties: every positive/negative pair contributes a half.
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    assert auroc(np.full(5, 3.0), y) == pytest.approx(0.5)


def test_auroc_known_value_with_one_swap():
    # 2 positives x 3 negatives = 6 pairs; the lowest positive loses to one
    # negative, so 5/6 pairs are correctly ordered.
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    scores = np.array([0.0, 1.0, 5.0, 4.0, 9.0])
    assert auroc(scores, y) == pytest.approx(5 / 6)


def test_auroc_is_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.15).astype(float)
    s = rng.normal(size=400) + y  # weakly informative, like a real probe
    base = auroc(s, y)
    assert auroc(10 * s - 7, y) == pytest.approx(base)
    assert auroc(1 / (1 + np.exp(-s)), y) == pytest.approx(base)


def test_auroc_of_a_binary_rule_equals_balanced_accuracy():
    """The identity the hand baselines rely on.

    `not_known` and `degree_aug` emit 0/1, not scores, so their AUROC and
    their balanced accuracy must coincide -- otherwise switching the probe
    sweep to AUROC would silently change what they are being compared against.
    """
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.15).astype(float)
    # A deliberately lopsided rule: high recall, poor precision, which is
    # exactly the regime that flatters a hand rule under balanced accuracy.
    pred = ((y > 0.5) | (rng.random(600) < 0.4)).astype(float)
    assert auroc(pred, y) == pytest.approx(_balanced_accuracy(pred, y))


def test_average_precision_chance_is_the_base_rate():
    rng = np.random.default_rng(2)
    y = (rng.random(20_000) < 0.15).astype(float)
    ap = average_precision(rng.normal(size=20_000), y)
    assert ap == pytest.approx(0.15, abs=0.02)


def test_average_precision_perfect_ranking_is_one():
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    assert average_precision(np.array([9.0, 8.0, 1.0, 0.5, 0.0]), y) == pytest.approx(1.0)


def test_average_precision_known_value():
    # Ranked: pos, neg, pos, neg -- precision at the two hits is 1/1 and 2/3.
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert average_precision(np.array([4.0, 3.0, 2.0, 1.0]), y) == pytest.approx(
        (1.0 + 2 / 3) / 2
    )


@pytest.mark.parametrize("labels", [np.zeros(8), np.ones(8)])
def test_degenerate_labels_are_nan_not_a_crash(labels):
    scores = np.arange(8, dtype=float)
    assert np.isnan(auroc(scores, labels))
    # AP is defined when there is at least one positive.
    if labels.sum() == 0:
        assert np.isnan(average_precision(scores, labels))
    else:
        assert average_precision(scores, labels) == pytest.approx(1.0)
