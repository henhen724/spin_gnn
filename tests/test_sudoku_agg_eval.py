import numpy as np
import pytest

from spin_gnn.sudoku_agg_eval import (
    check_comparable, instance_signature, mcnemar_exact_p, wilson_interval,
)


def test_no_disagreement_is_no_evidence():
    """Two models that never disagree cannot be ranked, whatever their rate."""
    assert mcnemar_exact_p(0, 0) == 1.0


def test_symmetric_discordance_is_a_coin_flip():
    assert mcnemar_exact_p(10, 10) == pytest.approx(1.0)
    assert mcnemar_exact_p(3, 3) == pytest.approx(1.0)


def test_hand_checked_small_case():
    # b=1, c=4: two-sided p = 2 * (C(5,0) + C(5,1)) / 2^5 = 2 * 6/32 = 0.375
    assert mcnemar_exact_p(1, 4) == pytest.approx(0.375)
    # b=0, c=5: 2 * 1/32 = 0.0625
    assert mcnemar_exact_p(0, 5) == pytest.approx(0.0625)


def test_direction_does_not_change_the_p_value():
    for b, c in [(2, 9), (0, 7), (13, 30)]:
        assert mcnemar_exact_p(b, c) == pytest.approx(mcnemar_exact_p(c, b))


def test_lopsided_discordance_is_significant():
    assert mcnemar_exact_p(2, 25) < 0.001
    assert mcnemar_exact_p(15, 25) > 0.05


def test_pairing_beats_independent_rates_near_the_ceiling():
    """Why this module exists, as an assertion rather than a claim.

    Two models at 0.98 and 0.9667 over 1500 instances have overlapping Wilson
    intervals, so on independent rates they are indistinguishable. If every
    instance the weaker one solves is also solved by the stronger one, the
    disagreement is 20-0 and the paired test is decisive.
    """
    n = 1500
    strong = np.zeros(n, dtype=bool); strong[:1470] = True   # 0.980
    weak = np.zeros(n, dtype=bool); weak[:1450] = True       # 0.9667

    s_lo, s_hi = wilson_interval(int(strong.sum()), n)
    w_lo, w_hi = wilson_interval(int(weak.sum()), n)
    assert w_hi > s_lo, "intervals should overlap, which is the point"

    b = int((weak & ~strong).sum())
    c = int((~weak & strong).sum())
    assert (b, c) == (0, 20)
    assert mcnemar_exact_p(b, c) < 1e-5


def test_wilson_interval_stays_inside_the_unit_range():
    lo, hi = wilson_interval(1500, 1500)
    assert 0.0 <= lo <= 1.0 and hi == pytest.approx(1.0)
    lo, hi = wilson_interval(0, 1500)
    assert lo == pytest.approx(0.0) and 0.0 <= hi <= 1.0
    # The normal interval would exceed 1 here; Wilson must not.
    lo, hi = wilson_interval(1485, 1500)
    assert hi < 1.0 and lo < 0.99


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(1470, 1500)
    assert lo < 1470 / 1500 < hi


def test_mismatched_instance_distributions_are_refused():
    # Built through instance_signature rather than by hand: check_comparable
    # indexes INSTANCE_KEYS directly, so a literal dict here goes stale (with a
    # KeyError, not a useful failure) the next time a key is added -- as it did
    # when shuffle_adj arrived on 2026-10-26.
    a = instance_signature({"n_nodes": 24, "degree": 5, "n_colors": 4, "rho": 0.3, "rho_max": 0.95,
                            "hint_mode": "removal", "scratchpad": False})
    b = dict(a, rho=0.5)
    with pytest.raises(SystemExit, match="different instance distribution"):
        check_comparable([a, b], ["ref", "other"])
    # Identical signatures must pass and return the shared one.
    assert check_comparable([a, dict(a)], ["ref", "other"]) == a


def test_a_config_predating_a_key_is_read_as_having_used_the_default():
    """Regression: the first guard I wrote refused the comparison I built it for.

    sudoku_hparam_sweep/d0256... was written before hint_mode and scratchpad
    existed, so config.get() returns None for both. That means "used the
    default", not "used something else"; treating it as a mismatch blocked
    every comparison against the original sweep, which is the whole point of a
    shared pool.
    """
    old = {"n_nodes": 24, "degree": 5, "n_colors": 4, "rho": 0.3, "rho_max": 0.95}
    new = dict(old, hint_mode="removal", scratchpad=False)
    assert instance_signature(old) == instance_signature(new)
    assert check_comparable(
        [instance_signature(new), instance_signature(old)], ["new", "old"]
    ) == instance_signature(new)

    # An explicit non-default must still be caught.
    odd = dict(old, scratchpad=True)
    with pytest.raises(SystemExit, match="different instance distribution"):
        check_comparable(
            [instance_signature(new), instance_signature(odd)], ["new", "odd"]
        )


def test_signature_ignores_everything_that_is_not_the_question():
    """d_model and epochs differ between the models being compared; those must
    not be treated as making the instances incomparable."""
    cfg = {"n_nodes": 24, "degree": 5, "n_colors": 4, "rho": 0.3, "rho_max": 0.95,
           "hint_mode": "removal", "scratchpad": False, "d_model": 384, "epochs": 300}
    other = dict(cfg, d_model=256, epochs=900, n_train_instances=5000)
    assert instance_signature(cfg) == instance_signature(other)
