"""rung_label must name the axis the cell actually varies.

The bug these cover: rung_label's seventh positional parameter was `epochs`, so
every tail element landing there was formatted as an epoch count whatever it
was. Stage T's T=24 cell printed "ep24" and stage U's k=2 cell would have
printed "epk2" -- in acc_vs_solve's held-out listings, escape_timing's tables
and any deck page built from them. Nothing raised; the label was simply wrong.

The matching half of the fix is in rung_key, which now writes the eval-T tail
as the string "T24". Epochs stays a bare int because stage G's scorer freezes
keys containing it, so the three tails are told apart by TYPE.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.minarch_analysis import EP_DEFAULT, cell_key, rung_key, rung_label, sort_key


def _row(k=1, t=None, L=16, arch="tied_no_mlp", d=320, ep=EP_DEFAULT):
    return {"num_layers": L, "nhead": 4, "arch_mode": arch, "d_model": d,
            "dim_feedforward": 1024, "warmup_epochs": 0, "epochs": ep,
            "n_blocks": k, "n_iters_eval": t if t is not None else L}


# --- the two labels that were wrong ------------------------------------------

def test_eval_t_is_labelled_as_applications_not_epochs():
    label = rung_label(*rung_key(_row(t=24)))
    assert "T24" in label, label
    assert "ep" not in label, label


def test_k_is_labelled_as_blocks_not_epochs():
    label = rung_label(*rung_key(_row(k=2)))
    assert "k2" in label, label
    assert "ep" not in label, label


def test_k_and_eval_t_both_appear_and_keep_their_axes():
    label = rung_label(*rung_key(_row(k=4, t=8)))
    assert "k4" in label and "T8" in label, label
    assert "ep" not in label, label


# --- what must not have changed ----------------------------------------------

def test_epochs_still_labels_as_epochs():
    """Stage G's shape: a bare int tail is still an epoch count."""
    label = rung_label(*rung_key(_row(arch="no_mlp", d=128, ep=150)))
    assert "ep150" in label, label


def test_default_epochs_never_appears():
    assert "ep" not in rung_label(*rung_key(_row(ep=EP_DEFAULT)))


def test_plain_rungs_are_unchanged():
    assert rung_label(6, 4, "full") == "L=6  h=4 full"
    assert rung_label(6, 4, "no_mlp", 384, 1024, 0) == "L=6  h=4 no_mlp d384"


def test_a_tail_element_of_an_unknown_shape_raises():
    """A new tail must get a case here rather than a plausible-looking label."""
    for bad in ("x2", "k", "kk", 1.5, None):
        try:
            rung_label(16, 4, "tied_no_mlp", 320, 1024, 0, bad)
        except ValueError:
            continue
        raise AssertionError("tail element %r was formatted instead of refused" % (bad,))


# --- the grouping is a relabelling, not a regrouping --------------------------

def test_eval_t_as_a_string_still_separates_every_t():
    keys = {rung_key(_row(t=t)) for t in (4, 8, 12, 20, 24)}
    assert len(keys) == 5, keys
    assert rung_key(_row(t=16)) == (16, 4, "tied_no_mlp", 320, 1024, 0)


def test_k_still_does_not_collide_with_an_eval_t_of_the_same_number():
    assert rung_key(_row(k=2)) != rung_key(_row(t=2))
    assert rung_key(_row(k=2, t=24)) != rung_key(_row(t=24))


# --- sorting keys of mixed shape ---------------------------------------------

def _cell(seed, **kw):
    r = _row(**kw)
    r["seed"] = seed
    return cell_key(r)


def test_a_string_tail_and_a_seed_can_be_sorted_together():
    """The TypeError job 51931433 found: a 'T24' tail meets a seed.

    (16, 4, tied_no_mlp, 320, 1024, 0, 'T24', 0) against
    (16, 4, tied_no_mlp, 320, 1024, 0, 0) -- position 6 is str against int.
    """
    keys = [_cell(0, t=24), _cell(0), _cell(1), _cell(2, t=4), _cell(0, k=2)]
    try:
        sorted(keys)
    except TypeError:
        pass
    else:
        raise AssertionError("expected the raw sort to raise; the guard is now untested")
    assert len(sorted(keys, key=sort_key)) == len(keys)


def test_sort_key_orders_ints_numerically_not_lexically():
    assert sorted([(2,), (10,)], key=sort_key) == [(2,), (10,)]
    assert sorted([(10,), (2,)], key=sort_key) == [(2,), (10,)]


def test_sort_key_is_total_over_every_tail_shape():
    keys = [_cell(s, k=k, t=t)
            for s in (0, 1, 2) for k in (1, 2, 4) for t in (None, 4, 24)]
    keys.append(_cell(0, arch="no_mlp", d=128, ep=150))
    out = sorted(keys, key=sort_key)
    assert len(out) == len(keys)
    assert sorted(out, key=sort_key) == out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print("  ok   %s" % fn.__name__)
        except AssertionError as e:
            failures.append((fn.__name__, e))
            print("  FAIL %s: %s" % (fn.__name__, e))
    print("%d/%d passed" % (len(tests) - len(failures), len(tests)))
    sys.exit(1 if failures else 0)
