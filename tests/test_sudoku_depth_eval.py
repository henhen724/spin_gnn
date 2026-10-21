"""Tests for depth binning at fixed hint density.

The point of sudoku_depth_eval is to separate reasoning depth from hint
density, and the separation is only as good as the binning, so the binning is
what gets pinned here. The model-dependent half is exercised by the job that
uses it, not by a test that would have to train something.
"""
import numpy as np
import pytest
from dataclasses import replace

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_depth_eval import bin_by_depth
from spin_gnn.train_sudoku_transformer import build_instances


def _fake(t):
    """An instance tuple with only the fields bin_by_depth touches."""
    return (None, None, (), {"rounds": [()] * t})


def test_bins_are_keyed_by_round_count_and_sorted():
    bins, dropped = bin_by_depth([_fake(3), _fake(1), _fake(3), _fake(2)], min_per_bin=1)
    assert list(bins) == [1, 2, 3]
    assert [len(v) for v in bins.values()] == [1, 1, 2]
    assert dropped == 0


def test_thin_bins_are_dropped_and_counted_not_merged():
    bins, dropped = bin_by_depth([_fake(1)] * 5 + [_fake(2)] * 2 + [_fake(3)], min_per_bin=3)
    assert list(bins) == [1]
    # 2 at T=2 plus 1 at T=3; merging them into a "T=2-3" bin would blur the
    # one axis this module exists to resolve, so they are discarded.
    assert dropped == 3


def test_no_bin_survives_an_impossible_threshold():
    bins, dropped = bin_by_depth([_fake(1), _fake(2)], min_per_bin=99)
    assert bins == {}
    assert dropped == 2


@pytest.mark.parametrize("rho", [0.6, 0.75])
def test_fixed_rho_really_pins_density_across_depth_bins(rho):
    """The validity condition for the whole measurement.

    If deep-T instances systematically carried fewer hints than shallow-T
    ones, binning by T at "fixed" rho would just be the rho sweep again under
    another name. Pinning rho_max to rho has to make realized density flat
    across bins, and that is cheap enough to assert directly.
    """
    cfg = SudokuReasoningConfig(n_nodes=24, degree=5, n_colors=4)
    pool = build_instances(replace(cfg, rho=rho, rho_max=rho), 120, 30_000_000_000)
    bins, _ = bin_by_depth(pool, min_per_bin=5)
    assert len(bins) >= 2, "no depth spread at this rho, so there is nothing to separate"

    realized = [
        sum(len(h) for _, _, h, _ in v) / (len(v) * cfg.n_nodes) for v in bins.values()
    ]
    # One hint out of 24 nodes is 0.042; the transition sweep spans rho 0.3-0.95,
    # so anything at the 0.01 level is negligible against the effect of interest.
    assert max(realized) - min(realized) < 0.01

    # And depth must not be a proxy for density within the pool.
    depth = np.array(list(bins), dtype=float)
    if np.std(realized) > 0:
        assert abs(np.corrcoef(np.array(realized), depth)[0, 1]) < 0.95
