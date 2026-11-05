"""Pin the stage D/E decision rule to the sbatch header that registered it.

scripts/capacity_back.py exists so that the verdict on stages D (51151957) and
E (51171824) cannot be chosen after the cells land. That guarantee is only
worth anything if the thresholds in the script still match the sbatch headers
they were transcribed from, and if the branches fire in the registered ORDER.
Both are easy to break by a one-character edit that no eval would catch: move
BARRIER_LO from 0.72 to 0.74 and three quarters of the tie-break region
silently becomes a falsification of the barrier reading.

So this file checks two things and nothing else:

  1. the constants still equal the numbers written in
     scripts/slurm/sudoku2_n48_minarch_E_2026-11-04.sbatch, both as literal
     text in the header and as the arithmetic that produced them (hub + s*),
  2. barrier_call returns the registered status at every boundary, including
     the ones that matter most -- that trainability outranks a crossing
     median, and that an incomplete rung can never falsify anything.

The third thing worth saying is not testable and is left to the docstring: the
header's E1 tie-break and its E2 falsifier DISAGREE on [0.72, 0.740), and this
script follows the tie-break. test_tiebreak_region_reports_the_conflict just
makes sure that disagreement stays visible in the printed verdict.

No `import pytest`: the only pytest module on Sherlock is py39 and this
project is py312 (STATUS Open problems). Run with

    python3 tests/test_capacity_back.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.capacity_back import (BARRIER_HI, BARRIER_LO, E2_CAPACITY, FREE,  # noqa: E402
                                   HUB, S_MEAN, S_RHO50, barrier_call)
from tests import _pytest_shim  # noqa: E402

HEADER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "slurm", "sudoku2_n48_minarch_E_2026-11-04.sbatch")

# The two measured rungs the registration built its thresholds on. Quoted here
# as bare numbers on purpose: if the analysis ever recomputes them, this test
# fails and the registration is re-read rather than quietly re-derived.
HUB_RHO50, FULL_L8_RHO50 = 0.592, 0.471


def _summary(n=3, n_cross=3, rho50=0.75, readable=True):
    """The subset of summarize()'s dict that barrier_call actually reads."""
    return {"n": n, "n_cross": n_cross, "rho50": rho50, "rho50_readable": readable}


def test_header_exists():
    assert os.path.exists(HEADER), HEADER


def test_constants_appear_verbatim_in_the_registration():
    text = open(HEADER, encoding="utf-8").read()
    for literal in (f"[{BARRIER_LO:.2f}, {BARRIER_HI:.2f}]",
                    f"{FREE:.3f}", f"{E2_CAPACITY:.3f}"):
        assert literal in text, f"{literal!r} is no longer in the stage E header"


def test_thresholds_are_the_arithmetic_the_header_wrote():
    # 0.592 + s* = 0.740 (E1 under (B)); 0.471 + s* = 0.619 (E2 under (B)).
    assert round(HUB_RHO50 + S_RHO50, 3) == FREE
    assert round(FULL_L8_RHO50 + S_RHO50, 3) == E2_CAPACITY


def test_frozen_bar_and_hub_are_not_recomputed():
    assert (S_RHO50, S_MEAN) == (0.1484, 0.2918)
    assert HUB == (6, 4, "full")


def test_trainability_outranks_a_crossing_median():
    """0 escapes wins even when the rung would otherwise kill (A).

    This is the trap stage C's P2 fell into, inverted: an untrained cell must
    not be read as evidence in EITHER direction.
    """
    text, status = barrier_call(_summary(rho50=0.40), n_esc=0)
    assert status == "untrained", status
    assert "did not train" in text.lower()


def test_incomplete_rung_cannot_falsify():
    assert barrier_call(_summary(n=2, rho50=0.40), n_esc=2)[1] == "incomplete"
    assert barrier_call(_summary(n=1, rho50=0.40), n_esc=1)[1] == "incomplete"


def test_one_crossing_seed_cannot_fire_the_falsifier():
    """The falsifier needs >=2 of 3 crossing seeds, so n_cross=1 is inconclusive."""
    assert barrier_call(_summary(n_cross=1, rho50=0.40), n_esc=3)[1] == "censored"


def test_censored_median_is_not_a_rung_that_held():
    status = barrier_call(_summary(rho50=float("nan"), readable=False), n_esc=3)[1]
    assert status == "censored", status


def test_falsification_boundary_is_exactly_BARRIER_LO():
    assert barrier_call(_summary(rho50=BARRIER_LO - 0.001), n_esc=3)[1] == "falsified"
    assert barrier_call(_summary(rho50=BARRIER_LO), n_esc=3)[1] == "held"


def test_two_crossing_seeds_are_enough_to_falsify():
    text, status = barrier_call(_summary(n_cross=2, rho50=0.60), n_esc=3)
    assert status == "falsified", status
    assert "2 of 3" in text


def test_free_boundary_goes_to_A_not_to_B():
    """0.740 exactly is (A). Registered in advance because P3 landed on its bar."""
    assert barrier_call(_summary(rho50=FREE), n_esc=3)[1] == "held"
    assert barrier_call(_summary(rho50=FREE + 0.001), n_esc=3)[1] == "held"


def test_tiebreak_region_reports_the_conflict():
    text, _ = barrier_call(_summary(rho50=0.730), n_esc=3)
    assert "falsifier" in text.lower() and "0.740" in text, text


def test_barrier_window_upper_edge():
    assert barrier_call(_summary(rho50=BARRIER_HI), n_esc=3)[1] == "held"
    assert barrier_call(_summary(rho50=BARRIER_HI + 0.001), n_esc=3)[1] == "worse"


def test_worse_than_the_band_is_not_a_vote_for_either_reading():
    text, status = barrier_call(_summary(rho50=0.90), n_esc=3)
    assert status == "worse", status
    assert "not a vote" in text


def test_every_branch_returns_a_known_status():
    known = {"untrained", "incomplete", "censored", "falsified", "held", "worse"}
    cases = [(_summary(rho50=r), e)
             for r in (0.40, 0.719, 0.72, 0.73, 0.740, 0.76, 0.80, 0.95)
             for e in (0, 1, 3)]
    cases += [(_summary(n=2), 2), (_summary(n_cross=0), 3),
              (_summary(rho50=float("nan"), readable=False), 3)]
    for s, e in cases:
        assert barrier_call(s, e)[1] in known


if __name__ == "__main__":
    sys.exit(_pytest_shim.run(globals()))
