#!/usr/bin/env python3
"""Stage DQ3 figures -- stage DQ's figure code over THREE seeds instead of two.

WHY THIS IS A WRAPPER AND NOT A COPY. The three seed-coloured figures carry
the Clopper-Pearson censoring logic, the draw-the-limit-once rule and the
right-hand label de-collision, all three of which were defects found by
reading the PNGs back on 2026-11-22. Re-typing that math for a second stage is
how a fixed defect comes back. So this rebinds `SCANS`, `SPLITS` and `OUTDIR`
in `scripts/stagedq_figs.py` and calls its figure functions unchanged.

Seeds 0 and 1 live in the stage DQ directories and seed 2 in the stage DQ3
ones, so the (N, seed) keys are disjoint and the two sets of cells merge into
one axis: three seeds per panel, which is exactly the 3-slot SERIES palette,
in fixed seed order. That merged picture is the whole point -- E1 asks whether
the colour head is bimodal, and bimodality in two samples per size is a
coincidence while in three it is a pattern.

ONLY THE SEED-COLOURED FIGURES ARE REDRAWN. `fig_depth_vs_rho`, `fig_collapse`
and `fig_depth_vs_n` are properties of the INSTANCE ENSEMBLE, need no model,
are identical for DQ3, and are already published -- redrawing them here would
put a second copy of an unchanged figure in a second directory.

    python3 -m scripts.stagedq3_figs [--partial]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.stagedq_figs as F                                  # noqa: E402

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_DQ3 = os.path.join(SCRATCH, "spin_gnn/sudoku_dq3_2026-11-22")
SPLIT_DQ3 = os.path.join(SCRATCH, "spin_gnn/runs/stagedq3_2026-11-22")
OUT_DQ3 = "/home/users/hshunt/spin_gnn/runs/stagedq3_2026-11-22/figures"

# The figure functions read these out of the module dict at CALL time, which is
# why rebinding the attribute reaches them.
assert F.fig_d4.__globals__ is F.__dict__
assert F.fig_d1_reason_color.__globals__ is F.__dict__
assert F.fig_d5_trainability.__globals__ is F.__dict__

_OUT_BEFORE = F.OUTDIR
F.SCANS = [F.SCAN, SCAN_DQ3]
F.SPLITS = [F.SPLIT, SPLIT_DQ3]
F.OUTDIR = OUT_DQ3

# OUTDIR MUST MOVE. Stage DQ's figures are the provenance of a deck that is
# already published; writing three seeds over them would leave no record of
# what the two-seed pages were drawn from.
assert F.OUTDIR != _OUT_BEFORE, "OUTDIR did not move off stage DQ's figures"
assert SCAN_DQ3 in F.SCANS and SPLIT_DQ3 in F.SPLITS


def seed2_cells():
    """What stage DQ3 itself has contributed so far, by source."""
    return {
        "rho_curve": sorted(glob.glob(os.path.join(
            SCAN_DQ3, "rho_curve_n*_s*_*.csv"))),
        "history": sorted(glob.glob(os.path.join(
            SCAN_DQ3, "n*_d4_q3_L12_s*", "*", "history.json"))),
        "dqsplit": sorted(glob.glob(os.path.join(SPLIT_DQ3, "dqsplit_*.csv"))),
    }


def main(argv):
    partial = "--partial" in argv
    have = seed2_cells()
    for kind, paths in have.items():
        print("DQ3 %-9s %d file(s)" % (kind, len(paths)))
        for p in paths:
            print("    ", p)

    # A FIGURE IN THE DQ3 DIRECTORY WITH NO SEED 2 IN IT IS A LIE BY FILENAME.
    # 53313361 lands one cell at a time, so the honest states are "not yet"
    # and "complete"; the dangerous one is a file that looks like a DQ3 result
    # and shows stage DQ's two seeds. Refuse by default, and make --partial the
    # explicit way to look at the merge while the array is still running.
    if not any(have.values()) and not partial:
        print("\nREFUSING: stage DQ3 has contributed no cell yet, so every "
              "figure would show stage DQ's seeds 0 and 1 under a DQ3 "
              "filename. Re-run with --partial to draw it anyway.")
        return 2
    if not all(have.values()):
        print("\nPARTIAL: not every DQ3 source has landed. Seeds present in a "
              "panel are whichever cells exist right now -- do NOT put this "
              "in a deck.")

    os.makedirs(OUT_DQ3, exist_ok=True)
    F.fig_d5_trainability()
    F.fig_d1_reason_color()
    F.fig_d4()

    # Provenance beside the PNGs: which directories fed them, and which cells.
    with open(os.path.join(OUT_DQ3, "SOURCES.txt"), "w") as fh:
        fh.write("stage DQ3 figures -- merged sources\n")
        for d in F.SCANS:
            fh.write("scan   %s\n" % d)
        for d in F.SPLITS:
            fh.write("split  %s\n" % d)
        fh.write("\nDQ3 contribution at draw time:\n")
        for kind, paths in have.items():
            fh.write("  %-9s %d\n" % (kind, len(paths)))
            for p in paths:
                fh.write("      %s\n" % p)
        if not all(have.values()):
            fh.write("\nPARTIAL -- not every DQ3 source had landed.\n")
    print("wrote", os.path.join(OUT_DQ3, "SOURCES.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
