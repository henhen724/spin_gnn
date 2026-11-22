#!/usr/bin/env python3
"""Stage DQ3 section split -- stage DQ's probe, pointed at the seed-2 cells.

THIS FILE DELIBERATELY CONTAINS NO MEASUREMENT CODE. `scripts/stagedq_probe.py`
produced the CSVs that stage DQ's verdict was scored from, and that scorer is
now a published artefact: editing it to take a second directory would mean the
code that produced `VERDICT.txt` is no longer the code in the tree. So this
rebinds three module globals and calls its `main()` unchanged.

The pattern is the one stage DQ itself used on `stageentropy_probe`, and that
redirect is not assumed to work -- DQ-P1 proved it by reproducing an already
published CSV on 20 of 20 quantities to 1e-8, and DQ-P1 still runs here, since
`main()` calls it before touching a single DQ3 checkpoint.

    python3 -m scripts.stagedq3_probe --task 0 --ntasks 2 --out <dir>
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dq = importlib.import_module("scripts.stagedq_probe")

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_DQ3 = os.path.join(SCRATCH, "spin_gnn/sudoku_dq3_2026-11-22")
OUT_DQ3 = os.path.join(SCRATCH, "spin_gnn/runs/stagedq3_2026-11-22")
CELLS_DQ3 = [(96, 2), (120, 2)]

# `resolve_cell` reads SCAN_DQ, and `main` reads CELLS and OUT, out of the
# module dict at CALL time -- which is why rebinding the attribute reaches
# them and rebinding a local name here would not.
assert dq.resolve_cell.__globals__ is dq.__dict__, \
    "stagedq_probe.resolve_cell does not resolve globals from its own module"
assert dq.main.__globals__ is dq.__dict__, \
    "stagedq_probe.main does not resolve globals from its own module"

_DQ_SCAN_BEFORE = dq.SCAN_DQ
dq.SCAN_DQ = SCAN_DQ3
dq.OUT = OUT_DQ3
dq.CELLS = CELLS_DQ3

# Prove the redirect took AND that it moved: a no-op patch would silently
# re-probe stage DQ's seeds 0 and 1 and write a CSV that looks exactly like a
# DQ3 result. That is the failure this whole file exists to make impossible.
assert dq.SCAN_DQ != _DQ_SCAN_BEFORE, "SCAN_DQ did not move off the DQ scan"
assert dq.resolve_cell.__globals__["SCAN_DQ"] == SCAN_DQ3
assert dq.main.__globals__["CELLS"] == CELLS_DQ3
assert all(s == 2 for _n, s in dq.CELLS), "DQ3 scores seed 2 and nothing else"

if __name__ == "__main__":
    print("stage DQ3 probe | scan %s | cells %s" % (SCAN_DQ3, CELLS_DQ3),
          flush=True)
    sys.exit(dq.main())
