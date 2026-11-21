"""Stage NSPLIT -- where does the N=192 excess live, deduction or floor?

Registered in `runs/stagensplit_2026-11-20/REGISTERED.md` before the N=192
checkpoint existed. This file is the probe; `stagensplit_fit.py` is the scorer.

HOW THIS RELATES TO STAGE ENTROPY.  The measurement is the same per-section
teacher-forced pass, so `probe_run` and `selftest` are imported from
`stageentropy_probe` rather than copied. Stage ENTROPY produced published
numbers and is not edited: this module only supplies a different cell list and
a different output directory. The N=96 `L12_s0` cell is deliberately measured
by BOTH stages, which is diagnostic DN3 -- a free cross-check that extending
the path to N=192 did not perturb the existing one.

WHY THE DEDUCTION SECTIONS ARE THE HANDLE.  `reason_node`, `reason_color`,
`round_mark` and `term` have a data-entropy floor of exactly ZERO at every N:
within-round emission order is sorted by node index (`sudoku_trace.py:83`) and
the colour is forced, so each is deterministic given the prefix. Stage ENTROPY
measured 0.0219 and 0.0000 nats on the best N=48 cell, confirming it. So any
NLL on those sections is pure excess and needs no floor model -- unlike the
graph section, whose floor is finding 109's `(5/16) log N` estimate and would
otherwise have to be trusted.

RESOLVING THE N=192 CELLS.  The sweep names the inner run directory from the
model config, not from `n_nodes`, so the N=192 tag should match N=96's. Rather
than hard-code that guess, the inner directory is found by looking for the one
subdirectory holding `model.pt`. If there is not exactly one, this raises
instead of silently probing the wrong checkpoint.

Run:  python3 scripts/stagensplit_probe.py --task I --ntasks K
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stageentropy_probe import probe_run, selftest
from scripts.stagescale_probe import RUNS, SCAN_192

SCRATCH = "/scratch/users/hshunt"
OUT = os.path.join(SCRATCH, "spin_gnn/runs/stagensplit_2026-11-20")

# The N=192 ladder cells, as (run_id, pair_key, cell directory under SCAN_192).
# The inner run directory is resolved at call time, not written down here.
N192_CELLS = [
    ("n192_L12_s0", "L12_s0", "t00_L12_ff01024_n240000_s0"),
    ("n192_L12_s1", "L12_s1", "t01_L12_ff01024_n240000_s1"),
]

# Measured by stage ENTROPY too; this is DN3.
N96_CELL = "n96_L12_s0"


def resolve_inner(cell_dir):
    """The single subdirectory of `cell_dir` that holds a model.pt.

    Raises rather than guessing: a wrong checkpoint here would be scored as a
    real result and there is no downstream check that would catch it."""
    root = os.path.join(SCAN_192, cell_dir)
    if not os.path.isdir(root):
        raise SystemExit("cell directory does not exist yet: %s" % root)
    hits = [d for d in sorted(os.listdir(root))
            if os.path.exists(os.path.join(root, d, "model.pt"))]
    if len(hits) != 1:
        raise SystemExit(
            "expected exactly one subdirectory with model.pt under %s, found %d"
            " %s -- the run may still be training (the sweep writes model.pt "
            "only at the end)" % (root, len(hits), hits))
    return os.path.join(cell_dir, hits[0])


def build_recs(include_192):
    """Cell records in `probe_run`'s (run_id, n_nodes, pair_key, rel) form."""
    recs = [r for r in RUNS if r[0] == N96_CELL]
    if len(recs) != 1:
        raise SystemExit("N=96 cross-check cell %s not found in stage SCALE's "
                         "RUNS" % N96_CELL)
    if include_192:
        for run_id, pair_key, cell_dir in N192_CELLS:
            recs.append((run_id, 192, pair_key, resolve_inner(cell_dir)))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap instances per bin; 0 = all cached")
    ap.add_argument("--n96-only", action="store_true",
                    help="run just the DN3 cross-check cell, which exists "
                         "before the N=192 checkpoints do")
    args = ap.parse_args()

    if not selftest():
        return 1
    os.makedirs(args.out, exist_ok=True)
    recs = build_recs(include_192=not args.n96_only)
    mine = recs[args.task::args.ntasks]
    print("task %d/%d: %d of %d cells%s"
          % (args.task, args.ntasks, len(mine), len(recs),
             " (N=96 cross-check only)" if args.n96_only else ""), flush=True)
    for r in mine:
        print("  %-12s N=%-4d %s" % (r[0], r[1], r[3]), flush=True)

    rows = []
    for rec in mine:
        probe_run(rec, rows, args.limit)

    suffix = "_n96" if args.n96_only else ""
    path = os.path.join(args.out, "nsplit%s_%dof%d.csv"
                        % (suffix, args.task, args.ntasks))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
