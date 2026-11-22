"""Stage DQ -- the per-section split that D1, D2 and D3 are scored on.

Registered in `runs/stagedq_2026-11-21/REGISTERED.md` (commit 496c766) BEFORE
array 53243550 was submitted. This file is the probe; `stagedq_fit.py` is the
scorer. Neither may change the registered bars; they only supply the numbers.

WHY THIS FILE EXISTS AT ALL. The stage-DQ array trains the four cells and runs
the density sweep (D4) and the trainability read (D5), but it emits NO
per-section NLL -- and D1, the discriminator the whole stage exists for, is a
per-section number. Stage NSPLIT got its section CSVs from a separate CPU job
chained behind the training array (53203962); this is the (d=4, q=3) analogue.

THE MEASUREMENT IS NOT REIMPLEMENTED. `probe_run` is imported from
`stageentropy_probe`, exactly as `stagensplit_probe` imports it, because stage
ENTROPY produced published numbers and must not be edited -- and because D1's
whole force comes from `reason_color` here being the SAME quantity that read
1.3863 nats at d5q4 N=192 and 0.0036 at d5q4 N=96. Re-typing the forward pass
would quietly break that.

Two of `probe_run`'s module globals are wrong for this ensemble, so they are
REDIRECTED IN PLACE rather than worked around:

  * `run_path(n_nodes, rel)` is `{48: ..., 96: ..., 192: ...}[n_nodes]` and
    would raise KeyError at N=120. The replacement passes absolute cell paths
    straight through and falls back to the original for the self-check cell.
  * `RHOS = [0.90, 0.75, 0.60, 0.45, 0.35]` is a q=4 grid. It spans from easy
    down to 0.35, which is 0.75x of that ensemble's rho_c = 0.465. The same
    RULE at rho_c = 0.25 gives 0.1875, and rho_min here is ~0.150, so the DQ
    grid bottoms at 0.18. This grid is fixed in this file BEFORE any cell has
    landed, and D1 is reported per density as well as pooled, so the pooling
    cannot be accused of hiding a density structure.

NO q=4 CONSTANT APPEARS HERE OR IN WHAT THIS WRITES. Chance is ln 3 = 1.0986.
The banned list (free line 0.740, s* 0.1484/0.119, val_nll floors
1.371/1.625/1.861, (5/16) log N, ln 4) is in REGISTERED.md.

WHAT IS CHECKED BEFORE THE REAL CELLS ARE TOUCHED.

  DQ-P0  `stageentropy_probe.selftest` -- the entropy estimator returns log V
         on a uniform distribution, in both precisions. Inherited unchanged.
  DQ-P1  PLUMBING. The redirected globals are used to re-measure the ALREADY
         PUBLISHED d5q4 N=96 L12_s0 cell at rho=0.90 and the result must
         reproduce `nsplit_0of1.csv` to 1e-6 on every section. This is a
         code-correctness check, NOT a science comparison: it is run on the
         q=4 grid against a q=4 cell precisely so that any disagreement is
         attributable to the redirection and nothing else. A d5q4 number is
         never carried into a q=3 verdict.
  DQ-P2  VOCABULARY. Every DQ cell must report `vocab_size == n_nodes + 6 + 3`.
         Sections are labelled by position, so a q=4 checkpoint accidentally
         globbed into this list would otherwise be scored against ln 3 and
         look spectacular. The scorer re-derives q from vocab_size and asserts
         it again.

MISSING CELLS ARE MEASURED AROUND, NOT WAITED FOR. If a training cell died,
this probe reports every cell that did land and writes `MISSING.txt`; the
SCORER then refuses any prediction whose cells are not all present. Measuring
what exists and scoring only what is complete beats a dependency that sits in
the queue forever as DependencyNeverSatisfied.

Run:  python3 scripts/stagedq_probe.py [--task I --ntasks K] [--plumbing-only]
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.stageentropy_probe as ep
from scripts.stagescale_probe import run_path as scale_run_path

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_DQ = os.path.join(SCRATCH, "spin_gnn/sudoku_dq_2026-11-21")
OUT = os.path.join(SCRATCH, "spin_gnn/runs/stagedq_2026-11-21")

DEGREE, N_COLORS = 4, 3
CHANCE = math.log(N_COLORS)              # ln 3 = 1.098612, NOT ln 4

# Fixed before any cell landed. See the module docstring for the rule.
DQ_RHOS = [0.90, 0.70, 0.50, 0.30, 0.18]

# (n_nodes, seed) -- array 53243550 tasks 0..3, in order.
CELLS = [(96, 0), (96, 1), (120, 0), (120, 1)]

# DQ-P1's reference: a PUBLISHED d5q4 cell, its stage-SCALE relative path, and
# the NSPLIT CSV its numbers already live in.
PLUMB_REC = ("n96_L12_s0", 96, "L12_s0",
             "t00_L12_ff01024_n240000_s0/"
             "d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall")
PLUMB_RHO = 0.90
PLUMB_CSV = os.path.join(SCRATCH, "spin_gnn/runs/stagensplit_2026-11-20",
                         "nsplit_0of1.csv")
PLUMB_TOL = 1e-6


def dq_run_path(n_nodes, rel):
    """Absolute cell paths pass through; relative ones keep SCALE's meaning.

    The original raises KeyError on any N outside {48, 96, 192}, which is every
    N this stage cares about at N=120.
    """
    if os.path.isabs(rel):
        return rel
    return scale_run_path(n_nodes, rel)


def install_redirects(rhos):
    """Point `probe_run`'s globals at this ensemble, and PROVE the patch took.

    `probe_run` resolves `RHOS` and `run_path` out of its own module dict at
    call time, so rebinding them there is what reaches it -- rebinding a local
    import in this file would not. The assertion is here because a silently
    ineffective monkeypatch would run the whole probe on the q=4 grid and
    produce a plausible-looking CSV.
    """
    assert ep.probe_run.__globals__ is ep.__dict__, \
        "probe_run does not resolve globals from stageentropy_probe"
    ep.RHOS = list(rhos)
    ep.run_path = dq_run_path
    assert ep.probe_run.__globals__["RHOS"] == list(rhos)
    assert ep.probe_run.__globals__["run_path"] is dq_run_path


def resolve_cell(n_nodes, seed):
    """(abs run directory, None) or (None, why not).

    The sweep names the inner directory from the model config, not from the
    ensemble, so it is FOUND rather than reconstructed: a wrong guessed path
    would be scored as a real result and nothing downstream would catch it.
    `model.pt` is written only at the end of training, so "not there yet" and
    "died" look the same here and are both reported as missing.
    """
    root = os.path.join(SCAN_DQ, "n%d_d%d_q%d_L12_s%d"
                        % (n_nodes, DEGREE, N_COLORS, seed))
    if not os.path.isdir(root):
        return None, "no cell directory: %s" % root
    hits = [d for d in sorted(os.listdir(root))
            if os.path.exists(os.path.join(root, d, "model.pt"))]
    if len(hits) != 1:
        return None, ("expected exactly one subdirectory with model.pt under "
                      "%s, found %d %s" % (root, len(hits), hits))
    return os.path.join(root, hits[0]), None


def load_reference(path, run_id, rho):
    """{section: (mean_H, mean_nll, n_tokens)} from a published NSPLIT CSV."""
    ref = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["run_id"] != run_id or abs(float(r["rho"]) - rho) > 1e-9:
                continue
            if not r["n_tokens"] or int(r["n_tokens"]) == 0:
                continue
            ref[r["section"]] = (float(r["mean_H"]), float(r["mean_nll"]),
                                 int(r["n_tokens"]))
    return ref


def plumbing_check():
    """DQ-P1 -- the redirection reproduces a published number exactly."""
    if not os.path.exists(PLUMB_CSV):
        print("DQ-P1 SKIPPED: no reference CSV at %s" % PLUMB_CSV, flush=True)
        return True
    ref = load_reference(PLUMB_CSV, PLUMB_REC[0], PLUMB_RHO)
    if not ref:
        print("DQ-P1 SKIPPED: no %s rows at rho=%.2f in %s"
              % (PLUMB_REC[0], PLUMB_RHO, PLUMB_CSV), flush=True)
        return True

    install_redirects([PLUMB_RHO])
    rows = []
    ep.probe_run(PLUMB_REC, rows, 0)

    worst, worst_what, n_cmp = 0.0, "", 0
    for r in rows:
        if r["section"] not in ref or not r["n_tokens"]:
            continue
        h_ref, nll_ref, tok_ref = ref[r["section"]]
        if int(r["n_tokens"]) != tok_ref:
            print("DQ-P1 FAIL: %s token count %s != published %d"
                  % (r["section"], r["n_tokens"], tok_ref), flush=True)
            return False
        for got, want, what in ((float(r["mean_H"]), h_ref, "mean_H"),
                                (float(r["mean_nll"]), nll_ref, "mean_nll")):
            d = abs(got - want)
            n_cmp += 1
            if d > worst:
                worst, worst_what = d, "%s %s" % (r["section"], what)
    ok = n_cmp > 0 and worst < PLUMB_TOL
    print("DQ-P1 plumbing: %d quantities vs published NSPLIT, max |diff| "
          "%.3e at %s  (tol %.0e)  %s"
          % (n_cmp, worst, worst_what or "-", PLUMB_TOL,
             "PASS" if ok else "FAIL"), flush=True)
    return ok


def check_vocab(rows, n_nodes):
    """DQ-P2 -- this checkpoint really is q=3 at this N."""
    want = n_nodes + 6 + N_COLORS
    seen = {int(r["vocab_size"]) for r in rows if r["vocab_size"]}
    if seen != {want}:
        raise SystemExit("DQ-P2 FAIL: N=%d vocab_size %s, expected {%d} "
                         "(= N + 6 + %d). Refusing to score a non-q=%d "
                         "checkpoint against ln %d."
                         % (n_nodes, sorted(seen), want, N_COLORS,
                            N_COLORS, N_COLORS))
    print("  DQ-P2 N=%d vocab_size %d == N + 6 + %d  PASS"
          % (n_nodes, want, N_COLORS), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap instances per bin; 0 = all cached")
    ap.add_argument("--plumbing-only", action="store_true",
                    help="run DQ-P0/DQ-P1 and stop; needs no DQ checkpoint "
                         "and so can be run while the array is still training")
    args = ap.parse_args()

    print("stage DQ section split | chance = ln %d = %.6f nats | rho grid %s"
          % (N_COLORS, CHANCE, DQ_RHOS), flush=True)

    if not ep.selftest():                       # DQ-P0
        return 1
    if not plumbing_check():                    # DQ-P1
        return 1
    if args.plumbing_only:
        print("plumbing only: stopping before the DQ cells", flush=True)
        return 0

    os.makedirs(args.out, exist_ok=True)
    recs, missing = [], []
    for n_nodes, seed in CELLS:
        path, why = resolve_cell(n_nodes, seed)
        if path is None:
            missing.append("n%d_s%d: %s" % (n_nodes, seed, why))
            continue
        recs.append(("n%d_d%dq%d_L12_s%d" % (n_nodes, DEGREE, N_COLORS, seed),
                     n_nodes, "L12_s%d" % seed, path))

    if missing:
        print("\n!!! %d of %d cells are MISSING -- measuring the rest and "
              "recording them; stagedq_fit.py will refuse any prediction that "
              "needs one:" % (len(missing), len(CELLS)), flush=True)
        for m in missing:
            print("    " + m, flush=True)
        with open(os.path.join(args.out, "MISSING.txt"), "w") as fh:
            fh.write("\n".join(missing) + "\n")
    if not recs:
        raise SystemExit("no stage DQ checkpoint exists yet; nothing to probe")

    install_redirects(DQ_RHOS)
    mine = recs[args.task::args.ntasks]
    print("\ntask %d/%d: %d of %d landed cells" % (args.task, args.ntasks,
                                                   len(mine), len(recs)),
          flush=True)
    for r in mine:
        print("  %-18s N=%-4d %s" % (r[0], r[1], r[3]), flush=True)

    rows = []
    for rec in mine:
        before = len(rows)
        ep.probe_run(rec, rows, args.limit)
        check_vocab(rows[before:], rec[1])

    path = os.path.join(args.out, "dqsplit_%dof%d.csv"
                        % (args.task, args.ntasks))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
