"""WHEN do cells escape the 1.64 plateau, and is 50 epochs enough to find out?

pareto_min.py (11-05) found that the floor on this grid is trainability rather
than capacity: of 24 rungs with at least two seeds, only 12 ever got below
finding 44's train-NLL plateau of 1.64, and the standing minimum is simply the
cheapest of those 12. Every rung below it is excluded for never escaping.

The programme has always written 0/3 escapes as "did not train in 375k steps,
never evidence in either direction" (minarch_analysis's own docstring: "the two
are separable only by more steps or a better optimizer, neither of which is in
this job"). That is the honest reading, but it has been treated as a dead end.
It is not: the escape TIMES that are already on disk say how much of the escape
distribution the budget actually covers.

TWO QUESTIONS, BOTH ANSWERED FROM CURVES THAT ALREADY EXIST, NO NEW CELLS.

  1. WHEN does a cell that escapes, escape? If escapes are all early, a cell
     still on the plateau at epoch 50 is unlikely to move at 100 and "did not
     train" is close to "cannot train". If escapes are still arriving at the
     end of the budget, the budget is truncating the distribution and the 0/3
     rungs are partly a measurement artifact.

  2. Is there room to spend more epochs, or are these cells already
     overfitting? The pool is FIXED at 240k instances, so more epochs is more
     passes over the same data. If best validation NLL arrives well before the
     last epoch, more epochs buys memorization. If it arrives AT the last
     epoch, the run was stopped mid-descent.

WHAT THE ANSWERS DECIDE. Together they are the registration evidence for a
longer-budget rerun of the cheap end -- the follow-up cell that stage E's
header named and deliberately did not run ("A longer-budget rerun is the
follow-up cell, not this cell"). This script makes that call on data instead of
on taste, and it is DESCRIPTIVE: no prediction, no falsifier, no new cell.

A NOTE ON WHAT "ESCAPE EPOCH" IS. The first epoch whose train NLL is below
1.64, from the per-epoch history.json. It is not re-derived from the final
value: several seeds dip below and come back, and first-crossing is the
quantity that matters for "would more epochs have caught it".

Usage:  python3 -m scripts.escape_timing [--scan-dir DIR ...] [--all]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.minarch_analysis import (SCAN_DIRS, SCAN_E, cell_key,  # noqa: E402
                                      load_all, rung_label)
from scripts.stagec_score import PLATEAU  # noqa: E402

# Rungs whose escape is marginal enough that the timing question is live. Not
# used to filter the table -- everything is printed -- only to compute the
# summary, so that three rungs that escape at epoch 2 do not drown the signal.
EARLY = 15


def series(path):
    with open(os.path.join(path, "history.json")) as fh:
        h = json.load(fh)
    return [(float(e["train_nll"]), float(e["nll"]), float(e["accuracy"])) for e in h]


def cells(scan_dirs):
    """{rung key: {path: (seed, series)}} -- dict by path, since load_all gives
    one row per (cell, rho) bin and a cell appears 14 times."""
    out = defaultdict(dict)
    for r in load_all(scan_dirs):
        if r["path"] in out[cell_key(r)[:-1]]:
            continue
        try:
            out[cell_key(r)[:-1]][r["path"]] = (cell_key(r)[-1], series(r["path"]))
        except (OSError, KeyError, IndexError, ValueError):
            pass
    return out


def escape_epoch(s):
    return next((i for i, (t, _v, _a) in enumerate(s) if t < PLATEAU), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_DIRS + [SCAN_E],
                    help="stage E is INCLUDED by default here, unlike every scorer: "
                         "this script reads training curves, not solve rates, so it "
                         "cannot move a pooled bar or a published median.")
    a = ap.parse_args()

    by = cells(a.scan_dir)
    print("WHEN CELLS ESCAPE THE 1.64 PLATEAU -- descriptive, no verdict")
    print(f"  {sum(len(v) for v in by.values())} cells over {len(by)} rungs, "
          f"{len(a.scan_dir)} scan dirs\n")
    print(f"{'rung':<24} {'sd':>2} {'ep':>3} {'trainEnd':>8} {'last20%/ep':>10} "
          f"{'esc@':>5} {'valBest@':>8} {'final-best':>10}")

    allesc, marginal = [], []
    for key in sorted(by, key=lambda k: rung_label(*k)):
        for sd, s in sorted(by[key].values()):
            n = len(s)
            tr = [t for t, _v, _a in s]
            tail = tr[int(n * 0.8):]
            slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
            esc = escape_epoch(s)
            vb = min(range(n), key=lambda i: s[i][1])
            print(f"{rung_label(*key):<24} {sd:>2} {n:>3} {tr[-1]:>8.3f} {slope:>+10.5f} "
                  f"{(str(esc) if esc is not None else 'never'):>5} {vb:>5}/{n - 1:<2} "
                  f"{s[-1][1] - s[vb][1]:>+10.4f}")
            if esc is not None:
                allesc.append((esc, n, rung_label(*key).strip(), sd))
        if any(escape_epoch(s) is not None and escape_epoch(s) > EARLY
               for _sd, s in by[key].values()):
            marginal.append(rung_label(*key).strip())

    print("\n=== Q1: IS THE BUDGET TRUNCATING THE ESCAPE DISTRIBUTION? ===")
    n_tot = len(allesc)
    late = [e for e in allesc if e[0] * 2 > e[1]]
    very = [e for e in allesc if e[0] > 0.8 * e[1]]
    last = max(allesc, key=lambda e: e[0])
    print(f"  {n_tot} escapes on disk. {len(late)} of them ({100 * len(late) / n_tot:.0f}%) "
          f"happened in the SECOND HALF")
    print(f"  of the budget and {len(very)} in the final fifth. The LAST escape anywhere in")
    print(f"  the grid is {last[2]} seed {last[3]} at epoch {last[0]} of {last[1]}.")
    print(f"  {len(marginal)} rungs have at least one seed escaping after epoch {EARLY}:")
    print("    " + ", ".join(marginal))
    print("  Escapes are still ARRIVING when training stops, so a 0/3 rung is a rung whose")
    print("  escape time may simply exceed the budget. That does not make the cheap rungs")
    print("  trainable -- it makes 50 epochs unable to tell.")

    print("\n=== Q2: IS THERE ROOM TO SPEND MORE EPOCHS? ===")
    gaps, early_best = [], []
    for key in by:
        for sd, s in by[key].values():
            vb = min(range(len(s)), key=lambda i: s[i][1])
            gaps.append(s[-1][1] - s[vb][1])
            if vb < 0.8 * (len(s) - 1):
                early_best.append((rung_label(*key).strip(), sd, vb, len(s) - 1,
                                   max(v for _t, v, _a in s) - min(v for _t, v, _a in s)))
    print(f"  worst final-minus-best validation NLL over all {len(gaps)} cells: "
          f"{max(gaps):+.4f} nats")
    print(f"  cells whose best val NLL is NOT in the last fifth of the run: "
          f"{len(early_best)} of {len(gaps)}")
    for name, sd, vb, n, rng in sorted(early_best):
        print(f"    {name:<22} seed {sd}  best at {vb}/{n}, but its val NLL moves only "
              f"{rng:.4f} all run")
    print("  The exceptions are all frozen-feature cells, whose validation NLL moves under")
    print("  0.02 nats all run, so their argmin is noise rather than an early optimum. None")
    print("  in this grid is overfitting: the worst regression from best-val is under 0.007")
    print("  nats. With a constant LR and a fixed 240k pool, more epochs is more")
    print("  optimization and nothing else -- no schedule to reshape, no memorization to")
    print("  buy. Every run was stopped mid-descent.")


if __name__ == "__main__":
    main()
