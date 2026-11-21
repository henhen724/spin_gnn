"""Stage NSPLIT scorer -- TA1..TA4 and DN1..DN3.

Committed BEFORE the N=192 checkpoints exist, so nothing below can have been
tuned to the numbers it scores. Predictions are in
`runs/stagensplit_2026-11-20/REGISTERED.md`.

THE QUESTION.  Finding 109 put the `val_nll` floor at 1.371 / 1.625 / 1.861
nats for N = 48 / 96 / 192. TB1 registered that the excess above that floor
would stay in [0.09, 0.15] at N=192; it is falsified, the N=192 run is heading
for ~0.27, and matched-epoch curves show the same descent slope with a
persistent ~0.14 offset. That is a different asymptote, not slow optimisation.
This stage asks which half of the loss the extra 0.14 sits in.

THE HANDLE.  The deduction sections have a data-entropy floor of exactly ZERO
at every N, so any NLL there is pure excess and needs no floor model. The
graph sections do not, and their floor is finding 109's `(5/16) log N`
estimate -- which is itself the thing that might be wrong. Splitting the loss
at that line makes the two accounts mutually exclusive.

ONE AMBIGUITY RESOLVED IN ADVANCE.  TA3 says the `graph_nbr_novel`
"contribution" grows by `(5/16) log 2 = 0.2166`. That could mean the raw
section NLL or the token-share-weighted contribution to the pooled loss.
**TA3 is scored on the token-share-weighted contribution**, since finding 109's
`f_graph` is a term in the pooled `val_nll`. The raw section delta is printed
beside it so a reader can see both, but the verdict is on the weighted one.
Fixing this here, before any N=192 number exists, is the point.

Run:  python3 scripts/stagensplit_fit.py
"""
import argparse
import csv
import glob
import math
import os
import sys

SCRATCH = "/scratch/users/hshunt"
NSPLIT = os.path.join(SCRATCH, "spin_gnn/runs/stagensplit_2026-11-20")
ENTROPY = os.path.join(SCRATCH, "spin_gnn/runs/stageentropy_2026-11-20")

# Finding 109. The floor is a property of the data at a given N, not of a run.
FLOOR = {48: 1.371, 96: 1.625, 192: 1.861}

# Zero data-entropy floor at every N (sudoku_trace.py:83 emits sorted, and the
# colour is forced), so NLL here is pure excess.
DETERMINISTIC = ("reason_node", "reason_color", "round_mark", "term")
# Bookkeeping, not a real section; excluded from shares exactly as in ENTROPY.
EXCLUDE = ("begin",)

TA3_TARGET = (5.0 / 16.0) * math.log(2.0)      # 0.21661
TA3_TOL, TA3_FALSIFY = 0.05, (0.10, 0.33)
PAIR = "L12_s0"


def load(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(p)):
            if r["section"] in EXCLUDE or not r["n_tokens"]:
                continue
            r["n_nodes"] = int(r["n_nodes"])
            r["n_tokens"] = int(r["n_tokens"])
            for k in ("mean_nll", "mean_H", "accuracy"):
                r[k] = float(r[k]) if r[k] else float("nan")
            rows.append(r)
    return rows


def pooled(rows, run_id):
    """Token-share weights and token-weighted mean NLL per section, pooled
    over the five densities. Returns (share, nll, pooled_nll, total_tokens)."""
    sel = [r for r in rows if r["run_id"] == run_id]
    if not sel:
        return None
    tot = sum(r["n_tokens"] for r in sel)
    share, nll = {}, {}
    for r in sel:
        s = r["section"]
        share[s] = share.get(s, 0) + r["n_tokens"]
        nll[s] = nll.get(s, 0.0) + r["mean_nll"] * r["n_tokens"]
    for s in share:
        nll[s] = nll[s] / share[s]
        share[s] = share[s] / tot
    pooled_nll = sum(share[s] * nll[s] for s in share)
    return share, nll, pooled_nll, tot


def ded(share, nll):
    """Pure excess: the deduction sections' contribution. Floor is 0."""
    return sum(share[s] * nll[s] for s in DETERMINISTIC if s in share)


def verdict(ok):
    return "HIT " if ok else "MISS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsplit-dir", default=NSPLIT)
    ap.add_argument("--entropy-dir", default=ENTROPY)
    a = ap.parse_args()

    rows = load(os.path.join(a.nsplit_dir, "nsplit*.csv"))
    if not rows:
        return err("no NSPLIT rows under %s" % a.nsplit_dir)
    runs = sorted({r["run_id"] for r in rows})
    print("stage NSPLIT -- cells present: %s\n" % ", ".join(runs))

    # ---------------------------------------------------------- diagnostics
    print("DIAGNOSTICS")
    dn1_worst, dn2_bad = 0.0, 0
    for rid in runs:
        share, nll, pooled_nll, tot = pooled(rows, rid)
        recon = sum(share[s] * nll[s] for s in share)
        dn1_worst = max(dn1_worst, abs(recon - pooled_nll))
        if abs(sum(share.values()) - 1.0) > 1e-9:
            dn2_bad += 1
        dn2_bad += sum(1 for s in nll if nll[s] < 0)
    print("  DN1 reconstruction           max |diff| %.3e   bar 1e-6   %s"
          % (dn1_worst, "PASS" if dn1_worst < 1e-6 else "FAIL"))
    print("  DN2 NLL >= 0, shares sum 1   %d violation(s)    bar 0      %s"
          % (dn2_bad, "PASS" if dn2_bad == 0 else "FAIL"))

    # DN3: the N=96 cell is measured by both stages and must agree.
    ent = load(os.path.join(a.entropy_dir, "entropy_*of*.csv"))
    dn3 = float("nan")
    n96 = [r for r in runs if r.startswith("n96_")]
    if ent and n96:
        rid = n96[0]
        a96 = pooled(rows, rid)
        b96 = pooled(ent, rid)
        if b96:
            keys = set(a96[0]) | set(b96[0])
            dn3 = max(max(abs(a96[0].get(k, 0) - b96[0].get(k, 0)),
                          abs(a96[1].get(k, 0) - b96[1].get(k, 0)))
                      for k in keys)
            print("  DN3 %s vs stage ENTROPY  max |diff| %.3e   bar 1e-3   %s"
                  % (rid, dn3, "PASS" if dn3 < 1e-3 else "FAIL"))
    else:
        print("  DN3 skipped -- need both the N=96 cell and stage ENTROPY's CSVs")

    # ------------------------------------------------------------- the split
    cells = {}
    for rid in runs:
        share, nll, pooled_nll, tot = pooled(rows, rid)
        n = [r for r in rows if r["run_id"] == rid][0]["n_nodes"]
        cells[rid] = dict(n=n, share=share, nll=nll, pooled=pooled_nll,
                          ded=ded(share, nll),
                          exc=pooled_nll - FLOOR[n],
                          gnov=share.get("graph_nbr_novel", 0.0)
                               * nll.get("graph_nbr_novel", 0.0),
                          gnov_raw=nll.get("graph_nbr_novel", float("nan")))

    print("\nPER-CELL SPLIT  (ded = deduction contribution, floor 0; "
          "exc = pooled NLL - finding-109 floor)")
    print("  %-13s %4s  %8s %8s %8s %8s %9s"
          % ("cell", "N", "pooled", "floor", "exc", "ded", "g_novel"))
    for rid in runs:
        c = cells[rid]
        print("  %-13s %4d  %8.4f %8.4f %8.4f %8.4f %9.4f"
              % (rid, c["n"], c["pooled"], FLOOR[c["n"]], c["exc"], c["ded"],
                 c["gnov"]))

    # ------------------------------------------------------------ TA1..TA4
    src = next((r for r in runs if r.startswith("n96_")), None)
    dsts = [r for r in runs if r.startswith("n192_")]
    if not src or not dsts:
        print("\nTA1-TA4 not scored: need the N=96 cell and at least one "
              "N=192 cell. The N=192 sweep writes model.pt only at the end.")
        return 0

    print("\nPREDICTIONS  (N=96 -> N=192, matched %s)" % PAIR)
    for dst in dsts:
        lo, hi = cells[src], cells[dst]
        d_ded, d_exc = hi["ded"] - lo["ded"], hi["exc"] - lo["exc"]
        frac = d_ded / d_exc if d_exc else float("nan")
        ta1, ta2 = frac >= 0.60, (1 - frac) >= 0.60
        d_gnov = hi["gnov"] - lo["gnov"]
        ta3 = abs(d_gnov - TA3_TARGET) <= TA3_TOL
        ta3_fals = not (TA3_FALSIFY[0] <= d_gnov <= TA3_FALSIFY[1])
        ta4 = hi["ded"] > lo["ded"]
        print("  --- %s ---" % dst)
        print("    d(ded) %+.4f   d(exc) %+.4f   deduction share %.3f"
              % (d_ded, d_exc, frac))
        print("    TA1 DEDUCTION account  share >= 0.60        %.3f   %s"
              % (frac, verdict(ta1)))
        print("    TA2 FLOOR account      1-share >= 0.60      %.3f   %s"
              % (1 - frac, verdict(ta2)))
        print("    TA3 g_novel contrib    %.4f +- %.2f         %+.4f  %s%s"
              % (TA3_TARGET, TA3_TOL, d_gnov, verdict(ta3),
                 "  FALSIFIED" if ta3_fals else ""))
        print("        (raw section NLL delta %+.4f, not the scored quantity)"
              % (hi["gnov_raw"] - lo["gnov_raw"]))
        print("    TA4 ded(192) > ded(96) strict               %+.4f  %s"
              % (d_ded, verdict(ta4)))
        if not ta1 and not ta2:
            print("    NEITHER account is sufficient -- the excess is split, "
                  "which is the registered third outcome.")
    return 0


def err(m):
    print("ERROR: %s" % m, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
