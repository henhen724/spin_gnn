"""Stage DQ scorer -- D1..D5 against `runs/stagedq_2026-11-21/REGISTERED.md`.

The registration (commit 496c766) predates array 53243550. This file may not
change a bar; it only computes the numbers and prints the verdict the bars
imply.

WHY THIS IS NOT `stagensplit_fit.py`. That scorer is hard-wired to q=4, d=5 in
two ways that are not cosmetic:

    FLOOR = {48: 1.371, 96: 1.625, 192: 1.861}      # keyed on N ALONE
    TA3_TARGET = (5.0 / 16.0) * math.log(2.0)       # finding 109, q=4

`FLOOR` would raise KeyError at N=120 and -- far worse -- at N=96 it would
quietly apply a q=4 val_nll floor to a q=3 cell and nothing would look wrong.
Both constants are on this stage's banned list, so the DQ scorer carries
neither. Chance here is `ln 3 = 1.098612`; `ln 4` is not a reference level in
this ensemble, it is above chance.

THE GUARD IS LOAD-BEARING, NOT DECORATION. Every density-curve row must carry
`degree == 4` and `n_colors == 3`, and every section row must satisfy
`vocab_size == n_nodes + 6 + 3`. A d5q4 CSV globbed in by accident would
otherwise be scored against ln 3 and would look like a triumph. `--dry-run`
checks that the guard actually FIRES on a real d5q4 curve, rather than
assuming it would.

ZERO SOLVES IS A BOUND, NOT A VALUE. `a_eff = solve_rate**(1/n_blank)` is 0
for every n_blank when nothing solved, which says only that the product
underflowed. The one-sided Clopper-Pearson 95% bound on 0 successes in n
trials is `p < 1 - 0.05**(1/n)`, and a_eff is monotone in p, so
`a_eff < p_upper**(1/n_blank)`. That is what is reported. This is the rule
that kept N1 honest at N=192 and it is carried forward verbatim.

Run:  python3 scripts/stagedq_fit.py            # score
      python3 scripts/stagedq_fit.py --dry-run  # check the machinery only
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_DQ = os.path.join(SCRATCH, "spin_gnn/sudoku_dq_2026-11-21")
SPLIT_DIR = os.path.join(SCRATCH, "spin_gnn/runs/stagedq_2026-11-21")
NSPLIT_CSV = os.path.join(SCRATCH, "spin_gnn/runs/stagensplit_2026-11-20",
                          "nsplit_0of1.csv")

DEGREE, N_COLORS = 4, 3
CHANCE = math.log(N_COLORS)              # 1.098612

# --- registered bars, transcribed from REGISTERED.md, changed by nobody ----
D1_HN, D1_HT = 0.10, 0.55        # H-N predicts below; H-T predicts at or above
D2_FALSIFY = 0.02                # N=120 BETTER than N=96 by more than this
D3_BAR, D3_BREAK = 0.01, 0.05    # graph_lead and term NLL
D4_RHO, D4_SOLVE = 0.25, 0.1     # a sub-rho_c bin solving better than this
D5_TAIL = 5                      # val_nll argmin inside the last 5 epochs
CONF = 0.95

CELLS = [(96, 0), (96, 1), (120, 0), (120, 1)]


# ------------------------------------------------------------------ helpers
def cp_upper(n_trials, conf=CONF):
    """One-sided Clopper-Pearson upper bound on p given ZERO successes."""
    return 1.0 - (1.0 - conf) ** (1.0 / n_trials)


def a_eff_or_bound(solve_rate, n_blank, n_trials):
    """(value, is_bound). `is_bound` True means 'strictly less than'."""
    if n_blank <= 0:
        return None, False
    if solve_rate > 0.0:
        return solve_rate ** (1.0 / n_blank), False
    return cp_upper(n_trials) ** (1.0 / n_blank), True


def fmt_a(pair):
    v, bound = pair
    if v is None:
        return "    n/a"
    return ("< %.4f" if bound else "  %.4f") % v


# ------------------------------------------------------------------- readers
def load_curves(pattern=None):
    """Density-curve rows, grouped by (n_nodes, seed), q=3 d=4 enforced."""
    pattern = pattern or os.path.join(SCAN_DQ, "rho_curve_n*_s*_*.csv")
    cells = {}
    for p in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(p)):
            d, q = int(r["degree"]), int(r["n_colors"])
            if (d, q) != (DEGREE, N_COLORS):
                raise SystemExit(
                    "GUARD: %s has degree=%d n_colors=%d, not (%d, %d). "
                    "Scoring that against ln %d would be a category error -- "
                    "refusing." % (p, d, q, DEGREE, N_COLORS, N_COLORS))
            row = {
                "n_nodes": int(r["n_nodes"]), "seed": int(r["seed"]),
                "rho": float(r["rho"]),
                "realized_rho": float(r["realized_rho"]),
                "solve_rate": float(r["solve_rate"]),
                "dd_mean_T": float(r["dd_mean_T"]),
                "n_instances": int(r["n_instances"]),
                "truncated_rate": float(r["truncated_rate"]),
                "path": r["path"], "src": p,
            }
            cells.setdefault((row["n_nodes"], row["seed"]), []).append(row)
    for k in cells:
        cells[k].sort(key=lambda r: -r["rho"])
    return cells


def load_sections(pattern=None):
    """Section rows, grouped by (n_nodes, seed). Enforces vocab = N + 6 + q."""
    pattern = pattern or os.path.join(SPLIT_DIR, "dqsplit_*.csv")
    cells = {}
    for p in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(p)):
            if not r["n_tokens"] or int(r["n_tokens"]) == 0:
                continue
            n = int(r["n_nodes"])
            vs = int(r["vocab_size"])
            if vs != n + 6 + N_COLORS:
                raise SystemExit(
                    "GUARD: %s row %s has vocab_size %d, expected %d "
                    "(= N + 6 + %d). Refusing." % (p, r["run_id"], vs,
                                                   n + 6 + N_COLORS, N_COLORS))
            seed = int(r["pair_key"].rsplit("_s", 1)[1])
            cells.setdefault((n, seed), []).append({
                "section": r["section"], "rho": float(r["rho"]),
                "n_tokens": int(r["n_tokens"]),
                "mean_nll": float(r["mean_nll"]),
                "mean_H": float(r["mean_H"]),
                "accuracy": float(r["accuracy"]),
            })
    return cells


def pooled(rows, section, rho=None):
    """Token-weighted mean NLL over densities. (value, n_tokens, accuracy)."""
    num = den = acc = 0.0
    for r in rows:
        if r["section"] != section:
            continue
        if rho is not None and abs(r["rho"] - rho) > 1e-9:
            continue
        num += r["mean_nll"] * r["n_tokens"]
        acc += r["accuracy"] * r["n_tokens"]
        den += r["n_tokens"]
    if not den:
        return None, 0, None
    return num / den, int(den), acc / den


def history(run_dir):
    with open(os.path.join(run_dir, "history.json")) as fh:
        return json.load(fh)


# -------------------------------------------------------------------- D1-D5
def score_d1(sec, out):
    per_seed = {}
    for seed in (0, 1):
        rows = sec.get((96, seed))
        if rows:
            per_seed[seed] = pooled(rows, "reason_color")
    if not per_seed:
        out.append("D1 UNSCOREABLE: no N=96 section rows.")
        return None
    if len(per_seed) < 2:
        out.append("D1 PARTIAL DATA: only seed(s) %s landed; the registration "
                   "asks for both. Reported, not treated as final."
                   % sorted(per_seed))
    allrows = [r for s in (0, 1) for r in sec.get((96, s), [])]
    val, ntok, acc = pooled(allrows, "reason_color")
    for seed, (v, n, a) in sorted(per_seed.items()):
        out.append("   N=96 s%d  reason_color NLL %.4f  acc %.4f  (%d tokens)"
                   % (seed, v, a, n))
    out.append("   pooled    reason_color NLL %.4f  acc %.4f  (%d tokens)   "
               "chance ln 3 = %.4f, chance acc = %.4f"
               % (val, acc, ntok, CHANCE, 1.0 / N_COLORS))
    if val < D1_HN:
        v = ("H-N SUPPORTED (%.4f < %.2f): at an N where the rule is known "
             "learnable the colour head holds up even though this cell is "
             "DEEPER than the d5q4 cell at the same N. Depth is not what "
             "breaks it." % (val, D1_HN))
    elif val >= D1_HT:
        v = ("H-T SUPPORTED (%.4f >= %.2f): the colour head degrades at an N "
             "that is known safe, so the collapse tracks DD depth rather "
             "than N." % (val, D1_HT))
    else:
        v = ("PARTIAL (%.4f, inside the %.2f-%.2f band registered IN ADVANCE "
             "as partial): depth contributes but does not account for the "
             "collapse. D2's DIRECTION decides this; D1 is not re-read."
             % (val, D1_HN, D1_HT))
    out.append("D1 %s" % v)
    split = sorted(v2[0] for v2 in per_seed.values())
    if len(split) == 2 and (split[0] < D1_HN) != (split[1] < D1_HN):
        out.append("   NOTE: the two seeds straddle the %.2f bar (%.4f, %.4f) "
                   "-- the pooled verdict is not seed-robust."
                   % (D1_HN, split[0], split[1]))
    return val


def score_d2(sec, out):
    vals = {}
    for n in (96, 120):
        rows = [r for s in (0, 1) for r in sec.get((n, s), [])]
        v, ntok, acc = pooled(rows, "reason_color")
        if v is None:
            continue
        vals[n] = (v, ntok, acc)
        out.append("   N=%-4d reason_color NLL %.4f  acc %.4f  (%d tokens)"
                   % (n, v, acc, ntok))
    if len(vals) < 2:
        out.append("D2 UNSCOREABLE: need both N=96 and N=120.")
        return
    delta = vals[120][0] - vals[96][0]
    out.append("   N=120 - N=96 = %+.4f nats" % delta)
    if delta < -D2_FALSIFY:
        out.append("D2 FALSIFIED: the deeper cell is BETTER by more than "
                   "%.2f nats, so the degradation does not track depth even "
                   "where depth is the thing varying. D1 may NOT then be "
                   "attributed to depth." % D2_FALSIFY)
    elif delta >= 0:
        out.append("D2 HIT: deeper is not better; the ordering is consistent "
                   "with depth driving the colour head.")
    else:
        out.append("D2 INSIDE TOLERANCE: deeper is better by %.4f, inside the "
                   "registered %.2f-nat falsifier. No direction claimed."
                   % (-delta, D2_FALSIFY))


def score_d3(sec, out):
    unreadable = []
    for key in sorted(sec):
        g, _, _ = pooled(sec[key], "graph_lead")
        t, _, _ = pooled(sec[key], "term")
        worst = max(x for x in (g, t) if x is not None)
        tag = ("OK" if worst < D3_BAR else
               "SOFT" if worst < D3_BREAK else "UNREADABLE")
        if tag == "UNREADABLE":
            unreadable.append(key)
        out.append("   N=%-4d s%d  graph_lead %.4f  term %.4f   %s"
                   % (key[0], key[1], g, t, tag))
    if not sec:
        out.append("D3 UNSCOREABLE: no section rows.")
    elif unreadable:
        out.append("D3 BREAKING POINT HIT for %s: those cells are sick rather "
                   "than specifically bad at colouring, and D1/D2 are "
                   "DECLARED UNREADABLE for them."
                   % ", ".join("N=%d s%d" % k for k in unreadable))
    else:
        out.append("D3 PASS: the deterministic sections stayed learned "
                   "(bar %.2f), so D1/D2 are readable." % D3_BAR)


def score_d4(curves, out):
    hit = []
    for key in sorted(curves):
        n, seed = key
        sub = [r for r in curves[key] if r["rho"] < D4_RHO]
        if not sub:
            out.append("   N=%-4d s%d  no bin below rho=%.2f" % (n, seed,
                                                                 D4_RHO))
            continue
        for r in sub:
            n_blank = r["n_nodes"] - round(r["n_nodes"] * r["realized_rho"])
            a = a_eff_or_bound(r["solve_rate"], n_blank, r["n_instances"])
            flag = " <== D4" if r["solve_rate"] > D4_SOLVE else ""
            if r["solve_rate"] > D4_SOLVE:
                hit.append((n, seed, r["rho"], r["solve_rate"]))
            out.append("   N=%-4d s%d  rho %.2f (real %.3f)  T %5.1f  "
                       "n_blank %3d  solve %.4f  a_eff %s%s"
                       % (n, seed, r["rho"], r["realized_rho"], r["dd_mean_T"],
                          n_blank, r["solve_rate"], fmt_a(a), flag))
    if not curves:
        out.append("D4 UNSCOREABLE: no density curves.")
    elif hit:
        out.append("D4 HIT: %d sub-rho_c bin(s) above solve_rate %.2f, e.g. "
                   "N=%d s%d at rho %.2f solving %.3f."
                   % (len(hit), D4_SOLVE, hit[0][0], hit[0][1], hit[0][2],
                      hit[0][3]))
    else:
        out.append("D4 FALSIFIED: no bin below rho=%.2f reaches solve_rate "
                   "%.2f on any seed. The depth lever does not rescue "
                   "slow-phase operation either. a_eff for a zero bin is the "
                   "Clopper-Pearson BOUND above, never 0." % (D4_RHO, D4_SOLVE))


def score_d5(curves, out):
    bad = []
    for key in sorted(curves):
        run_dir = curves[key][0]["path"]
        try:
            h = history(run_dir)
        except OSError as exc:
            out.append("   N=%-4d s%d  history unavailable: %s" % (key[0],
                                                                   key[1], exc))
            continue
        best = min(h, key=lambda r: r["nll"])
        gap = h[-1]["nll"] - best["nll"]
        in_tail = best["epoch"] >= h[-1]["epoch"] - (D5_TAIL - 1)
        flat = abs(h[0]["nll"] - h[-1]["nll"]) < 1e-3
        tag = "OK" if (in_tail and gap <= 0 and not flat) else "FLAG"
        if tag == "FLAG":
            bad.append(key)
        out.append("   N=%-4d s%d  %2d ep  first %.4f -> final %.4f  best "
                   "%.4f @ep%d  gap %+.4f  %s"
                   % (key[0], key[1], len(h), h[0]["nll"], h[-1]["nll"],
                      best["nll"], best["epoch"], gap, tag))
    if not curves:
        out.append("D5 UNSCOREABLE: no cells.")
    elif bad:
        out.append("D5 FLAGGED for %s. A cell flat at its initial plateau for "
                   "all 50 epochs is NOT trainable at this budget, and that "
                   "IS the result -- it is not retried larger."
                   % ", ".join("N=%d s%d" % k for k in bad))
    else:
        out.append("D5 PASS: every cell was still improving at the end and "
                   "none overfit.")


# ------------------------------------------------------------------ dry run
def dry_run():
    """Four checks that need no stage-DQ data, so they can run while the
    array is still training. Two bugs in stage N were caught exactly here."""
    ok = True

    # 1. The Clopper-Pearson arithmetic, against the number in REGISTERED.md.
    # REGISTERED.md quotes this to 3 s.f. as 0.00746; the exact value is
    # 1 - 0.05**(1/400) = 0.00746136. An earlier version of this check
    # asserted 0.0074564, which was a bad hand-expansion of exp(ln 0.05/400)
    # and not what the code computes -- the dry run caught the TEST, which is
    # what it is for. The bar is the registered figure at its own precision.
    p = cp_upper(400)
    good = abs(p - 0.00746) < 5e-6 and abs(p - (1 - 0.05 ** (1 / 400.0))) < 1e-15
    ok &= good
    print("DRY 1 CP bound, 0/400 at 95%%: p < %.8f  (registered 0.00746)  %s"
          % (p, "PASS" if good else "FAIL"))
    v, is_bound = a_eff_or_bound(0.0, 82, 400)
    good = is_bound and abs(v - p ** (1.0 / 82)) < 1e-12 and 0.9 < v < 1.0
    ok &= good
    print("DRY 1b zero-solve bin returns a BOUND a_eff < %.4f, not 0.0  %s"
          % (v, "PASS" if good else "FAIL"))

    # 2. The guard must actually fire on a real d5q4 curve.
    q4 = sorted(glob.glob(os.path.join(
        SCRATCH, "spin_gnn/sudoku_n192_ladder_2026-11-20/rho_curve_*.csv")))
    if not q4:
        print("DRY 2 SKIPPED: no d5q4 curve on disk to test the guard with")
    else:
        try:
            load_curves(q4[0])
            print("DRY 2 FAIL: the guard did NOT fire on %s -- a q=4 cell "
                  "would be scored against ln 3." % os.path.basename(q4[0]))
            ok = False
        except SystemExit as exc:
            print("DRY 2 guard fired on a d5q4 curve  PASS\n      (%s)"
                  % str(exc).split(" -- ")[0])

    # 3. Pooling, against an independent inline computation on real rows.
    if not os.path.exists(NSPLIT_CSV):
        print("DRY 3 SKIPPED: no NSPLIT CSV to pool")
    else:
        rows, num, den = [], 0.0, 0.0
        for r in csv.DictReader(open(NSPLIT_CSV)):
            if r["run_id"] != "n96_L12_s0" or r["section"] != "reason_color":
                continue
            if not r["n_tokens"] or int(r["n_tokens"]) == 0:
                continue
            rows.append({"section": "reason_color", "rho": float(r["rho"]),
                         "n_tokens": int(r["n_tokens"]),
                         "mean_nll": float(r["mean_nll"]),
                         "mean_H": float(r["mean_H"]),
                         "accuracy": float(r["accuracy"])})
            num += float(r["mean_nll"]) * int(r["n_tokens"])
            den += int(r["n_tokens"])
        got, ntok, _ = pooled(rows, "reason_color")
        good = den > 0 and abs(got - num / den) < 1e-12 and ntok == int(den)
        ok &= good
        print("DRY 3 token-weighted pooling over %d densities, %d tokens, "
              "matches an inline sum to 1e-12  %s"
              % (len(rows), ntok, "PASS" if good else "FAIL"))

    # 4. No banned q=4 constant is reachable from this module's own namespace.
    banned = (0.740, 0.1484, 0.119, 1.371, 1.625, 1.861, math.log(4))
    g = globals()
    hits = [k for k, val in g.items()
            if isinstance(val, float) and any(abs(val - b) < 1e-9
                                              for b in banned)]
    good = not hits
    ok &= good
    print("DRY 4 no banned q=4 constant among this module's globals  %s%s"
          % ("PASS" if good else "FAIL", "" if good else " -> %s" % hits))

    print("\nDRY RUN %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--curves", default=None)
    ap.add_argument("--sections", default=None)
    ap.add_argument("--out", default=None, help="also write the report here")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run()

    curves = load_curves(args.curves)
    sec = load_sections(args.sections)
    missing = os.path.join(SPLIT_DIR, "MISSING.txt")

    out = ["STAGE DQ -- scored against runs/stagedq_2026-11-21/REGISTERED.md",
           "ensemble d=%d q=%d   chance = ln %d = %.6f nats"
           % (DEGREE, N_COLORS, N_COLORS, CHANCE),
           "cells with a density curve: %s"
           % (", ".join("N=%d s%d" % k for k in sorted(curves)) or "none"),
           "cells with a section split: %s"
           % (", ".join("N=%d s%d" % k for k in sorted(sec)) or "none")]
    have = set(curves) | set(sec)
    absent = [c for c in CELLS if tuple(c) not in have]
    if absent or os.path.exists(missing):
        out.append("MISSING CELLS: %s -- any prediction needing one is "
                   "reported UNSCOREABLE below, not guessed."
                   % (", ".join("N=%d s%d" % tuple(c) for c in absent)
                      or "see MISSING.txt"))
    for title, fn, arg in (
            ("D1 -- THE DISCRIMINATOR: reason_color at N=96", score_d1, sec),
            ("D2 -- monotonicity in depth inside this ensemble", score_d2, sec),
            ("D3 -- gates whether D1 is readable", score_d3, sec),
            ("D4 -- slow-phase solving below rho_c", score_d4, curves),
            ("D5 -- trainability", score_d5, curves)):
        out.append("")
        out.append(title)
        fn(arg, out)

    text = "\n".join(out)
    print(text)
    if args.out:
        # Atomic: four array tasks race to be "last one out" and a dead heat
        # must not leave a half-written verdict on disk.
        tmp = "%s.%d.tmp" % (args.out, os.getpid())
        with open(tmp, "w") as fh:
            fh.write(text + "\n")
        os.replace(tmp, args.out)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
