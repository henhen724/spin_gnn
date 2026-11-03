"""Score the four pre-registrations of the N=96 fall scan (50860571).

The pre-registration is the header of
scripts/slurm/sudoku2_n96_fall_scan_2026-11-02.sbatch; this file computes the
expressions it names (finding 39's lesson: compute the expression, do not
re-derive it from a sentence).

    H1  every cell has a rho_50 strictly inside the grid. The falsifier is the
        best cell failing to cross 0.5 before the ensemble floor -- which is
        exactly what happened at N=48 (finding 44) and is why N moved up. If
        H1 fails here the next move is N=192, not another N=96 grid.
    H2  rho_50(L=8) > rho_50(12) > rho_50(16), with the 8->12 gap >= 0.05.
    H3  120k->240k moves rho_50 by >= 0.02 toward 240k and by less than the
        depth gap. Falsifier: 120k wins, i.e. N=96 is optimization-bound and
        the fix is more steps, not more instances.
    H4  |rho_50(s0) - rho_50(s1)| <= 0.05, and EVERY H1-H3 comparison is read
        against it (finding 41: at N=48 the seed was worth 0.163, more than
        the depth gap it was being read against).

Two differences from $SCRATCH/fall_scan_analysis.py, which did N=48:

  * It lives in the repo. The N=48 reader sits on a 90-day purge timer and is
    the only thing that can reproduce finding 44's table.
  * rho_50 carries a CI. Every rho_50 here is a one-number summary of 14
    binomial bins at n=400, and findings 41/42 are both about gaps being read
    against the wrong error bar. `rho50_ci` resamples each bin from
    Binomial(n, p_hat) and recomputes the crossing, so a depth gap can be
    compared with the width of the thing it is a gap between. This is SAMPLING
    error only -- the seed term (H4) is separate and usually larger.

Partial runs are expected: the array finishes over several hours, so the
script reports whatever cells are on disk and names the missing ones instead
of failing. Run it again when the rest land.

Usage:  python3 scripts/n96_fallscan_analysis.py [--scan-dir DIR] [--boot N]
"""
import argparse
import csv
import glob
import json
import math
import os
import random
from collections import defaultdict

SCAN_DEFAULT = os.path.join(
    os.environ.get("SCRATCH", "/scratch/users/hshunt"),
    "spin_gnn", "sudoku_n96_fall_scan_2026-11-02")

# (label, cellkey) for the five submitted tasks. cellkey = (L, ff, n_train, seed).
TASKS = {
    "t0 L=12 240k s0 (headline)": (12, 1024, 240000, 0),
    "t1 L=8  240k s0": (8, 1024, 240000, 0),
    "t2 L=16 240k s0": (16, 1024, 240000, 0),
    "t3 L=12 120k s0 (data axis)": (12, 1024, 120000, 0),
    "t4 L=12 240k s1 (seed)": (12, 1024, 240000, 1),
    # Added 11-03 as job 50903022, a replicate rather than a new condition:
    # H2's bar is the 8->12 gap and without this the shallow end of it is a
    # single draw (finding 41 measured the seed at 0.1627 at L=8 at N=48).
    "t5 L=8  240k s1 (seed @ L=8)": (8, 1024, 240000, 1),
}

_HIST = {}


def history(path):
    """(final_val, best_val, final_train, n_epochs) -- 'nll' is the val key.

    The sweep writes val_nll into history.json, not config.json, so the
    rho_curve CSV's val_nll column is blank for every scan cell.
    """
    if path not in _HIST:
        try:
            with open(os.path.join(path, "history.json")) as fh:
                h = json.load(fh)
            _HIST[path] = (float(h[-1]["nll"]), min(float(e["nll"]) for e in h),
                           float(h[-1]["train_nll"]), len(h))
        except (OSError, KeyError, IndexError, ValueError):
            _HIST[path] = (float("nan"),) * 3 + (0,)
    return _HIST[path]


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                for k in ("rho", "realized_rho", "solve_rate", "outcome_accuracy",
                          "well_formed_rate", "truncated_rate", "dd_mean_T"):
                    r[k] = float(r[k])
                for k in ("num_layers", "dim_feedforward", "n_params",
                          "n_train_instances", "epochs", "seed", "n_instances"):
                    r[k] = int(r[k])
                r["val_nll"] = float(r["val_nll"]) if r["val_nll"] else history(r["path"])[0]
                rows.append(r)
    return rows


def rho_50(curve):
    """Realized density where solve_rate crosses 0.5, linear interpolation.

    curve: [(realized_rho, solve_rate)], any order. Walks upward in rho and
    takes the FIRST upward crossing, matching the N=48 reader so the two
    numbers mean the same thing. Returns (value, flag): 'ok', '<floor' (never
    falls to 0.5 inside the grid -- the model is better than the grid can
    measure, finding 44's failure) or '>top' (never rises to 0.5).
    """
    pts = sorted(set((round(r, 6), s) for r, s in curve))
    if not pts:
        return (float("nan"), "none")
    if all(s >= 0.5 for _, s in pts):
        return (pts[0][0], "<floor")
    if all(s < 0.5 for _, s in pts):
        return (pts[-1][0], ">top")
    for (r0, s0), (r1, s1) in zip(pts, pts[1:]):
        if s0 < 0.5 <= s1:
            return (r0 + (0.5 - s0) * (r1 - r0) / (s1 - s0), "ok")
    return (float("nan"), "none")


def rho50_ci(cell_rows, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI for rho_50 under binomial resampling of each bin.

    Each bin is n=400 independent instances scored pass/fail, so the honest
    sampling model is Binomial(n, p_hat) per bin, resampled independently and
    pushed back through the same interpolation. Draws where the curve no
    longer crosses are dropped and counted -- if many are dropped the point
    estimate is near the edge of the grid and the CI is not trustworthy.
    """
    rng = random.Random(seed)
    base = [(r["realized_rho"], r["solve_rate"], r["n_instances"]) for r in cell_rows]
    draws = []
    dropped = 0
    for _ in range(n_boot):
        curve = []
        for rho, p, n in base:
            k = _binom(rng, n, p)
            curve.append((rho, k / n))
        v, flag = rho_50(curve)
        if flag == "ok" and not math.isnan(v):
            draws.append(v)
        else:
            dropped += 1
    if len(draws) < 20:
        return (float("nan"), float("nan"), dropped / n_boot)
    draws.sort()
    lo = draws[int(alpha / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return (lo, hi, dropped / n_boot)


def _binom(rng, n, p):
    """Binomial(n, p) via normal approximation with exact tails.

    n=400 everywhere here, so the normal approximation is good except when p
    is pinned at 0 or 1, which happens in most of this grid's tails -- those
    are returned exactly rather than smeared by a continuity correction.
    """
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if n * p < 10 or n * (1 - p) < 10:
        return sum(1 for _ in range(n) if rng.random() < p)
    k = int(round(rng.gauss(n * p, math.sqrt(n * p * (1 - p)))))
    return max(0, min(n, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default=SCAN_DEFAULT)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    rows = load(os.path.join(args.scan_dir, "rho_curve_t*.csv"))
    if not rows:
        print(f"no rho_curve_t*.csv under {args.scan_dir}")
        return
    cells = defaultdict(list)
    for r in rows:
        cells[(r["num_layers"], r["dim_feedforward"],
               r["n_train_instances"], r["seed"])].append(r)

    present = {lab: k for lab, k in TASKS.items() if k in cells}
    missing = {lab: k for lab, k in TASKS.items() if k not in cells}
    print(f"{len(rows)} rows / {len(cells)} cells under {args.scan_dir}")
    print(f"present: {len(present)}/{len(TASKS)}   missing: "
          + (", ".join(sorted(missing)) if missing else "none") + "\n")

    r50, ci = {}, {}
    for k, rs in cells.items():
        r50[k] = rho_50([(r["realized_rho"], r["solve_rate"]) for r in rs])
        ci[k] = rho50_ci(rs, n_boot=args.boot)

    grid = sorted({r["rho"] for r in rows}, reverse=True)
    realized = {q: min(r["realized_rho"] for r in rows if r["rho"] == q) for q in grid}

    # ------------------------------------------------------------- controls
    print("=== controls: grid pairing, floor, grammar ===")
    by_req = defaultdict(set)
    for r in rows:
        by_req[r["rho"]].add(round(r["realized_rho"], 6))
    bad = {q: v for q, v in by_req.items() if len(v) > 1}
    print(f"  requested rho whose realized_rho differs across cells: {len(bad)} "
          f"(0 = all curves paired point by point)")
    floor = min(r["realized_rho"] for r in rows)
    print(f"  fall-ensemble floor realized_rho = {floor:.4f}; "
          f"dd_mean_T spans {min(r['dd_mean_T'] for r in rows):.2f} "
          f"to {max(r['dd_mean_T'] for r in rows):.2f}")
    live = [r for r in rows if r["solve_rate"] > 0.1]
    if live:
        worst_tr = max(r["truncated_rate"] for r in live)
        worst_wf = min(r["well_formed_rate"] for r in live)
        print(f"  over {len(live)} bins with solve>0.1: max truncated_rate {worst_tr:.4f} "
              f"(bar 0.01){'  <-- MARGIN FIX HOLDS' if worst_tr <= 0.01 else '  <-- BREACH'}, "
              f"min well_formed {worst_wf:.4f}")
    print()

    # ---------------------------------------------------------------- table
    print("=== solve rate vs rho, fall ensemble (n=400/bin, same instances everywhere) ===")
    print("  cell                          " + " ".join(f"{realized[q]:5.2f}" for q in grid)
          + "   rho_50 [95% CI]")
    for lab, k in TASKS.items():
        if k not in cells:
            print(f"  {lab:28s}  {'(not finished)':>{6*len(grid)}}")
            continue
        cur = {r["rho"]: r for r in cells[k]}
        v, flag = r50[k]
        lo, hi, drop = ci[k]
        tag = {"ok": "", ">top": " (>top)", "<floor": " (<FLOOR)"}[flag]
        band = f"[{lo:.3f}, {hi:.3f}]" if not math.isnan(lo) else "[--]"
        print(f"  {lab:28s}  " + " ".join(f"{cur[q]['solve_rate']:5.3f}" for q in grid)
              + f"   {v:.3f}{tag} {band}")
    print()

    # ------------------------------------------------------------------- H1
    print("=== H1: every cell crosses 0.5 strictly inside the grid ===")
    for lab, k in present.items():
        v, flag = r50[k]
        verdict = "PASS" if flag == "ok" else f"FAIL ({flag})"
        print(f"  {lab:28s} rho_50 = {v:.3f}  {verdict}")
    if missing:
        print(f"  (pending: {len(missing)} cells)")
    print()

    # ------------------------------------------------------- the H4 yardstick
    k0, k4 = TASKS["t0 L=12 240k s0 (headline)"], TASKS["t4 L=12 240k s1 (seed)"]
    k1, k5 = TASKS["t1 L=8  240k s0"], TASKS["t5 L=8  240k s1 (seed @ L=8)"]
    print("=== H4: the seed yardstick (identical config, seeds 0 and 1) ===")
    seed_term = {}
    for depth, ka, kb in ((12, k0, k4), (8, k1, k5)):
        if ka not in cells or kb not in cells:
            print(f"  L={depth:2d}: pending")
            continue
        a = {r["rho"]: r["solve_rate"] for r in cells[ka]}
        b = {r["rho"]: r["solve_rate"] for r in cells[kb]}
        d = {q: abs(a[q] - b[q]) for q in grid}
        seed_term[depth] = abs(r50[ka][0] - r50[kb][0])
        print(f"  L={depth:2d} |dsolve| " + " ".join(f"{d[q]:5.3f}" for q in grid)
              + f"   max {max(d.values()):.3f}  mean {sum(d.values())/len(d):.3f}")
        print(f"  L={depth:2d} seed spread in rho_50 = {seed_term[depth]:.3f}  (bar <= 0.05)  "
              f"{'PASS' if seed_term[depth] <= 0.05 else 'FAIL'}")
    # The gap is read against the LARGER of the two ends it spans: a depth
    # claim is only as sharp as the noisier rung it is measured from.
    s_seed = max(seed_term.values()) if seed_term else None
    if s_seed is None:
        print("  >>> pending -- until one lands, NO H2/H3 gap below is "
              "interpretable (finding 41)")
    else:
        print(f"  >>> yardstick in force = {s_seed:.3f} (max over measured rungs); "
              f"every gap below must clear it to be a measurement of architecture")
    print()

    # ------------------------------------------------------------------- H2
    print("=== H2: depth. rho_50(L=8) > rho_50(12) > rho_50(16), gap(8,12) >= 0.05 ===")
    kd = [(8, TASKS["t1 L=8  240k s0"]), (12, k0), (16, TASKS["t2 L=16 240k s0"])]
    have = [(L, k) for L, k in kd if k in cells]
    for L, k in have:
        lo, hi, _ = ci[k]
        print(f"  L={L:2d}: rho_50 = {r50[k][0]:.3f} [{lo:.3f}, {hi:.3f}]")
    if len(have) == 3:
        v = [r50[k][0] for _, k in have]
        mono = v[0] > v[1] > v[2]
        gap = v[0] - v[1]
        print(f"  monotone decreasing in L: {mono}    gap(8->12) = {gap:+.3f} (bar >= 0.05) "
              f"{'PASS' if gap >= 0.05 else 'FAIL'}")
        if s_seed is not None:
            print(f"  gap vs seed yardstick {s_seed:.3f}: "
                  f"{'gap clears the seed term' if gap > s_seed else 'NOT RESOLVABLE above seed noise'}")
    else:
        print(f"  pending ({len(have)}/3 depth cells)")
    print()

    # ------------------------------------------------------------------- H3
    print("=== H3: data. 240k beats 120k by >= 0.02 in rho_50, by less than the depth gap ===")
    k3 = TASKS["t3 L=12 120k s0 (data axis)"]
    if k0 in cells and k3 in cells:
        d = r50[k3][0] - r50[k0][0]
        print(f"  rho_50: 120k {r50[k3][0]:.3f}   240k {r50[k0][0]:.3f}   "
              f"240k advantage = {d:+.3f} (bar >= +0.02) {'PASS' if d >= 0.02 else 'FAIL'}")
        if d < 0:
            print("  >>> 120k WINS: the pre-registered falsifier. N=96 is optimization-bound; "
                  "the fix is more steps, not more instances.")
        if s_seed is not None and abs(d) <= s_seed:
            print(f"  >>> |{d:+.3f}| <= seed yardstick {s_seed:.3f}: not resolvable either way")
    else:
        print("  pending")
    print()

    # ------------------------------------------------------- trainability
    print("=== val_nll per cell (steps-bound? plateau? overfit?) ===")
    print("  cell                          ep  train_nll  val_nll  best_val   gap  last5_drop")
    for lab, k in TASKS.items():
        if k not in cells:
            continue
        p = cells[k][0]["path"]
        final, best, train, nep = history(p)
        try:
            with open(os.path.join(p, "history.json")) as fh:
                h = json.load(fh)
            last5 = [float(e["nll"]) for e in h[-5:]]
            drop = last5[-1] - last5[0]
        except (OSError, KeyError, IndexError, ValueError):
            drop = float("nan")
        print(f"  {lab:28s} {nep:3d}  {train:9.4f}  {final:7.4f}  {best:8.4f}  "
              f"{final-train:+5.3f}  {drop:+.4f}")


if __name__ == "__main__":
    main()
