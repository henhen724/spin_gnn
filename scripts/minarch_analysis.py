"""Score M1-M7 of the minimal-architecture grid (51032450).

The pre-registration is the header of
scripts/slurm/sudoku2_n48_minarch_2026-11-04.sbatch; this file computes the
expressions it names and nothing else (finding 39's lesson: compute the
expression, do not re-derive it from a sentence).

    M1  there is a breaking depth B inside L in {1,2,3,4,6,8,12}, predicted 2
    M2  rho_50(L=2) - rho_50(L=4) >= 2 s*, the precondition for reading M1 as
        anything finer than a yes/no
    M3  linear is resolvably worse than full at L=4 (and a TIE is the headline
        of the project, not a null)
    M4  rf_mlp ties full at L=4 within s*
    M5  rf_all's rho_50 is censored and its mean_solve_rate < 0.10
    M6  no_mlp: no registered prediction, it brackets M3/M4 from below
    M7  nhead 4 -> 1 at L=4 ties within s*

THREE METRICS, BECAUSE rho_50 CENSORS. This grid deliberately walks every axis
until it breaks, so most of its cells are EXPECTED not to cross solve 0.5
anywhere in the grid, and a censored rho_50 carries no information about how
badly a cell failed. Every comparison below is therefore reported on

    rho_50           the metric of record, censored as ">0.95" and never
                     silently clipped to the grid edge
    mean_solve_rate  unweighted mean of solve rate over the 14 bins; never
                     censors, so it is what actually orders the broken cells
    solve_at_floor   solve rate at the lowest realized rho, the hardest bin

and where rho_50 is censored on either side of a comparison, the verdict is
taken from mean_solve_rate and SAID to be.

THE YARDSTICK IS COMPUTED, NOT ASSUMED. s* = the largest across-seed spread
(max - min) on any rung in this job, per metric. Finding 41/42 measured a
same-architecture seed spread of 0.1627 at N=48, larger than the depth gaps the
earlier ladders were built from, and finding 47 showed that term is not a
schedule artifact. So no gap smaller than s* is called a difference, and that
bar is read off this job's own replicates rather than carried over.

TRAINABILITY IS NOT CAPACITY. Each rung reports how many of its seeds got below
finding 44's train-NLL plateau of 1.64. A rung with 0/3 escapes is reported as
"did not train at this budget", never as "cannot represent DD" -- the two are
separable only by more steps or a better optimizer, neither of which is in this
job.

Partial runs are expected: 36 cells finish over ~11 h, so this reports whatever
is on disk and names what is missing. Run it again when the rest land.

Usage:  python3 scripts/minarch_analysis.py [--scan-dir DIR] [--boot N]
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

from scripts.n96_fallscan_analysis import rho_50, rho50_ci

SCAN_DEFAULT = os.path.join(
    os.environ.get("SCRATCH", "/scratch/users/hshunt"),
    "spin_gnn", "sudoku_n48_minarch_2026-11-04")

SEEDS = (0, 1, 2)
# (L, nhead, arch_mode) -> label. The hub L=4/h4/full appears in all three
# axes and is measured once.
DEPTH_LADDER = [12, 8, 6, 4, 3, 2, 1]
ARCH_AXIS = ["full", "linear", "rf_mlp", "rf_all", "no_mlp"]
PLATEAU = 1.64  # finding 44's L=4/ff=1024/240k train NLL, never escaped


def rung_label(L, nhead, arch):
    return f"L={L:<2d} h={nhead} {arch}"


def all_rungs():
    rungs = [(L, 4, "full") for L in DEPTH_LADDER]
    rungs += [(4, 4, a) for a in ARCH_AXIS if a != "full"]
    rungs += [(4, 1, "full")]
    return rungs


def load(scan_dir):
    """Rows from every rho_curve CSV, one per (cell, rho) bin."""
    rows = []
    for f in sorted(glob.glob(os.path.join(scan_dir, "rho_curve_t*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                for k in ("rho", "realized_rho", "solve_rate", "outcome_accuracy",
                          "well_formed_rate", "truncated_rate", "dd_mean_T"):
                    r[k] = float(r[k])
                for k in ("num_layers", "n_params", "seed", "n_instances"):
                    r[k] = int(r[k])
                # nhead and arch_mode were added to this CSV on 11-04; fall
                # back to the checkpoint's own config.json rather than
                # defaulting, since defaulting arch_mode wrong is exactly the
                # silent failure the column exists to prevent.
                if not r.get("arch_mode") or not r.get("nhead"):
                    cfg = _config(r["path"])
                    r["arch_mode"] = r.get("arch_mode") or cfg.get("arch_mode", "full")
                    r["nhead"] = r.get("nhead") or cfg.get("nhead")
                r["nhead"] = int(r["nhead"])
                rows.append(r)
    return rows


_CFG, _HIST = {}, {}


def _config(path):
    if path not in _CFG:
        try:
            with open(os.path.join(path, "config.json")) as fh:
                _CFG[path] = json.load(fh)
        except (OSError, ValueError):
            _CFG[path] = {}
    return _CFG[path]


def history(path):
    """(n_epochs, final_train_nll, final_val, best_val, last5_train_slope)."""
    if path not in _HIST:
        try:
            with open(os.path.join(path, "history.json")) as fh:
                h = json.load(fh)
            tail = [float(e["train_nll"]) for e in h[-5:]]
            slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
            _HIST[path] = (len(h), float(h[-1]["train_nll"]), float(h[-1]["nll"]),
                           min(float(e["nll"]) for e in h), slope)
        except (OSError, KeyError, IndexError, ValueError):
            _HIST[path] = (0, float("nan"), float("nan"), float("nan"), float("nan"))
    return _HIST[path]


def cell_metrics(rs, boot):
    """The three metrics plus controls for one cell (one checkpoint)."""
    v, flag = rho_50([(r["realized_rho"], r["solve_rate"]) for r in rs])
    lo, hi, _drop = rho50_ci(rs, n_boot=boot) if flag == "ok" else (float("nan"),) * 3
    floor_rho = min(r["realized_rho"] for r in rs)
    at_floor = min(r["solve_rate"] for r in rs if r["realized_rho"] == floor_rho)
    return {
        "rho50": v, "flag": flag, "ci": (lo, hi),
        "mean_solve": sum(r["solve_rate"] for r in rs) / len(rs),
        "solve_at_floor": at_floor,
        "best_solve": max(r["solve_rate"] for r in rs),
        "n_params": rs[0]["n_params"],
        "n_trainable": int(rs[0].get("n_trainable_params") or rs[0]["n_params"]),
        "path": rs[0]["path"],
        "bins": len(rs),
    }


def fmt_rho50(m):
    """Censored cells print as a bound, never as a number at the grid edge."""
    if m["flag"] == "ok":
        return f"{m['rho50']:.3f}"
    return ">0.95" if m["flag"] == ">top" else f"<{m['rho50']:.2f}"


def spread(vals):
    return max(vals) - min(vals) if len(vals) > 1 else float("nan")


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def compare(name, a, b, yard, metric, higher_is_better):
    """One cross-rung comparison, scored against the yardstick and nothing else."""
    if a is None or b is None:
        print(f"  {name}: pending")
        return None
    d = a[metric] - b[metric]
    better = (d > 0) == higher_is_better
    verdict = ("not separable at this budget" if abs(d) <= yard
               else ("BETTER" if better else "WORSE"))
    print(f"  {name}: {metric} {a[metric]:.4f} vs {b[metric]:.4f}  "
          f"delta {d:+.4f}  vs s*={yard:.4f}  ->  {verdict}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default=SCAN_DEFAULT)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    rows = load(args.scan_dir)
    if not rows:
        print(f"no rho_curve_t*.csv under {args.scan_dir} yet")
        return
    cells = defaultdict(list)
    for r in rows:
        cells[(r["num_layers"], r["nhead"], r["arch_mode"], r["seed"])].append(r)

    want = [(L, h, a, s) for (L, h, a) in all_rungs() for s in SEEDS]
    have = [k for k in want if k in cells]
    print(f"{len(rows)} rows / {len(cells)} cells under {args.scan_dir}")
    print(f"cells present: {len(have)}/{len(want)}\n")

    met = {k: cell_metrics(cells[k], args.boot) for k in have}

    # ------------------------------------------------------------- controls
    print("=== controls ===")
    leaky = {(r["shuffle_adj"], r["fall"]) for r in rows}
    print(f"  (shuffle_adj, fall) over every row: {leaky}  "
          f"{'OK -- leak-free fall ensemble throughout' if leaky == {('True', 'True')} else '<-- CHECK'}")
    # The cell key is (L, nhead, arch_mode, seed) and deliberately does NOT
    # include the training budget, because this grid holds it fixed at 375k
    # steps. Pointed at a directory that varies it, two cells would silently
    # merge into one, so say so rather than averaging over the difference.
    budgets = {(r.get("n_train_instances"), r.get("epochs")) for r in rows}
    print(f"  training budgets present: {sorted(budgets)}"
          + ("" if len(budgets) == 1 else "  <-- MORE THAN ONE: cells with the same "
                                          "(L, nhead, arch, seed) have been merged, re-key before reading anything below"))
    by_req = defaultdict(set)
    for r in rows:
        by_req[r["rho"]].add(round(r["realized_rho"], 6))
    bad = {q: v for q, v in by_req.items() if len(v) > 1}
    print(f"  requested rho whose realized_rho differs across cells: {len(bad)} "
          f"(0 = every curve paired point by point)")
    floor = min(r["realized_rho"] for r in rows)
    print(f"  fall-ensemble floor realized_rho = {floor:.4f}  "
          f"(measured rho_min at N=48 is 0.332 +- 0.024)")
    live = [r for r in rows if r["solve_rate"] > 0.1]
    if live:
        print(f"  over {len(live)} bins with solve>0.1: max truncated_rate "
              f"{max(r['truncated_rate'] for r in live):.4f} (bar 0.01), min well_formed "
              f"{min(r['well_formed_rate'] for r in live):.4f} (bar 0.99)")
    dead = [r for r in rows if r["solve_rate"] <= 0.1]
    if dead:
        print(f"  the {len(dead)} bins with solve<=0.1 are NOT held to that bar: a broken cell "
              f"is expected to babble (min well_formed {min(r['well_formed_rate'] for r in dead):.4f})")
    print()

    # ---------------------------------------------------------- the per-rung
    print("=== every rung: 3 seeds, median and spread ===")
    print("  rung             params/trainable    rho_50 (per seed)        "
          "median  spread | mean_solve med [spread] | floor | escapes")
    rung_med, rung_spread = {}, {}
    for (L, h, a) in all_rungs():
        ks = [(L, h, a, s) for s in SEEDS if (L, h, a, s) in met]
        if not ks:
            print(f"  {rung_label(L, h, a):16s} (not started)")
            continue
        ms = [met[k] for k in ks]
        r50s = [fmt_rho50(m) for m in ms]
        ok = [m["rho50"] for m in ms if m["flag"] == "ok"]
        med_r50 = f"{median(ok):.3f}" if len(ok) == len(ms) else "censored"
        sp_r50 = f"{spread(ok):.3f}" if len(ok) == len(ms) and len(ok) > 1 else "--"
        msolve = [m["mean_solve"] for m in ms]
        esc = sum(1 for m in ms if history(m["path"])[1] < PLATEAU)
        rung_med[(L, h, a)] = {
            "rho50": median(ok) if len(ok) == len(ms) else float("nan"),
            "censored": len(ok) < len(ms),
            "mean_solve": median(msolve),
            "solve_at_floor": median([m["solve_at_floor"] for m in ms]),
            "n_seeds": len(ms),
        }
        rung_spread[(L, h, a)] = {"rho50": spread(ok) if len(ok) == len(ms) else float("nan"),
                                  "mean_solve": spread(msolve)}
        print(f"  {rung_label(L, h, a):16s} {ms[0]['n_params']:>9,}/{ms[0]['n_trainable']:<9,} "
              f" {'/'.join(r50s):24s} {med_r50:>8s} {sp_r50:>7s} | "
              f"{median(msolve):.4f} [{spread(msolve):.4f}] | "
              f"{median([m['solve_at_floor'] for m in ms]):.3f} | {esc}/{len(ms)}"
              + ("" if len(ms) == 3 else f"   <-- only {len(ms)} seed(s) so far"))
    print()

    # ------------------------------------------------------------ yardstick
    s_r50 = [v["rho50"] for k, v in rung_spread.items()
             if not math.isnan(v["rho50"]) and rung_med[k]["n_seeds"] > 1]
    s_ms = [v["mean_solve"] for k, v in rung_spread.items()
            if not math.isnan(v["mean_solve"]) and rung_med[k]["n_seeds"] > 1]
    yard_r50 = max(s_r50) if s_r50 else float("nan")
    yard_ms = max(s_ms) if s_ms else float("nan")
    print("=== the yardstick s*, read off this job's own replicates ===")
    print(f"  largest spread on any complete rung:  rho_50 {yard_r50:.4f}"
          f"   mean_solve {yard_ms:.4f}")
    print(f"  (finding 41/42 measured 0.1627 in rho_50 at N=48; finding 47 showed it is not a "
          f"schedule artifact)")
    if math.isnan(yard_ms):
        print("  >>> no complete rung yet -- NO comparison below is interpretable")
    print()

    # ------------------------------------------------------------------- M1
    print("=== M1: the breaking depth B (largest L where some seed fails to cross 0.5) ===")
    B = None
    for L in DEPTH_LADDER:
        k = (L, 4, "full")
        if k not in rung_med:
            print(f"  L={L:<2d} pending")
            continue
        ks = [(L, 4, "full", s) for s in SEEDS if (L, 4, "full", s) in met]
        n_cross = sum(1 for kk in ks if met[kk]["flag"] == "ok")
        print(f"  L={L:<2d} {n_cross}/{len(ks)} seeds cross 0.5   "
              f"median mean_solve {rung_med[k]['mean_solve']:.4f}")
        if n_cross < len(ks) and B is None:
            B = L
    if B is not None:
        print(f"  >>> B = {B} (predicted 2).  "
              + ("PASS" if B == 2 else "the prediction was B=2; this is the measured number"))
    else:
        print("  >>> no rung yet has a seed that fails to cross -- B is below the measured range "
              "or the ladder is incomplete")
    print()

    # ------------------------------------------------------------------- M2
    print("=== M2: can the ladder resolve a knee? rho_50(L=2) - rho_50(L=4) >= 2 s* ===")
    k2, k4 = (2, 4, "full"), (4, 4, "full")
    if k2 in rung_med and k4 in rung_med:
        if rung_med[k2]["censored"] or rung_med[k4]["censored"]:
            d = rung_med[k2]["mean_solve"] - rung_med[k4]["mean_solve"]
            print(f"  rho_50 is censored on at least one side, so the comparison runs on "
                  f"mean_solve: {d:+.4f} against 2 s* = {2 * yard_ms:.4f}  "
                  f"-> {'resolvable' if abs(d) >= 2 * yard_ms else 'NOT resolvable'}")
            print("  (a censored L=2 against an uncensored L=4 is itself the knee: the rung "
                  "stopped crossing. M2's bar is then about the SIZE of the step, not its sign.)")
        else:
            d = rung_med[k2]["rho50"] - rung_med[k4]["rho50"]
            print(f"  rho_50: L=2 {rung_med[k2]['rho50']:.3f}  L=4 {rung_med[k4]['rho50']:.3f}  "
                  f"delta {d:+.3f}  against 2 s* = {2 * yard_r50:.3f}  "
                  f"-> {'PASS' if d >= 2 * yard_r50 else 'FAIL -- the knee is inside the seed term'}")
    else:
        print("  pending")
    print()

    # ---------------------------------------------------------------- M3-M7
    hub = rung_med.get((4, 4, "full"))
    print("=== M3-M7: what each removal costs, all against the L=4/h4/full hub ===")
    print("  (verdicts on mean_solve, which never censors; rho_50 medians are in the rung "
          "table above)")
    compare("M3 linear (nonlinearity removed)", rung_med.get((4, 4, "linear")), hub,
            yard_ms, "mean_solve", True)
    if (4, 4, "linear") in rung_med and hub and \
            abs(rung_med[(4, 4, "linear")]["mean_solve"] - hub["mean_solve"]) <= yard_ms:
        print("      >>> linear TIES full. This is the pre-registered headline, not a null: "
              "with an identity MLP activation there is no ELEMENTWISE nonlinearity left "
              "anywhere in the stack. What remains is the attention softmax and LayerNorm "
              "(pre-norm, so 2L+1 of them), and both act as input-dependent SCALARS on "
              "otherwise linear maps -- softmax sets the mixing coefficients, LayerNorm "
              "removes a mean (linear) and divides by an RMS (one scalar per token). The "
              "whole network is therefore an input-dependent linear operator on the embedded "
              "sequence, which is a far more tractable object than a ReLU stack even though "
              "it is NOT yet closed form. FOLLOW UP IMMEDIATELY.")
    compare("M4 rf_mlp (linear1 frozen)      ", rung_med.get((4, 4, "rf_mlp")), hub,
            yard_ms, "mean_solve", True)
    compare("M6 no_mlp (attention only)      ", rung_med.get((4, 4, "no_mlp")), hub,
            yard_ms, "mean_solve", True)
    compare("M7 nhead 1 vs 4                 ", rung_med.get((4, 1, "full")), hub,
            yard_ms, "mean_solve", True)
    print()

    print("=== M5: the convex cell (rf_all). Censored rho_50 AND mean_solve < 0.10? ===")
    kr = (4, 4, "rf_all")
    if kr in rung_med:
        ks = [(4, 4, "rf_all", s) for s in SEEDS if (4, 4, "rf_all", s) in met]
        n_cross = sum(1 for k in ks if met[k]["flag"] == "ok")
        ms = rung_med[kr]["mean_solve"]
        print(f"  {n_cross}/{len(ks)} seeds cross 0.5   median mean_solve {ms:.4f} (bar < 0.10)   "
              f"median solve_at_floor {rung_med[kr]['solve_at_floor']:.4f}")
        wf = [min(r["well_formed_rate"] for r in cells[k]) for k in ks]
        print(f"  grammar even where it cannot solve: min well_formed over bins "
              f"{min(wf):.4f}, best single-bin solve "
              f"{max(met[k]['best_solve'] for k in ks):.4f}")
        if ms >= 0.10:
            print("  >>> FALSIFIED, and this is the most important cell in the grid: a CONVEX "
                  "problem on features frozen at initialization does part of DD. The analytic "
                  "programme runs through this cell -- make it the main object of study.")
        else:
            print(f"  >>> as predicted: frozen attention cannot learn where to look. "
                  f"The informative number is how far it got: mean_solve {ms:.4f}.")
    else:
        print("  pending")
    print()

    # ------------------------------------------------------- trainability
    print("=== W: trainability, NOT capacity (finding 44's plateau = train NLL 1.64) ===")
    print("  rung             ep  train_nll (per seed)        final_val  best_val  last5 slope")
    for (L, h, a) in all_rungs():
        ks = [(L, h, a, s) for s in SEEDS if (L, h, a, s) in met]
        if not ks:
            continue
        hs = [history(met[k]["path"]) for k in ks]
        tr = "/".join(f"{x[1]:.3f}" for x in hs)
        esc = sum(1 for x in hs if x[1] < PLATEAU)
        note = "" if esc else "   <-- 0/3 escaped: DID NOT TRAIN at this budget, not 'cannot represent DD'"
        print(f"  {rung_label(L, h, a):16s} {hs[0][0]:3d}  {tr:26s} "
              f"{median([x[2] for x in hs]):9.4f} {median([x[3] for x in hs]):9.4f} "
              f"{median([x[4] for x in hs]):+.5f}{note}")


if __name__ == "__main__":
    main()
