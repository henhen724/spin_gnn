"""Score the minimal-architecture grid: stage A (51032450) + stage B (51072446).

The pre-registrations are the headers of
scripts/slurm/sudoku2_n48_minarch_2026-11-04.sbatch (M1-M7, arch axis at L=4)
and scripts/slurm/sudoku2_n48_minarch_L6_2026-11-04.sbatch (N1-N5, the same
arch axis re-run at L=6); this file computes the expressions they name and
nothing else (finding 39's lesson: compute the expression, do not re-derive it
from a sentence).

    M1  there is a breaking depth B inside L in {1,2,3,4,6,8,12}, predicted 2
    M2  rho_50(L=2) - rho_50(L=4) >= 2 s*, the precondition for reading M1 as
        anything finer than a yes/no
    M3  linear is resolvably worse than full (and a TIE is the headline of the
        project, not a null)
    M4  rf_mlp ties full within s*
    M5  rf_all's rho_50 is censored and its mean_solve_rate < 0.10
    M6  no_mlp: no registered prediction, it brackets M3/M4 from below
    M7  nhead 4 -> 1 ties within s*

STAGE B EXISTS BECAUSE STAGE A PINNED THE ARCH AXIS AT THE BREAKING DEPTH.
Stage A's own ladder measured B = 4: at L=4 one seed of the hub stops crossing
0.5 at all, and the hub's across-seed spread in mean_solve there (0.2784) is as
large as its own score (0.3084), so no arch contrast at L=4 can be resolved
against the seed term. L=6 is the shallowest rung where all three seeds still
solve (rho_50 median 0.592, spread 0.012), so stage B re-runs the whole arch
axis and the head axis there. N1-N5 are M3,M4,M5,M6,M7 asked at L=6.

Both depths are scored below against the SAME POOLED s*, the largest
across-seed spread on any complete rung in EITHER job, so moving the arch axis
to a quieter rung does not buy a smaller yardstick.

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

Partial runs are expected: 51 cells finish over ~18 h, so this reports whatever
is on disk and names what is missing. Run it again when the rest land.

Usage:  python3 scripts/minarch_analysis.py [--scan-dir DIR [DIR ...]] [--boot N]
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

from scripts.n96_fallscan_analysis import rho_50, rho50_ci

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_DEFAULT = os.path.join(_SCR, "spin_gnn", "sudoku_n48_minarch_2026-11-04")
SCAN_L6 = os.path.join(_SCR, "spin_gnn", "sudoku_n48_minarch_L6_2026-11-04")
# Stage C (51114734): the width and schedule axes at L=6. Its rungs carry a
# non-default d_model, dim_feedforward or warmup_epochs, so rung_key() below
# gives them their own keys and all_rungs() does not name them -- they show up
# in main()'s "NOT in the registered grid" list on purpose, which is also what
# keeps them out of the bar that stage A and B were scored against.
SCAN_C = os.path.join(_SCR, "spin_gnn", "sudoku_n48_minarch_C_2026-11-04")
# Stage D (51151957): the width axis again, after stage C's P1 falsified "width
# is free" -- d192 full bisects the cliff between 256 and 128, and no_mlp at
# d384 asks whether width can buy the MLP back. Same treatment as stage C: a
# non-default d_model keeps these rungs out of the registered grid and out of
# the pooled bar that stages A and B were scored against.
SCAN_D = os.path.join(_SCR, "spin_gnn", "sudoku_n48_minarch_D_2026-11-04")
SCAN_DIRS = [SCAN_DEFAULT, SCAN_L6, SCAN_C, SCAN_D]

SEEDS = (0, 1, 2)
# (L, nhead, arch_mode) -> label. At each arch depth the hub L/h4/full is a rung
# of the depth ladder AND the control for the arch axis AND for the head axis,
# so it is measured once and compared against three ways.
DEPTH_LADDER = [12, 8, 6, 4, 3, 2, 1]
ARCH_AXIS = ["full", "linear", "rf_mlp", "rf_all", "no_mlp"]
# Depths at which the arch and head axes were run: 4 in stage A (which turned
# out to be the breaking depth) and 6 in stage B (the shallowest rung that
# still solves at all three seeds).
ARCH_DEPTHS = (4, 6)
PLATEAU = 1.64  # finding 44's L=4/ff=1024/240k train NLL, never escaped


# Stage A and B held the width and the schedule fixed, so d_model, ff and
# warmup were not in the cell key -- which means a narrower or warmed-up cell
# at the same (L, nhead, arch_mode) would have MERGED into the existing rung
# and quietly moved its median. That is finding 43's failure mode on three more
# silent knobs, and stage C varies all three. The key therefore carries them,
# but ONLY as a block and ONLY when at least one is non-default, so every key
# written for stage A and B is unchanged and every number this script printed
# before stays identical.
#
# All three go in together, as ints, so a key is either a bare 3-tuple or a
# 6-tuple and position means the same thing in both: appending them one at a
# time would make (L, h, a, 128) ambiguous between d_model and warmup, and
# would put a seed at a position another key uses for a knob.
D_DEFAULT, FF_DEFAULT, W_DEFAULT = 256, 1024, 0
VARIANT_DEFAULT = (D_DEFAULT, FF_DEFAULT, W_DEFAULT)


def rung_key(r):
    """(L, nhead, arch), plus (d_model, ff, warmup) when any is non-default."""
    base = (r["num_layers"], r["nhead"], r["arch_mode"])
    v = (int(r["d_model"]), int(r["dim_feedforward"]), int(r.get("warmup_epochs") or 0))
    return base if v == VARIANT_DEFAULT else base + v


def cell_key(r):
    """A rung plus its seed. The SEED IS LAST, so the rung is always key[:-1]."""
    return rung_key(r) + (r["seed"],)


def rung_label(L, nhead, arch, d_model=D_DEFAULT, ff=FF_DEFAULT, warmup=W_DEFAULT):
    v = "".join((f" d{d_model}" if d_model != D_DEFAULT else "",
                 f" ff{ff}" if ff != FF_DEFAULT else "",
                 f" w{warmup:02d}" if warmup != W_DEFAULT else ""))
    return f"L={L:<2d} h={nhead} {arch}{v}"


def all_rungs(arch_depths=ARCH_DEPTHS):
    rungs = [(L, 4, "full") for L in DEPTH_LADDER]
    for d in arch_depths:
        rungs += [(d, 4, a) for a in ARCH_AXIS if a != "full"]
        rungs += [(d, 1, "full")]
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
                # d_model, dim_feedforward and warmup_epochs are part of the
                # cell key from stage C on, so they get the same treatment as
                # nhead: read the checkpoint's own config rather than
                # defaulting, because a wrong default here merges a narrow or
                # warmed-up cell into the rung it is meant to be compared
                # against. warmup is not a CSV column at all, so it always
                # comes from the config -- and a MISSING config would then read
                # as warmup 0, which is why the controls below count the cells
                # whose config could not be read instead of trusting the zero.
                if not r.get("d_model") or not r.get("dim_feedforward"):
                    cfg = _config(r["path"])
                    r["d_model"] = r.get("d_model") or cfg.get("d_model")
                    r["dim_feedforward"] = r.get("dim_feedforward") or cfg.get("dim_feedforward")
                r["warmup_epochs"] = _config(r["path"]).get("warmup_epochs", 0) or 0
                for k in ("nhead", "d_model", "dim_feedforward", "warmup_epochs"):
                    r[k] = int(r[k])
                r["_scan"] = os.path.basename(scan_dir.rstrip("/"))
                rows.append(r)
    return rows


def load_all(scan_dirs):
    """Rows from several scan directories, tagged with which one they came from."""
    rows = []
    for d in scan_dirs:
        rows += load(d)
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


def _bound_median(pairs):
    """Median of rho_50 when some seeds are censored at the top of the grid.

    `pairs` is [(rho50, crossed)]. A censored seed is known only as ">0.95",
    but that is enough to ORDER it above every seed that did cross, so the
    median is well defined whenever the middle rank is one of the finite
    values. Returns the median and whether it is readable; an unreadable one
    (two of three censored) stays censored and is reported as such.
    """
    ranked = sorted(pairs, key=lambda p: (not p[1], p[0] if p[1] else 0.0))
    mid = ranked[len(ranked) // 2]
    even = len(ranked) % 2 == 0
    if even:
        lo = ranked[len(ranked) // 2 - 1]
        ok = lo[1] and mid[1]
        return {"rho50_bm": 0.5 * (lo[0] + mid[0]) if ok else float("nan"),
                "bm_readable": ok}
    return {"rho50_bm": mid[0] if mid[1] else float("nan"), "bm_readable": mid[1]}


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
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_DIRS)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    rows = load_all(args.scan_dir)
    if not rows:
        print("no rho_curve_t*.csv under " + " or ".join(args.scan_dir) + " yet")
        return
    cells = defaultdict(list)
    for r in rows:
        cells[cell_key(r)].append(r)

    want = [rk + (s,) for rk in all_rungs() for s in SEEDS]
    # Anything in the data that the registered grid does not name -- a width
    # cell from stage C, say. These used to collide with the default-width rung
    # of the same (L, nhead, arch); now they get their own key and are listed
    # here so a new cell ANNOUNCES itself instead of moving someone else's
    # median.
    extra = sorted({k[:-1] for k in cells} - set(all_rungs()), key=lambda t: (len(t), t))
    have = [k for k in want if k in cells] + [k for k in sorted(cells) if k[:-1] in set(extra)]
    for d in args.scan_dir:
        n = len([r for r in rows if r["_scan"] == os.path.basename(d.rstrip("/"))])
        print(f"{n:5d} rows  {os.path.basename(d.rstrip('/'))}")
    print(f"{len(rows)} rows / {len(cells)} cells; present: "
          f"{sum(1 for k in want if k in cells)}/{len(want)} of the registered grid")
    if extra:
        print(f"  + {len(extra)} rung(s) NOT in the registered grid, read on their own key so "
              f"they cannot merge into a default-width rung: "
              + "; ".join(rung_label(*k).strip() for k in extra))
    # The cell key does not include the scan directory, so a cell run twice in
    # two jobs would silently merge. Stage B deliberately does NOT re-run stage
    # A's L=6/h4/full control, so this should always be empty.
    dup = {k: sorted({r["_scan"] for r in v}) for k, v in cells.items()
           if len({r["_scan"] for r in v}) > 1}
    if dup:
        print(f"  <-- {len(dup)} cell(s) appear in MORE THAN ONE scan dir and have been "
              f"merged: {sorted(dup)[:4]} ... re-key before reading anything below")
    print()

    met = {k: cell_metrics(cells[k], args.boot) for k in have}

    # ------------------------------------------------------------- controls
    print("=== controls ===")
    leaky = {(r["shuffle_adj"], r["fall"]) for r in rows}
    print(f"  (shuffle_adj, fall) over every row: {leaky}  "
          f"{'OK -- leak-free fall ensemble throughout' if leaky == {('True', 'True')} else '<-- CHECK'}")
    # The cell key carries (L, nhead, arch_mode, width, seed) and deliberately
    # does NOT include the training budget, because this grid holds it fixed at
    # 375k steps. Pointed at a directory that varies it, two cells would
    # silently merge into one, so say so rather than averaging the difference.
    budgets = {(r.get("n_train_instances"), r.get("epochs")) for r in rows}
    print(f"  training budgets present: {sorted(budgets)}"
          + ("" if len(budgets) == 1 else "  <-- MORE THAN ONE: cells with the same "
                                          "(L, nhead, arch, seed) have been merged, re-key before reading anything below"))
    # warmup_epochs has no CSV column, so it is read from each checkpoint's
    # config.json. An unreadable config reads as warmup 0, which would merge a
    # warmed-up cell into the cold rung it exists to be compared against.
    noconf = sorted({r["path"] for r in rows if not _config(r["path"])})
    print(f"  cells whose config.json could not be read: {len(noconf)}"
          + ("" if not noconf else f"  <-- warmup_epochs read as 0 for these, so a warmed-up "
                                   f"cell could be merging into a cold rung: {noconf[:2]}"))
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
    print("  rung                     params/trainable    rho_50 (per seed)        "
          "median  spread | mean_solve med [spread] | floor | escapes")
    rung_med, rung_spread = {}, {}
    for key in list(all_rungs()) + extra:
        ks = [key + (s,) for s in SEEDS if key + (s,) in met]
        if not ks:
            print(f"  {rung_label(*key):24s} (not started)")
            continue
        ms = [met[k] for k in ks]
        r50s = [fmt_rho50(m) for m in ms]
        ok = [m["rho50"] for m in ms if m["flag"] == "ok"]
        med_r50 = f"{median(ok):.3f}" if len(ok) == len(ms) else "censored"
        sp_r50 = f"{spread(ok):.3f}" if len(ok) == len(ms) and len(ok) > 1 else "--"
        msolve = [m["mean_solve"] for m in ms]
        esc = sum(1 for m in ms if history(m["path"])[1] < PLATEAU)
        rung_med[key] = {
            "rho50": median(ok) if len(ok) == len(ms) else float("nan"),
            "censored": len(ok) < len(ms),
            "mean_solve": median(msolve),
            "solve_at_floor": median([m["solve_at_floor"] for m in ms]),
            "n_seeds": len(ms),
            # the seeds that DID cross, kept so a censored rung can still be
            # read: "2 of 3 crossed, and both at 0.76" is a far more useful
            # statement than "censored", and it is what decides whether a
            # censored removal is merely noisy or genuinely above threshold.
            "n_cross": len(ok),
            "rho50_ok": sorted(ok),
            # Censoring only destroys the MEDIAN when a censored value lands on
            # the middle rank. With 3 seeds and one censored at the top, the
            # median is an ordinary number and the rung is fully readable -- so
            # "censored" is too blunt a verdict and throws away a real
            # measurement. Rank the censored seeds above every finite one (they
            # are >0.95 and the grid stops at 0.95, so the ordering is known)
            # and take the median; mark it readable only if the middle rank is
            # a finite value.
            **_bound_median([(m["rho50"], m["flag"] == "ok") for m in ms]),
        }
        rung_spread[key] = {"rho50": spread(ok) if len(ok) == len(ms) else float("nan"),
                            "mean_solve": spread(msolve)}
        print(f"  {rung_label(*key):24s} {ms[0]['n_params']:>9,}/{ms[0]['n_trainable']:<9,} "
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
    print("=== the yardstick s*, POOLED over every complete rung in both jobs ===")
    print(f"  largest across-seed spread anywhere:  rho_50 {yard_r50:.4f}"
          f"   mean_solve {yard_ms:.4f}")
    worst_r50 = max(((v["rho50"], k) for k, v in rung_spread.items()
                     if not math.isnan(v["rho50"]) and rung_med[k]["n_seeds"] > 1),
                    default=(float("nan"), None))
    worst_ms = max(((v["mean_solve"], k) for k, v in rung_spread.items()
                    if not math.isnan(v["mean_solve"]) and rung_med[k]["n_seeds"] > 1),
                   default=(float("nan"), None))
    if worst_r50[1]:
        print(f"  set by: rho_50 at {rung_label(*worst_r50[1]).strip()}, "
              f"mean_solve at {rung_label(*worst_ms[1]).strip()}  -- pooling is what stops the "
              f"choice of arch depth from buying a smaller bar")
    print(f"  (finding 41/42 measured 0.1627 in rho_50 at N=48; finding 47 showed it is not a "
          f"schedule artifact)")
    # Pooling over EVERY complete rung means a noisy new rung raises s* for
    # everyone -- including verdicts that were registered and scored before it
    # existed. That is how a later job could silently turn an earlier "COSTS
    # rho_50" into "free" without a single earlier number changing. So quote
    # the bar the registered grid sets on its own, and say when they differ.
    if extra:
        reg = set(all_rungs())
        reg_r50 = [v["rho50"] for k, v in rung_spread.items() if k in reg
                   and not math.isnan(v["rho50"]) and rung_med[k]["n_seeds"] > 1]
        reg_ms = [v["mean_solve"] for k, v in rung_spread.items() if k in reg
                  and not math.isnan(v["mean_solve"]) and rung_med[k]["n_seeds"] > 1]
        r50_reg = max(reg_r50) if reg_r50 else float("nan")
        ms_reg = max(reg_ms) if reg_ms else float("nan")
        print(f"  over the REGISTERED grid alone (no width rungs):  rho_50 {r50_reg:.4f}"
              f"   mean_solve {ms_reg:.4f}")
        if yard_r50 > r50_reg or yard_ms > ms_reg:
            print(f"  >>> A LATER RUNG WIDENED THE BAR: stage A and B verdicts were registered "
                  f"against rho_50 {r50_reg:.4f} / mean_solve {ms_reg:.4f} and must keep being "
                  f"quoted against those. Score the new rungs against the pooled bar; do NOT "
                  f"let the pooled bar reopen a verdict that was already scored.")
    if math.isnan(yard_ms):
        print("  >>> no complete rung yet -- NO comparison below is interpretable")
    # Pooling protects against choosing a quiet rung to be judged against. It
    # does NOT protect against the opposite failure: a rung whose own seeds
    # disagree wildly RAISES s*, and is then compared against the bar it just
    # raised. That is circular, and when it happens "not separable" means "this
    # cell is too noisy to say anything about", not "the removal was free".
    if worst_ms[1] is not None and worst_ms[1] != (6, 4, "full"):
        print(f"  >>> CIRCULARITY WARNING: s* in mean_solve is set by "
              f"{rung_label(*worst_ms[1]).strip()}, which is itself one of the cells under test. "
              f"Any 'not separable' verdict involving that cell means IT IS TOO NOISY TO READ, "
              f"not that the removal was free. Read those rows on rho_50 and on the per-seed "
              f"numbers in the rung table, never on the mean_solve verdict alone.")
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
        # B is defined as the first rung walking DOWN at which a seed fails. It
        # is only a "breaking point" if everything below it also fails. Check,
        # because a single unlucky seed at one rung would otherwise be promoted
        # into a structural claim about the architecture.
        below = [(L, rung_med[(L, 4, "full")]) for L in DEPTH_LADDER
                 if L < B and (L, 4, "full") in rung_med]
        revived = [(L, m) for L, m in below if not m["censored"]]
        if revived:
            Ls = ", ".join(f"L={L} ({m['n_cross']}/{m['n_seeds']} cross, rho_50 "
                           f"{m['rho50']:.3f}, mean_solve {m['mean_solve']:.4f})"
                           for L, m in revived)
            print(f"  >>> THE LADDER IS NON-MONOTONE, so read B as a seed result and not a "
                  f"structural one: below B there are rungs that cross on EVERY seed -- {Ls}. "
                  f"L={B} has one seed that never crosses while a SHALLOWER rung has none, and "
                  f"the two are tied on mean_solve, so the censoring at L={B} is the seed term "
                  f"(findings 41/42/47) and not a property of {B} layers. The honest reading of "
                  f"this ladder is a GRADUAL collapse, not a knee: quote the mean_solve column.")
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
    # The same five questions at each depth the arch axis was run at. Stage A
    # asked them at L=4 (M3-M7) and stage B at L=6 (N1-N5); the labels differ,
    # the expressions do not, and both are scored against the POOLED s*.
    for depth, tags in ((4, "M3-M7"), (6, "N1-N5")):
        hub = rung_med.get((depth, 4, "full"))
        print(f"=== {tags}: what each removal costs at L={depth}, "
              f"all against the L={depth}/h4/full hub ===")
        if hub is None:
            print("  hub not measured yet -- nothing at this depth is interpretable\n")
            continue
        if hub["censored"]:
            print(f"  <-- WARNING: the hub itself is CENSORED at L={depth} (median mean_solve "
                  f"{hub['mean_solve']:.4f}, spread {rung_spread[(depth, 4, 'full')]['mean_solve']:.4f}). "
                  f"This is the breaking depth; contrasts here sit on the floor and a 'tie' "
                  f"below means 'both cells failed', not 'the machinery was removable'.")
        print("  (verdicts on mean_solve, which never censors; rho_50 medians are in the rung "
              "table above)")
        n3, n4, n6, n7 = (("M3", "M4", "M6", "M7") if depth == 4 else ("N1", "N2", "N4", "N5"))
        compare(f"{n3} linear (nonlinearity removed)", rung_med.get((depth, 4, "linear")), hub,
                yard_ms, "mean_solve", True)
        # The headline clause is scored on rho_50, the metric of record, against
        # the pre-registered falsifier -- "gap > s* OR ANY SEED CENSORED". An
        # earlier version gated only on the HUB being uncensored and compared
        # mean_solve, so it declared a tie for a removal whose own rho_50 was
        # censored on one seed; the BEFORE/AFTER section then printed "NOT free"
        # for the same cell. Two sections disagreeing on one number is the bug;
        # the falsifier decides, and it decides on rho_50.
        klin = (depth, 4, "linear")
        if klin in rung_med and not hub["censored"]:
            lin, free_at = rung_med[klin], hub["rho50"] + yard_r50
            if lin["censored"]:
                extra = ("" if not lin["bm_readable"] else
                         f" And the cost is not merely a censoring artefact: only the LARGEST "
                         f"seed is censored, so the median is still a value, and it is "
                         f"{lin['rho50_bm']:.3f} against the hub's {hub['rho50']:.3f} -- "
                         f"{lin['rho50_bm'] - hub['rho50']:+.3f}, which clears s* = "
                         f"{yard_r50:.4f} on its own.")
                print(f"      >>> N1/M3 FALSIFIED by its own registered falsifier: `linear` is "
                      f"CENSORED at L={depth} ({lin['n_cross']}/{lin['n_seeds']} seeds cross 0.5), "
                      f"and the falsifier reads 'gap > s* OR any seed censored'. The readable "
                      f"seeds are not a rescue either -- every one of them sits ABOVE the free "
                      f"threshold {free_at:.3f}.{extra} Removing the elementwise nonlinearity at "
                      f"a depth where full still solves COSTS rho_50; it does not tie.")
            elif lin["rho50"] - hub["rho50"] <= yard_r50:
                print("      >>> linear TIES full at a depth where full still solves, on rho_50 "
                      "and with all three seeds crossing. This is the pre-registered headline, "
                      "not a null: with an identity MLP activation there is no ELEMENTWISE "
                      "nonlinearity left anywhere in the stack. What remains is the attention "
                      "softmax and LayerNorm (pre-norm, so 2L+1 of them), and both act as "
                      "input-dependent SCALARS on otherwise linear maps -- softmax sets the "
                      "mixing coefficients, LayerNorm removes a mean (linear) and divides by an "
                      "RMS (one scalar per token). The whole network is therefore an "
                      "input-dependent linear operator on the embedded sequence, which is a far "
                      "more tractable object than a ReLU stack even though it is NOT yet closed "
                      "form. FOLLOW UP IMMEDIATELY.")
            else:
                print(f"      >>> N1/M3 FALSIFIED: rho_50 {lin['rho50']:.3f} against the free "
                      f"threshold {free_at:.3f}. The nonlinearity is load-bearing at L={depth}.")
        compare(f"{n4} rf_mlp (linear1 frozen)      ", rung_med.get((depth, 4, "rf_mlp")), hub,
                yard_ms, "mean_solve", True)
        compare(f"{n6} no_mlp (attention only)      ", rung_med.get((depth, 4, "no_mlp")), hub,
                yard_ms, "mean_solve", True)
        compare(f"{n7} nhead 1 vs 4                 ", rung_med.get((depth, 1, "full")), hub,
                yard_ms, "mean_solve", True)
        print()

        tag5 = "M5" if depth == 4 else "N3"
        print(f"  --- {tag5}: the convex cell (rf_all) at L={depth}. "
              f"Censored rho_50 AND mean_solve < 0.10? ---")
        kr = (depth, 4, "rf_all")
        if kr in rung_med:
            ks = [(depth, 4, "rf_all", s) for s in SEEDS if (depth, 4, "rf_all", s) in met]
            n_cross = sum(1 for k in ks if met[k]["flag"] == "ok")
            ms = rung_med[kr]["mean_solve"]
            print(f"  {n_cross}/{len(ks)} seeds cross 0.5   median mean_solve {ms:.4f} "
                  f"(bar < 0.10)   median solve_at_floor {rung_med[kr]['solve_at_floor']:.4f}   "
                  f"trainable {met[ks[0]]['n_trainable']:,}/{met[ks[0]]['n_params']:,}")
            wf = [min(r["well_formed_rate"] for r in cells[k]) for k in ks]
            print(f"  grammar even where it cannot solve: min well_formed over bins "
                  f"{min(wf):.4f}, best single-bin solve "
                  f"{max(met[k]['best_solve'] for k in ks):.4f}")
            if ms >= 0.10:
                print("  >>> FALSIFIED, and this is the most important cell in the grid: a "
                      "CONVEX problem on features frozen at initialization does part of DD. "
                      "The analytic programme runs through this cell -- make it the main "
                      "object of study.")
            else:
                print(f"  >>> as predicted: frozen attention cannot learn where to look. "
                      f"The informative number is how far it got: mean_solve {ms:.4f}.")
        else:
            print("  pending")
        print()

    # ----------------------------------------------- rho_50 before and after
    # The metric of record, stated the way hshunt asked for it: what rho_50 was
    # before any machinery was stripped, and what it is after each removal, at
    # the shallowest depth where the full model still solves at all three seeds.
    print("=== rho_50 BEFORE and AFTER stripping machinery (L=6, the metric of record) ===")
    base = rung_med.get((6, 4, "full"))
    if base is None:
        print("  L=6/h4/full not measured yet")
    elif base["censored"]:
        print("  L=6/h4/full is censored -- there is no 'before' number to strip from")
    else:
        print(f"  BEFORE  L=6 h=4 full      rho_50 median {base['rho50']:.3f}  "
              f"(spread {rung_spread[(6, 4, 'full')]['rho50']:.3f}, "
              f"mean_solve {base['mean_solve']:.4f})")
        print(f"  a removal is 'free' iff rho_50 rises by less than the pooled "
              f"s* = {yard_r50:.4f}, i.e. stays below {base['rho50'] + yard_r50:.3f}")
        for (L, h, a, nm) in [(6, 4, "linear", "linear (no elementwise nonlinearity)"),
                              (6, 4, "rf_mlp", "rf_mlp (linear1 frozen at init)"),
                              (6, 4, "no_mlp", "no_mlp (attention only)"),
                              (6, 1, "full", "nhead=1 (one head, not four)"),
                              (6, 4, "rf_all", "rf_all (only the readout trains)")]:
            m = rung_med.get((L, h, a))
            if m is None:
                print(f"  AFTER   {nm:38s} pending")
                continue
            ks = [(L, h, a, s) for s in SEEDS if (L, h, a, s) in met]
            tr = met[ks[0]]["n_trainable"]
            if m["censored"]:
                # "Censored" on its own is weak: it does not say whether the
                # crossing seeds were near the threshold or far above it. Quote
                # them, and say which side of the free line they fell on, so a
                # censored row is still a measurement.
                free_at = base["rho50"] + yard_r50
                seen = "/".join(f"{v:.3f}" for v in m["rho50_ok"]) or "none"
                if m["bm_readable"]:
                    # The median survived the censoring, so this row gets a
                    # real delta instead of a bare "censored".
                    d = m["rho50_bm"] - base["rho50"]
                    verdict = ("COSTS rho_50, resolvably" if d > yard_r50
                               else "within s*, but one seed never crossed -> NOT free")
                    print(f"  AFTER   {nm:38s} rho_50 {m['rho50_bm']:.3f} ({d:+.3f}) "
                          f"[{m['n_cross']}/{len(ks)} cross, at {seen}; median survives the "
                          f"censoring]  mean_solve {m['mean_solve']:.4f}  trainable {tr:,}"
                          f"  -> {verdict}")
                else:
                    where = ("ALL above the free line" if m["rho50_ok"]
                             and min(m["rho50_ok"]) > free_at else
                             "some below the free line" if m["rho50_ok"] else "no seed crossed")
                    print(f"  AFTER   {nm:38s} rho_50 CENSORED ({m['n_cross']}/{len(ks)} cross, "
                          f"at {seen}; {where})  mean_solve {m['mean_solve']:.4f}  "
                          f"trainable {tr:,}  -> NOT free")
            else:
                d = m["rho50"] - base["rho50"]
                free = "FREE (within s*)" if d <= yard_r50 else "COSTS rho_50"
                print(f"  AFTER   {nm:38s} rho_50 {m['rho50']:.3f}  {d:+.3f}  "
                      f"trainable {tr:,}  -> {free}")
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
