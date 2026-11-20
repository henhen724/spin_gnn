"""Score the N=192 ladder rung (job 53062385) against N1-N5, registered in
`scripts/slurm/sudoku2_n192_ladder_2026-11-20.sbatch` and committed in 0ccce1e
BEFORE the cells were submitted and before this scorer existed.

WHY THIS FILE EXISTS NOW, WITH NO DATA TO READ. The standing rule is that the
scorer is registered in git before it has anything to score, because a scorer
written after the numbers land is a scorer tuned to them. Stage BOUND is the
best-scoring stage in this project (7 of 7) for exactly one reason: its
estimator was DERIVED from the writer's source before it was registered, not
named after what it was wanted to mean. Stage GATE is the worst (six VOID)
because its headline estimator was named and not read. So every quantity below
is defined against the columns `scripts/rho_transition_eval.py` actually
writes, and the three cross-N definitions are IMPORTED from the N=96 reader
rather than retyped.

THE CLAIM BEING SCORED, restated so this file stands alone. DD has two phases
(finding 1 + 34, Dario's Sudoku_SC slides): above rho_c ~ 0.46 the number of
deduction rounds T grows like log N, below it like N^gamma with gamma -> 0.73.
Finding 43's fall fix put the INSTANCE ensemble's floor at rho_min ~ 0.31,
below rho_c, so the instances span the transition. Finding 108 then found that
the MODELS have been moving the other way: at the identical cell, N=48 still
solves 0.7550 at rho=0.333 (below rho_c, inside the slow phase) while N=96 is
at 0.0075 by rho=0.448 and 0.0000 below, dying exactly AT rho_c. This rung asks
whether that is a trend or a pair of points.

THE NORMALISATION, AND WHY IT IS NOT OPTIONAL. Whole-instance solve rate cannot
be compared across N: a solve needs every blank cell right, so it is a product
of ~N(1-rho) per-deduction events and falls with N even at a FIXED per-step
competence. The per-deduction rate is

    a_eff = solve_rate ** (1 / n_blank),     n_blank = N - round(N * rho)

using the NOMINAL grid rho, because the generator plants exactly round(N*rho)
hints, so n_blank is an integer CONSTANT inside every density bin -- which is
finding 95's Q4/Q5 result and is the only reason this normalisation is well
defined rather than an average over a spread. D2 below re-checks it rather
than trusting it.

a_eff is compared at MATCHED DD DEPTH, never at matched rho: trap 4 of the
N=96 control is that the three rungs use different rho grids (floor 0.33 /
0.32 / 0.32) so the densities are not a common axis, while `dd_mean_T` is a
property of the instances that means the same thing at every N.

Run (any env -- stdlib only, no numpy, no torch, no matplotlib):
    python3 scripts/n192_ladder_fit.py [--scan DIR] [--out DIR]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import, never retype: rho_50 carries the censoring flags ('<floor' is
# finding 44's H1 failure and is NOT a number), rho50_ci is the binomial
# resample over 400-instance bins, history() knows that val_nll lives in
# history.json and not in the CSV. All three are the N=96 reader's, so the
# N=192 rung cannot drift from the rung it is paired against.
from n96_fallscan_analysis import history, load, rho_50, rho50_ci

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_192 = os.path.join(SCRATCH, "spin_gnn/sudoku_n192_ladder_2026-11-20")
SCAN_96 = os.path.join(SCRATCH, "spin_gnn/sudoku_n96_fall_scan_2026-11-02")
SCAN_48 = os.path.join(SCRATCH, "spin_gnn/sudoku_n48_fall_scan_2026-11-02")

# The paired cells, all the IDENTICAL architecture (full L=12 d256 ff1024,
# 240k x 50 ep, --fall --shuffle-adj). Trap 1 of the N=96 control: d256 /
# ff1024 / warmup0 are all defaults, so an N=192 cell reduces to the same
# rung_key (12, 4, 'full') as the N=48 and N=96 L=12 cells. n_nodes MUST be
# part of any key, and these directories stay OUT of SCAN_DIRS.
PAIRED = (
    (48, 0, os.path.join(SCAN_48, "rho_curve_t08_*.csv")),
    (96, 0, os.path.join(SCAN_96, "rho_curve_t00_*.csv")),
    (96, 1, os.path.join(SCAN_96, "rho_curve_t04_*.csv")),
)

RHO_C = 0.46           # q=4, d=5 (fastslow.json)
N1_BAR, N1_FALSIFIER = 0.9118, 0.9626   # N=96 seed 0 and seed 1 at T ~ 12.0
N1_T = 12.0
N3_BAR = 0.119         # s*_96, the published N=96 seed spread
N5_BAR, N5_SOLVE_FLOOR = 0.01, 0.1


def a_eff(row):
    """Per-deduction success rate. None where it is undefined or uninformative.

    solve_rate == 0 gives a_eff == 0 for every n_blank, which carries no
    information about per-step competence (it only says the product
    underflowed), so it is returned as None rather than plotted as a floor.
    """
    n_blank = row["n_nodes"] - round(row["n_nodes"] * row["rho"])
    if n_blank <= 0 or row["solve_rate"] <= 0.0:
        return None
    return row["solve_rate"] ** (1.0 / n_blank), n_blank


def curve(rows):
    """[(dd_mean_T, a_eff, n_blank, rho, solve_rate)] sorted by depth."""
    out = []
    for r in rows:
        v = a_eff(r)
        if v is None:
            continue
        out.append((r["dd_mean_T"], v[0], v[1], r["rho"], r["solve_rate"]))
    return sorted(out)


def at_depth(pts, t):
    """Linear interpolation of a_eff onto DD depth t, or None if outside.

    Depth is the cross-N axis because the rho grids differ by size (trap 4).
    Extrapolation is refused: a bar read off an extrapolated line is a
    property of the fit and not a measurement (finding 95).
    """
    xs = [(p[0], p[1]) for p in pts]
    if len(xs) < 2 or t < xs[0][0] or t > xs[-1][0]:
        return None
    for (t0, a0), (t1, a1) in zip(xs, xs[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return a0
            return a0 + (t - t0) * (a1 - a0) / (t1 - t0)
    return None


def cell_report(n_nodes, seed, pattern):
    # `load` casts only the columns the N=96 reader needed; n_nodes is not one
    # of them, so it arrives as the STRING '48'. Every CSV cell is a string
    # here and booleans arrive as 'True' -- a standing gotcha in this project,
    # and the reason this coercion is explicit rather than assumed.
    rows = []
    for r in load(pattern):
        r["n_nodes"] = int(r["n_nodes"])
        if r["n_nodes"] == n_nodes:
            rows.append(r)
    if not rows:
        return None
    pts = curve(rows)
    r50, flag = rho_50([(r["realized_rho"], r["solve_rate"]) for r in rows])
    lo, hi, dropped = (float("nan"), float("nan"), 0)
    try:
        ci = rho50_ci(rows)
        if isinstance(ci, (tuple, list)) and len(ci) >= 2:
            lo, hi = float(ci[0]), float(ci[1])
            dropped = int(ci[2]) if len(ci) > 2 else 0
    except Exception as exc:                       # noqa: BLE001
        print(f"  (rho50_ci unavailable for N={n_nodes} s{seed}: {exc})")
    path = rows[0]["path"]
    fin, best, fin_tr, n_ep = history(path)
    return {
        "n_nodes": n_nodes, "seed": seed, "n_bins": len(rows),
        "n_params": rows[0]["n_params"],
        "rho_50": r50, "rho_50_flag": flag, "rho_50_lo": lo, "rho_50_hi": hi,
        "rho_50_dropped": dropped,
        "val_nll_final": fin, "val_nll_best": best, "n_epochs": n_ep,
        "a_at_N1_T": at_depth(pts, N1_T),
        "deepest_T": pts[-1][0] if pts else None,
        "a_at_deepest": pts[-1][1] if pts else None,
        "rho_at_deepest": pts[-1][3] if pts else None,
        "solve_at_deepest": pts[-1][4] if pts else None,
        "max_trunc_live": max(
            [r["truncated_rate"] for r in rows
             if r["solve_rate"] > N5_SOLVE_FLOOR] or [0.0]),
        "n_live_bins": sum(1 for r in rows if r["solve_rate"] > N5_SOLVE_FLOOR),
        "curve": pts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=SCAN_192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = []
    for n, s, pat in PAIRED:
        c = cell_report(n, s, pat)
        if c:
            cells.append(c)
    for s in (0, 1):
        pat = os.path.join(args.scan, f"rho_curve_t0{s}_*.csv")
        c = cell_report(192, s, pat)
        if c:
            cells.append(c)
        else:
            print(f"N=192 seed {s}: no rho_curve CSV yet at {pat}")

    n192 = [c for c in cells if c["n_nodes"] == 192]
    print(f"\n{'N':>5} {'seed':>5} {'bins':>5} {'rho_50':>9} {'flag':>7} "
          f"{'a@T=12':>8} {'deepest T':>10} {'a@deepest':>10} {'rho@deep':>9}")
    for c in cells:
        def f(x, w=9, p=4):
            return f"{x:>{w}.{p}f}" if isinstance(x, float) else f"{'-':>{w}}"
        print(f"{c['n_nodes']:>5} {c['seed']:>5} {c['n_bins']:>5} "
              f"{f(c['rho_50'])} {c['rho_50_flag']:>7} "
              f"{f(c['a_at_N1_T'], 8)} {f(c['deepest_T'], 10, 2)} "
              f"{f(c['a_at_deepest'], 10)} {f(c['rho_at_deepest'], 9, 3)}")

    lines = []

    # ---- D2, run FIRST because N1-N3 are meaningless if it fails. --------
    bad = []
    for c in cells:
        for t, a, nb, rho, _ in c["curve"]:
            if nb != c["n_nodes"] - round(c["n_nodes"] * rho):
                bad.append((c["n_nodes"], c["seed"], rho, nb))
    lines.append(
        f"D2 n_blank constant inside every bin: {'PASS' if not bad else 'FAIL'} "
        f"({len(bad)} violations) -- finding 95's Q4/Q5, the precondition for "
        f"a_eff being well defined rather than an average over a spread.")

    # ---- D1, the cross-N reproduction. ----------------------------------
    rep = {(c["n_nodes"], c["seed"]): c for c in cells}
    d1 = []
    for (n, s), exp_solve, exp_rho in (((48, 0), 0.7550, 0.333),
                                       ((96, 0), 0.0075, 0.448)):
        c = rep.get((n, s))
        if not c:
            continue
        got = c["solve_at_deepest"] if n == 48 else None
        d1.append(f"N={n} s{s} deepest live bin rho={c['rho_at_deepest']:.3f} "
                  f"solve={c['solve_at_deepest']:.4f} "
                  f"(finding 108 recorded rho={exp_rho:.3f} "
                  f"solve={exp_solve:.4f})")
    lines.append("D1 finding 108 reproduction from the same CSVs: "
                 + "; ".join(d1) if d1 else "D1 unavailable (no paired cells)")

    # ---- N1 -------------------------------------------------------------
    a192 = [c["a_at_N1_T"] for c in n192 if c["a_at_N1_T"] is not None]
    if not a192:
        lines.append(f"N1 UNSCOREABLE: no N=192 cell spans T={N1_T}; a_eff is "
                     f"not extrapolated (finding 95).")
    else:
        m = max(a192)
        verdict = ("HIT" if m < N1_BAR else
                   "FALSIFIED" if m >= N1_FALSIFIER else "MISS")
        lines.append(
            f"N1 {verdict}: a_eff at T={N1_T} is "
            + "/".join(f"{a:.4f}" for a in a192)
            + f" against the N=96 pair {N1_BAR:.4f}/{N1_FALSIFIER:.4f}. "
            f"Below {N1_BAR} = the N-penalty at matched depth continues; "
            f">= {N1_FALSIFIER} = it does not and 108 was a two-point trend.")

    # ---- N2 -------------------------------------------------------------
    if n192:
        bits, ok, fals = [], True, False
        for c in n192:
            bits.append(f"s{c['seed']} {c['rho_50']:.4f} [{c['rho_50_flag']}]")
            if c["rho_50_flag"] == "<floor":
                ok = False          # censored below the floor is NOT > rho_c
            elif c["rho_50"] <= RHO_C:
                fals = True
        lines.append(
            f"N2 {'FALSIFIED' if fals else 'HIT' if ok else 'UNSCOREABLE'}: "
            f"rho_50 = " + ", ".join(bits) + f" against rho_c = {RHO_C}. "
            f"A '<floor' flag is censoring, not a value (finding 44's H1), and "
            f"is reported as unscoreable rather than as a pass.")

    # ---- N3 -------------------------------------------------------------
    r192 = [c["rho_50"] for c in n192 if c["rho_50_flag"] == "ok"]
    r96 = [c["rho_50"] for c in cells
           if c["n_nodes"] == 96 and c["rho_50_flag"] == "ok"]
    if len(r192) == 2:
        s192 = max(r192) - min(r192)
        note = ""
        if len(r96) == 2:
            s96 = max(r96) - min(r96)
            note = (f" DISCLOSURE: the same recipe on the N=96 pair gives "
                    f"{s96:.4f} against the published s*_96 = {N3_BAR}; if "
                    f"these differ the published value uses a different "
                    f"definition and N3's bar is NOT this statistic.")
        lines.append(
            f"N3 {'HIT' if s192 >= N3_BAR else 'MISS'}: seed spread at N=192 "
            f"is {s192:.4f} against >= {N3_BAR}." + note)
    else:
        lines.append("N3 UNSCOREABLE: needs two uncensored N=192 seeds, have "
                     f"{len(r192)}.")

    # ---- N4 -------------------------------------------------------------
    for c in n192:
        lines.append(
            f"N4 N=192 s{c['seed']}: {c['n_epochs']} epochs, final val_nll "
            f"{c['val_nll_final']:.4f}, best {c['val_nll_best']:.4f}, "
            f"{'still descending (final == best)' if c['val_nll_final'] <= c['val_nll_best'] + 1e-9 else 'final is ABOVE best -- overfit gap'}"
            f". NO escape count is registered here: the 1.64 plateau is an "
            f"N=48 number (trap 3) and N=192's plateau is unmeasured. Flat for "
            f"50 epochs = not trainable at this budget, reported not retried.")

    # ---- N5 -------------------------------------------------------------
    for c in n192:
        lines.append(
            f"N5 N=192 s{c['seed']} "
            f"{'HIT' if c['max_trunc_live'] <= N5_BAR else 'MISS'}: max "
            f"truncated_rate over the {c['n_live_bins']} bins with solve > "
            f"{N5_SOLVE_FLOOR} is {c['max_trunc_live']:.4f} (bar <= {N5_BAR}). "
            f"A truncation is max_len biting, which would make a low solve "
            f"rate an encoding artefact rather than a competence result.")

    print("\n=== VERDICT ===")
    for ln in lines:
        print("- " + ln)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "VERDICT.txt"), "w") as fh:
            fh.write("\n".join("- " + ln for ln in lines) + "\n")
        with open(os.path.join(args.out, "ladder_cells.json"), "w") as fh:
            json.dump([{k: v for k, v in c.items() if k != "curve"}
                       for c in cells], fh, indent=1)
        print(f"\nwrote {args.out}/VERDICT.txt and ladder_cells.json")


if __name__ == "__main__":
    main()
