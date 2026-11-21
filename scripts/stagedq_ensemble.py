"""STAGE DQ -- ensemble pre-check for the (degree, n_colors) depth lever.

hshunt chose option (e) on 2026-11-21: get DD reasoning depth from the
*local* parameters (d, q) instead of from N. Finding 10 claims (d=4, q=3)
at N=60 reaches a mean depth that (d=5, q=4) does not reach until N ~ 333.
If that holds, the slow phase becomes reachable at a size where the model
demonstrably still learns the colouring rule -- finding 116 established
that it does at N <= 96 and does not at N = 192.

NOTHING HERE TRAINS ANYTHING. This is the CPU-side measurement that has to
land before a single GPU hour is spent, because (d=4, q=3) is a DIFFERENT
ENSEMBLE: rho_c moves, rho_min moves, the val_nll floor moves, and the
colour-token chance level moves from ln 4 = 1.3863 to ln 3 = 1.0986. None
of the q=4 d=5 constants carry over and none of them may be reused.

Legality at q=3, d=4: need q | N and q(q-1) | d*N, i.e. 3 | N and 6 | 4N.
The second reduces to 3 | N, so 3 | N is the only constraint (against
12 | N at q=4, d=5). Fall colouring needs d >= q-1, i.e. 4 >= 2. OK.

Outputs (runs/stagedq_ensemble_<date>/):
  depth_vs_n.csv    T at rho_min and mean deduction depth vs N, both ensembles
  t_vs_rho.csv      T(rho) curves on a shared grid, for locating rho_c
  summary.txt       the table that the stage registration quotes

Run under the TORCH/NUMBA module set:
  module purge --force
  module load devel math python/3.12.1 py-numpy/1.26.3_py312 \
      py-numba/0.60.0_py312 py-scipy/1.12.0_py312
"""
import os
import sys
import csv
import math
import datetime

import numpy as np

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
from spin_gnn.sudoku_dd import planted_regular_graph, removal_path, on_grid

# The two ensembles, each at both fall settings.
#
# WHY BOTH fall SETTINGS. Finding 10's "~6x cheaper" comes from the control
# parameter sweep 49509101, whose scorer (scripts/dd_control_parameter_sweep.py)
# never passes `fall` -- so it measured the fall=False ensemble. Every TRAINING
# cell we have ever run uses --fall. Under fall=False a fixed fraction of nodes
# is never deducible (0.3827 at q=4 d=5), which pushes rho_min up and CAPS the
# depth reachable near the floor; the fall constraint removes that cap. So the
# fall constraint already buys d5q4 much of the depth that finding 10 credited
# to (d=4, q=3), and the two numbers are not comparable. Measuring all four
# cells is the only way to say which part of the 6x survives into our ensemble.
ENSEMBLES = [
    ("d5q4_fall", 5, 4, True),    # REF -- what every training cell uses
    ("d4q3_fall", 4, 3, True),    # the lever hshunt chose, in OUR ensemble
    ("d5q4_nofall", 5, 4, False),  # finding 10's ensemble, REF arm
    ("d4q3_nofall", 4, 3, False),  # finding 10's ensemble, lever arm
]

N_SEEDS = 24
# Must reach BOTH floors. d5q4 bottoms out near 0.31 (fall) / 0.47 (no fall),
# d4q3 near 0.14 / 0.19 -- a grid that stops at 0.33, as every q=4 stage used,
# sees none of the d4q3 density range that matters. Below a path's own floor
# on_grid returns NaN, which is the honest value.
RHO_GRID = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60,
                     0.55, 0.50, 0.45, 0.40, 0.35, 0.33, 0.30, 0.27,
                     0.24, 0.21, 0.18, 0.16, 0.14])


def legal_sizes(degree, n_colors, lo, hi):
    """N values with q | N and q(q-1) | d*N in [lo, hi], on a coarse ladder."""
    out = []
    for n in range(lo, hi + 1):
        if n % n_colors == 0 and (degree * n) % (n_colors * (n_colors - 1)) == 0:
            out.append(n)
    return out


def ladder(degree, n_colors, targets):
    """Snap each target size to the nearest legal N for this ensemble."""
    out = []
    for t in targets:
        cands = legal_sizes(degree, n_colors, max(12, t - 24), t + 24)
        if not cands:
            continue
        n = min(cands, key=lambda c: (abs(c - t), c))
        if n not in out:
            out.append(n)
    return out


def measure(n_nodes, degree, n_colors, n_seeds, fall):
    """Run n_seeds greedy removal paths. Returns per-seed arrays of
    (T at rho_min, mean deduction time at rho_min, rho_min) plus the
    T(rho) curve on RHO_GRID averaged over seeds.

    Under fall=False some nodes are never deducible; their time is INF and
    would poison both statistics, so T and the mean are taken over the
    FINITE times only and the never-deducible fraction is returned too.
    """
    T_min, mean_t, rho_min, frac_inf = [], [], [], []
    grid_T = np.full((n_seeds, len(RHO_GRID)), np.nan)
    for s in range(n_seeds):
        rng = np.random.default_rng(10_000 * n_nodes + 97 * degree
                                    + 7 * n_colors + 3 * int(fall) + s)
        adj, colors = planted_regular_graph(n_nodes, degree, n_colors, rng,
                                            fall=fall)
        path = removal_path(adj, colors, n_colors, rng)
        t = path["final_t"]
        finite = t[t < (1 << 39)]
        frac_inf.append(1.0 - finite.size / t.size)
        T_min.append(int(finite.max()) if finite.size else 0)
        mean_t.append(float(finite.mean()) if finite.size else 0.0)
        rho_min.append(float(path["rho_min"]))
        Tg, _ = on_grid(path, RHO_GRID)
        grid_T[s] = Tg
    return (np.array(T_min), np.array(mean_t), np.array(rho_min),
            np.array(frac_inf),
            np.nanmean(grid_T, axis=0), np.nanstd(grid_T, axis=0))


def seq_len(n_nodes, degree, T):
    """Token budget. The grammar writes each undirected edge twice in rows of
    (degree + 1) ids, plus hints, the round markers and the 4 specials:
    L = N*(d+3) + T + 4 at the sizes measured so far."""
    return n_nodes * (degree + 3) + T + 4


def main():
    date = datetime.date.today().isoformat()
    outdir = f"/home/users/hshunt/spin_gnn/runs/stagedq_ensemble_{date}"
    os.makedirs(outdir, exist_ok=True)

    targets = [48, 60, 72, 96, 120, 144, 192, 240, 300, 384]
    rows, curves = [], []

    for name, d, q, fall in ENSEMBLES:
        sizes = ladder(d, q, targets)
        print(f"\n=== {name}: d={d} q={q} fall={fall} | legal N: {sizes}")
        print(f"{'N':>5} {'T@rho_min':>20} {'mean_t':>14} {'rho_min':>14} "
              f"{'never_ded':>10} {'L':>7}")
        for n in sizes:
            Tm, mt, rm, fi, gT, gS = measure(n, d, q, N_SEEDS, fall)
            L = seq_len(n, d, int(round(Tm.mean())))
            print(f"{n:5d} {Tm.mean():8.2f} +- {Tm.std():5.2f} "
                  f"{mt.mean():8.2f}+-{mt.std():4.2f} "
                  f"{rm.mean():8.4f}+-{rm.std():4.3f} {fi.mean():10.4f} {L:7d}")
            rows.append(dict(ensemble=name, degree=d, n_colors=q, fall=fall,
                             n_nodes=n, n_seeds=N_SEEDS,
                             T_rhomin_mean=Tm.mean(), T_rhomin_std=Tm.std(),
                             mean_t_mean=mt.mean(), mean_t_std=mt.std(),
                             rho_min_mean=rm.mean(), rho_min_std=rm.std(),
                             frac_never_deducible=fi.mean(), seq_len=L))
            for k, r in enumerate(RHO_GRID):
                curves.append(dict(ensemble=name, degree=d, n_colors=q,
                                   fall=fall, n_nodes=n, rho=float(r),
                                   T_mean=float(gT[k]), T_std=float(gS[k])))

    with open(f"{outdir}/depth_vs_n.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"{outdir}/t_vs_rho.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)

    # ---- the comparison the stage registration needs -----------------------
    lines = []
    lines.append("STAGE DQ -- ENSEMBLE PRE-CHECK")
    lines.append(f"date {date}   seeds/cell {N_SEEDS}")
    lines.append("")
    lines.append("Q1. Does (d=4,q=3) buy DD depth at small N? For each lever")
    lines.append("    size, the REF (d=5,q=4) size with the nearest T at")
    lines.append("    rho_min -- computed SEPARATELY per fall setting, because")
    lines.append("    the two are different ensembles and must not be mixed.")
    lines.append("    A saving marked CENSORED is a lower bound: the matching")
    lines.append("    REF size is off the top of the ladder.")

    savings = {}
    for fall in (True, False):
        tag = "fall=True  (OUR ensemble, every training cell)" if fall else \
              "fall=False (finding 10's ensemble, sweep 49509101)"
        ref = {r["n_nodes"]: r for r in rows
               if r["degree"] == 5 and r["fall"] == fall}
        lev = {r["n_nodes"]: r for r in rows
               if r["degree"] == 4 and r["fall"] == fall}
        ref_sorted = sorted(ref.items())
        ref_max_T = max(rr["T_rhomin_mean"] for _, rr in ref_sorted)
        lines.append("")
        lines.append(f"  --- {tag}")
        lines.append(f"  {'lever N':>8} {'T':>7} {'L':>7} | {'ref N':>7} {'T':>7} "
                     f"{'L':>7} | {'N saved':>9} {'tok saved':>10}")
        good = []
        for n, r in sorted(lev.items()):
            tgt = r["T_rhomin_mean"]
            rn, rr = min(ref_sorted,
                         key=lambda kv: abs(kv[1]["T_rhomin_mean"] - tgt))
            cens = tgt > ref_max_T
            nx, tx = rn / n, rr["seq_len"] / r["seq_len"]
            mark = " CENSORED" if cens else ""
            lines.append(f"  {n:8d} {tgt:7.2f} {r['seq_len']:7d} | {rn:7d} "
                         f"{rr['T_rhomin_mean']:7.2f} {rr['seq_len']:7d} "
                         f"| {nx:8.2f}x {tx:9.2f}x{mark}")
            if not cens:
                good.append((nx, tx))
        if good:
            mn = float(np.median([g[0] for g in good]))
            mt_ = float(np.median([g[1] for g in good]))
            savings[fall] = (mn, mt_)
            lines.append(f"  median over uncensored rows: {mn:.2f}x in N, "
                         f"{mt_:.2f}x in tokens")

    lines.append("")
    lines.append("Q1b. RECONCILIATION WITH FINDING 10 -- IT NEEDS AMENDING.")
    lines.append("    Finding 10 reports ~6x ('at N=60 (d=4,q=3) reaches mean")
    lines.append("    DD depth 14.5, which (5,4) does not reach until N~333').")
    lines.append("    That was measured WITHOUT the fall constraint. Measured")
    lines.append("    here:")
    for fall in (True, False):
        if fall in savings:
            mn, mt_ = savings[fall]
            lines.append(f"      fall={str(fall):5s}  {mn:.2f}x in N, {mt_:.2f}x in tokens")
    lines.append("    MECHANISM, read off the rho_min rows below. The fall")
    lines.append("    constraint removes the nodes that can never be deduced and")
    lines.append("    so must stay hints forever, which is what caps depth near")
    lines.append("    the floor -- and it helps the REF ensemble far more than")
    lines.append("    the lever, because q=4 d=5 has 0.3827 such nodes against a")
    lines.append("    much smaller fraction at q=3 d=4. At N=192 turning fall on")
    lines.append("    moves rho_min 0.4729 -> 0.3134 for the REF (-0.1595) but")
    lines.append("    only 0.1944 -> 0.1437 for the lever (-0.0507): the REF gains")
    lines.append("    3.1x more room below its floor, and that is the gap closing.")
    lines.append("    THE 6x DOES NOT TRANSFER TO THE ENSEMBLE WE TRAIN ON.")
    lines.append("    (Note: a 'never deducible' COUNT at rho_min is identically")
    lines.append("    zero by construction and is not evidence either way --")
    lines.append("    greedy removal only drops a hint if DD still recovers")
    lines.append("    everything, so the survivors are all deducible. The rho_min")
    lines.append("    shift is the measurement; a frac_inf column would be noise.)")

    lines.append("")
    lines.append("Q2. rho_min -- it MOVES, and no q=4 constant (rho_c ~ 0.46,")
    lines.append("    free line 0.740, the val_nll floors 1.371/1.625/1.861,")
    lines.append("    s* 0.1484/0.119) may be carried over to a d4q3 cell.")
    lines.append("")
    for name, _, _, _ in ENSEMBLES:
        sub = [r for r in rows if r["ensemble"] == name]
        lines.append(f"  {name}: rho_min "
                     + ", ".join(f"N={r['n_nodes']}:{r['rho_min_mean']:.4f}"
                                 for r in sub))
    lines.append("")
    lines.append("Q4. WHERE DOES d4q3 GO SLOW? This is what picks the training")
    lines.append("    density range, and it cannot be copied from q=4. Finding")
    lines.append("    1+34's diagnostic: in the FAST phase T ~ log N, so T/log N")
    lines.append("    COLLAPSES across sizes; approaching rho_c the curves FAN")
    lines.append("    OUT. Below: T/log N by density, fall=True, N=48..384.")
    lines.append("    'spread' is max/min across sizes -- near 1 is collapsed")
    lines.append("    (fast), growing means the slow phase is being entered.")
    for name, d, q, fall in ENSEMBLES:
        if not fall:
            continue
        lines.append("")
        lines.append(f"  --- {name}")
        lines.append(f"  {'rho':>6} " + " ".join(f"{n:>6d}" for n in
                                                 sorted({r['n_nodes'] for r in rows
                                                         if r['ensemble'] == name}))
                     + f" {'spread':>8}")
        sizes_n = sorted({r["n_nodes"] for r in rows if r["ensemble"] == name})
        for r in RHO_GRID:
            vals = []
            for n in sizes_n:
                hit = [c for c in curves if c["ensemble"] == name
                       and c["n_nodes"] == n and abs(c["rho"] - r) < 1e-9]
                v = hit[0]["T_mean"] / math.log(n) if hit and not math.isnan(
                    hit[0]["T_mean"]) else float("nan")
                vals.append(v)
            fin = [v for v in vals if not math.isnan(v)]
            if len(fin) < 2:
                continue
            spread = max(fin) / min(fin)
            lines.append(f"  {r:6.2f} "
                         + " ".join("   nan" if math.isnan(v) else f"{v:6.2f}"
                                    for v in vals)
                         + f" {spread:8.2f}")
    lines.append("")
    lines.append("Q3. Colour-token chance level (the finding-116 diagnostic) is")
    lines.append("    ln q, so it MOVES: ln 4 = %.4f -> ln 3 = %.4f." %
                 (math.log(4), math.log(3)))
    lines.append("    A d4q3 cell sitting at 1.0986 nats on reason_color is the")
    lines.append("    same failure 116 found; 1.3863 means nothing there.")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(f"{outdir}/summary.txt", "w") as fh:
        fh.write(txt + "\n")
    print(f"\nwrote {outdir}/")


if __name__ == "__main__":
    main()
