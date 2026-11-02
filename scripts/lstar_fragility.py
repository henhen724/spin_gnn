"""How much noise would flip each L* verdict in this file?

L* is defined by a hard gate: the smallest num_layers whose solve_rate reaches
0.90 in EVERY depth bin at a fixed density. That is a threshold on a noisy
estimate, so two quantities decide whether an L* claim means anything:

  * the GATE MARGIN -- how far the binding bin sits from 0.90. A cell that
    clears by 0.003 (finding 31's L=7 at N=96) and a cell that clears by 0.05
    are not the same claim, and the file quotes them the same way.

  * the BIN SE -- sqrt(p(1-p)/n) on that binding bin. The deep bins are small
    (n=70 at T=8 in the N=48 leak-free ladder), so the gate decision there
    carries +-0.10 at 95% from sampling alone, before any seed enters.

Jobs 50691430/50691432 are measuring a third source, the seed spread at fixed
architecture. This script computes the first two from the CSVs already on
disk so that, when the spread lands, every L* statement in STATUS can be read
against all three at once. Nothing here depends on how those two jobs come
out -- written 2026-11-01 while they were at epoch 64 and 79 of 100.

Usage (module stack, not the dead venv):
    module load devel math viz python/3.12.1 py-numpy/1.26.3_py312 \
        py-pandas/2.2.1_py312
    python3 /scratch/users/hshunt/lstar_fragility.py
"""
import glob
import math
import os
import re
from statistics import NormalDist

import numpy as np
import pandas as pd

S = os.environ["SCRATCH"] + "/spin_gnn"
GATE = 0.90

# (label, csv glob). The N=48 leak-free ladder is the live one -- findings
# 33/38/40 and the blocked MLP-depth scan all rest on it. The N=96 ladder is
# leaked-order but is where findings 22/31/32 read their L*.
SOURCES = [
    ("N=48 leak-free ff=1024", [f"{S}/sudoku_n48_leakfree_lstar_2026-10-29/depth_lstar_*.csv",
                                f"{S}/sudoku_n48_leakfree_lstar_2026-10-31/depth_lstar_*.csv",
                                f"{S}/sudoku_n48_leakfree_L10_2026-10-31/depth_L10_*.csv"]),
    ("N=96 leaked-order",      [f"{S}/sudoku_n96_layer_scan_2026-10-26/depth_by_layers_n96.csv",
                                f"{S}/sudoku_n96_L07_2026-10-28/depth_n96_L07.csv",
                                f"{S}/sudoku_n96_ff256_2026-10-27/depth_by_layers_n96_ff256.csv",
                                f"{S}/sudoku_n96_warmup_2026-10-28/depth_n96_ff256_warmup.csv"]),
]


def tag_of(ckpt):
    """(num_layers, dim_feedforward, extra) from a checkpoint tag."""
    name = str(ckpt)
    L = int(re.search(r"_L(\d+)_", name).group(1))
    ff = int(re.search(r"_ff(\d+)_", name).group(1))
    extra = "".join(s for s in ("_w05", "_cos", "_shuf") if s in name)
    return L, ff, extra


def se(p, n):
    return math.sqrt(p * (1 - p) / n) if n else float("nan")


# stdlib, so the script keeps its numpy+pandas-only footprint (no py-scipy
# module needed to re-run it).
NORMAL_Q = NormalDist().inv_cdf


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    pd.set_option("display.width", 200)
    for label, globs in SOURCES:
        frames = []
        for g in globs:
            for p in sorted(glob.glob(g)):
                frames.append(pd.read_csv(p))
        if not frames:
            print(f"\n### {label}: no CSV found")
            continue
        df = pd.concat(frames, ignore_index=True)
        tags = df.checkpoint.map(tag_of)
        df["L"] = [t[0] for t in tags]
        df["ff"] = [t[1] for t in tags]
        df["extra"] = [t[2] for t in tags]
        df = df.drop_duplicates(subset=["checkpoint", "rho_requested", "T"])

        for rho in sorted(df.rho_requested.unique(), reverse=True):
            sub = df[df.rho_requested == rho]
            nb = sub["T"].nunique()
            rows = []
            for (L, ff, extra), r in sub.groupby(["L", "ff", "extra"]):
                if r["T"].nunique() < nb:          # an incomplete cell cannot be gated
                    continue
                binding = r.loc[r.solve_rate.idxmin()]
                clears = int((r.solve_rate >= GATE).sum())
                # The gate is an INTERSECTION hypothesis -- every bin at once --
                # so the bin that binds at 95% need not be the one that binds on
                # the point estimate: a bin can sit higher and still have the
                # lower bound, because n varies 3x across bins (n=70 at T=8 vs
                # n=213 at T=3). Take the worst lower bound over bins, and spend
                # alpha across them (Bonferroni) so "clears at 95%" is a
                # simultaneous statement rather than nb separate ones.
                z_sim = NORMAL_Q(1 - 0.05 / (2 * nb))
                los = [wilson(int(round(rr.solve_rate * rr["n"])), int(rr["n"]), z_sim)[0]
                       for _, rr in r.iterrows()]
                worst = r.iloc[int(np.argmin(los))]
                worst_lo = min(los)
                # the uniform shift that would flip this cell's verdict: for a
                # cell that clears, down to the gate; for one that misses, the
                # lift its worst bin needs
                margin = binding.solve_rate - GATE
                nb_n = int(binding["n"])
                rows.append(dict(
                    cell=f"L={L} ff={ff}{extra}", L=L, ff=ff, extra=extra,
                    bins=f"{clears}/{nb}", verdict="clears" if clears == nb else "misses",
                    binding_T=int(binding["T"]), n=nb_n,
                    binding_rate=binding.solve_rate,
                    flip_shift=abs(margin),
                    # how many INSTANCES in the binding bin would have to go the
                    # other way to flip the verdict -- the gate is a count, and
                    # at n=70 one instance is 0.014 of solve_rate
                    flip_inst=max(1, int(math.ceil(abs(margin) * nb_n))),
                    bin_se=se(binding.solve_rate, nb_n),
                    # the bin that binds AT 95%, which is not always the one that
                    # binds on the point estimate
                    lo_T=int(worst["T"]), lo_n=int(worst["n"]), w_lo=worst_lo,
                    # the gate is an ABSOLUTE threshold on one cell, so unlike
                    # every paired contrast in this project it gets no help from
                    # the shared instances: this is the honest 95% statement
                    gate_at_95="yes" if (clears == nb and worst_lo >= GATE) else "no",
                ))
            if not rows:
                continue
            t = pd.DataFrame(rows).sort_values(["ff", "L"], ascending=[True, False])
            t["flip_in_SE"] = t.flip_shift / t.bin_se
            print(f"\n### {label}, rho={rho:.2f}  ({nb} bins; gate {GATE:.2f})")
            print("    flip_shift = uniform shift in solve_rate that flips this cell's verdict")
            print("    bin_se     = binomial SE of the binding bin alone")
            print(t.drop(columns=["L", "ff", "extra"]).round(4).to_string(index=False))

            # One ladder per (ff, extra): a warmed-up cell is a different
            # training procedure, so its rungs must not be mixed into the plain
            # column when taking the min over clearing rungs.
            for (ff, extra), col in t.groupby(["ff", "extra"]):
                col = col.sort_values("L")
                name = f"ff={ff}{extra}"
                cle = col[col.verdict == "clears"]
                if cle.empty:
                    print(f"    {name}: L* UNDEFINED at the top of this column")
                    continue
                lstar = int(cle.L.min())
                row = cle[cle.L == lstar].iloc[0]
                below = col[col.L < lstar]
                msg = (f"    {name}: L*={lstar}, held by a {row.flip_shift:.4f} margin "
                       f"({row.flip_in_SE:.1f} binding-bin SE) at T={row.binding_T}, n={row.n}")
                if not below.empty:
                    b = below.iloc[-1]
                    msg += (f"; the rung below (L={int(b.L)}) "
                            f"would take it with +{b.flip_shift:.4f}")
                print(msg)
                # And the conservative ladder: the smallest rung that clears
                # every bin at a simultaneous 95% level. Where this differs from
                # L*, the point estimate is the only thing holding the claim up.
                safe = col[col.gate_at_95 == "yes"]
                if safe.empty:
                    print(f"    {name}: NO rung clears at a simultaneous 95% level "
                          f"(worst-bin lo tops out at {col.w_lo.max():.4f}) -- "
                          f"L*={lstar} is a point estimate only")
                elif int(safe.L.min()) != lstar:
                    print(f"    {name}: at a simultaneous 95% level the smallest "
                          f"clearing rung is L={int(safe.L.min())}, not {lstar}")


def design():
    """How big must a depth bin be for the gate verdict to mean anything?

    The audit above says the binding bins (n=43-70 at the deep end) are too
    small to support a 95% statement. The fix is not a different statistic, it
    is --min-per-bin: this prints the smallest n at which a cell whose TRUE
    solve rate is p would clear the 0.90 gate at a simultaneous 95% level.
    """
    print("\n### design: min n per bin to clear the 0.90 gate at simultaneous 95%")
    print("    (rows = the cell's true solve rate in that bin; columns = n bins gated)")
    hdr = "    true_rate " + "".join(f"{nb:>9d} bins" for nb in (3, 4, 6))
    print(hdr)
    for p in (0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 1.00):
        cells = []
        for nb in (3, 4, 6):
            z = NORMAL_Q(1 - 0.05 / (2 * nb))
            need = None
            for n in range(10, 4001):
                if wilson(int(round(p * n)), n, z)[0] >= GATE:
                    need = n
                    break
            cells.append(f"{need:>9d}     " if need else "    >4000     ")
        print(f"    {p:9.2f} " + "".join(cells))
    print("    current deep-end bins: n=70 (N=48 T=8), n=43 (N=96 T=9), n=57 (N=96 T=5)")


if __name__ == "__main__":
    main()
    design()
