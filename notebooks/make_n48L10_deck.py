"""Slide deck for the N=48 leak-free L=10 cell (job 50607549).

    module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 \
        py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 \
        py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
    cd ~/spin_gnn && python3 -m notebooks.make_n48L10_deck

Eight landscape pages to $SCRATCH/spin_gnn/notebooks/sudoku_n48_L10_2026-11-01/,
PDF plus one PNG per page (nothing on this cluster rasterizes a PDF, so the PNGs
are the only way to read the deck back before sending it).

The result is negative: the midpoint cell that was supposed to halve the (8, 12]
bracket landed BELOW L=8, so the deck argues a methodological point rather than
a new best number.
"""
import glob
import json
import math
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.expanduser("~/spin_gnn"))
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import (BIG, MED, SMALL, PdfAndPng, footer, set_style,
                                  textpage)

S = os.environ["SCRATCH"] + "/spin_gnn"
OUT = f"{S}/notebooks/sudoku_n48_L10_2026-11-01"
DIRS = [f"{S}/sudoku_n48_leakfree_lstar_2026-10-29",   # 50099945: L=6, 4, 3
        f"{S}/sudoku_n48_leakfree_lstar_2026-10-31",   # 50527633: L=12, 50527634: L=8
        f"{S}/sudoku_n48_leakfree_L10_2026-10-31"]     # 50607549: L=10, the new cell
DEPTH_GLOBS = ["depth_lstar_*.csv", "depth_L10_*.csv"]
CELLDIRS = {12: f"{DIRS[1]}/L12", 10: f"{DIRS[2]}/L10",
             8: f"{DIRS[1]}/L08",  6: f"{DIRS[0]}/L06"}
GATE, LADDER = 0.90, [12, 10, 8, 6, 4, 3]
NEW, REF = 10, 8          # the new cell and the rung it was supposed to beat
GREEN, RED, AMBER, GREY = "#276749", "#9b2c2c", "#975a16", "0.35"
COL = {12: "#2b6cb0", 10: "#c53030", 8: "#2f855a", 6: "#975a16",
        4: "#4a5568", 3: "#718096"}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ca_trend(ks, ns, xs):
    ks, ns, xs = np.asarray(ks, float), np.asarray(ns, float), np.asarray(xs, float)
    N, K = ns.sum(), ks.sum()
    pbar, xbar = K / N, (ns * xs).sum() / N
    var = pbar * (1 - pbar) * (ns * (xs - xbar) ** 2).sum()
    return (ks * (xs - xbar)).sum() / np.sqrt(var) if var > 0 else float("nan")


def paired(D, a, b):
    """delta, a-wins, b-wins, McNemar p for two cells on the same 1500 instances."""
    x, y = D["solved"][a], D["solved"][b]
    won, lost = int((x & ~y).sum()), int((~x & y).sum())
    return x.mean() - y.mean(), won, lost, mcnemar_exact(won, lost)


def load():
    lab = lambda c: int(re.search(r"_L(\d+)_", c).group(1))
    solved, seen = {}, {}
    for d in DIRS:
        for p in sorted(glob.glob(f"{d}/sharedpool_n1500_*_per_instance.npz")):
            job = re.search(r"_(\d+)_per_instance", p).group(1)
            z = np.load(p, allow_pickle=True)
            for i, name in enumerate(z["names"]):
                k, v = lab(str(name)), np.asarray(z["solved"][i], bool)
                if k in solved:
                    assert (solved[k] == v).all(), f"L={k} disagrees, {job} vs {seen[k]}"
                else:
                    solved[k], seen[k] = v, job
    rows = []
    for L in LADDER:
        v = solved[L]
        lo, hi = wilson(int(v.sum()), v.size)
        rows.append(dict(L=L, n=v.size, rate=v.mean(), lo=lo, hi=hi))
    pool = pd.DataFrame(rows)

    bins, grams = [], []
    for d in DIRS:
        for pat in DEPTH_GLOBS:
            bins += [pd.read_csv(p) for p in sorted(glob.glob(f"{d}/{pat}"))]
        grams += [pd.read_csv(p) for p in sorted(glob.glob(f"{d}/sharedpool_n1500_*.csv"))]
    bins = pd.concat(bins, ignore_index=True)
    bins["L"] = bins.checkpoint.map(lab)
    bins = bins.drop_duplicates(subset=["checkpoint", "rho_requested", "T"])
    gram = pd.concat(grams, ignore_index=True)
    gram["L"] = gram.checkpoint.map(lab)

    train = {}
    for L, cd in CELLDIRS.items():
        h = json.load(open(f"{cd}/d0256_h04_L{L:02d}_ff01024_lr1e-03_pre_shuf/history.json"))
        train[L] = dict(val_nll=h[-1]["nll"], acc=h[-1]["accuracy"],
                        npar=int(pd.read_csv(f"{cd}/sweep.csv").n_params.iloc[0]))
    return dict(pool=pool, solved=solved, bins=bins, gram=gram, train=train)


def clears(bins, rho):
    sub = bins[bins.rho_requested == rho]
    nb = sub["T"].nunique()
    return {int(L): int(((sub.L == L) & (sub.solve_rate >= GATE)).sum())
            for L in sorted(sub.L.unique(), reverse=True)}, nb


def hard_rho(bins):
    """The density with the most depth bins -- the one L* is read on."""
    return bins.groupby("rho_requested")["T"].nunique().idxmax()


def page_verdict(o, D):
    pool = D["pool"].set_index("L")
    d, won, lost, p = paired(D, NEW, REF)
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The midpoint cell that was supposed to halve the bracket",
             ha="center", size=BIG + 2, weight="bold")
    fig.text(0.5, 0.893,
             "Finding 38 left L* in (8, 12] and said L=10 was the cheapest way to narrow it. "
             "It did not narrow.",
             ha="center", va="top", size=MED, color="0.3")
    fig.text(0.5, 0.815,
             f"L=10 came back at {pool.rate[NEW]:.4f} -- BELOW L=8's {pool.rate[REF]:.4f}.",
             ha="center", size=BIG + 1, weight="bold", color=RED)
    fig.text(0.5, 0.762,
             f"paired on the same 1500 instances: {d:+.4f}, discordant {won}-{lost}, "
             f"McNemar exact p = {p:.2g}",
             ha="center", size=MED, color=RED)

    ax = fig.add_axes([0.07, 0.15, 0.40, 0.50])
    t = D["pool"].sort_values("L")
    ax.errorbar(t.L, t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], fmt="o-", lw=2.5,
                ms=10, capsize=5, color="#2b6cb0", rasterized=True)
    ax.plot([NEW], [pool.rate[NEW]], "o", ms=22, mfc="none", mec=RED, mew=3.5,
            rasterized=True)
    ax.annotate("the new cell", xy=(NEW, pool.rate[NEW]), xytext=(5.4, 0.47),
                size=SMALL + 1, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.axhline(GATE, color="k", ls="--", lw=2)
    ax.text(3.0, GATE + 0.03, "0.90 gate", size=SMALL)
    ax.set_xticks(sorted(t.L))
    ax.set_xlabel("num_layers")
    ax.set_ylabel("shared-pool solve rate (n=1500)")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("the ladder is NOT monotone in depth", size=MED, color=RED)

    said = ("WHAT WAS PRE-REGISTERED\n"
            "  pool in [0.93, 0.97]\n"
            "  clears all 6 bins at rho=0.60\n"
            "  L* narrows to (8, 10]\n\n"
            "WHAT HAPPENED\n"
            f"  pool {pool.rate[NEW]:.4f}\n"
            "  clears 2 of 6 bins\n"
            "  L* still 12; bracket still (8, 12]\n\n"
            "WHAT IT ACTUALLY SHOWS\n"
            "  not 'depth 10 is bad' --\n"
            "  'you cannot read a knee off a\n"
            "   single-seed ladder', because\n"
            "   this one is not monotone and\n"
            "   no rung has a seed replicate.")
    fig.text(0.56, 0.655, said, size=MED - 3, va="top", family="monospace",
             linespacing=1.55)
    footer(fig, f"Job 50607549, commit 44f581f, seed 0, {D['train'][NEW]['npar']/1e6:.2f} M params, "
                f"8 h 59 m on one GPU. num_layers 10 is the only difference from L=12 and L=8, checked "
                f"against each checkpoint's own config.json. L=6/4/3 are finding 33's cells, re-scored "
                f"inside this job and identical instance for instance.")
    o.savefig(fig)


def page_method(o, D):
    rho = hard_rho(D["bins"])
    _, nb = clears(D["bins"], rho)
    d, won, lost, p = paired(D, NEW, REF)
    textpage(
        o,
        "What the p-value does and does not buy",
        "The result is sharp. Its interpretation is the part that needs care.",
        [("THE COMPARISON IS PAIRED AND CLEAN.", GREEN,
          f"Both cells see the same 1500 held-out instances at the same seed_start, disjoint from "
          f"training. The discordant split is {won}-{lost} and McNemar exact gives p = {p:.2g}. Grammar "
          f"is clean on both sides, so no part of the gap is formatting. Nothing about the measurement "
          f"is in doubt."),
         ("BUT IT COMPARES CHECKPOINTS, NOT PROCEDURES.", RED,
          "What is established is that THESE TWO TRAINED MODELS differ by 0.065 on these instances. What "
          "is NOT established is that retraining L=10 would land below L=8 again. There is one seed per "
          "cell on this entire ladder and no rung has ever been replicated, so the p-value says nothing "
          "about the depth, only about the run."),
         ("TWO READINGS SURVIVE, AND ONE SEED CANNOT SEPARATE THEM.", AMBER,
          "H1, the leading one: optimization noise -- an unlucky trajectory at L=10. The final val_nll "
          "gap is 0.0026, about a thousandth, and it maps to 0.065 of solve rate, because free "
          "generation compounds per-token error over a whole trace; a noise-sized perturbation is "
          "therefore sufficient. H2, the interesting one: some depths genuinely train worse at this "
          "width and budget. H2 is the less likely and also unfalsifiable from a single seed."),
         (f"CONSEQUENCE FOR EVERY L* IN THIS PROJECT.", RED,
          f"L* is read as the first rung clearing {GATE:.2f} in all {nb} bins, which assumes depth helps "
          f"monotonically -- that a rung below the gate implies every shallower rung is too. This cell is "
          f"a counterexample, so finding 38's bracket cannot be narrowed by interpolation and findings "
          f"11/14/22/33/38 need re-reading with error bars.")],
        "Honest summary: a clean measurement of an uninterpretable quantity. The fix is cheap and is on "
        "the last page.",
        y0=0.80, width=108)


def page_bins(o, D):
    rho = hard_rho(D["bins"])
    sub = D["bins"][D["bins"].rho_requested == rho]
    cl, nb = clears(D["bins"], rho)
    piv = sub.pivot_table(index="T", columns="L", values="solve_rate")
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, f"Bin by bin at density {rho:.2f}: L=10 sits under L=8 everywhere",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893,
             "Bins group held-out instances by T, the DD rounds they need. A cell owns L* only if its "
             "whole line clears the dashed gate.",
             ha="center", size=MED - 2, color="0.3")

    ax = fig.add_axes([0.07, 0.22, 0.50, 0.57])
    for L in sorted(sub.L.unique(), reverse=True):
        r = sub[sub.L == L].sort_values("T")
        hot = L in (NEW, REF)
        ax.plot(r["T"], r.solve_rate, "o-", lw=3.5 if hot else 1.8,
                ms=11 if hot else 6, color=COL.get(L, GREY),
                alpha=1.0 if hot else 0.40, label=f"L={L}", rasterized=True)
    ax.axhline(GATE, color="k", ls="--", lw=2.5)
    ax.set_xticks(sorted(sub["T"].unique()))
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("T, the number of DD rounds the instance needs")
    ax.set_ylabel("solve rate")
    ax.legend(loc="lower left", ncol=3, fontsize=SMALL)

    ax2 = fig.add_axes([0.655, 0.49, 0.30, 0.30])
    dd = (piv[NEW] - piv[REF])
    ax2.bar([str(int(i)) for i in dd.index], dd.values, color=RED, rasterized=True)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xlabel("T", size=SMALL)
    ax2.set_ylabel(f"L={NEW} minus L={REF}", size=SMALL)
    ax2.tick_params(labelsize=SMALL - 2)
    ax2.set_title("negative in every bin", size=MED - 3, color=RED)

    txt = [f"bins cleared, of {nb}", ""]
    for L in sorted(cl, reverse=True):
        mark = "<-- clears, owns L*" if cl[L] == nb else ""
        txt.append(f"  L={L:<3} {cl[L]}/{nb}  {mark}")
    spread = sub.realized_rho.max() - sub.realized_rho.min()
    txt += ["", f"density {sub.realized_rho.min():.4f}-{sub.realized_rho.max():.4f}, spread {spread:.6f}",
            "  the control: deeper is not sparser"]
    fig.text(0.655, 0.44, "\n".join(txt), size=MED - 5, va="top", family="monospace",
             linespacing=1.45)

    def z_of(L):
        r = sub[sub.L == L].sort_values("T")
        return ca_trend((r.solve_rate * r.n).round(), r.n, r["T"])

    ns = sub[sub.L == NEW].sort_values("T").set_index("T")["n"]
    footer(fig, f"L=10 clears only T={', '.join(str(int(t)) for t in piv.index[piv[NEW] >= GATE])}; the four "
                f"it misses are the four deepest and contiguous at the top -- the one pre-registration the "
                f"result confirmed. Cochran-Armitage trend in T: L=10 z = {z_of(NEW):+.2f}, L=8 z = "
                f"{z_of(REF):+.2f}. The per-bin deltas are not monotone in T, and the three deepest bins "
                f"hold only {', '.join(str(int(ns[t])) for t in (6, 7, 8))} instances, so only the overall "
                f"sign is resolvable.")
    o.savefig(fig)


def page_mechanism(o, D):
    """Is the inversion in the eval, or in the trained model?"""
    rho = hard_rho(D["bins"])
    sub = D["bins"][D["bins"].rho_requested == rho]
    g = D["gram"].groupby("L")[["truncated_rate", "outcome_accuracy"]].max()
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Is it the eval, or is it the model?",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893,
             "Two independent checks, and they agree: the deficit is already in the trained weights.",
             ha="center", size=MED, color="0.3")

    ax = fig.add_axes([0.07, 0.25, 0.38, 0.52])
    for L in sorted(sub.L.unique(), reverse=True):
        r = sub[sub.L == L].sort_values("T")
        hot = L in (NEW, REF)
        ax.plot(r["T"], r.well_formed_rate, "o-", lw=3 if hot else 1.5,
                ms=9 if hot else 5, color=COL.get(L, GREY),
                alpha=1.0 if hot else 0.4, label=f"L={L}", rasterized=True)
    ax.axhline(1.0, color="k", ls=":", lw=1.5)
    ax.set_xticks(sorted(sub["T"].unique()))
    ax.set_ylim(0.93, 1.015)
    ax.set_xlabel("T")
    ax.set_ylabel("well-formed rate")
    ax.set_title("grammar is intact: this is deduction, not format", size=MED - 2)
    ax.legend(loc="lower left", ncol=3, fontsize=SMALL - 2)

    ax = fig.add_axes([0.57, 0.25, 0.37, 0.52])
    Ls = sorted(D["train"])
    ax.plot(Ls, [D["train"][L]["val_nll"] for L in Ls], "s-", lw=2.5, ms=11,
            color="#6b46c1", rasterized=True)
    pool = D["pool"].set_index("L")
    for L in Ls:
        ax.annotate(f"pool {pool.rate[L]:.3f}", (L, D["train"][L]["val_nll"]),
                    textcoords="offset points", xytext=(0, 11), ha="center", size=SMALL - 1)
    ax.plot([NEW], [D["train"][NEW]["val_nll"]], "s", ms=20, mfc="none", mec=RED, mew=3,
            rasterized=True)
    ax.margins(y=0.18)
    ax.set_xticks(Ls)
    ax.set_xlabel("num_layers")
    ax.set_ylabel("final val_nll (lower is better)")
    ax.set_title("teacher-forced NLL reproduces the inversion", size=MED - 2)

    acc = ", ".join(f"L={L} {D['train'][L]['acc']:.4f}" for L in sorted(Ls, reverse=True))
    footer(fig, f"L=10 pool controls: truncated_rate {g.truncated_rate[NEW]:.4f}, outcome_accuracy "
                f"{g.outcome_accuracy[NEW]:.4f}, minimum well-formed rate over all bins "
                f"{sub[sub.L == NEW].well_formed_rate.min():.4f} -- finding 25's grammar-collapse mechanism "
                f"is excluded. Token accuracy ranks the rungs the same way: {acc}. That is concordance, not "
                f"a rehabilitation -- val_nll stays disqualified as a cross-model ranking statistic "
                f"(findings 4/20/25, r = +0.33 over 34 runs) -- but it does put the gap before free generation.")
    o.savefig(fig)


def page_prev(o, D):
    pool = D["pool"].set_index("L")
    best = int(pool.rate.idxmax())
    d, won, lost, p = paired(D, NEW, best)
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Against the previous best on the same eval",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893,
             "Same 1500 instances, same protocol, so this is a paired test rather than a comparison of "
             "two averages.",
             ha="center", size=MED - 2, color="0.3")

    ax = fig.add_axes([0.08, 0.20, 0.36, 0.52])
    names = [f"L={best}\n(prev best)", f"L={NEW}\n(this run)"]
    vals = [pool.rate[best], pool.rate[NEW]]
    errs = [[pool.rate[best] - pool.lo[best], pool.rate[NEW] - pool.lo[NEW]],
            [pool.hi[best] - pool.rate[best], pool.hi[NEW] - pool.rate[NEW]]]
    ax.bar(names, vals, color=[COL[best], RED], yerr=errs, capsize=8, rasterized=True)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.035, f"{v:.4f}", ha="center", size=MED, weight="bold")
    ax.axhline(GATE, color="k", ls="--", lw=2)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("shared-pool solve rate (n=1500)")
    ax.set_title(f"no improvement: {d:+.4f}, p = {p:.1e}", size=MED, color=RED)

    rows = [("(a) HEADLINE", RED, "FALSIFIED",
             "'L=10 clears 0.90 in ALL SIX rho=0.60 bins, so L* narrows to (8, 10]'  ->  clears 2 of 6."),
            ("(b) POOL", RED, "FALSIFIED",
             "'in [0.93, 0.97], resolvably above L=8' -> 0.8007, below falsifier B (0.9220) AND below "
             "L=8. The stated prior said a miss above 0.97 was likelier; it missed low by 0.12. This is "
             "the project's first interval prediction to miss on the low side."),
            ("(c) CONDITIONAL", GREEN, "CONFIRMED",
             "scored because (a) failed: 'the missed bins are the deepest and contiguous' -> missed "
             "T=5,6,7,8, a contiguous run at the top. Its falsifier did not occur."),
            ("(d) CONTROLS", GREEN, "CONFIRMED",
             "realized density 0.583432-0.583920, within 0.001 of 0.5836; all repeat evaluations of the "
             "five reference cells identical instance for instance.")]
    y = 0.78
    for head, col, mark, body in rows:
        fig.text(0.50, y, f"{head}   {mark}", size=MED - 1, weight="bold", color=col)
        fig.text(0.50, y - 0.032, textwrap.fill(body, width=62), size=MED - 5, va="top",
                 linespacing=1.4)
        y -= 0.032 + (textwrap.fill(body, width=62).count("\n") + 1) * 0.030 + 0.022
    fig.text(0.50, y + 0.004, "2 of 4. The two that failed are the two that predicted the result.",
             size=MED - 2, weight="bold", color=GREY)
    footer(fig, "Pre-registrations were committed in the sbatch header (44f581f) before the job was "
                "submitted and are quoted from it verbatim; the analysis script was written while the job "
                "was at epoch 76 of 100. The best N=48 leak-free cell remains finding 38's L=12 at "
                f"{pool.rate[best]:.4f}. This run's value is the negative result, not a new top line.")
    o.savefig(fig)


def page_next(o, D):
    d, _, _, _ = paired(D, NEW, REF)
    textpage(
        o,
        "The next job buys seeds, not rungs",
        "Hypothesis first: retraining L=10 under a second seed lands ABOVE L=8, i.e. the inversion was "
        "noise (H1).",
        [("THE EXPERIMENT.", GREEN,
          "L=10 and L=8 retrained at seed 1, everything else identical -- same 240000 x 100 epochs, same "
          "ff=1024, same pool. Two cells, about 9 h and 7 h on one GPU each, and they can run "
          "concurrently. If both seeds put L=10 below L=8 the ladder really is non-monotone (H2) and "
          "every L* in this project needs re-reading. If seed 1 puts L=10 between L=8 and L=12, this "
          "cell was noise -- but the measured seed spread then becomes a required error bar on every "
          "future rung, which the ladder has never had."),
         ("WHY IT BLOCKS THE HYPER-PARAMETER SCAN.", AMBER,
          f"The planned scan asks whether required reasoning depth tracks MLP depth, and this cell was "
          f"going to be its centre. A scan whose rungs carry an unmeasured seed spread of at least "
          f"{abs(d):.3f} solve rate cannot see an effect smaller than that, and the depth effects it is "
          f"looking for are plausibly that size. Measuring the spread first is cheaper than running the "
          f"scan twice."),
         ("WHAT DOES NOT CHANGE.", GREY,
          "L=12 still clears all six bins at 0.9780 and L=6 still clears none, so the (8, 12] bracket "
          "from finding 38 stands. The leak-free task at N=48 still needs far more depth than the "
          "leaked version did. Nothing here reopens finding 16."),
         ("LIMITATIONS.", GREY,
          "One seed per cell, which is the headline rather than a caveat. L=9 and L=11 are unmeasured, "
          "so the bracket holds four integers of which two are known. Everything is at ff=1024 "
          "(finding 14) and in the fast phase above the removal floor (finding 34). The job's wall "
          "clock spans the DST fall-back; Elapsed 8:58:51 is real time.")],
        "Negative result, reported as one: 8 h 59 m of GPU bought a cell that cannot be interpreted "
        "without a replicate. The methodological finding is worth more than the number.",
        y0=0.80, width=108)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    D = load()
    ex = example_instance(n_nodes=48, rho=0.60, pick="deep")
    note = textwrap.fill(
        "T is the x-axis of every plot in this deck: it is the number of DD ROUNDS this instance needs, "
        "counted on the trace above -- not the number of tokens and not the number of nodes. The depth "
        "bins group held-out instances by exactly this number, at fixed density, and L* asks how many "
        "layers the network needs before it clears 0.90 in all of them. The cell on the previous pages "
        "is the one that was supposed to sit between L=8 and L=12 on that scale.", width=122)
    with PdfPages(os.path.join(OUT, "slides.pdf")) as pdf:
        o = PdfAndPng(pdf, OUT, total=8)
        page_verdict(o, D)
        page_method(o, D)
        page_tokenization(o, ex)
        page_trace(o, ex, note=note)
        page_bins(o, D)
        page_mechanism(o, D)
        page_prev(o, D)
        page_next(o, D)
    mb = os.path.getsize(os.path.join(OUT, "slides.pdf")) / 1e6
    print(f"wrote {OUT}/slides.pdf  ({mb:.2f} MB)")


if __name__ == "__main__":
    main()
