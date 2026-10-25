"""Slide deck: why N=24 cannot show a reasoning phase transition.

Built entirely from CSVs already on disk (jobs 48579545, 48581589, 48581590).
No GPU, no new measurement -- this is the finite-size scaling analysis of
data that was collected on 10-19 and summarised too optimistically at the
time. Run:

    python -m notebooks.make_finite_size_deck
"""
import csv
import glob
import math
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

S = os.environ.get("SCRATCH", "/scratch/users/hshunt") + "/spin_gnn"
OUT = os.path.join(S, "notebooks", "sudoku_finite_size_2026-10-24")
REMOVAL = f"{S}/sudoku_dd_removal_small_n_2026-10-19/removal_path_T_vs_rho.csv"
RANDOM = f"{S}/dd_random_hint_transition_2026-10-19/N*.csv"
MODEL = f"{S}/sudoku_hint_density_transition_2026-10-19/transition.csv"

FIX_RHO = (0.95, 0.90, 0.80, 0.70, 0.60, 0.55, 0.50)
BIG, MED, SMALL = 25, 17, 13
plt.rcParams.update({"font.size": MED, "axes.titlesize": MED + 2,
                     "axes.labelsize": MED, "legend.fontsize": SMALL})


def footer(fig, text):
    """Bottom caption, wrapped by hand -- matplotlib's wrap=True ignores the
    rect passed to tight_layout and spills off the slide."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 118:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    fig.text(0.5, 0.012, "\n".join(lines), ha="center", va="bottom",
             size=SMALL - 1, color="0.3")


def logN_axis(ax, Ns):
    """Log x with one labelled tick per measured N.

    The default log locator puts 3x10^1, 4x10^1 ... on top of each other when
    the range spans barely more than a decade.
    """
    ax.set_xscale("log")
    ax.set_xticks(list(Ns))
    ax.set_xticklabels([str(N) for N in Ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def load_removal():
    rows = [r for r in csv.DictReader(open(REMOVAL))]
    out = {}
    for N in sorted({int(r["n_nodes"]) for r in rows}):
        rs = [r for r in rows if int(r["n_nodes"]) == N]
        rs.sort(key=lambda r: float(r["rho"]))
        rho = np.array([float(r["rho"]) for r in rs])
        T = np.array([float(r["mean_T"]) if r["mean_T"] not in ("", "nan") else np.nan
                      for r in rs])
        # Bins that only a few unusually deep paths reach average over a
        # biased subsample, so they are excluded from every fit and curve.
        reach = np.array([float(r["frac_paths_reaching_rho"]) for r in rs])
        T = np.where(reach >= 0.8, T, np.nan)
        out[N] = {"rho": rho, "T": T, "rho_min": float(rs[0]["mean_rho_min"])}
    return out


def t_at(d, target):
    i = int(np.argmin(np.abs(d["rho"] - target)))
    if abs(d["rho"][i] - target) > 0.006 or not np.isfinite(d["T"][i]):
        return np.nan
    return float(d["T"][i])


def slope(xs, ys):
    """Least-squares d(ln y)/d(ln x)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def crossing(xs, ys, level):
    for i in range(1, len(ys)):
        if ys[i - 1] < level <= ys[i]:
            if ys[i] == ys[i - 1]:
                return xs[i]
            return xs[i - 1] + (level - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1])
    return float("nan")


def load_random():
    out = {}
    for f in sorted(glob.glob(RANDOM), key=lambda p: int(os.path.basename(p)[1:-4])):
        rs = sorted(csv.DictReader(open(f)), key=lambda r: float(r["rho"]))
        N = int(rs[0]["n_nodes"])
        xs = [float(r["rho"]) for r in rs]
        ys = [float(r["resolved_frac"]) for r in rs]
        out[N] = {"rho": np.array(xs), "resolved": np.array(ys),
                  "rho10": crossing(xs, ys, 0.1), "rho50": crossing(xs, ys, 0.5),
                  "rho90": crossing(xs, ys, 0.9)}
    return out


def load_model():
    """Model solve_rate vs REALIZED density, for the stronger checkpoint.

    Many target rho collapse onto the same realized density because
    removal_path floors to an integer hint count, so duplicates are averaged
    rather than plotted twice.
    """
    rows = [r for r in csv.DictReader(open(MODEL))]
    ckpt = sorted({r["checkpoint"] for r in rows})[-1]
    agg = defaultdict(list)
    dd = defaultdict(list)
    for r in rows:
        if r["checkpoint"] != ckpt:
            continue
        agg[round(float(r["realized_rho"]), 4)].append(float(r["solve_rate"]))
        dd[round(float(r["realized_rho"]), 4)].append(float(r["dd_resolved_frac"]))
    xs = sorted(agg)
    return (np.array(xs), np.array([statistics.mean(agg[x]) for x in xs]),
            np.array([statistics.mean(dd[x]) for x in xs]), ckpt)


def title_page(pdf, exps):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.86, "No reasoning phase transition at N = 24", size=BIG + 5, weight="bold")
    fig.text(0.06, 0.78, "Finite-size scaling of deductive-deduction depth, spin_gnn / sudoku-dd",
             size=MED + 1, color="0.35")
    fig.text(0.06, 0.63,
             "Headline: deduction depth T grows as a power of N at EVERY hint\n"
             f"density. The exponent d lnT / d lnN drifts from {exps[0.95]:+.2f} at rho=0.95\n"
             f"to {exps[0.50]:+.2f} at rho=0.50 and never reaches zero, so there is no\n"
             "N-independent regime to sit outside of and no sharp critical\n"
             "density to sit at. N=24 is not near a transition -- it is far\nfrom the asymptotics.",
             size=MED + 3, va="top")
    fig.text(0.06, 0.30,
             "Two ensembles, neither with a solvability transition:\n"
             "  - random hints: threshold slides to rho=1 as N grows, "
             "unhinted nodes at the crossing stay O(1)\n"
             "  - removal path (what we train on): DD-solvable by construction, "
             "dd_resolved_frac = 1.0 everywhere\n\n"
             "Source: jobs 48579545 / 48581589 / 48581590, 10-19. "
             "No new measurement, CPU-only re-analysis.",
             size=MED, va="top", color="0.2")
    fig.text(0.06, 0.05, "2026-10-24", size=SMALL, color="0.5")
    pdf.savefig(fig)
    plt.close(fig)


def page_two_ensembles(pdf, rnd, model):
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    ax = axes[0]
    for N, d in rnd.items():
        ax.plot(d["rho"], d["resolved"], "o-", ms=4, lw=2, label=f"N={N}", rasterized=True)
    ax.set_xlabel("hint density rho")
    ax.set_ylabel("fraction DD fully resolves")
    ax.set_title("Random hints: DD does fail,\nbut the threshold runs to rho = 1")
    ax.set_xlim(0.55, 1.005)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs, solve, dd, ckpt = model
    ax.plot(xs, dd, "s-", lw=3, ms=7, color="tab:green",
            label="DD (dd_resolved_frac)", rasterized=True)
    ax.plot(xs, solve, "o-", lw=3, ms=7, color="tab:red",
            label="transformer solve_rate", rasterized=True)
    ax.set_xlabel("realized hint density rho")
    ax.set_ylabel("fraction solved")
    ax.set_title("Removal path (the training ensemble):\nDD never fails; the model does")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.suptitle("The 0.99 -> 0.44 curve is the model degrading, not DD transitioning",
                 size=MED + 4, weight="bold")
    footer(fig, "STATUS finding 1 said 'hint-density transition is real at N=24'. "
                "It is a transformer competence curve against a flat DD baseline.")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def page_threshold_runs_off(pdf, rnd):
    Ns = sorted(rnd)
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    ax = axes[0]
    ax.plot(Ns, [rnd[N]["rho50"] for N in Ns], "o-", lw=3, ms=9, label="50% crossing")
    ax.fill_between(Ns, [rnd[N]["rho10"] for N in Ns], [rnd[N]["rho90"] for N in Ns],
                    alpha=0.25, label="10-90% width", rasterized=True)
    ax.axhline(1.0, color="k", ls="--", lw=1.5)
    logN_axis(ax, Ns)
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("hint density rho")
    ax.set_title("The crossing sharpens AND\nslides to rho = 1")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    unh = [N * (1 - rnd[N]["rho50"]) for N in Ns]
    ax.plot(Ns, unh, "o-", lw=3, ms=9, color="tab:purple", label="measured")
    ref = [unh[0] * N / Ns[0] for N in Ns]
    ax.plot(Ns, ref, "k--", lw=2, label="if it were extensive (prop. to N)")
    logN_axis(ax, Ns)
    ax.set_yscale("log")
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("unhinted nodes at the 50% crossing")
    ax.set_title("...because only O(1) nodes are\never unhinted there")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("Random-hint ensemble: a threshold at zero unhinted density, not at finite rho",
                 size=MED + 4, weight="bold")
    footer(fig, f"unhinted at crossing: {', '.join(f'{u:.1f}' for u in unh)} "
                f"for N = {', '.join(str(N) for N in Ns)}. A finite-density transition "
                "would need this to grow 16x across this range; it grows 2.9x.")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def page_removal_curves(pdf, rem):
    Ns = sorted(rem)
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(Ns)))
    for c, N in zip(cmap, Ns):
        d = rem[N]
        ok = np.isfinite(d["T"])
        ax.plot(d["rho"][ok], d["T"][ok], "-", lw=3, color=c, label=f"N={N}", rasterized=True)
    ax.axvline(0.4777, color="r", ls="--", lw=2)
    ax.text(0.485, ax.get_ylim()[1] * 0.92, "rho_min ~ 0.478", color="r", size=SMALL)
    ax.set_xlabel("hint density rho")
    ax.set_ylabel("mean DD depth T (rounds)")
    ax.set_title("Depth rises toward the removal floor\nat every N -- but so does the whole curve")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Ns, [rem[N]["rho_min"] for N in Ns], "o-", lw=3, ms=9, color="tab:red")
    ax.axhline(0.4777, color="k", ls="--", lw=1.5)
    logN_axis(ax, Ns)
    ax.set_ylim(0.40, 0.55)
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("mean rho_min reached by greedy removal")
    ax.set_title("The floor itself is already converged at N = 24")
    ax.grid(alpha=0.3)
    fig.suptitle("Removal-path ensemble: the floor location is N-independent, the depth is not",
                 size=MED + 4, weight="bold")
    footer(fig, "rho_min = " + ", ".join(f"{rem[N]['rho_min']:.3f}" for N in Ns)
                + " for N = " + ", ".join(str(N) for N in Ns)
                + ". So going to larger N buys nothing on the density axis. "
                "Curves show only bins that at least 80% of removal paths reach.")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def page_scaling_test(pdf, rem, exps):
    Ns = sorted(rem)
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    ax = axes[0]
    for rho in FIX_RHO:
        xs = [N for N in Ns if np.isfinite(t_at(rem[N], rho))]
        ys = [t_at(rem[N], rho) for N in xs]
        ax.plot(xs, ys, "o-", lw=2.5, ms=7,
                label=f"rho={rho:.2f}  (slope {exps[rho]:+.2f})", rasterized=True)
    logN_axis(ax, Ns)
    ax.set_yscale("log")
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("mean DD depth T at fixed rho")
    ax.set_title("T vs N at FIXED density")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    rhos = list(FIX_RHO)
    ax.plot(rhos, [exps[r] for r in rhos], "o-", lw=3, ms=10, color="tab:orange")
    ax.axhline(0.0, color="k", ls="--", lw=2)
    ax.text(0.80, 0.02, "a = 0 would mean 'no finite-size effect here'", size=SMALL, color="0.3")
    ax.axvline(0.4777, color="r", ls="--", lw=2)
    ax.set_ylim(-0.05, 0.45)
    ax.invert_xaxis()
    ax.set_xlabel("hint density rho  (floor on the right)")
    ax.set_ylabel("exponent a in  T ~ N^a")
    ax.set_title("The exponent drifts, it does not switch on")
    ax.grid(alpha=0.3)
    fig.suptitle("The scaling test: is rho_min a critical point, or just where greedy stops?",
                 size=MED + 4, weight="bold")
    footer(fig, "A true critical point needs a ~ 0 away from it and a > 0 only near it. "
                "Instead a > 0 everywhere, rising smoothly from 0.18 to 0.38. "
                "Verdict: no sharp critical density in this ensemble -- rho sets the "
                "exponent, it does not switch criticality on.")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def page_cost(pdf, rem):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("What N = 24 gives you, and what more would cost",
                 size=MED + 4, weight="bold")
    ax = fig.add_subplot(1, 2, 1)
    Ns = sorted(rem)
    span = [np.nanmax(rem[N]["T"]) for N in Ns]
    ax.plot(Ns, span, "o-", lw=3, ms=9, color="tab:blue")
    ax.axvline(24, color="r", ls="--", lw=2)
    ax.text(26, max(span) * 0.5, "trained here", color="r", size=SMALL)
    logN_axis(ax, Ns)
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("max mean T (bins >=80% of paths reach)")
    ax.set_title("Dynamic range of the order parameter")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 2, 2)
    ax.axis("off")
    # Sequences are ~11*N tokens and attention is quadratic, so cost per
    # instance scales ~N^2; the 5.5 GPU-h figure is the measured cost of the
    # current best recipe (d0256 x 80000 x 300 epochs, job 48912548).
    base_h = 5.5
    rowsN = [24, 48, 96, 192]
    cells = [["N", "tokens", "cost", "best recipe", "max T"]]
    for N in rowsN:
        rel = (11 * N + 9) ** 2 / (11 * 24 + 9) ** 2
        h = base_h * rel
        t = f"{np.nanmax(rem[N]['T']):.1f}" if N in rem else "-"
        cells.append([str(N), str(11 * N + 9), f"{rel:.0f}x",
                      f"{h:.1f} GPU-h" if h < 48 else f"{h / 24:.1f} GPU-days", t])
    tab = ax.table(cellText=cells[1:], colLabels=cells[0], loc="center", cellLoc="center",
                   colWidths=[0.13, 0.16, 0.13, 0.36, 0.15])
    tab.auto_set_font_size(False)
    tab.set_fontsize(MED - 2)
    tab.scale(1, 2.6)
    for j in range(5):
        tab[0, j].set_facecolor("#dddddd")
        tab[0, j].set_text_props(weight="bold")
    ax.set_title("Cost of raising N (quadratic attention)", pad=30)
    footer(fig, "Doubling the order parameter's range costs ~16x in GPU time, and a range "
                "comparable to N=996 is out of reach entirely. 'max T' is the deepest bin "
                "at least 80% of removal paths reach; 5.5 GPU-h is the measured cost of the "
                "current best recipe (d0256 x 80000 x 300 ep, job 48912548). This is why the "
                "next move is a CPU sweep over (degree, n_colors), not a bigger N.")
    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def page_next(pdf):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.88, "What is running, and what would change the picture",
             size=BIG, weight="bold")
    fig.text(0.06, 0.76,
             "Job 49431733 (CPU-only, ~65 min): sweep DD over 20 (degree, n_colors) cells\n"
             "at N = 24 ... 996, both ensembles. Pre-registered:",
             size=MED + 2, va="top")
    fig.text(0.09, 0.62,
             "(a)  rho_min rises as the propagation ratio d/(q-1) falls:\n"
             "       rho_min(d=3,q=4) > 0.477 > rho_min(d=10,q=4)\n"
             "(b)  some cell with d/(q-1) <= 1.2 reaches max T >= 22 at N=96,\n"
             "       i.e. 1.5x the 14.9 that (5,4) gives there\n"
             "(c)  falsifier: rho_min and max T flat in (d,q) to +-20%,\n"
             "       meaning no cheap corner exists\n"
             "(d)  scaling: T_at_rho90 flatter in N than T_at_rho55,\n"
             "       if anything here is critical at all\n"
             "(e)  control: unhinted-at-crossing stays O(1) in every random cell",
             size=MED - 2, va="top", family="monospace")
    fig.text(0.06, 0.28,
             "Decision rule set in advance:",
             size=MED + 2, weight="bold", va="top")
    fig.text(0.09, 0.22,
             "- If (b) holds, retrain at the winning (degree, n_colors) and keep N small -- cheap.\n"
             "- If (c) holds instead, there is no cheap corner: the transition needs N ~ few hundred,\n"
             "   which is GPU-weeks, and the honest move is to stop calling this a phase transition\n"
             "   and study the depth-vs-competence curve on its own terms.",
             size=MED, va="top")
    fig.text(0.06, 0.05,
             "Either way, STATUS finding 1 gets rewritten: DD does not transition at N=24.",
             size=SMALL, color="0.3")
    pdf.savefig(fig)
    plt.close(fig)


class PdfAndPng:
    """Writes the PDF and a PNG per page.

    There is no PDF rasterizer on this cluster, so the PNGs are the only way
    to actually look at the deck before sending it, and a results notebook
    needs them anyway.
    """

    def __init__(self, pdf, png_dir):
        self.pdf, self.png_dir, self.n = pdf, png_dir, 0

    def savefig(self, fig):
        self.n += 1
        self.pdf.savefig(fig)
        fig.savefig(os.path.join(self.png_dir, f"page{self.n}.png"), dpi=110)


def main():
    os.makedirs(OUT, exist_ok=True)
    rem, rnd, model = load_removal(), load_random(), load_model()
    exps = {}
    for rho in FIX_RHO:
        pts = [(N, t_at(rem[N], rho)) for N in sorted(rem)]
        pts = [(N, t) for N, t in pts if np.isfinite(t) and t > 0]
        exps[rho] = slope([p[0] for p in pts], [p[1] for p in pts])

    pdf_path = os.path.join(OUT, "finite_size_scaling.pdf")
    with PdfPages(pdf_path) as raw:
        pdf = PdfAndPng(raw, OUT)
        title_page(pdf, exps)
        page_two_ensembles(pdf, rnd, model)
        page_threshold_runs_off(pdf, rnd)
        page_removal_curves(pdf, rem)
        page_scaling_test(pdf, rem, exps)
        page_cost(pdf, rem)
        page_next(pdf)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB)")

    with open(os.path.join(OUT, "scaling_exponents.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "exponent_a"] + [f"T_N{N}" for N in sorted(rem)])
        for rho in FIX_RHO:
            w.writerow([rho, round(exps[rho], 4)]
                       + [round(t_at(rem[N], rho), 3) for N in sorted(rem)])
    print("exponents:", {k: round(v, 3) for k, v in exps.items()})


if __name__ == "__main__":
    main()
