"""Slide deck: the shared-pool paired eval and the probe competence ladder.

Reads
  sudoku_agg_paired_2026-10-21/agg_paired.csv              (48912134)
  sudoku_probes_competence_ladder_2026-10-21/
      probes_competence_ladder.csv, frontier_baseline_auc.json   (48913017)

The headline is a correction: ranking models on each run's own 300-instance
validation set had the top two in the wrong order.
"""
import csv
import json
import textwrap
import os
import collections
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 19, "axes.labelsize": 16,
    "savefig.dpi": 150, "figure.dpi": 150,
})

S = os.environ["SCRATCH"] + "/spin_gnn"
PAIRED = f"{S}/sudoku_agg_paired_2026-10-21/agg_paired.csv"
LADDER = f"{S}/sudoku_probes_competence_ladder_2026-10-21/probes_competence_ladder.csv"
BASE = f"{S}/sudoku_probes_competence_ladder_2026-10-21/frontier_baseline_auc.json"
OUT = f"{S}/sudoku_agg_paired_2026-10-21/paired_eval_slides.pdf"
# Sherlock has no pdftoppm, so layout cannot be eyeballed from the PDF itself.
# Set SLIDES_PNG_DIR to also dump each page as a PNG for a visual check.
PNG_DIR = os.environ.get("SLIDES_PNG_DIR")
_page_no = [0]


def save(pdf, fig):
    _page_no[0] += 1
    pdf.savefig(fig)
    if PNG_DIR:
        os.makedirs(PNG_DIR, exist_ok=True)
        fig.savefig(f"{PNG_DIR}/page{_page_no[0]}.png", dpi=110)
    plt.close(fig)

# Short label -> (sweep dir, own-val agg_solve_rate, own-val k0). The own-val
# numbers are what sweep.csv reported; the whole point of page 2 is that they
# are each measured on a DIFFERENT 300 instances.
MODELS = [
    ("d256 x 5000",  "sudoku_hparam_sweep",                        0.846, 0.24),
    ("d256 x 5000, 900ep", "sudoku_hparam_sweep_2026-10-21_d256_ep900", 0.816, 0.87),
    ("d384 x 5000",  "sudoku_hparam_sweep_2026-10-21_d384",        0.846, 0.99),
    ("d256 x 20000", "sudoku_hparam_sweep_2026-10-21_d256_data4x", 0.973, 0.60),
    ("d384 x 20000", "sudoku_hparam_sweep_2026-10-21_d384_data4x", 0.986, 0.91),
]
CKPT = "d0256_h04_L12_ff01024_lr1e-03_pre"


def load_paired():
    """sweep dir -> row, keyed off the '[parent]' disambiguation suffix."""
    out = {}
    for r in csv.DictReader(open(PAIRED)):
        parent = r["checkpoint"].split("[")[1].rstrip("]").strip()
        out[parent] = {k: (float(v) if k not in ("checkpoint", "vs_reference") else v)
                       for k, v in r.items()}
    return out


def peak_by_run(target, metric):
    """run -> (mean peak over seeds, min, max, peak layers, mean layer-0)."""
    rows = list(csv.DictReader(open(LADDER)))
    out = {}
    runs = {r["run"] for r in rows}
    for run in runs:
        per_seed = collections.defaultdict(dict)
        for r in rows:
            if r["run"] == run and r["target"] == target:
                try:
                    per_seed[r["seed"]][int(r["layer"])] = float(r[metric])
                except ValueError:
                    pass
        if not per_seed or any(not d for d in per_seed.values()):
            continue
        pk = {s: max(d.items(), key=lambda kv: kv[1]) for s, d in per_seed.items()}
        vals = [v for _, v in pk.values()]
        out[run.split("/")[0]] = (
            st.mean(vals), min(vals), max(vals),
            sorted(l for l, _ in pk.values()),
            st.mean([d[0] for d in per_seed.values()]),
        )
    return out


def page(pdf, title, draw, rect=(0.09, 0.27, 0.84, 0.54), caption=None):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, title, fontsize=23, fontweight="bold", va="top")
    ax = fig.add_axes(rect)
    draw(ax)
    if caption:
        # Re-wrap to a width that actually fits 13.33in at this font size.
        # matplotlib's wrap=True wraps to the figure box and then overflows the
        # bottom instead, and hand-placed newlines ran off the right edge.
        text = textwrap.fill(" ".join(caption.split()), width=112)
        fig.text(0.06, 0.205, text, fontsize=12, va="top", linespacing=1.5)
    save(pdf, fig)


def main():
    paired = load_paired()
    front = peak_by_run("frontier_mask", "auroc")
    nnode = peak_by_run("next_node", "accuracy")
    ncolor = peak_by_run("next_color", "accuracy")
    base = json.load(open(BASE))

    with PdfPages(OUT) as pdf:
        # ---------------- 1. headline
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.06, 0.88, "The 300-instance eval had the top two models\nbackwards",
                 fontsize=30, fontweight="bold", va="top")
        fig.text(0.06, 0.60,
                 "Scored on ONE shared pool of 1500 held-out instances, d256 x 20000 solves\n"
                 "0.9847 and d384 x 20000 solves 0.9720 -- the reverse of their own-validation\n"
                 "numbers (0.973 and 0.986). Paired McNemar on the 53 instances where they\n"
                 "disagree: 36-17 for d256, exact p = 0.013.",
                 fontsize=18, va="top")
        fig.text(0.06, 0.30,
                 "Jobs 48912134 (paired eval, 4 checkpoints x 1500 instances) and 48913017\n"
                 "(probe ladder, 5 checkpoints x 3 seeds).  2026-10-21.\n"
                 "Of the five predictions I pre-registered for these two jobs, 2 held and 3 were wrong; page 6 scores them.",
                 fontsize=14, va="top", color="0.3")
        save(pdf, fig)

        # ---------------- 2. the two evals side by side
        def draw2(ax):
            ax.axis("off")
            hdr = ["model", "own val\n(n=300)", "shared pool\n(n=1500)", "95% Wilson",
                   "paired p vs\nd384 x 20000"]
            rows = []
            for label, d, own_agg, _k0 in MODELS:
                if d not in paired:
                    rows.append([label.replace("\n", " "), f"{own_agg:.3f}", "not in this job", "", ""])
                    continue
                r = paired[d]
                p = r["mcnemar_exact_p"]
                pstr = "reference" if r["delta"] == 0 else f"{p:.2g}"
                rows.append([label.replace("\n", " "), f"{own_agg:.3f}", f"{r['solve_rate']:.4f}",
                             f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]", pstr])
            t = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(14)
            t.scale(1, 2.4)
            for (row, _col), cell in t.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontweight="bold")
                elif rows[row - 1][0].startswith("d256 x 20000"):
                    cell.set_facecolor("#e8f4e8")
        page(pdf, "Same models, two evals", draw2, rect=(0.06, 0.31, 0.88, 0.52),
             caption="Own-val numbers are NOT comparable across rows: validation instances are seeded at "
                     "seed + n_train_instances, so each n_train scores a different 300, and the\n"
                     "n_train=20000 runs train over the seeds the 5000-instance runs validated on. I had "
                     "called this 'the identical eval' -- it is the identical protocol only.\n"
                     "The shared pool is one instance set from seed 5e10 that no model saw in training.")

        # ---------------- 3. the reversal, drawn
        def draw3(ax):
            labs = [m[0].replace("\n", " ") for m in MODELS if m[1] in paired]
            own = [m[2] for m in MODELS if m[1] in paired]
            sh = [paired[m[1]]["solve_rate"] for m in MODELS if m[1] in paired]
            lo = [paired[m[1]]["wilson_lo"] for m in MODELS if m[1] in paired]
            hi = [paired[m[1]]["wilson_hi"] for m in MODELS if m[1] in paired]
            for i, (o, s, l, h, lab) in enumerate(zip(own, sh, lo, hi, labs)):
                ax.plot([0, 1], [o, s], "-o", lw=2.5, ms=9, rasterized=True)
                ax.errorbar([1], [s], yerr=[[s - l], [h - s]], fmt="none",
                            ecolor="0.4", capsize=5, rasterized=True)
                ax.annotate(lab, (1.02, s), fontsize=14, va="center")
            ax.set_xlim(-0.15, 1.55)
            ax.set_xticks([0, 1])
            # Three-line tick labels reached down into the caption.
            ax.set_xticklabels(["own val\n(n=300, differs per row)",
                                "shared pool\n(n=1500, identical)"])
            ax.set_ylabel("solve_rate")
            ax.grid(axis="y", alpha=0.3)
        page(pdf, "The ranking is not stable between them", draw3,
             rect=(0.09, 0.35, 0.60, 0.47),
             caption="The two data4x models swap places, and so do the two 5000-instance models (0.846 and "
                     "0.846 on their own vals; 0.876 vs 0.820 on the shared pool, 95% intervals\n"
                     "disjoint -- unpaired, as this job only saved pairings against the reference). Bars are "
                     "Wilson intervals, which unlike k/n +- z*SE stay inside [0, 1] at these rates.\n"
                     "What survives unchanged: both n_train=20000 models beat both 5000 models overwhelmingly "
                     "(p ~ 1e-29 and 1e-55).")

        # ---------------- 4. frontier probe across the ladder
        def draw4(ax):
            labs, vals, errs, lay = [], [], [], []
            for label, d, _a, _k in MODELS:
                if d not in front:
                    continue
                m, lo, hi, layers, _l0 = front[d]
                labs.append(label)
                vals.append(m)
                errs.append([m - lo, hi - m])
                lay.append(layers[len(layers) // 2])
            x = range(len(labs))
            ax.bar(x, vals, color="#4477aa", rasterized=True)
            ax.errorbar(x, vals, yerr=list(zip(*errs)), fmt="none", ecolor="k", capsize=5)
            for i, (v, l) in enumerate(zip(vals, lay)):
                ax.text(i, v + 0.004, f"{v:.3f}\nL{l}", ha="center", fontsize=13)
            ax.axhline(base["degree_aug"]["auroc"], color="crimson", ls="--", lw=2,
                       label=f"degree_aug label-only baseline ({base['degree_aug']['auroc']:.3f})")
            ax.axhline(0.863, color="darkorange", ls=":", lw=2,
                       label="0.863 = the d0128 series value (finding 3)")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labs, fontsize=13)
            ax.set_ylim(0.70, 0.98)
            ax.set_ylabel("frontier_mask peak AUROC")
            ax.legend(fontsize=12, loc="lower right")
        page(pdf, "Finding 3 overstated it: the frontier does strengthen", draw4,
             caption="Every model here is ABOVE the 0.863 I reported as flat, and the value rises with "
                     "competence. Caveat that cuts the claim down: 0.863 came from a d0128 series,\n"
                     "so part of the gap is width, not training. Clean within-width: d256 0.907 (5000) -> "
                     "0.916 (20000), d384 0.916 -> 0.938, against a seed spread of +-0.001.\n"
                     "But a 10x cut in failures buys only +0.03 AUROC, and the layer-0 control is 0.676 "
                     "throughout. Revised: represented early, strengthened slightly -- not 'never develops'.")

        # ---------------- 5. what actually tracks competence
        def draw5(ax):
            sh, nn, nc, labs = [], [], [], []
            for label, d, _a, _k in MODELS:
                if d not in paired or d not in nnode:
                    continue
                sh.append(paired[d]["solve_rate"])
                nn.append(nnode[d][0])
                nc.append(ncolor[d][0])
                labs.append(label.replace("\n", " "))
            ax.plot(sh, nn, "o", ms=14, color="#228833", label="next_node (peak accuracy)",
                    rasterized=True)
            ax.plot(sh, nc, "s", ms=13, color="#ccbb44", label="next_color (peak accuracy)",
                    rasterized=True)
            for s, n, lab in zip(sh, nn, labs):
                # The two n_train=20000 points sit 0.013 apart on x, so their
                # labels collide unless they are pushed in opposite directions.
                rank = sorted(sh).index(s)
                dx, dy, ha = (8, -20, "left")
                if rank == len(sh) - 2:
                    dx, dy, ha = (-10, -26, "right")
                elif rank == len(sh) - 1:
                    dx, dy, ha = (-6, 12, "right")
                ax.annotate(lab, (s, n), textcoords="offset points",
                            xytext=(dx, dy), fontsize=13, ha=ha)
            ax.set_xlim(min(sh) - 0.02, max(sh) + 0.035)
            ax.set_xlabel("shared-pool solve_rate (n = 1500)")
            ax.set_ylabel("probe peak accuracy")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=13, loc="upper left")
        page(pdf, "Next-step prediction tracks competence; the frontier does not",
             draw5, rect=(0.09, 0.30, 0.84, 0.52),
             caption="next_node peak accuracy orders the four models EXACTLY as the shared-pool eval does, "
                     "reversal included (0.964 > 0.957 >> 0.810 > 0.789 against 0.985 > 0.972 >>\n"
                     "0.876 > 0.820). frontier_mask does not: it puts d384 x 20000 on top, which the paired "
                     "eval says is second. Both next-step probes step up at n_train 5000 -> 20000\n"
                     "rather than sloping, and both peak at the LAST layer -- whereas the frontier peaks mid-stack.")

        # ---------------- 6. pre-registration scorecard
        def draw6(ax):
            ax.axis("off")
            rows = [
                ["48912134 (a)", "both n_train=20000 models beat both\n5000 models at p < 1e-6",
                 "HELD", "p = 6e-29 and 8e-55"],
                ["48912134 (b)", "d384 vs d256 at 20000 stays a tie\n(|delta| < 0.02 AND p > 0.05)",
                 "WRONG", "|delta| 0.013 but p = 0.013;\nresolved, and d256 is ahead"],
                ["48912134 (c)", "the two 5000-instance models are\nwithin noise of each other",
                 "WRONG", "0.876 vs 0.820, disjoint\n95% intervals"],
                ["48913017", "frontier AUROC 0.863 +- 0.02 on all\nfive, no trend, d384 not above d256",
                 "WRONG", "0.907-0.938, rising, and\nd384 > d256 at matched data"],
                ["48913017 (2nd)", "next_color tracks competence across\nmodels, not just within a run",
                 "HELD", "0.69 -> 0.83 with n_train"],
            ]
            t = ax.table(cellText=rows, colLabels=["job", "pre-registered", "verdict", "measured"],
                         loc="center", cellLoc="left", colWidths=[0.13, 0.36, 0.11, 0.32])
            t.auto_set_font_size(False)
            t.set_fontsize(12.5)
            t.scale(1, 3.1)
            for (row, col), cell in t.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontweight="bold")
                elif col == 2:
                    cell.set_facecolor("#e8f4e8" if rows[row - 1][2] == "HELD" else "#fbe6e6")
        page(pdf, "Pre-registration scorecard: 2 held, 3 wrong", draw6,
             rect=(0.05, 0.26, 0.90, 0.58),
             caption="The two that held were the ones I was confident about. All three failures are the same "
                     "mistake: I assumed the remaining differences were unresolvable, and they were\n"
                     "not once the eval stopped being 300 instances per model.")

        # ---------------- 7. state of play
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.06, 0.90, "Where this leaves the project", fontsize=23,
                 fontweight="bold", va="top")
        fig.text(0.06, 0.78,
                 "Best model (shared pool, n=1500):  d256 x 20000, 0.9847 [0.977, 0.990]\n"
                 "   archived to $OAK/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_..._sharedpool0.9847\n"
                 "Previous best on the old eval:  0.847 (d256 x 5000), which the shared pool puts at 0.820\n\n"
                 "Running\n"
                 "   48912548  d256 x n_train 80000, one change from the 0.9847 model. Pre-registered:\n"
                 "             agg a tie, k0 >= 0.85, mean_o1 >= 0.96.  ~6h.\n\n"
                 "Next\n"
                 "   1. Re-score 48912548 on the same 1500-instance pool, not on its own validation set.\n"
                 "   2. Finding 5 (depth costs accuracy at fixed density) is still unexplained; finding 7\n"
                 "      ruled out the token-level slots, so the next test has to use free generation.\n"
                 "   3. The frontier result needs a d0128 model trained at n_train=20000 to separate\n"
                 "      width from data in the 0.863 -> 0.907 gap.",
                 fontsize=16, va="top")
        save(pdf, fig)

    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.3f} MB)")


if __name__ == "__main__":
    main()
