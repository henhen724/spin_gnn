"""Slide deck: the easy-end collapse was never a collapse.

Results notebook for jobs 49430217 (diagnostic + shared pool at n=6000) and
49434190 (the same diagnostic re-run with the failure classifier fixed).

The open problem this closes, as STATUS phrased it on 10-22: "`solve_rate_k0`
0.60 -> 0.07, `mean_o1` 0.918 -> 0.776, with shared pool at 0.9960. k0 = 0.07
alongside 0.996 is a contradiction." It is not a contradiction, and the fix is
arithmetic, not training: `removal_path` snapshots at `int(floor(rho * N))`
hints, so `rho_max = 0.95` at N = 24 is 22 hints and the k = 0 and k = 1 bins
sit outside the training support entirely.

Run:  python -m notebooks.make_easy_end_deck
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

S = os.environ.get("SCRATCH", "/scratch/users/hshunt") + "/spin_gnn"
OUT = os.path.join(S, "notebooks", "sudoku_easy_end_2026-10-24")
HINTS = f"{S}/sudoku_easy_end_2026-10-24/easy_end_by_hint_count.csv"
RELABELLED = f"{S}/sudoku_easy_end_2026-10-24_relabelled/easy_end_by_hint_count_relabelled.csv"
POOL = f"{S}/sudoku_easy_end_2026-10-24/agg_paired_pool6000.csv"

# The three checkpoints, in the order they are discussed, with the short names
# used on every slide.
SERIES = [
    ("d256_data16x", "d0256 x 80000  (best)", "tab:blue"),
    ("d256_data4x", "d0256 x 20000", "tab:orange"),
    ("d384", "d0384 x 5000", "tab:green"),
]
CEILING = 22  # floor(0.95 * 24)

# n_train is not in config.json, only in each sweep's sweep.csv, so the
# directory suffixes are spelled out here rather than derived: `_d384` is the
# 5000-instance run and `_d384_data4x` the 20000-instance one, and chaining
# string replacements over those two names silently produced "d384 x 5000 x
# 20000" on the first build.
NTRAIN = {"d128_data4x": 20000, "d256_data4x": 20000, "d256_data16x": 80000,
          "d384": 5000, "d384_data4x": 20000}


def pretty(short_name, sep=" x "):
    return f"{short_name.split('_')[0]}{sep}{NTRAIN[short_name]}"


def short(checkpoint):
    """`sudoku_hparam_sweep_2026-10-21_d256_data16x/d0256_...` -> `d256_data16x`."""
    m = re.search(r"2026-10-21_([^/]+)/", checkpoint)
    return m.group(1) if m else checkpoint


def load_hints(path):
    rows = [r for r in csv.DictReader(open(path))]
    out = {}
    for r in rows:
        out.setdefault(short(r["checkpoint"]), {})[int(r["n_hints_target"])] = r
    return out


def load_pool():
    rows = [r for r in csv.DictReader(open(POOL))]
    for r in rows:
        m = re.search(r"\[sudoku_hparam_sweep_2026-10-21_([^\]]+)\]", r["checkpoint"])
        r["short"] = m.group(1) if m else r["checkpoint"]
    return rows


def labels_present(rows):
    """Failure-label columns actually in the CSV.

    The first run (49430217) predates the `empty_round` split and has no such
    column; the deck must render from whichever version is on disk and say
    which it used, not silently assume the newer schema.
    """
    any_row = next(iter(next(iter(rows.values())).values()))
    return [k[len("n_"):] for k in any_row
            if k.startswith("n_") and k not in ("n_instances", "n_failed", "n_hints_target")]


def page_title(pdf, relabelled):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.80, "The easy end never collapsed", ha="center", size=BIG + 7, weight="bold")
    fig.text(0.5, 0.695, "graph-colouring reasoning transformer, N = 24   ·   jobs 49430217"
             + (" + 49434190" if relabelled else ""),
             ha="center", size=MED, color="0.35")
    fig.text(0.5, 0.52,
             "The best model scores EXACTLY 1.000, with 0 failures of 300, at every\n"
             "hint density it was trained on. It only breaks above the training ceiling.",
             ha="center", va="center", size=MED + 7, linespacing=1.6)
    fig.text(0.5, 0.265,
             "`solve_rate_k0` = 0.07 next to a 0.996 shared-pool rate was logged as a\n"
             "contradiction for two days. It was an arithmetic error in what k = 0 means:\n"
             "that bin asks for 24 hints out of 24, and training never went past 22.",
             ha="center", va="center", size=MED, color="0.2", linespacing=1.7)
    footer(fig, "Open problem closed, not a new result: no model was retrained. The fix is to stop "
                "reading two out-of-distribution bins as the easy end.")
    pdf.savefig(fig)


def page_arithmetic(pdf, hints):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("Why k = 0 and k = 1 are out of distribution", size=MED + 7, weight="bold", y=0.95)
    # Explicit axes rather than add_subplot: a two-line header row drawn with
    # loc="upper center" overflows the axes box upward and clips against the
    # title, and the table has no padding option of its own.
    ax = fig.add_axes([0.05, 0.40, 0.90, 0.44])
    ax.axis("off")
    best = hints["d256_data16x"]
    body = [["k", "hints", "rho", "true DD\nrounds T", "in training\nsupport?", "solve_rate\n(best model)"]]
    for k in (0, 1, 2, 3, 4):
        n_h = 24 - k
        r = best.get(n_h)
        body.append([str(k), f"{n_h}/24", f"{n_h / 24:.4f}",
                     f"{float(r['mean_rounds_true']):.2f}" if r else "--",
                     "NO" if n_h > CEILING else "yes",
                     f"{float(r['solve_rate']):.3f}" if r else "--"])
    t = ax.table(cellText=body[1:], colLabels=body[0], loc="upper center",
                 cellLoc="center", colWidths=[0.07, 0.12, 0.13, 0.17, 0.19, 0.19])
    t.auto_set_font_size(False)
    t.set_fontsize(MED - 1)
    t.scale(1, 2.5)
    for j in range(6):
        t[(0, j)].set_facecolor("0.88")
        t[(0, j)].set_text_props(weight="bold")
    for i in (1, 2):                      # the two OOD rows
        for j in range(6):
            t[(i, j)].set_facecolor("#ffe8e0")
    fig.text(0.5, 0.205,
             "rho_max = 0.95, and removal_path stops at int(floor(rho * N)) = floor(22.8) = 22.\n"
             "mean_o1 averages these five bins, so it mixes two the model has never seen with\n"
             "three it has. k = 0 is doubly strange: with every node hinted the true trace has\n"
             "ZERO deduction rounds, a sequence shape no training instance has.",
             ha="center", va="center", size=MED - 2, linespacing=1.8, family="monospace")
    footer(fig, "300 instances per bin, seed_start = 6e10, disjoint from the shared pool (5e10) and from "
                "every training seed. Greedy decoding.")
    pdf.savefig(fig)


def page_profile(pdf, hints):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for key, label, colour in SERIES:
        d = hints[key]
        xs = sorted(d)
        ax.plot(xs, [float(d[x]["solve_rate"]) for x in xs], "o-", color=colour,
                label=label, lw=2.5, ms=9)
    ax.axvspan(CEILING + 0.5, 24.5, color="#ffe8e0", zorder=0)
    ax.text(23.5, 0.42, "outside\ntraining\nsupport", ha="center", size=SMALL + 1, color="#a03010")
    ax.axvline(CEILING + 0.5, color="#a03010", ls="--", lw=1.5)
    ax.set_xlabel("hints given (out of 24)")
    ax.set_ylabel("solve_rate")
    ax.set_title("solve_rate across and beyond the training ceiling", size=MED + 5, weight="bold")
    ax.set_xticks([16, 18, 20, 21, 22, 23, 24])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    footer(fig, "The best model (blue) is a flat line at 1.000 across every in-distribution bin and falls "
                "only past the dashed line. d0384 x 5000 (green) is the counter-example that keeps this "
                "from being an ensemble-wide claim: it handles 24 hints at 0.983 but declines INSIDE "
                "distribution, 0.970 at 18 hints and 0.933 at 16.")
    pdf.savefig(fig)


def page_mechanism(pdf, hints, labels):
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5), sharey=True)
    order = [l for l in ("empty_round", "malformed", "wrong_colors", "extra_rounds",
                         "wrong_outcome", "missing_rounds") if l in labels]
    colours = dict(zip(order, plt.cm.tab20(np.linspace(0, 0.95, len(order)))))
    for ax, (key, label, _) in zip(axes, SERIES):
        d = hints[key]
        xs = [x for x in sorted(d, reverse=True) if int(d[x]["n_failed"])]
        bottom = np.zeros(len(xs))
        for lab in order:
            v = np.array([int(d[x][f"n_{lab}"]) for x in xs], float)
            ax.bar(range(len(xs)), v, bottom=bottom, color=colours[lab],
                   label=lab, width=0.72)
            bottom += v
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_title(label, size=MED)
        ax.set_xlabel("hints")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("failures (of 300)")
    axes[0].legend(fontsize=SMALL - 1, loc="upper right")
    fig.suptitle("What the failures actually are", size=MED + 5, weight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    if "empty_round" in labels:
        best24 = hints["d256_data16x"][24]
        n_er, n_f = int(best24["n_empty_round"]), int(best24["n_failed"])
        footer(fig, f"At 24 hints {n_er} of the best model's {n_f} failures are `empty_round`: it opens a "
                    "deduction round on an instance with nothing left to deduce, and the parser raises "
                    "before the round is appended. That is the predicted out-of-distribution signature. "
                    "d0384 instead FILLS the round it opens, which is why its 24-hint failures parse and "
                    "land in `extra_rounds` -- the control showing this is not just relabelling every "
                    "off-distribution failure. Its in-distribution failures are a third thing again: it "
                    "emits FEWER rounds than the true trace (2.48 against 2.90 at 16 hints) and colours "
                    "wrongly, i.e. it truncates the deduction rather than over-running it.")
    else:
        footer(fig, "This is the FIRST run's labelling, which had no `empty_round` category: "
                    "`classify_failure` tested `well_formed` before `extra_rounds`, so the predicted "
                    "signature was charged to `malformed`. Verified separately: splicing one bare "
                    "BEGIN_ROUND into a real rho=1.0 sequence gives `empty BEGIN_ROUND at position 196`. "
                    "Job 49434190 re-runs with the split label to confirm the attribution per instance.")
    pdf.savefig(fig)


def page_pool(pdf, pool):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("Same job, second question: the shared pool at 4x resolution",
                 size=MED + 5, weight="bold", y=0.95)
    ax = fig.add_axes([0.05, 0.47, 0.90, 0.38])
    ax.axis("off")
    # The reference scores itself, so its row carries delta 0.0 and p 1.0
    # rather than a blank -- it is identified by name, not by an empty cell.
    rows = sorted(pool, key=lambda r: -float(r["solve_rate"]))
    body = [["model", "solve_rate", "Wilson 95%", "delta vs\nd256 x 20000", "discordant", "exact p"]]
    for r in rows:
        d = "" if r["checkpoint"] == r["vs_reference"] else r["delta"]
        body.append([
            pretty(r["short"]),
            f"{float(r['solve_rate']):.4f}",
            f"[{float(r['wilson_lo']):.4f}, {float(r['wilson_hi']):.4f}]",
            "reference" if d == "" else f"{float(d):+.4f}",
            "--" if d == "" else f"{r['discordant_won']} won / {r['discordant_lost']} lost",
            "--" if d == "" else f"{float(r['mcnemar_exact_p']):.1e}",
        ])
    t = ax.table(cellText=body[1:], colLabels=body[0], loc="center",
                 cellLoc="center", colWidths=[0.21, 0.13, 0.22, 0.16, 0.19, 0.11])
    t.auto_set_font_size(False)
    t.set_fontsize(MED - 3)
    t.scale(1, 2.8)
    for j in range(6):
        t[(0, j)].set_facecolor("0.88")
        t[(0, j)].set_text_props(weight="bold")
    fig.text(0.5, 0.37,
             "n = 6000 is a strict superset of the 1500-instance pool (instance i is seeded at\n"
             "seed_start + i), so this re-scores the same models on the same eval, 4x deeper.",
             ha="center", va="center", size=MED - 2, linespacing=1.7)
    fig.text(0.5, 0.185,
             "At n_train = 20000, d0128 (4.0M params) and d0384 (16.7M) are still tied with each other\n"
             "(0.9750 vs 0.9740) -- but BOTH are now resolvably below d0256, which n = 1500 could not\n"
             "separate. d_model = 256 is a real optimum at this budget, not a shrug.",
             ha="center", va="center", size=MED - 2, linespacing=1.7, color="0.15")
    footer(fig, "McNemar exact, paired on the identical 6000 instances. The reference model's own rate "
                "moved 0.9847 -> 0.9837 between pools, i.e. the n=1500 estimate was within its interval.")
    pdf.savefig(fig)


def page_versus(pdf, pool):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    names, n1500, n6000 = [], [], []
    prev = {"d256_data16x": 0.9960, "d256_data4x": 0.9847, "d128_data4x": 0.9733,
            "d384_data4x": 0.9720}
    for r in sorted(pool, key=lambda r: -float(r["solve_rate"])):
        if r["short"] not in prev:
            continue
        names.append(pretty(r["short"], sep="\nx "))
        n1500.append(prev[r["short"]])
        n6000.append(float(r["solve_rate"]))
    x = np.arange(len(names))
    ax.bar(x - 0.19, n1500, 0.36, label="previous best estimate, n = 1500", color="0.65")
    ax.bar(x + 0.19, n6000, 0.36, label="this job, n = 6000", color="tab:blue")
    for xi, (a, b) in enumerate(zip(n1500, n6000)):
        ax.text(xi - 0.19, a + 0.0012, f"{a:.4f}", ha="center", size=SMALL - 1)
        ax.text(xi + 0.19, b + 0.0012, f"{b:.4f}", ha="center", size=SMALL - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.96, 1.002)
    ax.set_ylabel("shared-pool solve_rate")
    ax.set_title("Comparison against the previous best on the same eval",
                 size=MED + 5, weight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")   # lower right sits on top of the d384 bars
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    footer(fig, "Every model reproduces its recorded rate to within its Wilson interval, so nothing "
                "drifted; what 6000 instances buy is the power to separate the bottom two from d0256. "
                "The best model remains d0256 x 80000, now 0.9955 [0.9935, 0.9969].")
    pdf.savefig(fig)


def page_next(pdf, relabelled):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("Scorecard and what this changes", size=MED + 7, weight="bold", y=0.95)
    scored = [
        ("(a) in-distribution bins flat and high", "HELD", "1.000, 0/300 failures, all 5 bins"),
        ("(b) cliff appears only above 22 hints", "HELD", "0.860 at 23, 0.060 at 24"),
        ("(c) >=70% of 24-hint failures are\n      the model deducing where there\n      is nothing to deduce",
         "HELD, after\nfixing the\nclassifier" if relabelled else "PENDING\n49434190",
         "280 of 282 (99%) once `empty_round` is\nsplit out; it was charged to `malformed`\nby a test-order bug"
         if relabelled else "charged to `malformed` by a test-order\nbug; parser confirms the mechanism"),
        ("(d) the profile is a property of the\n      ensemble, not of one model", "MISSED",
         "d0384 x 5000 declines in-distribution\n(0.933 at 16 hints) while acing 24"),
    ]
    ax = fig.add_subplot(111)
    ax.axis("off")
    t = ax.table(cellText=[[a, b, c] for a, b, c in scored],
                 colLabels=["pre-registered", "verdict", "evidence"],
                 loc="upper center", cellLoc="left", colWidths=[0.38, 0.15, 0.40])
    t.auto_set_font_size(False)
    t.set_fontsize(MED - 4)
    t.scale(1, 4.4)
    for j in range(3):
        t[(0, j)].set_facecolor("0.88")
        t[(0, j)].set_text_props(weight="bold")
    for i, (_, v, _) in enumerate(scored, start=1):
        t[(i, 1)].set_facecolor("#e8f4e8" if v.startswith("HELD") else
                                "#fff4d8" if v.startswith("PENDING") else "#ffe8e0")
    fig.text(0.5, 0.2,
             "Consequences: `mean_o1` and `solve_rate_k0` are retired as competence metrics -- they\n"
             "are extrapolation probes. Nothing needs retraining. The real open question moves to\n"
             "whether network depth must grow with reasoning depth (job 49433454), and to N = 48,\n"
             "where mean DD depth is 6.10 rather than 4.25 (job 49433453).",
             ha="center", va="center", size=MED - 1, linespacing=1.8, color="0.15")
    footer(fig, "Checkpoints from jobs 48912548 / 48935109 / 48913017, config "
                "d_model x nhead 4 x L12 x ff1024, lr 1e-3, pre-norm, 300 epochs, N=24 d=5 q=4, "
                "rho in [0.3, 0.95], hint_mode=removal.")
    pdf.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    path = RELABELLED if os.path.exists(RELABELLED) else HINTS
    relabelled = path == RELABELLED
    hints = load_hints(path)
    labels = labels_present(hints)
    pool = load_pool()
    print(f"hint profile from {os.path.basename(path)} "
          f"({'relabelled' if relabelled else 'ORIGINAL labelling'}); labels: {labels}")

    pdf_path = os.path.join(OUT, "easy_end_is_ood.pdf")
    with PdfPages(pdf_path) as raw:
        pdf = PdfAndPng(raw, OUT)
        page_title(pdf, relabelled)
        page_arithmetic(pdf, hints)
        page_profile(pdf, hints)
        page_mechanism(pdf, hints, labels)
        page_pool(pdf, pool)
        page_versus(pdf, pool)
        page_next(pdf, relabelled)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {pdf.n} pages) "
          f"+ page1..{pdf.n}.png")


if __name__ == "__main__":
    main()
