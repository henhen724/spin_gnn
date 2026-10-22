"""Slide deck for the 2026-10-22 data-scaling result (Slurm 48912548/48935109/48999377).

Six landscape pages built from the same figures the notebook shows, so the deck
and the notebook cannot drift: the plots are read back from
notebooks/sudoku_data_scaling_2026-10-22/*.png rather than re-drawn.

Writes PNGs of every page to $SLIDES_PNG_DIR when that is set, because nothing
on Sherlock can rasterize a PDF (no pdftoppm, and fitz / pypdfium2 / pdf2image
are all unimportable), so page-by-page checking has to come from matplotlib.
"""
import csv
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

S = os.environ.get("SCRATCH", "/scratch/users/hshunt") + "/spin_gnn"
FIG_DIR = f"{S}/notebooks/sudoku_data_scaling_2026-10-22"
GRID_DIR = f"{S}/sudoku_shared_pool_grid_2026-10-22"
OUT_PDF = f"{GRID_DIR}/data_scaling_slides.pdf"
PNG_DIR = os.environ.get("SLIDES_PNG_DIR")

_page = [0]


def save(pdf, fig):
    _page[0] += 1
    pdf.savefig(fig)
    if PNG_DIR:
        os.makedirs(PNG_DIR, exist_ok=True)
        fig.savefig(f"{PNG_DIR}/page{_page[0]}.png", dpi=110)
    plt.close(fig)


def caption(fig, text):
    fig.text(0.06, 0.145, textwrap.fill(" ".join(text.split()), width=112),
             fontsize=12.5, va="top", linespacing=1.5)


def figure_page(pdf, title, png, text):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.95, title, fontsize=23, fontweight="bold", va="top")
    ax = fig.add_axes((0.06, 0.20, 0.88, 0.68))
    ax.imshow(mpimg.imread(f"{FIG_DIR}/{png}"), rasterized=True)
    ax.axis("off")
    caption(fig, text)
    save(pdf, fig)


def rows():
    with open(f"{GRID_DIR}/agg_paired_full_grid.csv") as fh:
        return list(csv.DictReader(fh))


def label(name):
    parent = name.split("[")[-1].rstrip("]")
    d = 128 if "epoch00300" in name else int(name.split("d0")[1][:3])
    n = {"data4x": 20000, "data16x": 80000}.get(parent.split("_")[-1], 5000)
    return f"d{d} x {n}"


def main():
    data = sorted(rows(), key=lambda r: -float(r["solve_rate"]))
    best = data[0]

    with PdfPages(OUT_PDF) as pdf:
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.07, 0.72, "More training data is still the lever",
                 fontsize=40, fontweight="bold")
        fig.text(0.07, 0.60,
                 f"{label(best['checkpoint'])} solves "
                 f"{float(best['solve_rate']):.4f} of the shared 1500-instance pool,\n"
                 f"up from 0.9847, paired McNemar exact p = "
                 f"{float(best['mcnemar_exact_p']):.4f}.",
                 fontsize=22, va="top", linespacing=1.6)
        fig.text(0.07, 0.30,
                 "Slurm 48912548 (train, 5:33) + 48999377 (eval + probes, 10:27)\n"
                 "spin_gnn, branch sudoku-dd  |  2026-10-22  |  "
                 "metric of record: shared-pool solve_rate, 1500 instances, seed_start 5e10",
                 fontsize=13, va="top", linespacing=1.8, color="0.3")
        save(pdf, fig)

        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.06, 0.95, "Seven checkpoints, one eval pool", fontsize=23,
                 fontweight="bold", va="top")
        ax = fig.add_axes((0.04, 0.26, 0.92, 0.60))
        ax.axis("off")
        cells = [[label(r["checkpoint"]), f"{float(r['solve_rate']):.4f}",
                  f"[{float(r['wilson_lo']):.4f}, {float(r['wilson_hi']):.4f}]",
                  f"{float(r['delta']):+.4f}",
                  f"{float(r['mcnemar_exact_p']):.3g}",
                  f"{int(r['discordant_won'])}/{int(r['discordant_lost'])}"]
                 for r in data]
        t = ax.table(cellText=cells,
                     colLabels=["model", "shared-pool\nsolve_rate", "95% Wilson",
                                "delta vs\nd256 x 20000", "McNemar\nexact p",
                                "discordant\nwon/lost"],
                     cellLoc="center", loc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(13)
        t.scale(1, 2.5)
        for (r, _), cell in t.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold")
            elif r == 1:
                cell.set_facecolor("#e8f1fb")
        caption(fig, "Every row is the same 1500 held-out instances, so the "
                     "comparison is paired: only the instances where two models "
                     "disagree carry information, and the test is an exact "
                     "binomial on those. The six previously scored checkpoints "
                     "reproduced their recorded rates to four decimals, which is "
                     "the reproducibility check for the pool.")
        save(pdf, fig)

        figure_page(pdf, "Data scaling has not saturated", "fig1_data_scaling.png",
                    "Each width improves monotonically with training data and the "
                    "best cell is the largest data budget tried. d0128 x 20000 "
                    "(4.0M parameters) reaches 0.9733, statistically "
                    "indistinguishable from d0384 x 20000 (16.7M, 0.9720) -- at "
                    "this scale the data axis buys more than the width axis.")

        figure_page(pdf, "The frontier probe does not measure competence",
                    "fig2_probes_vs_competence.png",
                    "Spearman against shared-pool solve_rate is +0.96 for "
                    "next_node peak accuracy and +0.25 for frontier_mask peak "
                    "AUROC; frontier_mask against d_model is +0.85. The new best "
                    "model's frontier AUROC FELL (0.916 -> 0.907) while its solve "
                    "rate rose. Frontier representation strength is a function of "
                    "width; next-step prediction is what tracks solving.")

        figure_page(pdf, "What more data costs: the trivial cases",
                    "fig3_easy_end.png",
                    "solve_rate_k0 is the pure-copy case (no uncolored nodes, no "
                    "reasoning needed). It rose 0.24 -> 0.60 from 5000 to 20000 "
                    "instances and then collapsed to 0.07 at 80000, while the full "
                    "range kept rising. More width did the same thing on 10-18, so "
                    "this is not specific to the data axis. These are 100-instance "
                    "own-val bins and therefore noisy, but 0.60 -> 0.07 is not noise.")

        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.06, 0.95, "Pre-registration scorecard and what is next",
                 fontsize=23, fontweight="bold", va="top")
        fig.text(0.06, 0.82,
                 "HELD     solve_rate >= 0.9847 with a positive delta  ->  0.9960, +0.0113, p = 0.0015\n"
                 "HELD     next_node peak accuracy >= 0.96 (80000 run)  ->  0.9895\n"
                 "HELD     frontier AUROC for d0128 x 20000 in [0.855, 0.895]  ->  0.8831\n"
                 "HELD     control: d0128 x 5000 re-probe = 0.863 +/- 0.015  ->  0.8578, no drift\n"
                 "MISSED   predicted range [0.980, 0.995] for the 80000 run  ->  0.9960, just above\n"
                 "MISSED   k0 >= 0.85 for the 80000 run  ->  0.07, opposite direction\n"
                 "MISSED   mean_o1 >= 0.96  ->  0.776",
                 fontsize=14.5, va="top", linespacing=2.0, family="monospace")
        fig.text(0.06, 0.37,
                 "Next: push the data axis again (d0256 x 320000) now that 80000 did not saturate;\n"
                 "find out whether the easy-end collapse is a decoding failure or a representation\n"
                 "failure, since k0 = 0.07 with solve_rate 0.996 is a contradiction worth resolving;\n"
                 "and attribute generation failures by where the first error lands (finding 5).",
                 fontsize=15, va="top", linespacing=1.8)
        save(pdf, fig)

    size = os.path.getsize(OUT_PDF) / 1e6
    print(f"wrote {OUT_PDF} ({size:.3f} MB, {_page[0]} pages)")


if __name__ == "__main__":
    main()
