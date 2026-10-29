"""Shared slide-deck plumbing.

Factored out of make_finite_size_deck.py, which predates this module and still
carries its own copies -- it produced a deck that has already been sent, so it
is left alone rather than re-tested for a cosmetic refactor.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG, MED, SMALL = 25, 17, 13


def set_style():
    plt.rcParams.update({"font.size": MED, "axes.titlesize": MED + 2,
                         "axes.labelsize": MED, "legend.fontsize": SMALL})


def footer(fig, text, width=118):
    """Bottom caption, wrapped by hand -- matplotlib's wrap=True ignores the
    rect passed to tight_layout and spills off the slide."""
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    fig.text(0.5, 0.012, "\n".join(lines), ha="center", va="bottom",
             size=SMALL - 1, color="0.3")


class PdfAndPng:
    """Writes the PDF and a PNG per page.

    There is no PDF rasterizer on this cluster, so the PNGs are the only way
    to actually look at the deck before sending it, and a results notebook
    needs them anyway.
    """

    def __init__(self, pdf, png_dir, total=None):
        self.pdf, self.png_dir, self.n = pdf, png_dir, 0
        self.total = total

    def savefig(self, fig):
        self.n += 1
        stamp = f"{self.n} / {self.total}" if self.total else str(self.n)
        fig.text(0.985, 0.012, stamp, ha="right", va="bottom", size=SMALL - 1,
                 color="0.45")
        self.pdf.savefig(fig)
        fig.savefig(os.path.join(self.png_dir, f"page{self.n}.png"), dpi=110)
        plt.close(fig)
