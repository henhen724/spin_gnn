"""Shared slide-deck plumbing.

Factored out of make_finite_size_deck.py, which predates this module and still
carries its own copies -- it produced a deck that has already been sent, so it
is left alone rather than re-tested for a cosmetic refactor.
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG, MED, SMALL = 25, 17, 13

_HEAD_OFF = 0.035    # heading baseline -> top of its body text
_FOOTER_TOP = 0.09   # the footer caption starts about here


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


def _check_header(fig, title, subtitle, margin=0.96):
    """Raise if the title or subtitle is wider than the slide.

    `textpage` wraps and bounds-checks the body but draws the header as two
    single centred lines, so a long title silently ran off both edges of the
    page -- caught on the N=96 deck only by reading the rendered PNG. Measured
    rather than estimated from character counts: the two sizes have different
    metrics and the em dashes and digits in these titles are not average
    characters.
    """
    r = fig.canvas.get_renderer()
    w_fig = fig.get_window_extent().width
    for what, text, size, weight in (("title", title, BIG, "bold"),
                                     ("subtitle", subtitle, MED, "normal")):
        probe = fig.text(0.5, 0.5, text, ha="center", size=size, weight=weight)
        frac = probe.get_window_extent(renderer=r).width / w_fig
        probe.remove()
        if frac > margin:
            over = int(len(text) * (1 - margin / frac)) + 1
            raise ValueError(
                f"textpage {what} overflows the slide: {frac:.3f} of the page "
                f"width against a {margin:.2f} limit. Cut about {over} "
                f"characters from {text!r}.")


def textpage(o, title, subtitle, blocks, foot, y0=0.80, gap=0.030, width=100,
             pitch=None):
    """Title + subtitle + coloured verdict blocks.

    Each block advances by the height its body actually wraps to. A fixed
    pitch silently overprints the next heading the moment one block gains a
    line, which is how page 8 of the arms deck came to run into its footer;
    `pitch` survives only for the two decks laid out against it and already
    sent, and is checked the same way. Overflow raises rather than rendering
    an unreadable slide: a deck that cannot be built is a visible failure, an
    overprinted one is not.
    """
    body_size = MED - 4
    line_h = body_size * 1.45 / (9 * 72)   # figure fractions per wrapped line
    wrapped = [textwrap.fill(b, width=width) for _, _, b in blocks]

    tops, bottoms = [], []
    y = y0
    for text in wrapped:
        n = text.count("\n") + 1
        tops.append(y)
        bottoms.append(y - _HEAD_OFF - n * line_h)
        y -= pitch if pitch is not None else _HEAD_OFF + n * line_h + gap
    for i, bottom in enumerate(bottoms):
        limit = tops[i + 1] if i + 1 < len(tops) else _FOOTER_TOP
        if bottom < limit:
            raise ValueError(
                f"textpage overflow on '{title}': block {i} "
                f"({wrapped[i].count(chr(10)) + 1} lines) runs "
                f"{limit - bottom:.4f} into "
                f"{'the next heading' if i + 1 < len(tops) else 'the footer'}. "
                f"Widen `width`, shorten the body, or lower `gap`/`y0`.")

    fig = plt.figure(figsize=(16, 9))
    _check_header(fig, title, subtitle)
    fig.text(0.5, 0.93, title, ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, subtitle, ha="center", size=MED, color="0.3")
    for (head, col, _), text, y in zip(blocks, wrapped, tops):
        fig.text(0.06, y, head, size=MED, weight="bold", color=col)
        fig.text(0.06, y - _HEAD_OFF, text, size=body_size, va="top",
                 linespacing=1.45)
    footer(fig, foot)
    o.savefig(fig)


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
