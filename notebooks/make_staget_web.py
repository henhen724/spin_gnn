"""Stage T as a password-gated WEB deck: ~/slides/decks/spin-gnn/deck.md.

~/CLAUDE.md changed under this project: the default delivery is now a web deck at
https://slides-hshunt.pages.dev/<project>/ and a PDF is ask-only. Stage T shipped
as a PDF to OneDrive before that was noticed, so this builds the same reading as
the deck hshunt is now meant to open from a link.

NOTHING IS RE-DERIVED HERE. Every number and every figure comes from
notebooks.make_staget_deck, which reads them through the registered scorer's own
functions (scripts/staget_score.py, committed by job 51904633 BEFORE the first
grid, scored by 51919538). check_claims() runs here too, so this builder fails
closed on exactly the sentences the PDF's read-back corrected:

  p5  "the easy end never reaches the 0.90 bar, at any T" -- refuted by its own
      figure: seed 2 sits exactly ON 0.900 at the trained count, so what fails is
      UNANIMITY, not the bar.
  p9  "all three seeds peak at 16" -- seed 2 peaks at T=20, by 0.0004.

The page titles and notes are the PDF's, and they are run through the shared
library's own _check_claim/_check_note/_check_points so the web deck cannot drift
past the same 12-word title and 25-word body caps the PDF is held to.

Two deliberate differences from the PDF:

  * the comparison against the previous best is LAST. ~/CLAUDE.md says to end on
    it; the PDF ended on "the next cut is already running" instead.
  * the tokenization and trace pages get finding-sentence headings of their own,
    because the figures already carry their in-figure titles and repeating those
    as the slide heading wastes the one line that is supposed to make the claim.

Usage (in a job -- $HOME I/O on sh02-ln02 wedges, and this imports torch):
    python3 -u -m notebooks.make_staget_web --commit <sha> [--out <dir>]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

# The caps the PDF is held to, enforced here by the same functions rather than by
# a second implementation that could disagree about what "12 words" means.
from slides import _check_claim, _check_note, _check_points

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_deck import (
    EASY_BAR, FREE, J2_RHO50, J2_TRAINABLE, NEAR, S1_TRAINABLE, TRAINED_T,
    check_claims, deep_deltas, draw_curves, draw_deltas, draw_easy, draw_level,
    draw_mean, easy_worst, level_table, load_T, lvl_txt, mean_peaks)

DECK = "/home/users/hshunt/slides/decks/spin-gnn"


class WebDeck:
    """deck.md + figures/, with the shared library's claim and note caps.

    Deliberately NOT a subclass of Deck: Deck owns a PdfPages and draws its own
    titles into the figure, and here the title is markdown and the figure is only
    the plot. Sharing the page API is what matters, so that the page list below
    reads the same as make_staget_deck's and the two can be diffed by eye.
    """

    def __init__(self, out_dir, title, author, date, provenance):
        self.dir = out_dir
        self.fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)
        self.n = 0
        self._pending = None
        self.parts = []
        self.front = ("---\ntitle: %s\nauthor: %s\ndate: %s\n---" % (title, author, date))
        self.provenance = provenance

    # -- plumbing
    def _add(self, body):
        self.n += 1
        self.parts.append(body.rstrip())

    def _fig_md(self, name, claim, note):
        md = "## %s\n\n![](figures/%s.png)" % (_check_claim(claim), name)
        if note:
            md += "\n\n%s" % _check_note(note)
        return md

    def _save(self, fig, name):
        path = os.path.join(self.fig_dir, "%s.png" % name)
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return path

    # -- page types
    def title_page(self, claim, subtitle=None):
        body = "# %s" % _check_claim(claim)
        if subtitle:
            body += "\n\n%s" % subtitle.replace("\n", " ")
        # Provenance is provenance, not body text: it is the one line on the deck
        # exempt from the 25-word cap, as it is in the PDF's title_page.
        body += "\n\n*%s*" % self.provenance
        self._add(body)

    def claim_page(self, claim, draw, name, note=None):
        # 11x6 at dpi 150 is 1650x900 into colloquium's 1280x720 stock theme, so
        # the plot downsamples rather than being upscaled. The right margin is
        # kept clear because label_line() puts its direct labels there.
        fig = plt.figure(figsize=(11.0, 6.0))
        ax = fig.add_axes([0.07, 0.11, 0.78, 0.84])
        draw(ax)
        ax.set_rasterized(True)
        self._save(fig, name)
        self._add(self._fig_md(name, claim, note))

    def table_page(self, claim, headers, rows, note=None, highlight_row=None):
        out = ["## %s" % _check_claim(claim), ""]
        out.append("| " + " | ".join(str(h) for h in headers) + " |")
        out.append("|" + "---|" * len(headers))
        for r, row in enumerate(rows):
            cells = [str(c) for c in row]
            if highlight_row == r:
                cells = ["**%s**" % c for c in cells]
            out.append("| " + " | ".join(cells) + " |")
        if note:
            out += ["", _check_note(note)]
        self._add("\n".join(out))

    def points_page(self, claim, points):
        points = _check_points(points)
        self._add("## %s\n\n" % _check_claim(claim)
                  + "\n".join("- %s" % p for p in points))

    # -- the hook page_tokenization/page_trace call instead of a PdfPages
    def expect(self, name, claim, note=None):
        """Name the next figure those two helpers will hand to savefig()."""
        self._pending = (name, claim, note)

    def savefig(self, fig):
        if self._pending is None:
            raise RuntimeError("savefig() with no expect() -- which page is this?")
        name, claim, note = self._pending
        self._pending = None
        self._save(fig, name)
        self._add(self._fig_md(name, claim, note))

    def write(self):
        path = os.path.join(self.dir, "deck.md")
        with open(path, "w") as fh:
            fh.write(self.front + "\n\n" + "\n\n---\n\n".join(self.parts) + "\n")
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    rows, groups, cells = load_T()
    seeds = sorted({s for s, _ in groups})
    ts = [t for t in sorted({t for _, t in groups}) if any((s, t) in cells for s in seeds)]
    level = level_table(cells, seeds, ts)
    easy = easy_worst(groups, seeds, ts)
    deltas = deep_deltas(groups, seeds)
    # The same assertions the PDF ships behind, re-run: if a grid were ever
    # re-scored and a number moved, this builder stops instead of publishing a
    # sentence the data no longer supports -- to a URL, which is worse than a PDF.
    best, peak = check_claims(level, easy, seeds, ts)

    base = level[TRAINED_T]
    gap = abs(level[24]["rho50"] - base["rho50"])
    worst16 = min(easy[TRAINED_T])
    tstar = min([t for t in ts if t <= TRAINED_T
                 and abs(level[t]["mean_solve"] - base["mean_solve"]) <= NEAR])

    peaks_by_seed = mean_peaks(level, seeds, ts)
    n_at_trained = sum(1 for s in seeds if peaks_by_seed[s][0] == TRAINED_T)
    off = [s for s in seeds if peaks_by_seed[s][0] != TRAINED_T]
    if off:
        arg, hi, at16 = peaks_by_seed[off[0]]
        mean_note = ("Seed %d is flat to T=%d (%.4f against %.4f); the median peaks "
                     "at %d." % (off[0], arg, hi, at16, TRAINED_T))
    else:
        mean_note = "All %d seeds peak at %d, as does their median." % (len(seeds),
                                                                       TRAINED_T)

    prov = ("Arrays 51904636 +51905401, scored 51919538, commit %s. Frozen d320 "
            "weights, 3 seeds, 18 grids of 14 rho x 400 instances. No training."
            % args.commit)

    d = WebDeck(args.out, "One shared block is a 16-step schedule, not an operator",
                "hshunt", "2026-11-11", prov)

    d.title_page(
        "One shared block is a 16-step schedule, not an operator",
        subtitle="Stage S's frozen d320 weights, applied T = 4..24 times instead of "
                 "16. No training: T is the only thing that changes.")

    d.points_page(
        "A tied cell has a dial no untied cell has",
        ["Applications T are an eval-time knob on FIXED weights",
         "An operator tolerates T>16; a 16-step schedule peaks at 16",
         "18 grids, six T values, three seeds, no training"])

    exi = example_instance(n_nodes=48, degree=5, n_colors=4,
                           shuffle_adj=True, fall=True)
    d.expect("tokenization",
             "Every token is one categorical id in a %d-symbol vocabulary"
             % exi.vocab.size)
    page_tokenization(d, exi)
    d.expect("trace",
             "A deduction fires when a node sees all three other colours",
             note="The same distribution stage S trained on, re-scored at six values "
                  "of T.")
    page_trace(d, exi)

    # T1 first, as registered: it cannot censor, so it is readable whatever
    # rho_50 does, and it outranks T0.
    d.claim_page(
        "No T clears the easy-end bar in all three seeds",
        lambda ax: draw_easy(ax, easy, seeds, ts), "easy",
        note="At the trained count one seed sits exactly on the bar; the worst "
             "misses by %.4f." % (EASY_BAR - worst16))

    d.table_page(
        "At T=24 the near-miss becomes a collapse: one seed hits zero",
        ["T", "worst easy bin, per seed", "worst", "vs bar %.2f" % EASY_BAR],
        [["T=%d%s" % (t, " (trained)" if t == TRAINED_T else ""),
          " / ".join("%.3f" % v for v in easy[t]),
          "%.4f" % min(easy[t]), "%+.4f" % (min(easy[t]) - EASY_BAR)]
         for t in ts],
        highlight_row=ts.index(TRAINED_T),
        note="Per seed, not a median: at T=24 the three seeds are 0.000, 0.550 and "
             "0.882.")

    d.table_page(
        "rho_50 bottoms exactly at the trained count, and censors below T=12",
        ["T", "rho_50", "seeds crossing", "mean solve", "vs T=16"],
        [["T=%d%s" % (t, " (trained)" if t == TRAINED_T else ""),
          lvl_txt(level[t]["rho50"]),
          "%d/%d" % (level[t]["crossed"], level[t]["n"]),
          "%.4f" % level[t]["mean_solve"],
          "--" if t == TRAINED_T
          else "%+.4f" % (level[t]["mean_solve"] - base["mean_solve"])]
         for t in ts],
        highlight_row=ts.index(TRAINED_T),
        note="CENSORED means no density bin cleared a 0.5 solve rate. Lower rho_50 "
             "is better.")

    d.claim_page(
        "No T beats the trained count, and T=24 is %.3f worse" % gap,
        lambda ax: draw_level(ax, level, ts), "level",
        note="T0 required every T >= 16 within %.2f of the best. The spread above 16 "
             "is %.3f." % (NEAR, gap))

    d.claim_page(
        "No seed improves past T=20; %d of %d peak at %d"
        % (n_at_trained, len(seeds), TRAINED_T),
        lambda ax: draw_mean(ax, level, seeds, ts), "mean", note=mean_note)

    d.claim_page(
        "Too few applications censor; too many degrade the same curve",
        lambda ax: draw_curves(ax, groups, seeds), "curves",
        note="At T=4 no density reaches the dashed line; at T=24 one seed is flat on "
             "the floor.")

    d.points_page(
        "T0 falsified: the trained count is a peak, not a floor",
        ["|T24 - T16| is %.3f against the registered %.2f bar" % (gap, NEAR),
         "Both directions fall away from the trained count",
         "A clean negative, registered as equally publishable"])

    d.points_page(
        "All sixteen applications are used: none is padding",
         # NOT "T* = 16, within 0.01 of T=16's mean solve", which the first render
         # printed: T* is DEFINED as the smallest T within NEAR of the trained
         # count, so that sentence says only that 16 is within 0.01 of itself.
        ["T* = %d: the smallest T within %.2f of the trained count" % (tstar, NEAR),
         "T=12 already costs %.4f of mean solve"
         % (base["mean_solve"] - level[12]["mean_solve"]),
         "So the analysable object does not shrink below sixteen rounds"])

    d.claim_page(
        "Over-iterating costs at the easy end, not where depth should pay",
        lambda ax: draw_deltas(ax, deltas), "deltas",
        note="T3 required a positive paired delta at the two lowest-rho bins. It is "
             "+-0.005 in 1 of 3 seeds.")

    d.points_page(
        "Stage T speaks only for d320's frozen weights at N=48",
        ["One width, one trained count, no other tied cell",
         "Fixed weights, so the only spread is three trained seeds",
         "N=48 only; the N=96 bars are a different size"])

    d.points_page(
        "The next cut is k distinct blocks, and it is already running",
        ["Stage U: k blocks cycled T/k times, still one iterated map",
         "Array 51922710, six cells, k in {2,4} by three seeds",
         "If no k clears %.3f the next cut is optimization" % FREE])

    # Last, as ~/CLAUDE.md requires. The PDF ended on the next cut instead.
    d.table_page(
        "Against the previous best, the T sweep changes nothing: 0.757 stands",
        ["quantity", "J2 untied d192 (standing)", "S1 tied d320, best T"],
        [["rho_50 median", "%.3f" % J2_RHO50, lvl_txt(base["rho50"])],
         ["best T over the sweep", "n/a (depth is its weights)", "T=%d" % best],
         ["mean solve", "--", "%.4f" % base["mean_solve"]],
         ["trainable", "{:,}".format(J2_TRAINABLE), "{:,}".format(S1_TRAINABLE)],
         ["distinct blocks", "16", "1"],
         ["seeds crossing 0.5", "3/3", "%d/%d" % (base["crossed"], base["n"])]],
        highlight_row=0,
        note="Same grid, same 400 instances per bin, same --shuffle-adj --fall. No "
             "improvement is claimed.")

    path = d.write()
    print("wrote %s (%d pages, %d figures)"
          % (path, d.n, len(os.listdir(d.fig_dir))))
    print("figures: %s" % ", ".join(sorted(os.listdir(d.fig_dir))))
    print("level: " + "  ".join("T%d %s(%d/%d)" % (t, lvl_txt(level[t]["rho50"]),
                                                   level[t]["crossed"], level[t]["n"])
                                for t in ts))
    print("best T %d, mean_solve peak T %d, T* %d, |T24-T16| %.3f"
          % (best, peak, tstar, gap))


if __name__ == "__main__":
    main()
