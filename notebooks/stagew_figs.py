#!/usr/bin/env python3
"""Stage W's training-curve figure, written and checked BEFORE the cells land.

WHY NOW. The curve figure is the one page of stage W's deck whose content does
not depend on how W turns out: it shows each 100-epoch cell against its own
50-epoch control, and that comparison is defined whatever the verdict is. Every
other results page needs W-a's answer first. Across stages S-V the deck
read-backs caught seventeen defects, four of them in figures that rendered
clean, so rendering and reading this one a day early removes that risk from
tonight rather than discovering it at 21:00.

RE-DERIVES NOTHING. The log parse, the step definition and the three NLL levels
all come from scripts.step_epoch_census, which is the module that registered
them and that was validated against two independent ground truths (the
control's hand-read step at 21-24, and finding 64's two basin seeds). A second
copy of that regex here is exactly how two numbers in one deck come to disagree.

SEEDS ARE MATCHED BY PARSED SEED, never by array task index. The control array
and W's array were submitted months apart with different task orderings, and
pairing a 100-epoch cell against the wrong control would invent a difference
out of nothing.
"""

import os
import sys

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
sys.path.insert(0, "/scratch/users/hshunt/spin_gnn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slides import MUTED, SERIES, label_line, style_axes
from scripts.step_epoch_census import (BASIN, FLOOR, ROOT, config_of, history,
                                       step_epoch)

W_LOGS = "sud2N48BudgetW_52053221_%d.out"
CONTROL_LOGS = "sud2N48TiedWidthS_51853969_%d.out"


def by_seed(pattern):
    """{seed: [(epoch, train_nll, val_acc), ...]} for one array's three tasks."""
    out = {}
    for task in (0, 1, 2):
        path = os.path.join(ROOT, pattern % task)
        if not os.path.exists(path):
            continue
        cfg = config_of(path)
        seed = cfg.get("seed")
        if seed is None:
            raise AssertionError("no seed= line in %s; cannot pair it" % path)
        if seed in out:
            raise AssertionError("two tasks claim seed %d in %s" % (seed, pattern))
        out[seed] = history(path)
    return out


def fig_training_curves(outdir, fname="curves.png"):
    """One facet per seed: the 100-epoch cell against its own 50-epoch control.

    Three seeds would be three series in one axes, which the palette refuses
    above three and which would then have to carry two arms as well. Faceting is
    what ~/CLAUDE.md asks for here, and it also puts each pair on its own
    y-range, where a 0.06 gap is visible at all.
    """
    w, c = by_seed(W_LOGS), by_seed(CONTROL_LOGS)
    seeds = sorted(set(w) & set(c))
    if not seeds:
        raise AssertionError("no seed appears in both arrays -- check the log names")

    # The first read-back of this figure drew epoch 0 onwards, and the initial
    # collapse (2.6 -> 1.6 in five epochs) then owned the whole y-range: the
    # 0.01-0.06 differences the page is actually about sat inside the bottom
    # tenth of the axes, invisible. Start at the same MIN_EP the step census
    # scans from -- the project's own registered boundary for "after the
    # collapse" -- and let the y-range follow the data that remains.
    start = 8
    kept = [n for seed in seeds for src in (w[seed], c[seed])
            for e, n, _ in src if e >= start]
    lo, hi = min(kept), max(kept)
    pad = 0.06 * (hi - lo)

    fig, axes = plt.subplots(1, len(seeds), figsize=(13, 4.2), sharey=True)
    axes = [axes] if len(seeds) == 1 else list(axes)
    for ax, seed in zip(axes, seeds):
        # dy splits the two labels: both arms end near the same NLL on the
        # healthy seeds, and the first read-back had them printed on top of
        # each other. The offset has to FOLLOW THE DATA -- a fixed assignment
        # pushed them together again on seed 0, where the control ends above
        # the 100-epoch cell rather than below it.
        c_end, w_end = c[seed][-1][1], w[seed][-1][1]
        c_dy = 11 if c_end >= w_end else -11
        for hist, color, name, dy in ((c[seed], MUTED, "50 ep", c_dy),
                                      (w[seed], SERIES[0], "100 ep", -c_dy)):
            pts = [(e, n) for e, n, _ in hist if e >= start]
            ax.plot([e for e, _ in pts], [n for _, n in pts],
                    color=color, linewidth=1.8, rasterized=True)
            label_line(ax, pts[-1][0], pts[-1][1], name, color=color, dy=dy)
        # BOTH level lines, because the page's claim is which level a cell
        # reached: with only the floor drawn, "reached the basin" was a sentence
        # the reader had to take on trust from a figure that did not show 1.561.
        ax.axhline(FLOOR, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.axhline(BASIN, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)))
        style_axes(ax, xlabel="epoch", ylabel="train NLL" if ax is axes[0] else None)
        ax.set_title("seed %d" % seed, color=MUTED, fontsize=13)
        ax.set_xlim(start, 128)
        ax.set_ylim(lo - pad, hi + pad)
    # In the empty right half of the first facet, where no curve runs. The two
    # labels sit on opposite sides of their lines so neither reads as belonging
    # to the other: 1.600 above its line, 1.561 below its own.
    axes[0].annotate("floor %.3f" % FLOOR, xy=(112, FLOOR), xytext=(0, 5),
                     textcoords="offset points", color=MUTED, fontsize=11,
                     ha="right")
    axes[0].annotate("basin %.3f" % BASIN, xy=(112, BASIN), xytext=(0, -14),
                     textcoords="offset points", color=MUTED, fontsize=11,
                     ha="right")
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def describe():
    """What the figure currently shows, as text, so the page body can be checked
    against it rather than written from memory."""
    w, c = by_seed(W_LOGS), by_seed(CONTROL_LOGS)
    lines = []
    for seed in sorted(set(w) & set(c)):
        wh, ch = w[seed], c[seed]
        ws, cs = step_epoch(wh), step_epoch(ch)
        lines.append(
            "seed %d: W at epoch %d nll %.4f acc %.4f step %s | control final "
            "epoch %d nll %.4f acc %.4f step %s | delta %+.4f"
            % (seed, wh[-1][0], wh[-1][1], wh[-1][2],
               ws if ws is not None else "none",
               ch[-1][0], ch[-1][1], ch[-1][2],
               cs if cs is not None else "none",
               wh[-1][1] - ch[-1][1]))
    return lines


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    print("levels: basin <= %.3f, floor <= %.3f" % (BASIN, FLOOR))
    for line in describe():
        print(line)
    print("wrote", fig_training_curves(outdir))
