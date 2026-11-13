#!/usr/bin/env python3
"""Stage W's training-curve figure, written and checked BEFORE the cells land.

WHY NOW. The curve figure is the one page of stage W's deck whose content does
not depend on how W turns out: it shows each 100-epoch cell against the
50-epoch cell at its seed, and that comparison is defined whatever the verdict
is. (They are separate RUNS of one configuration -- a matched seed fixes the
data and the init, not the trajectory, since nothing pins GPU kernel
nondeterminism -- so the pairing is by seed and not a continuation.) Every
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
                                       onset, step_epoch)

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


def clearing_dy(ax, fig, y, dy):
    """A label offset, in points, that does not land the label on a level line.

    The floor and basin lines span the whole axes, so a label placed at a
    curve's right-hand end sits on one of them whenever the curve ends near
    that level -- which, on a figure about which level a cell reached, is
    exactly when it happens. The second read-back of this figure found seed 1's
    "100 ep" printed across the 1.600 floor.

    Doubling the offset clears the line in the SAME direction, which matters:
    the two arms' labels are on opposite sides of their curves, and flipping one
    would put both on one side. Returns dy unchanged when it is already clear,
    so only the colliding facet moves.
    """
    to_disp, to_data = ax.transData.transform, ax.transData.inverted().transform
    scale = fig.dpi / 72.0
    base = to_disp((0.0, y))[1]
    # Half a line of text, in data units, is how close counts as touching.
    half = abs(to_data((0.0, base + 7.0 * scale))[1] - y)
    for mult in (1, 2):
        y_lab = to_data((0.0, base + dy * mult * scale))[1]
        if all(abs(y_lab - lvl) > half for lvl in (FLOOR, BASIN)):
            return dy * mult
    return dy


def fig_training_curves(outdir, fname="curves.png"):
    """One facet per seed: the 100-epoch cell against the 50-epoch one at it.

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
    pending = []
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
            # Labels wait for tight_layout(): clearing_dy() asks the axes where
            # a point offset lands in data units, and that answer changes when
            # the axes is resized.
            pending.append((ax, pts[-1][0], pts[-1][1], name, color, dy))
        # BOTH level lines, because the page's claim is which level a cell
        # reached: with only the floor drawn, "reached the basin" was a sentence
        # the reader had to take on trust from a figure that did not show 1.561.
        ax.axhline(FLOOR, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.axhline(BASIN, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)))
        # The page's title names the epoch of the one visible fall, so the
        # figure has to name it too -- from the same function, so the two can
        # never disagree. The first read-back of this figure is what found the
        # fall at all: the title then said "no step anywhere" over a plot that
        # plainly showed one.
        on = onset(w[seed])
        if on is not None:
            ax.axvline(on["epoch"], color=SERIES[0], linewidth=0.9,
                       linestyle=(0, (2, 3)))
            ax.annotate("epoch %d" % on["epoch"], xy=(on["epoch"], hi),
                        xytext=(4, -2), textcoords="offset points",
                        color=SERIES[0], fontsize=10, ha="left", va="top")
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
    fig.canvas.draw()          # so transData reflects the laid-out axes
    for ax, x, y, name, color, dy in pending:
        label_line(ax, x, y, name, color=color, dy=clearing_dy(ax, fig, y, dy))
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
