"""The stage T deck: is the shared block an operator, or a 16-step schedule?

Stage S showed that ONE block applied 16 times does direct deduction (rho_50
0.757, 3/3 seeds, 0.017 past the free line). That leaves the question a tied
cell is the only architecture that can ask: T -- how many times the block is
applied -- is a dial at EVAL time, on weights that are already frozen. If the
block is a deduction step, more applications should be tolerated. If the
network instead learned one 16-step trajectory, T=16 is a peak with both sides
falling away, the recurrence is cosmetic, and "one application = one round" is
a reading of a trained network rather than a property of the architecture.

It is the second reading. T0 is falsified and T1 with it.

NO TRAINING HAPPENED IN THIS STAGE. The checkpoints are stage S's d320 cells,
unchanged; the only variable is T. Every T sees the SAME 400 instances per bin
(seed_start fixed), so the differences are paired per-bin differences.

WHAT THE EARLIER WRITE-UPS GOT THIN. The stage S deck's p17 and STATUS.md both
report T1 as a collapse at T=24. True, but incomplete: the registered easy-end
bar (solve_rate >= 0.90 at realized rho >= 0.85) is missed by at least one seed
at EVERY T >= 12, the trained count included, where the three seeds are 0.895,
0.887 and 0.900 -- so one seed sits exactly ON the bar at T=16 and T1 fails on
unanimity, not on a collapse. The collapse specific to over-iterating is the
step from 0.887 to 0.000 between T=16 and T=24 in seed 0. Both halves are on
the pages below; a deck that reported only the dramatic half would be quoting
the arm that makes the point.

Every number is read through the registered scorer's own functions
(scripts/staget_score.py, committed by register job 51904633 before the first
grid, scored by job 51919538; 51904146 is stage S's scorer run, not this one's
registration). The page titles that assert an ORDERING -- the level bottoming
at the trained count, no T clearing the easy-end bar in all three seeds, no
seed improving past T=20 -- are asserted against the data before anything is
drawn, so a drifted input fails the render instead of shipping a false
sentence. Draft 51939930 is why two of those assertions exist: it rendered
clean and said "never reaches the bar, at any T" and "all three seeds peak at
16", and the figures on its own pages refuted both.

Usage (from a job; see scripts/slurm/staget_writeup_2026-11-11.sbatch):
    python3 -u -m notebooks.make_staget_deck --commit <sha>
"""
import argparse
import os
import sys
from statistics import median

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slides import Deck, MUTED, SERIES, style_axes, label_line, series_colors

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace

from scripts.minarch_analysis import cell_metrics, load_all
from scripts.stagec_score import PLATEAU
from scripts.stagek_score import FREE
from scripts.staget_score import (EASY_BAR, EASY_RHO, NEAR, S1_RHO50,
                                 S1_TRAINABLE, SCAN_T, TRAINED_T, T_GRID,
                                 by_seed_t, paired_delta)

# The standing minimum on this same eval, frozen in stage J and quoted in stage
# S's deck the same way: 16 SEPARATE blocks, same L, same attention-only family.
# Stage T cannot beat it and does not try -- it is here because the brief
# requires every write-up to end against the previous best on the same eval.
J2_RHO50 = 0.580
J2_ACC = 0.4940
J2_TRAINABLE = 2504890

# Which T values get a curve of their own on the overlay page: the censor below,
# the trained count, and the degradation above. All six are in the level table.
SHOWN_T = (4, TRAINED_T, 24)


# ------------------------------------------------------------------ the data

def load_T(boot=2000):
    """Rows, {(seed, T): rows}, {(seed, T): metrics} -- the scorer's grouping."""
    if not os.path.isdir(SCAN_T):
        sys.exit("no stage T run dir at %s -- nothing to write up." % SCAN_T)
    rows = load_all([SCAN_T])
    if not rows:
        sys.exit("stage T run dir is empty -- nothing to write up.")
    for r in rows[:1]:
        assert int(r.get("n_trainable_params") or 0) == S1_TRAINABLE, (
            "these are not stage S's d320 checkpoints", r.get("n_trainable_params"))
        assert int(r.get("n_iters_trained") or TRAINED_T) == TRAINED_T, (
            "checkpoint was not trained at T=16", r.get("n_iters_trained"))
    groups = by_seed_t(rows)
    for rs in groups.values():
        rs.sort(key=lambda r: float(r["realized_rho"]))
    cells = {k: cell_metrics(v, boot) for k, v in groups.items()}
    return rows, groups, cells


def level_table(cells, seeds, ts):
    """The scorer's level, recomputed by its own reader: bound-aware median.

    A T is readable only when MORE THAN HALF its seeds crossed 0.5; otherwise
    the level censors and prints as a censor rather than as a number pulled off
    the grid edge.
    """
    level = {}
    for t in ts:
        ms = [cells[(s, t)] for s in seeds if (s, t) in cells]
        ok = [m for m in ms if m["flag"] == "ok"]
        level[t] = {
            "rho50": median([m["rho50"] for m in ok]) if len(ok) * 2 > len(ms)
                     else float("nan"),
            "crossed": len(ok), "n": len(ms),
            "mean_solve": median([m["mean_solve"] for m in ms]),
            "per_seed": [(s, cells[(s, t)]) for s in seeds if (s, t) in cells],
        }
    return level


def easy_worst(groups, seeds, ts):
    """{T: [worst easy-bin solve_rate, per seed]} -- T1's own statistic.

    The easy end is where dd_mean_T is near 1 and every free cell is at ceiling,
    so it cannot censor: a number here always exists, whatever rho_50 does.
    """
    out = {}
    for t in ts:
        per = []
        for s in seeds:
            rs = [r for r in groups.get((s, t), [])
                  if float(r["realized_rho"]) >= EASY_RHO]
            per.append(min(float(r["solve_rate"]) for r in rs) if rs else float("nan"))
        out[t] = per
    return out


def deep_deltas(groups, seeds):
    """Paired per-bin solve_rate deltas, T=24 minus T=16, by seed.

    paired_delta asserts the two sides saw the same instances (it checks the
    realized densities line up) rather than trusting that they did.
    """
    out = []
    for s in seeds:
        a, b = groups.get((s, 24)), groups.get((s, TRAINED_T))
        if not a or not b:
            continue
        d = sorted(paired_delta(a, b), key=lambda p: p[1])
        out.append((s, [p[1] for p in d], [p[2] for p in d]))
    return out


def mean_peaks(level, seeds, ts):
    """{seed: (argmax T, its mean_solve, the trained count's)} -- page 9's note.

    The MEDIAN over seeds peaks at the trained count (check_claims asserts it),
    and the first draft turned that into "all three seeds peak at 16", which is
    false: seed 2 is flat out to T=20 (0.3152 against 0.3148 at 16). The page
    now reads its sentence off this instead of asserting unanimity by hand.
    """
    out = {}
    for s in seeds:
        per = {t: dict(level[t]["per_seed"])[s]["mean_solve"]
               for t in ts if s in dict(level[t]["per_seed"])}
        arg = max(per, key=lambda t: per[t])
        out[s] = (arg, per[arg], per.get(TRAINED_T, float("nan")))
    return out


def check_claims(level, easy, seeds, ts):
    """The page titles assert an ORDERING. Assert it against the data first.

    Each of these is a sentence on a page below. If a re-render ever reads
    different grids, this fails the render; the alternative is a deck that
    renders clean and says something untrue, which is how stage S shipped two
    wrong pages.
    """
    readable = [t for t in ts if level[t]["rho50"] == level[t]["rho50"]]
    assert TRAINED_T in readable, "T=16 has no readable level; every page assumes it"
    best = min(readable, key=lambda t: level[t]["rho50"])
    assert best == TRAINED_T, (
        "rho_50 does NOT bottom at the trained count -- page titles say it does", best)
    peak = max(ts, key=lambda t: level[t]["mean_solve"])
    assert peak == TRAINED_T, ("mean_solve does not peak at the trained count", peak)
    above = [t for t in ts if t >= TRAINED_T and t in readable]
    assert max(level[t]["rho50"] for t in above) - level[TRAINED_T]["rho50"] > NEAR, (
        "T0 is not falsified on this data -- the deck's whole claim is wrong")
    # p5 claims no T clears the bar in ALL seeds, which is a claim about the
    # worst seed, and separately that one seed sits exactly on the bar at the
    # trained count. Both are asserted; the draft asserted only the first and
    # its title overstated it into "never reaches the bar, at any T".
    worst_each = {t: min(v for v in easy[t] if v == v) for t in ts if any(v == v for v in easy[t])}
    for t in [t for t in ts if t >= 12]:
        assert worst_each[t] < EASY_BAR, (
            "the easy end clears its bar in EVERY seed at T=%d -- p5's title says "
            "no T does" % t, worst_each[t])
    assert abs(max(easy[TRAINED_T]) - EASY_BAR) < 1e-9, (
        "no seed sits exactly on the easy-end bar at the trained count -- p5's "
        "note says one does", easy[TRAINED_T])
    if 20 in ts and 24 in ts:
        deep, last = dict(level[20]["per_seed"]), dict(level[24]["per_seed"])
        for s in seeds:
            assert last[s]["mean_solve"] < deep[s]["mean_solve"], (
                "seed %d improves from T=20 to T=24 -- p9's title says none does" % s)
    assert worst_each[24] < 0.5 <= worst_each[TRAINED_T], (
        "the T=16 -> T=24 step is not a near-miss becoming a collapse", worst_each)
    assert abs(level[TRAINED_T]["rho50"] - S1_RHO50) < 5e-4, (
        "T=16 here is not stage S's frozen 0.757 -- the checkpoints drifted",
        level[TRAINED_T]["rho50"])
    return best, peak


def lvl_txt(v):
    return "%.3f" % v if v == v else "CENSORED"


# ------------------------------------------------------------------- the pages
# Every draw_* is module level so the notebook can import and reuse it: the
# notebook must not be able to disagree with the deck about what a curve shows.

def declutter(ax, ys, min_pts=13.0):
    """Right-hand label heights pushed apart to one line of clearance.

    Lifted unchanged from make_stages_deck, where the first render had two
    labels struck through each other. slides.py does not de-collide, and
    label_line writes at the curve's own end, which is the honest place for it
    and the wrong place when two curves end together. Call after set_ylim: the
    points-to-data conversion reads the transform as it stands.
    """
    inv = ax.transData.inverted()
    px = min_pts * ax.figure.dpi / 72.0
    gap = abs(inv.transform((0.0, px))[1] - inv.transform((0.0, 0.0))[1])
    out = list(ys)
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    for above, i in zip(order, order[1:]):
        if out[above] - out[i] < gap:
            out[i] = out[above] - gap
    return out


def draw_easy(ax, easy, seeds, ts):
    """T1: the worst easy-bin solve_rate against T, one line per seed.

    Per seed and not a median, because the whole content of T1 at T=24 is that
    the seeds disagree violently -- 0.000, 0.550, 0.882 -- and a median of those
    is 0.550, which describes none of them.
    """
    cols = series_colors(len(seeds))
    ends = []
    for col, s in zip(cols, seeds):
        ys = [easy[t][seeds.index(s)] for t in ts]
        ax.plot(ts, ys, "-o", color=col, linewidth=1.8, markersize=6,
                rasterized=True)
        ends.append((ts[-1], ys[-1], "seed %d" % s, col))
    ax.axhline(EASY_BAR, color="0.3", linestyle="--", linewidth=1.2)
    ax.text(ts[0], EASY_BAR + 0.02, "registered bar %.2f" % EASY_BAR, size=11,
            color="0.3")
    ax.axvline(TRAINED_T, color="0.6", linestyle=":", linewidth=1.2)
    ax.text(TRAINED_T - 0.3, 0.30, "trained count", size=11, color="0.5",
            rotation=90, va="center", ha="right")
    ax.set_xticks(list(ts))
    ax.set_xlabel("applications T at eval (weights frozen, trained at 16)")
    ax.set_ylabel("worst easy-bin solve rate (realized rho >= %.2f)" % EASY_RHO)
    ax.set_ylim(-0.03, 1.03)
    for (x, _, lbl, col), yy in zip(ends, declutter(ax, [e[1] for e in ends])):
        label_line(ax, x, yy, lbl, color=col)
    style_axes(ax)


def draw_level(ax, level, ts):
    """rho_50 per T against the free line, censors drawn as censors.

    LOWER IS BETTER. A censored T is plotted at the top of the range and
    labelled, never as a number at the grid edge -- "no bin cleared 0.5" is not
    "rho_50 = 0.95".
    """
    y = np.arange(len(ts))[::-1]
    vals = [level[t]["rho50"] for t in ts]
    cens = [v != v for v in vals]
    cols = [SERIES[0] if t == TRAINED_T else MUTED for t in ts]
    ax.barh(y, [1.0 if c else v for v, c in zip(vals, cens)], height=0.55,
            color=cols, rasterized=True)
    ax.axvline(FREE, color="0.25", linestyle="--", linewidth=1.3)
    ax.text(FREE, len(ts) - 0.3, " free line %.3f" % FREE, size=11, color="0.25",
            va="center")
    for yy, v, c in zip(y, vals, cens):
        ax.text((1.0 if c else v) - 0.008, yy, "CENSORED" if c else "%.3f" % v,
                va="center", ha="right", size=13, color="white", weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(["T=%d%s" % (t, "  (trained)" if t == TRAINED_T else "")
                        for t in ts])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("rho_50  (lower is better)")
    style_axes(ax)
    p = ax.get_position()
    ax.set_position([p.x0 + 0.06, p.y0, p.width - 0.06, p.height])


def draw_mean(ax, level, seeds, ts):
    """mean_solve against T, one line per seed. The peak is the whole claim."""
    cols = series_colors(len(seeds))
    ends = []
    for col, s in zip(cols, seeds):
        ys = [dict(level[t]["per_seed"])[s]["mean_solve"] for t in ts]
        ax.plot(ts, ys, "-o", color=col, linewidth=1.8, markersize=6,
                rasterized=True)
        ends.append((ts[-1], ys[-1], "seed %d" % s, col))
    ax.axvline(TRAINED_T, color="0.6", linestyle=":", linewidth=1.2)
    ax.text(TRAINED_T + 0.3, 0.02, "trained count", size=11, color="0.5")
    ax.set_xticks(list(ts))
    ax.set_xlabel("applications T at eval (weights frozen)")
    ax.set_ylabel("mean solve rate over the 14-point rho grid")
    # Headroom off the GLOBAL maximum, not off the right-hand ends: the ends are
    # T=24's values, which are the lowest on the page, and scaling from them
    # would clip the peak at 16 -- which is the entire claim.
    ax.set_ylim(-0.02, max(m["mean_solve"] for t in ts
                           for _, m in level[t]["per_seed"]) * 1.25)
    for (x, _, lbl, col), yy in zip(ends, declutter(ax, [e[1] for e in ends])):
        label_line(ax, x, yy, lbl, color=col)
    style_axes(ax)


def draw_curves(ax, groups, seeds, ts=SHOWN_T):
    """Solve rate against REALIZED rho, three seeds per T, one colour per T."""
    cols = series_colors(len(ts), highlight=list(ts).index(TRAINED_T))
    ends = []
    for col, t in zip(cols, ts):
        last = None
        for s in seeds:
            rs = groups.get((s, t))
            if not rs:
                continue
            x = [float(r["realized_rho"]) for r in rs]
            yv = [float(r["solve_rate"]) for r in rs]
            ax.plot(x, yv, "-", color=col, linewidth=1.8, alpha=0.85,
                    rasterized=True)
            last = (x[-1], yv[-1])
        if last:
            ends.append((last[0], last[1], "T=%d" % t, col))
    ax.axhline(0.5, color="0.3", linestyle="--", linewidth=1.2)
    ax.text(0.345, 0.52, "rho_50 crossing", size=11, color="0.3")
    ax.set_xlabel("realized hint density rho")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.02, 1.02)
    for (x, _, lbl, col), yy in zip(ends, declutter(ax, [e[1] for e in ends])):
        label_line(ax, x, yy, lbl, color=col)
    style_axes(ax)


def draw_deltas(ax, deltas):
    """T3: paired per-bin delta, T=24 minus T=16, across the whole grid.

    Paired, so this is not two curves subtracted after the fact: each point is
    the same 400 instances scored twice, once at each T.
    """
    cols = series_colors(len(deltas))
    ends = []
    for col, (s, rho, dv) in zip(cols, deltas):
        ax.plot(rho, dv, "-o", color=col, linewidth=1.8, markersize=5,
                rasterized=True)
        ends.append((rho[-1], dv[-1], "seed %d" % s, col))
    ax.axhline(0.0, color="0.25", linewidth=1.0)
    ax.axvspan(0.33, 0.35, color="0.9", zorder=0)
    ax.text(0.34, ax.get_ylim()[1], " the two deep bins T3 registered", size=11,
            color="0.45", va="top")
    ax.set_xlabel("realized hint density rho")
    ax.set_ylabel("solve rate at T=24 minus solve rate at T=16")
    for (x, _, lbl, col), yy in zip(ends, declutter(ax, [e[1] for e in ends])):
        label_line(ax, x, yy, lbl, color=col)
    style_axes(ax)


class PngDeck(Deck):
    """slides.Deck plus a PNG per page.

    There is no PDF rasterizer on this cluster, so the PNGs are the only way to
    LOOK at the deck before shipping, and every page is read back before any
    upload. Twice, on stage S's evidence: the second read-back is the one that
    found two pages whose CLAIMS were wrong rather than their layout.
    """

    def __init__(self, path, png_dir, **kw):
        Deck.__init__(self, path, **kw)
        self.png_dir = png_dir
        os.makedirs(png_dir, exist_ok=True)

    def _save(self, fig):
        fig.savefig(os.path.join(self.png_dir, "page%d.png" % self._n), dpi=110)
        Deck._save(self, fig)

    def savefig(self, fig):
        """The hook a page drawn OUTSIDE the deck hands its figure back through.

        slides.Deck has no public savefig -- its page helpers call _save
        themselves and advance the counter -- so deck_tokenization's pages, which
        build their own figure, need this. Omitting it cost draft 51939561 at
        page 3; the tokenization and trace pages are mandatory in every deck, so
        any deck class here needs it.
        """
        self._n += 1
        self._save(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/users/hshunt/spin_gnn/notebooks/staget")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    pdf_path = args.pdf or os.path.join(args.out, "staget_slides.pdf")
    os.makedirs(args.out, exist_ok=True)

    rows, groups, cells = load_T()
    seeds = sorted({s for s, _ in groups})
    ts = [t for t in T_GRID if any((s, t) in cells for s in seeds)]
    level = level_table(cells, seeds, ts)
    easy = easy_worst(groups, seeds, ts)
    deltas = deep_deltas(groups, seeds)
    best, peak = check_claims(level, easy, seeds, ts)

    base = level[TRAINED_T]
    t24 = level[24]
    gap = abs(t24["rho50"] - base["rho50"])
    worst16 = min(easy[TRAINED_T])
    worst24 = min(easy[24])
    # T2 as the scorer computes it: the smallest T at or below the trained count
    # whose mean_solve is within NEAR of the trained count's.
    tstar = min([t for t in ts if t <= TRAINED_T
                 and abs(level[t]["mean_solve"] - base["mean_solve"]) <= NEAR])

    # Page 9's sentence, read off the data rather than assumed. "All three seeds
    # peak at 16" was the draft's note and it is false by 0.0004 in seed 2 --
    # small, but a page that states it wrongly is still a page stating it wrongly.
    peaks_by_seed = mean_peaks(level, seeds, ts)
    n_at_trained = sum(1 for s in seeds if peaks_by_seed[s][0] == TRAINED_T)
    off = [s for s in seeds if peaks_by_seed[s][0] != TRAINED_T]
    if off:
        arg, hi, at16 = peaks_by_seed[off[0]]
        mean_note = ("Seed %d is flat out to T=%d (%.4f against %.4f). The median "
                     "over seeds peaks at %d. Nothing was trained."
                     % (off[0], arg, hi, at16, TRAINED_T))
    else:
        mean_note = ("All %d seeds peak at %d, as does their median. Nothing was "
                     "trained." % (len(seeds), TRAINED_T))

    # One line, one fig.text, no wrapping: the first draft (51939930) ran ~170
    # chars and clipped "NO TRAINING" off the right edge of page 1. Stage S's
    # line fits at ~128, so this one is held to about that. What it drops --
    # N=48, --shuffle-adj --fall -- is on the tokenization and trace pages.
    prov = ("T 51904636 +51905401 | scored 51919538 | commit %s | frozen d320 "
            "weights, 3 seeds | 18 grids, 14 rho x 400 | no training" % args.commit)

    with PngDeck(pdf_path, args.out, project="spin_gnn stage T",
                 provenance=prov) as d:

        d.title_page(
            "One shared block is a 16-step schedule, not an iterable operator",
            # Hand-broken: title_page's subtitle is one fig.text and does not
            # wrap (stage S's first render ran off the right edge here).
            subtitle="Stage S's frozen d320 weights, applied T = 4..24 times instead\n"
                     "of 16. No training: T is the only thing that changes.")

        d.points_page(
            "A tied cell has a dial no untied cell has",
            ["Applications T are an eval-time knob on FIXED weights, not retraining",
             "An operator tolerates T>16; a 16-step schedule peaks at 16",
             "18 grids, six T values, three seeds, no training at all"])

        # Mandatory in every deck, and taking the CHECKPOINT's instance keys:
        # shuffle_adj and fall select the distribution (finding 43).
        exi = example_instance(n_nodes=48, degree=5, n_colors=4,
                               shuffle_adj=True, fall=True)
        page_tokenization(d, exi)
        page_trace(d, exi, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the "
                                "same distribution stage S trained on, re-scored here "
                                "at six values of T.")

        # T1 FIRST, as registered: it cannot censor, so it is readable whatever
        # rho_50 does, and it outranks T0.
        # NOT "the easy end never reaches the bar": the first draft said that and
        # the figure refutes it -- seed 2 sits exactly ON 0.90 at the trained
        # count. T1's statistic is the WORST seed, so what fails is unanimity.
        d.claim_page(
            "No T clears the easy-end bar in all three seeds",
            lambda ax: draw_easy(ax, easy, seeds, ts),
            note="T1, read first because it cannot censor. At the trained count one "
                 "seed sits exactly on the bar and the worst misses by %.4f."
                 % (EASY_BAR - worst16))

        d.table_page(
            "At T=24 the near-miss becomes a collapse: one seed hits zero",
            ["T", "worst easy bin, per seed", "worst", "vs bar %.2f" % EASY_BAR],
            [["T=%d%s" % (t, " (trained)" if t == TRAINED_T else ""),
              " / ".join("%.3f" % v for v in easy[t]),
              "%.4f" % min(easy[t]), "%+.4f" % (min(easy[t]) - EASY_BAR)]
             for t in ts],
            highlight_row=ts.index(TRAINED_T),
            note="Per seed, not a median: at T=24 the three seeds are 0.000, 0.550 and "
                 "0.882, and their median describes none of them.")

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
            note="CENSORED means no density bin cleared a 0.5 solve rate, so there is "
                 "no rho_50 to quote. Lower rho_50 is better.")

        d.claim_page(
            "No T beats the trained count, and T=24 is %.3f worse" % gap,
            lambda ax: draw_level(ax, level, ts),
            note="T0 was registered to hold if every T >= 16 landed within %.2f of the "
                 "best. The spread above 16 is %.3f." % (NEAR, gap))

        d.claim_page(
            "No seed improves past T=20; %d of %d peak at %d"
            % (n_at_trained, len(seeds), TRAINED_T),
            lambda ax: draw_mean(ax, level, seeds, ts),
            note=mean_note)

        d.claim_page(
            "Too few applications censor; too many degrade the same curve",
            lambda ax: draw_curves(ax, groups, seeds),
            note="Three seeds per T. At T=4 no density reaches the dashed line at all; "
                 "at T=24 one seed is flat on the floor.")

        d.points_page(
            "T0 falsified: the trained count is a peak, not a floor",
            ["|T24 - T16| is %.3f against the registered %.2f bar" % (gap, NEAR),
             "Both directions fall away; the median mean solve peaks there",
             "A clean negative, registered in advance as equally publishable"])

        d.points_page(
            "All sixteen applications are used: none is padding",
            ["T* = %d, the smallest T within %.2f of T=16's mean solve" % (tstar, NEAR),
             "T=12 already costs %.4f of mean solve against T=16"
             % (base["mean_solve"] - level[12]["mean_solve"]),
             "So the analysable object does not shrink below sixteen rounds"])

        d.claim_page(
            "Over-iterating costs at the easy end, not where depth should pay",
            lambda ax: draw_deltas(ax, deltas),
            note="T3 registered a positive paired delta at the two lowest-rho bins. It "
                 "is +-0.005 there in 1 of 3 seeds: nothing.")

        d.points_page(
            "Stage T speaks only for d320's frozen weights at N=48",
            ["One width, one trained count: no tied cell trained at another T",
             "Fixed weights, so the only spread is the three trained seeds",
             "N=48 only; the N=96 bars are a different size (finding 55)"])

        d.table_page(
            "Against the previous best, the T sweep changes nothing: 0.757 stands",
            ["quantity", "J2 untied d192 (standing)", "S1 tied d320, best T"],
            [["rho_50 median", "%.3f" % J2_RHO50, lvl_txt(base["rho50"])],
             ["best T over the sweep", "n/a (depth is its weights)", "T=%d" % best],
             ["mean solve", "--", "%.4f" % base["mean_solve"]],
             ["trainable", "{:,}".format(J2_TRAINABLE),
              "{:,}".format(S1_TRAINABLE)],
             ["distinct blocks", "16", "1"],
             ["seeds crossing 0.5", "3/3", "%d/%d" % (base["crossed"], base["n"])]],
            highlight_row=0,
            note="Same 14-point grid, same 400 instances per bin, same --shuffle-adj "
                 "--fall. No improvement is claimed: no T beats 0.757.")

        d.points_page(
            "The next cut is k distinct blocks, and it is already running",
            ["Stage U: k blocks cycled T/k times, still one iterated map",
             "Array 51922710, six cells, k in {2,4} by three seeds",
             "If no k clears %.3f the next cut is optimization, not capacity" % FREE])

    print("wrote %s (%d KB, %d pages)" % (pdf_path, os.path.getsize(pdf_path) // 1024,
                                          d._n))
    print("PNGs at %s/page1..%d.png -- READ EVERY ONE before shipping"
          % (args.out, d._n))
    print("level: " + "  ".join("T%d %s(%d/%d)" % (t, lvl_txt(level[t]["rho50"]),
                                                   level[t]["crossed"], level[t]["n"])
                                for t in ts))
    print("easy end, worst per T: " + "  ".join(
        "T%d %.4f" % (t, min(easy[t])) for t in ts))
    print("best T %d (rho_50 %.3f), mean_solve peak T %d, T* %d, |T24-T16| %.3f"
          % (best, level[best]["rho50"], peak, tstar, gap))
    print("T=16 easy-end worst %.4f (bar %.2f, miss %.4f); T=24 worst %.4f"
          % (worst16, EASY_BAR, EASY_BAR - worst16, worst24))


if __name__ == "__main__":
    main()
