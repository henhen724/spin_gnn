"""Stage N's deck: does KB (`no_mlp L=16 d128`, N=48) replicate on fresh seeds?

WRITTEN BEFORE THE SCORE EXISTED, AND THAT IS SAFE ONLY BECAUSE NOTHING HERE IS
A SENTENCE CHOSEN BY HAND. Every page title and every verdict block is computed
from the cells on disk through the same functions scripts/stagen_score.py uses --
`qualifies6` below is the registered pooling rule transcribed, not re-decided.
The failure mode this guards against is a deck that states a conclusion its own
figures contradict, which is what the page-by-page PNG read-back exists to catch.

Run it with fewer than six seeds and it refuses to write the shipping PDF; pass
--allow-partial to render a clearly-named dry-run deck instead. That is how this
file was tested on the day it was written, against stage K's three seeds alone.

Pages, in order, each titled with its finding:
  1 the pooled call against the standing minimum          (N2, the headline)
  2 did the level and the crossing replicate              (N0, N1)
  3 the transition curves all six seeds were scored on
  4 the plateau                                           (N3)
  5 validation token accuracy, read before rho_50         (N4)
  6 how an instance is tokenized                          (mandatory)
  7 one real reasoning trace                              (mandatory)
  8 what this stage settles and what it does not
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import (BIG, MED, SMALL, PdfAndPng, _check_header,
                                  footer, set_style, textpage)
from notebooks.make_stagehi_deck import acc_of, collect, curve
from scripts.minarch_analysis import SCAN_J
from scripts.stagec_score import (PLATEAU, S_RHO50, escapes, rho_txt, rung,
                                  summarize)
from scripts.stagej_score import LOO_BAR
from scripts.stagek_score import FREE, SCAN_K
from scripts.stagen_score import (J2, K_SEEDS, KB, KB_K_ESCAPES, KB_K_RHO50,
                                  KB_TRAINABLE, N_SEEDS, PRED_HI, PRED_LO,
                                  SCAN_N, STANDING_RHO50, STANDING_TRAINABLE)

K_C = "#b06000"        # stage K's original triple
N_C = "#13507a"        # stage N's fresh triple
BAND_C = "#c1272d"     # the free line, the plateau, a falsified prediction
OK_C = "#1a7a3c"       # a held prediction, and the standing minimum
DEAD_C = "0.45"        # censored / never escaped
NOTE_C = "#6a3d9a"     # registered in advance

GRID_TOP = 0.95

# One geometry for all four plot pages: figure, then the plot box, then the
# registered-call column to its right. The column used to start at 0.74 and ran
# off the slide -- caught by reading page2.png, which is what the read-back is
# for -- so the plot is narrower than it looks like it could be, on purpose.
PLOT_BOX = [0.07, 0.17, 0.55, 0.58]
COL_X, COL_TOP = 0.655, 0.73


def header(fig, title, subtitle):
    """Title and subtitle for a plot page, width-checked like textpage's.

    Pages 2-5 computed a title and never drew it: the verdict was in the
    variable and not on the slide. Routing every plot page through one helper
    is what stops that from coming back one page at a time.
    """
    _check_header(fig, title, subtitle)
    fig.text(0.5, 0.945, title, ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, subtitle, ha="center", size=MED - 3, color="0.3")


def column(fig, lines):
    """The fixed-width verdict column down the right-hand side."""
    fig.text(COL_X, COL_TOP, "\n".join(lines), va="top", size=SMALL - 2,
             family="monospace", linespacing=1.7)


def end_labels(ax, ends, close):
    """Label each curve at its right-hand end, nudged apart where they coincide.

    `ends` is [(x, y, colour, label)]; `close` is the y-distance in data units
    below which two labels would print on top of each other. Six seeds of one
    architecture land within a percent of each other at the top bin, so without
    this the labels stack into an unreadable smudge -- seen on page3.png.
    """
    ordered = sorted(ends, key=lambda e: -e[1])
    for i, (x, y, color, lab) in enumerate(ordered):
        dy = -13 * sum(1 for y2 in (e[1] for e in ordered[:i]) if abs(y2 - y) < close)
        ax.annotate(lab, (x, y), xytext=(7, dy), textcoords="offset points",
                    color=color, size=SMALL, va="center")


# --------------------------------------------------------------------------
# The registered calls. These encode stagen_score.py's falsifiers and nothing
# else; if the two ever disagree, the scorer is right and this file is wrong.
# --------------------------------------------------------------------------
def qualifies6(s):
    """N2's pooling rule, fixed at registration: 6 of 6 cross AND median <= FREE.

    NOT stage K's 3-of-3 rule. Pooling six seeds and then applying a
    three-seed rule is exactly the slip the registration was written to stop.
    """
    return (s["n"] == 6 and s["n_cross"] == 6 and s["rho50_readable"]
            and s["rho50"] <= FREE)


def verdict_n1(fresh):
    """Did the crossing replicate?"""
    if not fresh or fresh["s"]["n"] < 3:
        return "pending"
    return "held" if fresh["s"]["n_cross"] == 3 else "falsified"


def verdict_n0(fresh):
    """Did the level replicate, and in which direction did it miss?"""
    if not fresh or not fresh["s"]["rho50_readable"]:
        return "pending"
    r = fresh["s"]["rho50"]
    if PRED_LO <= r <= PRED_HI:
        return "held"
    return "low" if r < PRED_LO else "high"


def verdict_n3(pooled):
    if not pooled or pooled["s"]["n"] < 6:
        return "pending"
    return "held" if pooled["esc"] >= 4 else "falsified"


def rows_for(cells, key, seeds=None):
    """One rung's seeds, with everything the pages need, or None."""
    ms = rung(cells, key)
    if not ms:
        return None
    if seeds is not None:
        ms = [m for m in ms if m["seed"] in seeds]
        if not ms:
            return None
    s = summarize(ms)
    esc, nll = escapes(ms)
    return dict(ms=sorted(ms, key=lambda r: r["seed"]), s=s, esc=esc, nll=nll,
                acc=acc_of(ms))


def seedmap(cells):
    """cell_key carries the seed last; surface it as a field the pages can read."""
    for k, v in cells.items():
        v.setdefault("seed", k[-1])
    return cells


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_headline(o, pooled, fresh, old, j2, commit):
    s = pooled["s"]
    took = qualifies6(s)
    shrink = STANDING_TRAINABLE / KB_TRAINABLE

    if took:
        title = (f"Attention-only L=16 d128 takes the minimum at "
                 f"{KB_TRAINABLE / 1e6:.2f}M parameters")
    elif s["n"] < 6:
        title = f"Only {s['n']} of 6 seeds have landed, so the rule is not applied"
    elif not s["rho50_readable"]:
        title = "Pooled over six seeds, d128 censors, so the minimum holds"
    else:
        title = (f"Six seeds of d128 still miss the free line, so the minimum "
                 f"holds at {STANDING_TRAINABLE / 1e6:.2f}M")

    gap = (s["rho50"] - FREE) / S_RHO50 if s["rho50_readable"] else float("nan")
    call = (f"median {rho_txt(pooled['s'])} is {gap:+.2f} s* "
            f"{'inside' if s['rho50'] <= FREE else 'past'} the free line {FREE:.3f}")

    before = (f"BEFORE -- {STANDING_RHO50:.3f} at {STANDING_TRAINABLE:,} trainable "
              f"(`no_mlp L=16 d192`, J2, finding 53), measured from disk on this "
              f"same pooled eval as {rho_txt(j2['s'])} over seeds "
              + "/".join(f"{m['rho50']:.3f}" for m in j2["ms"])
              + f", {j2['s']['n_cross']}/{j2['s']['n']} crossing.")
    if s["n"] < 6:
        moved = (f"The rule is NOT applied: it takes six seeds and {s['n']} are on "
                 f"disk, so there is no 'after' yet and no claim is made here.")
    elif took:
        moved = (f"The minimum MOVES: {STANDING_TRAINABLE:,} -> {KB_TRAINABLE:,} "
                 f"trainable, {shrink:.2f}x smaller.")
    else:
        moved = (f"The minimum does NOT move; it stays {STANDING_RHO50:.3f} at "
                 f"{STANDING_TRAINABLE:,}, unchanged by this stage.")
    after = (f"AFTER -- {rho_txt(pooled['s'])} at {s['trainable']:,} trainable, "
             f"{s['n_cross']}/{s['n']} crossing, {call}. " + moved)
    rule = (f"N2's pooling rule was fixed at registration -- 6 of 6 seeds crossing "
            f"AND the six-seed median at or below {FREE:.3f} -- because choosing "
            f"how to pool after seeing the fresh numbers is how a narrow miss "
            f"becomes a hit. Stage K's three seeds gave {KB_K_RHO50:.3f}, a miss "
            f"of {(KB_K_RHO50 - FREE) / S_RHO50:+.2f} s*.")
    size = (f"This is an N=48 claim. Finding 55 measured that the N=48 minimum "
            f"does not transfer to N=96, so nothing on this page says anything "
            f"about N=96, and the two sizes' rho_50 are never pooled.")

    textpage(o, title,
             f"stage N -- KB on fresh seeds {'/'.join(str(x) for x in N_SEEDS)}, "
             f"pooled with stage K's {'/'.join(str(x) for x in K_SEEDS)}",
             [("The pooled call (N2)", OK_C if took else BAND_C, after),
              ("The previous best, same eval", OK_C, before),
              ("Why the rule was fixed first", NOTE_C, rule),
              ("What size this is about", "0.25", size)],
             f"commit {commit} · {KB_TRAINABLE:,} trainable · "
             f"14-point grid to 0.33, 400 instances per bin · "
             f"shuffle_adj on every cell")


def page_replication(o, pooled, fresh, old, commit):
    v1, v0 = verdict_n1(fresh), verdict_n0(fresh)
    if v1 == "falsified":
        n_cens = 3 - fresh["s"]["n_cross"]
        title = f"The crossing does not replicate: {n_cens} of 3 fresh seeds censor"
    elif v0 == "held":
        title = "Both the level and the crossing replicate on fresh seeds"
    elif v0 == "low":
        title = "Fresh seeds land below stage K's level, which was pessimistic"
    elif v0 == "high":
        title = "Fresh seeds land above stage K's level, which was optimistic"
    else:
        title = "The fresh triple has not landed yet"

    fig = plt.figure(figsize=(16, 9))
    header(fig, title, "one rung, two independent seed triples, identical in "
                       "every other flag")
    ax = fig.add_axes(PLOT_BOX)
    for grp, color, label in ((old, K_C, "stage K, seeds 0/1/2"),
                              (fresh, N_C, "stage N, seeds 3/4/5")):
        if not grp:
            continue
        xs = [m["seed"] for m in grp["ms"]]
        ys = [m["rho50"] for m in grp["ms"]]
        ok = [m["flag"] == "ok" for m in grp["ms"]]
        ax.scatter([x for x, o_ in zip(xs, ok) if o_],
                   [y for y, o_ in zip(ys, ok) if o_],
                   s=150, color=color, zorder=3, label=label)
        ax.scatter([x for x, o_ in zip(xs, ok) if not o_],
                   [y for y, o_ in zip(ys, ok) if not o_],
                   s=150, facecolors="none", edgecolors=DEAD_C, zorder=3)
        if grp["s"]["rho50_readable"]:
            ax.hlines(grp["s"]["rho50"], min(xs) - 0.35, max(xs) + 0.35,
                      color=color, lw=2.5, zorder=2)
            # Labelled at the bar, not in a legend box: a legend placed anywhere
            # in this panel sits on either the J2 line or a seed point.
            ax.annotate(label, (min(xs) - 0.35, grp["s"]["rho50"]),
                        xytext=(2, 7), textcoords="offset points",
                        color=color, size=SMALL)
    ax.axhspan(PRED_LO, PRED_HI, color=NOTE_C, alpha=0.10, zorder=0)
    ax.axhline(FREE, color=BAND_C, lw=2, ls="--", zorder=1)
    ax.text(5.4, FREE + 0.004, f"free line {FREE:.3f}", color=BAND_C, ha="right",
            va="bottom", size=SMALL)
    ax.axhline(STANDING_RHO50, color=OK_C, lw=2, ls=":", zorder=1)
    ax.text(5.4, STANDING_RHO50 + 0.004, f"standing minimum J2 {STANDING_RHO50:.3f}",
            color=OK_C, ha="right", va="bottom", size=SMALL)
    ax.text(-0.4, PRED_HI + 0.004,
            f"N0's registered bracket [{PRED_LO:.3f}, {PRED_HI:.3f}]",
            color=NOTE_C, va="bottom", size=SMALL)
    ax.set_xlim(-0.5, 5.5)
    ax.set_xticks(range(6))
    ax.set_xlabel("seed")
    ax.set_ylabel(r"$\rho_{50}$   (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)

    column(fig, ["N0 -- did the LEVEL replicate?",
                 "  registered  fresh median in",
                 f"              [{PRED_LO:.3f}, {PRED_HI:.3f}]",
                 f"  measured    {rho_txt(fresh['s']) if fresh else 'pending'}",
                 f"  -> {v0.upper()}", "",
                 "N1 -- did the CROSSING replicate?",
                 "  registered  3 of 3 fresh seeds",
                 f"  measured    {fresh['s']['n_cross'] if fresh else 0} of "
                 f"{fresh['s']['n'] if fresh else 0}",
                 f"  -> {v1.upper()}", "",
                 "N1 and N3 are read BEFORE N2:",
                 "a level from a censoring rung",
                 "is a bound, not a measurement."])
    footer(fig, f"commit {commit} · open circles are censored seeds · "
                f"horizontal bars are each triple's bound-aware median · "
                f"a censor is a result, not a missing number")
    o.savefig(fig)


def page_transition(o, raw, pooled, commit):
    s = pooled["s"]
    n = s["n"]
    # The title is a finding about the curves, not a description of the grid:
    # "scored on the same grid" is true of every page in every deck.
    readable = [m["rho50"] for m in pooled["ms"] if m["flag"] == "ok"]
    spread = (max(readable) - min(readable)) / S_RHO50 if len(readable) > 1 else 0.0
    if not n:
        title = "No transition curve has landed yet"
    elif s["n_cross"] < n:
        title = f"{n - s['n_cross']} of {n} seeds never reach a 0.5 solve rate"
    elif spread >= 1.0:
        title = (f"The same architecture spreads {spread:.1f} s* across "
                 f"{n} seeds")
    else:
        title = f"All {n} transitions fall within one s* of each other"

    fig = plt.figure(figsize=(16, 9))
    header(fig, title, "solve rate against realized hint density -- the curve "
                       "rho_50 is read off")
    ax = fig.add_axes([0.07, 0.17, 0.80, 0.58])
    ends = []
    for m in pooled["ms"]:
        rows = raw.get(KB + (m["seed"],), [])
        if not rows:
            continue
        pts = sorted((float(r["realized_rho"]), float(r["solve_rate"])) for r in rows)
        color = K_C if m["seed"] in K_SEEDS else N_C
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4, lw=1.8,
                color=color, alpha=0.85, rasterized=True)
        ends.append((pts[-1][0], pts[-1][1], color, f"s{m['seed']}"))
    end_labels(ax, ends, 0.035)
    ax.axhline(0.5, color=BAND_C, lw=2, ls="--")
    ax.text(0.335, 0.52, "solve rate 0.5 -- where rho_50 is read off",
            color=BAND_C, size=SMALL)
    ax.set_xlabel(r"realized hint density $\rho$")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    footer(fig, f"commit {commit} · orange = stage K's seeds 0/1/2, blue = stage N's "
                f"3/4/5 · 400 instances per bin · grid floor 0.33, top {GRID_TOP} · "
                f"rho_50 censors when no bin clears 0.5")
    o.savefig(fig)


def page_plateau(o, pooled, fresh, old, commit):
    v = verdict_n3(pooled)
    esc, n = pooled["esc"], pooled["s"]["n"]
    if v == "pending":
        title = f"Only {n} of 6 seeds have landed, so the plateau call waits"
    elif v == "held":
        title = f"{esc} of {n} seeds train out of the plateau, so the level is capacity"
    else:
        title = f"Only {esc} of {n} seeds escape the plateau: d128 is optimization-limited"

    fig = plt.figure(figsize=(16, 9))
    header(fig, title, "training loss against the N=48 plateau -- capacity or "
                       "trainability")
    ax = fig.add_axes(PLOT_BOX)
    ends, lo = [], PLATEAU
    for m in pooled["ms"]:
        tr, _acc = curve(m["path"])
        color = K_C if m["seed"] in K_SEEDS else N_C
        ax.plot(range(1, len(tr) + 1), tr, lw=2, color=color, alpha=0.9,
                rasterized=True)
        ends.append((len(tr), tr[-1], color, f"s{m['seed']}"))
        lo = min(lo, min(tr))
    end_labels(ax, ends, 0.012)
    ax.axhline(PLATEAU, color=BAND_C, lw=2, ls="--")
    ax.annotate(f"plateau {PLATEAU} (finding 44, an N=48 number)",
                (2, PLATEAU), xytext=(0, 7), textcoords="offset points",
                color=BAND_C, size=SMALL)
    # Cropped to the decision region: epoch 1 is near 2.4 and squashes the only
    # part of the curve the plateau call is about. Said in the footer, not hidden.
    ax.set_ylim(lo - 0.02, 1.95)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train NLL")
    ax.spines[["top", "right"]].set_visible(False)

    lines = ["N3 -- the plateau",
             "  registered  >= 4 of 6 escape",
             f"  stage K     {KB_T_ESC(old)}",
             f"  stage N     {KB_T_ESC(fresh)}",
             f"  pooled      {esc} of {n}", f"  -> {v.upper()}", "",
             "final train NLL"]
    for m in pooled["ms"]:
        tr, _ = curve(m["path"])
        mark = "" if tr[-1] < PLATEAU else "  never escaped"
        lines.append(f"  s{m['seed']}  {tr[-1]:.3f}{mark}")
    column(fig, lines)
    footer(fig, f"commit {commit} · orange = stage K, blue = stage N · y-axis "
                f"cropped at 1.95; epoch 1 starts near 2.4 · every checkpoint is "
                f"the final epoch, no best-val selection · an optimization-limited "
                f"rung outranks N2")
    o.savefig(fig)


def KB_T_ESC(grp):
    return f"{grp['esc']} of {grp['s']['n']}" if grp else "pending"


def page_accuracy(o, pooled, fresh, old, commit):
    accs = [(m["seed"], curve(m["path"])[1][-1]) for m in pooled["ms"]]
    if not accs:
        title = "No accuracy has landed yet"
    else:
        lo, hi = min(a for _, a in accs), max(a for _, a in accs)
        # Measured against the registered LOO bar, not against the word "only":
        # a span is small or large only relative to something declared earlier.
        title = (f"All {len(accs)} seeds fall inside the {LOO_BAR:.4f} LOO bar"
                 if hi - lo < LOO_BAR else
                 f"Seed spread of {hi - lo:.4f} exceeds the {LOO_BAR:.4f} LOO bar")

    fig = plt.figure(figsize=(16, 9))
    header(fig, title, "the measure that never censors, read before rho_50")
    ax = fig.add_axes(PLOT_BOX)
    for seed, a in accs:
        color = K_C if seed in K_SEEDS else N_C
        ax.scatter([seed], [a], s=150, color=color, zorder=3)
    ax.set_xlim(-0.5, 5.5)
    ax.set_xticks(range(6))
    ax.set_xlabel("seed")
    ax.set_ylabel("final validation token accuracy")
    ax.spines[["top", "right"]].set_visible(False)

    body = ("Accuracy is read BEFORE rho_50 on every stage of this programme, "
            "because it never censors and because a rung that falls OFF "
            "scripts/acc_vs_solve.py's fitted curve is the interesting one. "
            "That curve is an N=48 object and is applied here only because this "
            "stage is N=48.")
    column(fig, [f"  s{seed}  {a:.4f}" for seed, a in accs]
           + ["", f"  median K  {old['acc']:.4f}" if old else "  median K  pending",
              f"  median N  {fresh['acc']:.4f}" if fresh else "  median N  pending",
              "", f"  LOO bar   {LOO_BAR:.4f}"])
    # In the column, not under the axes: at y=0.11 this ran through the x-label
    # and into the footer.
    fig.text(COL_X, 0.44, textwrap.fill(body, 46), size=SMALL - 1, va="top",
             color="0.25")
    footer(fig, f"commit {commit} · orange = stage K, blue = stage N · "
                f"N4's out-of-sample check runs separately over the nine "
                f"sufficiency dirs plus K and N")
    o.savefig(fig)


def page_next(o, pooled, fresh, j2, commit):
    s = pooled["s"]
    took = qualifies6(s)

    # The "go narrower" question, measured instead of asserted: the width gap
    # this rung already spans, against the seed spread inside it. An earlier
    # draft simply claimed the next steps were smaller than the spread.
    readable = [m["rho50"] for m in pooled["ms"] if m["flag"] == "ok"]
    spread = max(readable) - min(readable) if len(readable) > 1 else float("nan")
    width_gap = (s["rho50"] - j2["s"]["rho50"]) if s["rho50_readable"] else float("nan")
    grid = (f"d128 -> d192 moves rho_50 by {width_gap:.3f} while the seeds inside "
            f"d128 already spread {spread:.3f}. The width effect this ladder is "
            f"measuring is about the size of its own seed noise, so a rung below "
            f"d128 would need more than three seeds to say anything -- which is a "
            f"statement about this grid's resolution, not about the task.")

    if s["n"] < 6:
        textpage(o, f"Only {s['n']} of 6 seeds have landed, so nothing is settled yet",
                 "what stage N will settle once its array closes",
                 [("Not a result", BAND_C,
                   f"This page is rendered from a partial pool ({s['n']} of 6 "
                   f"seeds) and states no verdict. The shipping deck refuses to "
                   f"build in this state; only --allow-partial reaches here."),
                  ("What the rule will be", NOTE_C,
                   f"6 of 6 crossing AND six-seed median at or below {FREE:.3f}, "
                   f"fixed at registration before any fresh cell started."),
                  ("What is already measured", "0.25", grid)],
                 f"commit {commit} · dry run, not for sending")
        return

    if took:
        title = "The minimum moved, and the next question is whether it transfers"
        blocks = [
            ("What moved", OK_C,
             f"The N=48 standing minimum moves to `no_mlp L=16 d128`, "
             f"{KB_TRAINABLE:,} trainable, {STANDING_TRAINABLE / KB_TRAINABLE:.2f}x "
             f"smaller than J2, on 6 of 6 seeds crossing."),
            ("What is NOT settled", BAND_C,
             "Whether it transfers to N=96. Finding 55 measured that the previous "
             "N=48 minimum did not, so the honest default is that this one does "
             "not either until a paired N=96 cell says so."),
            ("The next cut", NOTE_C, grid),
        ]
    else:
        title = "The minimum holds, and this width grid is at its resolution limit"
        why = ("not all six seeds crossed" if s["n_cross"] < s["n"]
               else f"the six-seed median is "
                    f"{(s['rho50'] - FREE) / S_RHO50:+.2f} s* past {FREE:.3f}"
               if s["rho50_readable"] else "the pooled median censors")
        blocks = [
            ("The negative result, stated as one", BAND_C,
             f"{s['n']} seeds of the smallest crossing rung still do not make it "
             f"free: {why}. `no_mlp L=16 d192` keeps the minimum at "
             f"{STANDING_TRAINABLE:,} trainable, {STANDING_RHO50:.3f}, unchanged "
             f"by this stage."),
            ("What it does settle", OK_C,
             f"Stage K's KB row is no longer a three-seed number. Its level and "
             f"its crossing now rest on {s['n']} seeds, which is what K1's own "
             f"registration asked for before the d128 result is quoted outside "
             f"the scorer."),
            ("Where this grid runs out", NOTE_C, grid),
        ]
    textpage(o, title, "what stage N settles, and what it leaves open", blocks,
             f"commit {commit} · the replacement rule H3 = I3 = J3 = K2, pooled "
             f"to six seeds as N2 registered it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="notebooks/stagen_2026-11-10",
                    help="directory; gets slides.pdf and one PNG per page")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--allow-partial", action="store_true",
                    help="render a dry-run deck from fewer than six seeds")
    args = ap.parse_args()

    set_style()
    dirs = [d for d in (SCAN_N, SCAN_K, SCAN_J) if os.path.isdir(d)]
    cells, raw = collect(dirs, args.boot)
    cells = seedmap(cells)

    pooled = rows_for(cells, KB)
    if pooled is None:
        raise SystemExit("no KB cell on disk at all -- nothing to build")
    fresh = rows_for(cells, KB, N_SEEDS)
    old = rows_for(cells, KB, K_SEEDS)
    j2 = rows_for(cells, J2)
    if j2 is None:
        raise SystemExit("the standing minimum J2 is not on disk; the "
                         "before/after comparison cannot be made honestly")

    n = pooled["s"]["n"]
    if n < 6:
        if not args.allow_partial:
            raise SystemExit(
                f"only {n} of 6 seeds are on disk. The shipping deck is not "
                f"built from a partial pool -- re-run when stage N's array has "
                f"closed, or pass --allow-partial for a dry run.")
        args.out = args.out + "_DRY_PARTIAL"
        print(f"DRY RUN: {n} of 6 seeds; writing {args.out}")

    print(f"pooled KB: {rho_txt(pooled['s'])}, {pooled['s']['n_cross']}/"
          f"{pooled['s']['n']} cross, {pooled['esc']} escapes, "
          f"{pooled['s']['trainable']:,} trainable -> "
          f"{'TAKES THE MINIMUM' if qualifies6(pooled['s']) else 'minimum holds'}")
    print(f"J2 (before): {rho_txt(j2['s'])}, {j2['s']['trainable']:,} trainable")

    os.makedirs(args.out, exist_ok=True)
    # fall=True and shuffle_adj=True are the INSTANCE KEYS stage N's cells were
    # trained and scored on (--fall --shuffle-adj), not deck decoration: a deck
    # that defaults fall to False illustrates a shallower distribution than the
    # one the numbers come from. pick="deep" because the subject is reasoning.
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33,
                          rho_max=GRID_TOP, shuffle_adj=True, fall=True,
                          pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=8)
        page_headline(o, pooled, fresh, old, j2, args.commit)
        page_replication(o, pooled, fresh, old, args.commit)
        page_transition(o, raw, pooled, args.commit)
        page_plateau(o, pooled, fresh, old, args.commit)
        page_accuracy(o, pooled, fresh, old, args.commit)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the "
                               "task every cell in this deck is trained and scored "
                               "on. Stage N changes nothing about it and nothing "
                               "about the architecture either -- only the seed "
                               "triple moves, from 0/1/2 to 3/4/5.")
        page_next(o, pooled, fresh, j2, args.commit)

    mb = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf  ({mb:.2f} MB, {o.n} pages)")
    print(f"PNGs at {args.out}/page1..{o.n}.png -- READ EVERY ONE before shipping")
    if mb > 2.0:
        print("OVER THE 2 MB UPLOAD CAP -- reduce dpi or rasterize more")


if __name__ == "__main__":
    main()
