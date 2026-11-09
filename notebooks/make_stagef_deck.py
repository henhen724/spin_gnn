"""Deck: stage F -- does the N=48 minimum survive at N=96?

Every "smallest architecture" result in this programme was measured on one
problem size. Stage F (51656823) is the transfer test: the surviving
attention-only stack (L=16 d192) at N=96 against a PAIRED complete-model control
(full L=12 d256), three seeds each, on the identical 14 x 400 leak-free grid.

    python3 -m notebooks.make_stagef_deck --commit $(git rev-parse --short HEAD)

WRITTEN BEFORE THE FIRST F CELL LANDED, like scripts/stagef_score.py and the
sbatch registration -- git log is the timestamp. Every page is read off disk and
every headline is selected by the branch the data falls in, from the branches
named in the registration. Nothing here is a sentence chosen after seeing a
number.

TWO GUARDS CARRIED FROM THE SCORER, because they are the ways this deck could
quietly lie:

  1. s*_96 = 0.119 is frozen from the three two-seed N=96 rungs that existed at
     registration. s* = 0.1484 is an N=48 number and is NEVER used here, and the
     re-estimated s*_96 is plotted but never substituted into a verdict.
  2. Finding 44's 1.64 plateau is an N=48 number. NO escape count is quoted at
     N=96 -- its plateau level has never been measured. The training panel is
     drawn against the measured N=96 control NLL band instead, which is a
     description, not a threshold.

And the N=48 rho grid ends at 0.33 while the N=96 one ends at 0.32, so no panel
on these slides puts an N=48 and an N=96 rho_50 on the same axis.
"""
import argparse
import os
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_stagehi_deck import acc_of, curve
from scripts.minarch_analysis import cell_metrics, load_all
from scripts.n96_control import n96_key
from scripts.stagef_score import (CONTROL, CONTROL_2SEED, CONTROL_BRACKET,
                                  CONTROL_NLL_BAND, CONTROL_TOTAL, GRID_FLOOR_96,
                                  N48_HUB, N48_MINIMUM, PREDICTED_TOTAL_96,
                                  PREDICTED_TRAINABLE_96, S48, S96, S96_PARTS, SCAN_F,
                                  SCAN_F_DIRS, STRIPPED, rho_txt, rung, summarize)

STRIP_C = "#1a7a3c"     # the attention-only arm
CTRL_C = "#13507a"      # the complete-model control
BAND_C = "#c1272d"      # a falsified prediction, a band edge
NOTE_C = "#6a3d9a"      # registered in advance
DEAD_C = "0.45"         # censored

GRID_TOP = 0.95         # where a censored seed is drawn, above the grid's top bin


def collect96(scan_dirs, boot):
    """Every leak-free N=96 cell, plus the raw per-rho rows the curves need.

    stagef_score.collect() returns only the metrics; the transition panel needs
    the per-rho rows as well, so this is the scorer's collect with `raw` kept.
    Keys are n96_key tuples, which cannot collide with an N=48 cell_key.
    """
    by = defaultdict(list)
    for r in load_all(scan_dirs):
        if int(r["n_nodes"]) != 96:
            continue
        by[n96_key(r) + (int(r["seed"]),)].append(r)
    cells = {k: dict(cell_metrics(v, boot), seed=k[-1], path=v[0]["path"])
             for k, v in by.items()}
    return cells, by


# --------------------------------------------------------------------------
# Which registered branch did the data land in? These three encode the
# falsifiers from the sbatch header and stagef_score.py, nothing else.
# --------------------------------------------------------------------------
def verdict_f1(st):
    """F1 gates everything: an unmeasured rho_50 cannot be compared to a level."""
    if st is None:
        return "absent"
    if st["n"] < 3:
        return "incomplete"
    n_cens = st["n"] - st["n_cross"]
    if n_cens == 0:
        return "held"
    return "all_censored" if n_cens == 3 else "part_censored"


def verdict_f0(st, ct, f1_call):
    if f1_call != "held":
        return "not_scored"
    if ct is None or not ct["rho50_readable"]:
        return "control_censored"
    d = st["rho50"] - ct["rho50"]
    if abs(d) <= S96:
        return "held"
    return "higher" if d > 0 else "lower"


def verdict_f2(ct):
    if ct is None:
        return "absent"
    if ct["n"] < 3:
        return "incomplete"
    new = [(sd, v, ok) for sd, v, ok in ct["per_seed"] if sd == 2]
    if not new:
        return "absent"
    _sd, v, ok = new[0]
    if not ok:
        return "falsified"
    lo, hi = CONTROL_BRACKET
    return "held" if (lo <= v <= hi
                      and abs(ct["rho50"] - CONTROL_2SEED) <= S96) else "falsified"


# The headline is keyed on F1 first, because F1 gates whether F0 is a level at
# all. Every string below was written before the run closed.
TITLE_F1 = {
    "all_censored": "The breaking point: the N=48 stack does not train at N=96",
    "part_censored": "A seed story, not yet an architecture statement",
    "incomplete": "Incomplete: fewer than three stripped seeds landed",
    "absent": "No stripped cell on disk",
}
TITLE_F0 = {
    "held": "The stripped stack reaches the complete model at a SECOND problem size",
    "higher": "The N=48 minimum does NOT transfer: the cut is load-bearing at N=96",
    "lower": "The stripped arm comes in LOWER, which reads as an undertrained control",
    "control_censored": "The control itself censored at N=96: no level to compare against",
    "not_scored": "F0 not scored as a level -- see F1",
}


def title_for(f1_call, f0_call):
    return TITLE_F1.get(f1_call) or TITLE_F0[f0_call]


def pair(cells):
    """The two stage F arms, with everything the pages need."""
    out = {}
    for key, tag, name in ((STRIPPED, "stripped", "attention-only L=16 d192"),
                           (CONTROL, "control", "full L=12 d256")):
        ms = rung(cells, key)
        out[tag] = None if not ms else dict(
            key=key, tag=tag, name=name, ms=ms, s=summarize(ms),
            trainable=ms[0]["n_trainable"], acc=acc_of(ms))
    return out


def gap(st, ct):
    """stripped - control in rho_50, or None when either side has no level."""
    if st is None or ct is None:
        return None
    if not (st["rho50_readable"] and ct["rho50_readable"]):
        return None
    return st["rho50"] - ct["rho50"]


def page_headline(o, p, f1_call, f0_call, f2_call, commit):
    st = p["stripped"]["s"] if p["stripped"] else None
    ct = p["control"]["s"] if p["control"] else None
    d = gap(st, ct)
    tot_s = st["n_params"] if st else PREDICTED_TOTAL_96
    tot_c = ct["n_params"] if ct else CONTROL_TOTAL
    tr_s = p["stripped"]["trainable"] if p["stripped"] else PREDICTED_TRAINABLE_96

    if f1_call == "held" and f0_call == "held":
        first = (
            "THE CUT IS FREE AT A SECOND PROBLEM SIZE, AND THE CLAIM IS 'SMALLER'", STRIP_C,
            f"stripped {st['rho50']:.3f} vs control {ct['rho50']:.3f}, difference "
            f"{d:+.3f} = {abs(d) / S96:.2f} s*_96, inside the registered {S96:.3f}. The "
            f"claim earned is {tot_c / tot_s:.1f}x fewer parameters (total {tot_c:,} -> "
            f"{tot_s:,}, {tr_s:,} trainable in the stripped arm) at no separable cost, AT "
            f"TWO PROBLEM SIZES. It is not 'better' -- a difference below s*_96 is not a "
            f"difference, and the registration said so before this number existed.")
    elif f0_call == "higher":
        first = (
            "THE TRANSFER FAILS, AND EVERY SIZE CLAIM HERE BECOMES AN N=48 CLAIM", BAND_C,
            f"stripped {st['rho50']:.3f} vs control {ct['rho50']:.3f}, {d:+.3f} = "
            f"{abs(d) / S96:.2f} s*_96, OUTSIDE the registered {S96:.3f} and in the "
            f"direction that matters. The MLP, the width, or both are load-bearing at "
            f"N=96. Every 'smallest architecture' claim in this programme is relabelled "
            f"as an N=48 claim. This was a named outcome before the run closed and it is "
            f"reported as a result, not as a disappointment.")
    elif f0_call == "lower":
        first = (
            "LOWER THAN THE CONTROL -- WHICH THE REGISTRATION READS AS A CONTROL FAULT", NOTE_C,
            f"stripped {st['rho50']:.3f} vs control {ct['rho50']:.3f}, {d:+.3f} = "
            f"{abs(d) / S96:.2f} s*_96 below. The registered reading is that the N=96 "
            f"CONTROL IS UNDERTRAINED at 375k steps, not that stripping helps: a cut "
            f"cannot add capacity. The next move is a step-budget check on the control, "
            f"not a victory lap. Writing that down in advance is the only reason it can "
            f"be said now without sounding like an excuse.")
    elif f1_call == "all_censored":
        first = (
            "THE BREAKING POINT, AND IT IS THE RESULT", DEAD_C,
            f"All three stripped seeds censored on a grid whose floor is "
            f"{GRID_FLOOR_96}. This is an ARCHITECTURE statement: the attention-only "
            f"stack that works at N=48 does not train at N=96 at this budget. F0 is NOT "
            f"quoted as a level -- there is no median to compare. mean_solve, which never "
            f"censors, is {st['mean_solve']:.4f} against the control's "
            f"{ct['mean_solve']:.4f} if the control read, and that gap is the rest of "
            f"the story.")
    elif f1_call == "part_censored":
        first = (
            "A SEED STORY, NOT AN ARCHITECTURE STATEMENT", DEAD_C,
            f"{st['n'] - st['n_cross']} of 3 stripped seeds censored. Finding 51 named "
            f"this case in advance -- a 2-of-3 at 50 epochs was once ONE UNFINISHED SEED "
            f"-- so no median level is quoted here. Read mean_solve, which never censors: "
            f"{st['mean_solve']:.4f} stripped against "
            + (f"{ct['mean_solve']:.4f} control." if ct else "no control on disk.")
            + " Three seeds that all read is what it would take to make this a statement "
              "about shape.")
    else:
        first = (
            "NOT SCORED AS A LEVEL", DEAD_C,
            f"F1 came back '{f1_call.replace('_', ' ')}' and F0 '{f0_call.replace('_', ' ')}'. "
            f"The registration forbids comparing an unmeasured or unfinished arm to a "
            f"level, so what is on disk is shown on the following pages with no verdict "
            f"attached. This page exists in this state on purpose: a deck that cannot "
            f"reach a verdict should say so rather than find one.")

    f2_tail = (
        f" It landed inside [{CONTROL_BRACKET[0]:.3f}, {CONTROL_BRACKET[1]:.3f}] and the "
        f"3-seed median is {ct['rho50']:.3f} against the 2-seed {CONTROL_2SEED:.3f}, so "
        f"the control's level is quotable."
        if f2_call == "held" and ct and ct["rho50_readable"] else
        " It did not land where the registration bracketed it, so the control is "
        "seed-dominated and every verdict on these slides survives only as an ORDERING.")

    sub = (f"N=96, degree 5, 4 colours, leak-free | job 51656823, 4 cells | "
           f"s*_96 = {S96:.3f}, frozen at registration")
    if f2_call == "falsified":
        sub += " | F2 FALSIFIED"

    textpage(
        o,
        title_for(f1_call, f0_call),
        sub,
        [
            first,
            (f"THE CONTROL IS NOT ONE DRAW (F2 {f2_call.upper()})", CTRL_C,
             f"Two N=96 control seeds existed at registration, median "
             f"{CONTROL_2SEED:.3f} over the bracket [{CONTROL_BRACKET[0]:.3f}, "
             f"{CONTROL_BRACKET[1]:.3f}]. The third seed was bought by this job "
             f"precisely because a two-seed control was named as the most likely way "
             f"stage F ends up weaker than it looks." + f2_tail),
            ("WHAT IS DELIBERATELY NOT ON THESE SLIDES", DEAD_C,
             f"No N=48 rho_50 shares an axis with an N=96 one -- the grids end at 0.33 "
             f"and {GRID_FLOOR_96}. No plateau-escape count is quoted: finding 44's 1.64 "
             f"is an N=48 number and N=96's plateau has never been measured, so the "
             f"training panel is drawn against a measured BAND instead. s* = {S48} is an "
             f"N=48 number and is used nowhere here. The N=48 standing minimum "
             f"({desc_min()}) does not move on anything in this job (F3)."),
        ],
        footer_text(commit),
        y0=0.80, gap=0.024, width=120)


def footer_text(commit):
    return (f"spin_gnn @ {commit} | job 51656823 | scorer scripts/stagef_score.py, "
            f"registered before the first cell landed | N=96, 14 x 400 leak-free "
            f"(--shuffle-adj --fall), 3 seeds")


def page_contrast(o, p, raw, commit):
    """The money plot: per-seed levels, the transition curves, the training curves."""
    st, ct = p["stripped"], p["control"]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The transfer test at N=96, arm against paired control",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893,
             "Left: every seed, not just the median. Middle: the curves rho_50 is "
             "read off. Right: training, against the MEASURED N=96 band",
             ha="center", size=MED - 3, color="0.3")

    arms = [a for a in (st, ct) if a]
    col = {"stripped": STRIP_C, "control": CTRL_C}

    # --- left: per-seed rho_50 ---------------------------------------------
    ax = fig.add_axes([0.055, 0.22, 0.195, 0.56])
    ctrl_i = None
    for i, r in enumerate(arms):
        c = col[r["tag"]]
        if r["tag"] == "control":
            ctrl_i = i
        # Censored seeds all land on exactly GRID_TOP, so three of them draw as one
        # marker unless they are spread -- and "three censors" is precisely the
        # reading F1 turns on. Offset them; crossing seeds have distinct levels.
        cens = [m for m in r["ms"] if m["flag"] != "ok"]
        for m in r["ms"]:
            if m["flag"] == "ok":
                ax.plot(i, m["rho50"], "o", ms=9, color=c, alpha=0.75)
        for j, _m in enumerate(cens):
            dx = 0.10 * (j - (len(cens) - 1) / 2)
            ax.plot(i + dx, GRID_TOP, "^", ms=10, mfc="none", mew=1.8, color=c)
        if r["s"]["rho50_readable"]:
            ax.plot([i], [r["s"]["rho50"]], "_", ms=40, mew=3, color=c)
    # The bracket F2 registered for the third control seed, drawn on the control
    # column only -- it is a prediction about that seed, not about the arm.
    if ctrl_i is not None:
        lo, hi = CONTROL_BRACKET
        ax.add_patch(plt.Rectangle((ctrl_i - 0.3, lo), 0.6, hi - lo, color=CTRL_C,
                                   alpha=0.10, zorder=0))
        ax.text(ctrl_i + 0.32, (lo + hi) / 2, "F2\nbracket", size=SMALL - 3,
                color=CTRL_C, va="center")
    ax.errorbar(len(arms) - 0.5, 0.90, yerr=S96 / 2, color="0.25", capsize=5, lw=1.8)
    ax.text(len(arms) - 0.56, 0.90, f"s*_96 {S96:.3f} ", size=SMALL - 2, color="0.25",
            va="center", ha="right")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([("stripped\nL=16 d192" if r["tag"] == "stripped"
                         else "control\nL=12 d256") for r in arms], size=SMALL - 1)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel("rho_50   (lower is better)")
    ax.set_ylim(GRID_FLOOR_96 - 0.02, 1.0)
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("Triangles = censored,\ndrawn above the grid's top bin", size=MED - 5)

    # --- middle: the transition curves -------------------------------------
    ax2 = fig.add_axes([0.315, 0.22, 0.195, 0.56])
    for r in arms:
        for m in r["ms"]:
            rows = next((v for k, v in raw.items()
                         if k[:-1] == r["key"] and v[0]["path"] == m["path"]), None)
            if not rows:
                continue
            pts = sorted((x["realized_rho"], x["solve_rate"]) for x in rows)
            ax2.plot([q[0] for q in pts], [q[1] for q in pts], "-", lw=1.7,
                     color=col[r["tag"]], alpha=0.8)
    ax2.axhline(0.5, color="0.5", lw=1.2, ls=":")
    ax2.text(0.015, 0.5, " solve rate 0.5", size=SMALL - 1, color="0.4", va="bottom",
             transform=ax2.get_yaxis_transform())
    ax2.axvline(GRID_FLOOR_96, color="0.6", lw=1.1, ls="--")
    ax2.text(GRID_FLOOR_96, 0.97, f" grid floor {GRID_FLOOR_96}", size=SMALL - 3,
             color="0.4", va="top", transform=ax2.get_xaxis_transform())
    ax2.set_xlabel("realized hint density rho")
    ax2.set_ylabel("solve rate")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(alpha=0.25)
    ax2.set_title("Identical 14 x 400 grid, both arms", size=MED - 4)

    # --- right: training curves against the measured N=96 band ---------------
    ax3 = fig.add_axes([0.565, 0.22, 0.195, 0.56])
    for r in arms:
        for m in r["ms"]:
            nll, _acc = curve(m["path"])
            ax3.plot(range(1, len(nll) + 1), nll, "-", lw=1.5, color=col[r["tag"]],
                     alpha=0.8)
    lo, hi = CONTROL_NLL_BAND
    ax3.axhspan(lo, hi, color="0.6", alpha=0.18, zorder=0)
    # Top-right of the axes, not on the band itself: at these widths the band sits
    # right where the curves plateau, and a label there lands on top of them.
    ax3.text(0.97, 0.96, f"band = every leak-free\nN=96 cell to date:\n"
                         f"{lo:.3f}-{hi:.3f}",
             size=SMALL - 3, color="0.35", va="top", ha="right",
             transform=ax3.transAxes)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("train NLL")
    ax3.grid(alpha=0.25)
    ax3.set_title("A band, NOT a threshold:\nN=96's plateau is unmeasured", size=MED - 5)

    fig.legend(handles=[Line2D([], [], color=STRIP_C, lw=2.4,
                               label="stripped  attention-only L=16 d192"),
                        Line2D([], [], color=CTRL_C, lw=2.4,
                               label="control  full L=12 d256")],
               loc="lower center", bbox_to_anchor=(0.41, 0.09), ncol=2, frameon=False,
               fontsize=SMALL)

    # --- the metrics table, to the right of the axes ------------------------
    rows = [f"{'arm':<10}{'rho_50':>10}{'mean':>8}{'sprd':>7}{'cross':>6}"]
    rows.append("-" * 41)
    for r in arms:
        s = r["s"]
        rows.append(f"{r['tag'][:9]:<10}{rho_txt(s):>10}{s['mean_solve']:>8.3f}"
                    f"{s['spread_mean']:>7.3f}{s['n_cross']:>4}/{s['n']}")
    rows.append("")
    rows.append(f"{'arm':<10}{'trainable':>13}{'total':>13}")
    rows.append("-" * 36)
    for r in arms:
        rows.append(f"{r['tag'][:9]:<10}{r['trainable']:>13,}{r['s']['n_params']:>13,}")
    d = gap(st["s"] if st else None, ct["s"] if ct else None)
    rows.append("")
    rows.append(f"gap  {d:+.3f} = {abs(d) / S96:.2f} s*_96" if d is not None
                else "gap  n/a -- one side censors")
    fig.text(0.785, 0.78, "\n".join(rows), size=SMALL - 4, family="monospace",
             va="top", ha="left", color="0.15")

    footer(fig, footer_text(commit))
    o.savefig(fig)


def page_two_sizes(o, p, commit):
    """N=48 and N=96 side by side, each in ITS OWN units, never as a ratio."""
    st = p["stripped"]["s"] if p["stripped"] else None
    ct = p["control"]["s"] if p["control"] else None
    desc48, tr48, rho48 = N48_MINIMUM
    hdesc, htr, hrho = N48_HUB

    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Two problem sizes, two yardsticks, deliberately not one axis",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893,
             "Each panel is a gap against ITS OWN control, in ITS OWN s*. The grids "
             f"end at 0.33 and {GRID_FLOOR_96}; the numbers are not comparable",
             ha="center", size=MED - 3, color="0.3")

    for i, (ttl, a, b, alab, blab, bar, barname) in enumerate([
            ("N=48 (frozen, finding 53)", rho48, hrho, desc48, hdesc, S48, "s*"),
            ("N=96 (this job)",
             st["rho50"] if st and st["rho50_readable"] else None,
             ct["rho50"] if ct and ct["rho50_readable"] else None,
             "attention-only L=16 d192", "full L=12 d256", S96, "s*_96")]):
        ax = fig.add_axes([0.085 + 0.42 * i, 0.24, 0.30, 0.56])
        if a is None or b is None:
            ax.text(0.5, 0.5, "no level to plot\n(an arm censored;\nsee mean_solve "
                              "on the\nprevious page)", ha="center", va="center",
                    size=MED - 2, color=DEAD_C, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            # The titled branch below is the only one that names the panel; without
            # this the censored panel renders as an unlabelled empty box.
            ax.set_title(f"{ttl}\ngap n/a  -- an arm censored", size=MED - 4)
        else:
            ax.plot(0, a, "o", ms=13, color=STRIP_C)
            ax.plot(1, b, "o", ms=13, color=CTRL_C)
            ax.plot([0, 1], [a, b], "-", lw=1.4, color="0.5", zorder=0)
            ax.errorbar(0.5, (a + b) / 2, yerr=bar / 2, color="0.25", capsize=6, lw=2)
            ax.text(0.53, (a + b) / 2, f" {barname} {bar:.3f}", size=SMALL - 1,
                    color="0.25", va="center")
            ax.set_xticks([0, 1])
            ax.set_xticklabels([textwrap.fill(alab, 18), textwrap.fill(blab, 18)],
                               size=SMALL - 1)
            ax.set_xlim(-0.45, 1.45)
            ax.set_ylabel("rho_50   (lower is better)")
            ax.grid(alpha=0.25, axis="y")
            sep = abs(a - b) > bar
            ax.set_title(f"{ttl}\ngap {a - b:+.3f} = {abs(a - b) / bar:.2f} {barname}"
                         f"  -- {'SEPARABLE' if sep else 'not separable'}",
                         size=MED - 4)
    fig.text(0.5, 0.125,
             "The left panel does not move on anything in this job (F3). F can "
             "corroborate or undermine the N=48 minimum; it can never replace it.",
             ha="center", size=SMALL, color=NOTE_C, style="italic")
    footer(fig, footer_text(commit))
    o.savefig(fig)


def page_predictions(o, p, f1_call, f0_call, f2_call, commit):
    st = p["stripped"]["s"] if p["stripped"] else None
    ct = p["control"]["s"] if p["control"] else None
    d = gap(st, ct)
    ok = {"held": STRIP_C, "not_scored": DEAD_C, "absent": DEAD_C,
          "incomplete": DEAD_C}

    textpage(
        o,
        "The registered predictions, scored",
        "Named in scripts/stagef_score.py before the first F cell had an eval on disk",
        [
            (f"F1  IT TRAINS AT ALL -- {f1_call.replace('_', ' ').upper()}",
             ok.get(f1_call, BAND_C),
             f"Registered: 3 of 3 stripped seeds produce a MEASURED rho_50 on a grid "
             f"whose floor is {GRID_FLOOR_96}. Observed: "
             + (f"{st['n_cross']} of {st['n']} crossing."
                if st else "no stripped cell on disk.")
             + " F1 gates F0, because an unmeasured rho_50 cannot be compared to a "
               "level, and it gates the READING of a censor: three censors is an "
               "architecture statement, one or two is finding 51's seed story."),

            (f"F0  THE HEADLINE -- {f0_call.replace('_', ' ').upper()}",
             ok.get(f0_call, BAND_C),
             f"Registered: |stripped - control| <= s*_96 = {S96:.3f}, frozen from the "
             f"three two-seed N=96 rungs that existed at registration ("
             + ", ".join(f"{k} {v:.3f}" for k, v in S96_PARTS.items()) + "). Observed: "
             + (f"stripped {st['rho50']:.3f} vs control {ct['rho50']:.3f}, {d:+.3f} = "
                f"{abs(d) / S96:.2f} s*_96." if d is not None else
                "not scored as a level -- see F1.")
             + " BOTH falsification directions were given a reading in advance, which "
               "is what stops either one from being re-narrated after the fact."),

            (f"F2  THE CONTROL IS NOT ONE DRAW -- {f2_call.upper()}",
             ok.get(f2_call, BAND_C),
             f"Registered: the third control seed lands inside "
             f"[{CONTROL_BRACKET[0]:.3f}, {CONTROL_BRACKET[1]:.3f}] AND the 3-seed "
             f"median stays within s*_96 of the 2-seed {CONTROL_2SEED:.3f}. Observed: "
             + (", ".join(f"s{sd} " + (f"{v:.3f}" if c else "CENSORED")
                          for sd, v, c in ct["per_seed"]) + f", median {rho_txt(ct)}."
                if ct else "no control cell on disk.")
             + " If F2 falsifies, F0 survives only as an ORDERING and the levels are "
               "not quotable."),

            ("F3  THE N=48 STANDING MINIMUM DOES NOT MOVE ON N=96 EVIDENCE", NOTE_C,
             f"A REGISTRATION, not a measurement. The standing minimum is "
             f"{desc_min()}, an N=48 ladder result under rule H3 = I3 = J3, on a grid "
             f"whose floor is 0.33. Stage F can CORROBORATE or UNDERMINE it; it can "
             f"never replace it. Written down so that no number on these slides can "
             f"later be read as having moved it."),

            ("THE YARDSTICK, RE-ESTIMATED BUT NEVER SUBSTITUTED", DEAD_C,
             f"s*_96 = {S96:.3f} is what every verdict above uses, and it was frozen "
             f"before these cells ran. The scorer re-estimates it from whatever is on "
             f"disk now and prints both; if the two disagree, the registered one still "
             f"stands and the disagreement is itself the thing to report."),
        ],
        footer_text(commit),
        y0=0.81, gap=0.018, width=130)


def desc_min():
    desc, tr, rho = N48_MINIMUM
    return f"{desc}, {tr:,} trainable at rho_50 {rho:.3f}"


def page_next(o, p, f1_call, f0_call, f2_call, commit):
    if f0_call == "held":
        first = ("WHAT THIS LICENSES, AND WHAT IT DOES NOT", STRIP_C,
                 "Two problem sizes is not a scaling law. The claim earned here is "
                 "'the cut is free at N=48 and at N=96', not 'the cut is free'. A "
                 "third size -- or better, a size where it breaks -- is what would "
                 "turn two points into a trend, and nothing in this job tells you "
                 "which direction to look.")
    elif f0_call == "higher":
        first = ("THE RELABELLING IS THE WORK", BAND_C,
                 "Every 'smallest architecture' claim in this programme becomes an "
                 "N=48 claim, including the ladder's own ~2M floor. The next question "
                 "is WHICH cut is load-bearing at N=96 -- the MLP or the width -- and "
                 "that is a two-cell experiment against this same control, not a "
                 "sweep.")
    elif f0_call == "lower":
        first = ("THE NEXT MOVE IS A CONTROL CHECK", NOTE_C,
                 "The registration reads a lower stripped arm as an undertrained N=96 "
                 "control at 375k steps. Until the control is rerun at a longer "
                 "budget, the ordering here says nothing about architecture, and the "
                 "honest headline is about the control rather than the arm.")
    else:
        first = ("A BREAKING POINT IS A RESULT", DEAD_C,
                 "The registration says so in advance, which is the only reason it "
                 "can be said now without sounding like a consolation. What it "
                 "establishes is a BOUNDARY on the N=48 claim, and boundaries are "
                 "what this programme is looking for -- the brief asks for the "
                 "smallest architecture that still learns direct deduction, not for "
                 "the highest solve rate.")

    textpage(
        o,
        "What stage F leaves open",
        "Written before the run closed, so these are not the questions the answer "
        "happened to leave",
        [
            first,
            ("THE N=48 LADDER IS NOT CLOSED BY THIS", CTRL_C,
             "Stage K (51668331) tested whether the ~2M floor measured on the L=12 "
             "width ladder belongs to the task or only to L=12, at L=16 d160 and "
             "d128. Its verdict is on its own slides, in N=48's own yardstick; "
             "nothing here moves it either way, and nothing there moves anything "
             "here."),
            ("THE OPEN N=48 QUESTION, STILL DELIBERATELY UNBUNDLED", NOTE_C,
             "The width curve is convex on [192, 256] at L=12: all three registered "
             "interpolations overestimated rho_50 at d224 (J1). A fourth width point "
             "is the only way to know the shape of that stretch. It is worth 3 cells "
             "and it was queued behind this job, because F decides whether any N=48 "
             "number is worth generalizing in the first place."),
            ("WHAT A READER SHOULD CHECK FIRST", DEAD_C,
             f"The per-seed levels, not the medians. The seed spread at a fixed "
             f"architecture has exceeded the depth gaps throughout this programme, "
             f"and s*_96 = {S96:.3f} is a MEASURED spread, not a confidence interval. "
             f"If a gap on these slides is smaller than that bar, the reading is 'no "
             f"separable difference' -- and no page here says otherwise."),
        ],
        footer_text(commit),
        y0=0.80, gap=0.022, width=124)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_F_DIRS)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "stagef_2026-11-09"))
    a = ap.parse_args()

    cells, raw = collect96(a.scan_dir, a.boot)
    print(f"{len(cells)} N=96 cells from {len(a.scan_dir)} dirs")
    p = pair(cells)
    for tag in ("stripped", "control"):
        print(f"  {tag:<9} "
              + ("absent" if not p[tag] else
                 f"{p[tag]['s']['n']} seed(s), {p[tag]['s']['n_cross']} crossing, "
                 f"rho_50 {rho_txt(p[tag]['s'])}"))
    st = p["stripped"]["s"] if p["stripped"] else None
    ct = p["control"]["s"] if p["control"] else None
    f1_call = verdict_f1(st)
    f0_call = verdict_f0(st, ct, f1_call)
    f2_call = verdict_f2(ct)
    print(f"  branches: F1 {f1_call} | F0 {f0_call} | F2 {f2_call}")
    print(f"  headline: {title_for(f1_call, f0_call)}")

    os.makedirs(a.out, exist_ok=True)
    set_style()
    # The tokenization and trace pages are drawn at a legible N=24; the scheme is
    # size-independent and a 96-node adjacency does not fit on a slide. The note
    # says what the sequence length actually is at N=96.
    ex = example_instance(n_nodes=24, degree=5, n_colors=4, rho=0.30)
    pdf_path = os.path.join(a.out, "stagef_slides.pdf")
    with PdfPages(pdf_path) as pdf:
        o = PdfAndPng(pdf, a.out, total=7)
        page_headline(o, p, f1_call, f0_call, f2_call, a.commit)
        page_tokenization(o, ex)
        page_trace(o, ex, note="Drawn at N=24 so the adjacency fits on a slide. The "
                              "scheme is identical at N=96, where the run lives; only "
                              "the sequence gets longer.")
        page_contrast(o, p, raw, a.commit)
        page_two_sizes(o, p, a.commit)
        page_predictions(o, p, f1_call, f0_call, f2_call, a.commit)
        page_next(o, p, f1_call, f0_call, f2_call, a.commit)
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
