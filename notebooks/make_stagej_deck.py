"""Deck: stage J -- at a matched parameter budget, does SHAPE matter?

Stages A-I cut the N=48 model down one knob at a time and left two separate
findings: depth buys transition depth (the L ladder) and width is a second,
graded floor (stages H and I). Both were measured by moving ONE axis while the
parameter count moved with it, so neither could say whether the model cares
about shape or only about size. Stage J (51280737) is the controlled contrast:
two attention-only rungs 2.6% apart in trainable parameters, one shallow and
wide (L=12 d224), one deep and narrow (L=16 d192).

    python3 -m notebooks.make_stagej_deck --commit $(git rev-parse --short HEAD)

Every verdict on these slides comes from scripts/stagej_score.py, committed as
88cb7cb while 51280737 was still in its first wave -- git log is the timestamp.
SCAN_J stays out of SCAN_DIRS so J's rungs cannot pool into the frozen bar
s* = 0.1484 / 0.2918, which is quoted here and never recomputed.
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_stagehi_deck import acc_of, collect, curve, sufficiency
from scripts.minarch_analysis import SCAN_DIRS, SCAN_E, SCAN_G, SCAN_H, SCAN_I, SCAN_J
from scripts.stagec_score import PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt, rung, summarize
from scripts.stagej_score import (BAND_HI, BAND_LO, DEPTH_EVIDENCE, E2, FREE, H1, H2,
                                  HUB_RHO50, HUB_TRAINABLE, I1, J1, J2, LOO_BAR,
                                  MATCH_AT_REGISTRATION, PRED_J1_HI, PRED_J1_INTERP,
                                  PRED_J1_LO, PRED_J2_BELOW, STANDING)

FULL_C = "#13507a"      # the complete model, and the reference depth ladder
NOMLP_C = "#1a7a3c"     # attention-only
BAND_C = "#c1272d"      # the plateau, the barrier band, the free line
NOTE_C = "#6a3d9a"      # registered in advance
DEAD_C = "0.45"         # censored / never escaped

J1_C = "#b06000"        # shallow-wide, L=12 d224
J2_C = "#1a7a3c"        # deep-narrow, L=16 d192

HUB = (6, 4, "full")                # the reference architecture s* is anchored to
FULL_L12 = (12, 4, "full")          # best solver on this eval, at any size
GRID_TOP = 0.95

# The whole attention-only frontier, smallest to largest, plus the two anchors.
# d128/d160/d192 are stages H and I at L=12; d224 and L=16 d192 are stage J;
# d256 is stage E. Trainable counts are read off disk, never from this table.
FRONTIER = [(H1, "H1", "no_mlp L=12 d128", BAND_C),
            (I1, "I1", "no_mlp L=12 d160", NOTE_C),
            (H2, "H2", "no_mlp L=12 d192", "#b06000"),
            (J2, "J2", "no_mlp L=16 d192", J2_C),
            (J1, "J1", "no_mlp L=12 d224", J1_C),
            (E2, "E2", "no_mlp L=12 d256", NOMLP_C)]


def pair(cells):
    """The two matched rungs, with everything the pages need."""
    out = {}
    for key, tag in ((J1, "J1"), (J2, "J2")):
        ms = rung(cells, key)
        if not ms:
            raise SystemExit(f"{tag} has no cell on disk yet")
        s = summarize(ms)
        esc, nll = escapes(ms)
        out[tag] = dict(key=key, tag=tag, ms=ms, s=s, esc=esc, nll=nll, acc=acc_of(ms))
    return out


def frontier_rows(cells):
    rows = []
    for key, tag, name, col in FRONTIER:
        ms = rung(cells, key)
        if not ms:
            continue
        s = summarize(ms)
        rows.append(dict(key=key, tag=tag, name=name, col=col, ms=ms, s=s,
                         esc=escapes(ms)[0]))
    return rows


def page_headline(o, p, anchors, commit):
    j1, j2 = p["J1"], p["J2"]
    gap = j1["s"]["rho50"] - j2["s"]["rho50"]
    hub, best = anchors["hub"], anchors["best"]
    shrink = STANDING[2] / j2["s"]["trainable"]
    textpage(
        o,
        "At a matched budget, shape is free: only size buys anything",
        f"N=48, degree 5, 4 colours, leak-free | two attention-only rungs "
        f"{MATCH_AT_REGISTRATION * 100:.1f}% apart in parameters | J 51280737, 6 cells, 3 seeds each",
        [
            ("THE CONTRAST, AND IT IS AS CLEAN AS THIS GRID CAN MAKE IT", NOMLP_C,
             f"Shallow-wide (L=12 d224, {j1['s']['trainable']:,}) lands at rho_50 "
             f"{rho_txt(j1['s'])}; deep-narrow (L=16 d192, {j2['s']['trainable']:,}) at "
             f"{rho_txt(j2['s'])}. The difference is {abs(gap):.3f} = {abs(gap) / S_RHO50:.2f} s* "
             f"against a frozen bar of {S_RHO50}. Both cross on 3 of 3 seeds and clear finding "
             f"44's {PLATEAU} plateau on 3 of 3. Two shapes, one price, one answer: over this grid "
             f"model quality is ONE-DIMENSIONAL, and the depth and width ladders were reading the "
             f"same axis at different prices."),
            ("THE STANDING MINIMUM MOVES, AND THE CLAIM IS 'SMALLER', NOT 'BETTER'", NOTE_C,
             f"The registered rule (3 of 3 crossing AND median at or below {FREE}) fires for BOTH "
             f"rungs, so the tie-break named at scoring time -- smaller parameter count -- gives "
             f"J2. The minimum goes from {STANDING[1]} at {STANDING[2]:,} / {STANDING[3]} to "
             f"attention-only L=16 d192 at {j2['s']['trainable']:,} / {rho_txt(j2['s'])}: "
             f"{shrink:.2f}x smaller, {HUB_TRAINABLE / j2['s']['trainable']:.2f}x below the hub, "
             f"rho_50 moving {j2['s']['rho50'] - STANDING[3]:+.3f} = "
             f"{abs(j2['s']['rho50'] - STANDING[3]) / S_RHO50:.2f} s* -- below the bar. The two J "
             f"rungs differ by less than the job's own 5% tolerance, so the real statement is that "
             f"the minimum is ~2,500,000 AT EITHER SHAPE."),
            ("THE PREDICTION THAT MISSED, REPORTED FIRST BECAUSE IT MISSED", BAND_C,
             f"J1 was registered to land in [{PRED_J1_LO}, {PRED_J1_HI}], from three interpolations "
             f"of its neighbours ({', '.join(f'{k} {v}' for k, v in PRED_J1_INTERP.items())}). It "
             f"came in at {rho_txt(j1['s'])} -- MISSED LOW, all three falsified in the same "
             f"direction. d224 beats any straight line through d192 and d256, so the width curve is "
             f"CONVEX there: the gain is bought in the first 32 units of width, not the last. "
             f"Nothing on this axis should be interpolated again without a fourth point."),
            ("WHAT IS STILL NOT CLAIMED", DEAD_C,
             f"The best solver on this eval at any size is still the COMPLETE model at L=12, "
             f"rho_50 {rho_txt(best['s'])} at {best['s']['trainable']:,} trainable; 'minimal' has "
             f"never meant 'best'. Every number here is N=48 -- whether any of it survives a second "
             f"problem size is stage F, running now as 51656823."),
        ],
        f"rho_50 = realized hint density at which solve rate crosses 0.5 on a 14-point x "
        f"400-instance grid, LOWER IS BETTER, reported CENSORED when no bin crosses. mean_solve = "
        f"unweighted mean over the 14 bins, which never censors. s* = {S_RHO50} / {S_MEAN} was "
        f"frozen before any of these cells ran, and the free line {FREE} is the hub plus one s*. "
        f"Commit {commit}.",
        width=122)


def page_contrast(o, p, raw, commit):
    """The matched pair: every seed, both transition curves, both training curves."""
    j1, j2 = p["J1"], p["J2"]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Two shapes at one price, and they are indistinguishable",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, "Left: every seed, not just the median. Middle: the curves rho_50 is read "
                         "off. Right: training, where the two arms also coincide",
             ha="center", size=MED - 3, color="0.3")

    # --- left: per-seed rho_50 at a matched budget -------------------------
    ax = fig.add_axes([0.065, 0.28, 0.225, 0.52])
    for r in (j1, j2):
        col = J1_C if r["tag"] == "J1" else J2_C
        x = r["s"]["trainable"]
        for m in r["ms"]:
            if m["flag"] == "ok":
                ax.plot(x, m["rho50"], "o", ms=8, color=col, alpha=0.75)
            else:
                ax.plot(x, GRID_TOP, "^", ms=9, mfc="none", mew=1.8, color=col)
        ax.plot([x], [r["s"]["rho50"]], "_", ms=34, mew=3, color=col)
    # A legend rather than point annotations: the two medians are 0.001 apart,
    # so any label anchored to them lands on the other arm's seed dots.
    ax.legend(handles=[Line2D([], [], color=J1_C, marker="o", ls="", ms=8,
                              label=f"J1  {j1['name_short']}"),
                       Line2D([], [], color=J2_C, marker="o", ls="", ms=8,
                              label=f"J2  {j2['name_short']}")],
              loc="center left", framealpha=0.95, fontsize=SMALL - 1)
    ax.errorbar(2.61e6, 0.64, yerr=S_RHO50 / 2, color="0.25", capsize=5, lw=1.8)
    ax.text(2.60e6, 0.64 + S_RHO50 / 2, f"s* {S_RHO50} ", size=SMALL - 2, color="0.25",
            va="bottom", ha="right")
    ax.axhline(FREE, color=BAND_C, lw=1.5, ls="--")
    ax.text(0.99, FREE, f" free {FREE}", size=SMALL - 2, color=BAND_C, va="bottom",
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("rho_50   (lower is better)")
    ax.set_ylim(0.48, 0.82)
    ax.set_xlim(2.42e6, 2.68e6)
    ax.set_xticks([2.5e6, 2.6e6])
    ax.set_xticklabels(["2.50M", "2.60M"])
    ax.grid(alpha=0.25)
    ax.set_title("The whole result in one panel:\nmedian bars differ by "
                 f"{abs(j1['s']['rho50'] - j2['s']['rho50']):.3f}", size=MED - 4)

    # --- middle: the transition curves -------------------------------------
    ax2 = fig.add_axes([0.375, 0.28, 0.225, 0.52])
    for r in (j1, j2):
        col = J1_C if r["tag"] == "J1" else J2_C
        for m in r["ms"]:
            seed_rows = next(v for k, v in raw.items()
                             if k[:-1] == r["key"] and v[0]["path"] == m["path"])
            pts = sorted((x["realized_rho"], x["solve_rate"]) for x in seed_rows)
            ax2.plot([q[0] for q in pts], [q[1] for q in pts], "-", lw=1.7, color=col, alpha=0.8)
    ax2.axhline(0.5, color="0.5", lw=1.2, ls=":")
    ax2.text(0.015, 0.5, " solve rate 0.5", size=SMALL - 1, color="0.4", va="bottom",
             transform=ax2.get_yaxis_transform())
    ax2.set_xlabel("realized hint density rho")
    ax2.set_ylabel("solve rate")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(alpha=0.25)
    ax2.set_title("Six curves, identical 14 x 400 grid", size=MED - 2)
    ax2.legend(handles=[Line2D([], [], color=J1_C, lw=2.2, label="J1  L=12 d224"),
                        Line2D([], [], color=J2_C, lw=2.2, label="J2  L=16 d192")],
               loc="upper left", framealpha=0.95, fontsize=SMALL - 1)

    # --- right: training curves --------------------------------------------
    ax3 = fig.add_axes([0.685, 0.28, 0.225, 0.52])
    for r in (j1, j2):
        col = J1_C if r["tag"] == "J1" else J2_C
        for m in r["ms"]:
            nll, _acc = curve(m["path"])
            ax3.plot(range(1, len(nll) + 1), nll, "-", lw=1.5, color=col, alpha=0.8)
    ax3.axhline(PLATEAU, color=BAND_C, lw=1.6, ls="--")
    ax3.text(0.98, PLATEAU, f" finding 44 plateau {PLATEAU} ", size=SMALL - 2, color=BAND_C,
             va="bottom", ha="right", transform=ax3.get_yaxis_transform())
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("train NLL")
    ax3.grid(alpha=0.25)
    ax3.set_title(f"All 6 cells escape the plateau\nfinal NLL "
                  f"{min(j1['nll'] + j2['nll']):.4f}-{max(j1['nll'] + j2['nll']):.4f}",
                  size=MED - 4)

    footer(fig, f"Open triangles would mark censored seeds; there are none in stage J. The median "
                f"bar is the 3-seed median rho_50, the quantity every verdict is read on. Both "
                f"arms were trained on the same 240k-instance pool for 50 epochs at constant lr "
                f"1e-3, batch 32, no warmup -- the only things that differ between them are "
                f"num_layers and d_model. Commit {commit}.")
    o.savefig(fig)


def page_frontier(o, rows, anchors, p, commit):
    """Size against rho_50 for the whole attention-only programme."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The size-quality frontier, and where stage J put the knee",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, "Every attention-only rung this programme has run at N=48, against the "
                         "two complete-model anchors",
             ha="center", size=MED - 3, color="0.3")

    ax = fig.add_axes([0.085, 0.30, 0.40, 0.50])
    for r in rows:
        x = r["s"]["trainable"]
        for m in r["ms"]:
            if m["flag"] == "ok":
                ax.plot(x, m["rho50"], "o", ms=6, color=r["col"], alpha=0.55)
            else:
                ax.plot(x, GRID_TOP, "^", ms=9, mfc="none", mew=1.8, color=r["col"])
        if r["s"]["rho50_readable"]:
            ax.plot([x], [r["s"]["rho50"]], "o", ms=12, mfc="white", mew=2.6, color=r["col"],
                    zorder=5)
            ax.annotate(r["tag"], xy=(x, r["s"]["rho50"]), xytext=(0, 14),
                        textcoords="offset points", ha="center", size=SMALL,
                        color=r["col"], weight="bold")
        else:
            ax.annotate(f"{r['tag']} CENSORED", xy=(x, GRID_TOP), xytext=(6, 6),
                        textcoords="offset points", ha="left", size=SMALL - 1, color=r["col"])
    hub, best = anchors["hub"], anchors["best"]
    for a, name, mark in ((hub, "hub  full L=6 d256", "s"), (best, "full L=12 d256", "D")):
        ax.plot([a["s"]["trainable"]], [a["s"]["rho50"]], mark, ms=11, color=FULL_C, zorder=5)
        ax.annotate(name, xy=(a["s"]["trainable"], a["s"]["rho50"]), xytext=(0, -24),
                    textcoords="offset points", ha="center", size=SMALL - 1, color=FULL_C)
    ax.axhspan(BAND_LO, BAND_HI, color=BAND_C, alpha=0.08, zorder=0)
    ax.text(0.99, (BAND_LO + BAND_HI) / 2, " barrier band ", size=SMALL - 2, color=BAND_C,
            va="center", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(FREE, color=BAND_C, lw=1.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters  (log scale)")
    ax.set_ylabel("rho_50   (lower is better)")
    ax.set_ylim(0.33, 1.00)
    ax.grid(alpha=0.25)
    ax.set_title("A cliff at the bottom, a plateau across the top:\nbelow ~2M the model stops "
                 "solving; above it, size barely moves rho_50", size=MED - 4)

    j2 = p["J2"]
    block = [
        "  rung                    trainable    rho_50    cross   plateau esc",
        "  " + "-" * 62,
    ]
    for r in rows:
        block.append(f"  {r['name']:<22} {r['s']['trainable']:>10,}   {rho_txt(r['s']):>8}"
                     f"   {r['s']['n_cross']}/{r['s']['n']}      {r['esc']}/{r['s']['n']}")
    block.append("  " + "-" * 62)
    for a, name in ((hub, "full L=6 d256 (hub)"), (best, "full L=12 d256")):
        block.append(f"  {name:<22} {a['s']['trainable']:>10,}   {rho_txt(a['s']):>8}"
                     f"   {a['s']['n_cross']}/{a['s']['n']}      -")
    fig.text(0.545, 0.78, "\n".join(block), size=SMALL - 1, va="top", family="monospace")

    note = (
        f"WHAT THE SHAPE OF THIS CURVE SAYS. From {rows[-1]['s']['trainable']:,} down to "
        f"{j2['s']['trainable']:,} -- a {rows[-1]['s']['trainable'] / j2['s']['trainable']:.2f}x cut "
        f"-- rho_50 does not move outside the bar. Below that the curve turns over hard: d192 at "
        f"L=12 is already in the barrier band, d160 is deeper into it, and d128 does not train on 2 "
        f"of 3 seeds. The useful reading is not 'smaller is free' but 'there is a floor near 2M "
        f"parameters on this task, and everything above it is flat'. Stage J's two rungs sit "
        f"exactly on the last flat step before the drop, which is why the matched contrast is "
        f"informative: if shape mattered anywhere, it would matter at the edge."
    )
    fig.text(0.545, 0.40, "\n".join(textwrap.wrap(note, 74)),
             size=SMALL, va="top", linespacing=1.5)
    footer(fig, f"Open triangles are censored rungs, drawn at the grid edge {GRID_TOP} because "
                f"that is all the eval knows about them. All cells attention-only unless marked "
                f"'full'; all at 240k instances x 50 epochs, batch 32, constant lr 1e-3, no "
                f"warmup, --shuffle-adj --fall. Commit {commit}.")
    o.savefig(fig)


def page_predictions(o, p, suf, commit):
    j1, j2 = p["J1"], p["J2"]
    gap = j1["s"]["rho50"] - j2["s"]["rho50"]
    d_h2 = j2["s"]["rho50"] - PRED_J2_BELOW
    textpage(
        o,
        "Every stage J prediction, scored against what was registered",
        f"stagej_score.py was committed (88cb7cb) while 51280737 was in its first wave -- "
        f"no verdict was chosen after seeing a number",
        [
            (f"J0  HELD -- shape is not a free parameter  (registered: |J1-J2| <= s* = {S_RHO50})",
             NOMLP_C,
             f"Observed {abs(gap):.3f} = {abs(gap) / S_RHO50:.2f} s*. The budgets came back "
             f"{j1['s']['trainable']:,} vs {j2['s']['trainable']:,}, "
             f"{100 * abs(j1['s']['trainable'] - j2['s']['trainable']) / j2['s']['trainable']:.1f}% "
             f"apart against the 5% tolerance that would have VOIDED this contrast, so it is a "
             f"controlled comparison and not a size verdict wearing a hat. The falsifier named a "
             f"winner in advance in either direction; neither fired."),
            (f"J1  MISSED LOW  (registered: median in [{PRED_J1_LO}, {PRED_J1_HI}] on 3/3 cross, 3/3 escape)",
             BAND_C,
             f"Observed {rho_txt(j1['s'])} on {j1['s']['n_cross']}/3 crossing and {j1['esc']}/3 "
             f"escapes -- the counts held, the level did not. All three registered interpolations "
             f"({', '.join(f'{k} {v}' for k, v in PRED_J1_INTERP.items())}) are falsified in the "
             f"same direction, which is the informative part: the width curve is convex on "
             f"[192, 256], so interpolating d_model between measured rungs systematically "
             f"UNDERESTIMATES the in-between model. No fourth point, no more interpolation."),
            (f"J2  HELD -- depth past L=12 still pays  (registered: below H2's {PRED_J2_BELOW} on >= 2/3 escapes)",
             NOTE_C,
             f"Observed {rho_txt(j2['s'])} on {j2['esc']}/3 escapes, {d_h2:+.3f} = "
             f"{abs(d_h2) / S_RHO50:.2f} s* against H2 at the SAME width -- lower and separable. "
             f"The depth evidence this rested on stopped at L=12 "
             f"({', '.join(f'L={k} {v}' for k, v in sorted(DEPTH_EVIDENCE.items()))}), so L=16 was "
             f"an extrapolation of the ladder and it came back on the line. This is the first "
             f"leak-free point above L=12 at N=48."),
            (f"J3  FIRED  |  J4  HELD -- {suf['n_exp']}/{suf['n_rungs']} held-out rungs on the curve, worst miss {suf['worst']:.4f} vs bar {suf['bar']:.4f}",
             FULL_C,
             f"J3's rule (3/3 crossing AND median at or below {FREE}) fired for both rungs and the "
             f"minimum moved to J2 on the pre-named tie-break. J4 refits the depth ladder's "
             f"accuracy-to-solve curve over all nine scan dirs and asks whether J's rungs sit on "
             f"it: {suf['n_exp']} of {suf['n_rungs']} well-determined held-out rungs do, over "
             f"{suf['n_cells']} cells, {suf['n_sharp']} of them discriminating. That is UP from "
             f"21/21 over 46 before stage J. Token accuracy remains a sufficient statistic over "
             f"this grid -- including across a shape change it was never fit on."),
        ],
        f"The LOO bar {LOO_BAR} is the leave-one-out RMS of the depth ladder's own fit and was "
        f"frozen with it. 'Discriminating' means the curve predicts solve > 0.10 for that cell, so "
        f"landing on it is a real constraint rather than two numbers agreeing near zero. "
        f"Commit {commit}.",
        width=124, y0=0.81, gap=0.026)


def page_next(o, p, rows, suf, commit):
    j2 = p["J2"]
    textpage(
        o,
        "Where this leaves the programme, and what 51656823 is testing now",
        f"stage J closes the N=48 ladder | stage F (51656823) asks whether any of it survives a "
        f"second problem size",
        [
            ("THE N=48 LADDER IS CLOSED, AND THE CLOSING ARGUMENT IS J0", NOMLP_C,
             f"There is no further architecture cut this grid can resolve. Width below d192 does "
             f"not train (stage H), the d160-d192 stretch buys nothing a seed could not (stage I), "
             f"and stage J has now shown that trading depth against width at a fixed budget "
             f"changes nothing measurable. What is left to vary is size, and the frontier is flat "
             f"from ~2.5M up to the hub's {HUB_TRAINABLE:,} and falls off a cliff below. The "
             f"standing minimum is attention-only L=16 d192, {j2['s']['trainable']:,} trainable at "
             f"rho_50 {rho_txt(j2['s'])}."),
            ("STAGE F (51656823): THE ONE QUESTION A SINGLE PROBLEM SIZE CANNOT ANSWER", NOTE_C,
             f"Every cut above was measured at N=48. F runs the surviving architecture at N=96 "
             f"against a PAIRED complete-model control -- 4 cells, 36.5 GPU-h, cheap only because "
             f"the control already exists: 8 N=96 cells trained leak-free at the same 240k x 50 "
             f"budget on the identical 14 x 400 grid (full L=12 d256 -> 0.621 over two seeds). F "
             f"buys the stripped arm at 3 seeds plus the missing THIRD control seed. Registered: "
             f"|stripped - control| <= s*_96 = 0.119, frozen from the three two-seed N=96 rungs "
             f"that exist today. s* = {S_RHO50} is an N=48 number and is NOT reused there."),
            ("THE FALSIFIER THAT MATTERS, STATED BEFORE THE NUMBERS ARRIVE", BAND_C,
             f"If the stripped arm comes back HIGHER than the N=96 control by more than 0.119, the "
             f"N=48 minimum does not transfer and every 'smallest architecture' claim in this "
             f"programme gets relabelled as an N=48 claim -- which is a real possible outcome, not "
             f"a formality. If any of the three stripped seeds CENSORS, the breaking point is the "
             f"result and it becomes the headline. And finding 44's {PLATEAU} plateau is an N=48 "
             f"number: N=96's own plateau has never been measured, so no escape count will be "
             f"quoted for F."),
            ("WHAT IS NOT BEING CLAIMED, AND THE OPEN N=48 QUESTION J LEFT", DEAD_C,
             f"J4's sufficiency result is CORRELATIONAL and is a statement about this grid's "
             f"range. 'One-dimensional' means no cut opened a second, deduction-specific way to "
             f"fail; it does not mean these are the same model. The open N=48 question is J1's: "
             f"the width curve is convex, so a fourth width point is the only way to know the "
             f"shape of the d192-d256 stretch. That is deliberately NOT bundled into F, which "
             f"changes the problem size and must change nothing else."),
        ],
        f"Stage J: 6 cells, 3 seeds per rung, 51280737, all COMPLETED 2026-11-06. Sufficiency "
        f"refit over 9 scan dirs: {suf['n_exp']}/{suf['n_rungs']} rungs, {suf['n_cells']} cells, "
        f"worst discriminating miss {suf['worst']:.4f} against bar {suf['bar']:.4f}. Commit {commit}.",
        width=124, y0=0.825, gap=0.021)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+",
                    default=SCAN_DIRS + [SCAN_E, SCAN_G, SCAN_H, SCAN_I, SCAN_J])
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "stagej_2026-11-09"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--commit", default="(uncommitted)")
    args = ap.parse_args()

    set_style()
    cells, raw = collect(args.scan_dir, args.boot)
    p = pair(cells)
    p["J1"]["name_short"] = "L=12 d224"
    p["J2"]["name_short"] = "L=16 d192"
    for tag in ("J1", "J2"):
        if p[tag]["s"]["n"] < 3:
            raise SystemExit(f"{tag} has {p[tag]['s']['n']} seeds; the registered rule is "
                             f"median-vs-median at three seeds")
    rows = frontier_rows(cells)
    anchors = {"hub": dict(s=summarize(rung(cells, HUB))),
               "best": dict(s=summarize(rung(cells, FULL_L12)))}
    suf = sufficiency(args.scan_dir)
    print(f"{len(cells)} cells | J1 {rho_txt(p['J1']['s'])} ({p['J1']['s']['n_cross']}/3 cross, "
          f"{p['J1']['esc']}/3 esc, {p['J1']['s']['trainable']:,}) | "
          f"J2 {rho_txt(p['J2']['s'])} ({p['J2']['s']['n_cross']}/3 cross, "
          f"{p['J2']['esc']}/3 esc, {p['J2']['s']['trainable']:,})")
    print(f"sufficiency: {suf['n_exp']}/{suf['n_rungs']} rungs, {suf['n_cells']} cells, "
          f"{suf['n_sharp']} discriminating, worst {suf['worst']:.4f} vs bar {suf['bar']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=7)
        page_headline(o, p, anchors, args.commit)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task every "
                               "cell in this deck is trained and scored on. Stage J changes "
                               "nothing about it -- only num_layers and d_model move, and they "
                               "move against each other so the two budgets land 2.6% apart, "
                               "inside the 5% tolerance the contrast was registered under.")
        page_contrast(o, p, raw, args.commit)
        page_frontier(o, rows, anchors, p, args.commit)
        page_predictions(o, p, suf, args.commit)
        page_next(o, p, rows, suf, args.commit)
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
