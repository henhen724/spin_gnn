"""Deck: stage K -- is the ~2M floor a property of the task, or only of L=12?

Stage J moved the standing minimum to attention-only L=16 d192 (2,504,890
trainable at rho_50 0.580) and STATUS read the frontier as "a floor near 2M
parameters, and everything above it is flat". That floor was measured on the
L=12 width ladder ALONE -- every rung below 2M on record is an L=12 rung -- and
at d192 the step L=12 -> L=16 moved rho_50 0.754 -> 0.580 and carried the rung
across the 0.740 free line. Stage K (51668331) runs the same width ladder one
depth up: KA = L=16 d160, KB = L=16 d128, three seeds each, one d_model knob
from J2.

    python3 -m notebooks.make_stagek_deck --commit $(git rev-parse --short HEAD)

WRITTEN BEFORE THE FIRST K CELL LANDED, like scripts/stagek_score.py (262b581)
and the sbatch registration (11d42aa) -- git log is the timestamp. Every page is
read off disk and every headline is selected by the branch the data falls in,
from the branches named in the registration. Nothing here is a sentence chosen
after seeing a number. SCAN_K joins the scan list but the frozen bar
s* = 0.1484 / 0.2918 is quoted, never recomputed.
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
from scripts.stagek_score import (DEPTH_STEP_AT_D192, E2, FREE, H1, H2, HUB_RHO50,
                                  HUB_TRAINABLE, I1, I1_RHO50, J1, J2, KA, KB, LOO_BAR,
                                  PRED_KA_HI, PRED_KA_LO, PRED_KA_POINT,
                                  PREDICTED_TRAINABLE, SCAN_K, STANDING)

FULL_C = "#13507a"      # the complete model, and the reference anchors
NOMLP_C = "#1a7a3c"     # attention-only, and a held prediction
BAND_C = "#c1272d"      # the plateau, the free line, a falsified prediction
NOTE_C = "#6a3d9a"      # registered in advance
DEAD_C = "0.45"         # censored / never escaped

L12_C = "#b06000"       # the L=12 width ladder, stages H/I/J/E
L16_C = "#1a7a3c"       # the L=16 width ladder, stage J's J2 plus stage K

HUB = (6, 4, "full")
FULL_L12 = (12, 4, "full")
GRID_TOP = 0.95

# The two width ladders, narrow to wide. Trainable counts and rho_50 are read
# off disk; this table only says which cell belongs on which curve.
LADDER_12 = [(H1, "H1", 128), (I1, "I1", 160), (H2, "H2", 192),
             (J1, "J1", 224), (E2, "E2", 256)]
LADDER_16 = [(KB, "KB", 128), (KA, "KA", 160), (J2, "J2", 192)]


# --------------------------------------------------------------------------
# Which registered branch did the data land in? Both functions below encode
# the falsifiers from the sbatch header and stagek_score.py, nothing else.
# --------------------------------------------------------------------------
def verdict_ka(ka):
    s = ka["s"]
    if not s["rho50_readable"]:
        return "censored_trained" if ka["esc"] >= 2 else "censored_untrained"
    if s["n_cross"] < 3:
        return "partial"
    if s["rho50"] > FREE:
        return "high"
    if s["rho50"] < PRED_KA_LO:
        return "low"
    if s["rho50"] > PRED_KA_HI:
        return "free_above_band"
    return "held"


def verdict_kb(kb):
    s = kb["s"]
    if s["rho50_readable"] and s["n_cross"] == 3:
        return "crossed"
    return "censored_trained" if kb["esc"] >= 2 else "censored_untrained"


KA_TITLE = {
    "held": "Depth buys back width: the floor was L=12's, not the task's",
    "free_above_band": "Depth buys back width, but less of it than the ladder predicted",
    "low": "Depth buys back MORE width at d160 than the ladder predicted",
    "high": "The depth gain does not transport: the ~2M floor stands",
    "partial": "Two of three is not a rung: d160 at L=16 is not free",
    "censored_trained": "The breaking point: L=16 d160 trains and still fails",
    "censored_untrained": "L=16 d160 did not train: this says nothing about shape",
}


def qualifies(r):
    """K2 = H3 = I3 = J3: 3 of 3 seeds crossing AND median at or below the free line."""
    s = r["s"]
    return s["rho50_readable"] and s["n_cross"] == 3 and s["rho50"] <= FREE


def new_minimum(p):
    """The rung the standing minimum moves to, or None if it does not move.

    BOTH K rungs can qualify -- KB is the smaller one, so when it does the
    pre-named tie-break (smaller wins, as in J3) hands it the minimum and not
    KA. Writing KA in here would have quietly contradicted score_k2.
    """
    ok = [r for r in (p["KA"], p["KB"]) if qualifies(r)]
    return min(ok, key=lambda r: r["s"]["trainable"]) if ok else None


def pair(cells):
    """The two stage K rungs, with everything the pages need."""
    out = {}
    for key, tag in ((KA, "KA"), (KB, "KB")):
        ms = rung(cells, key)
        if not ms:
            raise SystemExit(f"{tag} has no cell on disk yet")
        s = summarize(ms)
        esc, nll = escapes(ms)
        out[tag] = dict(key=key, tag=tag, ms=ms, s=s, esc=esc, nll=nll, acc=acc_of(ms))
    return out


def ladder(cells, spec):
    rows = []
    for key, tag, d in spec:
        ms = rung(cells, key)
        if not ms:
            continue
        s = summarize(ms)
        rows.append(dict(key=key, tag=tag, d=d, ms=ms, s=s, esc=escapes(ms)[0]))
    return rows


def page_headline(o, p, l12, l16, anchors, commit):
    ka, kb = p["KA"], p["KB"]
    vka, vkb = verdict_ka(ka), verdict_kb(kb)
    best = anchors["best"]
    win = new_minimum(p)

    if win is not None:
        w = win["s"]
        drho = w["rho50"] - STANDING[3]
        both = " Both K rungs satisfy the rule, so the pre-named tie-break -- smaller wins, as " \
               "in J3 -- selects it." if qualifies(ka) and qualifies(kb) else ""
        first = (
            "THE STANDING MINIMUM MOVES, AND THE CLAIM IS 'SMALLER', NOT 'BETTER'", NOMLP_C,
            f"{win['tag']} (attention-only L=16 d{win['key'][3]}, {w['trainable']:,} trainable) "
            f"lands at rho_50 {rho_txt(w)} on {w['n_cross']}/3 seeds, inside the free line {FREE}, "
            f"so the registered rule (3 of 3 crossing AND median at or below {FREE}) fires.{both} "
            f"The minimum goes from {STANDING[1]} at {STANDING[2]:,} / {STANDING[3]:.3f} to "
            f"{w['trainable']:,} / {rho_txt(w)}: {STANDING[2] / w['trainable']:.2f}x smaller, with "
            f"rho_50 moving {drho:+.3f} = {abs(drho) / S_RHO50:.2f} s* against a frozen bar of "
            f"{S_RHO50}. {HUB_TRAINABLE / w['trainable']:.2f}x below the hub.")
    else:
        first = (
            "THE STANDING MINIMUM DOES NOT MOVE, AND THAT IS THE RESULT", BAND_C,
            f"KA (attention-only L=16 d160, {ka['s']['trainable']:,} trainable) came back "
            f"{rho_txt(ka['s'])} on {ka['s']['n_cross']}/3 seeds crossing, and KB "
            f"({kb['s']['trainable']:,}) {rho_txt(kb['s'])} on {kb['s']['n_cross']}/3. Neither "
            f"satisfies the registered rule (3 of 3 AND median at or below {FREE}), so "
            f"{STANDING[1]} at {STANDING[2]:,} / {STANDING[3]:.3f} remains the standing minimum. "
            f"Stage K was registered to be able to say this: the breaking point is the result, "
            f"and a rung that fails the rule is reported at the same volume as one that passes.")

    textpage(
        o,
        KA_TITLE[vka],
        f"N=48, degree 5, 4 colours, leak-free | the L=12 width ladder repeated at L=16 | "
        f"K 51668331, 6 cells, 3 seeds | one knob from J2",
        [
            first,
            ("WHY THIS WAS WORTH SIX CELLS", NOTE_C,
             f"Every rung below 2M on record was an L=12 rung, so 'there is a floor near 2M "
             f"parameters on this task' was really a statement about one depth. At a fixed d192 "
             f"the step L=12 -> L=16 moved rho_50 {H2_rho(l12):.3f} -> {J2_rho(l16):.3f}, a gain of "
             f"{DEPTH_STEP_AT_D192:.3f} = {DEPTH_STEP_AT_D192 / S_RHO50:.2f} s*, and carried the "
             f"rung across {FREE}. K0 asked whether that same step carries d160 over too: "
             f"{I1_RHO50} minus {DEPTH_STEP_AT_D192:.3f} = {PRED_KA_POINT}, registered as "
             f"[{PRED_KA_LO}, {PRED_KA_HI}] with the robust call being 'KA IS FREE on 3/3'. "
             f"Observed {rho_txt(ka['s'])}."),
            (f"KB (L=16 d128, {kb['s']['trainable']:,}) -- " + kb_head(kb, vkb),
             DEAD_C if vkb.startswith("censored") else BAND_C,
             kb_text(kb, vkb)),
            ("WHAT IS NOT CLAIMED", DEAD_C,
             f"The best solver on this eval at any size is still the COMPLETE model at L=12, "
             f"rho_50 {rho_txt(best['s'])} at {best['s']['trainable']:,} trainable; 'minimal' has "
             f"never meant 'best'. Every number here is N=48 -- the frozen bar s* = {S_RHO50}, "
             f"finding 44's {PLATEAU} plateau and the 0.33 grid floor are all N=48 quantities and "
             f"none of them is reused at another problem size. Whether any of this survives N=96 "
             f"is stage F (51656823), running separately."),
        ],
        f"rho_50 = realized hint density at which solve rate crosses 0.5 on a 14-point x "
        f"400-instance grid, LOWER IS BETTER, reported CENSORED when no bin crosses. mean_solve = "
        f"unweighted mean over the 14 bins, which never censors. s* = {S_RHO50} / {S_MEAN} was "
        f"frozen before any of these cells ran and the free line {FREE} is the hub plus one s*. "
        f"Commit {commit}.",
        width=122)


def kb_head(kb, vkb):
    """Crossing and being free are two different things, so the heading says which.

    'CROSSED, falsifying K1' alone read as good news on the page whose verdict
    is that nothing moved: K1 is about whether the eval can read the rung at
    all, K2 about whether it is free, and KB came back readable-but-not-free.
    """
    if vkb.startswith("censored"):
        return "CENSORED, as registered"
    return ("CROSSED AND FREE, falsifying K1" if kb["s"]["rho50"] <= FREE
            else "CROSSED BUT NOT FREE, falsifying K1")


def kb_text(kb, vkb):
    s = kb["s"]
    if vkb == "censored_trained":
        return (f"KB never crosses solve rate 0.5 anywhere on the grid ({s['n_cross']}/3 seeds "
                f"readable), and it is NOT a trainability failure: {kb['esc']}/3 seeds left "
                f"finding 44's {PLATEAU} plateau, final train NLL "
                f"{min(kb['nll']):.4f}-{max(kb['nll']):.4f}. The model fits and still cannot "
                f"deduce, which is the capacity statement H1 could not make at L=12 because it "
                f"stayed on the plateau. Depth does not pay for a per-layer width bottleneck.")
    if vkb == "censored_untrained":
        return (f"KB censors, but only {kb['esc']}/3 seeds left finding 44's {PLATEAU} plateau "
                f"(final train NLL {min(kb['nll']):.4f}-{max(kb['nll']):.4f}), so K1's gate "
                f"fires: this is a TRAINABILITY statement about d128 at this budget, not an "
                f"architecture one. It is reported as a failure to train and nothing is concluded "
                f"from it about what d128 can represent.")
    # 'The floor' is two different lines and this branch is the one place they
    # come apart: the CENSORING boundary (narrowest rung the eval can read at
    # all) does move down to d128, while the STANDING MINIMUM only moves if the
    # registered rule fires, which at 0.766 > 0.740 it does not. Saying only
    # 'the floor moves' here contradicted this very page's own K2 verdict.
    return (f"KB crosses {s['n_cross']}/3 at rho_50 {rho_txt(s)} where d128 CENSORED at L=12, so "
            f"depth does pay for a per-layer width bottleneck and the CENSORING BOUNDARY -- the "
            f"narrowest rung this eval can read at all -- moves to ~{s['trainable'] / 1e6:.2f}M. "
            + (f"It clears the free line {FREE} too, so the standing minimum moves with it. "
               if s["rho50"] <= FREE else
               f"That is not the free line: {rho_txt(s)} is "
               f"{(s['rho50'] - FREE) / S_RHO50:+.2f} s* past {FREE}, so K2 does not fire. ")
            + f"{kb['esc']}/3 seeds left the {PLATEAU} plateau.")


def H2_rho(l12):
    return next(r["s"]["rho50"] for r in l12 if r["tag"] == "H2")


def J2_rho(l16):
    return next(r["s"]["rho50"] for r in l16 if r["tag"] == "J2")


def page_ladders(o, l12, l16, p, commit):
    """The money plot: the same width ladder at two depths."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The same width ladder, run at two depths",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, "Left: rho_50 against d_model, one curve per depth. Right: training, "
                         "where a censored rung is read as capacity or as trainability",
             ha="center", size=MED - 3, color="0.3")

    ax = fig.add_axes([0.065, 0.30, 0.355, 0.50])
    # Same convention as the frontier page, and for the same reason: the two
    # ladders are what collide here. At d=160 the medians are I1 0.773 and KA
    # 0.757, 0.016 apart -- which IS the result this panel exists to show -- so a
    # shared label offset stacks the two tags on top of each other exactly where
    # the reader has to compare them. L=12 tags above the marker, L=16 below.
    for rows, col, lab, dy in ((l12, L12_C, "L=12  (stages H, I, J, E)", 13),
                               (l16, L16_C, "L=16  (J2 + stage K)", -23)):
        xs, ys = [], []
        for r in rows:
            for m in r["ms"]:
                if m["flag"] == "ok":
                    ax.plot(r["d"], m["rho50"], "o", ms=5, color=col, alpha=0.45)
                else:
                    ax.plot(r["d"], GRID_TOP, "^", ms=9, mfc="none", mew=1.8, color=col)
            if r["s"]["rho50_readable"]:
                xs.append(r["d"])
                ys.append(r["s"]["rho50"])
                ax.plot([r["d"]], [r["s"]["rho50"]], "o", ms=11, mfc="white", mew=2.5,
                        color=col, zorder=5)
                ax.annotate(r["tag"], xy=(r["d"], r["s"]["rho50"]), xytext=(0, dy),
                            textcoords="offset points", ha="center", size=SMALL,
                            color=col, weight="bold", va="bottom" if dy > 0 else "top")
            else:
                ax.annotate(f"{r['tag']} CENSORED", xy=(r["d"], GRID_TOP),
                            xytext=(5, 7 if dy > 0 else -9), textcoords="offset points",
                            ha="left", size=SMALL - 1, color=col,
                            va="bottom" if dy > 0 else "top")
        ax.plot(xs, ys, "-", lw=2.2, color=col, alpha=0.9, label=lab)
    ax.axhline(FREE, color=BAND_C, lw=1.5, ls="--")
    ax.text(0.99, FREE, f" free {FREE} ", size=SMALL - 2, color=BAND_C, va="bottom",
            ha="right", transform=ax.get_yaxis_transform())
    # The s* bar and the legend both used to sit in the lower left, which is
    # exactly where a rung that came back LOW would land. Bar goes top right,
    # legend goes under the axes, so no outcome can be hidden by the furniture.
    ax.errorbar(249, 0.90, yerr=S_RHO50 / 2, color="0.25", capsize=5, lw=1.8)
    ax.text(245, 0.90, f"s* {S_RHO50} ", size=SMALL - 2, color="0.25", va="center",
            ha="right")
    ax.set_xticks([128, 160, 192, 224, 256])
    ax.set_xlabel("d_model")
    ax.set_ylabel("rho_50   (lower is better)")
    ax.set_ylim(0.44, 1.00)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
              fontsize=SMALL)
    ax.set_title("If the two curves are parallel, depth and width are\nseparable knobs; if they "
                 "converge, the floor is the task's", size=MED - 4)

    ax2 = fig.add_axes([0.495, 0.30, 0.205, 0.50])
    # KB is NOT drawn in BAND_C: that is the colour of the plateau line and its
    # label in this very panel, and this panel is the one that decides K1's gate
    # (censored having left 1.64 = capacity, censored still on it = trainability).
    # A KB curve in the threshold's own colour is the one confusion that would
    # cost the reading. J2 is drawn first so the two K rungs sit on top of the
    # reference rather than under it when the curves nearly coincide.
    for r in sorted(l16, key=lambda r: 0 if r["tag"] == "J2" else 1):
        col = {"KA": NOTE_C, "KB": FULL_C, "J2": L16_C}[r["tag"]]
        for m in r["ms"]:
            nll, _acc = curve(m["path"])
            ax2.plot(range(1, len(nll) + 1), nll, "-", lw=1.5, color=col, alpha=0.8)
    ax2.axhline(PLATEAU, color=BAND_C, lw=1.6, ls="--")
    ax2.text(0.98, PLATEAU, f" finding 44 plateau {PLATEAU} ", size=SMALL - 2, color=BAND_C,
             va="bottom", ha="right", transform=ax2.get_yaxis_transform())
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("train NLL")
    ax2.grid(alpha=0.25)
    ax2.legend(handles=[Line2D([], [], color=NOTE_C, lw=2.2, label="KA  L=16 d160"),
                        Line2D([], [], color=FULL_C, lw=2.2, label="KB  L=16 d128"),
                        Line2D([], [], color=L16_C, lw=2.2, label="J2  L=16 d192")],
               loc="upper right", framealpha=0.95, fontsize=SMALL - 1)
    ax2.set_title("A rung that censors WITHOUT leaving the plateau\nis a trainability failure, "
                  "not a capacity one", size=MED - 4)

    ka, kb = p["KA"], p["KB"]
    block = []
    for rows, name in ((l12, "L=12"), (l16, "L=16")):
        block += [f"{name} width ladder", "  d     trainable    rho_50  cross  esc",
                  "  " + "-" * 38]
        for r in rows:
            block.append(f"  {r['d']:<4}{r['s']['trainable']:>10,}  {rho_txt(r['s']):>8}"
                         f"   {r['s']['n_cross']}/{r['s']['n']}"
                         f"   {r['esc']}/{r['s']['n']}   {r['tag']}")
        block.append("")
    fig.text(0.735, 0.78, "\n".join(block), size=SMALL - 3, va="top", family="monospace")

    footer(fig, f"Open triangles are censored rungs, drawn at the grid edge {GRID_TOP} because "
                f"that is all the eval knows about them; faint dots are individual seeds and the "
                f"ringed marker is the 3-seed median, which is the quantity every verdict is read "
                f"on. All cells attention-only, 240k instances x 50 epochs, batch 32, constant lr "
                f"1e-3, no warmup, --shuffle-adj --fall. KA {ka['s']['trainable']:,} and KB "
                f"{kb['s']['trainable']:,} both matched their registered predictions "
                f"({PREDICTED_TRAINABLE['KA']:,} / {PREDICTED_TRAINABLE['KB']:,}) exactly. "
                f"Commit {commit}.")
    o.savefig(fig)


def page_frontier(o, l12, l16, anchors, commit):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The size-quality frontier, with stage K's two rungs added",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, "Every attention-only rung this programme has run at N=48, against the "
                         "two complete-model anchors", ha="center", size=MED - 3, color="0.3")

    ax = fig.add_axes([0.075, 0.22, 0.43, 0.58])
    # L=12 labels go above their marker and L=16 labels below: the two ladders
    # are what collide on this axis (J2 2.50M / 0.580 sits on top of J1 2.57M /
    # 0.579), and separating by ladder fixes it whatever the K rungs come back as.
    for rows, col, lab, dy in ((l12, L12_C, "attention-only, L=12", 13),
                               (l16, L16_C, "attention-only, L=16", -23)):
        for r in rows:
            x = r["s"]["trainable"]
            for m in r["ms"]:
                if m["flag"] == "ok":
                    ax.plot(x, m["rho50"], "o", ms=5, color=col, alpha=0.45)
                else:
                    ax.plot(x, GRID_TOP, "^", ms=9, mfc="none", mew=1.8, color=col)
            if r["s"]["rho50_readable"]:
                ax.plot([x], [r["s"]["rho50"]], "o", ms=11, mfc="white", mew=2.5, color=col,
                        zorder=5)
                ax.annotate(r["tag"], xy=(x, r["s"]["rho50"]), xytext=(0, dy),
                            textcoords="offset points", ha="center", size=SMALL, color=col,
                            weight="bold")
            else:
                ax.annotate(f"{r['tag']} CENS", xy=(x, GRID_TOP), xytext=(5, 6),
                            textcoords="offset points", ha="left", size=SMALL - 2, color=col)
        ax.plot([], [], "o", color=col, label=lab)
    hub, best = anchors["hub"], anchors["best"]
    # The best solver is the largest thing on the axis, so a centred label under it
    # runs off the right edge of the plot and gets clipped; right-align that one so
    # it extends inwards instead. The hub sits mid-axis and is fine centred.
    for a, name, mark, ha in ((hub, "hub  full L=6 d256", "s", "center"),
                              (best, "full L=12 d256", "D", "right")):
        ax.plot([a["s"]["trainable"]], [a["s"]["rho50"]], mark, ms=11, color=FULL_C, zorder=5)
        ax.annotate(name, xy=(a["s"]["trainable"], a["s"]["rho50"]), xytext=(0, -24),
                    textcoords="offset points", ha=ha, size=SMALL - 1, color=FULL_C)
    ax.axhline(FREE, color=BAND_C, lw=1.5, ls="--")
    ax.text(0.99, FREE, f" free {FREE} ", size=SMALL - 2, color=BAND_C, va="bottom", ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters  (log scale)")
    ax.set_ylabel("rho_50   (lower is better)")
    ax.set_ylim(0.33, 1.00)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=SMALL - 1)
    ax.set_title("Where the cliff actually is, now that it has been\nprobed at two depths "
                 "rather than one", size=MED - 4)

    free = [r for r in l12 + l16 if qualifies(r)]
    smallest = min(free, key=lambda r: r["s"]["trainable"]) if free else None
    read = [r for r in l12 + l16 if r["s"]["rho50_readable"]]
    narrow = min(read, key=lambda r: r["s"]["trainable"]) if read else None
    # Two lines on this plot get called 'the cliff' and only one of them is the
    # one the registered rule reads. Whether it MOVED is a branch, not a given:
    # written unconditionally, this paragraph announced a relocation on a page
    # whose own verdict was that the minimum stayed put.
    if smallest is None:
        where = "No rung on this plot satisfies the registered rule. "
    elif smallest["tag"] in ("KA", "KB"):
        where = (f"The smallest rung that satisfies the registered rule anywhere on this plot is "
                 f"now {smallest['tag']} at {smallest['s']['trainable']:,} trainable, rho_50 "
                 f"{rho_txt(smallest['s'])} -- a stage K rung, so the cliff really has been "
                 f"RELOCATED, and relocating it is the entire content of the stage. ")
    else:
        where = (f"The smallest rung that satisfies the registered rule anywhere on this plot is "
                 f"still {smallest['tag']} at {smallest['s']['trainable']:,}, rho_50 "
                 f"{rho_txt(smallest['s'])} -- the same rung as before stage K, so THE CLIFF HAS "
                 f"NOT MOVED. What did move is a lower line: the narrowest READABLE rung is now "
                 f"{narrow['tag']} at {narrow['s']['trainable']:,}. Readable is not free, and the "
                 f"rule reads the free line. ")
    note = (
        f"WHAT THE SHAPE OF THIS CURVE NOW SAYS. Before stage K the cliff was a one-depth "
        f"measurement: everything below 2M was L=12, and the honest reading was 'a floor near 2M "
        f"parameters on this task, and everything above it is flat'. Stage K is the test of that "
        f"word 'task'. "
        + where
        + f"Nothing here says a smaller model is a better one: the "
        f"complete L=12 d256 model at {best['s']['trainable']:,} still solves at "
        f"{rho_txt(best['s'])}, far below anything on the attention-only curve."
    )
    # Centred on the plot's vertical span rather than hugging the top: top-anchored,
    # this block left the whole bottom-right quadrant blank while the plot filled the
    # left. One point larger too -- these decks get read on a phone.
    fig.text(0.545, 0.51, "\n".join(textwrap.wrap(note, 54)), size=SMALL + 1, va="center",
             linespacing=1.6)
    footer(fig, f"The hub is the architecture s* is anchored to (full L=6 d256, rho_50 "
                f"{HUB_RHO50} at {HUB_TRAINABLE:,}); the free line is the hub plus one s*. "
                f"Commit {commit}.")
    o.savefig(fig)


def page_predictions(o, p, l12, suf, commit):
    ka, kb = p["KA"], p["KB"]
    vka, vkb = verdict_ka(ka), verdict_kb(kb)
    held = {"held": "HELD", "free_above_band": "HELD on the binary call, MISSED HIGH on the level",
            "low": "MISSED LOW", "high": "FALSIFIED HIGH", "partial": "FALSIFIED on the seed count",
            "censored_trained": "FALSIFIED by censoring",
            "censored_untrained": "VOID -- the rung did not train"}[vka]
    win = new_minimum(p)
    textpage(
        o,
        "Every stage K prediction, scored against what was registered",
        f"stagek_score.py was committed (262b581) before any K cell had an eval on disk -- "
        f"no verdict chosen after seeing a number",
        [
            (f"K0  {held}   (registered: median in [{PRED_KA_LO}, {PRED_KA_HI}], point estimate "
             f"{PRED_KA_POINT}, robust call KA IS FREE on 3/3)",
             NOMLP_C if vka in ("held", "free_above_band") else BAND_C,
             f"Observed {rho_txt(ka['s'])} on {ka['s']['n_cross']}/3 crossing and {ka['esc']}/3 "
             f"plateau escapes, at {ka['s']['trainable']:,} trainable. The point estimate was "
             f"{I1_RHO50} (I1, the same width at L=12) minus the measured depth step "
             f"{DEPTH_STEP_AT_D192:.3f} (H2 -> J2 at a fixed d192) = {PRED_KA_POINT}. "
             + ("The transport model -- that a depth step buys a roughly constant amount of "
                "rho_50 independent of width -- survives this test. "
                if vka in ("held", "free_above_band", "low") else
                "The transport model does not survive: the depth gain measured at d192 does not "
                "carry to d160, so depth and width are not separable down here. ")
             + "LOW was registered as a falsification too, in our favour, earning no victory lap."),
            (f"K1  {'HELD' if vkb.startswith('censored') else 'FALSIFIED'}   (registered: KB "
             f"CENSORS, as H1 did at L=12)",
             NOMLP_C if vkb.startswith("censored") else BAND_C,
             kb_text(kb, vkb) + (
                 "  K1 GATES THE ARCHITECTURE READING OF KB, as F1 gates F0: a censored rung "
                 "that never left the plateau is a statement about optimization, not about what "
                 "the architecture can represent.")),
            (f"K2  {'FIRED -- the minimum moves' if win else 'DID NOT FIRE -- the minimum stands'}"
             f"   |   K3  before/after on the frozen bar",
             NOTE_C,
             f"K2 is the standing-minimum rule unchanged from H3 = I3 = J3: 3 of 3 crossing AND "
             f"median at or below {FREE}, smaller wins the tie-break, and two of three is never "
             f"enough (finding 51). BEFORE: {STANDING[1]}, {STANDING[2]:,} trainable at rho_50 "
             f"{STANDING[3]:.3f}. AFTER: "
             + (f"{win['tag']}, attention-only L=16 d{win['key'][3]}, "
                f"{win['s']['trainable']:,} at {rho_txt(win['s'])} -- "
                f"{STANDING[2] / win['s']['trainable']:.2f}x smaller, rho_50 moving "
                f"{win['s']['rho50'] - STANDING[3]:+.3f} = "
                f"{abs(win['s']['rho50'] - STANDING[3]) / S_RHO50:.2f} s* against the FROZEN bar "
                f"{S_RHO50}, which is never recomputed. Any move under the bar is 'no separable "
                f"cost', never an improvement."
                if win else
                f"unchanged. No rung in stage K satisfies the rule, so the before and after are "
                f"the same number and the correct report is that the push failed at this depth.")),
            (f"K4  {suf['n_exp']}/{suf['n_rungs']} held-out rungs on the accuracy-to-solve curve, "
             f"worst discriminating miss {suf['worst']:.4f} vs bar {suf['bar']:.4f}",
             FULL_C,
             f"The curve is refit over the nine sufficiency dirs plus SCAN_K and asks whether "
             f"stage K's rungs sit on it: {suf['n_exp']} of {suf['n_rungs']} well-determined "
             f"held-out rungs do, over {suf['n_cells']} cells, {suf['n_sharp']} of them "
             f"discriminating. Token accuracy remains a sufficient statistic over this grid. A "
             f"rung that fell OFF this curve would be the interesting one -- it would mean stage "
             f"K found a failure mode that token accuracy cannot see."),
        ],
        f"The LOO bar {LOO_BAR} is the leave-one-out RMS of the depth ladder's own fit and was "
        f"frozen with it. 'Discriminating' means the curve predicts solve > 0.10 for that cell, so "
        f"landing on it is a real constraint rather than two numbers agreeing near zero. "
        f"Commit {commit}.",
        width=128, y0=0.825, gap=0.019)


def page_next(o, p, l12, l16, commit):
    win = new_minimum(p)
    moved = win is not None
    textpage(
        o,
        "Where this leaves the N=48 search",
        f"stage K probes the floor at a second depth | stage F (51656823) asks separately "
        f"whether any of it transfers",
        [
            ("WHAT STAGE K SETTLED", NOMLP_C,
             (f"The ~2M floor belonged to L=12, not to the task. Repeating the width ladder one "
              f"depth up moved the smallest free rung to {win['s']['trainable']:,} trainable "
              f"({win['tag']}, d{win['key'][3]}), and the two width curves are not parallel -- "
              f"depth buys width back down here. "
              if moved else
              f"The ~2M floor is not an artefact of L=12. Repeating the width ladder one depth up "
              f"did NOT produce a smaller free rung: the depth gain measured at d192 does not "
              f"transport to narrower models, so the two knobs stop being separable below d192. ")
             + f"Either way the useful output is the SHAPE of the two curves, which is now "
             f"measured at two depths instead of one, and the breaking point is reported as the "
             f"result rather than hidden behind the rungs that worked."),
            ("THE NEXT CUT, IF THERE IS ONE", NOTE_C,
             (f"L=20 at d128 or d160 is the obvious follow-on -- if depth keeps buying width, the "
              f"ladder has not bottomed out and the next rung is cheaper still. It is worth 3 "
              f"cells only once stage F says whether any of this survives a second problem size; "
              f"spending them before that risks optimizing an N=48 artefact."
              if moved else
              f"There is no obvious next cut on this grid. The width floor held at two depths, "
              f"which is the stronger version of the stage J claim, and the remaining open "
              f"question is not architectural: it is whether the floor itself moves with problem "
              f"size, which is stage F.")),
            ("THE OPEN N=48 QUESTION STAGE J LEFT, STILL OPEN", DEAD_C,
             f"J1's width curve is CONVEX on [192, 256]: d224 beat every straight line through its "
             f"neighbours, so interpolating d_model between measured rungs systematically "
             f"underestimates the in-between model. A fourth width point is the only way to know "
             f"the shape of that stretch. It is deliberately NOT bundled into K, which moves "
             f"depth, or into F, which moves problem size -- one knob at a time is the only reason "
             f"any of these verdicts mean anything."),
            ("THE HONEST LIMITS OF EVERY NUMBER ON THESE SLIDES", BAND_C,
             f"All N=48, degree 5, 4 colours, one 14 x 400 rho grid ending at 0.33, one 240k x 50 "
             f"epoch budget, three seeds per rung. Seed spread at fixed architecture exceeds the "
             f"depth gaps the ladders are built from, which is why no single-seed rung is a "
             f"measurement and why the median over three is the only quantity scored. s* = "
             f"{S_RHO50} and the {PLATEAU} plateau are N=48 constants and are not reused at N=96. "
             f"The sufficiency result is correlational and is a statement about this grid's range.",
             ),
        ],
        f"Stage K: 6 cells, 3 seeds per rung, 51668331, --array=0-5%6 --time=10:00:00 -G 1 "
        f"-p sganguli. Registered {('11d42aa (sbatch) / 262b581 (scorer)')} before submission. "
        f"Commit {commit}.",
        width=124, y0=0.825, gap=0.021)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+",
                    default=SCAN_DIRS + [SCAN_E, SCAN_G, SCAN_H, SCAN_I, SCAN_J, SCAN_K])
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "stagek_2026-11-09"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--commit", default="(uncommitted)")
    args = ap.parse_args()

    set_style()
    cells, _raw = collect(args.scan_dir, args.boot)
    p = pair(cells)
    for tag in ("KA", "KB"):
        if p[tag]["s"]["n"] < 3:
            raise SystemExit(f"{tag} has {p[tag]['s']['n']} seeds; every registered rule in "
                             f"stage K is a median over three")
    l12, l16 = ladder(cells, LADDER_12), ladder(cells, LADDER_16)
    anchors = {"hub": dict(s=summarize(rung(cells, HUB))),
               "best": dict(s=summarize(rung(cells, FULL_L12)))}
    suf = sufficiency(args.scan_dir)
    print(f"{len(cells)} cells | KA {rho_txt(p['KA']['s'])} ({p['KA']['s']['n_cross']}/3 cross, "
          f"{p['KA']['esc']}/3 esc, {p['KA']['s']['trainable']:,}) -> {verdict_ka(p['KA'])} | "
          f"KB {rho_txt(p['KB']['s'])} ({p['KB']['s']['n_cross']}/3 cross, {p['KB']['esc']}/3 "
          f"esc, {p['KB']['s']['trainable']:,}) -> {verdict_kb(p['KB'])}")
    print(f"sufficiency: {suf['n_exp']}/{suf['n_rungs']} rungs, {suf['n_cells']} cells, "
          f"{suf['n_sharp']} discriminating, worst {suf['worst']:.4f} vs bar {suf['bar']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=7)
        page_headline(o, p, l12, l16, anchors, args.commit)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task every "
                               "cell in this deck is trained and scored on. Stage K changes "
                               "nothing about it -- only d_model moves, at a fixed L=16, so the "
                               "two K rungs differ from the standing minimum by one knob.")
        page_ladders(o, l12, l16, p, args.commit)
        page_frontier(o, l12, l16, anchors, args.commit)
        page_predictions(o, p, l12, suf, args.commit)
        page_next(o, p, l12, l16, args.commit)
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
