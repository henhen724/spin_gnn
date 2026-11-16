"""Figures for the `spin-gnn/loop` deck -- findings 87 and 88.

Every number is READ OUT OF THE JOBS' OWN JSON and none is retyped;
`check_claims()` at the bottom asserts the values the deck's prose states
against what was read, so a figure and its caption cannot drift apart. Same
pattern as make_predicate_web.py, which this follows page for page.

Sources, all COMPLETED 0:0:
  52476255_[0-2]  stageloopProbe  c LEARNED from zero   (finding 87)
  52478871_[0-2]  stageloopFixed  c FIXED and frozen    (finding 88, if landed)

The forced-arm pages are SKIPPED rather than faked when 52478871 has not landed;
`check_claims` reports which pages exist.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors

sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")
import stagepair_score as SP

ROOT = "/scratch/users/hshunt/spin_gnn"
OUT = "/home/users/hshunt/slides/decks/spin-gnn/loop/figures"
LEARNED = os.path.join(ROOT, "runs/stageloop_probe_2026-11-15")

os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def read_seeds(suffix):
    """{seed: summary} for the three array tasks, skipping any that is absent.

    Seed 0 writes the unsuffixed directory; seeds 1 and 2 add `_s{N}`. A partial
    array still produces the pages it can support.
    """
    out = {}
    for s in range(3):                       # the array is 0-2, by construction
        d = LEARNED + ("" if s == 0 else f"_s{s}") + suffix
        f = os.path.join(d, "loop_summary.json")
        if os.path.isfile(f):
            out[s] = json.load(open(f))
    return out


LRN = read_seeds("")
FIX = read_seeds("_fixed")
FST = read_seeds("_fixed_first")          # finding 90, the corrected calibration
print("learned-c seeds on disk:", sorted(LRN))
print("forced-c  seeds on disk:", sorted(FIX))
print("first-cal seeds on disk:", sorted(FST))
assert LRN, "finding 87's artifacts are missing -- nothing to draw"

SEEDS = sorted(LRN)
# the labels the forced run uses, read off the artifact rather than assumed
FIX_ARMS = ([a for a in FIX[sorted(FIX)[0]]["per_arm"] if a.startswith("fix")]
            if FIX else [])
print("forced arms:", FIX_ARMS)


def col(src, arm, key):
    return [src[s]["per_arm"][arm][key] for s in sorted(src)]


def label_ends(ax, x, pairs, min_pts=16):
    """Label several series at a shared right-hand x, staggered when they tie.

    `pairs` is [(value, text, colour)]. Three seeds routinely land within 0.004
    of each other here, which overprints their labels -- the first read-back
    pass caught it on the AUROC, guard and churn pages -- so anything closer
    than `min_pts` typographic points to the label above it is pushed down by
    exactly that much. The leader is still the data point, not the text.
    """
    h_pts = ax.get_position().height * ax.figure.get_figheight() * 72.0
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    order = sorted(pairs, key=lambda t: -t[0])
    placed = []
    for v, _txt, _c in order:
        y = (v - lo) / span * h_pts
        if placed and placed[-1] - y < min_pts:
            y = placed[-1] - min_pts
        placed.append(y)
    for (v, txt, c), y in zip(order, placed):
        label_line(ax, x, v, txt, color=c,
                   dy=y - (v - lo) / span * h_pts)


def hist(src, s, arm, key):
    """History plus the FINAL value, which lives in `per_arm`, not `history`.

    The snapshots stop at step 2,500; the number every table and this deck
    quotes is the one at 3,000. Plotting the history alone would end a curve at
    a value the prose never states -- on seed 2 that is 1.3e-05 against a
    reported 0.0002, a factor of fifteen.
    """
    h = src[s]["history"][arm]
    x = [r["step"] for r in h] + [src[s]["train"]["steps"]]
    y = [r[key] for r in h] + [src[s]["per_arm"][arm][key]]
    return x, y


# ---------------------------------------------------------------------------
# 1. the scalar's trajectory: the whole finding of 87 on one page
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.3))
lo, hi = LRN[SEEDS[0]]["prediction"]["loop_scale_abs"]
ax.axhspan(lo, hi, color=MUTED, alpha=0.16, zorder=0)
ax.text(60, lo * 1.25, "registered prediction: $|c|$ ends in this band",
        color=MUTED, fontsize=12, va="bottom")
cols = series_colors(len(SEEDS))
# seeds 0 and 1 end at 0.00065 and 0.00064, so their right-hand labels collide
# unless they are staggered; the first read-back pass caught exactly that.
ends = []
for k, s in enumerate(SEEDS):
    x, y = hist(LRN, s, "loop L1", "loop_scale")
    y = [max(abs(v), 1e-5) for v in y]
    ax.plot(x, y, color=cols[k], lw=2.0, rasterized=True)
    ends.append((y[-1], x[-1], s, cols[k]))
for rank, (yv, xv, s, c) in enumerate(sorted(ends, reverse=True)):
    close = any(abs(np.log10(yv / o[0])) < 0.12 for j, o in enumerate(ends)
                if o[2] != s)
    label_line(ax, xv, yv, f"seed {s}", color=c,
               dy=(9 if rank == 0 else -9) if close else 0)
ax.set_yscale("log")
ax.set_xlim(-60, 3000 * 1.14)
style_axes(ax, xlabel="training step", ylabel="$|c|$, the feedback scalar")
save(fig, "c_trajectory.png")

# ---------------------------------------------------------------------------
# 2. what that means in units of the residual stream it feeds
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for k, s in enumerate(SEEDS):
    x, y = hist(LRN, s, "loop L1", "fb_rel")
    y = [100 * v for v in y]
    ax.plot(x, y, color=cols[k], lw=2.0, rasterized=True)
    label_line(ax, x[-1], y[-1], f"seed {s}", color=cols[k])
ax.set_xlim(-60, 3000 * 1.14)
ax.set_ylim(bottom=0)
style_axes(ax, xlabel="training step",
           ylabel=r"feedback as % of $\overline{|h|}$")
save(fig, "fb_rel_trajectory.png")

# ---------------------------------------------------------------------------
# 3. the predicate: corrupt against placebo, per arm, per seed
# ---------------------------------------------------------------------------
ARMS4 = ["Y", "pair L1", "loop L0", "loop L1"]
PRETTY = {"Y": "stage Y", "pair L1": r"pair $\lambda{=}1$",
          "loop L0": r"loop $\lambda{=}0$", "loop L1": r"loop $\lambda{=}1$"}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
for ax, key, ttl in zip(axes, ["pair_auroc_corrupt", "pair_auroc_placebo"],
                        ["corrupt (a neighbour of $v$)",
                         "placebo (not a neighbour)"]):
    for k, s in enumerate(SEEDS):
        ax.plot(range(len(ARMS4)), [LRN[s]["per_arm"][a][key] for a in ARMS4],
                "o-", color=cols[k], lw=1.6, ms=7, rasterized=True)
    ax.set_xticks(range(len(ARMS4)))
    ax.set_xticklabels([PRETTY[a] for a in ARMS4], fontsize=11)
    ax.set_title(ttl, color=MUTED, fontsize=13)
    style_axes(ax)
axes[0].axhline(SP.PAIR_BAR, color=MUTED, ls="--", lw=1.2)
# The bar's label does NOT sit on its own line. Three placements were tried and
# all three were struck: the arms make a V whose legs cross 0.65 at x ~ 0.3, 1.6
# and 2.4, so both the band just above the line and the band just below it are
# crossed somewhere along a label's width. Raising the ceiling buys an empty
# strip at the top of the panel instead, and the word "dashed" carries the link.
axes[0].set_ylim(top=0.99)                      # shared: the right panel too
axes[0].text(-0.08, 0.965, "dashed: P1 bar 0.65", color=MUTED, fontsize=11,
             va="top")
axes[1].axhspan(*SP.PLACEBO_BAND, color=MUTED, alpha=0.16, zorder=0)
axes[1].text(0.04, SP.PLACEBO_BAND[1] + 0.018, "P2 band", color=MUTED,
             fontsize=11)
# the seeds are labelled once, past the RIGHT panel's right edge so nothing
# spills onto the other panel, and staggered: all three land within 0.004 of
# each other, so unstaggered labels overprint. The colours carry across panels.
fin = sorted(((LRN[s]["per_arm"][ARMS4[-1]]["pair_auroc_placebo"], s, cols[k])
              for k, s in enumerate(SEEDS)), reverse=True)
for rank, (yv, s, c) in enumerate(fin):
    label_line(axes[1], len(ARMS4) - 1, yv, f"seed {s}", color=c,
               dy=(1 - rank) * 14)
for ax in axes:
    ax.set_xlim(-0.3, len(ARMS4) - 0.7)
axes[0].set_ylabel("matched-pair AUROC", color=MUTED, fontsize=13)
axes[0].set_ylim(0.35, 0.95)
save(fig, "predicate_arms.png")

# ---------------------------------------------------------------------------
# 4. the load-bearing column: guard rate, loop L1 against its own pair L1
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.3))
for k, s in enumerate(SEEDS):
    y = [LRN[s]["per_arm"]["pair L1"]["guard_rate"],
         LRN[s]["per_arm"]["loop L1"]["guard_rate"]]
    ax.plot([0, 1], y, "o-", color=cols[k], lw=2.0, ms=8, rasterized=True)
    label_line(ax, 1, y[1], f"seed {s}", color=cols[k])
ax.axhline(SP.GUARD_BAR, color=MUTED, ls="--", lw=1.2)
ax.set_ylim(bottom=SP.GUARD_BAR - 0.018)      # room for the bar's own label
ax.text(-0.06, SP.GUARD_BAR - 0.003, "P4 bar 0.10", color=MUTED, fontsize=11,
        va="top")
ax.set_xticks([0, 1])
ax.set_xticklabels([r"pair $\lambda{=}1$ (no loop)", r"loop $\lambda{=}1$"],
                   fontsize=12)
ax.set_xlim(-0.12, 1.3)
style_axes(ax, ylabel="guard rate, conditional on clean emits $v$")
save(fig, "guard_paired.png")

# ---------------------------------------------------------------------------
# 5. the churn column, which only means anything next to the guard
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for k, s_ in enumerate(SEEDS):
    ax.plot(range(len(ARMS4)),
            [LRN[s_]["per_arm"][a]["total_flips"] for a in ARMS4],
            "o-", color=cols[k], lw=1.8, ms=8, rasterized=True)
ax.set_xticks(range(len(ARMS4)))
ax.set_xticklabels([PRETTY[a] for a in ARMS4], fontsize=11)
ax.set_xlim(-0.2, len(ARMS4) - 0.62)
ax.set_ylim(-8, 196)
style_axes(ax, ylabel="cases of 1,000 where one edit flips $v$")
label_ends(ax, len(ARMS4) - 1,
           [(LRN[s_]["per_arm"][ARMS4[-1]]["total_flips"], f"seed {s_}",
             cols[k]) for k, s_ in enumerate(SEEDS)])
save(fig, "flips_arms.png")

# ---------------------------------------------------------------------------
# 6. validation token accuracy, which is read before any AUROC
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.3))
for k, s_ in enumerate(SEEDS):
    ax.plot(range(len(ARMS4)),
            [LRN[s_]["per_arm"][a]["val_accuracy"] for a in ARMS4],
            "o-", color=cols[k], lw=1.6, ms=7, rasterized=True)
ax.set_xticks(range(len(ARMS4)))
ax.set_xticklabels([PRETTY[a] for a in ARMS4], fontsize=11)
ax.set_xlim(-0.2, len(ARMS4) - 0.55)
style_axes(ax, ylabel="held-out token accuracy")
label_ends(ax, len(ARMS4) - 1,
           [(LRN[s_]["per_arm"][ARMS4[-1]]["val_accuracy"], f"seed {s_}",
             cols[k]) for k, s_ in enumerate(SEEDS)])
save(fig, "val_accuracy.png")

# ---------------------------------------------------------------------------
# 7. the mechanism page: how far Adam COULD have moved the scalar
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.0))
steps = LRN[SEEDS[0]]["train"]["steps"]
lr = LRN[SEEDS[0]]["train"]["lr"]
reach = steps * lr
bars = [("Adam's reach\n%d steps x lr %g" % (steps, lr), reach),
        ("the predicted\nband's floor", lo)] + \
       [("seed %d's\nfinal $|c|$" % s, abs(LRN[s]["per_arm"]["loop L1"]
                                           ["loop_scale"])) for s in SEEDS]
cl = [MUTED, MUTED] + list(cols)
ax.barh(range(len(bars)), [b[1] for b in bars], color=cl, height=0.62,
        rasterized=True)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars], fontsize=11)
ax.invert_yaxis()
ax.set_xscale("log")
for i, (_n, v) in enumerate(bars):
    ax.text(v * 1.3, i, f"{v:.2g}", va="center", color=MUTED, fontsize=11)
ax.set_xlim(right=reach * 9)        # headroom, or the widest bar's label clips
style_axes(ax, xlabel="$|c|$ (log scale)")
save(fig, "adam_reach.png")

# ---------------------------------------------------------------------------
# the forced arm, only if 52478871 has landed
# ---------------------------------------------------------------------------
if FIX_ARMS:
    FSEEDS = sorted(FIX)
    fcols = series_colors(len(FSEEDS))
    FARMS = ["Y", "pair L1"] + FIX_ARMS
    FP = dict(PRETTY)
    for a in FIX_ARMS:
        frac = float(a.split()[0][3:])
        # two lines: side by side on one line these three run into each other
        FP[a] = "$c$ at %d%% of $h$\n" % round(100 * frac) + (
            r"$\lambda{=}%s$" % a.split()[1][1:])

    l0 = [a for a in FIX_ARMS if a.endswith("L0")][0]
    l1 = [a for a in FIX_ARMS if a.endswith("L1")][0]        # the 0.10 one
    l1s = [a for a in FIX_ARMS if a.endswith("L1")][-1]      # the 0.30 one
    BREAK_VA = FIX[FSEEDS[0]]["prediction"]["break_val_acc_below"]
    BREAK_NLL = FIX[FSEEDS[0]]["prediction"]["break_nll_above"]

    # 8. THE HEADLINE: a forced loop does not cost accuracy, it removes it
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for k, s_ in enumerate(FSEEDS):
        y = [FIX[s_]["per_arm"][a]["val_accuracy"] for a in FARMS]
        ax.plot(range(len(FARMS)), y, "o-", color=fcols[k], lw=1.6, ms=7,
                rasterized=True)
    ax.axhline(BREAK_VA, color=MUTED, ls="--", lw=1.2)
    ax.text(len(FARMS) - 1.0, BREAK_VA + 0.012,
            f"registered falsifier: below {BREAK_VA:g} the row is damage",
            color=MUTED, fontsize=10, ha="right", va="bottom")
    ax.set_xticks(range(len(FARMS)))
    ax.set_xticklabels([FP[a] for a in FARMS], fontsize=10)
    ax.set_xlim(-0.2, len(FARMS) - 0.55)
    ax.set_ylim(0.0, 0.56)
    style_axes(ax, ylabel="held-out token accuracy")
    label_ends(ax, len(FARMS) - 1,
               [(FIX[s_]["per_arm"][FARMS[-1]]["val_accuracy"], f"seed {s_}",
                 fcols[k]) for k, s_ in enumerate(FSEEDS)])
    save(fig, "fixed_val_accuracy.png")

    # 9. WHY: the very first forward pass is worse than guessing uniformly, and
    # 3,000 steps only claw it back to roughly uniform. Three seeds of the mild
    # arm; the two reference lines are the body's own nll and log 58.
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for k, s_ in enumerate(FSEEDS):
        # `history` stops at 2,500 and the final nll lives in `per_arm` under a
        # different key, so this one cannot go through `hist`
        rows = FIX[s_]["history"][l1]
        x = [r["step"] for r in rows] + [FIX[s_]["train"]["steps"]]
        y = ([r["train_nll"] for r in rows]
             + [FIX[s_]["per_arm"][l1]["final_train_nll"]])
        ax.plot(x, y, "o-", color=fcols[k], lw=1.8, ms=5, rasterized=True)
    ax.axhline(float(np.log(58)), color=MUTED, ls="--", lw=1.2)
    ax.text(1500, float(np.log(58)) + 0.15,
            r"$\log 58 = 4.06$: uniform over the vocabulary", color=MUTED,
            fontsize=10, ha="center", va="bottom")
    ybody = FIX[FSEEDS[0]]["per_arm"]["Y"]["final_train_nll"]
    ax.axhline(ybody, color=MUTED, ls=":", lw=1.2)
    ax.text(1500, ybody + 0.15, "the stage Y body it started from: 1.60",
            color=MUTED, fontsize=10, ha="center", va="bottom")
    ax.set_ylim(0.0, 10.2)
    style_axes(ax, xlabel="training step", ylabel="training nll (nats)")
    label_ends(ax, 3000, [(FIX[s_]["per_arm"][l1]["final_train_nll"],
                           f"seed {s_}", fcols[k])
                          for k, s_ in enumerate(FSEEDS)])
    save(fig, "fixed_nll_trajectory.png")

    # 10. THE DENOMINATOR TRAP, drawn: the guard's denominator is the number of
    # cases the clean prompt emits v at all, and the forced arms stopped doing it
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for k, s_ in enumerate(FSEEDS):
        y = [FIX[s_]["per_arm"][a]["guard_denom"] for a in FARMS]
        ax.plot(range(len(FARMS)), y, "o-", color=fcols[k], lw=1.6, ms=7,
                rasterized=True)
    # no per-point numbers: at the forced end all three seeds sit within a few
    # cases of zero and three labels there overprint each other. The collapse is
    # the figure's claim and is visible without them; the page's body states the
    # counts.
    ax.set_xticks(range(len(FARMS)))
    ax.set_xticklabels([FP[a] for a in FARMS], fontsize=10)
    ax.set_xlim(-0.2, len(FARMS) - 0.55)
    ax.set_ylim(bottom=-60)
    style_axes(ax, ylabel="cases of 1,000 where the clean prompt emits $v$")
    label_ends(ax, len(FARMS) - 1,
               [(FIX[s_]["per_arm"][FARMS[-1]]["guard_denom"], f"seed {s_}",
                 fcols[k]) for k, s_ in enumerate(FSEEDS)])
    save(fig, "fixed_denominator.png")

    # 11. the calibration itself missed: c was solved for from ONE application's
    # linearity, but the loop composes it 15 times, so the ratio at step 0 is not
    # the target. Measured-at-step-0 against the two targets, log y.
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    targets = FIX[FSEEDS[0]]["prediction"]["fix_targets"]
    for k, s_ in enumerate(FSEEDS):
        y = [100 * FIX[s_]["history"][a][0]["fb_rel"] for a in (l1, l1s)]
        ax.plot([0, 1], y, "o-", color=fcols[k], lw=2.0, ms=8, rasterized=True)
        # left of the x=0 points, above the x=1 ones: over the left column the
        # dashed 10% target runs straight through where a centred label goes
        ax.annotate(f"{y[0]:.0f}%", xy=(0, y[0]), xytext=(-9, 0),
                    textcoords="offset points", color=fcols[k], fontsize=10,
                    ha="right", va="center")
        ax.annotate(f"{y[1]:.0f}%", xy=(1, y[1]), xytext=(0, 9),
                    textcoords="offset points", color=fcols[k], fontsize=10,
                    ha="center")
    for xi, t in enumerate(targets):
        ax.plot([xi - 0.17, xi + 0.17], [100 * t] * 2, color=MUTED, ls="--",
                lw=1.4)
    # top left: the only corner no curve enters, and seed 2's climb struck the
    # middle of the panel where this sentence first sat
    ax.text(-0.28, 115, "dashed: what was asked for", color=MUTED, fontsize=10,
            ha="left", va="top")
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"asked for {100 * t:g}% of $|h|$" for t in targets],
                       fontsize=11)
    ax.set_xlim(-0.3, 1.45)
    style_axes(ax, ylabel="feedback as % of $\\overline{|h|}$, step 0")
    label_ends(ax, 1, [(100 * FIX[s_]["history"][l1s][0]["fb_rel"],
                        f"seed {s_}", fcols[k])
                       for k, s_ in enumerate(FSEEDS)])
    save(fig, "fixed_calibration.png")


# ---------------------------------------------------------------------------
# finding 89: the geometry of the injection, and what the calibration actually
# set. CPU job 52484098, one summary for all three bodies.
# ---------------------------------------------------------------------------
DIRF = os.path.join(ROOT, "runs/stageloop_direction_2026-11-15",
                    "direction_summary.json")
DIR = json.load(open(DIRF)) if os.path.isfile(DIRF) else None
if DIR:
    DSEEDS = sorted(int(k) for k in DIR["per_seed"])
    dcols = series_colors(len(DSEEDS))

    def dseed(s):
        return DIR["per_seed"][str(s)]

    def dmode(s, mode):
        return dseed(s)["modes"][mode]

    # 12. THE HEADLINE OF 89: what "10% of h" actually set. The first injection
    # at the c finding 88 solved for, as a multiple of the stream it is added to.
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    first = [100 * dseed(s)["c_solved"] * dmode(s, "closed")[0]["r1"]
             for s in DSEEDS]
    bars = [("asked for", 10.0)] + [(f"seed {s} actually got", v)
                                    for s, v in zip(DSEEDS, first)]
    ax.barh(range(len(bars)), [b[1] for b in bars], height=0.62,
            color=[MUTED] + list(dcols), rasterized=True)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=11)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for i, (_n, v) in enumerate(bars):
        ax.text(v * 1.25, i, f"{v:,.0f}%", va="center", color=MUTED,
                fontsize=11)
    ax.set_xlim(right=max(first) * 6)
    style_axes(ax, xlabel="first injection as % of $\\overline{|h|}$ "
                          "(log scale)")
    save(fig, "direction_first_injection.png")

    # 13. WHY THE CALIBRATION MISSED: on the body's own trajectory the ratio is
    # large at every application and RISES; the number 88 divided by was read at
    # the last one with the loop already open, which is a different quantity.
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    implied = [dseed(s)["r1_implied_by_calibration"] for s in DSEEDS]
    ax.axhspan(min(implied), max(implied), color=MUTED, alpha=0.18, zorder=0)
    for k, s in enumerate(DSEEDS):
        rows = dmode(s, "closed")
        ax.plot([r["t"] for r in rows], [r["r1"] for r in rows], "o-",
                color=dcols[k], lw=1.8, ms=5, rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 30)
    # the band sits at 0.04-0.08 and the curves run across the top, so the only
    # clear strip is just above the band
    ax.text(0.2, 0.115, "shaded: the ratio finding 88's calibration divided by",
            color=MUTED, fontsize=10, va="bottom")
    style_axes(ax, xlabel="application $t$ of the shared block",
               ylabel="injection / $\\overline{|h|}$ at $c=1$ (log scale)")
    label_ends(ax, 14, [(dmode(s, "closed")[-1]["r1"], f"seed {s}", dcols[k])
                        for k, s in enumerate(DSEEDS)])
    save(fig, "direction_ratio_vs_t.png")

    # 14. the registered question: is the injection common-mode? One panel per
    # body, three series each -- the injection and its two controls.
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9), sharey=True)
    # three DISTINCT colours, not highlight+mute: the two controls run within
    # 0.05 of each other and cross, so two greys made them one curve with two
    # labels -- caught in the first read-back of this page
    cc = list(SERIES[:3])
    for k, s in enumerate(DSEEDS):
        a = axes[k]
        rows = dmode(s, "closed")
        t = [r["t"] for r in rows]
        for j, (key, name) in enumerate((("cos_delta", "the injection"),
                                         ("cos_dh", "the block's own update"),
                                         ("cos_h", "the stream $h$"))):
            a.plot(t, [r[key] for r in rows], "-", color=cc[j], lw=2.0,
                   rasterized=True)
        a.set_ylim(-0.05, 1.0)
        a.set_title(f"seed {s}", fontsize=11, color=MUTED)
        style_axes(a, xlabel="application $t$",
                   ylabel="mean cosine between positions" if k == 0 else None)
    # labelled once, on the right-hand panel, at each series' own right end
    label_ends(axes[2], 14,
               [(dmode(DSEEDS[-1], "closed")[-1][key], name, cc[j])
                for j, (key, name) in enumerate(
                    (("cos_delta", "the injection"),
                     ("cos_dh", "the block's update"),
                     ("cos_h", "the stream $h$")))])
    save(fig, "direction_cosine.png")


# ---------------------------------------------------------------------------
# finding 90: the SAME forced-loop experiment with r1 read at the FIRST
# application, i.e. the mild loop 88 meant to run. Job 52484423, arms
# `fix0.10 L0` and `fix0.10 L1` in `..._fixed_first`.
# ---------------------------------------------------------------------------
if FST:
    TSEEDS = sorted(FST)
    tcols = series_colors(len(TSEEDS))
    TARMS = ["Y", "pair L1", "fix0.10 L0", "fix0.10 L1"]
    TP = {"Y": "stage Y", "pair L1": r"pair $\lambda{=}1$",
          "fix0.10 L0": "$c$ at 10%\n" + r"$\lambda{=}0$",
          "fix0.10 L1": "$c$ at 10%\n" + r"$\lambda{=}1$"}
    T_L0, T_L1 = "fix0.10 L0", "fix0.10 L1"

    # 15. THE HEADLINE OF 90: at a correct calibration the body is untouched.
    # 88's forced rows are drawn as a shaded reference band rather than three
    # more curves -- six series on one panel is past this deck's limit, and the
    # contrast is a range, not a trajectory.
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    if FIX_ARMS:
        f88 = [FIX[s_]["per_arm"][l1]["val_accuracy"] for s_ in sorted(FIX)]
        ax.axhspan(min(f88), max(f88), color=MUTED, alpha=0.16, zorder=0)
        ax.text(-0.14, max(f88) + 0.012,
                "shaded: finding 88's same arm, broken calibration",
                color=MUTED, fontsize=10, va="bottom")
    # seeds 0 and 1 agree to 0.001 on every arm here, so an unjittered seed 0
    # is drawn and then painted over -- two visible curves under three labels,
    # the same failure the cosine page was caught on. A 0.05 offset on an axis
    # whose ticks are 1.0 apart separates them without moving a point off its
    # arm.
    for k, s_ in enumerate(TSEEDS):
        xs = [i + (k - 1) * 0.05 for i in range(len(TARMS))]
        ax.plot(xs, [FST[s_]["per_arm"][a]["val_accuracy"] for a in TARMS],
                "o-", color=tcols[k], lw=1.6, ms=7, rasterized=True)
    ax.set_xticks(range(len(TARMS)))
    ax.set_xticklabels([TP[a] for a in TARMS], fontsize=10)
    ax.set_xlim(-0.2, len(TARMS) - 0.55)
    ax.set_ylim(0.0, 0.56)
    style_axes(ax, ylabel="held-out token accuracy")
    label_ends(ax, len(TARMS) - 1,
               [(FST[s_]["per_arm"][TARMS[-1]]["val_accuracy"], f"seed {s_}",
                 tcols[k]) for k, s_ in enumerate(TSEEDS)])
    save(fig, "first_val_accuracy.png")

    # 16. the first forward pass is still damaged -- and 500 steps repair it.
    # Same axes as `fixed_nll_trajectory.png` deliberately, so the two pages can
    # be compared by eye; this one ends on the body's own line instead of above
    # uniform.
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for k, s_ in enumerate(TSEEDS):
        rows = FST[s_]["history"][T_L1]
        x = [r["step"] for r in rows] + [FST[s_]["train"]["steps"]]
        y = ([r["train_nll"] for r in rows]
             + [FST[s_]["per_arm"][T_L1]["final_train_nll"]])
        ax.plot(x, y, "o-", color=tcols[k], lw=1.8, ms=5, rasterized=True)
    ax.axhline(float(np.log(58)), color=MUTED, ls="--", lw=1.2)
    ax.text(1500, float(np.log(58)) + 0.15,
            r"$\log 58 = 4.06$: uniform over the vocabulary", color=MUTED,
            fontsize=10, ha="center", va="bottom")
    ybody = FST[TSEEDS[0]]["per_arm"]["Y"]["final_train_nll"]
    ax.axhline(ybody, color=MUTED, ls=":", lw=1.2)
    ax.text(1500, ybody - 0.2, "the stage Y body it started from: 1.60",
            color=MUTED, fontsize=10, ha="center", va="top")
    ax.set_ylim(0.0, 10.2)                       # identical to finding 88's
    style_axes(ax, xlabel="training step", ylabel="training nll (nats)")
    label_ends(ax, 3000, [(FST[s_]["per_arm"][T_L1]["final_train_nll"],
                           f"seed {s_}", tcols[k])
                          for k, s_ in enumerate(TSEEDS)])
    save(fig, "first_nll_trajectory.png")

    # 17. the frozen scalar now holds its share, which 88's never did
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    band = FST[TSEEDS[0]]["prediction"]["fb_rel_last_band"]
    ax.axhspan(100 * band[0], 100 * band[1], color=MUTED, alpha=0.16, zorder=0)
    ax.text(60, 100 * band[1] + 0.6, "registered prediction: the ratio ends in "
            "this band", color=MUTED, fontsize=10, va="bottom")
    for k, s_ in enumerate(TSEEDS):
        rows = FST[s_]["history"][T_L1]
        x = [r["step"] for r in rows] + [FST[s_]["train"]["steps"]]
        y = [100 * r["fb_rel"] for r in rows] + [
            100 * FST[s_]["per_arm"][T_L1]["fb_rel"]]
        ax.plot(x, y, "o-", color=tcols[k], lw=1.8, ms=5, rasterized=True)
    ax.set_xlim(-60, 3000 * 1.14)
    ax.set_ylim(0, 30)
    style_axes(ax, xlabel="training step",
               ylabel=r"last application as % of $\overline{|h|}$")
    label_ends(ax, 3000, [(100 * FST[s_]["per_arm"][T_L1]["fb_rel"],
                           f"seed {s_}", tcols[k])
                          for k, s_ in enumerate(TSEEDS)])
    save(fig, "first_fb_rel.png")

    # 18. the predicate, same two panels as finding 87's page
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, key, ttl in zip(axes, ["pair_auroc_corrupt", "pair_auroc_placebo"],
                            ["corrupt (a neighbour of $v$)",
                             "placebo (not a neighbour)"]):
        for k, s_ in enumerate(TSEEDS):
            ax.plot(range(len(TARMS)),
                    [FST[s_]["per_arm"][a][key] for a in TARMS],
                    "o-", color=tcols[k], lw=1.6, ms=7, rasterized=True)
        ax.set_xticks(range(len(TARMS)))
        ax.set_xticklabels([TP[a] for a in TARMS], fontsize=10)
        ax.set_title(ttl, color=MUTED, fontsize=13)
        ax.set_xlim(-0.3, len(TARMS) - 0.7)
        style_axes(ax)
    axes[0].axhline(SP.PAIR_BAR, color=MUTED, ls="--", lw=1.2)
    axes[0].set_ylim(0.35, 0.99)
    axes[0].text(-0.2, 0.965, "dashed: P1 bar 0.65", color=MUTED, fontsize=11,
                 va="top")
    axes[1].axhspan(*SP.PLACEBO_BAND, color=MUTED, alpha=0.16, zorder=0)
    axes[1].text(0.04, SP.PLACEBO_BAND[1] + 0.018, "P2 band", color=MUTED,
                 fontsize=11)
    fin = sorted(((FST[s_]["per_arm"][TARMS[-1]]["pair_auroc_placebo"], s_,
                   tcols[k]) for k, s_ in enumerate(TSEEDS)), reverse=True)
    for rank, (yv, s_, c) in enumerate(fin):
        label_line(axes[1], len(TARMS) - 1, yv, f"seed {s_}", color=c,
                   dy=(1 - rank) * 14)
    axes[0].set_ylabel("matched-pair AUROC", color=MUTED, fontsize=13)
    save(fig, "first_predicate.png")

    # 19. THE LOAD-BEARING PAGE, and the only way to read it honestly: the same
    # margin arm trained three times -- no loop, a loop the optimizer shut, and
    # a loop forced open at a corrected 10%. The spread between the first two is
    # the envelope the third has to beat, and it is not a bar anyone registered;
    # it is simply what two trainings of one arm do.
    X3 = [r"pair $\lambda{=}1$" + "\nno loop",
          r"loop $\lambda{=}1$" + "\n$c$ learned, $10^{-3}$",
          r"loop $\lambda{=}1$" + "\n$c$ forced, $0.02$"]
    for key, ylab, fname, ylim in (
            ("guard_rate", "guard rate, conditional on clean emits $v$",
             "first_guard_three_trainings.png", None),
            ("total_flips", "cases of 1,000 where one edit flips $v$",
             "first_flips_three_trainings.png", (0, 210))):
        fig, ax = plt.subplots(figsize=(7.8, 4.3))
        for k, s_ in enumerate(TSEEDS):
            y = [FST[s_]["per_arm"]["pair L1"][key],
                 LRN[s_]["per_arm"]["loop L1"][key],
                 FST[s_]["per_arm"][T_L1][key]]
            ax.plot(range(3), y, "o-", color=tcols[k], lw=2.0, ms=8,
                    rasterized=True)
        ax.set_xticks(range(3))
        ax.set_xticklabels(X3, fontsize=10)
        ax.set_xlim(-0.2, 2.45)
        if ylim:
            ax.set_ylim(*ylim)
        style_axes(ax, ylabel=ylab)
        label_ends(ax, 2, [(FST[s_]["per_arm"][T_L1][key], f"seed {s_}",
                            tcols[k]) for k, s_ in enumerate(TSEEDS)])
        save(fig, fname)


# ---------------------------------------------------------------------------
# check_claims: every number the deck's prose states, asserted against the JSON
# ---------------------------------------------------------------------------
def check_claims():
    ok = []

    def chk(name, got, want, tol):
        assert abs(got - want) <= tol, f"{name}: read {got!r}, deck says {want!r}"
        ok.append(name)

    assert sorted(LRN) == [0, 1, 2], sorted(LRN)
    for s, c_ in zip([0, 1, 2], [0.0007, 0.0006, 0.0002]):
        chk(f"|c| seed {s}", abs(LRN[s]["per_arm"]["loop L1"]["loop_scale"]),
            c_, 0.00005)
    for s, f_ in zip([0, 1, 2], [0.51, 0.65, 0.41]):
        chk(f"fb_rel seed {s}",
            100 * LRN[s]["per_arm"]["loop L1"]["fb_rel"], f_, 0.02)
    for s, a_ in zip([0, 1, 2], [0.8720, 0.8710, 0.8680]):
        chk(f"corrupt seed {s}",
            LRN[s]["per_arm"]["loop L1"]["pair_auroc_corrupt"], a_, 1e-6)
    for s, a_ in zip([0, 1, 2], [0.4380, 0.4460, 0.4470]):
        chk(f"placebo seed {s}",
            LRN[s]["per_arm"]["loop L1"]["pair_auroc_placebo"], a_, 1e-6)
    for s, g_ in zip([0, 1, 2], [0.1072, 0.2211, 0.1416]):
        chk(f"loop guard seed {s}",
            LRN[s]["per_arm"]["loop L1"]["guard_rate"], g_, 5e-5)
    for s, g_ in zip([0, 1, 2], [0.1087, 0.2072, 0.1632]):
        chk(f"pair guard seed {s}",
            LRN[s]["per_arm"]["pair L1"]["guard_rate"], g_, 5e-5)
    for s, n_ in zip([0, 1, 2], [139, 174, 140]):
        chk(f"loop flips seed {s}",
            LRN[s]["per_arm"]["loop L1"]["total_flips"], n_, 0)
    for s, n_ in zip([0, 1, 2], [4, 1, 7]):
        chk(f"Y flips seed {s}", LRN[s]["per_arm"]["Y"]["total_flips"], n_, 0)
    # the deck states the arm is one scalar away from the registered count
    chk("Adam reach", steps * lr, 3.0, 1e-9)
    # P1 and P2 as the deck reads them, from the registered bars
    for s in [0, 1, 2]:
        p = LRN[s]["per_arm"]["loop L1"]
        assert p["pair_auroc_corrupt"] >= SP.PAIR_BAR, s
        assert SP.PLACEBO_BAND[0] < p["pair_auroc_placebo"] < SP.PLACEBO_BAND[1], s
    ok.append("P1 and P2 on 3/3 from the imported bars")
    # val accuracy within the registered tolerance of this job's own Y row
    for s in [0, 1, 2]:
        d = abs(LRN[s]["per_arm"]["loop L1"]["val_accuracy"]
                - LRN[s]["per_arm"]["Y"]["val_accuracy"])
        assert d <= LRN[s]["prediction"]["val_acc_tol_vs_Y"], (s, d)
    ok.append("val accuracy within 0.02 of Y on 3/3")

    # ---- finding 88, only once its artifacts are on disk -------------------
    if FIX_ARMS:
        assert sorted(FIX) == [0, 1, 2], sorted(FIX)
        a10_0, a10_1, a30_1 = l0, l1, l1s
        for s, v_ in zip([0, 1, 2], [0.0306, 0.1258, 0.0661]):
            chk(f"fix0.10 L1 val seed {s}",
                FIX[s]["per_arm"][a10_1]["val_accuracy"], v_, 5e-5)
        for s, v_ in zip([0, 1, 2], [0.0522, 0.1785, 0.0706]):
            chk(f"fix0.10 L0 val seed {s}",
                FIX[s]["per_arm"][a10_0]["val_accuracy"], v_, 5e-5)
        for s, v_ in zip([0, 1, 2], [0.1564, 0.0766, 0.0444]):
            chk(f"fix0.30 L1 val seed {s}",
                FIX[s]["per_arm"][a30_1]["val_accuracy"], v_, 5e-5)
        # the deck says EVERY forced row trips BOTH registered falsifiers
        for s in [0, 1, 2]:
            for a in FIX_ARMS:
                r = FIX[s]["per_arm"][a]
                assert r["val_accuracy"] < FIX[s]["prediction"][
                    "break_val_acc_below"], (s, a, r["val_accuracy"])
                assert r["final_train_nll"] > FIX[s]["prediction"][
                    "break_nll_above"], (s, a, r["final_train_nll"])
        ok.append("all 9 forced rows below val 0.30 and above nll 2.5")
        # the first forward pass, before any step on the forced body
        for s, n_ in zip([0, 1, 2], [9.18, 6.45, 6.73]):
            chk(f"fix0.10 L1 step-0 nll seed {s}",
                FIX[s]["history"][a10_1][0]["train_nll"], n_, 0.005)
        # the calibration missed its target at step 0, in both directions
        for s, f_ in zip([0, 1, 2], [5.1, 8.4, 3.1]):
            chk(f"fix0.10 step-0 fb% seed {s}",
                100 * FIX[s]["history"][a10_1][0]["fb_rel"], f_, 0.05)
        for s, f_ in zip([0, 1, 2], [9.1, 13.1, 96.0]):
            chk(f"fix0.30 step-0 fb% seed {s}",
                100 * FIX[s]["history"][a30_1][0]["fb_rel"], f_, 0.05)
        for s, c_ in zip([0, 1, 2], [2.572, 1.288, 2.682]):
            chk(f"solved c at 0.10 seed {s}",
                FIX[s]["per_arm"][a10_1]["loop_scale"], c_, 0.001)
        # the denominator trap, which is the whole reading of the guard column
        for s, d_ in zip([0, 1, 2], [656, 637, 584]):
            chk(f"Y denom seed {s}", FIX[s]["per_arm"]["Y"]["guard_denom"],
                d_, 0)
        for s, d_ in zip([0, 1, 2], [0, 46, 17]):
            chk(f"fix0.10 L1 denom seed {s}",
                FIX[s]["per_arm"][a10_1]["guard_denom"], d_, 0)
        chk("seed 1's only 'win'", FIX[1]["per_arm"][a10_1]["guard_rate"],
            0.2826, 5e-5)
        for s, n_ in zip([0, 1, 2], [0, 29, 0]):
            chk(f"fix0.10 L1 flips seed {s}",
                FIX[s]["per_arm"][a10_1]["total_flips"], n_, 0)
        # the deck's remaining forced-run numbers
        for s, n_ in zip([0, 1, 2], [3.98, 3.38, 3.71]):
            chk(f"fix0.10 L1 final nll seed {s}",
                FIX[s]["per_arm"][a10_1]["final_train_nll"], n_, 0.005)
        for s, a_ in zip([0, 1, 2], [0.4275, 0.874, 0.569]):
            chk(f"fix0.10 L1 corrupt seed {s}",
                FIX[s]["per_arm"][a10_1]["pair_auroc_corrupt"], a_, 0.0005)
        for s, n_ in zip([0, 1, 2], [0, 3, 3]):
            chk(f"fix0.10 L0 flips seed {s}",
                FIX[s]["per_arm"][a10_0]["total_flips"], n_, 0)
        # c frozen does not freeze the ratio: seed 0 ended at 377% of mean|h|
        chk("seed 0's end-of-training ratio",
            100 * FIX[0]["per_arm"][a10_1]["fb_rel"], 377.0, 0.5)

    # ---- finding 89, the geometry job ------------------------------------
    if DIR:
        cl = {s: [r["cos_delta"] for r in dmode(s, "closed")] for s in [0, 1, 2]}
        dh = {s: [r["cos_dh"] for r in dmode(s, "closed")] for s in [0, 1, 2]}
        ch = {s: [r["cos_h"] for r in dmode(s, "closed")] for s in [0, 1, 2]}
        for s, (a_, b_) in zip([0, 1, 2], [(0.70, 0.82), (0.65, 0.83),
                                           (0.37, 0.61)]):
            chk(f"cos(delta) min seed {s}", min(cl[s]), a_, 0.005)
            chk(f"cos(delta) max seed {s}", max(cl[s]), b_, 0.005)
        # the two controls, quoted as one range across all seeds and all t
        chk("cos(dh) min", min(min(v) for v in dh.values()), 0.06, 0.005)
        chk("cos(dh) max", max(max(v) for v in dh.values()), 0.30, 0.005)
        chk("cos(h) min", min(min(v) for v in ch.values()), 0.06, 0.005)
        chk("cos(h) max", max(max(v) for v in ch.values()), 0.16, 0.005)
        # the gap the deck reports is the PER-APPLICATION minimum, which is the
        # statistic the registered prediction named ("at every application")
        for s, g_ in zip([0, 1, 2], [0.50, 0.53, 0.15]):
            chk(f"cos gap min seed {s}",
                min(a - b for a, b in zip(cl[s], dh[s])), g_, 0.005)
        for s in [0, 1, 2]:
            assert min(a - b for a, b in zip(cl[s], dh[s])) > DIR[
                "prediction"]["falsifier_cos_gap_below"], s
        ok.append("the registered falsifier did not fire on any seed")
        for s, r_ in zip([0, 1, 2], [4.66, 3.94, 4.90]):
            chk(f"first injection r1 seed {s}", dmode(s, "closed")[0]["r1"],
                r_, 0.005)
        for s, c_ in zip([0, 1, 2], [2.57, 1.29, 2.68]):
            chk(f"89's read of 88's c seed {s}", dseed(s)["c_solved"], c_,
                0.005)
        for s, p_ in zip([0, 1, 2], [1197, 508, 1315]):
            chk(f"first injection % seed {s}",
                100 * dseed(s)["c_solved"] * dmode(s, "closed")[0]["r1"], p_,
                1.0)
        for s, r_ in zip([0, 1, 2], [0.0402, 0.0785, 0.0384]):
            chk(f"_fb_rel at c=1 seed {s}", dseed(s)["r1_unit_last"], r_,
                5e-5)
        for s, d_ in zip([0, 1, 2], [3.5, 1.2, 2.9]):
            chk(f"reproduction of 88's c seed {s}",
                100 * dseed(s)["r1_rel_diff"], d_, 0.05)
        allr = [r["r1"] for s in [0, 1, 2] for r in dmode(s, "closed")]
        chk("ratio range low", min(allr), 3.94, 0.005)
        chk("ratio range high", max(allr), 8.16, 0.005)
        for s in [0, 1, 2]:          # larger at the end than at the start, 3/3
            assert dmode(s, "closed")[-1]["r1"] > dmode(s, "closed")[0]["r1"], s
        ok.append("the ratio is larger at t=14 than at t=0 on 3/3")
        chk("open-mode cos(dh) at t=14", dmode(0, "open")[-1]["cos_dh"], 0.99,
            0.005)
        chk("open-mode cos(h) at t=14", dmode(0, "open")[-1]["cos_h"], 0.95,
            0.005)

    # ---- finding 90, the corrected calibration ---------------------------
    if FST:
        assert sorted(FST) == [0, 1, 2], sorted(FST)
        A1, A0 = "fix0.10 L1", "fix0.10 L0"
        for s, c_ in zip([0, 1, 2], [0.021342, 0.025401, 0.020268]):
            chk(f"90 c seed {s}", FST[s]["per_arm"][A1]["loop_scale"], c_,
                5e-7)
        # the body survives: the registered reading, and the first thing read
        for s, v_ in zip([0, 1, 2], [0.4749, 0.4749, 0.4498]):
            chk(f"90 L1 val seed {s}", FST[s]["per_arm"][A1]["val_accuracy"],
                v_, 5e-5)
        for s, v_ in zip([0, 1, 2], [0.4828, 0.4821, 0.4564]):
            chk(f"90 Y val seed {s}", FST[s]["per_arm"]["Y"]["val_accuracy"],
                v_, 5e-5)
        for s in [0, 1, 2]:
            d = abs(FST[s]["per_arm"][A1]["val_accuracy"]
                    - FST[s]["per_arm"]["Y"]["val_accuracy"])
            assert d <= 0.008, (s, d)
            assert FST[s]["per_arm"][A1]["val_accuracy"] > FST[s][
                "prediction"]["break_val_acc_below"], s
            assert FST[s]["per_arm"][A1]["final_train_nll"] < FST[s][
                "prediction"]["nll_below"], s
        ok.append("90: val within 0.008 of Y, neither falsifier fires, 3/3")
        for s, n_ in zip([0, 1, 2], [5.2816, 4.9908, 5.8986]):
            chk(f"90 step-0 nll seed {s}",
                FST[s]["history"][A1][0]["train_nll"], n_, 0.005)
        for s, n_ in zip([0, 1, 2], [1.6887, 1.6871, 1.8617]):
            chk(f"90 step-500 nll seed {s}",
                FST[s]["history"][A1][1]["train_nll"], n_, 0.005)
        assert [r["step"] for r in FST[0]["history"][A1]][1] == 500
        for s, n_ in zip([0, 1, 2], [1.6851, 1.6737, 1.7132]):
            chk(f"90 final nll seed {s}",
                FST[s]["per_arm"][A1]["final_train_nll"], n_, 0.005)
        # the ratio stayed where it was put, which 88's never did
        for s, f_ in zip([0, 1, 2], [5.1, 7.4, 5.0]):
            chk(f"90 end fb% seed {s}", 100 * FST[s]["per_arm"][A1]["fb_rel"],
                f_, 0.05)
        for s in [0, 1, 2]:
            lo_, hi_ = FST[s]["prediction"]["fb_rel_last_band"]
            assert lo_ <= FST[s]["per_arm"][A1]["fb_rel"] <= hi_, s
        ok.append("90: the last application's ratio ends inside 5-25%, 3/3")
        # the predicate
        for s, a_ in zip([0, 1, 2], [0.8710, 0.8680, 0.8680]):
            chk(f"90 P1 seed {s}", FST[s]["per_arm"][A1]["pair_auroc_corrupt"],
                a_, 5e-5)
        for s, a_ in zip([0, 1, 2], [0.4430, 0.4420, 0.4410]):
            chk(f"90 P2 seed {s}", FST[s]["per_arm"][A1]["pair_auroc_placebo"],
                a_, 5e-5)
        for s, a_ in zip([0, 1, 2], [0.9210, 0.9163, 0.9132]):
            chk(f"90 pooled seed {s}", FST[s]["per_arm"][A1]["pooled_auroc"],
                a_, 5e-5)
        for s in [0, 1, 2]:
            p = FST[s]["per_arm"][A1]
            assert p["pair_auroc_corrupt"] >= SP.PAIR_BAR, s
            assert SP.PLACEBO_BAND[0] < p["pair_auroc_placebo"] < SP.PLACEBO_BAND[1], s
            assert p["pooled_auroc"] >= SP.POOLED_BAR, s
        ok.append("90: P1, P2 and P3 on 3/3")
        # the denominator did NOT collapse this time
        for s, d_ in zip([0, 1, 2], [481, 480, 451]):
            chk(f"90 L1 denom seed {s}", FST[s]["per_arm"][A1]["guard_denom"],
                d_, 0)
        for s, d_ in zip([0, 1, 2], [506, 502, 429]):
            chk(f"90 pair denom seed {s}",
                FST[s]["per_arm"]["pair L1"]["guard_denom"], d_, 0)
        # the load-bearing column, and the envelope it has to beat
        for s, g_ in zip([0, 1, 2], [0.1622, 0.1646, 0.1530]):
            chk(f"90 L1 guard seed {s}", FST[s]["per_arm"][A1]["guard_rate"],
                g_, 5e-5)
        for s, d_ in zip([0, 1, 2], [0.0535, -0.0426, -0.0102]):
            chk(f"90 guard delta seed {s}",
                FST[s]["per_arm"][A1]["guard_rate"]
                - FST[s]["per_arm"]["pair L1"]["guard_rate"], d_, 5e-5)
        beats = sum(FST[s]["per_arm"][A1]["guard_rate"]
                    >= FST[s]["per_arm"]["pair L1"]["guard_rate"]
                    for s in [0, 1, 2])
        assert beats == 1, beats          # the deck says 1 of 3; needed 2
        ok.append("90: the guard beats its own pair row on exactly 1 of 3")
        for s, n_ in zip([0, 1, 2], [166, 184, 157]):
            chk(f"90 L1 flips seed {s}", FST[s]["per_arm"][A1]["total_flips"],
                n_, 0)
        under = sum(FST[s]["per_arm"][A1]["total_flips"]
                    <= FST[s]["per_arm"]["pair L1"]["total_flips"]
                    for s in [0, 1, 2])
        assert under == 1, under
        ok.append("90: churn at or below its own pair row on exactly 1 of 3")
        # no seed has BOTH a higher guard and no more churn -- the overturn
        # condition the sbatch registered, checked rather than asserted away
        both = sum((FST[s]["per_arm"][A1]["guard_rate"]
                    > FST[s]["per_arm"]["pair L1"]["guard_rate"])
                   and (FST[s]["per_arm"][A1]["total_flips"]
                        <= FST[s]["per_arm"]["pair L1"]["total_flips"])
                   for s in [0, 1, 2])
        assert both == 0, both
        ok.append("90: the overturn condition is met on 0 of 3")
        # the same column in cases rather than rates: rate x denom is an
        # integer count of withheld emissions and the deck states it as one
        for s, (a_, p_) in zip([0, 1, 2], [(78, 55), (79, 104), (69, 70)]):
            for arm, want in ((A1, a_), ("pair L1", p_)):
                r = FST[s]["per_arm"][arm]
                chk(f"withheld count {arm} seed {s}",
                    round(r["guard_rate"] * r["guard_denom"]), want, 0)
        # the placebo floor, reported but NOT applied: whether P4 should be the
        # raw conditional or this excess is hshunt's call, not this script's
        for s, g_ in zip([0, 1, 2], [0.0936, 0.0938, 0.1153]):
            chk(f"90 placebo guard seed {s}",
                FST[s]["per_arm"][A1]["placebo_guard_rate"], g_, 5e-5)
        for s, g_ in zip([0, 1, 2], [0.0672, 0.0777, 0.1026]):
            chk(f"pair placebo guard seed {s}",
                FST[s]["per_arm"]["pair L1"]["placebo_guard_rate"], g_, 5e-5)
        # the no-margin control, and 87's closed-loop control for its churn
        for s, g_ in zip([0, 1, 2], [0.0016, 0.0032, 0.0035]):
            chk(f"90 L0 guard seed {s}", FST[s]["per_arm"][A0]["guard_rate"],
                g_, 5e-5)
        for s, n_ in zip([0, 1, 2], [9, 10, 10]):
            chk(f"90 L0 flips seed {s}", FST[s]["per_arm"][A0]["total_flips"],
                n_, 0)
        for s, n_ in zip([0, 1, 2], [7, 5, 11]):
            chk(f"87 L0 flips seed {s}", LRN[s]["per_arm"]["loop L0"]
                ["total_flips"], n_, 0)
        for s, n_ in zip([0, 1, 2], [4, 1, 7]):
            chk(f"Y flips again seed {s}", FST[s]["per_arm"]["Y"]
                ["total_flips"], n_, 0)
        # the guard spread across three trainings of the same margin arm, which
        # is the number the deck uses to say the movement is not a movement
        for s, (a_, b_) in zip([0, 1, 2], [(0.1072, 0.1622), (0.1646, 0.2211),
                                           (0.1416, 0.1632)]):
            vals = [FST[s]["per_arm"]["pair L1"]["guard_rate"],
                    LRN[s]["per_arm"]["loop L1"]["guard_rate"],
                    FST[s]["per_arm"][A1]["guard_rate"]]
            chk(f"guard spread low seed {s}", min(vals), a_, 5e-5)
            chk(f"guard spread high seed {s}", max(vals), b_, 5e-5)

    print(f"\ncheck_claims: {len(ok)} assertions pass")
    print("forced-arm pages:", "built" if FIX_ARMS else "SKIPPED (52478871 "
          "has not landed; the deck must not state its numbers)")
    print("finding 89 pages:", "built" if DIR else "SKIPPED")
    print("finding 90 pages:", "built" if FST else "SKIPPED")
    return ok


if __name__ == "__main__":
    check_claims()
