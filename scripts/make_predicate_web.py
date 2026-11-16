"""Figures for the `spin-gnn/predicate` deck -- findings 82 and 83.

Every number is READ OUT OF THE JOBS' OWN JSON and none is retyped; the
McNemar decomposition is recomputed from the saved per-case rows. `check_claims()`
at the bottom asserts the values the deck's prose states against what was read,
so a figure and its caption cannot drift apart -- the pattern
make_margin_web.py, make_channel_web.py and make_stagez_web.py established.

Sources, all COMPLETED 0:0:
  52452867  stagepairProbe      three models, one case list, P1-P4   (finding 82)
  52455457  stagepairGuardSpec  the conditional placebo control       (finding 83)
  52456376  stagepairSeeds12    seeds 1 and 2, for MIN_SEEDS = 2      (if present)

The seed-1/2 panels are SKIPPED rather than faked when that job has not landed,
and `check_claims` reports which pages exist.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors

sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")
import stagepair_score as SP

ROOT = "/scratch/users/hshunt/spin_gnn"
OUT = "/home/users/hshunt/slides/decks/spin-gnn/predicate/figures"
PROBE = os.path.join(ROOT, "runs/stagepair_pairprobe_2026-11-15")
GUARD = os.path.join(ROOT, "runs/stagepair_guardspec_2026-11-15")

ARMS = ["Y", "warm L0", "warm L1"]
PRETTY = {"Y": "stage Y", "warm L0": r"warm $\lambda=0$",
          "warm L1": r"warm $\lambda=1$"}

os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
probe = json.load(open(os.path.join(PROBE, "pair_summary.json")))["per_arm"]
gspec = json.load(open(os.path.join(GUARD, "guardspec_summary.json")))["per_arm"]

# Seeds 1 and 2, if 52456376 has landed. Keyed by seed so a partial job (seed 1
# done, seed 2 still running) still produces the pages it can support.
seeds = {0: probe}
for d in sorted(glob.glob(os.path.join(ROOT,
                                       "runs/stagepair_pairprobe_2026-11-15_s*"))):
    f = os.path.join(d, "pair_summary.json")
    if os.path.isfile(f):
        seeds[int(d.rsplit("_s", 1)[1])] = json.load(open(f))["per_arm"]
print("seeds with a summary on disk:", sorted(seeds))


def mcnemar(arm):
    """Recomputed from the per-case rows, not read from a printed table.

    Returns the two discordant counts among clean-emitting cases, the exact
    binomial p, and the placebo-corrected excess with a normal interval on the
    difference of two CORRELATED proportions (SE = sqrt(b+c)/n, the standard
    form -- the cases are paired, so the independent-sample SE would be wrong).
    """
    rows = [json.loads(l) for l in
            open(os.path.join(GUARD, "rows_%s.jsonl" % arm.replace(" ", "_")))]
    on = [r for r in rows if r["clean_emits_v"]]
    off = [r for r in rows if not r["clean_emits_v"]]
    n = len(on)
    b = sum(1 for r in on if not r["corrupt_emits_v"] and r["placebo_emits_v"])
    c = sum(1 for r in on if r["corrupt_emits_v"] and not r["placebo_emits_v"])
    both = sum(1 for r in on
               if not r["corrupt_emits_v"] and not r["placebo_emits_v"])
    diff, se = (b - c) / n, np.sqrt(b + c) / n
    # the other tail: among clean-SILENT cases, which perturbation turns v on
    b2 = sum(1 for r in off if r["corrupt_emits_v"] and not r["placebo_emits_v"])
    c2 = sum(1 for r in off if not r["corrupt_emits_v"] and r["placebo_emits_v"])
    return {
        "n_on": n, "corrupt_only": b, "placebo_only": c, "both": both,
        "p": stats.binomtest(b, b + c, 0.5).pvalue if b + c else float("nan"),
        "excess": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
        "n_off": len(off), "on_corrupt_only": b2, "on_placebo_only": c2,
        "p_on": (stats.binomtest(b2, b2 + c2, 0.5).pvalue
                 if b2 + c2 else float("nan")),
        "total_flips": sum(1 for r in rows
                           if r["clean_emits_v"] != r["corrupt_emits_v"]
                           or r["clean_emits_v"] != r["placebo_emits_v"]),
    }


MC = {a: mcnemar(a) for a in ARMS}
for a in ARMS:
    print(f"  {a:<9} corruptOnly={MC[a]['corrupt_only']:3d} "
          f"placeboOnly={MC[a]['placebo_only']:3d} both={MC[a]['both']:3d} "
          f"p={MC[a]['p']:.2e} excess={MC[a]['excess']:+.4f} "
          f"flips={MC[a]['total_flips']:3d}")

# Stage Y's own three bodies, read from their training histories. This is the
# comparison the convergence threshold has to survive, and it is not a number
# the probe printed.
Y_HIST = []
for d in sorted(glob.glob(os.path.join(
        ROOT, "sudoku_n48_determinacy_Y_2026-11-14b/t0*/d0320*/history.json"))):
    Y_HIST.append(float(json.load(open(d))[-1]["train_nll"]))
print("stage Y epoch-mean train nll by seed:", [round(v, 4) for v in Y_HIST])

THRESH = SP.FLOOR + SP.STUCK_TOL


def bars(ax, vals, colors, width=0.38, offset=0.0, hatch=None, label_fmt="%.4f"):
    x = np.arange(len(vals)) + offset
    b = ax.bar(x, vals, width, color=colors, hatch=hatch, rasterized=True)
    for xi, v in zip(x, vals):
        ax.annotate(label_fmt % v, (xi, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=9)
    return b


# ---- 1. the headline: corrupt vs placebo matched-pair AUROC ---------------
# Two groups per arm, so the arms are the x axis and the perturbation is the
# pair. Three arms x two bars is six, which is a table's worth of numbers, but
# the two reference LINES are the whole point and a table cannot draw them.
fig, ax = plt.subplots(figsize=(7.6, 4.4))
cor = [probe[a]["pair_auroc_corrupt"] for a in ARMS]
pla = [probe[a]["pair_auroc_placebo"] for a in ARMS]
c_hi, c_lo = series_colors(2, highlight=0)
bars(ax, cor, [c_hi] * 3, offset=-0.20)
bars(ax, pla, [c_lo] * 3, offset=+0.20)
ax.axhline(0.5, color=MUTED, lw=1.0, ls=":")
ax.axhline(SP.PAIR_BAR, color=MUTED, lw=1.0, ls="--")
ax.text(-0.45, SP.PAIR_BAR + 0.012, "P1 bar 0.65", color=MUTED, fontsize=9)
ax.text(-0.45, 0.512, "chance 0.50", color=MUTED, fontsize=9)
# The series words go on the LAST group, not the first: on the first group the
# two bars are 0.06 apart, so both words land in the same strip of air as the
# two reference-line labels and the four of them pile up.
ax.annotate("broken prompt", (1.80, cor[2]), textcoords="offset points",
            xytext=(0, 20), ha="center", color=c_hi, fontsize=9.5)
# 44 and not 20: at 20 the word lands a pixel under the dotted chance line.
# But 44 lifts it clear of its own bar and into the strip between the two grey
# reference lines, where a grey word reads as a line label. A thin leader back
# down to the bar top is the cheapest way to say which thing it names.
ax.annotate("placebo", (2.20, pla[2]), textcoords="offset points",
            xytext=(0, 44), ha="center", color=c_lo, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=c_lo, lw=0.7,
                            shrinkA=3, shrinkB=14))
ax.set_xticks(np.arange(3))
ax.set_xticklabels([PRETTY[a] for a in ARMS])
ax.set_ylabel("matched-pair AUROC")
ax.set_ylim(0, 1.0)
style_axes(ax)
save(fig, "auroc_pair.png")

# ---- 2. the guard, with its control ---------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.4))
g = [gspec[a]["corrupt"]["rate_off_given_on"] for a in ARMS]
p = [gspec[a]["placebo"]["rate_off_given_on"] for a in ARMS]
bars(ax, g, [c_hi] * 3, offset=-0.20)
bars(ax, p, [c_lo] * 3, offset=+0.20)
ax.axhline(SP.GUARD_BAR, color=MUTED, lw=1.0, ls="--")
ax.text(-0.45, SP.GUARD_BAR + 0.004, "P4 bar 0.10", color=MUTED, fontsize=9)
ax.annotate("broken prompt", (1.80, g[2]), textcoords="offset points",
            xytext=(0, 20), ha="center", color=c_hi, fontsize=9.5)
ax.annotate("placebo", (2.20, p[2]), textcoords="offset points",
            xytext=(0, 34), ha="center", color=c_lo, fontsize=9.5)
ax.set_xticks(np.arange(3))
ax.set_xticklabels([PRETTY[a] for a in ARMS])
ax.set_ylabel(r"$P(\mathrm{stops\ emitting}\ v \mid \mathrm{clean\ emits}\ v)$")
ax.set_ylim(0, 0.145)
style_axes(ax)
save(fig, "guard_vs_placebo.png")

# ---- 3. the placebo-corrected excess against the bar ----------------------
fig, ax = plt.subplots(figsize=(7.2, 4.0))
m = MC["warm L1"]
ax.errorbar([0], [m["excess"]], yerr=[[m["excess"] - m["lo"]],
                                      [m["hi"] - m["excess"]]],
            fmt="o", color=SERIES[0], capsize=5, lw=1.8, ms=7)
ax.plot([1], [g[2]], "o", color=MUTED, ms=7)
ax.axhline(SP.GUARD_BAR, color=MUTED, lw=1.0, ls="--")
# va="bottom" at the far left, because va="center" on the line's own y puts the
# dashes straight through the words -- the defect the margin deck shipped three
# of before the second read-back pass caught them.
ax.text(-0.42, SP.GUARD_BAR + 0.003, "P4 bar 0.10", color=MUTED, fontsize=9,
        va="bottom", ha="left")
ax.axhline(0.0, color=MUTED, lw=0.8, ls=":")
ax.annotate("excess over the placebo\n%.4f [%.4f, %.4f]"
            % (m["excess"], m["lo"], m["hi"]), (0, m["excess"]),
            textcoords="offset points", xytext=(12, 0), va="center",
            color=SERIES[0], fontsize=9.5)
ax.annotate("P4 as registered\n%.4f" % g[2], (1, g[2]),
            textcoords="offset points", xytext=(12, 0), va="center",
            color=MUTED, fontsize=9.5)
ax.set_xlim(-0.5, 2.3)
ax.set_xticks([])
ax.set_ylim(-0.01, 0.145)
ax.set_ylabel("guard rate")
style_axes(ax)
save(fig, "guard_excess.png")

# ---- 4. the channel got loud on BOTH perturbations ------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.4))
dc = [probe[a]["median_abs_dg_corrupt"] for a in ARMS]
dp = [probe[a]["median_abs_dg_placebo"] for a in ARMS]
bars(ax, dc, [c_hi] * 3, offset=-0.20, label_fmt="%.3f")
bars(ax, dp, [c_lo] * 3, offset=+0.20, label_fmt="%.3f")
ax.set_yscale("log")
# Stacked, not side by side: at equal height the two words butt into each other,
# because "broken prompt" is wider than the 0.4 of x it is centred in.
ax.annotate("broken prompt", (-0.20, dc[0]), textcoords="offset points",
            xytext=(0, 16), ha="center", color=c_hi, fontsize=9.5)
# Same leader as figure 1: stacking buys clearance at the cost of a 40-point gap
# between the word and the bar it names, and on a log axis that gap is wide.
ax.annotate("placebo", (0.20, dp[0]), textcoords="offset points",
            xytext=(0, 40), ha="center", color=c_lo, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=c_lo, lw=0.7,
                            shrinkA=3, shrinkB=14))
ax.set_xticks(np.arange(3))
ax.set_xticklabels([PRETTY[a] for a in ARMS])
ax.set_ylabel(r"median $|g_{\rm clean}-g_{\rm perturbed}|$")
style_axes(ax)
save(fig, "dg_magnitude.png")

# ---- 5. emission stability: how often one token flips the answer ----------
fig, ax = plt.subplots(figsize=(7.2, 4.0))
fl = [MC[a]["total_flips"] for a in ARMS]
bars(ax, fl, series_colors(3, highlight=2), offset=0.0, width=0.5,
     label_fmt="%d")
ax.set_xticks(np.arange(3))
ax.set_xticklabels([PRETTY[a] for a in ARMS])
ax.set_ylabel("cases of 1,000 where a one-token edit\nchanged whether $v$ is emitted")
style_axes(ax)
save(fig, "flips.png")

# ---- 6. the convergence threshold sits inside stage Y's own spread --------
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, v in enumerate(Y_HIST):
    ax.plot([i], [v], "o", color=MUTED, ms=8)
    ax.annotate("seed %d" % i, (i, v), textcoords="offset points",
                xytext=(0, 10), ha="center", color=MUTED, fontsize=9)
ax.annotate("stage Y's own three bodies", (1, min(Y_HIST)),
            textcoords="offset points", xytext=(0, -22), ha="center",
            color=MUTED, fontsize=9.5)
arm_nll = probe["warm L1"]["final_train_nll"]
# One point per seed that has landed. With all three the page stops being "the
# arm missed" and becomes "every arm missed, and so did one of Y's own bodies",
# which is the actual question about the threshold.
arm_x = [2.9 + 0.45 * i for i in range(len(seeds))]
arm_y = [seeds[k]["warm L1"]["final_train_nll"] for k in sorted(seeds)]
ax.plot(arm_x, arm_y, "o", color=SERIES[0], ms=8)
for x, y, k in zip(arm_x, arm_y, sorted(seeds)):
    ax.annotate("%.4f" % y, (x, y), textcoords="offset points",
                xytext=(0, 10), ha="center", color=SERIES[0], fontsize=9)
# ha="right", because at the right-hand end of the x range a left-aligned label
# runs off the axis -- the clipping the margin deck's second pass caught.
ax.annotate(r"the $\lambda=1$ arms", (arm_x[0], min(arm_y)),
            textcoords="offset points", xytext=(-12, 0), va="center",
            ha="right", color=SERIES[0], fontsize=9.5)
ax.axhline(THRESH, color=SERIES[1], lw=1.2, ls="--")
ax.text(-0.42, THRESH + 0.003, "the rule admits anything below %.3f" % THRESH,
        color=SERIES[1], fontsize=9, va="bottom")
ax.set_xlim(-0.5, max(arm_x) + 0.45)
# Headroom top and bottom: the extreme points sat on the frame without it.
ax.set_ylim(min(min(Y_HIST), min(arm_y)) - 0.019,
            max(max(arm_y), max(Y_HIST)) + 0.010)
ax.set_xticks([])
ax.set_ylabel("train nll at the end of training")
style_axes(ax)
save(fig, "converged.png")

# ---- 7. seeds, if 52456376 has landed -------------------------------------
if len(seeds) > 1:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ks = sorted(seeds)
    cs = [seeds[k]["warm L1"]["pair_auroc_corrupt"] for k in ks]
    ps = [seeds[k]["warm L1"]["pair_auroc_placebo"] for k in ks]
    ax.plot(ks, cs, "o-", color=c_hi, lw=1.8, ms=7)
    ax.plot(ks, ps, "o-", color=c_lo, lw=1.8, ms=7)
    label_line(ax, ks[-1], cs[-1], "broken prompt", color=c_hi)
    label_line(ax, ks[-1], ps[-1], "placebo", color=c_lo)
    ax.axhline(SP.PAIR_BAR, color=MUTED, lw=1.0, ls="--")
    ax.text(ks[0] - 0.08, SP.PAIR_BAR + 0.012, "P1 bar 0.65", color=MUTED,
            fontsize=9)
    for y in SP.PLACEBO_BAND:
        ax.axhline(y, color=MUTED, lw=0.8, ls=":")
    ax.text(ks[0] - 0.08, SP.PLACEBO_BAND[1] + 0.012, "P2 band",
            color=MUTED, fontsize=9)
    ax.set_xticks(ks)
    ax.set_xlabel("stage Y body warmed from (seed)")
    ax.set_ylabel("matched-pair AUROC")
    ax.set_ylim(0, 1.0)
    style_axes(ax)
    save(fig, "seeds.png")
else:
    print("SKIPPED seeds.png -- only seed 0 on disk")


# ---------------------------------------------------------------------------
# what the deck's prose says, asserted against what was read
# ---------------------------------------------------------------------------
def check_claims():
    def eq(name, got, want, tol=5e-5):
        assert abs(got - want) <= tol, "%s: read %.6f, deck says %.6f" % (
            name, got, want)
        print("  ok  %-26s %.4f" % (name, got))

    eq("P1 corrupt (L1)", probe["warm L1"]["pair_auroc_corrupt"], 0.8710)
    eq("P2 placebo (L1)", probe["warm L1"]["pair_auroc_placebo"], 0.4360)
    eq("P3 pooled (L1)", probe["warm L1"]["pooled_auroc"], 0.9271)
    eq("P1 corrupt (Y)", probe["Y"]["pair_auroc_corrupt"], 0.4560)
    eq("P1 corrupt (L0)", probe["warm L0"]["pair_auroc_corrupt"], 0.4890)
    eq("pooled (Y)", probe["Y"]["pooled_auroc"], 0.9495)
    eq("guard (L1)", gspec["warm L1"]["corrupt"]["rate_off_given_on"], 0.1087,
       tol=5e-4)
    eq("guard (Y)", gspec["Y"]["corrupt"]["rate_off_given_on"], 0.0015,
       tol=5e-4)
    eq("placebo guard (L1)", gspec["warm L1"]["placebo"]["rate_off_given_on"],
       0.0672, tol=5e-4)
    eq("excess (L1)", MC["warm L1"]["excess"], 0.0415, tol=5e-4)
    eq("excess lo", MC["warm L1"]["lo"], 0.0108, tol=5e-4)
    eq("excess hi", MC["warm L1"]["hi"], 0.0722, tol=5e-4)
    eq("|dg| placebo (L1)", probe["warm L1"]["median_abs_dg_placebo"], 0.6686,
       tol=5e-4)
    eq("|dg| placebo (Y)", probe["Y"]["median_abs_dg_placebo"], 0.0111,
       tol=5e-4)
    eq("train nll (L1)", probe["warm L1"]["final_train_nll"], 1.6636, tol=5e-4)

    assert gspec["warm L1"]["corrupt"]["denom_clean_on"] == 506
    assert gspec["Y"]["corrupt"]["denom_clean_on"] == 656
    assert gspec["warm L0"]["corrupt"]["denom_clean_on"] == 648
    assert (MC["warm L1"]["corrupt_only"], MC["warm L1"]["placebo_only"],
            MC["warm L1"]["both"]) == (42, 21, 13)
    assert (MC["warm L1"]["on_corrupt_only"],
            MC["warm L1"]["on_placebo_only"]) == (15, 38)
    assert MC["warm L1"]["p"] < 0.05 and MC["warm L1"]["p_on"] < 0.05
    assert MC["warm L1"]["hi"] < SP.GUARD_BAR, "the excess CI must clear-fail"
    assert [round(v, 4) for v in Y_HIST] == [1.5969, 1.5933, 1.6577], Y_HIST
    assert Y_HIST[2] > THRESH > Y_HIST[1], "Y's spread must straddle 1.630"
    assert SP.N_TRAINABLE == 614138
    print("  ok  the registered bars 0.65 / (0.40,0.60) / 0.90 / 0.10 and the")
    print("      614,138-parameter cell, imported from stagepair_score")
    if len(seeds) > 1:
        for k in sorted(seeds):
            print("  seed %d: corrupt %.4f  placebo %.4f  nll %.4f" % (
                k, seeds[k]["warm L1"]["pair_auroc_corrupt"],
                seeds[k]["warm L1"]["pair_auroc_placebo"],
                seeds[k]["warm L1"]["final_train_nll"]))
    else:
        print("  NOTE seeds 1-2 absent; the deck must not claim a spread")


check_claims()
print("check_claims: all green")
