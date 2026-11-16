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


def mcnemar(arm, guard_dir=None):
    """Recomputed from the per-case rows, not read from a printed table.

    Returns the two discordant counts among clean-emitting cases, the exact
    binomial p, and the placebo-corrected excess with a normal interval on the
    difference of two CORRELATED proportions (SE = sqrt(b+c)/n, the standard
    form -- the cases are paired, so the independent-sample SE would be wrong).
    """
    rows = [json.loads(l) for l in
            open(os.path.join(guard_dir or GUARD,
                              "rows_%s.jsonl" % arm.replace(" ", "_")))]
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

# The same decomposition for the other two seeds, from 52460087. Seed 0's rows
# live in the unsuffixed directory; _recheck is that job's bit-for-bit rerun of
# seed 0 and is deliberately NOT read here -- it is a regression test, not data.
GUARD_SEEDS = {0: GUARD}
for d in sorted(glob.glob(os.path.join(
        ROOT, "runs/stagepair_guardspec_2026-11-15_s*"))):
    if os.path.isfile(os.path.join(d, "guardspec_summary.json")):
        GUARD_SEEDS[int(d.rsplit("_s", 1)[1])] = d
MCS = {k: mcnemar("warm L1", v) for k, v in GUARD_SEEDS.items()}
GSEED = {k: json.load(open(os.path.join(v, "guardspec_summary.json")))["per_arm"]
         for k, v in GUARD_SEEDS.items()}
for k in sorted(MCS):
    m, raw = MCS[k], GSEED[k]["warm L1"]["corrupt"]["rate_off_given_on"]
    print(f"  seed {k} guard: raw {raw:.4f}  excess {m['excess']:+.4f} "
          f"[{m['lo']:+.4f}, {m['hi']:+.4f}]  fraction {m['excess']/raw:.2f}  "
          f"cOnly={m['corrupt_only']} pOnly={m['placebo_only']} "
          f"both={m['both']} of {m['n_on']}")

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

# ---- 3b. the same correction on all three seeds ---------------------------
# The page this figure answers: seed 0's excess fell below the bar, so does the
# correction hold everywhere? It does not, and the figure has to show that
# rather than an average that hides it.
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ks = sorted(MCS)
ex = [MCS[k]["excess"] for k in ks]
raw = [GSEED[k]["warm L1"]["corrupt"]["rate_off_given_on"] for k in ks]
ax.errorbar(ks, ex,
            yerr=[[MCS[k]["excess"] - MCS[k]["lo"] for k in ks],
                  [MCS[k]["hi"] - MCS[k]["excess"] for k in ks]],
            fmt="o-", color=c_hi, capsize=5, lw=1.8, ms=7)
ax.plot(ks, raw, "o--", color=c_lo, lw=1.4, ms=6)
ax.axhline(SP.GUARD_BAR, color=SERIES[1], lw=1.2, ls="--")
label_line(ax, ks[-1], raw[-1], "raw P4 rate", color=c_lo)
label_line(ax, ks[-1], ex[-1], "corrected excess", color=c_hi)
# BELOW the line here, not above it: seed 0's raw rate is 0.1087 and its marker
# sits exactly where the word "bar" would go.
ax.text(ks[0] - 0.08, SP.GUARD_BAR - 0.004, "P4 bar 0.10", color=SERIES[1],
        fontsize=9, va="top", ha="left")
ax.set_xticks(ks)
ax.set_xlabel("stage Y body warmed from (seed)")
ax.set_ylabel("guard rate")
ax.set_ylim(0, max(raw) + 0.03)
style_axes(ax)
save(fig, "guard_excess_seeds.png")

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
arm_x = [3.2 + 0.45 * i for i in range(len(seeds))]
arm_y = [seeds[k]["warm L1"]["final_train_nll"] for k in sorted(seeds)]
ax.plot(arm_x, arm_y, "o", color=SERIES[0], ms=8)
for x, y, k in zip(arm_x, arm_y, sorted(seeds)):
    ax.annotate("%.4f" % y, (x, y), textcoords="offset points",
                xytext=(0, 10), ha="center", color=SERIES[0], fontsize=9)
# BELOW the group and centred on it, not to its left: left of the group is where
# stage Y's seed 2 sits, and with three arms the label landed on that point.
ax.annotate(r"the $\lambda=1$ arms",
            (sum(arm_x) / len(arm_x), min(arm_y)),
            textcoords="offset points", xytext=(0, -18), va="top",
            ha="center", color=SERIES[0], fontsize=9.5)
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
# 7-9. the trajectory: 52461470, finding 84
# ---------------------------------------------------------------------------
# One lambda=1 arm re-run with the identical recipe and snapshotted ten times.
# TRAJ is None when that job has not landed, and every page below is skipped
# rather than faked -- the same rule the seed pages follow.
_tj = os.path.join(ROOT, "runs/stagepair_trajectory_2026-11-15",
                   "trajectory_summary.json")
TRAJ = json.load(open(_tj))["per_step"] if os.path.exists(_tj) else None

if TRAJ:
    ts = [r["step"] for r in TRAJ]
    tc = [r["pair_auroc_corrupt"] for r in TRAJ]
    tp = [r["pair_auroc_placebo"] for r in TRAJ]
    tf = [r["total_flips"] for r in TRAJ]
    tg = [r["guard_rate"] for r in TRAJ]
    te = [r["excess"] for r in TRAJ]

    # ---- 7. when the predicate arrives -----------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(ts, tc, "o-", color=SERIES[0], lw=1.8, ms=5)
    ax.plot(ts, tp, "o-", color=SERIES[1], lw=1.8, ms=5)
    ax.axhline(SP.PAIR_BAR, color=MUTED, lw=1.1, ls="--")
    ax.axhspan(*SP.PLACEBO_BAND, color=MUTED, alpha=0.12, lw=0)
    label_line(ax, ts[-1], tc[-1], "broken prompt", color=SERIES[0])
    label_line(ax, ts[-1], tp[-1], "placebo", color=SERIES[1])
    # Left edge, not right: the right is where the two curve labels go, and the
    # 0.65 line passes within 0.02 of the placebo band's top.
    # Mid-right and BELOW the line, matching traj_guard.png. At the left edge it
    # is unplaceable: on the line the dashes strike the glyph baselines, and 10
    # points above it the blue curve -- which goes from 0.456 to 0.748 between
    # step 0 and step 125 -- runs straight through the word "P1". Both of those
    # were caught by a PNG pass, in that order.
    ax.annotate("P1 bar 0.65", (1750, SP.PAIR_BAR), textcoords="offset points",
                xytext=(0, -7), ha="center", va="top", color=MUTED, fontsize=9)
    ax.annotate("P2 band", (ts[0], SP.PLACEBO_BAND[0]),
                textcoords="offset points", xytext=(4, 4), ha="left",
                color=MUTED, fontsize=9)
    ax.set_xlabel("fine-tuning steps from stage Y")
    ax.set_ylabel("matched-pair AUROC")
    ax.set_ylim(0.38, 0.95)
    style_axes(ax)
    save(fig, "traj_auroc.png")

    # ---- 8. and when the churn arrives -----------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(ts, tf, "o-", color=SERIES[0], lw=1.8, ms=5)
    # The crossing step, marked on the figure the claim is about rather than
    # stated only in the prose. `cross` is recomputed here from the same rows.
    _x = next(r["step"] for r in TRAJ if r["pair_auroc_corrupt"] >= SP.PAIR_BAR)
    _y = next(r["total_flips"] for r in TRAJ if r["step"] == _x)
    ax.plot([_x], [_y], "o", color=SERIES[1], ms=11, mfc="none", mew=2)
    ax.annotate("the predicate is already\nin place here", (_x, _y),
                textcoords="offset points", xytext=(26, 14), ha="left",
                color=SERIES[1], fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.7,
                                shrinkA=2, shrinkB=10))
    ax.set_xlabel("fine-tuning steps from stage Y")
    ax.set_ylabel("cases of 1,000 where a one-token edit\n"
                  "changed whether $v$ is emitted")
    ax.set_ylim(0, max(tf) + 25)
    style_axes(ax)
    save(fig, "traj_flips.png")

    # ---- 9. the guard saturates, the targeting does not ------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(ts, tg, "o-", color=SERIES[0], lw=1.8, ms=5)
    ax.plot(ts, te, "o-", color=SERIES[1], lw=1.8, ms=5)
    ax.axhline(SP.GUARD_BAR, color=MUTED, lw=1.1, ls="--")
    label_line(ax, ts[-1], tg[-1], "raw guard", color=SERIES[0])
    label_line(ax, ts[-1], te[-1], "targeted excess", color=SERIES[1])
    # Mid-right and below the line, which is the only clear band on this figure:
    # the raw-guard curve crosses 0.10 at about step 400, so at the left edge the
    # label is struck through by the rise (the first PNG pass caught exactly
    # that), and above the line from step 500 on the blue curve is sitting there.
    ax.annotate("P4 bar 0.10", (1750, SP.GUARD_BAR),
                textcoords="offset points", xytext=(0, -7), ha="center",
                va="top", color=MUTED, fontsize=9)
    ax.set_xlabel("fine-tuning steps from stage Y")
    ax.set_ylabel("rate, conditional on the clean prompt emitting $v$")
    ax.set_ylim(0, max(tg) + 0.02)
    style_axes(ax)
    save(fig, "traj_guard.png")


# ---------------------------------------------------------------------------
# 10-12. the same trajectory on all three bodies: 52466922_[1-2], finding 85
# ---------------------------------------------------------------------------
# The identical probe at TRAJ_SEED=1 and 2, with its predictions committed to git
# (`stagepair_trajectory.PREDICTION`) before it was submitted. Two of those
# predictions fail on seed 2 -- including the load-bearing one -- so these pages
# CORRECT finding 84 rather than extend it, and the figures are built to show the
# disagreement rather than the average.
TRAJ_ALL = {}
for _s in (0, 1, 2):
    _f = os.path.join(ROOT, "runs/stagepair_trajectory_2026-11-15"
                      + ("" if _s == 0 else "_s%d" % _s),
                      "trajectory_summary.json")
    if os.path.isfile(_f):
        TRAJ_ALL[_s] = json.load(open(_f))["per_step"]
print("trajectory seeds on disk:", sorted(TRAJ_ALL))

# The measured run-to-run noise on a guard RATE, from finding 84: the same code,
# recipe and cases moved seed 0's raw guard 0.1087 -> 0.1399. It is drawn on the
# crossing figure because seed 2's crossing guard sits 0.0056 above the bar, which
# is a fifth of that noise -- a figure that showed the point without the bar would
# be claiming a resolution the instrument does not have.
GUARD_NOISE = 0.03


def _crossing(rows):
    """The first snapshot where corrupt clears P1, with its whole row."""
    return next((r for r in rows if r["pair_auroc_corrupt"] >= SP.PAIR_BAR), None)


if len(TRAJ_ALL) == 3:
    _seeds = sorted(TRAJ_ALL)
    _cross = {s: _crossing(TRAJ_ALL[s]) for s in _seeds}
    assert all(c is not None for c in _cross.values()), "a seed never crossed P1"

    # ---- 10. the predicate's arrival, faceted --------------------------------
    # Six curves on one axis would need a legend and the three corrupt endpoints
    # land within 0.007 of each other (0.8720 / 0.8680 / 0.8650), so right-end
    # labels would overlap illegibly. Small multiples instead, per the deck rule.
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharex=True, sharey=True)
    for ax, s in zip(axes, _seeds):
        rows = TRAJ_ALL[s]
        st = [r["step"] for r in rows]
        ax.axhspan(*SP.PLACEBO_BAND, color=MUTED, alpha=0.12, lw=0)
        ax.axhline(SP.PAIR_BAR, color=MUTED, lw=1.1, ls="--")
        ax.plot(st, [r["pair_auroc_corrupt"] for r in rows], "o-",
                color=SERIES[0], lw=1.8, ms=4)
        ax.plot(st, [r["pair_auroc_placebo"] for r in rows], "o-",
                color=SERIES[1], lw=1.8, ms=4)
        ax.plot([_cross[s]["step"]], [_cross[s]["pair_auroc_corrupt"]], "o",
                color=SERIES[0], ms=11, mfc="none", mew=2)
        ax.set_title("seed %d" % s, fontsize=10)
        ax.set_xlabel("steps from stage Y")
        style_axes(ax)
    # One label per colour, on the LEFT panel only: repeating it three times is
    # the clutter small multiples exist to avoid.
    axes[0].annotate("broken prompt", (3000, TRAJ_ALL[0][-1]["pair_auroc_corrupt"]),
                     textcoords="offset points", xytext=(-4, -12), ha="right",
                     color=SERIES[0], fontsize=9)
    axes[0].annotate("placebo", (3000, TRAJ_ALL[0][-1]["pair_auroc_placebo"]),
                     textcoords="offset points", xytext=(-4, 6), ha="right",
                     color=SERIES[1], fontsize=9)
    # The circle and the two reference marks have to be NAMED or the reader sees
    # an unexplained ring, an unexplained dashed line and an unexplained grey
    # band -- the first PNG pass on this figure had exactly those three holes.
    # The ring note goes in the left panel, where the gap between 0.65 and the
    # curve is widest; the bar and band notes go in the right panel, whose
    # mid-height is empty in all three because the curves sit at 0.86 and 0.44.
    axes[0].annotate("first snapshot\npast the bar", (125, 0.7480),
                     textcoords="offset points", xytext=(34, -26), ha="left",
                     color=SERIES[0], fontsize=9,
                     arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.7,
                                     shrinkA=2, shrinkB=9))
    axes[2].annotate("P1 bar 0.65", (1600, SP.PAIR_BAR),
                     textcoords="offset points", xytext=(0, -7), ha="center",
                     va="top", color=MUTED, fontsize=9)
    axes[2].annotate("P2 band", (1600, SP.PLACEBO_BAND[1]),
                     textcoords="offset points", xytext=(0, -7), ha="center",
                     va="top", color=MUTED, fontsize=9)
    axes[0].set_ylabel("matched-pair AUROC")
    axes[0].set_ylim(0.38, 0.95)
    save(fig, "traj_seeds_auroc.png")

    # ---- 11. the guard where the predicate arrives ---------------------------
    # THE finding's figure: three points, one bar, and the instrument's own noise.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    gx = [_cross[s]["guard_rate"] for s in _seeds]
    cols = series_colors(len(_seeds), highlight=_seeds.index(2))
    ax.axhline(SP.GUARD_BAR, color=MUTED, lw=1.2, ls="--")
    # Asymmetric on purpose: a rate cannot go below zero, so the lower whisker is
    # clipped at 0 by construction rather than hidden by the axis limit, which is
    # what the first draft did to seed 0 (0.0140 - 0.03 = -0.016).
    ax.errorbar(range(len(_seeds)), gx,
                yerr=[[min(g, GUARD_NOISE) for g in gx],
                      [GUARD_NOISE] * len(gx)],
                fmt="o", ms=9, lw=0, elinewidth=1.4, capsize=5, ecolor=MUTED,
                zorder=3)
    for i, (s, g) in enumerate(zip(_seeds, gx)):
        ax.plot([i], [g], "o", color=cols[i], ms=9, zorder=4)
        ax.annotate("%.4f" % g, (i, g), textcoords="offset points",
                    xytext=(12, 0), ha="left", va="center", color=cols[i],
                    fontsize=9.5)
    # Left of the points and above the line: the three markers occupy x = 0, 1, 2
    # and the seed-2 error bar reaches 0.136, so the mid-right placement used on
    # the single-seed guard figure would sit inside that bar here.
    ax.annotate("P4 bar 0.10", (-0.42, SP.GUARD_BAR), textcoords="offset points",
                xytext=(0, 5), ha="left", va="bottom", color=MUTED, fontsize=9)
    # An unexplained whisker is worse than no whisker. Placed in headroom made
    # for it at the top: at (0.08, 0.062) -- the first attempt -- seed 1's upper
    # whisker ran straight through the word "guard" on the second line, and the
    # bottom-right alternative collides with seed 1's own value label. A PNG pass
    # caught the first; the second was ruled out by arithmetic before rendering.
    _top = max(gx) + GUARD_NOISE + 0.05
    # ONE line, not two: wrapped onto two lines in the headroom, the 0.175
    # gridline ran between them and clipped the first line's descenders. The
    # second PNG pass caught that.
    ax.annotate("whiskers: $\\pm 0.03$, this instrument's own run-to-run noise "
                "on a guard rate",
                (-0.42, _top - 0.004), ha="left", va="top", color=MUTED,
                fontsize=8.5)
    ax.set_xticks(range(len(_seeds)))
    ax.set_xticklabels(["seed %d\n(step %d)" % (s, _cross[s]["step"])
                        for s in _seeds])
    ax.set_xlim(-0.5, len(_seeds) - 0.4)
    ax.set_ylim(0, _top)
    ax.set_ylabel("raw guard rate at the snapshot\nwhere the predicate arrives")
    style_axes(ax)
    save(fig, "traj_seeds_cross.png")

    # ---- 12. how much of the churn is already paid ---------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    shr = [100.0 * _cross[s]["total_flips"] / TRAJ_ALL[s][-1]["total_flips"]
           for s in _seeds]
    for i, (s, v) in enumerate(zip(_seeds, shr)):
        ax.plot([i], [v], "o", color=cols[i], ms=9)
        ax.annotate("%d%%  (%d of %d)" % (round(v), _cross[s]["total_flips"],
                                          TRAJ_ALL[s][-1]["total_flips"]),
                    (i, v), textcoords="offset points", xytext=(12, 0),
                    ha="left", va="center", color=cols[i], fontsize=9.5)
    # The registered cap, drawn so the miss is visible rather than asserted.
    ax.axhline(35, color=MUTED, lw=1.2, ls="--")
    ax.annotate("my registered cap, 35%", (-0.42, 35),
                textcoords="offset points", xytext=(0, 5), ha="left",
                va="bottom", color=MUTED, fontsize=9)
    ax.set_xticks(range(len(_seeds)))
    ax.set_xticklabels(["seed %d" % s for s in _seeds])
    ax.set_xlim(-0.5, len(_seeds) - 0.25)
    ax.set_ylim(0, max(shr) + 12)
    ax.set_ylabel("flips already present at the crossing,\n"
                  "as % of that body's own final count")
    style_axes(ax)
    save(fig, "traj_seeds_flips.png")
else:
    print("SKIPPED traj_seeds_*.png -- trajectory seeds on disk: %s"
          % sorted(TRAJ_ALL))


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
    if len(seeds) == 3:
        # The deck's seed pages, asserted against stagepair_score's own
        # held_against constants -- which this probe was never shown.
        for i, k in enumerate(sorted(seeds)):
            assert abs(seeds[k]["Y"]["within_stratum_auroc"]
                       - SP.Z_WITHIN_STRATUM[i]) < 5e-5, k
            assert abs(seeds[k]["Y"]["pooled_auroc"]
                       - SP.Y_AUROC_PUBLISHED[i]) < 5e-4, k
        print("  ok  stage Y within-stratum == Z_WITHIN_STRATUM on 3/3 seeds,")
        print("      pooled within 0.0004 of Y_AUROC_PUBLISHED on 3/3")
        cs = [seeds[k]["warm L1"]["pair_auroc_corrupt"] for k in sorted(seeds)]
        ps = [seeds[k]["warm L1"]["pair_auroc_placebo"] for k in sorted(seeds)]
        assert all(c >= SP.PAIR_BAR for c in cs), cs
        assert all(SP.PLACEBO_BAND[0] < p < SP.PLACEBO_BAND[1] for p in ps), ps
        assert all(seeds[k]["warm L1"]["pooled_auroc"] >= SP.POOLED_BAR
                   for k in seeds)
        assert round(max(cs) - min(cs), 4) == 0.0090, cs
        ys = [seeds[k]["Y"]["pair_auroc_corrupt"] for k in sorted(seeds)]
        assert round(max(ys) - min(ys), 4) == 0.1310, ys
        assert all(seeds[k]["warm L1"]["guard_rate"] > SP.GUARD_BAR
                   for k in seeds), "the deck says all three cross raw"
        print("  ok  3/3 seeds clear P1 and P3 and sit inside P2; spreads"
              " 0.0090 (arm) vs 0.1310 (baseline)")
    if len(MCS) == 3:
        ks3 = sorted(MCS)
        assert [MCS[k]["total_flips"] for k in ks3] == [142, 175, 177]
        assert [round(MCS[k]["excess"], 4) for k in ks3] == [0.0415, 0.1295,
                                                             0.0606]
        # the other tail, the direction check: the placebo turns v back ON more
        # often than the break does, on every seed, each at p < 0.01
        for k in ks3:
            assert MCS[k]["on_placebo_only"] > MCS[k]["on_corrupt_only"], k
            assert MCS[k]["p_on"] < 0.01, (k, MCS[k]["p_on"])
        assert [MCS[k]["on_corrupt_only"] for k in ks3] == [15, 10, 22]
        assert [MCS[k]["on_placebo_only"] for k in ks3] == [38, 35, 47]
        fr = [MCS[k]["excess"] / GSEED[k]["warm L1"]["corrupt"]
              ["rate_off_given_on"] for k in ks3]
        assert [round(f, 2) for f in fr] == [0.38, 0.62, 0.37], fr
        print("  ok  the guard pages: excess 0.0415/0.1295/0.0606, targeted"
              " share 38/62/37%, flips 142/175/177,")
        print("      and the other tail favours the placebo on 3/3 at p < 0.01")
    if TRAJ:
        d = {r["step"]: r for r in TRAJ}
        # THE STRONGEST FORM OF THE STEP-0 CHECK: not "0.4560 as the deck says"
        # but "the same number the SEPARATE probe job measured". Step 0 is stage
        # Y's weights meeting stage Y's case list in a different process, so
        # anything other than equality means the two pipelines disagree.
        assert abs(d[0]["pair_auroc_corrupt"]
                   - probe["Y"]["pair_auroc_corrupt"]) < 5e-5
        assert abs(d[0]["pair_auroc_placebo"]
                   - probe["Y"]["pair_auroc_placebo"]) < 5e-5
        assert d[0]["guard_denom"] == gspec["Y"]["corrupt"]["denom_clean_on"]
        assert d[0]["total_flips"] == MC["Y"]["total_flips"] == 4
        print("  ok  step 0 of 52461470 == stage Y's own row from 52452867 and")
        print("      52455457, across processes: 0.4560 / 0.5140 / 4 flips")

        x = next(r["step"] for r in TRAJ
                 if r["pair_auroc_corrupt"] >= SP.PAIR_BAR)
        assert x == 125, x
        eq("crossing corrupt", d[125]["pair_auroc_corrupt"], 0.7480)
        eq("crossing placebo", d[125]["pair_auroc_placebo"], 0.4670)
        eq("crossing guard", d[125]["guard_rate"], 0.0140, tol=5e-4)
        assert d[125]["total_flips"] == 29, d[125]["total_flips"]
        assert d[125]["guard_denom"] == 572, d[125]["guard_denom"]
        # the deck's central claim: the predicate at a fifth of the churn, with
        # the guard BELOW P4's bar -- so early stopping trades P4 away
        assert d[125]["total_flips"] < 0.25 * d[3000]["total_flips"]
        assert d[125]["guard_rate"] < SP.GUARD_BAR < d[3000]["guard_rate"]
        print("  ok  the crossing is step 125: P1 cleared at 29 flips of the"
              " final 160, guard 0.0140 < 0.10")

        eq("final corrupt", d[3000]["pair_auroc_corrupt"], 0.8720)
        eq("final placebo", d[3000]["pair_auroc_placebo"], 0.4370)
        eq("final guard", d[3000]["guard_rate"], 0.1399, tol=5e-4)
        assert (d[3000]["total_flips"], d[3000]["guard_denom"]) == (160, 529)
        # The rerun caveat the deck states: the AUROCs reproduce to 0.001 and the
        # EMISSION COUNTS do not -- emission is an argmax, so a logit shift too
        # small to move a ranking still flips a discrete decision.
        # 0.0011 and not 0.0010: both gaps ARE 0.0010 at four decimals, and
        # 0.8720 - 0.8710 comes out as 1.0000000000000009e-3 in binary float.
        assert abs(d[3000]["pair_auroc_corrupt"]
                   - probe["warm L1"]["pair_auroc_corrupt"]) <= 0.0011
        assert abs(d[3000]["pair_auroc_placebo"]
                   - probe["warm L1"]["pair_auroc_placebo"]) <= 0.0011
        assert abs(d[3000]["guard_rate"]
                   - gspec["warm L1"]["corrupt"]["rate_off_given_on"]) > 0.02
        print("  ok  the rerun: AUROCs within 0.001 of 52452867, guard off by"
              " 0.0312 and the denominator by 23")

        assert all(SP.PLACEBO_BAND[0] < r["pair_auroc_placebo"]
                   < SP.PLACEBO_BAND[1] for r in TRAJ), "P2 held at every step"
        rho = stats.spearmanr([r["guard_rate"] for r in TRAJ],
                              [r["total_flips"] for r in TRAJ]).statistic
        assert round(rho, 3) == 0.879, rho
        # the guard plateaus after step 500 while the targeted part keeps rising
        assert all(abs(r["guard_rate"] - d[500]["guard_rate"]) < 0.025
                   for r in TRAJ if r["step"] >= 500)
        assert d[2500]["excess"] > 1.4 * d[500]["excess"]
        assert round(max(r["excess"] for r in TRAJ), 4) == 0.0730
        # The "share targeted" column, every row of it. Step 0 is excluded on the
        # page because its raw rate is 0.0015 -- one case -- so the ratio is a
        # ratio of one, and the deck says that rather than quietly dropping it.
        share = [round(100 * r["excess"] / r["guard_rate"])
                 for r in TRAJ if r["step"] > 0]
        assert share == [25, 39, 39, 25, 36, 46, 49, 54, 43], share
        # and the non-monotonicity the page admits to, asserted rather than
        # glossed: 750 dips below 250, and 3,000 falls back from 2,500
        assert share[3] < share[1] and share[-1] < share[-2]
        print("  ok  P2 held at all 10 steps; Spearman(guard, flips) +0.879;")
        print("      guard flat after step 500 while the excess rises"
              " 0.0471 -> 0.0730")
    else:
        print("  NOTE 52461470 absent; the deck must not claim a trajectory")

    if len(TRAJ_ALL) == 3:
        ss = sorted(TRAJ_ALL)
        cr = {s: _crossing(TRAJ_ALL[s]) for s in ss}
        fin = {s: TRAJ_ALL[s][-1] for s in ss}

        # Same cross-process check as seed 0's, now on all three: step 0 is that
        # seed's stage Y weights meeting that seed's case list, measured by a
        # DIFFERENT job (52452867 / 52456376), so anything but equality is a
        # pipeline disagreement. This is what localizes the guard noise to
        # training rather than to the eval, and it now holds 3/3.
        for s in ss:
            z = TRAJ_ALL[s][0]
            assert abs(z["pair_auroc_corrupt"]
                       - seeds[s]["Y"]["pair_auroc_corrupt"]) < 5e-5, s
            assert abs(z["pair_auroc_placebo"]
                       - seeds[s]["Y"]["pair_auroc_placebo"]) < 5e-5, s
            assert abs(z["guard_rate"]
                       - seeds[s]["Y"]["guard_rate"]) < 5e-6, s
        assert [TRAJ_ALL[s][0]["total_flips"] for s in ss] == [4, 1, 7]
        print("  ok  step 0 == that seed's own stage Y row on 3/3, across"
              " processes (flips 4 / 1 / 7)")

        # P1 clears at the FIRST trained snapshot on all three bodies.
        assert [cr[s]["step"] for s in ss] == [125, 125, 125]
        assert [round(cr[s]["pair_auroc_corrupt"], 4)
                for s in ss] == [0.7480, 0.8070, 0.8520]
        print("  ok  corrupt clears 0.65 at step 125 on 3/3"
              " (0.7480 / 0.8070 / 0.8520)")

        # THE CORRECTION TO FINDING 84, AND THE WHOLE REASON THESE PAGES EXIST.
        # The registered prediction was that the raw guard at the crossing is
        # below P4's bar on both new seeds. It is not: seed 2 is at 0.1056.
        gx = [round(cr[s]["guard_rate"], 4) for s in ss]
        assert gx == [0.0140, 0.0345, 0.1056], gx
        assert [cr[s]["guard_denom"] for s in ss] == [572, 580, 483]
        assert sum(g >= SP.GUARD_BAR for g in gx) == 1
        assert gx[2] >= SP.GUARD_BAR > gx[1] > gx[0]
        # and the honest qualifier the page must carry: seed 2 clears the bar by
        # 0.0056, which is a fifth of the instrument's own run-to-run noise
        assert 0 < gx[2] - SP.GUARD_BAR < GUARD_NOISE / 4
        print("  ok  crossing guard 0.0140 / 0.0345 / 0.1056 -- 1 of 3 seeds is"
              " already at P4's bar,")
        print("      and it clears it by 0.0056 against ~0.03 of run-to-run"
              " noise")

        # the registered 35% cap on the churn already paid at the crossing
        shr = [round(100.0 * cr[s]["total_flips"] / fin[s]["total_flips"])
               for s in ss]
        assert shr == [18, 31, 77], shr
        assert [cr[s]["total_flips"] for s in ss] == [29, 52, 127]
        assert [fin[s]["total_flips"] for s in ss] == [160, 168, 166]
        assert shr[0] < 35 and shr[1] < 35 and shr[2] > 35
        print("  ok  churn already paid at the crossing: 18 / 31 / 77% --"
              " the 35% cap holds on 2 of 3")

        # P2 on every snapshot of every seed: 30 of 30, the one prediction that
        # has now hit on all three bodies.
        assert all(SP.PLACEBO_BAND[0] < r["pair_auroc_placebo"]
                   < SP.PLACEBO_BAND[1]
                   for s in ss for r in TRAJ_ALL[s]), "P2 left the band"
        rhos = [round(stats.spearmanr([r["guard_rate"] for r in TRAJ_ALL[s]],
                                      [r["total_flips"] for r in TRAJ_ALL[s]]
                                      ).statistic, 3) for s in ss]
        assert rhos == [0.879, 0.806, 0.881], rhos
        assert all(r >= 0.7 for r in rhos)
        print("  ok  P2 in band at 30 of 30 snapshots; Spearman(guard, flips)"
              " +0.879 / +0.806 / +0.881")

        # The reproduction caveat, now with three bodies instead of one: the
        # AUROCs come back to 0.006 or better and the guard moves by up to 0.031.
        # Seed 0's 0.0010 was the best of the three, not the typical case -- the
        # deck says 0.006 on 3/3 and attributes 0.001 to seed 0 only.
        da = [abs(fin[s]["pair_auroc_corrupt"]
                  - seeds[s]["warm L1"]["pair_auroc_corrupt"]) for s in ss]
        dg = [abs(fin[s]["guard_rate"]
                  - GSEED[s]["warm L1"]["corrupt"]["rate_off_given_on"])
              if s in GSEED else None for s in ss]
        assert [round(x, 4) for x in da] == [0.0010, 0.0060, 0.0060], da
        assert all(x <= 0.0061 for x in da)
        if all(x is not None for x in dg):
            assert [round(x, 4) for x in dg] == [0.0312, 0.0275, 0.0046], dg
            assert max(dg) > 4 * max(da)
        print("  ok  rerun: AUROCs within 0.006 on 3/3 (0.0010 / 0.0060 /"
              " 0.0060), guard off by up to 0.031")

        # THE LIMIT OF THE GRID, stated because the figure cannot show it: 125 is
        # the first TRAINED snapshot, so every crossing is an upper bound. Seed 2
        # is at 0.8520 there against a final 0.8650 -- 98% of the way -- which is
        # the clearest sign the real crossing is earlier still.
        assert min(r["step"] for r in TRAJ_ALL[0] if r["step"] > 0) == 125
        assert (cr[2]["pair_auroc_corrupt"] / fin[2]["pair_auroc_corrupt"]
                > 0.98)
        print("  ok  125 is the first trained snapshot, so each crossing is an"
              " UPPER BOUND (seed 2 is at 98%")
        print("      of its final AUROC there)")

        # the "share of final" column on the upper-bound page
        shq = [round(100.0 * cr[s]["pair_auroc_corrupt"]
                     / fin[s]["pair_auroc_corrupt"]) for s in ss]
        assert shq == [86, 93, 98], shq

        # the conjecture page: three columns that rank the seeds identically.
        # Asserted as an ORDERING and never as a cause -- n = 3 and ||s|| is
        # perfectly confounded with the stage Y train nll along this axis.
        sn = [round(TRAJ_ALL[s][0]["det_scale_norm"], 3) for s in ss]
        yn = [round(seeds[s]["Y"]["mean_train_nll_last250"], 4) for s in ss]
        assert sn == [7.053, 7.774, 9.137], sn
        assert yn == [1.5610, 1.5933, 1.6577], yn
        assert sn == sorted(sn) and yn == sorted(yn) and gx == sorted(gx)
        print("  ok  the conjecture page: ||s|| 7.053/7.774/9.137 and stage Y"
              " nll 1.5610/1.5933/1.6577")
        print("      rank the seeds the same way the crossing guard does"
              " (ordering only, n = 3)")

        # the P2 page's per-seed range, every cell of it
        pr = [(round(min(r["pair_auroc_placebo"] for r in TRAJ_ALL[s]), 4),
               round(max(r["pair_auroc_placebo"] for r in TRAJ_ALL[s]), 4))
              for s in ss]
        assert pr == [(0.4350, 0.5140), (0.4410, 0.4730), (0.4390, 0.5060)], pr
        # the page claims the HIGHEST value in each row is step 0, i.e. stage Y
        for s in ss:
            assert TRAJ_ALL[s][0]["pair_auroc_placebo"] == max(
                r["pair_auroc_placebo"] for r in TRAJ_ALL[s]), s
        print("  ok  placebo ranges 0.4350-0.5140 / 0.4410-0.4730 /"
              " 0.4390-0.5060, max at step 0 on 3/3")
    elif TRAJ:
        print("  NOTE trajectory seeds 1-2 absent; the deck must not claim a"
              " replication")

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
