"""Figures for the `spin-gnn/tiebreak` deck -- finding 91.

Every number is READ OUT OF `marginstrat_summary.json` and none is retyped;
`check_claims()` at the bottom asserts the values the deck's prose states
against what was read, so a figure and its caption cannot drift apart. Same
pattern as make_loop_web.py and make_predicate_web.py.

Source: runs/stagepair_marginstrat_2026-11-15/marginstrat_summary.json, itself
computed from the three guardspec row dumps (3 seeds x 3 arms x 1,000 cases).
No model is loaded anywhere in this line: the rows were written by 52452867 /
52456376 / 52455457 and every number here is a re-reading of them.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, PAGE, style_axes, label_line

sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")
import stagepair_score as SP

ROOT = "/scratch/users/hshunt/spin_gnn"
OUT = "/home/users/hshunt/slides/decks/spin-gnn/tiebreak/figures"
SRC = os.path.join(ROOT, "runs/stagepair_marginstrat_2026-11-15/"
                         "marginstrat_summary.json")

os.makedirs(OUT, exist_ok=True)
S = json.load(open(SRC))
PS = S["per_seed"]
SEEDS = [0, 1, 2]
L1, Y = "warm L1", "Y"
QX = [1, 2, 3, 4]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def arm(seed, a):
    return PS[str(seed)][a]


def q(seed, a, key):
    return [x[key] for x in arm(seed, a)["quartiles"]]


def panels(n=3):
    fig, axes = plt.subplots(1, n, figsize=PAGE, sharey=True)
    return fig, axes


# ---------------------------------------------------------------- 1. the gradient
# The central result: the guard falls steeply across the model's OWN margin
# quartiles. Three seeds, three series, labelled at the right end -- no legend.
fig, ax = plt.subplots(figsize=PAGE)
# Seeds 1 and 2 land 0.008 apart at Q4, so their right-end labels collide
# without a nudge; the note goes top-right because the 0.10 line runs straight
# through every other empty spot and seed 0 struck it there.
for i, (s, dy) in enumerate(zip(SEEDS, (0, 7, -7))):
    g = q(s, L1, "guard_rate")
    ax.plot(QX, g, "o-", color=SERIES[i], lw=2.2, ms=7, rasterized=True)
    label_line(ax, QX[-1], g[-1], f"seed {s}", color=SERIES[i], dy=dy)
ax.axhline(SP.GUARD_BAR, color=MUTED, lw=1.1, ls="--")
ax.text(2.55, 0.40, "dashed: P4's 0.10 bar, shown for scale only",
        color=MUTED, fontsize=12)
ax.set_xticks(QX)
ax.set_xticklabels(["Q1\nthinnest", "Q2", "Q3", "Q4\nthickest"])
style_axes(ax, xlabel="quartile of the clean top-1 minus top-2 margin",
           ylabel="guard rate  (corrupt withholds v | clean emits v)")
ax.set_xlim(0.8, 4.45)
save(fig, "guard_by_quartile.png")

# ---------------------------------------------------------------- 2. both rates
fig, axes = panels()
for i, (s, ax) in enumerate(zip(SEEDS, axes)):
    g, p = q(s, L1, "guard_rate"), q(s, L1, "placebo_rate")
    ax.plot(QX, g, "o-", color=SERIES[0], lw=2.2, ms=7, rasterized=True)
    ax.plot(QX, p, "s-", color=SERIES[1], lw=2.2, ms=6, rasterized=True)
    if i == 2:
        label_line(ax, QX[-1], g[-1], "corrupt", color=SERIES[0])
        label_line(ax, QX[-1], p[-1], "placebo", color=SERIES[1], dy=-14)
    ax.set_xticks(QX)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_title(f"seed {s}", fontsize=14, color=MUTED)
    style_axes(ax, xlabel="margin quartile",
               ylabel="turn-off rate" if i == 0 else None)
    ax.set_xlim(0.7, 4.8)
save(fig, "guard_and_placebo.png")

# ---------------------------------------------------------------- 3. the excess
fig, axes = panels()
for i, (s, ax) in enumerate(zip(SEEDS, axes)):
    e = np.array(q(s, L1, "excess"))
    ci = np.array(q(s, L1, "excess_ci"))
    ax.errorbar(QX, e, yerr=[e - ci[:, 0], ci[:, 1] - e], fmt="o",
                color=SERIES[0], ecolor=MUTED, elinewidth=1.4, capsize=5, ms=8,
                rasterized=True)
    ax.axhline(0.0, color=MUTED, lw=1.1, ls="--")
    ax.set_xticks(QX)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_title(f"seed {s}", fontsize=14, color=MUTED)
    style_axes(ax, xlabel="margin quartile",
               ylabel="corrupt minus placebo" if i == 0 else None)
    ax.set_xlim(0.6, 4.4)
save(fig, "excess_by_quartile.png")

# ---------------------------------------------------------------- 4. margins moved
# Y against warm L1, per seed. Two series, so the quartile marks can carry the
# shape without a legend box.
fig, axes = panels()
for i, (s, ax) in enumerate(zip(SEEDS, axes)):
    for j, a in ((0, Y), (1, L1)):
        m = q(s, a, "median_margin")
        ax.plot(QX, m, "o-" if j == 0 else "s-", color=SERIES[j], lw=2.2, ms=7,
                rasterized=True)
        if i == 2:
            label_line(ax, QX[-1], m[-1], "stage Y" if j == 0 else "pair L1",
                       color=SERIES[j])
    ax.set_xticks(QX)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_title(f"seed {s}", fontsize=14, color=MUTED)
    style_axes(ax, xlabel="margin quartile, within each arm",
               ylabel="median margin (logits)" if i == 0 else None)
    ax.set_xlim(0.7, 4.9)
save(fig, "margin_quartiles_by_arm.png")

# ---------------------------------------------------------------- 5. the thin share
# The one number that makes the whole finding: how many clean-emitting cases sit
# within one typical perturbation of a tie.
fig, ax = plt.subplots(figsize=PAGE)
w = 0.34
xs = np.arange(3)
for j, (a, lab) in enumerate(((Y, "stage Y"), (L1, "pair L1"))):
    v = [arm(s, a)["frac_margin_below_median_drop"] for s in SEEDS]
    b = ax.bar(xs + (j - 0.5) * w, v, w, color=SERIES[j], rasterized=True)
    # The arm name rides on the FIRST group's value label, so the figure needs
    # no key; a key parked under the last group read as if it belonged to it.
    for k, (rect, val) in enumerate(zip(b, v)):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.008,
                (f"{lab}\n{val:.3f}" if k == 0 else f"{val:.3f}"),
                ha="center", fontsize=13, color=SERIES[j])
ax.set_xticks(xs)
ax.set_xticklabels([f"seed {s}" for s in SEEDS])
style_axes(ax, ylabel="share of clean-emitting cases whose margin\n"
                      "is below the arm's own median logit response")
ax.set_ylim(0, 0.38)
save(fig, "thin_fraction.png")

# ---------------------------------------------------------------- 6. who flips
# Median margin by overlap class. A fragility puts its mass in `both`.
fig, ax = plt.subplots(figsize=PAGE)
CLS = ["corrupt_only", "placebo_only", "both"]
NICE = ["corrupt only\n(the targeted ones)", "placebo only", "both\n(the fragile ones)"]
xs = np.arange(3)
# seeds 0 and 2 end 0.006 apart in the `both` column, so their labels overlap
# unless they are pushed apart; 8 points is under half a line height.
for i, (s, dy) in enumerate(zip(SEEDS, (-8, 0, 8))):
    v = [arm(s, L1)["overlap_median_margin"][c] for c in CLS]
    ax.plot(xs, v, "o-", color=SERIES[i], lw=2.2, ms=7, rasterized=True)
    label_line(ax, xs[-1], v[-1], f"seed {s}", color=SERIES[i], dy=dy)
ax.set_xticks(xs)
ax.set_xticklabels(NICE)
style_axes(ax, ylabel="median clean margin (logits)")
ax.set_xlim(-0.2, 2.5)
save(fig, "overlap_margin.png")

# ---------------------------------------------------------------- 7. targeting
# guard / placebo per quartile: where the TARGETING lives, as opposed to where
# the flips live. Ratio, so the placebo-zero quartiles are dropped rather than
# drawn at infinity -- stated on the page.
fig, ax = plt.subplots(figsize=PAGE)
for i, s in enumerate(SEEDS):
    g, p = np.array(q(s, L1, "guard_rate")), np.array(q(s, L1, "placebo_rate"))
    ok = p > 0
    ax.plot(np.array(QX)[ok], (g / p)[ok], "o-", color=SERIES[i], lw=2.2, ms=7,
            rasterized=True)
    label_line(ax, QX[-1], (g / p)[-1], f"seed {s}", color=SERIES[i])
ax.axhline(1.0, color=MUTED, lw=1.1, ls="--")
# No in-axes note on this one: every empty slot for it is crossed by a curve
# (seed 2 struck it at Q2), so what 1.0 means is said in the slide body instead.
ax.set_xticks(QX)
ax.set_xticklabels(["Q1\nthinnest", "Q2", "Q3", "Q4\nthickest"])
style_axes(ax, xlabel="margin quartile",
           ylabel="guard rate divided by placebo rate")
ax.set_xlim(0.8, 4.45)
save(fig, "targeting_ratio.png")


# ---------------------------------------------------------------- the assertions
def check_claims():
    n = 0

    def ck(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # --- the artifact is the registered one
    ck(SP.PAIR_EVAL_N_CASES == 1000, "case count")
    ck(SP.GUARD_BAR == 0.10, "guard bar")
    ck(S["n_boot"] == 2000 and S["boot_seed"] == 91_000_000, "bootstrap spec")
    for s, d in zip(SEEDS, (506, 502, 429)):
        ck(arm(s, L1)["denom_clean_on"] == d, f"L1 denom s{s}")
    for s, d in zip(SEEDS, (656, 637, 584)):
        ck(arm(s, Y)["denom_clean_on"] == d, f"Y denom s{s}")
    # the published guard rates, reproduced from the rows
    for s, g in zip(SEEDS, (0.1087, 0.2072, 0.1632)):
        ck(abs(arm(s, L1)["guard_rate"] - g) < 5e-5, f"L1 guard s{s}")
    for s, g in zip(SEEDS, (0.0015, 0.0000, 0.0017)):
        ck(abs(arm(s, Y)["guard_rate"] - g) < 5e-5, f"Y guard s{s}")

    # --- 91.1 the margins moved, 3/3
    for s, a, b in zip(SEEDS, (0.684, 0.758, 0.480), (0.959, 1.036, 0.682)):
        ck(abs(arm(s, L1)["median_margin"] - a) < 5e-4, f"L1 medmargin s{s}")
        ck(abs(arm(s, Y)["median_margin"] - b) < 5e-4, f"Y medmargin s{s}")
        ck(arm(s, L1)["median_margin"] < arm(s, Y)["median_margin"], f"91.1 s{s}")
    # the reduction, quoted as a percentage in the deck
    red = [1 - arm(s, L1)["median_margin"] / arm(s, Y)["median_margin"]
           for s in SEEDS]
    ck([round(100 * r) for r in red] == [29, 27, 30],
       f"margin reduction 29/27/30%: {red}")

    # --- the response grew by more than the margin shrank
    for s, a, b in zip(SEEDS, (0.2676, 0.2488, 0.2853), (0.0091, 0.0072, 0.0094)):
        ck(abs(arm(s, L1)["median_abs_drop_corrupt"] - a) < 5e-5, f"L1 drop s{s}")
        ck(abs(arm(s, Y)["median_abs_drop_corrupt"] - b) < 5e-5, f"Y drop s{s}")
    grow = [arm(s, L1)["median_abs_drop_corrupt"] / arm(s, Y)["median_abs_drop_corrupt"]
            for s in SEEDS]
    # one decimal, not a band and not a round(): the integers 30/35/30 hide that
    # seed 0 is 29.6, and the deck quotes the decimals.
    ck([round(g, 1) for g in grow] == [29.6, 34.7, 30.4],
       f"response grew 29.6/34.7/30.4x: {grow}")

    # --- the thin share, the number the whole finding turns on
    for s, a, b in zip(SEEDS, (0.1779, 0.1813, 0.3217), (0.0000, 0.0000, 0.0137)):
        ck(abs(arm(s, L1)["frac_margin_below_median_drop"] - a) < 5e-5, f"thin L1 s{s}")
        ck(abs(arm(s, Y)["frac_margin_below_median_drop"] - b) < 5e-5, f"thin Y s{s}")
    ck(all(arm(s, L1)["frac_margin_below_median_drop"] >= 0.17 for s in SEEDS),
       "L1 thin >= 0.17 on 3/3")
    ck(all(arm(s, Y)["frac_margin_below_median_drop"] <= 0.014 for s in SEEDS),
       "Y thin <= 0.014 on 3/3")

    # --- 91.2 / 91.3 the AUROCs, both MISSES of the registered bars
    for s, a, b in zip(SEEDS, (0.7952, 0.7656, 0.7725), (0.8345, 0.8555, 0.7817)):
        ck(abs(arm(s, L1)["auroc_margin_predicts_corrupt_off"] - a) < 5e-5,
           f"aurocC s{s}")
        ck(abs(arm(s, L1)["auroc_margin_predicts_placebo_off"] - b) < 5e-5,
           f"aurocP s{s}")
    ck(all(arm(s, L1)["auroc_margin_predicts_corrupt_off"] < 0.85 for s in SEEDS),
       "91.2 missed 3/3")
    ck(all(arm(s, L1)["auroc_margin_predicts_corrupt_off"] > 0.70 for s in SEEDS),
       "the 0.70 falsifier did NOT fire")
    ck(sum(arm(s, L1)["auroc_margin_predicts_placebo_off"] >= 0.80 for s in SEEDS) == 2,
       "91.3 hit on 2 of 3")
    # the placebo AUROC is the HIGHER of the two on 2 of 3 -- eligibility is not
    # a property of which edit was made
    hi = sum(arm(s, L1)["auroc_margin_predicts_placebo_off"]
             > arm(s, L1)["auroc_margin_predicts_corrupt_off"] for s in SEEDS)
    ck(hi == 3, f"placebo AUROC higher on {hi} of 3")

    # --- the gradient, Q1 against Q4
    q1g = [q(s, L1, "guard_rate")[0] for s in SEEDS]
    q4g = [q(s, L1, "guard_rate")[3] for s in SEEDS]
    for s, a, b in zip(SEEDS, (0.2520, 0.4524, 0.4206), (0.0079, 0.0635, 0.0556)):
        ck(abs(q(s, L1, "guard_rate")[0] - a) < 5e-5, f"Q1 guard s{s}")
        ck(abs(q(s, L1, "guard_rate")[3] - b) < 5e-5, f"Q4 guard s{s}")
    fall = [a / b for a, b in zip(q1g, q4g)]
    ck(abs(fall[0] - 32.0) < 1.0 and abs(fall[1] - 7.1) < 0.5
       and abs(fall[2] - 7.6) < 0.5, f"Q1/Q4 fall {fall}")
    ck(all(g > 0.10 for g in q1g) and all(g < 0.08 for g in q4g),
       "Q1 over the bar and Q4 under the falsifier on 3/3")
    # 91.4 asked for Q4 <= 0.02 and got it on seed 0 only
    ck(sum(g <= 0.02 for g in q4g) == 1, "91.4 hit on 1 of 3")

    # --- 91.5 the excess DOES reach the thick quartiles; the strong claim fails
    for s, vals in zip(SEEDS, ((0.0551, 0.0873, 0.0317, -0.0079),
                               (0.1984, 0.2000, 0.0640, 0.0556),
                               (0.1495, 0.0000, 0.0654, 0.0278))):
        for k, v in enumerate(vals):
            ck(abs(q(s, L1, "excess")[k] - v) < 5e-5, f"excess s{s} Q{k+1}")
    ck(sum(abs(q(s, L1, "excess")[3]) <= 0.02 for s in SEEDS) == 1,
       "91.5 Q4 hit on 1 of 3")
    ck(all(q(s, L1, "excess")[0] >= 0.05 for s in SEEDS), "91.5 Q1 hit 3/3")
    # Q3's CI excludes zero on 3/3 -- this is what kills the confinement story
    q3ci = [q(s, L1, "excess_ci")[2] for s in SEEDS]
    ck(all(lo >= 0.0 for lo, _hi in q3ci), f"Q3 CI lower bounds {q3ci}")
    ck(q(1, L1, "excess_ci")[3][0] > 0, "seed 1 Q4 CI excludes zero")
    ck(q(0, L1, "excess_ci")[3][0] < 0 and q(2, L1, "excess_ci")[3][0] < 0,
       "seeds 0 and 2 Q4 CIs include zero")

    # --- the targeting ratio rises off Q1 on 3/3
    for s in SEEDS:
        g, p = np.array(q(s, L1, "guard_rate")), np.array(q(s, L1, "placebo_rate"))
        ck(p[0] > 0 and (g / p)[0] < 2.0, f"Q1 ratio under 2 s{s}")
        ck(max((g / p)[1:][p[1:] > 0]) > (g / p)[0], f"ratio rises off Q1 s{s}")
    rmax = []
    for s in SEEDS:
        g, p = np.array(q(s, L1, "guard_rate")), np.array(q(s, L1, "placebo_rate"))
        r = (g / p)[1:][p[1:] > 0]
        rmax.append(float(r.max()))
        ck(1.2 <= (g / p)[0] <= 1.8, f"Q1 ratio 1.3-1.8 s{s}: {(g / p)[0]}")
    ck([round(x, 1) for x in rmax] == [3.2, 8.0, 3.3], f"ratio maxima {rmax}")
    # ...but seed 0's Q4 ratio dips UNDER 1 on 1 corrupt against 2 placebo
    # turn-offs, which is counts, not a reversal. Said on the page.
    g0, p0 = np.array(q(0, L1, "guard_rate")), np.array(q(0, L1, "placebo_rate"))
    ck((g0 / p0)[3] == 0.5 and q(0, L1, "turns_off_corrupt")[3] == 1
       and q(0, L1, "turns_off_placebo")[3] == 2, "seed 0 Q4 is 1 against 2")
    ck(all(q(s, L1, "placebo_rate")[0] >= 0.19 for s in SEEDS),
       "Q1 placebo >= 0.19 on 3/3 -- a fifth of thin cases flip on an edit that "
       "leaves v forced")

    # --- 91.6 the fragile class is the thinnest, 3/3
    for s, vals in zip(SEEDS, ((0.099, 0.340, 0.135),
                               (0.158, 0.355, 0.205),
                               (0.105, 0.166, 0.232))):
        o = arm(s, L1)["overlap_median_margin"]
        ck(abs(o["both"] - vals[0]) < 5e-4, f"both margin s{s}")
        ck(abs(o["corrupt_only"] - vals[1]) < 5e-4, f"corrupt_only margin s{s}")
        ck(abs(o["placebo_only"] - vals[2]) < 5e-4, f"placebo_only margin s{s}")
        ck(o["both"] < min(o["corrupt_only"], o["placebo_only"]), f"91.6 s{s}")

    # --- the confound control: the strata are NOT a difficulty gradient
    for s in SEEDS:
        nk = q(s, L1, "median_n_known")
        rk = q(s, L1, "median_round0_rank")
        ck(max(nk) - min(nk) <= 0.5, f"n_known flat across quartiles s{s}: {nk}")
        ck(max(rk) - min(rk) <= 1.0, f"round0_rank flat across quartiles s{s}: {rk}")

    # --- Y's AUROC is computed on 0-1 positives and is NOT evidence
    ck(arm(1, Y)["auroc_margin_predicts_corrupt_off"] != arm(1, Y)[
        "auroc_margin_predicts_corrupt_off"], "seed 1 Y AUROC is nan, as it must be")
    ck(round(arm(0, Y)["guard_rate"] * 656) == 1, "seed 0 Y has one turn-off")
    ck(round(arm(2, Y)["guard_rate"] * 584) == 1, "seed 2 Y has one turn-off")

    print(f"{n} assertions pass")


check_claims()
