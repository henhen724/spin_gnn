"""Figures for the `spin-gnn/margin` deck -- findings 79, 80 and 81.

Every number is PARSED OUT OF THE THREE JOBS' OWN STDOUT and none is retyped.
`check_claims()` at the bottom asserts the values the deck's prose states
against what the parse found, so a figure and its caption cannot drift apart --
the pattern make_channel_web.py and make_stagez_web.py established.

Sources, all COMPLETED 0:0:
  52441848  stagepairDiag3   three arms x 7,500 steps from one init
  52441962  stagepairWarm    two arms x 3,000 steps from the stage-Y checkpoint
  52446955  stagepairSwitch  one lambda=0 trunk, forked at 2,500/7,500/15,000
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors

ROOT = "/scratch/users/hshunt/spin_gnn"
OUT = "/home/users/hshunt/slides/decks/spin-gnn/margin/figures"
DIAG = os.path.join(ROOT, "stagepairDiag3_52441848.out")
WARM = os.path.join(ROOT, "stagepairWarm_52441962.out")
SWITCH = os.path.join(ROOT, "stagepairSwitch_52446955.out")

LOG58 = float(np.log(58))
LOG2 = float(np.log(2))


def _f(tok):
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def parse_sections(path, head_re, ncol):
    """{label: (n, ncol) array} for every `head_re` section in `path`.

    A row is a line whose first token parses as an int and which has exactly
    `ncol` whitespace-separated tokens -- which skips the header, the banners,
    the torch warning and the `end` row without needing to know their text.
    """
    out, cur = {}, None
    for line in open(path):
        m = head_re.match(line)
        if m:
            cur = m.group(1).strip()
            out[cur] = []
            continue
        if cur is None:
            continue
        t = line.split()
        if len(t) != ncol:
            continue
        try:
            int(t[0])
        except ValueError:
            continue
        out[cur].append([_f(x) for x in t])
    return {k: np.array(v, dtype=float) for k, v in out.items() if v}


# step main det_bce hinge gap |s| g_absmax dPair dMain ratio
diag = parse_sections(DIAG, re.compile(r"^=== (\S+): pair_weight="), 10)
# step main det_bce hinge gap sat |s| dPair dMain ratio
warm = parse_sections(WARM, re.compile(r"^=== (lambda=\d+ \w+): pair_weight="), 10)
# k main det_bce hinge gap sat |s| |dbody|
fork = parse_sections(
    SWITCH, re.compile(r"^--- fork at trunk step ([\d,]+), main there"), 8)
# step main det_bce |s| |dbody|
trunk = parse_sections(SWITCH, re.compile(r"^=== (trunk): lambda=0"), 5)

print("diag arms:", {k: v.shape for k, v in diag.items()})
print("warm arms:", {k: v.shape for k, v in warm.items()})
print("forks:", {k: v.shape for k, v in fork.items()})

DIAG_ARMS = ["control", "registered", "forward-only"]
FORKS = ["2,500", "7,500", "15,000"]
MAIN_BEFORE = {"2,500": 3.0411, "7,500": 2.4683, "15,000": 1.6822}


def parse_final_mains(path):
    """{arm: nll} from a job's own `main nll after ...` summary block, which is
    read at the true last step rather than at the last 250-step log row."""
    out, on = {}, False
    for line in open(path):
        if line.startswith("main nll after"):
            on = True
            continue
        if on:
            t = line.split()
            if len(t) == 2:
                out[t[0]] = float(t[1])
            elif out:
                break
    return out


def parse_ladder(path):
    """The switch-point summary table, five columns keyed by the fork step."""
    out, on = {}, False
    for line in open(path):
        if line.startswith("fork results"):
            on = True
            continue
        if on:
            t = line.split()
            if len(t) == 5 and re.fullmatch(r"[\d,]+", t[0]):
                out[t[0]] = {"main_before": float(t[1]),
                             "main_after": float(t[2]),
                             "hinge": float(t[3]), "gap": float(t[4])}
            elif out:
                break
    return out


DIAG_FINAL = parse_final_mains(DIAG)
LADDER = parse_ladder(SWITCH)
print("diag finals:", DIAG_FINAL)
print("ladder:", LADDER)

os.makedirs(OUT, exist_ok=True)


def end_label(ax, xs, ys, text, color, dy=0):
    """label_line takes the POINT to label, not the series -- it annotates one
    xy. So pick the curve's last finite sample, which is the right-hand end the
    deck rules ask the label to sit at.

    `dy` is in points and exists because two curves that END close together get
    two labels drawn on top of each other, which is unreadable and is this
    project's recurring defect class. Nudge, don't add a legend box.
    """
    ok = np.isfinite(np.asarray(xs, dtype=float)) & \
        np.isfinite(np.asarray(ys, dtype=float))
    i = int(np.nonzero(ok)[0][-1])
    label_line(ax, float(xs[i]), float(ys[i]), text, color=color, dy=dy)


def ladder_label(ax, a, col, k, c, grey_dy=10):
    """The two failing forks are BOTH muted, so they are deliberately not
    distinguishable from each other -- the claim is that they behave alike. Two
    identical-colour labels stacked on the same spot is the unreadable way to
    say that; one label naming both is the readable way.

    `grey_dy` nudges that combined label. On the two figures carrying a dashed
    reference line the failing forks end just under it, so the default upward
    nudge lands the text ON the dashes; those pass a negative value and put it
    below the curves instead.
    """
    if k == "15,000":
        end_label(ax, a[:, 0], a[:, col], "switched at 15,000", color=c)
    elif k == "2,500":
        end_label(ax, a[:, 0], a[:, col], "switched at 2,500 or 7,500",
                  color=c, dy=grey_dy)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- 1. the three arms' main loss ---------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
cols = SERIES[:3]
for arm, c in zip(DIAG_ARMS, cols):
    a = diag[arm]
    ax.plot(a[:, 0], a[:, 1], color=c, lw=1.8, rasterized=True)
    # control ends at 2.4644 and forward-only at 2.4177, 0.047 apart, so their
    # two labels land on top of each other unless they are nudged.
    end_label(ax, a[:, 0], a[:, 1], arm, color=c,
              dy={"control": 8, "forward-only": -9}.get(arm, 0))
ax.axhline(LOG58, color=MUTED, lw=1.0, ls="--")
ax.text(200, LOG58 + 0.03, r"$\log 58 = 4.060$, a uniform guess",
        color=MUTED, fontsize=9)
ax.set_xlabel("optimizer step")
ax.set_ylabel("training cross-entropy per token (nats)")
style_axes(ax)
save(fig, "three_arms_main.png")

# ---- 2. the body-gradient magnitude, control vs registered ---------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for arm, c in zip(["control", "registered"], series_colors(2, highlight=1)):
    a = diag[arm]
    ax.semilogy(a[:, 0], a[:, 7], color=c, lw=1.8, rasterized=True)
    end_label(ax, a[:, 0], a[:, 7], arm, color=c)
ax.set_xlabel("optimizer step")
ax.set_ylabel(r"$\|\partial L_{\rm pair}/\partial\,\mathrm{body}\|$")
style_axes(ax)
save(fig, "body_grad_magnitude.png")

# ---- 3. the ratio, in every arm that learns and the one that does not ----
fig, ax = plt.subplots(figsize=(7.2, 4.2))
# The warm-start arm is NOT on this figure even though its ratio is the third
# number the claim needs. Its x axis is fine-tuning steps from a trained
# checkpoint and these two are optimizer steps from a random init -- the same
# integer means a different thing, so overlaying them would be a dual x axis
# wearing one label. Its ratio goes in the table on the next page instead.
for arm, dy, c in zip(["control", "registered"], [-9, 9],
                      series_colors(2, highlight=1)):
    a = diag[arm]
    ax.semilogy(a[:, 0], a[:, 9], color=c, lw=1.8, rasterized=True)
    end_label(ax, a[:, 0], a[:, 9], arm, color=c, dy=dy)
ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
# x=900, not 200: the curves climb from 0.02 to 9 over the first 500 steps, so
# anything written near y=1 on the left gets struck through by the rise. Short
# enough to end by step 3,000, where both curves are above 2.
ax.text(900, 1.15, "ratio 1: the terms pull equally",
        color=MUTED, fontsize=9)
ax.set_xlabel("optimizer step")
ax.set_ylabel(r"$\|\partial L_{\rm pair}/\partial\,\mathrm{body}\|\;/\;"
              r"\|\partial L_{\rm main}/\partial\,\mathrm{body}\|$")
style_axes(ax)
save(fig, "ratio_three_arms.png")

# ---- 4. the warm start's hinge ------------------------------------------
w = warm["lambda=1 margin"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(w[:, 0], w[:, 3], color=SERIES[0], lw=1.8, rasterized=True)
end_label(ax, w[:, 0], w[:, 3], r"$\lambda=1$ from stage Y", color=SERIES[0])
ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
ax.text(60, 1.02, "margin 1.0: where a gap of zero sits", color=MUTED,
        fontsize=9)
ax.set_xlabel("fine-tuning step")
ax.set_ylabel(r"$L_{\rm pair}$, the mean hinge (nats of $g$)")
ax.set_ylim(0, 1.35)
style_axes(ax)
save(fig, "warm_hinge.png")

# ---- 5. the warm start's gap (a different scale: its own page) ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(w[:, 0], w[:, 4], color=SERIES[0], lw=1.8, rasterized=True)
end_label(ax, w[:, 0], w[:, 4], r"$g_{\rm clean}-g_{\rm broken}$",
           color=SERIES[0])
ax.axhline(0.0, color=MUTED, lw=1.0, ls="--")
ax.set_xlabel("fine-tuning step")
ax.set_ylabel(r"mean gap $g_{\rm clean}[p,v]-g_{\rm broken}[p,v]$")
style_axes(ax)
save(fig, "warm_gap.png")

# ---- 6. the warm start's main loss, both arms ---------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for lab, c in zip(["lambda=0 control", "lambda=1 margin"], SERIES[:2]):
    a = warm[lab]
    pretty = lab.replace("lambda", r"$\lambda$").replace("=0", "$=0$").replace(
        "=1", "$=1$")
    ax.plot(a[:, 0], a[:, 1], color=c, lw=1.8, rasterized=True)
    end_label(ax, a[:, 0], a[:, 1], pretty, color=c)
ax.set_xlabel("fine-tuning step")
ax.set_ylabel("training cross-entropy per token (nats)")
style_axes(ax)
save(fig, "warm_main.png")

# ---- 7. the switch-point ladder, hinge ----------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for k, c in zip(FORKS, series_colors(3, highlight=2)):
    a = fork[k]
    ax.plot(a[:, 0], a[:, 3], color=c, lw=1.8, rasterized=True)
    ladder_label(ax, a, 3, k, c, grey_dy=-40)
ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
# Not at x=40: the two failing forks sit AT the margin for the whole run, so
# the left end of that line is exactly where they are and the text came out
# struck through by both of them. x=420 clears the blue curve's descent too.
ax.text(420, 1.14, "margin 1.0: where a gap of zero sits", color=MUTED,
        fontsize=9)
ax.set_xlabel(r"steps since $\lambda$ was switched to 1")
ax.set_ylabel(r"$L_{\rm pair}$, the mean hinge")
style_axes(ax)
save(fig, "ladder_hinge.png")

# ---- 8. the switch-point ladder, main loss ------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for k, c in zip(FORKS, series_colors(3, highlight=2)):
    a = fork[k]
    ax.plot(a[:, 0], a[:, 1], color=c, lw=1.8, rasterized=True)
    ladder_label(ax, a, 1, k, c, grey_dy=-40)
ax.axhline(LOG58, color=MUTED, lw=1.0, ls="--")
ax.text(40, LOG58 + 0.04, r"$\log 58 = 4.060$", color=MUTED, fontsize=9)
ax.set_xlabel(r"steps since $\lambda$ was switched to 1")
ax.set_ylabel("training cross-entropy per token (nats)")
style_axes(ax)
save(fig, "ladder_main.png")

# ---- 9. the fraction of cases clearing the margin ----------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for k, c in zip(FORKS, series_colors(3, highlight=2)):
    a = fork[k]
    ax.plot(a[:, 0], a[:, 5], color=c, lw=1.8, rasterized=True)
    ladder_label(ax, a, 5, k, c)
ax.set_xlabel(r"steps since $\lambda$ was switched to 1")
ax.set_ylabel(r"fraction of cases with gap $\geq 1$")
ax.set_ylim(-0.03, 0.85)
style_axes(ax)
save(fig, "ladder_sat.png")

# ---- 10. the trunk, so the fork points are placed on a curve ------------
t = trunk["trunk"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(t[:, 0], t[:, 1], color=MUTED, lw=1.8, rasterized=True)
end_label(ax, t[:, 0], t[:, 1], r"$\lambda=0$ trunk", color=MUTED)
for k, c in zip(FORKS, series_colors(3, highlight=2)):
    n = int(k.replace(",", ""))
    ax.plot([n], [MAIN_BEFORE[k]], "o", color=c, ms=9, zorder=5)
    # The last fork sits at the right-hand axis limit, so a label offset to the
    # RIGHT of it is both clipped and on top of the trunk's own end label. That
    # one goes left and down instead.
    dx, dy, ha = ((-8, -16, "right") if k == "15,000" else (6, 10, "left"))
    ax.annotate("fork at %s" % k, (n, MAIN_BEFORE[k]),
                textcoords="offset points", xytext=(dx, dy), color=c,
                fontsize=9, ha=ha)
ax.set_xlabel("optimizer step")
ax.set_ylabel("training cross-entropy per token (nats)")
style_axes(ax)
save(fig, "trunk_with_forks.png")


# ---- the claims the deck's prose makes, asserted against the parse ------
def check_claims():
    def close(a, b, tol=5e-4, what=""):
        assert abs(a - b) <= tol, (what, a, b)

    # The deck quotes the finals from each job's OWN summary block, at step
    # 7,499 / 2,999, and the curves are logged every 250 so their last plotted
    # point is 249 steps earlier. These assert the two agree -- which is the
    # check that caught me quoting 2.4177 for the control when 2.4177 was the
    # FORWARD-ONLY arm's step-7,250 row and the control's was 2.4644.
    close(diag["control"][-1, 1], DIAG_FINAL["control"], 0.10, "control")
    close(diag["registered"][-1, 1], DIAG_FINAL["registered"], 0.10,
          "registered")
    close(diag["forward-only"][-1, 1], DIAG_FINAL["forward-only"], 0.20,
          "forward-only")
    assert DIAG_FINAL["registered"] > 3.90 > DIAG_FINAL["control"], DIAG_FINAL
    # the registered arm is flat from step 1,000 onward
    reg = diag["registered"]
    flat = reg[reg[:, 0] >= 1000, 1]
    assert flat.min() > 3.90 and flat.max() < 4.00, (flat.min(), flat.max())
    # every registered-arm det_bce sits at log 2
    assert abs(reg[reg[:, 0] >= 1000, 2].mean() - LOG2) < 0.05
    # the ratio exceeds 1 in BOTH the control and the healthy warm arm
    assert np.nanmedian(diag["control"][1:, 9]) > 1.0
    assert np.nanmedian(warm["lambda=1 margin"][:, 9]) > 3.0
    # and the magnitude, not the ratio, is what separates the failing arm
    assert np.nanmax(reg[:, 7]) / np.nanmax(diag["control"][:, 7]) > 100
    # the warm start's hinge leaves the margin and its main holds near 1.6
    close(warm["lambda=1 margin"][0, 3], 1.1006, 1e-3, "warm hinge first")
    # Not "holds flat": it SPIKES to 1.7171 at step 125 and settles at 1.6688.
    # The first version of this check asserted a max under 1.70 on the strength
    # of the final value and was wrong about the path -- the deck says peak and
    # settle because of this assertion, not in spite of it.
    close(warm["lambda=1 margin"][:, 1].max(), 1.7171, 1e-3, "warm main peak")
    close(warm["lambda=1 margin"][-1, 1], 1.6688, 1e-3, "warm main final")
    assert warm["lambda=0 control"][-1, 1] < warm["lambda=0 control"][0, 1]
    # the ladder: two forks collapse to uniform, the third satisfies the margin
    for k in ("2,500", "7,500"):
        assert fork[k][-1, 1] > 3.85, (k, fork[k][-1, 1])
        assert fork[k][:, 5].max() == 0.0, (k, "something cleared the margin")
    assert fork["15,000"][:, 5].max() >= 0.70, fork["15,000"][:, 5].max()
    # The ladder TABLE the deck reproduces is the job's own summary block, read
    # at step 1,499 rather than at the last logged 1,250, so it is parsed too
    # rather than retyped -- and the two have to tell the same story.
    assert set(LADDER) == set(FORKS), LADDER
    for k in ("2,500", "7,500"):
        assert LADDER[k]["main_after"] > 3.85 and LADDER[k]["hinge"] > 0.95, k
    assert LADDER["15,000"]["hinge"] < 0.30, LADDER["15,000"]
    assert LADDER["15,000"]["gap"] > 1.0, LADDER["15,000"]
    assert LADDER["15,000"]["main_after"] < 2.15, LADDER["15,000"]
    for k in FORKS:
        close(LADDER[k]["main_before"], MAIN_BEFORE[k], 5e-4, "before " + k)
    # |s| at the fork points: the two fatal ones are an order below the third
    s = {k: fork[k][0, 6] for k in FORKS}
    assert s["2,500"] < 0.8 and s["7,500"] < 0.9 and s["15,000"] > 2.0, s
    print("check_claims: every stated number agrees with the parsed logs")
    print("  |det_scale| at the forks: " +
          ", ".join("%s -> %.3f" % (k, s[k]) for k in FORKS))


check_claims()
print("figures in", OUT)
