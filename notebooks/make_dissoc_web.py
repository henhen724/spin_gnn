"""Render decks/spin-gnn/same-edit from the one-edit-two-positions probe.

Nothing is re-derived here. Every number comes from scripts/dissoc_score.py's own
helpers over the per-case CSVs, every bar is imported from the module that owns
it, and check_claims() asserts what the pages print against dissoc_summary.json,
which the scorer wrote inside the measurement job.

The HEADLINE branches on the verdict, including the H3 branch where the edit turns
out not to reach the tied model's colour computation either -- which was
pre-registered at a29fbc2 as the falsifier, and which NARROWS finding 68 rather
than producing a different question.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_web import WebDeck

from scripts.ddguard_score import (ALPHA, ARMS, RULE_BAR, SEEDS, ratio, rows_of,
                                   truthy)
from scripts.dissoc_score import FREED_BAR, RUN, cell_dir, fmean, sign_test
from scripts.stagev_score import S1_RHO50_SEEDS, S1_TRAINABLE
from scripts.stagej_score import PREDICTED_TRAINABLE

from spin_gnn.sudoku_dissoc import VARIANTS

DECK = "/home/users/hshunt/slides/decks/spin-gnn/same-edit"
J2_TRAINABLE = PREDICTED_TRAINABLE["J2"]
CELL_ORDER = [(arm, s) for arm in ARMS for s in SEEDS]
# What each variant is, in the order every figure and table prints them. The
# order is clean, placebo, rule, broken: reference, null, positive control, then
# the cell the finding turns on, so a reader meets the controls first.
VARIANT_GLOSS = {
    "clean": "no edit",
    "placebo": "far hint recoloured",
    "rule": "$w \\rightarrow c$",
    "broken": "$w \\rightarrow b$",
}


def cell_label(arm, s):
    return "%s s%d" % (arm, s)


def load_cells():
    """{(arm, seed): every quantity this deck quotes}."""
    out = {}
    for arm, s in CELL_ORDER:
        rows = rows_of(os.path.join(cell_dir(arm, s), "dissoc_behavior.csv"))
        d = {"rows": rows, "n": len(rows)}
        for v in VARIANTS:
            d["freed_%s" % v] = fmean(rows, "%s_p_freed" % v)[0]
            d["true_%s" % v] = fmean(rows, "%s_p_true" % v)[0]
            d["cand_%s" % v] = fmean(rows, "%s_p_cand" % v)[0]
            d["emits_%s" % v] = fmean(rows, "%s_emits_v" % v)[0]
        # Two denominators, kept apart on purpose. The figures use the
        # unconditional rate (one denominator, so the variants are comparable);
        # the tables use the conditional one, which is finding 68's published
        # statistic, and every header that shows it says "broken | clean".
        d["emits_cond_broken"], d["clean_n"] = ratio(rows, "clean_emits_v",
                                                     "broken_emits_v")
        n_valid = sum(1 for r in rows if truthy(r["rule_valid"]))
        d["rule_valid_n"] = n_valid
        d["rule_flips"] = (sum(1 for r in rows
                               if truthy(r["rule_valid"]) and truthy(r["rule_flips"]))
                           / n_valid) if n_valid else float("nan")
        b, c, n, p, med = sign_test(rows, "broken_p_freed", "placebo_p_freed")
        d.update({"b": b, "c": c, "sign_n": n, "p": p, "median": med})
        d["dissoc"] = fmean(rows, "dissoc")[0]
        d["drift"] = d["freed_placebo"] - d["freed_clean"]
        out[(arm, s)] = d
    return out


def check_claims(cells):
    """Refuse to render a deck that disagrees with the scorer's own appendix."""
    with open(os.path.join(RUN, "dissoc_summary.json")) as fh:
        js = json.load(fh)
    for arm in ARMS:
        for s in SEEDS:
            k, c = str(s), cells[(arm, s)]
            assert abs(js["rule"][arm][k] - c["rule_flips"]) < 1e-12, \
                "rule_flips disagrees at %s s%d" % (arm, s)
            for v in VARIANTS:
                assert abs(js["freed"][arm][k][v] - c["freed_%s" % v]) < 1e-12, \
                    "p_freed(%s) disagrees at %s s%d" % (v, arm, s)
            assert abs(js["emits"][arm][k] - c["emits_broken"]) < 1e-12, \
                "broken_emits_v disagrees at %s s%d" % (arm, s)
            assert abs(js["emits_cond"][arm][k] - c["emits_cond_broken"]) < 1e-12, \
                "conditional broken|clean disagrees at %s s%d" % (arm, s)
            assert abs(js["placebo_drift"][arm][k] - c["drift"]) < 1e-12, \
                "placebo drift disagrees at %s s%d" % (arm, s)
            sg = js["sign"]["%s_s%d" % (arm, s)]
            assert sg["b"] == c["b"] and sg["c"] == c["c"], \
                "sign test disagrees at %s s%d" % (arm, s)
    assert js["n_cases"] == cells[("S1", 0)]["n"]
    assert abs(js["FREED_BAR"] - FREED_BAR) < 1e-12
    return js["H1"], js["H2"], js["H4"], js["H5"], js


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def draw_variants(ax, cells, arm):
    """p_freed across the four one-token variants, one line per seed.

    Three series, the cap, in SEEDS order. The x axis is categorical and the
    variants are not a scale, so the line is a reading aid between paired points
    of one model, never a trend.
    """
    xs = list(range(len(VARIANTS)))
    dys = (12, 0, -12)
    for i, s in enumerate(SEEDS):
        ys = [cells[(arm, s)]["freed_%s" % v] for v in VARIANTS]
        ax.plot(xs, ys, "o-", ms=12 - 3 * i, lw=2, color=SERIES[i], zorder=3 + i)
        label_line(ax, xs[-1], ys[-1], "seed %d" % s, color=SERIES[i], dy=dys[i])
    ax.axhline(0.0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([VARIANT_GLOSS[v] for v in VARIANTS], fontsize=11)
    ax.set_xlim(-0.35, len(VARIANTS) + 0.55)
    ax.set_ylim(-0.06, 1.08)
    style_axes(ax, xlabel=None, ylabel="$p_{\\mathrm{freed}}$")


def draw_diff(ax, cells):
    """Median paired p_freed(broken) - p_freed(placebo), per cell.

    One series, six cells, drawn as two segments so no line crosses the
    tied/untied divider -- a categorical axis has no trend along it.
    """
    xs = list(range(len(CELL_ORDER)))
    ys = [cells[k]["median"] for k in CELL_ORDER]
    ax.plot(xs[:3], ys[:3], "o-", ms=11, lw=2, color=SERIES[0], zorder=3)
    ax.plot(xs[3:], ys[3:], "o-", ms=11, lw=2, color=SERIES[0], zorder=3)
    ax.axhline(0.0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.axhline(FREED_BAR, color=SERIES[1], lw=1.5, ls="--", zorder=2)
    label_line(ax, xs[-1], FREED_BAR, "bar %.2f" % FREED_BAR, color=SERIES[1])
    ax.axvline(2.5, color=MUTED, lw=1, ls=":", zorder=1)
    for x, name in ((1.0, "TIED, one shared block"), (4.0, "UNTIED, 16 blocks")):
        ax.annotate(name, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=11, color=MUTED,
                    annotation_clip=False)
    ax.set_xticks(xs)
    ax.set_xticklabels([cell_label(*k) for k in CELL_ORDER], fontsize=11)
    ax.set_xlim(-0.4, len(CELL_ORDER) + 0.9)
    style_axes(ax, xlabel=None, ylabel="median $\\Delta p_{\\mathrm{freed}}$")


def draw_positions(ax, cells):
    """The two readouts under the SAME broken edit, on one 0-1 scale.

    Both series are probabilities, so this is one axis and not a disguised dual
    axis: the node series is a rate of naming $v$, the colour series is the mass
    on the freed colour.
    """
    xs = list(range(len(CELL_ORDER)))
    series = (("names $v$ (node position)", "emits_broken", SERIES[0]),
              ("$p_{\\mathrm{freed}}$ (colour position)", "freed_broken", SERIES[1]))
    for i, (label, key, color) in enumerate(series):
        ys = [cells[k][key] for k in CELL_ORDER]
        ax.plot(xs[:3], ys[:3], "o-", ms=12 - 3 * i, lw=2, color=color, zorder=3 + i)
        ax.plot(xs[3:], ys[3:], "o-", ms=12 - 3 * i, lw=2, color=color, zorder=3 + i)
        label_line(ax, xs[-1], ys[-1], label, color=color, dy=(12, -12)[i])
    ax.axvline(2.5, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([cell_label(*k) for k in CELL_ORDER], fontsize=11)
    ax.set_xlim(-0.4, len(CELL_ORDER) + 2.2)
    ax.set_ylim(-0.06, 1.08)
    style_axes(ax, xlabel=None, ylabel="probability")


# --------------------------------------------------------------------------- #
# prose
# --------------------------------------------------------------------------- #
GAP = """## Finding 68 used two different edits, which is the gap this closes

The previous deck in this project measured two things and called the pair a
decomposition.

**The rule test** recoloured one hint token and asked whether the predicted
*colour* followed. Both architectures followed it on 350 of 350 cases.

**The guard test** recoloured the same hint to a *different* value and asked
whether the model still named the node whose colour had stopped being
determined. The tied model always did; the untied model stopped.

Those are **different edits read at different positions**. Two readings fit that
pattern equally well, and nothing in finding 68 separates them:

1. the edit reaches the model, changes what is deducible, and the node schedule
   does not consult it — a genuine rule/guard dissociation;
2. the second edit never reaches the model at all — in which case the honest
   claim is *insensitive to this edit*, and "no guard" is an artifact of which
   edit was used where.
"""

SEPARATING = """## The separating measurement is one edit read at both positions

Write the determinacy-destroying edit — the one whose node readout the tied model
ignores — and read the **colour** position on the same sequence.

Under that edit the node $v$ has exactly **two** candidate colours instead of
one. If the model's elimination machinery ran at all, the colour it was
previously certain of must now share probability with the colour the edit freed.

- If the freed colour's probability **rises**, the edit demonstrably reached the
  computation, and the node schedule ignored information the model had. That is
  a same-edit dissociation.
- If it **does not rise**, the edit reached nothing, and finding 68's headline
  has to narrow.

Both outcomes were written down before the measurement ran.
"""

QUAD = """## Four one-token variants of a single prompt, three at the same index

Every variant below differs from the clean prompt in **exactly one token**, and
that is asserted in code rather than intended. Let $v$ be a node whose colour is
forced, $c$ that forced colour, $w$ the single hint neighbour holding colour $a$,
and $b$ a colour some other neighbour already holds.

A **candidate** for $v$ is a colour that no known neighbour of $v$ holds. $v$ is
**forced** when it has exactly one candidate, and that is the only situation in
which direct deduction may write a colour for $v$.

| variant | the edit | what direct deduction then says |
| --- | --- | --- |
| clean | none | $v$ is forced to $c$; $a$ is excluded by $w$ |
| placebo | a hint far from $v$ | unchanged: $v$ is still forced to $c$ |
| rule | $w$'s colour $a \\rightarrow c$ | $v$ is forced to $a$ instead |
| broken | $w$'s colour $a \\rightarrow b$ | $v$ is **not** forced: candidates $\\{c, a\\}$ |

**rule** and **broken** rewrite the *same token at the same index* — only the
value written differs. That is what makes the comparison between them clean.
"""

POSITIONS = """## The two readouts are adjacent tokens in one forward pass

The reasoning section of the prompt is a sequence of (node, colour) pairs. For
the deduction about $v$ that gives two positions one apart:

- the **node position** is the token *before* $v$'s node token; the model's
  next-token distribution there is *which node it names next*. That is where a
  guard would act, by naming someone else.
- the **colour position** is $v$'s node token itself; the next-token
  distribution there is *what colour it gives $v$*. That is where the
  elimination rule shows.

$v$'s node token is physically present in all four variants, because every edit
is to a hint colour earlier in the prompt. So the colour readout is **teacher
forced** on "the node is $v$": the model is not asked to choose $v$, it is told
$v$ and asked for the colour. That is exactly the conditional direct deduction's
rule computes.
"""

PFREED = """## $p_{\\mathrm{freed}}$ is the probability mass on the colour the edit released

Let $z$ be the model's output logits at $v$'s colour position and $a$ the colour
that hint $w$ was holding before the edit. Then

$$p_{\\mathrm{freed}} = \\frac{e^{z_a}}{\\sum_{k} e^{z_k}}$$

where $k$ runs over the whole 58-symbol vocabulary, so this is the model's actual
output probability and not a renormalised one. $p_{\\mathrm{true}}$ is the same
quantity at $c$, the colour $v$ was forced to before the edit.

Under direct deduction the predicted values are exact:

| variant | $p_{\\mathrm{freed}}$ | because |
| --- | --- | --- |
| clean | $0$ | $a$ is held by a neighbour, so $a$ is eliminated |
| placebo | $0$ | the edit has nothing to do with $v$ |
| rule | $1$ | $a$ is the unique surviving colour |
| broken | $>0$ | $a$ is live again and shares with $c$ |
"""

SIGNTEST = """## The test is a paired sign test, one case at a time

Each case contributes two numbers from the same model and the same prompt:
$p_{\\mathrm{freed}}$ under the broken edit and under the placebo edit. Pairing
per case removes every source of variation except the edit itself.

Let $b$ count the cases with broken $>$ placebo and $c$ the cases with broken
$<$ placebo. Under the null that the edit does nothing, each discordant case is
a fair coin, so

$$p = 2 \\cdot \\Pr\\left[X \\le \\min(b, c)\\right], \\qquad X \\sim \\mathrm{Binomial}(b + c, \\tfrac{1}{2})$$

computed exactly, not approximated. **Holm correction** across the three seeds
multiplies the smallest $p$ by 3, the next by 2, the largest by 1, and enforces
that the adjusted values do not decrease — which controls the chance of any
false positive among the three at $\\alpha = %.2f$.
""" % ALPHA


def hypotheses_page():
    return """## What was registered before the measurement ran

Committed at `a29fbc2`, before the job was submitted.

- **H1 — rule replication.** The rule edit flips the emitted colour to $a$ in at
  least %.2f of cases, in all six cells, against a chance rate of $1/q = 0.250$.
  If this fails the construction is wrong and nothing else is read.
- **H2 — the dissociation.** For the tied model, $p_{\\mathrm{freed}}$ under the
  broken edit exceeds it under the placebo edit: sign test significant after Holm
  correction **and** a median difference of at least %.2f, in at least 2 of 3
  seeds. The direction was fixed in advance.
- **H3 — the falsifier.** If H2 misses, the registered conclusion is that the
  edit does not reach the tied model's colour computation either, and finding
  68's headline narrows to *insensitive to this edit at both positions*. That
  narrowing is the published result. No new arm and no third edit.
- **H4 — specificity.** The placebo must move $p_{\\mathrm{freed}}$ by at most
  %.2f, otherwise H2 has no null.
- **H5 — reproduction.** The node readout must reproduce finding 68's, case by
  case, from the same argmax at the same position.
""" % (RULE_BAR, FREED_BAR, FREED_BAR)


def arms_page(cells):
    return """## The two models differ in weight sharing and in size

| | tied (S1) | untied (J2) |
| --- | --- | --- |
| blocks | **1, applied 16 times** | **16, each its own weights** |
| width | 320 | 192 |
| trainable parameters | %s | %s |
| $\\rho_{50}$, 3 seeds | %.3f – %.3f | 0.580 |

$\\rho_{50}$ is the hint density at which a model solves half the instances;
lower is better, and the project's gate is $0.740$. The tied model is the
analytically tractable object — one block, applied repeatedly — and it is the
one that misses the gate.

**The arms differ by %.1f$\\times$ in parameters as well as in tying.** Nothing on
the following pages separates those two differences, and no page claims to.
""" % ("{:,}".format(S1_TRAINABLE), "{:,}".format(J2_TRAINABLE),
       min(S1_RHO50_SEEDS), max(S1_RHO50_SEEDS), J2_TRAINABLE / S1_TRAINABLE)


OFFPOLICY = """## The untied model's colour readout is off-policy, and is only context

Under the broken edit the untied model **stops naming $v$** — that was finding
68's result. Asking what colour it would give $v$ is therefore a question it
never faces in its own rollout: the readout is teacher forced onto a node the
model declined to choose.

That is a legitimate probe of the machinery and not a measurement of behaviour.
It is reported here for contrast only. **The registered hypothesis H2 is about
the tied model alone**, and was written that way before the numbers existed,
precisely so the untied column could not be promoted to a comparator after the
fact.
"""

SCOPE = """## What this measurement does not establish

- **Nothing was trained.** No model changed, $\\rho_{50}$ did not move, and the
  0.011 token-accuracy gap between the two architectures is exactly where it was.
- **The arms still differ in size as well as in tying,** by a factor of 4.1 in
  trainable parameters. Every difference here is tying *and* budget.
- **Round 0 only.** Both edits require every known neighbour of $v$ to be a hint,
  which is a deduction in the first round. Later rounds, where a deduction rests
  on an earlier deduction, are a separate registered test.
- **No localisation.** These are decision-level readouts. Nothing here says
  *where* in the network the behaviour lives, and on a tied model the usual
  per-layer ablation is not even defined.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--job", default="unknown")
    args = ap.parse_args()

    deck_explain.check()
    # Only CENSOR is dropped: nothing here is scored on a density grid. LEAK is
    # emitted because the shared tokenization figure asserts "THE ORDER WITHIN A
    # ROW IS THE LEAK" in bold, and a term a FIGURE introduces must be defined.
    shared = [p for p in deck_explain.PAGES if p is not deck_explain.CENSOR]

    cells = load_cells()
    h1, h2, h4, h5, js = check_claims(cells)
    n = cells[("S1", 0)]["n"]

    s1med = [cells[("S1", s)]["median"] for s in SEEDS]
    s1fb = [cells[("S1", s)]["freed_broken"] for s in SEEDS]

    if not h1:
        head = "The edits do not do what they were built to do"
    elif not h4:
        head = "An irrelevant edit moves the readout, so nothing is measurable"
    elif not h5:
        head = "This run disagrees with finding 68 on the same argmax"
    elif h2:
        head = "The tied model represents the ambiguity and schedules as if certain"
    else:
        head = "The tied model does not notice the edit at either position"

    prov = ("Job %s, commit %s. Six existing checkpoints read four times each over "
            "the %d round-0 cases of finding 68, case list identical. Registered by "
            "a29fbc2 before the job ran. Nothing trained."
            % (args.job, args.commit, n))

    d = WebDeck(args.out, head, "hshunt", "2026-11-13", prov)
    d.title_page(head, subtitle=(
        "A follow-up that finding 68 registered against itself: its rule test and "
        "its guard test used two different edits, so the pair was weaker than a "
        "dissociation. This uses one edit and reads two positions."))

    d.points_page(
        "Nothing is trained and no published number can change",
        ["Six checkpoints that already existed are read four times each.",
         "The gate, the free line and every stage result are untouched.",
         "The case list is finding 68's, asserted row for row."])

    # ---- shared explanation set, with the two shared figures
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, shuffle_adj=True,
                          fall=True)
    for page in shared:
        if page is deck_explain.FIGURE_SLOT:
            d.expect("tokenization",
                     "Every token is one categorical id in a %d-symbol vocabulary"
                     % ex.vocab.size)
            page_tokenization(d, ex)
            d.expect("trace",
                     "A deduction fires when a node sees all three other colours",
                     note="Every edit below changes exactly one colour token in "
                          "the hint section of a prompt like this one.")
            page_trace(d, ex)
        else:
            d._add(page)

    own = (GAP, SEPARATING, QUAD, POSITIONS, PFREED, SIGNTEST, hypotheses_page(),
           arms_page(cells), OFFPOLICY)
    for page in own:
        d._add(page)
    n_explain = (len([p for p in shared if p is not deck_explain.FIGURE_SLOT])
                 + 2 + len(own))

    # ---- H1, the construction check, first because everything rests on it
    d.table_page(
        "The rule edit flips the answer in every cell, as it must",
        ["cell", "cases", "flips to freed colour"],
        [[cell_label(*k), "%d" % cells[k]["rule_valid_n"],
          "%.4f" % cells[k]["rule_flips"]] for k in CELL_ORDER],
        note=("Bar %.2f, chance 0.250. Cases where recolouring $w$ to $c$ would "
              "break the colouring are excluded, not counted as misses."
              % RULE_BAR))

    # ---- H4, the null, before the headline that needs it
    d.table_page(
        "An edit far from $v$ moves the freed colour by thousandths",
        ["cell", "clean", "placebo", "difference"],
        [[cell_label(*k), "%.4f" % cells[k]["freed_clean"],
          "%.4f" % cells[k]["freed_placebo"], "%+.4f" % cells[k]["drift"]]
         for k in CELL_ORDER],
        note=("Mean $p_{\\mathrm{freed}}$ over all %d cases. This is the null "
              "the broken edit is read against; the bar is %.2f." % (n, FREED_BAR)))

    # ---- the four-variant figures, one page per arm
    d.claim_page(
        "One token moves the freed colour from zero to one half",
        lambda ax: draw_variants(ax, cells, "S1"), "variants_s1",
        note=("Tied model, mean $p_{\\mathrm{freed}}$ over %d cases. Clean and "
              "placebo sit at the floor, the rule edit at the ceiling." % n))

    d.claim_page(
        "The untied model lands on the same four values",
        lambda ax: draw_variants(ax, cells, "J2"), "variants_j2",
        note=("Same axes as the previous page. Its broken column is teacher "
              "forced onto a node it would not have named."))

    # ---- the headline
    d.claim_page(
        "The two readouts under one edit, on one probability scale",
        lambda ax: draw_positions(ax, cells), "positions",
        note=("Both series are probabilities under the SAME broken edit: naming "
              "$v$, and the mass freed at $v$'s colour."))

    d.claim_page(
        "How far the broken edit moves the freed colour past the placebo",
        lambda ax: draw_diff(ax, cells), "diff",
        note=("Median paired difference per cell, over %d cases. The dashed line "
              "is the registered effect-size bar." % n))

    d.table_page(
        "The paired sign test, three tied seeds against the registered bar",
        ["seed", "$b$ up", "$c$ down", "median $\\Delta$", "exact $p$", "Holm $p$"],
        [["%d" % s, "%d" % cells[("S1", s)]["b"], "%d" % cells[("S1", s)]["c"],
          "%+.4f" % cells[("S1", s)]["median"], "%.2e" % cells[("S1", s)]["p"],
          "%.2e" % js["sign"]["S1_s%d" % s]["holm"]] for s in SEEDS],
        note=("Tied model only: H2 was registered for it alone. The bar is Holm "
              "$p < %.2f$ with median $\\ge %.2f$." % (ALPHA, FREED_BAR)))

    d.table_page(
        "Under the broken edit the mass splits over exactly two colours",
        ["cell", "$p_{\\mathrm{true}}$", "$p_{\\mathrm{freed}}$", "sum"],
        [[cell_label(*k), "%.4f" % cells[k]["true_broken"],
          "%.4f" % cells[k]["freed_broken"], "%.4f" % cells[k]["cand_broken"]]
         for k in CELL_ORDER],
        note=("$c$ and $a$ are exactly $v$'s candidate set after the edit. The sum "
              "is the mass elimination did not leave elsewhere."))

    d.points_page(
        "Both architectures compute the same ambiguity; one acts on it",
        ["Every cell splits near evenly over the two surviving colours.",
         "The untied model then stops naming $v$; the tied model does not.",
         "So the difference is in the schedule, not in the elimination."])

    d.table_page(
        "Whether $v$ is still named, on both denominators",
        ["cell", "all %d cases" % n, "broken $\\mid$ clean", "clean cases"],
        [[cell_label(*k), "%.4f" % cells[k]["emits_broken"],
          "%.4f" % cells[k]["emits_cond_broken"],
          "%d of %d" % (cells[k]["clean_n"], n)] for k in CELL_ORDER],
        note=("The second column is finding 68's published statistic. The two "
              "differ because they are over different case sets."))

    # ---- H5, the reproduction
    d.table_page(
        "The node readout reproduces finding 68 case by case",
        ["cell", "agree", "broken $\\mid$ clean", "finding 68"],
        [[cell_label(*k),
          "%d of %d" % (js["reproduce"]["%s_s%d" % k]["agree"], n),
          "%.4f" % js["reproduce"]["%s_s%d" % k]["here"],
          "%.4f" % js["reproduce"]["%s_s%d" % k]["published"]]
         for k in CELL_ORDER],
        note=("Same argmax, same position, same checkpoint, a separate job. Both "
              "rate columns are conditional, so they are comparable."))

    d._add(SCOPE)

    # Last page by the project's own rule: the previous result on this eval.
    if h2:
        close = ["Finding 68 used two edits and could not rule out insensitivity.",
                 "One edit now splits the colour and leaves the node choice identical.",
                 "So determinacy is computed, and then not consulted by the schedule."]
        claim = "This is the dissociation finding 68 could not make"
    else:
        close = ["Finding 68 claimed the tied model runs the rule, not the guard.",
                 "The same edit moves neither readout, so insensitivity is not ruled out.",
                 "The registered narrowing applies and the earlier headline is too strong."]
        claim = "Finding 68's headline narrows, by its own registered falsifier"
    d.points_page(claim, close)

    path = d.write()
    print("wrote %s (%d pages, %d explanation, %d figures)"
          % (path, d.n, n_explain, len(os.listdir(d.fig_dir))))
    print("H1 %s  H2 %s  H4 %s  H5 %s" % (h1, h2, h4, h5))
    print("S1 median diff %s" % ["%+.4f" % x for x in s1med])
    print("S1 p_freed broken %s" % ["%.4f" % x for x in s1fb])
    print("headline: %s" % head)


if __name__ == "__main__":
    main()
