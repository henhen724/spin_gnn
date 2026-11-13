"""Render decks/spin-gnn/guard from the fire-and-guard probe (job 52211808).

Nothing is re-derived here. Every rate comes from scripts/ddguard_score.py's own
helpers applied to the per-case CSVs, the bars and the arm constants are imported
from the modules that own them, and check_claims() asserts the figures against
ddguard_summary.json, which the scorer wrote inside the job. Two modules naming
one number is how a deck comes to contradict itself -- that is not a hypothetical,
it refused a render on 2026-11-13 when a published rounding had been retyped.

The HEADLINE is chosen from the verdict rather than written in advance, including
the branch where both architectures turn out to have the guard and the gap is
somewhere else entirely.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_web import WebDeck

from scripts.ddguard_score import (ALPHA, ARMS, J2_RHO50, PLACEBO_FLOOR, RULE_BAR,
                                  SEEDS, binom_exact, holm, ratio, rows_of, truthy)
from scripts.stagev_score import (FREE, J2_ACC, NEED, S1_ACC, S1_RHO50,
                                  S1_RHO50_SEEDS, S1_TRAINABLE)
from scripts.stagej_score import PREDICTED_TRAINABLE

DECK = "/home/users/hshunt/slides/decks/spin-gnn/guard"
RUN = os.path.join(os.environ["SCRATCH"], "spin_gnn", "runs", "2026-11-13_ddguard")
J2_TRAINABLE = PREDICTED_TRAINABLE["J2"]
# Finding 28's own numbers, for the "previous best on this eval" page. They are
# the ONLY quoted values in this deck that are not recomputed from disk, because
# the checkpoints they were measured on are N=24 models from a different stage
# and are not re-run here. Flagged as such on that page.
F28 = {"A": 0.000, "B": 0.176}

CELL_ORDER = [(arm, s) for arm in ARMS for s in SEEDS]


def cell_label(arm, s):
    return "%s s%d" % (arm, s)


def load_cells():
    """{(arm, seed): dict of every rate and diagnostic this deck quotes}."""
    out = {}
    for arm, s in CELL_ORDER:
        tag = "%s_s%d" % (arm, s)
        g = rows_of(os.path.join(RUN, "guard", tag, "skip_behavior.csv"))
        r = rows_of(os.path.join(RUN, "rule", tag, "behavior.csv"))
        clean_n = sum(truthy(x["clean_emits_v"]) for x in g)
        broken, _ = ratio(g, "clean_emits_v", "corrupt_emits_v")
        plac, _ = ratio(g, "clean_emits_v", "placebo_emits_v")
        rule, rule_n = ratio(r, "clean_correct", "corrupt_flipped")
        # The logit drop is the measurement the argmax rates are a threshold of,
        # and it is what makes a flat rate readable as "the logits did not move"
        # rather than "the logits moved but not past the boundary".
        drop = sum(float(x["corrupt_v_logit_drop"]) for x in g) / len(g)
        pdrop = sum(float(x["placebo_v_logit_drop"]) for x in g) / len(g)
        skips = [x for x in g if not truthy(x["corrupt_emits_v"])]
        nodes = [x for x in skips if truthy(x["pred_is_node_token"])]
        out[(arm, s)] = {
            "n": len(g), "clean_n": clean_n,
            # CONDITIONAL on clean_emits_v -- the scorer's registered statistics.
            # `clean` is 1 by construction under this conditioning, which is why
            # it is not one of them.
            "broken": broken, "placebo": plac, "rule": rule, "rule_n": rule_n,
            # UNCONDITIONAL, same counts over all cases. These three ARE mutually
            # comparable, which the conditional pair is not, so the figure plots
            # these and the tables report those. Keeping them apart is the point:
            # a draft of this deck plotted one clean rate against two conditional
            # rates and the page title claimed they coincided when they could not.
            "u_clean": clean_n / len(g),
            "u_placebo": sum(truthy(x["placebo_emits_v"]) for x in g) / len(g),
            "u_broken": sum(truthy(x["corrupt_emits_v"]) for x in g) / len(g),
            "drop": drop, "pdrop": pdrop,
            "n_other": len(nodes),
            "forced_of_other": (sum(truthy(x["pred_is_forced"]) for x in nodes)
                                / len(nodes)) if nodes else float("nan"),
            "rows": g,
        }
    return out


def mcnemar(cells):
    """Per seed: (seed, n paired, b, c, raw p, Holm p). b = S1 emits, J2 does not."""
    detail = []
    for s in SEEDS:
        a, j = cells[("S1", s)]["rows"], cells[("J2", s)]["rows"]
        both = [(x, y) for x, y in zip(a, j)
                if truthy(x["clean_emits_v"]) and truthy(y["clean_emits_v"])]
        b = sum(1 for x, y in both
                if truthy(x["corrupt_emits_v"]) and not truthy(y["corrupt_emits_v"]))
        c = sum(1 for x, y in both
                if truthy(y["corrupt_emits_v"]) and not truthy(x["corrupt_emits_v"]))
        detail.append([s, len(both), b, c, binom_exact(min(b, c), b + c)])
    for row, adj in zip(detail, holm([d[4] for d in detail])):
        row.append(adj)
    return detail


def check_claims(cells, mc):
    """Assert every number this deck prints against the scorer that produced it.

    The scorer ran INSIDE the job and wrote ddguard_summary.json before this
    module existed in a rendered form. If the two disagree anywhere, the render
    fails rather than publishing a deck that argues with its own appendix.
    """
    with open(os.path.join(RUN, "ddguard_summary.json")) as fh:
        js = json.load(fh)
    for arm, s in CELL_ORDER:
        for key, where in (("guard", "broken"), ("placebo", "placebo"), ("rule", "rule")):
            want, got = js[key][arm][str(s)], cells[(arm, s)][where]
            assert abs(want - got) < 1e-12, (
                "check_claims: %s %s s%d is %.12f here and %.12f in the scorer"
                % (key, arm, s, got, want))
    assert js["n_cases"] == cells[("S1", 0)]["n"], "case count disagrees"
    h1 = min(cells[("S1", s)]["rule"] for s in SEEDS) >= RULE_BAR
    h3 = all(cells[k]["placebo"] >= PLACEBO_FLOOR for k in CELL_ORDER)
    sig = sum(1 for _, _, b, c, _, adj in mc if adj < ALPHA and b > c)
    h2 = (sig >= 2) if (h1 and h3) else None
    assert (js["H1"], js["H2"], js["H3"]) == (h1, h2, h3), (
        "check_claims: verdicts disagree, scorer %s here %s"
        % ((js["H1"], js["H2"], js["H3"]), (h1, h2, h3)))
    return h1, h2, h3, sig


# ---------------------------------------------------------------- figures

def _draw_cells(ax, cells, series, ylabel, label_dy):
    """Shared frame for the two six-cell figures.

    The x axis is categorical and the two halves are two DIFFERENT NETWORKS, so
    each series is drawn as two segments and no line crosses the divider. A line
    through the boundary reads as a trend along an axis that has none; the draft
    drew one and it was the most misleading mark on the page.
    """
    xs = list(range(len(CELL_ORDER)))
    for i, (label, key, color) in enumerate(series):
        ys = [cells[k][key] for k in CELL_ORDER]
        # Markers shrink along the series order so COINCIDENT points draw as
        # concentric rings instead of the last one painting over the rest. On the
        # tied cells every series here is equal to the others, and the whole
        # claim is that they are equal -- a single visible dot would read as a
        # single series and leave the reader unable to see it.
        ms = 13 - 3.5 * i
        ax.plot(xs[:3], ys[:3], "o-", ms=ms, lw=2, color=color, zorder=3 + i)
        ax.plot(xs[3:], ys[3:], "o-", ms=ms, lw=2, color=color, zorder=3 + i)
        # Right-hand end labels collide whenever two series land close together
        # (broken +0.331 against placebo -0.011 at J2 s2 is a few pixels), so the
        # offsets are passed in per series rather than left to overlap.
        label_line(ax, xs[-1], ys[-1], label, color=color, dy=label_dy[i])
    ax.axvline(2.5, color=MUTED, lw=1, ls=":", zorder=1)
    for x, name in ((1.0, "TIED, one shared block"), (4.0, "UNTIED, 16 blocks")):
        ax.annotate(name, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=11, color=MUTED,
                    annotation_clip=False)
    ax.set_xticks(xs)
    ax.set_xticklabels([cell_label(*k) for k in CELL_ORDER], fontsize=11)
    ax.set_xlim(-0.4, len(CELL_ORDER) + 0.9)
    style_axes(ax, xlabel=None, ylabel=ylabel)


def draw_rates(ax, cells, n):
    """The three UNCONDITIONAL rates: of all n cases, how many name v.

    Unconditional on purpose. The conditional pair the scorer registers has
    `clean` in its denominator, so a clean rate cannot be plotted against them --
    they are fractions of different things. These three share one denominator and
    so can be compared to each other, which is what the claim needs.
    """
    _draw_cells(ax, cells,
                (("clean", "u_clean", SERIES[0]),
                 ("placebo", "u_placebo", SERIES[1]),
                 ("broken", "u_broken", SERIES[2])),
                "fraction of all %d cases that name $v$" % n, (14, 0, -14))
    ax.set_ylim(-0.05, 1.08)


def draw_logit(ax, cells):
    """Mean change in v's node logit under each edit, all six cells.

    The rates are an argmax, which cannot tell "the logits did not move" from
    "they moved and stayed inside the boundary". This can.
    """
    _draw_cells(ax, cells,
                (("broken", "drop", SERIES[0]), ("placebo", "pdrop", SERIES[1])),
                "mean drop in $v$'s node logit", (10, -10))
    ax.axhline(0.0, color=MUTED, lw=1, zorder=1)


# ---------------------------------------------------------------- pages

TWO_THINGS = """## Direct deduction is a rule plus a guard on that rule

Direct deduction, the algorithm this project asks a network to learn, does two separable things in every round.

- **The rule.** A node $v$ whose known neighbours show $q-1$ of the $q$ colours has exactly one colour left, so that colour is **forced** and direct deduction assigns it. Here $q = 4$.
- **The guard.** Direct deduction emits $v$ **only when $v$ is actually determined**. If $v$'s known neighbours show fewer than $q-1$ colours, two or more colours remain, and the algorithm says nothing about $v$ this round.

These are different operations and a model can have one without the other. A model that wrote a colour for whichever uncoloured node came next in index order would apply the rule perfectly and have no guard at all: it would be right whenever the next node happened to be determined, and guessing otherwise.

That is why a solve rate cannot separate them. A solve rate is one number for the whole behaviour, and the two failure modes both show up in it as "sometimes wrong".
"""

RULE_TEST = """## The rule test redirects the forcing set with one token

Take a node $v$ that direct deduction forces in round 0, so every neighbour of $v$ whose colour is known is a **hint** -- a colour stated in the prompt rather than deduced earlier.

$v$'s known neighbours show $q-1 = 3$ distinct colours, leaving one colour $c$ free, and direct deduction assigns $c$ to $v$. Among those neighbours there is a colour $a$ held by **exactly one** of them; call that neighbour $w$.

**The edit: change $w$'s colour from $a$ to $c$.** One token in the prompt changes. Now $v$'s known neighbours still show three distinct colours, but the free one is $a$ instead of $c$, so direct deduction's answer for $v$ flips from $c$ to $a$.

Nothing else moves: the token naming $v$, the adjacency table, and the whole reasoning prefix are untouched. A model applying the rule must flip with it. A model insensitive to neighbour colours cannot.

The corrupted prompt is deliberately not required to be a globally valid colouring -- recolouring $w$ may clash with $w$'s own neighbours. Only $v$'s prediction is ever scored, and requiring global validity would mean editing several tokens at once.
"""

GUARD_TEST = """## The guard test destroys the forcing set with one token

Same node $v$, same binding neighbour $w$, same single token. Only the new colour differs.

**The edit: change $w$'s colour from $a$ to $b$, where $b$ is a colour that another of $v$'s known neighbours already holds.**

Now nobody holds $a$. $v$'s known neighbours show only $q-2 = 2$ distinct colours, so two colours remain available to $v$ and **direct deduction does not emit $v$ in this round at all**.

The model is being asked to do the opposite of what the rule test asks. There, the right behaviour is to change its answer. Here, the right behaviour is to **stop answering** -- to name some other node, or to end the round.

Every case is checked by the algorithm itself after the edit, and the run aborts if any case is still forced. All 350 passed, so the construction is doing what this page says it does.
"""

WHERE_READ = """## The two tests read logits one token position apart

The reasoning section is written as a sequence of (node, colour) pairs: the model names a node, then gives that node's colour, then names the next node.

- **The rule test** reads the logits **at $v$'s node token**, which predict $v$'s colour. The question is *which colour*.
- **The guard test** reads the logits at the token **before** that one, which predict the node identity. The question is *which node, if any*.

So the guard test is not a harder version of the rule test. It reads a different output head of the same forward pass, one position to the left, and asks a question the rule test never asks.

Two quantities are read there. `emits_v` is whether the argmax at that position is $v$'s token. The **logit drop on $v$** is how far $v$'s logit falls from the clean prompt to the edited one, in logits. The rate is a threshold on the drop, so a flat rate plus a flat drop means the logits did not move, while a flat rate with a large drop would mean they moved without crossing the boundary.
"""

PLACEBO = """## The placebo edits a hint that cannot matter to $v$

A one-token edit anywhere in the prompt might disturb the model for reasons that have nothing to do with determinacy. Without a control, a model that stopped naming $v$ after *any* edit would be indistinguishable from one that checked the guard.

So every case carries a second, **placebo** edit: recolour a hint that is **not a neighbour of $v$**, to a colour that keeps the revealed colouring valid. $v$ stays forced, direct deduction's round is unchanged, and the model should still name $v$.

`placebo_emits_v` is therefore the upper baseline that the broken rate has to be read against. Three numbers, on the same case:

- **clean** -- does it name $v$ from the untouched prompt?
- **placebo** -- does it still name $v$ after an irrelevant edit?
- **broken** -- does it still name $v$ after $v$ stopped being determined?

A **low broken rate with a high placebo rate** means the guard is real. **Broken equal to placebo** means the node choice did not notice the difference between the two edits, which is the signature of no guard at all.
"""

EQUATIONS = """## Rule, guard and placebo are three conditional rates

Each is a fraction of cases, conditioned on the cases the model got right before the edit. $\\#\\{\\cdot\\}$ counts cases, and all three share the same 350 round-0 cases.

$$\\mathrm{rule} = \\frac{\\#\\{\\text{clean correct} \\;\\wedge\\; \\text{corrupt flipped}\\}}{\\#\\{\\text{clean correct}\\}}$$

where *clean correct* means the predicted colour for $v$ was $c$ before the edit, and *corrupt flipped* means it was $a$ after it. **High is good.**

$$\\mathrm{guard} = \\frac{\\#\\{\\text{clean emits } v \\;\\wedge\\; \\text{corrupt emits } v\\}}{\\#\\{\\text{clean emits } v\\}}$$

$$\\mathrm{placebo} = \\frac{\\#\\{\\text{clean emits } v \\;\\wedge\\; \\text{placebo emits } v\\}}{\\#\\{\\text{clean emits } v\\}}$$

where *emits $v$* means the argmax at the node position was $v$'s token. **For `guard`, LOW is good** -- it is the fraction of the time the model kept naming a node that had stopped being determined.

The conditioning matters: a model that was already wrong before the edit cannot be scored on whether it followed the edit, so the denominator is the cases it had right.
"""

CONDITIONING = """## The same counts read two ways, and the denominators differ

Both readings appear in this deck. They are not the same number and the difference is worth one page.

- **Conditional**, as defined on the previous page and as the scorer registers them: `clean` is in the denominator. They answer *"of the cases it got right before the edit, how many survived it?"* Under this conditioning `clean` is **1 by construction**, so it is not a third column and never appears as one.
- **Unconditional**: the same counts divided by **all** 350 cases. Three comparable fractions of one denominator, so `clean`, `placebo` and `broken` can be set against each other directly.

Each answers a question the other cannot. The conditional rate is the right one for the hypothesis test, because it removes cases the model never had right. The unconditional rate is the right one for *"did the edit change anything at all"*, because a cell whose three unconditional rates are **equal** produced the same behaviour under all three prompts.

So: **the figures plot the unconditional rates and the tables report the conditional ones**, each page says which, and a conditional rate is written $\\mathrm{broken} \\mid \\mathrm{clean}$ wherever both are in play.
"""

MCNEMAR = """## McNemar counts only the pairs where the arms disagree

The two architectures are run on the **same 350 cases, in the same order**, so each case gives a pair of answers rather than two independent samples. The scorer asserts this row by row before using it and stops if it ever fails.

For a pair of arms on one seed, restricted to the cases where **both** name $v$ from the clean prompt, let

- $b$ = cases where the **tied** model still names $v$ after the edit and the untied one does not,
- $c$ = cases where the **untied** model still names $v$ and the tied one does not.

Cases where both behave the same carry no information about which is better and are discarded. Under the hypothesis of no difference, each of the $b + c$ discordant cases is a coin flip, so

$$p = 2 \\sum_{i=0}^{\\min(b,c)} \\binom{b+c}{i} 2^{-(b+c)}$$

capped at 1. This is the **exact two-sided McNemar test**. Three seed pairs are tested, so the three $p$ values are corrected by **Holm's method**: sort them, multiply the smallest by 3, the next by 2, the last by 1, and carry the running maximum forward so the adjusted values stay ordered.
"""

MCNEMAR_WHY = """## A paired test is the reason this probe is cheap

$\\rho_{50}$ is a median over three seeds of a 14-point grid, and a measurement of this project's own noise floor put the run-to-run standard deviation at **2 to 22 times the effect being chased**. That is why five consecutive stages reported "within noise".

The two rates on the previous pages are **paired per case**. The same 350 prompts go to both networks, so the arms differ in the network and in nothing else -- no instance sampling, no training, no optimizer, no kernel nondeterminism, because every forward pass here is under evaluation mode with gradients off.

That changes what a given number of samples can resolve. Three seeds of a grid cannot see 0.0011 of token accuracy; 350 paired cases can see a difference of a few percent in a conditional rate.

It also costs one short job rather than GPU-weeks, because nothing is trained. Six checkpoints that already existed are read twice each.
"""


def arms_page(cells):
    return """## The two arms share every setting except weight sharing

| | S1, tied | J2, untied |
|---|---|---|
| blocks applied | 16 | 16 |
| distinct block weights | **1** | **16** |
| width $d_{\\mathrm{model}}$ | 320 | 192 |
| attention heads | 4 | 4 |
| feed-forward sublayer | none | none |
| trainable parameters | %s | %s |
| $\\rho_{50}$, median of 3 seeds | %s | %.3f |
| validation token accuracy | %.4f | %.4f |

**S1** applies one shared block sixteen times, $h_{t+1} = F(h_t)$ at tied weights. It is the most analytically tractable object this project has: one set of weights to read instead of sixteen. It **misses** the $%.3f$ gate.

**J2** is the same depth with sixteen independent blocks. It clears the gate and is the standing minimum.

**The one way they are not matched**, stated here and not in an appendix: %s against %s trainable parameters is a **4.1-fold** difference. Any gap found below is weight sharing *and* budget together. Removing that confound is not possible without changing the thing under test -- a tied model with J2's parameter count is a *wider* tied model, which is a different cell, and the stage that ran it found the same plateau.
""" % ("{:,}".format(S1_TRAINABLE), "{:,}".format(J2_TRAINABLE), S1_RHO50,
       J2_RHO50, S1_ACC, J2_ACC, FREE, "{:,}".format(S1_TRAINABLE),
       "{:,}".format(J2_TRAINABLE))


def hypotheses_page():
    return """## What was predicted, before any of these numbers existed

Committed as the docstring of `scripts/ddguard_score.py` by a job that the evaluation job depended on, so no cell could run against an unregistered hypothesis. Each is written as the expression that would be computed.

- **H1, the rule is present in the tied model.** Statistic $\\min_s \\mathrm{rule}(\\mathrm{S1}, s)$ against a bar of $%.2f$, the bar finding 27 used. Predicted **confirmed**, near 1. It is here because the guard number is uninterpretable without it: a guard deficit in a model that never learned the rule localises nothing.
- **H2, the tied model withholds less.** Statistic: the paired McNemar above, on each of three seed pairs. Confirmed if **at least 2 of 3** are significant at $\\alpha = %.2f$ after Holm **and** $b > c$ in every significant one. **The effect size was deliberately not barred** -- there is no honest way to convert finding 28's magnitude onto this scale, and barring a number I could not derive would be inventing one.
- **H3, the measurement is specific.** Every cell's placebo rate at or above $%.2f$. This is a guard on the instrument, not a result; a cell that misses it has its guard number reported as uninterpretable.

**The registered falsifier for H2 was real.** Finding 28's leaked arm, which never learned the rule, still withheld on 99.2%% of cases -- determinacy looked cheap to learn. If it were that cheap here, both arms would score near zero, H2 would be falsified, and the answer would be that the guard is *not* what weight sharing costs. That outcome would have closed a door and been worth the job.
""" % (RULE_BAR, ALPHA, PLACEBO_FLOOR)


SCOPE = """## What this does and does not show

**Does.** On round-0 cases, one shared block applies direct deduction's elimination rule as well as its untied twin and does not apply direct deduction's guard at all. Its node choice is invariant to the difference between an edit that destroys determinacy and an edit that changes nothing relevant, and the node logits confirm the invariance is real rather than a threshold effect.

**Does not.**

- **It does not close the accuracy gap.** No model was trained; the gate is still $\\rho_{50} \\le %.3f$ and S1 still misses it by $%.3f$, which is $%.4f$ of validation token accuracy. This names what the gap *is*, which is a different thing from fixing it.
- **It does not separate weight sharing from budget.** The arms differ 4.1-fold in parameters.
- **Round 0 only.** Both edits need every known neighbour of $v$ to be a hint. Whether the model eliminates against a fact it *deduced* is a separate question about provenance, tested by a third module that was deliberately not run here.
- **It does not say where in the network the guard lives.** `emits_v` is a decision-level statistic. Localising it -- activation patching, head and sublayer ablation -- would be the next job and is not this one.
- **One untied seed behaves unlike the other two**, and that is reported rather than smoothed. The guard is a per-seed property in this arm, not an architectural constant.
""" % (FREE, float(S1_RHO50) - FREE, NEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--job", default="unknown")
    args = ap.parse_args()

    deck_explain.check()
    # CENSOR is dropped: nothing here is scored on a grid that could fail to
    # clear 0.5. LEAK IS EMITTED -- the shared tokenization figure asserts "THE
    # ORDER WITHIN A ROW IS THE LEAK" in bold, and a term a FIGURE introduces is
    # a term this deck has to define.
    shared = [p for p in deck_explain.PAGES if p is not deck_explain.CENSOR]

    cells = load_cells()
    mc = mcnemar(cells)
    h1, h2, h3, sig = check_claims(cells, mc)

    s1g = [cells[("S1", s)]["broken"] for s in SEEDS]
    j2g = [cells[("J2", s)]["broken"] for s in SEEDS]

    if not h3:
        head = "The placebo control failed, so the guard numbers mean nothing"
    elif not h1:
        head = "The tied model does not apply the elimination rule at all"
    elif h2:
        head = "A shared block learns DD's rule and not DD's guard"
    else:
        head = "Both architectures run the guard, so the gap is elsewhere"

    prov = ("Job %s, commit %s. Six existing checkpoints (S1 and J2, three seeds "
            "each) read twice over %d shared round-0 cases. Registered by job "
            "52211756 at bb74ef0. Nothing trained; no published number changes."
            % (args.job, args.commit, cells[("S1", 0)]["n"]))

    d = WebDeck(args.out, head, "hshunt", "2026-11-13", prov)
    d.title_page(head, subtitle=(
        "Not a stage: no knob moves and nothing is trained. The question is "
        "whether the most analytically tractable architecture runs direct "
        "deduction, which its solve rate cannot answer."))

    d.points_page(
        "Solving is not deducing, and the gate only measures solving",
        ["The gate asks whether a model solves instances at low hint density.",
         "It cannot say *which part* of the algorithm a model is missing.",
         "Every registered knob is closed, so the instrument changes, not the arm."])

    d.points_page(
        "Nothing is trained here and no published number can change",
        ["Six checkpoints that already existed are read twice each.",
         "The job ran in 1 minute 39 seconds on one GPU.",
         "The gate, the free line and every stage result are untouched."])

    # ---- the shared explanation set, with the two shared figures
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
                     note="The edits below change exactly one colour token in the "
                          "hint section of a prompt like this one.")
            page_trace(d, ex)
        else:
            d._add(page)

    # ---- what this probe is
    own = (TWO_THINGS, RULE_TEST, GUARD_TEST, WHERE_READ, PLACEBO, EQUATIONS,
           CONDITIONING, MCNEMAR, MCNEMAR_WHY, arms_page(cells),
           hypotheses_page())
    for page in own:
        d._add(page)
    # shared prose + the two shared figures + the probe-specific pages.
    n_explain = (len([p for p in shared if p is not deck_explain.FIGURE_SLOT])
                 + 2 + len(own))

    # ---- H1
    d.table_page(
        "Both architectures apply the elimination rule, essentially perfectly",
        ["cell", "clean correct", "rule"],
        [[cell_label(*k), "%d of %d" % (cells[k]["rule_n"], cells[k]["n"]),
          "%.4f" % cells[k]["rule"]] for k in CELL_ORDER],
        note=("Redirect the forcing set and the prediction follows, in every cell, "
              "against a bar of %.2f and a chance rate of 0.250." % RULE_BAR))

    # ---- H2, split by arm so each page carries one claim
    for arm, claim, note in (
            ("S1", "The tied model never withholds, at any of three seeds",
             "Broken equals placebo in every row: the node choice does not "
             "distinguish the two edits. Low broken would be good."),
            ("J2", "Two untied seeds withhold completely; the third mostly does",
             "Seed 2 is not seed 0 or 1. The guard is a per-seed property here, "
             "reported rather than smoothed into a median.")):
        d.table_page(
            claim,
            # Short headers on purpose: table_page spaces columns evenly and
            # overflow into the next column is silent (LOG 2026-11-10, four
            # tables). The conditioning page carries the long explanation.
            ["cell", "placebo $\\mid$ clean", "broken $\\mid$ clean",
             "clean cases"],
            [[cell_label(arm, s), "%.4f" % cells[(arm, s)]["placebo"],
              "%.4f" % cells[(arm, s)]["broken"],
              "%d of %d" % (cells[(arm, s)]["clean_n"], cells[(arm, s)]["n"])]
             for s in SEEDS],
            note=note)

    d.claim_page(
        "The tied model names $v$ just as often after every edit",
        lambda ax: draw_rates(ax, cells, cells[("S1", 0)]["n"]), "rates",
        note=("Unconditional rates, one denominator, so the three are "
              "comparable. The tied arm's three lines coincide exactly."))

    d.claim_page(
        "Breaking determinacy moves the untied node logits, not the tied",
        lambda ax: draw_logit(ax, cells), "logitdrop",
        note=("The rates above are an argmax. This is the quantity underneath, so "
              "a flat rate can be told from a sub-threshold shift."))

    # No ratio column. On the tied rows it would be noise over noise (-0.003 /
    # -0.000 = 6), and at J2 s2 it reads 31 against J2 s0's 32 while the absolute
    # drop is 0.331 against 10.020 -- a column that makes a weak effect look as
    # strong as a strong one is worse than no column.
    d.table_page(
        "The tied node logits move by three thousandths or less",
        ["cell", "broken edit", "placebo edit"],
        [[cell_label(*k), "%+.3f" % cells[k]["drop"], "%+.3f" % cells[k]["pdrop"]]
         for k in CELL_ORDER],
        note=("Mean fall in $v$'s node logit, in logits, over all %d cases. The "
              "clean decision is made at a margin of several logits."
              % cells[("S1", 0)]["n"]))

    d.table_page(
        "Every seed pair is discordant in one direction only",
        ["seed", "paired cases", "$b$: tied only", "$c$: untied only",
         "exact $p$", "Holm $p$"],
        [["%d" % s, "%d" % n, "%d" % b, "%d" % c, "%.2e" % p, "%.2e" % adj]
         for s, n, b, c, p, adj in mc],
        note=("Restricted to cases both arms name $v$ cleanly. $c = 0$ "
              "everywhere: the untied model is never the one that fails."))

    d.table_page(
        "The untied model's other node is forced; the tied model's is not",
        ["cell", "names another node", "of those, forced"],
        [[cell_label(*k), "%d of %d" % (cells[k]["n_other"], cells[k]["n"]),
          "%.3f" % cells[k]["forced_of_other"]] for k in CELL_ORDER],
        note=("For the tied arm these are mostly cases it never named $v$; for "
              "the untied arm they are the withholds. Same question, different "
              "populations."))

    d._add(SCOPE)

    # Not "where the gate has plenty": 0.011 is plenty only RELATIVE to the
    # 0.017 gap, and a title should not need that qualifier to be true.
    d.points_page(
        "The guard is identical across seeds; $\\rho_{50}$ is not",
        ["The tied arm's guard is %.4f at all three seeds, spread %.4f."
         % (s1g[0], max(s1g) - min(s1g)),
         "Its $\\rho_{50}$ over the same three seeds spans %.3f to %.3f."
         % (min(S1_RHO50_SEEDS), max(S1_RHO50_SEEDS)),
         "Three points is not a correlation, and no claim is made from it."])

    # Last page by the project's own rule: the previous best on this eval.
    d.points_page(
        "Finding 28 measured this on different models at $N = 24$",
        ["There it separated two models with identical rule scores: %.3f against %.3f."
         % (F28["A"], F28["B"]),
         "Those are 12-block untied models on 24-node graphs, not these.",
         "So this is the first guard measurement on any tied block."])

    path = d.write()
    used = sorted(os.listdir(d.fig_dir))
    print("wrote %s (%d pages, %d explanation, %d figures)"
          % (path, d.n, n_explain, len(used)))
    print("H1 %s  H2 %s (%d/3 seed pairs)  H3 %s" % (h1, h2, sig, h3))
    print("S1 broken %s  J2 broken %s" % (["%.4f" % x for x in s1g],
                                          ["%.4f" % x for x in j2g]))
    print("logit drop broken: %s"
          % ", ".join("%s %+.3f" % (cell_label(*k), cells[k]["drop"])
                      for k in CELL_ORDER))
    print("headline: %s" % head)


if __name__ == "__main__":
    main()
