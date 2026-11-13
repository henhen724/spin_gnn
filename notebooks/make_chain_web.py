"""Render decks/spin-gnn/chained from the derived-fact probe.

Nothing is re-derived here. Every number comes from scripts/chain_score.py's own
helpers over the per-case CSVs, every bar is imported from the module that owns
it, and check_claims() asserts what the pages print against chain_summary.json,
which the scorer wrote inside the measurement job.

The HEADLINE branches on the registered verdict, including the two branches that
are NOT "it chains": the detection fork, where the model keeps its own answer and
the test is confounded, and the provenance branch, where the rule simply does not
apply to derived facts. All of them were written at 968c355 before the job ran.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_web import WebDeck

from scripts.ddguard_score import ALPHA, ARMS, RULE_BAR, SEEDS, ratio, rows_of, truthy
from scripts.chain_score import GAP_BAR, REVERT_BAR, RUN, cell_dir, round0_rule_flip
from scripts.dissoc_score import FREED_BAR, fmean, sign_test
from scripts.stagev_score import S1_RHO50_SEEDS, S1_TRAINABLE
from scripts.stagej_score import PREDICTED_TRAINABLE

from spin_gnn.sudoku_dissoc import VARIANTS

DECK = "/home/users/hshunt/slides/decks/spin-gnn/chained"
J2_TRAINABLE = PREDICTED_TRAINABLE["J2"]
CELL_ORDER = [(arm, s) for arm in ARMS for s in SEEDS]
# Same order as the same-edit deck, so the two can be flipped between. The
# placebo gloss differs because this deck's placebo is another DEDUCTION, not a
# hint -- that difference is the whole measurement.
VARIANT_GLOSS = {
    "clean": "no edit",
    "placebo": "far deduction recoloured",
    "rule": "$w \\rightarrow c$",
    "broken": "$w \\rightarrow b$",
}


def cell_label(arm, s):
    return "%s s%d" % (arm, s)


def load_cells():
    """{(arm, seed): every quantity this deck quotes}."""
    out = {}
    for arm, s in CELL_ORDER:
        rows = rows_of(os.path.join(cell_dir(arm, s), "chain4_behavior.csv"))
        d = {"rows": rows, "n": len(rows)}
        for v in VARIANTS:
            d["freed_%s" % v] = fmean(rows, "%s_p_freed" % v)[0]
            d["true_%s" % v] = fmean(rows, "%s_p_true" % v)[0]
            d["cand_%s" % v] = fmean(rows, "%s_p_cand" % v)[0]
            d["emits_%s" % v] = fmean(rows, "%s_emits_v" % v)[0]
        # Two denominators, kept apart on purpose, for the reason finding 69's
        # read-back caught twice in one day: the figures use the unconditional
        # rate so the four variants share a denominator, the tables use the
        # conditional one and every header that shows it says "broken | clean".
        d["emits_cond_broken"], d["clean_n"] = ratio(rows, "clean_emits_v",
                                                     "broken_emits_v")
        d["rule_flips"] = fmean(rows, "rule_flips")[0]
        d["rule_reverts"] = fmean(rows, "rule_reverts")[0]
        b, c, nn, p, med = sign_test(rows, "broken_p_freed", "placebo_p_freed")
        d.update({"b": b, "c": c, "sign_n": nn, "p": p, "median": med})
        d["dissoc"] = fmean(rows, "dissoc")[0]
        d["drift"] = d["freed_placebo"] - d["freed_clean"]
        # C6: the same rate, split by how visibly wrong the edit is
        one = [r for r in rows if int(r["rule_clashes"]) == 1]
        many = [r for r in rows if int(r["rule_clashes"]) >= 2]
        d["flip_clash1"] = fmean(one, "rule_flips")[0] if one else float("nan")
        d["flip_clash2"] = fmean(many, "rule_flips")[0] if many else float("nan")
        d["flip_clash1_n"], d["flip_clash2_n"] = len(one), len(many)
        d["round0"], d["round0_n"] = round0_rule_flip(arm, s)
        d["gap"] = d["round0"] - d["rule_flips"]
        # the depth profile: rule-flip rate by the round v was deduced in
        by = {}
        for r in rows:
            by.setdefault(int(r["round"]), []).append(truthy(r["rule_flips"]))
        d["by_round"] = {k: (sum(v) / len(v), len(v)) for k, v in by.items()}
        out[(arm, s)] = d
    return out


def check_claims(cells):
    """Refuse to render a deck that disagrees with the scorer's own appendix."""
    with open(os.path.join(RUN, "chain_summary.json")) as fh:
        js = json.load(fh)
    for arm, s in CELL_ORDER:
        k, c = "%s_s%d" % (arm, s), cells[(arm, s)]
        j = js["cells"][k]
        for key, mine in (("rule_flips", c["rule_flips"]),
                          ("rule_reverts", c["rule_reverts"]),
                          ("emits", c["emits_broken"]),
                          ("emits_cond", c["emits_cond_broken"]),
                          ("placebo_minus_clean", c["drift"]),
                          ("round0_rule_flips", c["round0"]),
                          ("gap", c["gap"])):
            assert abs(j[key] - mine) < 1e-12, "%s disagrees at %s" % (key, k)
        assert j["flip_clash1_n"] == c["flip_clash1_n"], "clash split n at %s" % k
        assert j["flip_clash2_n"] == c["flip_clash2_n"], "clash split n at %s" % k
        for v in VARIANTS:
            assert abs(j["%s_p_freed" % v] - c["freed_%s" % v]) < 1e-12, \
                "p_freed(%s) disagrees at %s" % (v, k)
            assert abs(j["%s_p_cand" % v] - c["cand_%s" % v]) < 1e-12, \
                "p_cand(%s) disagrees at %s" % (v, k)
        assert j["sign_b"] == c["b"] and j["sign_c"] == c["c"], \
            "sign test disagrees at %s" % k
    assert js["n_cases"] == cells[("S1", 0)]["n"]
    assert abs(js["rule_bar"] - RULE_BAR) < 1e-12
    assert abs(js["freed_bar"] - FREED_BAR) < 1e-12
    assert abs(js["gap_bar"] - GAP_BAR) < 1e-12
    return js


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _stack(ax, ends, labels, colors, x, span=0.085, lo=0.10, hi=0.98):
    """Right-hand labels at guaranteed-separated heights, ordered by their own y.

    Fixed dy offsets collided on the same-edit deck when two endpoints happened
    to be 0.06 apart, so the separation is computed rather than assumed.
    """
    base = min(max(sum(ends) / len(ends), lo), hi)
    rank = {i: r for r, i in enumerate(sorted(range(len(ends)), key=ends.__getitem__))}
    for i, text in enumerate(labels):
        label_line(ax, x, base + (rank[i] - 1) * span, text, color=colors[i])


def draw_variants(ax, cells, arm):
    """p_freed across the four one-token variants, one line per seed."""
    xs = list(range(len(VARIANTS)))
    series = [[cells[(arm, s)]["freed_%s" % v] for v in VARIANTS] for s in SEEDS]
    for i, s in enumerate(SEEDS):
        ax.plot(xs, series[i], "o-", ms=12 - 3 * i, lw=2, color=SERIES[i], zorder=3 + i)
    _stack(ax, [y[-1] for y in series], ["seed %d" % s for s in SEEDS],
           list(SERIES[:3]), xs[-1])
    ax.axhline(0.0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([VARIANT_GLOSS[v] for v in VARIANTS], fontsize=11)
    ax.set_xlim(-0.35, len(VARIANTS) + 0.55)
    ax.set_ylim(-0.06, 1.08)
    style_axes(ax, xlabel=None, ylabel="$p_{\\mathrm{freed}}$")


def draw_provenance(ax, cells):
    """The headline: rule-flip rate on a DERIVED fact against a GIVEN one."""
    xs = list(range(len(CELL_ORDER)))
    given = [cells[k]["round0"] for k in CELL_ORDER]
    derived = [cells[k]["rule_flips"] for k in CELL_ORDER]
    for ys, color in ((given, SERIES[0]), (derived, SERIES[1])):
        ax.plot(xs[:3], ys[:3], "o-", ms=11, lw=2, color=color, zorder=3)
        ax.plot(xs[3:], ys[3:], "o-", ms=11, lw=2, color=color, zorder=3)
    _stack(ax, [given[-1], derived[-1]],
           ["given fact (round 0)", "derived fact (chained)"],
           [SERIES[0], SERIES[1]], xs[-1], span=0.10)
    ax.axhline(RULE_BAR, color=MUTED, lw=1.5, ls="--", zorder=2)
    label_line(ax, xs[-1], RULE_BAR, "bar %.2f" % RULE_BAR, color=MUTED, dy=9)
    ax.axhline(0.25, color=MUTED, lw=1, ls=":", zorder=1)
    label_line(ax, xs[-1], 0.25, "chance 0.25", color=MUTED, dy=9)
    ax.axvline(2.5, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([cell_label(*k) for k in CELL_ORDER], fontsize=11)
    ax.set_xlim(-0.4, len(CELL_ORDER) + 1.4)
    ax.set_ylim(-0.04, 1.08)
    style_axes(ax, xlabel=None, ylabel="rule edit flips the answer")


def draw_rounds(ax, cells):
    """Rule-flip rate by the round v was deduced in, tied arm, three seeds."""
    ks = sorted(set(k for s in SEEDS for k in cells[("S1", s)]["by_round"]))
    ks = [k for k in ks if min(cells[("S1", s)]["by_round"].get(k, (0, 0))[1]
                               for s in SEEDS) >= 10]
    series = [[cells[("S1", s)]["by_round"][k][0] for k in ks] for s in SEEDS]
    for i, s in enumerate(SEEDS):
        ax.plot(ks, series[i], "o-", ms=12 - 3 * i, lw=2, color=SERIES[i], zorder=3 + i)
    _stack(ax, [y[-1] for y in series], ["seed %d" % s for s in SEEDS],
           list(SERIES[:3]), ks[-1])
    ax.axhline(RULE_BAR, color=MUTED, lw=1.5, ls="--", zorder=2)
    label_line(ax, ks[-1], RULE_BAR, "bar %.2f" % RULE_BAR, color=MUTED, dy=9)
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.4, ks[-1] + 1.3)
    ax.set_ylim(-0.04, 1.08)
    style_axes(ax, xlabel="the round $v$ was deduced in",
               ylabel="rule edit flips the answer")


def draw_positions(ax, cells):
    """The two readouts under the SAME broken edit, on one 0-1 scale."""
    xs = list(range(len(CELL_ORDER)))
    series = (("names $v$ (node position)", "emits_broken", SERIES[0]),
              ("$p_{\\mathrm{freed}}$ (colour position)", "freed_broken", SERIES[1]))
    ends = []
    for i, (label, key, color) in enumerate(series):
        ys = [cells[k][key] for k in CELL_ORDER]
        ax.plot(xs[:3], ys[:3], "o-", ms=12 - 3 * i, lw=2, color=color, zorder=3 + i)
        ax.plot(xs[3:], ys[3:], "o-", ms=12 - 3 * i, lw=2, color=color, zorder=3 + i)
        ends.append(ys[-1])
    _stack(ax, ends, [s[0] for s in series], [s[2] for s in series], xs[-1], span=0.10)
    ax.axvline(2.5, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([cell_label(*k) for k in CELL_ORDER], fontsize=11)
    ax.set_xlim(-0.4, len(CELL_ORDER) + 1.6)
    ax.set_ylim(-0.06, 1.08)
    style_axes(ax, xlabel=None, ylabel="probability")


# --------------------------------------------------------------------------- #
# this deck's own explanation pages
# --------------------------------------------------------------------------- #
PROVENANCE = """## A given fact is a hint; a derived fact is one the model wrote

Direct deduction eliminates a colour for node $v$ when a neighbour of $v$ is
known to hold it. Where that neighbour's colour came from is its **provenance**,
and there are exactly two possibilities:

- **given** — the neighbour is a **hint**, printed in the prompt. The model reads
  it.
- **derived** — the neighbour's colour is one the model itself emitted earlier in
  the reasoning section, as the conclusion of an earlier round.

Findings 68 and 69 only ever edited **given** facts: both of their edits require
every known neighbour of $v$ to be a hint, which restricts them to deductions
that fire in **round 0**. Everything they established is therefore about
applying the rule once, to what the model was handed.

Running the algorithm means something more: eliminating against your own
conclusions. This deck edits a **derived** fact instead, and changes nothing
else.
"""

ROUNDK = """## A round-$k$ deduction rests on colours produced in rounds before it

One **round** of direct deduction resolves every node whose known neighbours
already cover $q - 1 = 3$ of the four colours, all at once, and then those new
colours become available to the next round.

- A node resolved in **round 0** is resolved from hints alone.
- A node $v$ resolved in **round $k \\ge 1$** has at least one neighbour whose
  colour was itself deduced in some round $< k$.

The **reasoning section** of the sequence is where those deductions are written,
as (node, colour) pairs grouped by `BEGIN_ROUND`. Editing a colour there is
editing a fact the model produced, which is the one thing this measurement does
that no earlier one did.

Cases here run from round 1 upward; the round each case sits in is recorded and
one page plots the result against it.
"""

QUAD = """## Four one-token variants, three of them at the same reasoning index

Let $v$ be a node deduced in round $k \\ge 1$, $c$ its forced colour, $w$ the
single visible neighbour holding colour $a$ — **and $w$ a node the model
deduced, not a hint** — and $b$ another colour already covering $v$.

A **candidate** for $v$ is a colour no visible neighbour of $v$ holds. $v$ is
**forced** when it has exactly one candidate, which is the only situation in
which direct deduction may write a colour for $v$.

| variant | the edit | what direct deduction then says |
| --- | --- | --- |
| clean | none | $v$ is forced to $c$; $a$ is excluded by $w$ |
| placebo | another deduction, far from $v$ | unchanged: $v$ is still forced to $c$ |
| rule | $w$'s colour $a \\rightarrow c$ | $v$ is forced to $a$ instead |
| broken | $w$'s colour $a \\rightarrow b$ | $v$ is **not** forced: candidates $\\{c, a\\}$ |

Every edit is one token, asserted in code, and lands strictly before $v$'s own
node token. **rule** and **broken** rewrite the *same* token; only the value
written differs.
"""

PLACEBOPAGE = """## The placebo is another deduction, so the null matches the edit

A null has to differ from the thing it is a null for in exactly one respect. The
edit this deck measures is *a recoloured deduction in the reasoning section*, so
its placebo is also a recoloured deduction in the reasoning section — a node the
model derived, chosen because it is **not** a neighbour of $v$.

That is a deliberate change from the same-edit deck, whose placebo recoloured a
distant **hint**. A hint placebo here would differ from the real edit in two
ways at once, provenance and relevance, and could not isolate either.

The placebo is also **clash-matched** to the real edit — it clashes with the
same number of already-visible neighbours. The next page explains why that is
not optional, and why it is what makes this null strong.
"""

THEOREM = """## A derived fact cannot be recoloured without a visible clash

Call a colour **clashing** at a node if some neighbour of that node, whose
colour is already visible in the prefix, holds it. The first version of this
measurement required every edit to be **non-clashing**, exactly as the
same-edit deck's edits are. It found **zero** usable cases in 1200 instances.

That is a theorem, not a thin tail. A node $w$ is deduced to colour $a$
*precisely because* its visible neighbours already cover all $q - 1 = 3$ other
colours. So every colour other than $a$ clashes at $w$, by the very fact that
made $w$ deducible, and later rounds only reveal more neighbours:

$$w \\ \\text{deduced to}\\ a \\ \\implies\\ \\text{clashes}(w, x) \\ge 1
\\ \\ \\text{for every}\\ x \\neq a$$

**Editing a derived fact and keeping the prefix locally consistent are mutually
exclusive.** The set of locally consistent one-token edits to the reasoning
section is empty. A hint has no such property, which is why the same-edit deck
never met this.

So the edit does not avoid the clash. It **minimises** it, **records** it, and
**matches the placebo to it**.
"""

BLIND = """## The construction was amended before any model was run

The zero-yield census is a property of the instance generator: it builds the
case list and stops, with **no forward pass and no checkpoint output**. So the
amendment could not have been tuned to any model's behaviour, and nothing that
is scored moved.

What changed: the edit minimises its visible clashes instead of forbidding them,
the count is recorded, the placebo is matched on it, and cases are capped per
**round** rather than per instance so the depth axis spans rounds 1 to 10
instead of being mostly round 1.

What did not change: every bar, every hypothesis, every verdict branch. They
were fixed at commit `968c355` before the first submission and are quoted on a
later page as they were written.

The clash count is also read as a control of its own: if the model were
reacting to *how visibly broken* the prefix looks rather than to *what the edit
says*, the flip rate would depend on the clash count. One page reports that
split.
"""

CAVEAT = """## Editing a derived fact makes the prefix inconsistent, and that is a confound

This is the honest limit of the measurement and it was written down before the
run.

Editing a hint is unremarkable: a different hint is simply a different problem.
Editing a **deduction** shows the model a reasoning step it would not itself
have written — the graph and the hints still imply the original colour, so the
prefix contradicts itself, and by the clash theorem it must do so **visibly**,
right next to the edited node.

A model that *notices* that and keeps its own answer would score as "does not
chain" here while chaining perfectly in free generation. So a **low** flip rate
is ambiguous. Two things bound it:

- `rule_reverts` — how often the model answers $c$, its original value, under
  the rule edit. Above %.2f that is the signature of **detection**; neither
  following nor reverting is the signature of **provenance**.
- the **clash-matched placebo**, which is the same kind of object as the real
  edit — a derived token recoloured, clashing with the same number of visible
  neighbours, just nowhere near $v$. If the model merely reacts to a corrupted
  prefix, the placebo moves too and the whole reading is withdrawn.

A **high** flip rate has no such ambiguity. It can only mean the edited fact was
used.
""" % REVERT_BAR

PAIRING = """## The comparison against round 0 is between cells, not between cases

The number this deck is read against is finding 69's rule-flip rate at the same
six checkpoints, which was **1.0000 everywhere**.

That comparison is **cell-level**. The chained cases are a different case list by
construction — round $k \\ge 1$, with a binding neighbour the model deduced —
so there is no case in common with finding 69 and no paired difference to take.
Same checkpoints, same readout, same bars; different cases.

A gap is therefore called conservatively: the chained rate must be more than
%.2f below the round-0 rate in at least 2 of the 3 seeds of an arm. Anything
smaller is not separated from the two case lists simply being different.
""" % GAP_BAR

PFREED = """## $p_{\\mathrm{freed}}$ is the probability mass on the colour the edit released

Let $z$ be the model's output logits at $v$'s colour position and $a$ the colour
that neighbour $w$ was holding before the edit. Then

$$p_{\\mathrm{freed}} = \\frac{e^{z_a}}{\\sum_{k} e^{z_k}}$$

where $k$ runs over the whole 58-symbol vocabulary, so this is the model's actual
output probability and not a renormalised one. $p_{\\mathrm{true}}$ is the same
quantity at $c$, the colour $v$ was forced to before the edit, and
$p_{\\mathrm{cand}} = p_{\\mathrm{true}} + p_{\\mathrm{freed}}$ is the mass on
$v$'s candidate set after the broken edit.

Under direct deduction the predicted values are exact:

| variant | $p_{\\mathrm{freed}}$ | because |
| --- | --- | --- |
| clean | $0$ | $a$ is held by $w$, so $a$ is eliminated |
| placebo | $0$ | the edit has nothing to do with $v$ |
| rule | $1$ | $a$ is the unique surviving colour |
| broken | $>0$ | $a$ is live again and shares with $c$ |
"""

POSITIONS = """## The two readouts are adjacent tokens in one forward pass

The reasoning section is a sequence of (node, colour) pairs, so the deduction
about $v$ gives two positions one apart:

- the **node position** is the token *before* $v$'s node token; the next-token
  distribution there is *which node the model names next*. That is where a
  determinacy check would act, by naming someone else.
- the **colour position** is $v$'s node token itself; the next-token
  distribution there is *what colour it gives $v$*. That is where the
  elimination rule shows.

$v$'s node token is physically present in all four variants, because every edit
is to a colour written earlier in the prefix. So the colour readout is **teacher
forced** on "the node is $v$": the model is not asked to choose $v$, it is told
$v$ and asked for the colour. That is exactly the conditional that direct
deduction's rule computes.
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

Committed at `968c355`, before the job was submitted.

- **C1 — chaining, the headline.** The rule edit flips the emitted colour to $a$
  in at least %.2f of cases, in all six cells, against chance $1/q = 0.250$.
  Round 0 scored 1.0000 in all six.
- **C2 — the dissociation,** tied model only, exactly as finding 69 registered
  it: $p_{\\mathrm{freed}}$ under the broken edit exceeds it under the placebo,
  Holm-corrected sign test **and** a median difference of at least %.2f, in at
  least 2 of 3 seeds.
- **C3 — the falsifier, which is a fork.** If C1 misses, `rule_reverts` decides
  between *detection* (the model keeps its own answer, so the test is
  confounded and finding 69 stays round-0 only) and *provenance* (the rule does
  not apply to derived facts).
- **C4 — specificity.** The placebo must move $p_{\\mathrm{freed}}$ by at most
  %.2f, otherwise C2 has no null and nothing else is read.
- **C5 — the provenance gap.** Chained against round-0 rule-flip at the same
  checkpoint, called only at more than %.2f lower in at least 2 of 3 seeds.
- **C6 — the clash control,** added with the amendment and model-blind like the
  rest of it. The flip rate split by how many visible neighbours the edit
  clashes with. Descriptive, not a gate.
""" % (RULE_BAR, FREED_BAR, FREED_BAR, GAP_BAR)


def arms_page():
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


SCOPE = """## What this measurement does not establish

- **Nothing was trained.** No model changed, $\\rho_{50}$ did not move, and the
  0.011 token-accuracy gap between the two architectures is exactly where it was.
- **The arms still differ in size as well as in tying,** by a factor of 4.1 in
  trainable parameters. Every difference here is tying *and* budget.
- **The edited prefix is inconsistent, visibly, and no case selection can fix
  that** — the clash theorem says there is nothing to select. `rule_reverts`
  and the clash-matched placebo bound the confound; they do not remove it.
- **Not paired with round 0.** Different case list by construction, so the
  comparison is between two rates at one checkpoint.
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
    js = check_claims(cells)
    n = cells[("S1", 0)]["n"]
    verdict = js["verdict"]

    head = {
        "no_null": "An irrelevant edit moves the readout, so nothing is measurable",
        "chains": "The elimination rule runs over facts the model derived itself",
        "chains_weaker": "The rule follows a derived fact less often than a given one",
        "detection": "The model notices the edited prefix, so chaining stays unmeasured",
        "provenance": "The learned rule applies to given facts, not to derived ones",
    }[verdict]

    prov = ("Job %s, commit %s. Six existing checkpoints read four times each over "
            "%d chained cases, rounds %d-%d, from the same instance stream as "
            "findings 68 and 69. Registered by 968c355 before the job ran. "
            "Nothing trained." % (args.job, args.commit, n,
                                  js["round_min"], js["round_max"]))

    d = WebDeck(args.out, head, "hshunt", "2026-11-13", prov)
    d.title_page(head, subtitle=(
        "Finding 69 showed the tied model eliminates against a fact it was GIVEN. "
        "Direct deduction is iterative, so this asks the same question about a "
        "fact the model DERIVED, changing only which token is edited."))

    d.points_page(
        "Nothing is trained and no published number can change",
        ["Six checkpoints that already existed are read four times each.",
         "The gate, the free line and every stage result are untouched.",
         "One token differs between variants, asserted in code."])

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
                     note="Every edit in this deck rewrites one colour token "
                          "inside the reasoning section of a prompt like this.")
            page_trace(d, ex)
        else:
            d._add(page)

    own = (PROVENANCE, ROUNDK, QUAD, PLACEBOPAGE, THEOREM, BLIND, POSITIONS,
           PFREED, SIGNTEST, CAVEAT, PAIRING, hypotheses_page(), arms_page())
    for page in own:
        d._add(page)
    n_explain = (len([p for p in shared if p is not deck_explain.FIGURE_SLOT])
                 + 2 + len(own))

    # ---- C4, the null, before anything that needs it
    d.table_page(
        "An edit far from $v$ moves the freed colour by thousandths",
        ["cell", "clean", "placebo", "difference"],
        [[cell_label(*k), "%.4f" % cells[k]["freed_clean"],
          "%.4f" % cells[k]["freed_placebo"], "%+.4f" % cells[k]["drift"]]
         for k in CELL_ORDER],
        note=("Mean $p_{\\mathrm{freed}}$ over %d cases. The placebo is clash-matched "
              "to the real edit in %.4f of them; the bar is %.2f."
              % (n, js["clash_matched"], FREED_BAR)))

    # ---- C1, the headline.
    # Every title below states what its own numbers show, so each one branches
    # on the measurement. None of them can be a label for a table.
    d.table_page(
        "Every cell clears the bar on a fact the model derived"
        if js["C1"] else
        "The rule edit misses the bar in at least one cell",
        ["cell", "chained", "round 0", "gap"],
        [[cell_label(*k), "%.4f" % cells[k]["rule_flips"],
          "%.4f" % cells[k]["round0"], "%+.4f" % cells[k]["gap"]]
         for k in CELL_ORDER],
        note=("Bar %.2f, chance 0.250. A gap counts at more than %.2f in two "
              "seeds of an arm." % (RULE_BAR, GAP_BAR)))

    d.claim_page(
        "Provenance %s the rate at which the rule is applied"
        % ("changes" if js["C5_gap_arms"] else "does not change"),
        lambda ax: draw_provenance(ax, cells), "provenance",
        note=("Rule-flip rate at the same six checkpoints. The round-0 series is "
              "finding 69's, over its own case list."))

    d.claim_page(
        "The rule still fires on deductions made late in the trace"
        if js["C1"] else "Depth does not rescue the rule edit",
        lambda ax: draw_rounds(ax, cells), "rounds",
        note=("Tied model, rule-flip rate split by the round $v$ was deduced in. "
              "Rounds with under 10 cases in a seed are dropped."))

    d.table_page(
        "Visible wrongness tracks the flip rate"
        if js["C6_tracks"] else
        "How visibly wrong the edit looks does not change the answer",
        ["cell", "1 clash", "$n$", "2+ clashes", "$n$"],
        [[cell_label(*k), "%.4f" % cells[k]["flip_clash1"],
          "%d" % cells[k]["flip_clash1_n"], "%.4f" % cells[k]["flip_clash2"],
          "%d" % cells[k]["flip_clash2_n"]] for k in CELL_ORDER],
        note=("Rule-flip rate split by clash count. A dependence here would mean "
              "the model reads corruption, not content; the bar is %.2f."
              % js["clash_bar"]))

    d.table_page(
        "The model reverts to its own answer under the edit"
        if js["C3_detects"] else
        "The model rarely falls back on the colour it first wrote",
        ["cell", "flips to $a$", "reverts to $c$", "neither"],
        [[cell_label(*k), "%.4f" % cells[k]["rule_flips"],
          "%.4f" % cells[k]["rule_reverts"],
          "%.4f" % max(0.0, 1 - cells[k]["rule_flips"] - cells[k]["rule_reverts"])]
         for k in CELL_ORDER],
        note=("Reverting is the signature of noticing the edited prefix; the "
              "registered bar for that reading is %.2f." % REVERT_BAR))

    # ---- C2, the dissociation, carried over from finding 69
    d.claim_page(
        "One token moves the freed colour in the tied model",
        lambda ax: draw_variants(ax, cells, "S1"), "variants_s1",
        note=("Tied model, mean $p_{\\mathrm{freed}}$ over %d cases. Clean and "
              "placebo sit at the floor." % n))

    same_profile = max(
        abs(sum(cells[("S1", s)]["freed_%s" % v] for s in SEEDS) / 3
            - sum(cells[("J2", s)]["freed_%s" % v] for s in SEEDS) / 3)
        for v in VARIANTS) <= 0.12
    d.claim_page(
        "The untied model lands on the same four values"
        if same_profile else
        "The untied model splits the freed colour differently",
        lambda ax: draw_variants(ax, cells, "J2"), "variants_j2",
        note=("Same axes as the previous page. Its broken column is teacher "
              "forced onto a node it may not have named."))

    d.table_page(
        "The freed colour rises under the broken edit in every seed"
        if js["C2"] else
        "The paired test does not clear the bar in the tied model",
        ["seed", "$b$ up", "$c$ down", "median $\\Delta$", "exact $p$"],
        [["%d" % s, "%d" % cells[("S1", s)]["b"], "%d" % cells[("S1", s)]["c"],
          "%+.4f" % cells[("S1", s)]["median"], "%.3g" % cells[("S1", s)]["p"]]
         for s in SEEDS],
        note=("Tied model only: C2 was registered for it alone. The bar is Holm "
              "$p < %.2f$ with median $\\ge$ %.2f." % (ALPHA, FREED_BAR)))

    d.table_page(
        "Under the broken edit the mass splits over the candidate set"
        if min(cells[k]["cand_broken"] for k in CELL_ORDER) >= 0.90 else
        "Under the broken edit the mass leaves the candidate set",
        ["cell", "$p_{\\mathrm{true}}$", "$p_{\\mathrm{freed}}$", "sum"],
        [[cell_label(*k), "%.4f" % cells[k]["true_broken"],
          "%.4f" % cells[k]["freed_broken"], "%.4f" % cells[k]["cand_broken"]]
         for k in CELL_ORDER],
        note=("$c$ and $a$ are exactly $v$'s candidate set after the edit. The "
              "sum is the mass elimination did not leave elsewhere."))

    node_shift = max(abs(cells[("S1", s)]["emits_broken"]
                         - cells[("S1", s)]["emits_clean"]) for s in SEEDS)
    d.claim_page(
        "The colour readout moves and the node readout does not"
        if node_shift <= 0.05 else
        "The same edit moves both of the two adjacent readouts",
        lambda ax: draw_positions(ax, cells), "positions",
        note=("Both series are under the SAME broken edit. Naming $v$ is over all "
              "%d cases, not the conditional denominator." % n))

    d.table_page(
        "The guard rate depends on which cases you count",
        ["cell", "all %d cases" % n, "broken $\\mid$ clean", "clean cases"],
        [[cell_label(*k), "%.4f" % cells[k]["emits_broken"],
          "%.4f" % cells[k]["emits_cond_broken"],
          "%d of %d" % (cells[k]["clean_n"], n)] for k in CELL_ORDER],
        note=("Column 3 counts only cases where the clean prompt named $v$; "
              "column 2 counts all. Findings 68 and 69 published column 3."))

    d._add(SCOPE)

    closing = {
        "chains": ["The rule fires on a fact the model wrote, not only on hints.",
                   "So finding 69's decomposition is not confined to round 0.",
                   "Chaining is now a causal result, not read off accuracy."],
        "chains_weaker": ["The rule fires on a derived fact, above the bar.",
                          "It fires less often than on a given one.",
                          "Provenance costs something; it does not decide the outcome."],
        "detection": ["The model keeps its own answer under the edited prefix.",
                      "That is detection, so chaining is not measured either way.",
                      "Finding 69 stays round-0 only, as its scope page says."],
        "provenance": ["The rule follows a given fact and not a derived one.",
                       "The edit is neither followed nor reverted.",
                       "Finding 69's rule half is bounded to round 0."],
        "no_null": ["The placebo moved the readout, so there is no null.",
                    "No dissociation is claimed from this run.",
                    "The construction is reported as broken, not the model."],
    }[verdict]
    d.points_page(head, closing)

    path = d.write()
    print("wrote %s (%d pages, %d explanation, %d figures)"
          % (path, d.n, n_explain, len(os.listdir(d.fig_dir))))
    print("verdict %s   C1 %s  C2 %s  C4 %s  detects %s  gap %s"
          % (verdict, js["C1"], js["C2"], js["C4"], js["C3_detects"],
             js["C5_gap_arms"] or "none"))
    print("S1 chained rule flip %s" % ["%.4f" % cells[("S1", s)]["rule_flips"]
                                       for s in SEEDS])
    print("headline: %s" % head)


if __name__ == "__main__":
    main()
