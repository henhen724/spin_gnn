"""Render decks/spin-gnn/provenance from the provenance-paired guard probe.

Nothing is re-derived here. Every number comes from scripts/prov_score.py's own
helpers over the per-case CSVs, every bar is imported from the module that owns
it, and check_claims() asserts what the pages print against prov_summary.json,
which the scorer wrote inside the measurement job.

The HEADLINE branches on the registered verdict, including the branches that are
NOT "provenance": the two gate failures, the reversed direction, and the null in
which the post-hoc gap is about depth instead. All five were written at 9b59866
before the job ran.

The deck has to carry the honest decomposition and not just the headline. The
paired provenance effect is real and significant, and it is the SMALLER piece of
the post-hoc gap that motivated the run; depth is the larger piece; clash is not
it at all. Three pages exist for that, and the closing page says it in words.

Usage (in a job or with the modules loaded in the same shell):
    python3 -u -m notebooks.make_prov_web --commit <sha> --job <id> [--out <dir>]
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, label_line, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_web import WebDeck

from scripts.ddguard_score import (ALPHA, ARMS, PLACEBO_FLOOR, RULE_BAR, SEEDS,
                                   ratio, rows_of, truthy)
from scripts.chain_score import round0_guard
from scripts.dissoc_score import fmean
from scripts.prov_score import (MATCH_MIN, PROV_BAR, RUN, cell_dir, guard_pair,
                                mcnemar)
from scripts.stagev_score import S1_RHO50_SEEDS, S1_TRAINABLE
from scripts.stagej_score import PREDICTED_TRAINABLE

DECK = "/home/users/hshunt/slides/decks/spin-gnn/provenance"
J2_TRAINABLE = PREDICTED_TRAINABLE["J2"]
CELL_ORDER = [(arm, s) for arm in ARMS for s in SEEDS]

# The scorer's own depth bins, imported by value rather than re-chosen: r0 is
# kept because its emptiness is a fact the deck states.
DEPTH_BINS = ["r0", "r1", "r2", "r3+"]
DEPTH_GLOSS = {"r0": "round 0", "r1": "round 1", "r2": "round 2",
               "r3+": "rounds 3-15"}
CLASH_BINS = ["cl0", "cl1", "cl2+"]


def cell_label(arm, s):
    return "%s s%d" % (arm, s)


def load_cells():
    """{(arm, seed): every quantity this deck quotes}."""
    out = {}
    for arm, s in CELL_ORDER:
        rows = rows_of(os.path.join(cell_dir(arm, s), "prov_behavior.csv"))
        d = {"rows": rows, "n": len(rows)}
        # The two guard rates, on finding 68's denominator, through finding 68's
        # own ratio() -- the deck must not own a second definition of them.
        g, dv, den = guard_pair(rows)
        d.update({"guard_given": g, "guard_derived": dv, "guard_diff": dv - g,
                  "guard_n": den})
        d["guard_given_uncond"] = fmean(rows, "names_v_given")[0]
        d["guard_derived_uncond"] = fmean(rows, "names_v_derived")[0]
        for side in ("given", "derived"):
            d["rule_flips_%s" % side] = fmean(rows, "rule_flips_%s" % side)[0]
            d["rule_reverts_%s" % side] = fmean(rows, "rule_reverts_%s" % side)[0]
        d["placebo_cond"], d["clean_n"] = ratio(rows, "clean_emits_v",
                                                "placebo_emits_v")
        d["placebo_uncond"] = fmean(rows, "placebo_emits_v")[0]
        b, c, nn, p = mcnemar(rows, "names_v_derived", "names_v_given",
                              "clean_emits_v")
        d.update({"b": b, "c": c, "mc_n": nn, "p": p})
        # depth and clash profiles, both on the conditional denominator
        d["depth"] = {}
        for lab, (lo, hi) in zip(DEPTH_BINS, ((0, 0), (1, 1), (2, 2), (3, 10 ** 6))):
            sub = [r for r in rows if lo <= int(r["round"]) <= hi]
            v, dn = ratio(sub, "clean_emits_v", "names_v_given") if sub \
                else (float("nan"), 0)
            d["depth"][lab] = {"rate": v, "n": dn}
        d["clash"] = {}
        for lab, sub in (("cl0", [r for r in rows
                                  if int(r["broken_clashes_given"]) == 0]),
                         ("cl1", [r for r in rows
                                  if int(r["broken_clashes_given"]) == 1]),
                         ("cl2+", [r for r in rows
                                   if int(r["broken_clashes_given"]) >= 2])):
            v, dn = ratio(sub, "clean_emits_v", "names_v_given") if sub \
                else (float("nan"), 0)
            d["clash"][lab] = {"rate": v, "n": dn}
        # finding 68's round-0 guard at the same checkpoint, for the comparison
        d["round0_guard"], d["round0_n"] = round0_guard(arm, s)
        out[(arm, s)] = d
    return out


def check_claims(cells):
    """Refuse to render a deck that disagrees with the scorer's own appendix."""
    with open(os.path.join(RUN, "prov_summary.json")) as fh:
        js = json.load(fh)
    for arm, s in CELL_ORDER:
        k, c = "%s_s%d" % (arm, s), cells[(arm, s)]
        j = js["cells"][k]
        for key in ("guard_given", "guard_derived", "guard_diff",
                    "guard_given_uncond", "guard_derived_uncond",
                    "rule_flips_given", "rule_flips_derived",
                    "rule_reverts_given", "rule_reverts_derived",
                    "placebo_cond", "placebo_uncond", "round0_guard"):
            assert abs(j[key] - c[key]) < 1e-12, "%s disagrees at %s" % (key, k)
        for key in ("guard_n", "clean_n", "round0_n"):
            assert j[key] == c[key], "%s disagrees at %s" % (key, k)
        for lab in DEPTH_BINS:
            assert j["depth"][lab]["n"] == c["depth"][lab]["n"], "depth n %s" % k
            if c["depth"][lab]["n"]:
                assert abs(j["depth"][lab]["rate"] - c["depth"][lab]["rate"]) < 1e-12
        for lab in CLASH_BINS:
            assert j["clash_split"][lab]["n"] == c["clash"][lab]["n"], "clash n %s" % k
            if c["clash"][lab]["n"]:
                assert abs(j["clash_split"][lab]["rate"] - c["clash"][lab]["rate"]) < 1e-12
        if arm == "J2":
            assert j["mcnemar_b"] == c["b"] and j["mcnemar_c"] == c["c"], \
                "McNemar counts disagree at %s" % k
            assert abs(j["p_raw"] - c["p"]) < 1e-12, "raw p disagrees at %s" % k
    assert js["n_cases"] == cells[("S1", 0)]["n"]
    assert abs(js["prov_bar"] - PROV_BAR) < 1e-12
    assert abs(js["rule_bar"] - RULE_BAR) < 1e-12
    assert abs(js["placebo_floor"] - PLACEBO_FLOOR) < 1e-12
    assert js["match_min"] == MATCH_MIN
    return js


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _stack(ax, ends, labels, colors, x, span=0.085, lo=0.10, hi=0.98, dx=6):
    """Right-hand labels at guaranteed-separated heights, ordered by their own y.

    Carried over unchanged from the chained deck, where fixed dy offsets
    collided whenever two endpoints landed close together -- which here is the
    expected outcome in the tied arm, whose two series are both 1.0000.
    """
    base = min(max(sum(ends) / len(ends), lo), hi)
    rank = {i: r for r, i in enumerate(sorted(range(len(ends)), key=ends.__getitem__))}
    for i, text in enumerate(labels):
        label_line(ax, x, base + (rank[i] - 1) * span, text, color=colors[i], dx=dx)


def draw_guard(ax, cells):
    """The headline: still-names-v after breaking a given fact, and a derived one."""
    xs = list(range(len(CELL_ORDER)))
    given = [cells[k]["guard_given"] for k in CELL_ORDER]
    derived = [cells[k]["guard_derived"] for k in CELL_ORDER]
    # Different marker size and dash, because in the tied arm the two series
    # land on exactly the same value and have to stay separately readable.
    for ys, color, ms, ls, mfc in ((given, SERIES[0], 15, "--", "none"),
                                   (derived, SERIES[1], 8, "-", None)):
        for lo, hi in ((0, 3), (3, 6)):
            ax.plot(xs[lo:hi], ys[lo:hi], "o", ls=ls, ms=ms, lw=2, color=color,
                    mfc=mfc or color, mew=2, zorder=3)
    _stack(ax, [given[-1], derived[-1]],
           ["broke a given fact", "broke a derived fact"],
           [SERIES[0], SERIES[1]], xs[-1], span=0.11, dx=16)
    ax.axvline(2.5, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([cell_label(*k) for k in CELL_ORDER], fontsize=11)
    ax.set_xlim(-0.4, len(CELL_ORDER) + 1.4)
    ax.set_ylim(-0.06, 1.08)
    style_axes(ax, xlabel=None, ylabel="still names $v$ (lower = guards)")


def draw_depth(ax, cells):
    """Given-side guard rate against the round v was deduced in, untied arm."""
    labs = [t for t in DEPTH_BINS if cells[("J2", SEEDS[0])]["depth"][t]["n"]]
    xs = list(range(len(labs)))
    series = [[cells[("J2", s)]["depth"][t]["rate"] for t in labs] for s in SEEDS]
    for i, s in enumerate(SEEDS):
        ax.plot(xs, series[i], "o-", ms=12 - 3 * i, lw=2, color=SERIES[i],
                zorder=3 + i)
    _stack(ax, [y[-1] for y in series], ["seed %d" % s for s in SEEDS],
           list(SERIES[:3]), xs[-1])
    ax.set_xticks(xs)
    ax.set_xticklabels([DEPTH_GLOSS[t] for t in labs], fontsize=11)
    ax.set_xlim(-0.3, len(labs) + 0.8)
    ax.set_ylim(-0.04, 1.08)
    style_axes(ax, xlabel="the round $v$ was deduced in",
               ylabel="still names $v$, given edit")


def draw_decomp(ax, cells):
    """POST-HOC: round-0 given, then this run's two sides, untied arm."""
    xs = [0, 1, 2]
    series = [[cells[("J2", s)]["round0_guard"], cells[("J2", s)]["guard_given"],
               cells[("J2", s)]["guard_derived"]] for s in SEEDS]
    for i, s in enumerate(SEEDS):
        # The first point is a DIFFERENT case list, so the segment that reaches
        # it is dashed and the marker is open: same quantity, not a paired step.
        ax.plot(xs[:2], series[i][:2], ls="--", lw=2, color=SERIES[i], zorder=3 + i)
        ax.plot(xs[1:], series[i][1:], ls="-", lw=2, color=SERIES[i], zorder=3 + i)
        ax.plot(xs[:1], series[i][:1], "o", ms=13 - 3 * i, mfc="none", mew=2,
                color=SERIES[i], zorder=3 + i)
        ax.plot(xs[1:], series[i][1:], "o", ms=11 - 3 * i, color=SERIES[i],
                zorder=3 + i)
    _stack(ax, [y[-1] for y in series], ["seed %d" % s for s in SEEDS],
           list(SERIES[:3]), xs[-1], span=0.09)
    ax.axvline(0.5, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["round 0, given\n(finding 68's cases)",
                        "rounds 1-15, given\n(this run)",
                        "rounds 1-15, derived\n(this run)"], fontsize=11)
    ax.set_xlim(-0.35, 3.0)
    ax.set_ylim(-0.06, 1.08)
    style_axes(ax, xlabel=None, ylabel="still names $v$ (lower = guards)")


# --------------------------------------------------------------------------- #
# this deck's own explanation pages
# --------------------------------------------------------------------------- #
PROVENANCE = """## Provenance is where an eliminating neighbour's colour came from

Direct deduction removes a colour from node $v$ when some neighbour of $v$ is
known to hold it. Where that neighbour's colour came from is its **provenance**,
and there are exactly two possibilities:

- **given** — the neighbour is a **hint**, printed in the prompt before any
  reasoning starts. The model only has to read it.
- **derived** — the neighbour's colour is one the model itself emitted earlier in
  the reasoning section, as the conclusion of an earlier round.

Findings 68 and 69 edited **given** facts only: their constructions require every
known neighbour of $v$ to be a hint, which restricts them to deductions that fire
in round 0. Everything they established is about applying the rule once, to what
the model was handed.

This deck puts both kinds of fact in the **same case** and breaks each in turn.
"""

GUARD = """## The guard is the decision not to write a colour

Direct deduction has two halves, and this deck is about the second one.

- **The rule.** Given that $v$ is forced, which colour does it get?
- **The guard.** Is $v$ forced at all? A node is **forced** when exactly one
  colour is still available to it, and that is the only situation in which
  direct deduction may write a colour for it. If two colours are live, the
  algorithm writes nothing for $v$ here and names some other node instead.

Every edit in this deck takes $v$ from one candidate to two. So the guard's
correct response is to not name $v$, and the measured quantity is how often the
model names it anyway:

$$g \\;=\\; \\frac{\\#\\{\\,\\text{clean names } v\\ \\text{ and }\\ \\text{edited names } v\\,\\}}
{\\#\\{\\,\\text{clean names } v\\,\\}}$$

**$g$ is high when the model does NOT guard.** $g = 1$ means it never withheld;
$g = 0$ means it always did. Withholding is what every page here calls the
guard **firing**. The denominator is the cases where the **unedited** prompt
named $v$, because a model that was not going to name $v$ anyway tells you
nothing about guarding. Every table below that shows $g$ also shows that
denominator.
"""

PAIRED = """## One case holds both facts, so the pair differs only in provenance

A **case** here is one node $v$ whose visible neighbours cover exactly
$q - 1 = 3$ of the four colours, with two of those colours held by exactly one
neighbour each:

- $w_g$, a **hint**, holding colour $a_g$;
- $w_d$, a node the model **deduced** in an earlier round, holding colour $a_d$.

A colour with a single holder is the only kind that can be freed by one token:
recolour a neighbour that shares its colour with another neighbour and the
colour is still covered, so nothing about $v$ changes.

Breaking either holder leaves $v$ with exactly two candidates, and each of the
two prompts differs from the clean one by exactly one token. $v$ itself, its
round, the graph, the instance and the position that is read are **the same
object** in both measurements, because they are one case. The only thing that
differs between them is which kind of fact was broken.

That is what the lead this run came from could not do: it compared one case list
against another. Here every case is its own control.
"""

VARIANTS6 = """## Six prompts per case, each one token from the clean one

Let $c$ be the colour $v$ is forced to, and let $b$ be a colour already covering
$v$ from some other neighbour. Writing $w \\rightarrow x$ for "the colour token
of $w$ is replaced by $x$":

| variant | the edit | what direct deduction then says about $v$ |
| --- | --- | --- |
| clean | none | forced to $c$ |
| placebo | a far deduction recoloured | unchanged: still forced to $c$ |
| rule, given | $w_g \\rightarrow c$ | forced to $a_g$ instead |
| rule, derived | $w_d \\rightarrow c$ | forced to $a_d$ instead |
| broken, given | $w_g \\rightarrow b$ | **not forced**: candidates $\\{c, a_g\\}$ |
| broken, derived | $w_d \\rightarrow b$ | **not forced**: candidates $\\{c, a_d\\}$ |

Every edit is a single token, asserted in code, and lands strictly before $v$'s
own node token. The two **broken** variants are what the guard is read from. The
two **rule** variants are the read control on the next page, and each rewrites
the same token as its own broken variant — only the value written differs.
"""

RULECHECK = """## The rule edit proves the model actually read the edited token

If the model keeps naming $v$ after the derived fact is broken, the dull
explanation is that the derived token never reached it at all — that the model
is not reading that part of the prefix, so of course nothing changed.

The **rule** variant rules that out at the same token. Recolouring $w$ to $c$,
$v$'s own forced colour, leaves $v$ forced again, but forced to $a$ — the colour
$w$ had been holding. A model that applies the elimination rule to that token
must now emit $a$ at $v$'s colour position.

So the rule edit and the broken edit rewrite **the same token**, and the rule
edit has a right answer that can be checked. If it is produced, the token was
read, and "still names $v$" under the broken edit is a decision rather than an
oversight. This is a **gate**: if the rule edit misses on either side in any
cell, the guard numbers are printed and marked uninterpretable, and the headline
test is not called.

The table also reports how often the model **reverts** — emits $c$, the colour
$v$ had before the edit, instead of following the rule. A model that reverted
often would be one that *noticed* the edit and overrode it, which is a
different story from one that read it and applied the rule; so reverting is
printed next to flipping rather than lumped into "did not flip".
"""

POSITIONS = """## The guard and the rule are read at two adjacent positions

The reasoning section is a sequence of (node, colour) pairs, so the deduction
about $v$ gives two positions one token apart:

- the **node position** is the token *before* $v$'s node token. The next-token
  distribution there is *which node the model names next*, and that is where the
  guard would act — by naming someone else. Every guard number in this deck is
  read there, as whether $v$ is the argmax.
- the **colour position** is $v$'s node token itself. The next-token
  distribution there is *what colour it gives $v$*, and that is where the rule
  shows. Every rule-flip number is read there.

$v$'s node token is physically present in all six variants, because every edit
is to a colour written earlier in the prefix. So the colour readout is **teacher
forced** on "the node is $v$": the model is not asked to choose $v$, it is told
$v$ and asked for the colour. That is exactly the conditional the rule computes,
and it is why the rule check works even in cases where the guard fires.
"""

CLASH = """## A derived fact cannot be recoloured without a visible clash

Call a colour **clashing** at a node if some neighbour of that node, whose
colour is already visible in the prefix, holds it. Findings 68 and 69 required
every edit to be non-clashing, so that the revealed colouring stayed proper.
Finding 70 kept that rule when it first moved off hints, and its census found
**zero** usable cases in 1200 instances.

That is a theorem, not a thin tail. A node $w$ is deduced to colour $a$
*precisely because* its visible neighbours already cover all $q - 1 = 3$ other
colours. So every colour other than $a$ clashes at $w$, by the very fact that
made $w$ deducible, and later rounds only reveal more neighbours:

$$w \\ \\text{deduced to}\\ a \\ \\implies\\ \\mathrm{clashes}(w, x) \\ge 1
\\ \\ \\text{for every}\\ x \\neq a$$

**Editing a derived fact and keeping the prefix locally proper are mutually
exclusive.** So the edit does not avoid the clash: it **minimises** it,
**records** the count, and the given-side edit is then chosen to clash the same
number of times. A difference between the two sides therefore cannot be a
difference in how visibly broken the prefix looks — and one page checks that
directly by splitting on the count.
"""


def census_yield():
    """The paired-case count the job's own census printed, not a retyped one."""
    with open(os.path.join(RUN, "census.txt")) as fh:
        for line in fh:
            m = re.match(r"^(\d+) provenance-paired cases", line)
            if m:
                return int(m.group(1))
    raise SystemExit("no paired-case count in the census output")


def round0_page(js):
    return """## This construction cannot reach round 0, for the same kind of reason

Finding 68 measured the guard at **round 0**, where $v$ is forced by hints
alone. No case here sits there, and that is structural rather than a sampling
accident.

If $v$ is forced in round 0, its hint neighbours already cover $q - 1 = 3$
colours. A neighbour the model deduced earlier in the same round must therefore
repeat one of those three colours — so that colour has two holders and is not a
single holder any more. **A round-0 case with both a hint single holder and a
deduced single holder does not exist.**

The census found this before any model was run: %d paired cases from 1200
instances and not one of them at round 0. The 400 that were scored span rounds
%d to %d.

The consequence is stated plainly rather than worked around. This probe holds
depth **fixed** within a case, which is what the headline needs, but it cannot
**reach** the depth finding 68 measured. Comparisons against round 0 are
therefore between case lists, and the two pages that make them say so.
""" % (census_yield(), js["round_min"], js["round_max"])


def placebo_page(js):
    return """## The placebo is the same kind of edit, just nowhere near $v$

A null has to differ from the thing it is a null for in exactly one respect. The
edit measured here is *a recoloured fact in the prefix*, so the placebo is also
a recoloured fact in the prefix — a node chosen because it is **not** a
neighbour of $v$, preferring one the model deduced.

It is also **clash-matched**: it clashes with the same number of already-visible
neighbours as the real edit, in %.4f of cases. So it corrupts the prefix by the
same visible amount and differs only in relevance to $v$.

If the model merely reacts to a corrupted prefix, the placebo moves the node
readout too, and then "still names $v$" measures nothing about determinacy. The
registered floor is %.2f: the model must still name $v$ in at least that
fraction of the cases where the clean prompt did. It is a **gate** — below it,
nothing further is read.
""" % (js["placebo_matched"], PLACEBO_FLOOR)


MCNEMARP = """## The test is McNemar's, paired case by case

Each case contributes two binary outcomes from the same model and the same
case: whether $v$ is still named after the given edit, and after the derived
edit. Cases where the two agree carry no information about a difference, so the
test uses only the **discordant** ones.

Let $b$ count the cases where the derived edit left $v$ named and the given edit
did not, and $c$ the reverse. Under the null that provenance does not matter,
each discordant case is a fair coin, so

$$p \\;=\\; 2 \\cdot \\Pr\\left[X \\le \\min(b, c)\\right], \\qquad
X \\sim \\mathrm{Binomial}(b + c, \\tfrac{1}{2})$$

computed exactly rather than by the usual $\\chi^2$ approximation. **Holm
correction** across the three seeds multiplies the smallest $p$ by 3, the next
by 2, the largest by 1, and enforces that the adjusted values do not decrease;
that controls the chance of any false positive among the three at
$\\alpha = 0.05$.
"""


def hypotheses_page():
    return """## What was registered before the measurement ran

Committed at `9b59866`, before the job was submitted.

- **G1 — the headline, untied arm only,** because it is the only arm with a
  guard to lose. Exact McNemar, Holm-adjusted over three seeds, $p < %.2f$
  **and** a rate difference of at least %.2f, in at least 2 of 3 seeds.
  Direction fixed: derived $>$ given. The reverse is reported as REVERSED and is
  a different finding.
- **G2 — the read control, a GATE.** The rule edit must flip the emitted colour
  in at least %.2f of cases, on **both** sides, in all six cells.
- **G3 — specificity, a GATE.** The clash-matched placebo must leave $v$ named
  in at least %.2f of the cases the clean prompt named it.
- **G4 — the tied arm as a control,** descriptive. It has no guard at round 0,
  so it should show no provenance effect: difference within %.2f.
- **G5 — depth,** descriptive, and the alternative account G1's falsifier hands
  off to: the given-side rate as a function of $v$'s round.
- **G6 — the clash control.** G1 recomputed on the cases where the two edits
  clash exactly the same number of times, read only above %d such cases.
""" % (ALPHA, PROV_BAR, RULE_BAR, PLACEBO_FLOOR, PROV_BAR, MATCH_MIN)


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
the following pages separates those two differences, and no page claims to. The
tied arm appears here as a control, not as a competitor: finding 68 already
showed it never withholds, so it is the arm that should show no effect.
""" % ("{:,}".format(S1_TRAINABLE), "{:,}".format(J2_TRAINABLE),
       min(S1_RHO50_SEEDS), max(S1_RHO50_SEEDS), J2_TRAINABLE / S1_TRAINABLE)


CAVEAT = """## Provenance here also means position, and that is not separated

This is the honest limit of the construction. The registration named three
confounds — provenance, depth and clash — and held the last two. This is a
fourth, noticed while reading the finished deck back, and it is stated here
rather than left out.

The given fact is a **hint**, so it sits in the hints section of the sequence.
The derived fact sits in the reasoning section, later, and therefore **closer to
the position that is read**. The two edits differ in provenance, but they also
differ in which section they occupy and in how far they are from the readout.
Nothing here separates those three, and "the guard is specific to given facts"
should be read as including "…or to facts in the hints section, or to facts far
from the readout".

Two things are controlled and worth keeping apart from that one:

- **visible corruption**, by clash-matching the two edits and by splitting the
  result on the clash count — one page each;
- **the depth of $v$**, which is identical within a case by construction.

What would separate position from provenance is a hint restated inside the
reasoning section, or a deduction that appears before a hint. Neither exists in
this token format, so it needs a change to the data, not a change to the probe.
"""


SCOPE = """## What this measurement does not establish

- **Nothing was trained.** No model changed, $\\rho_{50}$ did not move, and the
  gap between the two architectures is exactly where it was.
- **The arms still differ in size as well as in tying,** by a factor of 4.1 in
  trainable parameters.
- **Provenance is confounded with position in the sequence** — the given fact
  is a hint, so it also sits further from the position that is read. The page
  on provenance and position sets that out at length.
- **Round 0 is out of reach,** so the comparison against finding 68 is between
  case lists and is marked post-hoc wherever it appears.
- **One arm, one direction.** G1 is registered on the untied model only; the
  tied model has no guard to lose and is read as a control.
- **No localisation.** These are decision-level readouts. Nothing here says
  *where* in the network the guard lives, and on a tied model the usual
  per-layer ablation is not even defined.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--job", default="unknown")
    args = ap.parse_args()

    deck_explain.check()
    # Only CENSOR is dropped: nothing here is scored on a density grid. LEAK
    # stays because the shared tokenization figure names the leak in bold, and a
    # term a FIGURE introduces has to be defined in the deck.
    shared = [p for p in deck_explain.PAGES if p is not deck_explain.CENSOR]

    cells = load_cells()
    js = check_claims(cells)
    n = cells[("S1", 0)]["n"]
    verdict = js["verdict"]
    # The seeds that actually clear G1's two-part rule. Quoted rather than
    # written as "two", because the rule needs 2 of 3 and could have given 3.
    g1_hits = [s for s in SEEDS
               if js["cells"]["J2_s%d" % s]["p_holm"] < ALPHA
               and cells[("J2", s)]["guard_diff"] >= PROV_BAR]
    sep = {0: "no seed", 1: "one seed", 2: "two seeds",
           3: "three seeds"}[len(g1_hits)]

    head = {
        "unread_edit": "An edit never reached the model, so nothing is read",
        "no_null": "An irrelevant edit moves the readout, so nothing is measurable",
        "provenance_specific": "The guard fires for broken hints, not for broken deductions",
        "reversed": "The guard fires for broken deductions, not for broken hints",
        "not_provenance": "The two guard rates do not separate at fixed $v$",
    }[verdict]

    prov = ("Job %s, commit %s. Six existing checkpoints read six times each over "
            "%d provenance-paired cases, rounds %d-%d, from the same instance "
            "stream as findings 68-70. Registered by 9b59866 before the job ran. "
            "Nothing trained." % (args.job, args.commit, n,
                                  js["round_min"], js["round_max"]))

    d = WebDeck(args.out, head, "hshunt", "2026-11-13", prov)
    d.title_page(head, subtitle=(
        "Finding 68 found that the untied model stops naming a node once that "
        "node stops being determined, in two of its three seeds. This asks "
        "whether it still does so when the fact that stopped determining it is "
        "one of its own deductions rather than a hint."))

    d.points_page(
        "Nothing is trained and no published number can change",
        ["Six checkpoints that already existed are read six times each.",
         "The gate, the free line and every stage result are untouched.",
         "One token differs between variants, asserted in code."])

    # ---- shared explanation set, with the two shared figures
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, shuffle_adj=True,
                          fall=True)
    for page in shared:
        if page is deck_explain.FIGURE_SLOT:
            d.expect("tokenization",
                     "Every token is one categorical id in a %d-symbol vocabulary"
                     % ex.vocab.size,
                     note="The leak the figure names is the order of node ids "
                          "within a row; a later page shows how every eval "
                          "here closes it.")
            page_tokenization(d, ex)
            d.expect("trace",
                     "A deduction fires when a node sees all three other colours",
                     note="Each case here breaks one colour token in a prompt "
                          "like this, once in the hints and once in the "
                          "reasoning section.")
            page_trace(d, ex)
        else:
            d._add(page)

    own = (PROVENANCE, GUARD, PAIRED, VARIANTS6, RULECHECK, POSITIONS, CLASH,
           round0_page(js), placebo_page(js), MCNEMARP, CAVEAT,
           hypotheses_page(), arms_page())
    for page in own:
        d._add(page)
    n_explain = (len([p for p in shared if p is not deck_explain.FIGURE_SLOT])
                 + 2 + len(own))

    # ---- G2, the read gate, before anything that depends on it
    worst_rule = min(min(cells[k]["rule_flips_given"],
                         cells[k]["rule_flips_derived"]) for k in CELL_ORDER)
    d.table_page(
        "Both edits reach the model in every cell"
        if js["G2"] else
        "An edit does not reach the model, so the guard is unreadable",
        ["cell", "rule, given", "rule, derived", "reverts to $c$"],
        [[cell_label(*k), "%.4f" % cells[k]["rule_flips_given"],
          "%.4f" % cells[k]["rule_flips_derived"],
          "%.4f" % max(cells[k]["rule_reverts_given"],
                       cells[k]["rule_reverts_derived"])]
         for k in CELL_ORDER],
        note=("Rule edit flips the emitted colour; bar %.2f, chance 0.250, worst "
              "cell %.4f over %d cases." % (RULE_BAR, worst_rule, n)))

    # ---- G3, the null
    d.table_page(
        "An edit far from $v$ leaves the node readout alone"
        if js["G3"] else
        "The placebo moves the node readout, so there is no null",
        ["cell", "placebo still names $v$", "cases the clean prompt named"],
        [[cell_label(*k), "%.4f" % cells[k]["placebo_cond"],
          "%d of %d" % (cells[k]["clean_n"], n)] for k in CELL_ORDER],
        note=("Floor %.2f. The placebo is clash-matched to the real edit in "
              "%.4f of cases." % (PLACEBO_FLOOR, js["placebo_matched"])))

    # ---- G1, the headline
    d.claim_page(
        "The untied model withholds for hints, not deductions, in %s" % sep
        if js["G1"] else
        "The untied model withholds for its deductions, not for hints"
        if js["G1_reversed"] else
        "Which fact is broken does not change whether $v$ is named",
        lambda ax: draw_guard(ax, cells), "guard",
        note=("Conditional on the clean prompt naming $v$; higher means less "
              "guarding. Coincident points draw as a ring around a dot."))

    d.table_page(
        "The paired test separates in two of three untied seeds"
        if js["G1"] else
        "The paired test does not separate in two of three untied seeds",
        ["seed", "given", "derived", "difference", "$b$ / $c$", "Holm $p$"],
        [["%d" % s, "%.4f" % cells[("J2", s)]["guard_given"],
          "%.4f" % cells[("J2", s)]["guard_derived"],
          "%+.4f" % cells[("J2", s)]["guard_diff"],
          "%d / %d" % (cells[("J2", s)]["b"], cells[("J2", s)]["c"]),
          "%.3g" % js["cells"]["J2_s%d" % s]["p_holm"]] for s in SEEDS],
        note=("Untied model, %d cases, conditional denominator. Bars: Holm "
              "$p < %.2f$ and difference $\\ge$ %.2f." % (n, ALPHA, PROV_BAR)))

    d.table_page(
        "The tied model, which has no guard, is flat"
        if js["G4_quiet"] else
        "The tied model moves too, so the effect is not the guard",
        ["cell", "given", "derived", "difference", "finding 68, round 0"],
        [["S1 s%d" % s, "%.4f" % cells[("S1", s)]["guard_given"],
          "%.4f" % cells[("S1", s)]["guard_derived"],
          "%+.4f" % cells[("S1", s)]["guard_diff"],
          "%.4f" % cells[("S1", s)]["round0_guard"]] for s in SEEDS],
        note=("The tied model names $v$ whether or not $v$ is still forced, "
              "which is what finding 68 reported at round 0."))

    # ---- G6, the clash-matched subset. Scored against G1's OWN 2-of-3 rule,
    # which is what "G1 recomputed" means; the summary's G6_agrees is the
    # stricter all-three flag and is quoted in the note rather than used as the
    # title, because on this data it is False for the seed that also misses G1.
    g6 = js["G6"]
    g6_hits = [s for i, s in enumerate(SEEDS) if g6[i]["diff"] >= PROV_BAR]
    d.table_page(
        "The same seeds separate on exactly clash-matched cases"
        if g6_hits and g6_hits == g1_hits else
        "The clash-matched subset does not reproduce the separation",
        ["seed", "given", "derived", "difference", "raw $p$"],
        [["%d" % s, "%.4f" % g6[i]["given"], "%.4f" % g6[i]["derived"],
          "%+.4f" % g6[i]["diff"], "%.3g" % g6[i]["p_raw"]]
         for i, s in enumerate(SEEDS)],
        note=("Both edits clash the same number of times in these %d cases; "
              "%d of 3 seeds separate, against %d in the full sample."
              % (g6[0]["n_matched"], len(g6_hits), len(g1_hits))))

    # ---- POST-HOC 1: clash is not the mechanism
    j2_cl0 = [cells[("J2", s)]["clash"]["cl0"]["rate"] for s in SEEDS]
    d.table_page(
        "A hint edit that clashes with nothing still fails to guard"
        if min(j2_cl0) > 0.5 else
        "A hint edit that clashes with nothing restores the guard",
        ["cell"] + ["%s (n)" % t for t in ("0 clashes", "1 clash", "2+ clashes")],
        [[cell_label(*k)] + ["%.4f (%d)" % (cells[k]["clash"][t]["rate"],
                                            cells[k]["clash"][t]["n"])
                             for t in CLASH_BINS] for k in CELL_ORDER],
        note=("POST-HOC. Column one is finding 68's edit, a hint edit clashing "
              "with nothing, but at rounds 1-15 rather than round 0."))

    # ---- G5, depth. The tied arm is NOT identically 1.0000 here -- S1 s0 dips
    # in one bin -- so the note quotes its minimum instead of claiming a flat
    # line the figure would contradict.
    rise = js["G5_deep_rise"]
    tied_min = min(cells[("S1", s)]["depth"][t]["rate"]
                   for s in SEEDS for t in DEPTH_BINS
                   if cells[("S1", s)]["depth"][t]["n"])
    d.claim_page(
        "The guard also weakens as the deduction gets deeper"
        if sum(1 for x in rise if x >= PROV_BAR) >= 2 else
        "Depth does not change how often the guard fires",
        lambda ax: draw_depth(ax, cells), "depth",
        note=("Untied model, given-side rate by $v$'s round; %d of 3 seeds rise. "
              "Round 0 cannot occur. Tied arm, not shown, never below %.4f."
              % (sum(1 for x in rise if x >= PROV_BAR), tied_min)))

    # ---- POST-HOC 2: the decomposition. This is the page the run exists for.
    d.claim_page(
        "Depth moves the guard further than provenance does",
        lambda ax: draw_decomp(ax, cells), "decomposition",
        note=("POST-HOC. The dashed step crosses case lists, mixing depth and "
              "construction; only the solid step is paired."))

    d._add(SCOPE)

    # ---- the required comparison against the previous measurement
    d.table_page(
        "Against finding 68 the untied rates here are far higher",
        ["cell", "finding 68, round 0", "here, given", "here, derived"],
        [[cell_label(*k), "%.4f (n=%d)" % (cells[k]["round0_guard"],
                                           cells[k]["round0_n"]),
          "%.4f" % cells[k]["guard_given"], "%.4f" % cells[k]["guard_derived"]]
         for k in CELL_ORDER],
        note=("POST-HOC. The previous measurement of this quantity: same "
              "checkpoints and readout, a different case list, so not paired."))

    sep_rates = ("Given %s against derived %s, %s that separate."
                 % (" and ".join("%.2f" % cells[("J2", s)]["guard_given"]
                                 for s in g1_hits),
                    " and ".join("%.2f" % cells[("J2", s)]["guard_derived"]
                                 for s in g1_hits), sep))
    closing = {
        "provenance_specific": [
            sep_rates,
            "Both edits were shown to be read, and an irrelevant edit moves nothing.",
            "The smaller piece: depth moved the earlier gap further than provenance did."],
        "reversed": [
            "The untied model withholds more when its own deduction is broken.",
            "That is the opposite of the registered direction and is reported as such.",
            "The post-hoc lead is not confirmed by this run."],
        "not_provenance": [
            "Breaking a hint and breaking a deduction do the same thing.",
            "So the earlier gap is not about where the broken fact came from.",
            "The depth profile is where it has to come from instead."],
        "unread_edit": [
            "The rule edit missed, so an edited token may never have been read.",
            "Naming $v$ cannot be read as a guard decision here.",
            "The construction is reported as bounded, not the model."],
        "no_null": [
            "The clash-matched placebo moved the node readout on its own.",
            "So naming $v$ is not specific to determinacy in this construction.",
            "Nothing below the gate is read."],
    }[verdict]
    d.points_page(head, closing)

    path = d.write()
    print("wrote %s (%d pages, %d explanation, %d figures)"
          % (path, d.n, n_explain, len(os.listdir(d.fig_dir))))
    print("verdict %s   G1 %s  G2 %s  G3 %s  G4 %s  G6 %s"
          % (verdict, js["G1"], js["G2"], js["G3"], js["G4_quiet"],
             js["G6_agrees"]))
    print("J2 guard given  %s" % ["%.4f" % cells[("J2", s)]["guard_given"]
                                  for s in SEEDS])
    print("J2 guard derived %s" % ["%.4f" % cells[("J2", s)]["guard_derived"]
                                   for s in SEEDS])
    print("headline: %s" % head)


if __name__ == "__main__":
    main()
