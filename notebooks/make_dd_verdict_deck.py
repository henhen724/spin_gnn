"""Slide deck: IS THE LEAK-FREE NETWORK RUNNING DD?

Run as `python -m notebooks.make_dd_verdict_deck` from the repo root.

Jobs 50075339 (round-0 interventions, 3 arms) and 50077245 (the chaining
interventions, 3 arms). This deck answers one question with causal evidence
rather than probe scores, which is the whole reason it exists: every earlier
answer in this project came from decoding accuracy, and finding 8's `next_node`
probe was retired by a hand rule that does no deduction at all.

THE EVAL IS AN INTERVENTION, NOT A SCORE. Each case: a node DD forces, whose
forcing set covers q-1 colours with exactly one known neighbour holding the
colour that is doing the work. Recolour that one neighbour -- a ONE TOKEN change,
asserted in build_pair -- so DD's answer changes from c to a, and ask whether the
model's prediction follows. Chance is 1/q = 0.25. This cannot be passed by
pattern-matching the prompt, because the prompt is almost identical in the two
conditions and the correct answers differ.

Nothing is hand-typed from a result: every number, verdict and headline is
computed from the CSVs the two jobs wrote.
"""
import json
import math
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

SCRATCH = os.environ["SCRATCH"]
CIRC = f"{SCRATCH}/spin_gnn/sudoku_circuits_leakfree_2026-10-28"
CHAIN = f"{SCRATCH}/spin_gnn/sudoku_chain_2026-10-28"
SKIP = f"{SCRATCH}/spin_gnn/sudoku_skip_2026-10-28"
# Job 50098957. The -29 (no b) directory is attempt 1's output, which is VOID:
# it hooked encoder.layers[layer] while the dictionaries are indexed one higher,
# so every condition was spliced one block too late. Do not point this here.
ABL = f"{SCRATCH}/spin_gnn/sudoku_sae_ablate_2026-10-29b"
# SAE dictionaries. The 1.0000 arm is job 50075499; the other two were trained
# by 49744985 with byte-identical hyper-parameters, so the three are one table.
SAE = {"leak-free 1.0000": f"{SCRATCH}/spin_gnn/sudoku_sae_solved_2026-10-28/leakfree1000_l0_64",
       "leak-free 0.8540": f"{SCRATCH}/spin_gnn/sudoku_sae_kindsplit_2026-10-27/leakfree_l0_64",
       "leaked 0.9960": f"{SCRATCH}/spin_gnn/sudoku_sae_kindsplit_2026-10-27/leaked_l0_64"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_dd_verdict_2026-10-28")

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "0.45"
# Red and green are reserved for MODEL IDENTITY (they key every plot). Caveat
# text used to be red too, which made "red" mean both "the leaked model" and
# "be careful" on the same page; caveats are this orange instead.
CAVEAT = "#b35c00"
CHANCE = 0.25
SITES = ["edited_hint", "graph_row_u", "graph_row_w", "predict_pos"]
# u is a known neighbour of v that is NOT the binding one; w IS the binding
# neighbour, the hint the intervention edits. Both graph rows are controls.
SITE_LABELS = {"edited_hint": "edited\nhint w",
               "graph_row_u": "row of u\n(control)",
               "graph_row_w": "row of w\n(control)",
               "predict_pos": "predicts\nv's colour"}
# (label, circuits dir, chain dir, colour). Arm A is the model that actually
# solved the task; C is the leaked twin finding 2 was measured on.
ARMS = [("leak-free 1.0000", "A_leakfree_1000", "A_leakfree_1000", BLUE),
        ("leak-free 0.8540", "B_leakfree_0854", "B_leakfree_0854", GREEN),
        ("leaked 0.9960", "C_leaked_0996", "C_leaked_0996", RED)]
# Display names. The dict KEYS above stay as the checkpoint identifiers because
# every path and every D[...] lookup uses them, but NO SLIDE shows them: three
# four-decimal names collide with three dozen four-decimal results, and the
# highest-looking number (0.9960) belongs to the model that fails. The rate in
# the gloss is parsed out of the key, so it still is not hand-typed.
ROLE = {"leak-free 1.0000": "SOLVER",
        "leak-free 0.8540": "WEAK TWIN",
        "leaked 0.9960": "LEAK-READER"}


def gloss(label, sep=" "):
    """'SOLVER (leak-free, 100.0%)'. `sep` is '\\n' where a chart tick needs it."""
    kind = "leaked" if label.startswith("leaked") else "leak-free"
    return f"{ROLE[label]}{sep}({kind}, {float(label.split()[-1]):.1%})"


def wilson(k, n, z=1.96):
    """95% Wilson interval. Normal intervals are useless here -- several bins
    sit at exactly 0 or 1, where the normal interval has zero width."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load():
    """Round-0 and chained results per arm. Arms still running are skipped so
    the deck can be built incrementally; the title page says which are missing."""
    out = {}
    for label, cdir, hdir, colour in ARMS:
        rec = {"label": label, "colour": colour}
        bpath = f"{CIRC}/{cdir}/behavior.csv"
        if os.path.exists(bpath):
            b = pd.read_csv(bpath)
            rec["r0"] = {"n": len(b),
                         "clean": b.clean_correct.mean(),
                         "flip": b.corrupt_flipped.mean(),
                         "clean_ld": b.clean_logit_diff.mean(),
                         "corrupt_ld": b.corrupt_logit_diff.mean()}
            rec["patch"] = pd.read_csv(f"{CIRC}/{cdir}/patching.csv")
            rec["heads"] = pd.read_csv(f"{CIRC}/{cdir}/head_ablation.csv")
        spath = f"{CHAIN}/{hdir}/chain_summary.json"
        if os.path.exists(spath):
            with open(spath) as f:
                s = json.load(f)
            rec["chain"] = s
            rec["chain_rows"] = pd.read_csv(f"{CHAIN}/{hdir}/chain_behavior.csv")
        kpath = f"{SKIP}/{hdir}/skip_summary.json"
        if os.path.exists(kpath):
            with open(kpath) as f:
                rec["skip"] = json.load(f)
            rec["skip_rows"] = pd.read_csv(f"{SKIP}/{hdir}/skip_behavior.csv")
        out[label] = rec
    return out


def load_sae():
    """By-kind SAE summaries, one frame per arm, or {} if none are on disk.

    `justifies_next[u]` = "u is a KNOWN neighbour of the node being emitted",
    i.e. exactly the set DD eliminates over. `n_just` is how many dictionary
    features matched one of those predicates well enough to be called
    interpretable; `best_justifier_f1` is the best single feature's F1.
    """
    out = {}
    for label, d in SAE.items():
        p = f"{d}/sae_summary_by_kind.csv"
        if not os.path.exists(p):
            continue
        k = pd.read_csv(p)
        counts = k.family_counts.map(json.loads)
        k["n_just"] = counts.map(lambda c: c.get("justifies_next", 0))
        k["n_nbr"] = counts.map(lambda c: c.get("nbr_of_next", 0))
        out[label] = k
    return out


def load_ablation():
    """SAE feature-ablation summaries, or {} if the job has not landed yet.

    `interp` is finding 19/21's own interpretability gate, which is the
    headline; `null` is the same test widened to every feature that beats its
    shuffled-label null, which buys paired cases at the cost of precision.
    `node_ran` records whether the pre_node arm found anything to ablate at
    all: the pre-registered prediction is that it does not, and a missing
    summary file is that prediction confirmed, not a failure.
    """
    out = {}
    for key, sub in [("interp", "solved_pre_color_interp"), ("null", "solved_pre_color_null")]:
        p = f"{ABL}/{sub}/ablate_summary.csv"
        if os.path.exists(p):
            out[key] = pd.read_csv(p).sort_values("layer").reset_index(drop=True)
    nd = f"{ABL}/solved_pre_node_null/ablate_summary.csv"
    if os.path.isdir(os.path.dirname(nd)):
        out["node_ran"] = os.path.exists(nd)
        if out["node_ran"]:
            out["node"] = pd.read_csv(nd).sort_values("layer").reset_index(drop=True)
    return out


def paired_gap_ci(layer, sub="solved_pre_color_interp"):
    """95% interval on the binding-minus-other gap, from the per-case file.

    `ablate_summary.csv` carries the McNemar p but not the discordant counts,
    and a null result is unreadable without an interval: "no effect" and "too
    few paired cases to see one" give the same p. Returns None if the per-case
    file is missing, in which case the page simply omits the interval.
    """
    p = f"{ABL}/{sub}/ablate_layer{int(layer):02d}.csv"
    if not os.path.exists(p):
        return None
    r = pd.read_csv(p)
    r = r[(r.n_bind > 0) & (r.n_nonbind > 0)]
    if not len(r):
        return None
    nb = r.zero_jn_nonbind_correct.astype(bool)
    bd = r.zero_jn_bind_correct.astype(bool)
    b, c, n = int((nb & ~bd).sum()), int((bd & ~nb).sum()), len(r)
    if b + c == 0:
        # Every pair concordant. The normal approximation collapses to the
        # single point 0.000, which would claim infinite precision from a
        # result that is really "no discordant pair was observed". The rule of
        # three bounds the discordant rate at 3/n with 95% confidence, and the
        # gap cannot exceed the discordant rate.
        return -3.0 / n, 3.0 / n, 0
    diff = (b - c) / n
    # paired normal approximation on the difference of two correlated
    # proportions: only the discordant pairs carry information about it.
    h = 1.96 * float(np.sqrt(max(b + c - (b - c) ** 2 / n, 0.0))) / n
    return diff - h, diff + h, b + c


def layer_spans(layers):
    """[6, 7, 8, 10] -> 'L6-L8, L10'. Honest about gaps, short enough for a legend."""
    out, i = [], 0
    while i < len(layers):
        j = i
        while j + 1 < len(layers) and layers[j + 1] == layers[j] + 1:
            j += 1
        out.append(f"L{layers[i]}" if i == j else f"L{layers[i]}-L{layers[j]}")
        i = j + 1
    return ", ".join(out)


def twin_skip_mcnemar(D, a="leak-free 1.0000", b="leak-free 0.8540"):
    """Paired twin comparison on the withhold test: (n, b_only, a_only, p).

    One arm ran 350 cases and the others 250, but the case stream is the same,
    so the SHORTER length is the paired sample and the rows line up positionally
    within it -- checked here on the four identifying columns rather than
    assumed. `b_only` = cases where the WEAK TWIN kept naming v and the SOLVER
    withheld it; `a_only` = the reverse. Exact two-sided McNemar (binomial on
    the discordant pairs), computed rather than typed: the footer used to carry
    44 / 0 / p = 1.1e-13 by hand.
    """
    if "skip_rows" not in D.get(a, {}) or "skip_rows" not in D.get(b, {}):
        return None
    ra, rb = D[a]["skip_rows"], D[b]["skip_rows"]
    n = min(len(ra), len(rb))
    key = ["node", "round0_rank", "binding_nbr", "n_known"]
    if not ra[key].head(n).equals(rb[key].head(n)):
        return None
    ra, rb = ra.head(n), rb.head(n)
    ea, eb = ra.corrupt_emits_v.astype(bool).values, rb.corrupt_emits_v.astype(bool).values
    n01, n10 = int((eb & ~ea).sum()), int((ea & ~eb).sum())
    m = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return len(ra), n01, n10, p


def page_ablation(out, A):
    """Correlational -> causal: remove the features and see if the rule breaks.

    Everything is scored against the reconstruction condition (the SAE's own
    output spliced back in, nothing zeroed), so reconstruction error is netted
    out of every bar instead of being charged to the ablation. The headline is
    the binding-vs-other pair, matched on family, count, position, layer and
    cases, differing only in which neighbour the removed features refer to --
    so no reconstruction-error or feature-count story can open a gap.
    """
    k = A["interp"]
    d = k.zero_jn_nonbind_acc - k.zero_jn_bind_acc
    bi = int(d.idxmax())
    best = k.loc[bi]
    gap = float(d.max())
    used = bool(best.bind_vs_nonbind_p < 0.05 and gap > 0)
    # Pre-registered control (b), and it is read FIRST because it decides
    # whether the designed contrast is readable at all: removing the WHOLE
    # forcing set has to cost something. If it costs nothing, no condition on
    # the page moved the output and the answer is "inconclusive", not "null".
    drop = float((k.recon_acc - k.zero_jn_all_acc).max())
    sensitive = drop > 0.01
    mcols = [f"{c}_margin" for c in ("recon", "zero_jn_all", "zero_jn_bind", "zero_jn_nonbind", "zero_random")]
    dm = k[mcols].sub(k.clean_margin, axis=0)
    mv = float(dm.abs().max().max())
    # signed, because the sign is the point: the forcing-set ablation moves the
    # margin UP. The largest shift in the panel is the random control's, down.
    jn = float(dm.zero_jn_all_margin.iloc[0])
    ci = paired_gap_ci(best.layer)
    cis = f", 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
    if used:
        head = ("Zero the features for the one neighbour that forces the answer and the deduction breaks;\n"
                f"zero as many for an equally-known neighbour and it does not -- layer {int(best.layer)}, "
                f"gap {gap:+.2f}, p = {best.bind_vs_nonbind_p:.3g}")
    elif sensitive:
        head = ("Zeroing those features does not change the answer: the forcing set is encoded here,\n"
                f"and this test cannot show it being used -- best gap {gap:+.2f}, "
                f"p = {best.bind_vs_nonbind_p:.3g}")
    else:
        head = ("Nothing we remove moves the answer -- not the binding neighbour's features, not the whole\n"
                "forcing set, not a random set of the same size. This test has no sensitivity to report from.")

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(head, size=BIG - 6, weight="bold", linespacing=1.4, y=0.985)
    fig.text(0.5, 0.888, "The binding neighbour is the one whose colour rules out the node's last alternative; its fellow "
                         "known neighbours are just as known, but not decisive.",
             ha="center", va="top", size=MED - 4, color="0.35")
    ax1 = fig.add_axes([0.085, 0.46, 0.375, 0.30])
    ax2 = fig.add_axes([0.590, 0.46, 0.375, 0.30])
    x = np.arange(len(k))
    w = 0.34
    b1 = ax1.bar(x - 0.19, k.zero_jn_bind_acc, w, color=RED)
    b2 = ax1.bar(x + 0.19, k.zero_jn_nonbind_acc, w, color=GREY)
    # One segment per layer, drawn only over that layer's pair: the ceiling is
    # measured per layer and a line spanning the axis would imply one number.
    for i, r in enumerate(k.itertuples()):
        ax1.plot([i - 0.30, i + 0.30], [r.recon_acc] * 2, color="k", lw=3, ls="--")
    ln, = ax1.plot([], [], color="k", lw=3, ls="--")
    # p and n go under the axis, not over the bars: these accuracies sit near
    # the top of the panel and per-bar annotations would land in the legend.
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"layer {int(r.layer)} of 12\n{int(r.n_paired)} cases, p = {r.bind_vs_nonbind_p:.3g}"
                         for r in k.itertuples()], size=SMALL - 1)
    ax1.set_ylabel("share of cases still answered\nwith the forced colour", size=SMALL)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    # Pinned, not auto: only one layer yielded features under the headline gate,
    # and a single auto-scaled group stretches two bars across the whole panel,
    # which reads as a full-width block rather than as a matched pair.
    ax1.set_xlim(-0.6, len(k) - 0.4)
    # Legends above the axes, not inside: these bars run to ~0.97, so an inside
    # legend either covers them or needs dead headroom that reads as accuracy.
    # Handles passed explicitly because legend() collects lines before patches,
    # which would otherwise split the red/grey pair across the two columns.
    # The dashed line's label states its VALUE. It used to read "the ceiling,
    # not 1.0" on a run where the ceiling is exactly 1.000, so the legend
    # contradicted the line it was labelling.
    ceil = (f"{k.recon_acc.iloc[0]:.3f}" if k.recon_acc.nunique() == 1 else
            f"{k.recon_acc.min():.3f}-{k.recon_acc.max():.3f}")
    ax1.legend([b1, b2, ln], ["zero the binding neighbour's features",
                              "zero another known neighbour's, same count",
                              f"nothing zeroed: the ceiling ({ceil})"],
               loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
               frameon=False, fontsize=SMALL - 2)

    # The controls are read as a change in the decision margin, not as a second
    # accuracy panel: when accuracy sits at the ceiling in every condition, five
    # accuracy bars are five identical bars. The margin is where a perturbation
    # shows up at all, and its size is what decides whether the left panel's
    # answer is a null or an unreadable test. Hatching keeps red and grey from
    # reading as the left panel's two series continued across the page.
    # As a PERCENTAGE of the clean margin, on a +-100% axis. In logits the axis
    # auto-scaled to the largest shift, so four changes of under 1% filled the
    # panel top to bottom while the text below called them negligible -- the
    # figure argued against its own caption. The point of this panel is that
    # nothing moved, and only a fixed axis can show that.
    for j, (c, lab, col, hat) in enumerate([
            ("recon", "nothing zeroed (SAE in)", "0.35", ""),
            ("zero_jn_all", "the whole forcing set", RED, "//"),
            ("zero_jn_bind", "the binding neighbour's", RED, ""),
            ("zero_random", "as many at random", GREY, "\\\\")]):
        ax2.bar(x + (j - 1.5) * 0.2,
                100 * (k[f"{c}_margin"] - k.clean_margin) / k.clean_margin, 0.2,
                color=col, hatch=hat, edgecolor="w", label=lab)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xlim(-0.6, len(k) - 0.4)
    ax2.set_ylim(-100, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"layer {int(r.layer)} of 12\n{int(r.n_jn_features)} features removed"
                         for r in k.itertuples()], size=SMALL - 1)
    ax2.set_ylabel("change in the decision margin,\n% of the clean margin "
                   f"({k.clean_margin.iloc[0]:.1f} logits)", size=SMALL)
    # Above the zero line, not on it: centred at 0.5 the axhline struck the
    # words through. The second line is there because an empty-looking panel
    # reads as a figure that failed to draw rather than as four null bars.
    ax2.text(0.5, 0.72, f"every bar is within {100 * mv / k.clean_margin.iloc[0]:.1f}% of zero\n"
                        "(all four are drawn -- they are flat against the line)",
             transform=ax2.transAxes, ha="center", va="center", size=SMALL,
             style="italic", color="0.35", linespacing=1.5)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
               frameon=False, fontsize=SMALL - 2)

    fig.text(0.2725, 0.845, "THE PAIR THAT CARRIES THE CLAIM" if used else "THE PAIR THAT WOULD CARRY THE CLAIM",
             ha="center", va="top", size=MED - 2, weight="bold")
    fig.text(0.7775, 0.845, "THE CONTROLS: did the forward pass move at all?",
             ha="center", va="top", size=MED - 2, weight="bold")
    fig.text(0.2725, 0.410, "p compares red against grey on the same cases, exact McNemar, two-sided.\nA case counts only if it "
                            "owns features for both kinds of neighbour.",
             ha="center", va="top", size=SMALL - 2, color="0.35", linespacing=1.4)

    rep = ""
    if "null" in A:
        n = A["null"]
        nb = n.loc[(n.zero_jn_nonbind_acc - n.zero_jn_bind_acc).idxmax()]
        rep = (f" The powered replication -- the same test widened to every feature that beats its null, "
               f"{int(nb.n_paired)} paired cases at layer {int(nb.layer)} -- gives "
               f"{nb.zero_jn_nonbind_acc - nb.zero_jn_bind_acc:+.2f}, p = {nb.bind_vs_nonbind_p:.3g}.")
    nodeline = ""
    if A.get("node_ran") is False:
        nodeline = (" At the token that picks the node there was nothing to ablate at any layer, as "
                    "pre-registered: the forcing set is encoded before the colour only.")
    elif "node" in A:
        nd = A["node"]
        # No "prediction (d)" and no finding number: a reader who has not seen
        # the pre-registration cannot use either, and the observation stands
        # without them.
        nodeline = (" The same test at the token that picks the NODE found features to remove there too "
                    f"({int(nd.n_jn_features.max())} at layer {int(nd.loc[nd.n_jn_features.idxmax(), 'layer'])}), "
                    "and removing those changed nothing either.")
    design = (
        "THE DESIGN: every bar is read against the reconstruction condition (the SAE's own output spliced back in, nothing zeroed), "
        "so reconstruction error is netted out rather than charged to the ablation. The two bars on the left remove the same number "
        "of features, of the same kind, at the same position, in the same layer, on the same cases; they differ only in which "
        "neighbour the features refer to, the one whose colour does the forcing or another known neighbour whose colour does not. "
        "Only the model reading that neighbour's colour to eliminate a candidate can open a gap between them." + rep + nodeline)
    if used:
        verdict = (
            "READ IT AS: these features are necessary for the answer, not merely correlated with it -- the one causal SAE result on "
            "this deck. Two caveats stay attached. It does not separate computing the elimination from reading out an elimination "
            "made elsewhere: zeroing a readout the rest of the network consumes breaks the answer too. And layer "
            f"{int(best.layer)} is the best of the layers probed{cis}; the others are on the chart, unselected.")
    elif sensitive:
        verdict = (
            "READ IT AS: this does not show the features being used. They are present and decodable (previous page), but removing "
            f"them costs no more than removing a count-matched random set{cis} -- which rules out a large effect, not every effect. "
            "Two readings survive and this run cannot separate them: the rule may be spread over more features than our selection "
            "threshold picks up -- the head-ablation page found exactly that redundancy -- or these features may be a readout of a "
            "decision made elsewhere. Either way the verdict on the next page rests on the interventions, which do not depend on "
            "this page.")
    else:
        disc = ("not one discordant answer" if ci and ci[2] == 0 else
                f"{ci[2]} discordant answers" if ci else "the paired comparison")
        verdict = (
            "READ IT AS: inconclusive -- and the control says so before the headline does. Removing the whole forcing set had to "
            f"cost more than removing a count-matched random set; at layer {int(best.layer)} both cost nothing, "
            f"{k.zero_jn_all_acc.iloc[0]:.3f} against a {k.recon_acc.iloc[0]:.3f} baseline, and the paired comparison has "
            f"{int(best.n_paired)} cases with {disc}{cis}. The largest of these ablations moves the decision margin by "
            f"{100 * mv / k.clean_margin.iloc[0]:.1f}% of its clean value, and removing the forcing set moves it "
            + ("the wrong way, towards MORE confidence" if jn > 0 else "down by less than that") +
            ". So this is not evidence that the features go unused. It is a test in which zeroing "
            f"{int(best.n_jn_features)} of 4096 features at one token cannot perturb a decision this overdetermined -- no "
            "sensitivity, so no null to report. The verdict on the next page rests on the interventions, which do move the answer.")
    # Wrapped here rather than hard-wrapped in the literal: `rep`, `nodeline`
    # and `cis` are spliced into the middle of the paragraphs and only exist on
    # some runs, so any \n placed by hand lands in the wrong spot once they
    # appear. The null branch is the longer of the two and sets the height.
    body = textwrap.fill(design, width=180) + "\n\n" + textwrap.fill(verdict, width=180)
    fig.text(0.5, 0.358, body, ha="center", va="top", size=MED - 7, linespacing=1.4)
    # 3 footer lines, no more: the null branch's body reaches y ~ 0.107 and
    # footer() grows upward from 0.012 at ~0.022 a line.
    footer(fig, "Dictionaries: the SOLVER, 4096 features a layer, one seed. The "
                f"{int(best.n_cases)} cases are held-out first-round deductions whose forced colour one neighbour's recolouring can "
                "flip; features come from the previous page's threshold.",
           width=150)
    out.savefig(fig)


# Training-set sizes, in instances. CONFIG, not a result -- the same kind of
# typed-in fact as "N = 24, degree 5, q = 4" -- but the whole twin story rests
# on this difference, so the slide has to state it.
NTRAIN = {"leak-free 1.0000": 240_000, "leak-free 0.8540": 80_000,
          "leaked 0.9960": 80_000}


def page_title(out, D, ex, O, S):
    """The verdict slide. The task, DD and the models get pages 2-4; repeating
    their definitions here made a first slide nobody finishes reading, and the
    results table is page 8's job."""
    fig = plt.figure(figsize=(16, 9))
    a = D["leak-free 1.0000"]
    r0, ch = a["r0"]["flip"], a.get("chain", {}).get("corrupt_flipped")
    leaked = D["leaked 0.9960"]["r0"]["flip"]
    fig.text(0.5, 0.95, "Is the leak-free network running DD?", ha="center",
             size=BIG + 5, weight="bold")
    fig.text(0.5, 0.895, "Causal interventions, not probe scores",
             ha="center", size=MED + 2, color="0.35")

    sk = a.get("skip", {}).get("corrupt_emits_v")
    others = [1 - D[l].get("skip", {}).get("corrupt_emits_v", 0.0)
              for l, _, _, _ in ARMS if l != "leak-free 1.0000" and "skip" in D[l]]
    # The withhold line is written from whether the OTHER arms pass it too: a
    # verdict line that claims the guard while pages 7 and 10 disown it is the
    # one thing a reader checks and the deck cannot afford to get wrong.
    shared = bool(others) and min(others) >= 0.80
    if ch is not None and ch >= 0.60:
        verdict = ["YES -- it applies DD's elimination rule, and chains it off its own deductions."]
        if sk is not None:
            verdict.append("The withhold test is passed by ALL THREE models, so it is not evidence on its own."
                           if shared else
                           "It also withholds a deduction that stops being forced.")
        verdict.append("DD's other half -- WHICH node to emit -- is a fixed index sweep, not a deduction.")
    else:
        verdict = ["YES for DD's one-step rule; chaining still measuring."]
    # MED+1 at three lines, not MED+4: the third line ran off the right edge and
    # the block's last line landed on top of the model list below it.
    fig.text(0.5, 0.845, "\n".join(verdict), ha="center", va="top", size=MED + 1,
             weight="bold", color=BLUE, linespacing=1.6)

    def _g(label):
        s = D[label].get("skip")
        return f"{1 - s['corrupt_emits_v']:.3f}" if s else "(running)"

    def _c(label):
        s = D[label].get("chain")
        return f"{s['corrupt_flipped']:.3f}" if s else "(running)"

    def _p(label):
        # placebo_emits_v, NOT its complement. Under the placebo edit v is STILL
        # forced, so following DD means naming it -- the same "HIGH = followed
        # DD" direction as every other column. Quoted as the complement (0.060)
        # it read as a floor on a page that called it a ceiling.
        s = D[label].get("skip")
        return f"{s['placebo_emits_v']:.3f}" if s else "  -- "

    lines = [
        "THREE 12-LAYER TRANSFORMERS, trained on Direct Deduction's own solution traces.",
        "",
        f"   SOLVER        leak-free encoding, {NTRAIN['leak-free 1.0000']:,} training instances. Solves 100.0% of held-out graphs.",
        f"   WEAK TWIN     leak-free encoding, {NTRAIN['leak-free 0.8540']:,} instances -- the only difference. Solves 85.4%.",
        f"   LEAK-READER   LEAKED encoding, {NTRAIN['leaked 0.9960']:,} instances. Solves 99.6% by reading the answer off the input.",
        "",
        "One-token counterfactuals, scored so that HIGH = followed DD in every column:",
        "",
        "                          RULE              RULE            WITHHOLD       placebo edit",
        "                      on given hints   on its OWN facts   forcing broken   still names v",
        f"   SOLVER                 {r0:.3f}            {_c('leak-free 1.0000')}            "
        f"{_g('leak-free 1.0000')}            {_p('leak-free 1.0000')}",
        f"   WEAK TWIN              {D['leak-free 0.8540']['r0']['flip']:.3f}            "
        f"{_c('leak-free 0.8540')}            {_g('leak-free 0.8540')}            {_p('leak-free 0.8540')}",
        f"   LEAK-READER            {leaked:.3f}            {_c('leaked 0.9960')}            "
        f"{_g('leaked 0.9960')}            {_p('leaked 0.9960')}",
        "   chance                 0.250            0.250              --                 --",
    ]
    fig.text(0.045, 0.680, "\n".join(lines), ha="left", va="top", size=MED - 5,
             family="monospace", linespacing=1.4)
    # Both numbers come from the files page 3 reads, so the footer's "every
    # number is computed" holds here too -- and computing them is what caught the
    # version of this sentence that quoted 0.545, which is the LEAKED model's
    # score at a different layer, not this model's.
    sa = S.get("leak-free 1.0000")
    probe = (float(sa[sa.position_kind == "pre_node"].best_next_node_f1.max())
             if sa is not None else None)
    hand = O["rules"]["min_unknown"]["max_f1"] if O else None
    if probe is not None and hand is not None:
        fig.text(0.045, 0.265,
                 "WHY INTERVENTIONS AND NOT PROBES: the best single feature for this model's own 'which node is next' "
                 f"scores {probe:.3f} -- and a hand\nrule that deduces nothing at all, 'the next unknown index after "
                 f"the last one', already scores {hand:.3f} on the same metric.\nA decoding score cannot separate "
                 "deducing from counting. Editing one token and watching the answer move can.",
                 ha="left", va="top", size=MED - 3, linespacing=1.5)
    fig.text(0.045, 0.112,
             "CAVEAT, up front: one task configuration (N = 24, degree 5, q = 4) and ONE CHECKPOINT PER MODEL, no "
             "seed replicates --\nso every leak-free vs leaked contrast is also two separate training runs.",
             ha="left", va="top", size=MED - 3, color="0.15", weight="bold", linespacing=1.5)
    missing = [ROLE[k] for k, v in D.items() if "chain" not in v or "skip" not in v]
    footer(fig, "Prepared 2026-10-29. Every number on every page is computed from the run outputs, not typed in."
                + (f" Arms still running: {', '.join(missing)}." if missing else ""))
    out.savefig(fig)


def page_task(out, ex):
    """The task and the algorithm. Split from page_model on purpose: the two
    together were ~450 words of monospace, which nobody reads in a room."""
    c = ex.cfg
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The task, and the algorithm the model imitates", size=BIG,
                 weight="bold")
    left = [
        "THE TASK",
        "",
        f"  Colour a random regular graph: N = {c.n_nodes} nodes, every",
        f"  node of degree {c.degree}, using q = {c.n_colors} colours so that no edge",
        "  joins two nodes of the same colour.",
        "",
        f"  Some nodes are given as HINTS ({len(ex.hints)} of {c.n_nodes} here);",
        "  the rest must be deduced.",
        "",
        "  Each instance is built from a planted colouring by",
        "  removing hints one at a time, keeping a removal only",
        "  if DD can still recover everything. So every instance",
        "  is solvable by DD alone, and instances differ only in",
        "  how deep the deduction has to go.",
    ]
    right = [
        "DD -- DIRECT DEDUCTION",
        "",
        "  SINGLE-CANDIDATE ELIMINATION (sudoku's 'naked single'),",
        "  on a graph: a node becomes determined one round after its",
        f"  known neighbours show all q-1 = {c.n_colors - 1} colours other than its",
        "  own. Sweep the nodes, emit every determined one, repeat",
        "  until none are left.",
        "",
        "  THE FORCING SET of a node v = the known neighbours whose",
        "  colours do the eliminating. DD fires on v exactly when",
        f"  that set shows q-1 = {c.n_colors - 1} distinct colours; v then takes the",
        "  one colour left over.",
        "",
        "  THE BINDING NEIGHBOUR of v = the one member of that set",
        "  whose colour rules out v's last alternative; its fellow",
        "  members are just as known, but not decisive. Every",
        "  intervention in this deck edits the binding neighbour.",
        "",
        "  REASONING DEPTH T = how many rounds an instance needs,",
        f"  1 to 6 at this hint count ({len(ex.hints)} of {c.n_nodes}). The example",
        f"  on the next slide needs {ex.n_rounds}.",
    ]
    # 14pt at 1.45, not 16 at 1.5: the right column is 21 lines once the binding
    # neighbour is defined here (it used to be defined on the page that uses it
    # last), and 16pt runs it into the paragraph below.
    fig.text(0.05, 0.855, "\n".join(left), ha="left", va="top", size=MED - 3,
             family="monospace", linespacing=1.45)
    fig.text(0.50, 0.855, "\n".join(right), ha="left", va="top", size=MED - 3,
             family="monospace", linespacing=1.45)
    fig.text(0.05, 0.168,
             "Solvable-by-DD is a property of the INSTANCES, not a proof that no other method works -- which is why "
             "the deck goes on to\nrule out the obvious non-deductive routes (reading the leak, next slide; emitting "
             "nodes by index order, the last slide) rather\nthan assuming they are closed.",
             ha="left", va="top", size=MED - 2, linespacing=1.5)
    out.savefig(fig)


def load_order():
    """`scripts/order_baseline_next_node.py --json`, or {} if it has not run.

    The node-ORDER claim is the one assertion on the verdict page that no job in
    this deck measures, and it used to appear as a bare "100% of rounds" with no
    n. These are its numbers: no model, no GPU, hand rules only.
    """
    p = os.path.join(OUT, "order_baseline.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def page_model(out, O, S):
    """The model, the leak, and the question. Second half of the old page 2."""
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The leak, and why there are two kinds of model", size=BIG,
                 weight="bold")
    left = [
        "THE MODEL",
        "",
        "  A 12-layer decoder-only transformer (d_model 256,",
        "  4 heads per layer, MLP 1024), trained from scratch",
        "  with next-token cross-entropy on DD's own traces:",
        "  graph, hints, then round by round the nodes DD",
        "  determines and the colours it gives them.",
        "",
        "  SOLVE RATE, which also names each model, is scored by",
        "  FREE GENERATION: the fraction of 1500 held-out",
        "  instances it colours completely and correctly with",
        "  no teacher forcing.",
    ]
    right = [
        "THE LEAK",
        "",
        "  The generator listed each node's neighbours sorted by",
        "  the neighbour's TRUE COLOUR. So a neighbour's POSITION",
        "  in the adjacency row encoded its colour, and 61-67% of",
        "  nodes could be read off with no deduction at all.",
        "",
        "  Permuting the rows at test time collapses a model trained",
        "  on them: 0.9975 -> 0.1450. Load-bearing, not incidental.",
        "",
        f"  LEAK-READER  sorted rows, {NTRAIN['leaked 0.9960']:,} instances. Solves 99.6% --",
        "               and the rule tests show it does so WITHOUT",
        "               the elimination rule.",
        f"  SOLVER       rows permuted, {NTRAIN['leak-free 1.0000']:,} instances. Solves 100.0%.",
        f"  WEAK TWIN    rows permuted, {NTRAIN['leak-free 0.8540']:,} instances -- the ONLY",
        "               difference from SOLVER. Solves 85.4%.",
    ]
    # Sized so the taller (right) column clears THE QUESTION below it: 17 lines
    # at 1.45 spacing is the limit this page holds.
    fig.text(0.05, 0.865, "\n".join(left), ha="left", va="top", size=MED - 3,
             family="monospace", linespacing=1.45)
    fig.text(0.50, 0.865, "\n".join(right), ha="left", va="top", size=MED - 3,
             family="monospace", linespacing=1.45)
    # Bold sans runs ~20% wider than the monospace above it, so this wraps
    # narrower than it looks like it needs to; the right-hand block starts at
    # 0.52 and the two used to collide.
    fig.text(0.05, 0.305,
             "THE QUESTION: the SOLVER colours a task whose\ninstances are ALL solvable by DD. Is it\n"
             "running DD, or doing something else that happens to\nwork here?\n\nEvery BEHAVIOURAL number from here on "
             "is causal: edit\none input token, ask whether the output follows DD's\nanswer to the EDITED problem. "
             "(The one correlational\npage, the sparse autoencoder, is labelled as such.)",
             ha="left", va="top", size=MED - 3, weight="bold", linespacing=1.5)
    # The other non-deductive route, with its numbers, on the page that sets up
    # the question -- not as a bare assertion on the verdict slide. This is the
    # analysis page 2 promises, and it is the reason the verdict is split: DD
    # has two halves and only one of them turns out to need deducing.
    if O:
        hand = O["rules"]["min_unknown"]["max_f1"]
        # the model's own score on the same metric, from the SAE table rather
        # than typed: pre_node positions, best feature for "which node is next"
        a = S.get("leak-free 1.0000")
        mine = (float(a[a.position_kind == "pre_node"].best_next_node_f1.max())
                if a is not None else None)
        fig.text(0.52, 0.305,
                 "THE OTHER NON-DEDUCTIVE ROUTE -- INDEX ORDER. DD's traces emit each\n"
                 f"round's nodes in ASCENDING NODE INDEX in {O['ascending_frac']:.1%} of rounds "
                 f"({O['n_rounds']:,} rounds,\n{O['n_instances']:,} instances), so the ORDER needs no deduction at all."
                 + (f" The model's best\n'which node is next' feature scores {mine:.3f} on this metric -- but a hand\n"
                    f"rule that deduces nothing, 'the next unknown index after the last\none', already scores "
                    f"{hand:.3f} on it. Most of that score is counting.\n" if mine is not None else
                    f" A hand rule that\ndeduces nothing scores {hand:.3f} on the metric these features are scored on.\n")
                 + "Only WHICH nodes get emitted can carry deduction -- what the\nwithhold test measures.",
                 ha="left", va="top", size=MED - 3, linespacing=1.5)
    out.savefig(fig)


def page_design(out, ex):
    """What the intervention actually does, in tokens."""
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The intervention: one token, two different correct answers",
                 size=BIG, weight="bold")
    body = [
        "DD forces node v because v's FORCING SET -- its known neighbours -- already uses q-1 = 3",
        "of the 4 colours. Exactly one of them, the neighbour w, holds colour a. So v must take",
        "the one colour left over: c. Recolour w to c, and now a is the colour nobody holds, so",
        "DD's answer for v changes from c to a.",
        "",
        "   clean     ... BEGIN_ROUND  v  ->  [c]        w holds a, so c is v's only option",
        "   corrupt   ... BEGIN_ROUND  v  ->  [a]        w recoloured to c, so a is now free",
        "",
        "The ONLY difference between the two prompts is w's single colour token, and the code",
        "checks that mechanically on every case rather than assuming it. Scored as 'moved to",
        "DD's new answer' -- did the prediction at v become a? Not 'is it still accurate': the",
        "correct answer CHANGES, so a memorised output scores zero. The model must pick one of",
        "q = 4 colours, so chance on this test is 0.25.",
        "",
        "Two versions, differing only in WHERE the edited token sits:",
        "",
        "   HINT EDIT     w is a HINT, handed to the model in the prompt, so the edit lands",
        "                 before any reasoning. Tests the elimination rule on GIVEN facts.",
        "",
        "   CHAINED EDIT  w was DEDUCED by the model itself in an earlier round, so the edited",
        "                 token sits in the REASONING section. Tests elimination against a fact",
        "                 the model produced -- propagation, not one rule application.",
    ]
    fig.text(0.06, 0.825, "\n".join(body), ha="left", va="top", size=MED - 3,
             family="monospace", linespacing=1.4)
    # two lines, not four: the monospace block above it reaches y ~ 0.17
    fig.text(0.06, 0.125,
             "One weak direction, stated up front: editing an emitted deduction makes the prefix inconsistent, so a "
             "model that DETECTS\nthe edit and keeps its own answer scores as 'does not chain'. Those cases are "
             "counted separately, as 'kept its old answer'.",
             ha="left", va="top", size=MED - 3, color=CAVEAT, linespacing=1.5)
    out.savefig(fig)


def page_guard_design(out):
    """The converse intervention: break the forcing instead of redirecting it."""
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The converse intervention: break the forcing, and v should vanish",
                 size=BIG, weight="bold")
    # Wrapped to ~95 monospace characters, not ~84: at 24 lines the block's last
    # two lines printed on top of the caveats below it. The page has the width.
    body = [
        "Both tests on the previous page SELECT a node that DD forces and then REDIRECT the forcing. A",
        "model that simply wrote a colour for whichever unknown node came next in index order would pass",
        "them both. So break the forcing instead of redirecting it -- the WITHHOLD test:",
        "",
        "   recolour w to a colour that ANOTHER known neighbour of v already holds.",
        "",
        "Now the freed colour a is held by nobody, v's forcing set covers only q-2 = 2 colours, v has",
        "TWO candidates, and DD does not emit v in this round at all.",
        "",
        "The measurement moves ONE TOKEN TO THE LEFT: from the logits that predict v's COLOUR to the",
        "logits that predict WHICH NODE comes next. Does the model stop naming v?",
        "",
        "   clean      ... BEGIN_ROUND  [v]  ->  c       DD emits v, and so should the model",
        "   broken     ... BEGIN_ROUND  [ ? ]            v is no longer determined",
        "",
        "PLACEBO EDIT, run on every single case: the same kind of one-token edit, but on a hint that is",
        "NOT a neighbour of v. v stays forced and DD's round is unchanged, so the model should still",
        "name v. That is the CONTROL the broken rate is read against -- without it, a low rate would",
        "only show that edits perturb the node stream. Reported the same way round as every other",
        "number in this deck: how often the model did what DD does, so HIGH = followed DD.",
    ]
    fig.text(0.06, 0.855, "\n".join(body), ha="left", va="top", size=MED - 3,
             family="monospace", linespacing=1.4)
    fig.text(0.06, 0.185,
             "Two caveats, both stated before the result. (1) If the placebo collapsed too, the test would say nothing. It does\n"
             "not collapse, but it does not reach 1.000 either (0.90-0.96 across models), so a few points of every rate are edit\n"
             "perturbation -- which is why all three models are read against it. (2) ALL THREE MODELS PASS THIS TEST, the\n"
             "leaked one included. Detecting that a node stopped being determined is evidently easier than applying the rule,\n"
             "so the withhold test is not evidence of DD on its own. Its job is to separate the two leak-free twins.",
             ha="left", va="top", size=MED - 4, color=CAVEAT, linespacing=1.5)
    out.savefig(fig)


def page_table(out, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Every causal measurement", size=BIG, weight="bold")
    # centre of this box sets where the table sits: scale() fixes the row
    # height, so only the centre moves it, and the bottom rows need to clear
    # the explanatory line below
    ax = fig.add_axes([0.03, 0.345, 0.94, 0.54]); ax.axis("off")
    # "followed DD", not "moved to DD's new answer": on the withhold rows the
    # measured event is "stopped naming the node", which is not a move to a new
    # answer at all. One header for one question, with the per-test meaning in
    # the 'test' column and the gloss below.
    cols = ["model", "test", "n", "clean", "followed DD\n(95% Wilson)", "kept its\nold answer",
            "control", "verdict"]
    # the verdict column has to hold "keeps its own answer" on one line
    widths = [0.17, 0.13, 0.05, 0.08, 0.17, 0.10, 0.13, 0.17]

    def ci(p, n):
        """The interval the audience asks for before anything else."""
        lo, hi = wilson(int(round(p * n)), int(n))
        return f"{p:.3f}  [{lo:.2f}, {hi:.2f}]"

    rows = []
    for label, _, _, _ in ARMS:
        rec = D[label]
        if "r0" in rec:
            r = rec["r0"]
            rows.append([ROLE[label], "hint edit", r["n"], f"{r['clean']:.3f}",
                         ci(r["flip"], r["n"]), "--", f"chance {CHANCE:.3f}",
                         "uses the rule" if r["flip"] >= 0.60 else "above chance, barely"])
        if "chain" in rec:
            c = rec["chain"]
            # 0.008 is thirty times BELOW chance: that model is not guessing,
            # it is actively re-asserting its own answer. Calling that "near
            # chance" (the old two-way test) was simply the wrong word.
            f = c["corrupt_flipped"]
            v = ("chains" if f >= 0.60 else
                 "keeps its own answer" if f <= CHANCE / 2 else "near chance")
            rows.append([ROLE[label], "chained edit", c["n_cases"], f"{c['clean_correct']:.3f}",
                         ci(f, c["n_cases"]), f"{c['reverts_to_original']:.3f}",
                         f"chance {CHANCE:.3f}", v])
        if "skip" in rec:
            s = rec["skip"]
            # the withhold row is scored the other way round: 1 - (still names
            # v), so "followed DD" stays one direction down the whole column.
            # The placebo stays in ITS natural direction for the same reason --
            # under the placebo v is still forced, so naming it IS following DD.
            rows.append([ROLE[label], "withhold", s["n_cases"], f"{s['clean_emits_v']:.3f}",
                         ci(1 - s["corrupt_emits_v"], s["n_cases"]), f"{s['corrupt_emits_v']:.3f}",
                         f"placebo {s['placebo_emits_v']:.3f}",
                         "withholds" if s["corrupt_emits_v"] <= 0.20 else "names it anyway"])
    t = ax.table(cellText=rows, colLabels=cols, colWidths=widths, loc="center",
                 cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(MED - 3); t.scale(1, 3.0)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#dddddd"); t[0, j].set_text_props(weight="bold")
    arm_colour = {ROLE[label]: colour for label, _, _, colour in ARMS}
    for i, r in enumerate(rows, start=1):
        # the same colour each model carries in every plot, or the legend on
        # one page silently contradicts the table on another
        t[i, 0].set_text_props(color=arm_colour[r[0]], weight="bold")
        # r[4] is "0.388  [0.34, 0.44]" now, so the point estimate is parsed out
        t[i, 4].set_text_props(weight="bold",
                               color=GREEN if float(r[4].split()[0]) >= 0.60 else RED)
    gaps = "   ".join(f"{ROLE[l]} {D[l]['r0']['clean_ld']:+.1f} -> {D[l]['r0']['corrupt_ld']:+.1f}"
                      for l, _, _, _ in ARMS if "r0" in D[l])
    fig.text(0.5, 0.305,
             "Every column is scored so that HIGH = followed DD, the placebo included; what 'followed DD' MEANS is "
             "set by the test --\nemitting DD's new colour on the two rule rows, withholding the node on the withhold "
             "row. A low 'followed DD' next to a high\n'clean' means the model is right for some reason other than "
             "the elimination: exactly LEAK-READER's profile on its two rule rows.",
             ha="center", va="top", size=MED - 3, linespacing=1.5)
    # Two lines, not three: the third ("a sign change is a reversal...") moved
    # into the footer, which is what kept this block off the footer's top line.
    fig.text(0.5, 0.195,
             "LOGIT GAP on the hint-edit rows, logit(DD's edited answer) - logit(DD's clean answer) at the position "
             f"that predicts v's colour:\n{gaps}",
             ha="center", va="top", size=MED - 3, linespacing=1.5, family="monospace")
    footer(fig, "A sign change in that gap is a reversal of preference, not a shift in confidence. 'kept its old "
                "answer' = named the node anyway (withhold row) or re-asserted its own colour (chained row). "
                "'control' is chance, 1/q, on the rule rows and the placebo edit on the withhold row -- the same "
                "one-token edit made on a hint that is NOT a neighbour of v, where DD's round is unchanged, so naming "
                "v is still correct and HIGH is still right. n is 350 or 250 by when each run was submitted, not by "
                "filtering; paired tests use the 250 shared cases.", width=132)
    out.savefig(fig)


def page_rates(out, D):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("The rule holds on given facts, and on facts it derived itself",
                 size=BIG, weight="bold")

    labels, r0v, chv, cols = [], [], [], []
    for label, _, _, colour in ARMS:
        rec = D[label]
        if "r0" not in rec:
            continue
        labels.append(gloss(label, sep="\n")); cols.append(colour)
        r0v.append(rec["r0"]["flip"])
        chv.append(rec.get("chain", {}).get("corrupt_flipped", np.nan))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, r0v, 0.38, color=cols, edgecolor="k")
    ax1.bar(x + 0.2, chv, 0.38, color=cols, edgecolor="k", hatch="///", alpha=0.75)
    ax1.axhline(CHANCE, color="k", ls="--", lw=2)
    ax1.text(len(labels) - 0.5, CHANCE + 0.02, f"chance {CHANCE:.2f}", ha="right", size=SMALL)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, size=SMALL)
    ax1.set_ylabel("fraction that moved to DD's new answer"); ax1.set_ylim(0, 1.18)
    # Legend ABOVE the axes and no panel title: the old in-axes "center right"
    # box sat on top of the 0.8540 bars, and the title it duplicated said the
    # same thing twice. Grey keys, not the bars' own colours, which key models.
    ax1.legend(handles=[Patch(facecolor="0.7", edgecolor="k", hatch=h, label=t)
                        for h, t in (("", "HINT EDIT (edited a given hint)"),
                                     ("///", "CHAINED EDIT (edited its own deduction)"))],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               fontsize=SMALL - 1, frameon=False)
    for xi, (a, b) in enumerate(zip(r0v, chv)):
        ax1.text(xi - 0.2, a + 0.02, f"{a:.3f}", ha="center", size=SMALL, weight="bold")
        if not np.isnan(b):
            ax1.text(xi + 0.2, b + 0.02, f"{b:.3f}", ha="center", size=SMALL, weight="bold")

    # chaining does not decay with how deep the deduction sits
    a = D["leak-free 1.0000"]
    if "chain" in a:
        br = a["chain"]["by_round"]
        ks = sorted(int(k) for k in br)
        v = [br[str(k)]["flipped"] for k in ks]
        ns = [br[str(k)]["n"] for k in ks]
        lo, hi = zip(*[wilson(round(y * n), n) for y, n in zip(v, ns)])
        ax2.errorbar(ks, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="o-",
                     color=BLUE, lw=3, ms=13, capsize=7, elinewidth=2.5)
        ax2.axhline(CHANCE, color="k", ls="--", lw=2)
        ax2.text(ks[-1] + 0.1, CHANCE + 0.02, f"chance {CHANCE:.2f}", ha="right", size=SMALL)
        for k, y, n in zip(ks, v, ns):
            # Labels go BELOW a point only when there is room below it. At y
            # near 1.0 the -52pt offset lands inside the point's own error bar,
            # which is exactly where the two 1.000 bins sit.
            above = y > 0.95
            ax2.annotate(f"{y:.3f}\nn={n}", (k, y), textcoords="offset points",
                         xytext=(0, 16 if above else -52), ha="center",
                         va="bottom" if above else "baseline", size=SMALL)
        # MED-2: at the default size this label is wider than the panel.
        ax2.set_xlabel("round in which the edited (binding) neighbour was deduced",
                       size=MED - 2)
        ax2.set_ylabel("fraction that moved to DD's new answer")
        ax2.set_ylim(0, 1.25); ax2.set_xticks(ks)
        # The title names the last bin that can carry the claim. "No decay with
        # depth" over a bin of n=4 asserts an absence that n=4 cannot establish:
        # its Wilson interval runs from ~0.51 to 1.0.
        MIN_N = 20
        ok = [k for k, n in zip(ks, ns) if n >= MIN_N]
        thin = [(k, n) for k, n in zip(ks, ns) if n < MIN_N]
        tail = ("" if not thin else
                "; round " + ", ".join(str(k) for k, _ in thin)
                + f": n={', '.join(str(n) for _, n in thin)}, uninformative")
        ax2.set_title(f"No decay out to round {max(ok)}{tail}\nSOLVER, 95% Wilson",
                      size=MED - 2, pad=20)
    fig.tight_layout(rect=[0, 0.19, 1, 0.93])
    # Wrapped by hand: one line of this length overran both page edges.
    fig.text(0.5, 0.175,
             "If the model chained one step but not three, round 1 would be high and the later rounds would fall off.\n"
             "Through round 3 (n=24) the rate stays at 1.00 with the interval clear of chance; round 4 has four cases "
             "and is shown for completeness only.",
             ha="center", va="top", size=MED - 3, linespacing=1.5)
    footer(fig, "Right panel splits the chained test by how deep the edited deduction sat -- the round in which the "
                "BINDING NEIGHBOUR was deduced, not the round of the node under test. Deep bins are small because "
                "deep chained forcings are rare in these instances, not because anything was filtered. Rounds 5-6 are "
                "absent because no case in the sample had its binding neighbour deduced that late.")
    out.savefig(fig)


def page_guard(out, D):
    """The converse test: does it WITHHOLD a deduction that stops being forced?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # Headline computed from the right panel's own categories, below: "it stops
    # naming the node" undersold the result and shared a title with a test all
    # three models pass. What it does instead is the evidence.
    a = D["leak-free 1.0000"]
    shares = None
    if "skip_rows" in a:
        r = a["skip_rows"]
        sk = r[~r.corrupt_emits_v]
        n_forced = int((sk.pred_is_node_token & sk.pred_is_forced).sum())
        n_unforced = int((sk.pred_is_node_token & ~sk.pred_is_forced).sum())
        n_round = int((~sk.pred_is_node_token).sum())
        shares = (n_forced, n_round, n_unforced, len(sk))
    if shares:
        f, rd, uf, tot = shares
        fig.suptitle("Break the forcing and it does not just go quiet: it names whatever else DD would\n"
                     f"name ({f / tot:.1%}), or ends the round ({rd / tot:.1%}). Only {uf / tot:.1%} name an "
                     "unforced node.",
                     size=BIG - 4, weight="bold", linespacing=1.4)
    else:
        fig.suptitle("And it withholds: break the forcing and it stops naming the node",
                     size=BIG, weight="bold")

    labels, cl, br, pl, cols = [], [], [], [], []
    for label, _, _, colour in ARMS:
        s = D[label].get("skip")
        if s is None:
            continue
        labels.append(gloss(label, sep="\n")); cols.append(colour)
        # EVERY bar is "did what DD does", which is naming v under clean and the
        # placebo (v is still forced there) and WITHHOLDING it under broken. The
        # panel used to plot "still names v" in all three conditions, so the
        # correct direction flipped inside one chart and the model that fails
        # every other test owned the three tallest bars on the page.
        cl.append(s["clean_emits_v"]); br.append(1 - s["corrupt_emits_v"])
        pl.append(s["placebo_emits_v"])
    x = np.arange(len(labels))
    ax1.bar(x - 0.26, cl, 0.25, color=cols, edgecolor="k", label="clean prompt")
    ax1.bar(x, pl, 0.25, color=cols, edgecolor="k", alpha=0.6, hatch="...",
            label="placebo edit")
    ax1.bar(x + 0.26, br, 0.25, color=cols, edgecolor="k", hatch="///", alpha=0.85,
            label="forcing BROKEN")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, size=SMALL - 1)
    ax1.set_ylabel("fraction that did what DD does:\nnamed v (clean, placebo) / withheld it (broken)",
                   size=MED - 3)
    ax1.set_ylim(0, 1.30)
    ax1.text(0.5, 0.985, "HIGH is correct in all three conditions, as everywhere else in this deck.\n"
                         "ALL THREE MODELS PASS: this test separates the twins, not leaked from leak-free.",
             transform=ax1.transAxes, ha="center", va="top", size=SMALL - 3)
    # legend above the axes, not inside: the 1.000 bars reach the top and any
    # in-axes box lands on their value labels. Grey handles, not the bars' own:
    # matplotlib takes the first colour of the list, so a blue key would read
    # as "leak-free 1.0000" rather than "clean prompt".
    keys = [Patch(facecolor="0.7", edgecolor="k", hatch=h, label=t)
            for h, t in (("", "clean prompt"), ("...", "placebo edit"),
                         ("///", "forcing BROKEN"))]
    ax1.legend(handles=keys, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=SMALL - 1, frameon=False)
    for xi in range(len(labels)):
        for dx, v in ((-0.26, cl[xi]), (0.0, pl[xi]), (0.26, br[xi])):
            ax1.text(xi + dx, v + 0.02, f"{v:.3f}", ha="center", size=SMALL - 2,
                     weight="bold")
            if v == 0:  # a 0.000 bar draws nothing at all -- mark the baseline
                ax1.plot([xi + dx - 0.12, xi + dx + 0.12], [0.004, 0.004],
                         color="k", lw=3, solid_capstyle="butt")

    # what it names INSTEAD, for the arm that actually solved the task
    a = D["leak-free 1.0000"]
    if "skip_rows" in a:
        r = a["skip_rows"]
        sk = r[~r.corrupt_emits_v]
        n_forced = int((sk.pred_is_node_token & sk.pred_is_forced).sum())
        n_unforced = int((sk.pred_is_node_token & ~sk.pred_is_forced).sum())
        n_round = int((~sk.pred_is_node_token).sum())
        vals = [n_forced, n_round, n_unforced]
        names = [f"a node that IS forced\nunder the broken hints\n(n={n_forced})",
                 f"BEGIN_ROUND\n'the round is over'\n(n={n_round})",
                 f"a node that is not\nforced (n={n_unforced})"]
        # greys, not the arm colours: this panel is one model's outcome
        # categories, and reusing blue/green/red here would read as the arms
        ax2.bar(range(3), vals, color=["0.35", "0.6", "0.82"], edgecolor="k")
        ax2.set_xticks(range(3)); ax2.set_xticklabels(names, size=SMALL - 2)
        ax2.set_ylabel(f"cases (of {len(sk)} withheld)")
        ax2.set_title(f"What it names instead -- {ROLE['leak-free 1.0000']}", size=MED)
        for i, v in enumerate(vals):
            ax2.text(i, v + max(vals) * 0.02, f"{v / len(sk):.1%}", ha="center",
                     size=SMALL, weight="bold")
    fig.tight_layout(rect=[0, 0.235, 1, 0.90])
    mn = twin_skip_mcnemar(D)
    # The p-value belongs to "the discordance is one-sided", NOT to "training on
    # more data causes the guard": there is one seed per arm, so no p can speak
    # to the cause. It used to sit at the end of a sentence about the twins
    # differing, where a reader reasonably takes it for the latter.
    # Two lines, and the "it does not just go quiet" sentence is gone: the title
    # says it and the right panel is it, while the footer below needs five lines.
    twin = ("The two leak-free twins differ here and nowhere else." if mn is None else
            f"Against its WEAK TWIN the discordance is entirely one-sided: of {mn[0]} paired cases, {mn[1]} where the "
            f"twin kept naming v and the SOLVER withheld it,\n{mn[2]} the other way (exact McNemar p = {mn[3]:.1g}). "
            "ONE SEED PER ARM, so that is a within-pair consistency result, not a measured effect of training-set size.")
    fig.text(0.5, 0.235, twin, ha="center", va="top", size=MED - 3, linespacing=1.5)
    footer(fig, "Same cases as the hint-edit test, and the case lists are byte-identical across models, so the arms "
                "are paired instance for instance. Conditioning on cases the twin gets right on the clean prompt "
                "gives 0.188 (n=234). Its 0.176 guard gap is the same order as its 0.146 solve-rate gap -- different "
                "measurements on different samples, so a consistency check, not a decomposition. The LEAK-READER "
                "withholds too (0.992), which is why this page is a twin comparison and not evidence for DD on its "
                "own: a model can stop answering for reasons that have nothing to do with elimination.")
    out.savefig(fig)


def page_patching(out, D):
    arms = [(l, D[l]) for l, _, _, _ in ARMS if "patch" in D[l]]
    fig, axes = plt.subplots(1, len(arms), figsize=(16, 9), squeeze=False)
    fig.suptitle("Where the recoloured fact travels (activation patching)",
                 size=BIG, weight="bold")
    # Only the two informative columns are drawn. The graph rows contributed 72
    # cells reading 0.00 -- a third of the grid spent on a control that says one
    # thing, which it says better in one sentence. Still COMPUTED here, so that
    # sentence's bound is measured rather than remembered.
    shown = ["edited_hint", "predict_pos"]
    ctrl_max = 0.0
    # Where each arm's grip on the edited hint falls below half, by linear
    # interpolation between the bracketing layers. Computed, not typed: the
    # footer used to claim "one to two layers" and the data says three.
    release = {}
    for ax, (label, rec) in zip(axes[0], arms):
        full = (rec["patch"].groupby(["layer", "site"]).recovery.mean()
                .unstack()[SITES])
        ctrl_max = max(ctrl_max,
                       float(full[["graph_row_u", "graph_row_w"]].abs().max().max()))
        m = full[shown]
        h = m["edited_hint"]
        below = [L for L in h.index if h[L] < 0.5]
        if below and below[0] > h.index[0]:
            L1 = below[0]
            L0 = max(x for x in h.index if x < L1)
            release[label] = L0 + (h[L0] - 0.5) / (h[L0] - h[L1])
        n_cases = len(rec["patch"]) // (len(m.index) * len(SITES))
        ax.set_xlabel(f"n = {n_cases} cases", size=SMALL)
        im = ax.imshow(m.values, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       origin="lower", rasterized=True)
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels([SITE_LABELS[s] for s in shown], size=SMALL)
        ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index, size=SMALL - 1)
        ax.set_title(gloss(label, sep="\n"), size=MED - 2)
        ax.set_ylabel("layer" if ax is axes[0][0] else "")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                val = m.values[i, j]
                # -0.00 is a rounding artefact of a tiny negative recovery and
                # reads as a signed quantity to anyone scanning the grid
                txt = f"{0.0 if abs(val) < 5e-3 else val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", size=SMALL - 1,
                        color="w" if val < 0.6 else "k")
    # subplots_adjust, not tight_layout -- the shared colorbar fights it
    fig.subplots_adjust(bottom=0.355, top=0.845, left=0.07, right=0.88)
    fig.colorbar(im, ax=axes[0], fraction=0.025, label="mean recovery")
    # Three lines: the footer below runs to five, and at four these two blocks
    # printed on top of each other.
    fig.text(0.5, 0.265,
             "RECOVERY = copy one activation from the clean run into the corrupted run at that (layer, position) and "
             "measure how much of the flip\nit undoes: 1.00 = that one activation carries the whole edited fact, 0.00 "
             "= none of it. w is the neighbour whose colour the edit changes.\nThe two CONTROL columns -- the adjacency "
             f"rows of w and of another known neighbour u -- never exceed {ctrl_max:.2f} anywhere, so the deduction "
             "never\nroutes through the adjacency table; they are left out of the grid to keep it readable.",
             ha="center", va="top", size=MED - 4, linespacing=1.5)
    a, b = "leak-free 1.0000", "leak-free 0.8540"
    # "about N layers earlier" between two single checkpoints is a difference
    # between two networks, not a mechanism: one seed per arm cannot separate
    # the handoff depth from seed noise, and the verb used to assert it did.
    gap = (f" Descriptively, the SOLVER releases the fact from the hint token about "
           f"{release[b] - release[a]:.0f} layers earlier than the WEAK TWIN (crossing half recovery at "
           f"L{release[a]:.1f} against L{release[b]:.1f}), both delivering it by L10. One seed per arm, so "
           f"read that as a difference between these two checkpoints, not as a depth-of-handoff effect."
           if a in release and b in release else "")
    footer(fig, "Read upwards: the edited colour is held AT the edited hint through L5, handed off between L6 and "
                "L10, and has arrived at the prediction position by L10. L11 at the prediction position is the "
                "built-in positive control and must read 1.00; it does." + gap
                + " In the LEAK-READER's panel recovery is a fraction of an effect that is near chance to begin with "
                  "(it follows the edit on 0.388 of cases), so its numbers localise a weak sensitivity, not a rule.",
           width=132)
    out.savefig(fig)


def page_heads(out, D):
    fig, ax = plt.subplots(figsize=(16, 9))
    # headline from arm A's worst head, so the title cannot drift from the plot
    keep = (D["leak-free 1.0000"]["heads"].groupby(["layer", "head"])
            .effect_retained.mean().min())
    fig.suptitle(f"No single head carries the rule: removing the worst one leaves "
                 f"{keep:.0%} of it", size=BIG, weight="bold")
    for label, _, _, colour in ARMS:
        rec = D[label]
        if "heads" not in rec:
            continue
        h = (rec["heads"].groupby(["layer", "head"]).effect_retained.mean()
             .sort_values().reset_index())
        ax.plot(range(len(h)), h.effect_retained, "o-", color=colour, lw=2.5, ms=8,
                # h["head"], not h.head -- the latter is DataFrame.head()
                label=f"{gloss(label)}  (worst: layer {int(h['layer'].iloc[0])}, "
                      f"head {int(h['head'].iloc[0])} = {h.effect_retained.iloc[0]:.3f})")
    ax.axhline(1.0, color="k", ls="--", lw=2)
    ax.text(2, 1.02, "1.00 = ablating this head changes the effect not at all;\n"
                     "below 1.00 = it shrinks the effect; above 1.00 = it GROWS the effect",
            size=SMALL, va="bottom")
    # Each curve is sorted by its OWN damage, so position k is a different head
    # in each model. Unsaid, the three curves read as one head index.
    ax.set_xlabel("attention heads, EACH MODEL SORTED BY ITS OWN damage (most damaging first),\n"
                  "so a given x is a different head in each curve", size=MED - 3)
    ax.set_ylabel("fraction of the first-round rule effect retained\nafter ablating that one head")
    ax.legend(loc="lower right", fontsize=SMALL)
    fig.tight_layout(rect=[0, 0.19, 1, 0.93])
    fig.text(0.5, 0.175,
             "The LEAK-READER's curve only looks steeper because its effect is near chance to begin with: "
             "a fraction of almost nothing.",
             ha="center", va="top", size=MED - 2)
    footer(fig, "Each point is one of the 48 heads (12 layers x 4 heads), zeroed one at a time, averaged over the "
                "same 350 (SOLVER) or 250 (the other two) hint-edit cases as the rule test. Values above "
                "1.00 mean ablating that head makes the model follow the edit MORE, which happens when a head is "
                "pulling toward the pre-edit answer. So there is no single circuit to point at: the rule is "
                "distributed across heads, and this is a negative result about localisation, not about DD.")
    out.savefig(fig)


def page_sae(out, S):
    """The one non-causal page: what the features themselves encode.

    Kept after the interventions on purpose. A feature that correlates with
    DD's forcing set is not evidence that the model uses it -- that is what
    pages 7-11 are for. What this page adds is WHERE the structure sits.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    a = S["leak-free 1.0000"]
    pc = a[a.position_kind == "pre_color"]
    peak = pc.loc[pc.best_justifier_f1.idxmax()]
    # The F1 floor a feature must clear to be COUNTED. Without it the right
    # panel reads as a contradiction: a curve sitting above the null while the
    # legend says zero features. Both gates are needed, and only this one shows.
    MIN_F1 = 0.30
    # The title used to read as "the model that solves the task encodes DD",
    # which the page's own legend contradicts three lines later: all three arms
    # carry these features. WHERE they sit is the only thing that separates them.
    fig.suptitle("ALL THREE models carry sparse features for DD's own forcing set. In the SOLVER they peak\n"
                 f"at L{int(peak.layer)} -- inside the L6-L10 window where the causal patch hands the fact off",
                 size=BIG - 7, weight="bold", linespacing=1.4)

    for label, _, _, colour in ARMS:
        if label not in S:
            continue
        k = S[label]
        k = k[k.position_kind == "pre_color"].sort_values("layer")
        n_tot = int(k.n_just.sum())
        # The layers that actually carry features, not "from L6 up": the counts
        # are not contiguous, and "up" asserted a run that the data does not have.
        ls = [int(x) for x in k.layer[k.n_just > 0]]
        where = f", at {layer_spans(ls)}" if n_tot else ""
        ax1.plot(k.layer, k.best_justifier_f1, "o-", color=colour, lw=3, ms=11,
                 label=f"{gloss(label)}  ({n_tot} features{where})")
    null = pd.concat([S[l][S[l].position_kind == "pre_color"]
                      for l in S]).groupby("layer").median_best_f1_null.max()
    ax1.plot(null.index, null.values, ls="--", lw=2.5, color="k",
             label="best F1 a SHUFFLED label gets")
    ax1.axhline(MIN_F1, color="0.4", ls="-.", lw=2)
    ax1.text(0.1, MIN_F1 + 0.015, f"F1 >= {MIN_F1:.2f}: the bar a feature must clear to be COUNTED",
             size=SMALL - 1, color="0.3")
    ax1.set_title("while the model picks the COLOUR", size=MED + 1)
    ax1.set_xlabel("transformer layer the dictionary was trained on", size=MED - 3)
    ax1.set_ylabel("best single feature's F1 for\n'this node is a KNOWN neighbour\nof the node being emitted'",
                   size=MED - 4)
    # Headroom to 1.35 so the legend has somewhere to sit: at ylim 1.0 the
    # three-line box landed squarely on the mid-stack peak it is describing.
    ax1.set_ylim(0, 1.35)
    ax1.legend(loc="upper left", fontsize=SMALL - 1, framealpha=0.95)

    for kind, name, ls, ms in [("pre_color", "before a COLOUR token", "-", "o"),
                               ("pre_node", "before a NODE token", ":", "s")]:
        k = a[a.position_kind == kind].sort_values("layer")
        n = int(k.n_just.sum())
        why = "" if n else " -- never clears the bar"
        ax2.plot(k.layer, k.best_justifier_f1, ls=ls, marker=ms, color=BLUE, lw=3,
                 ms=11, label=f"{name}: {n} features{why}")
    ax2.plot(null.index, null.values, ls="--", lw=2.5, color="k", label="shuffled label")
    ax2.axhline(MIN_F1, color="0.4", ls="-.", lw=2)
    ax2.set_title(f"{ROLE['leak-free 1.0000']}: the same dictionary, by position", size=MED + 1)
    # The same words as ax1: two different labels on two shared axes read as
    # two different quantities.
    ax2.set_xlabel("transformer layer the dictionary was trained on", size=MED - 3)
    ax2.set_ylim(0, 1.35)
    ax2.legend(loc="upper left", fontsize=SMALL - 1, framealpha=0.95)

    fig.tight_layout(rect=[0, 0.305, 1, 0.88])
    # The leaked arm's count and its layers, read from the same table the legend
    # reads, so this sentence cannot drift from the panel above it.
    lk = S.get("leaked 0.9960")
    if lk is not None:
        lk = lk[lk.position_kind == "pre_color"].sort_values("layer")
        lk_txt = (f"{int(lk.n_just.sum())} features, but only at "
                  f"{layer_spans([int(x) for x in lk.layer[lk.n_just > 0]])}")
    else:
        lk_txt = "features of its own"
    fig.text(0.5, 0.29,
             "A feature is COUNTED only when it matches one of DD's own predicates at F1 >= 0.30 (F1 = the harmonic "
             "mean of precision and recall)\nAND beats a shuffled-label null. Three things to read: (1) the forcing "
             "set is encoded where DD's rule needs it -- at the token that predicts\nthe COLOUR, and never at the "
             "token that picks the node; (2) the SOLVER is NOT more interpretable than the WEAK TWIN, which is "
             f"what\ntheir equal rule scores already said; (3) THE LEAK-READER ENCODES THE FORCING SET TOO -- {lk_txt}"
             ", after the\nwindow where the SOLVER uses it. Encoding it is not "
             "using it: that is why this page is not the evidence, and the four intervention pages are.",
             ha="center", va="top", size=MED - 4, linespacing=1.5)
    footer(fig, "CORRELATIONAL -- this page says WHERE the structure sits, not whether the model uses it. Sparse "
                "autoencoder, 4096 features per layer, ~64 active per token, trained on 53,200 activations per layer, "
                "then scored separately on the 26,600 positions before a colour token and the 26,600 before a node "
                "token, so counts compare within a mask only. Layer 12 is the residual stream after the last block, "
                "which is why this axis runs one further than the patching page's. One dictionary per layer per arm, "
                "one seed.")
    out.savefig(fig)


def page_verdict(out, D, O):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The verdict, and what it is not", size=BIG, weight="bold")
    a = D["leak-free 1.0000"]
    ch = a.get("chain", {}).get("corrupt_flipped")
    sk = a.get("skip", {}).get("corrupt_emits_v")
    pl = a.get("skip", {}).get("placebo_emits_v")
    bsk = D["leak-free 0.8540"].get("skip", {}).get("corrupt_emits_v")
    worst = (a["heads"].groupby(["layer", "head"]).effect_retained.mean().min()
             if "heads" in a else None)
    lk = D["leaked 0.9960"]
    lkf = lk["r0"]["flip"]
    lo, hi = wilson(int(round(lkf * lk["r0"]["n"])), lk["r0"]["n"])
    left = [
        "WHAT IT DOES DO  (causal unless marked)",
        "",
        f"* Applies DD's elimination rule: {a['r0']['flip']:.3f}",
        f"  on {a['r0']['n']} one-token counterfactuals.",
        "",
        f"* Chains it off its OWN deductions: {ch:.3f}," if ch is not None else
        "* Chaining: measuring (50077245).",
        "  with no decay across the rounds.",
        "",
        "* Guards it: break the forcing and it stops",
        f"  naming the node {1 - sk:.3f} of the time, while a"
        if sk is not None else "  (measuring, job 50082120)",
        # same direction as the line above it. Quoting the placebo as "leaves
        # it at 0.940" next to a stop rate invited reading 0.940 as a second
        # stop rate, i.e. the placebo doing the same thing as the real edit.
        f"  placebo edit stops it only {1 - pl:.3f}. But ALL" if sk is not None else "",
        "  THREE models pass this one, so passing",
        "  it is not by itself evidence of DD.",
        "",
        f"* The WEAK TWIN guards only {1 - bsk:.3f} -- the" if bsk is not None else "",
        "  first test here to separate the TWINS,",
        "  and the same order as its 0.146 solve gap.",
        "",
        "* Localised in depth: edited hint -> L5,",
        "  handoff L6-L10, prediction by L10.",
        "",
        "* CORRELATIONAL: sparse features for DD's",
        "  forcing set peak mid-stack -- but the",
        "  LEAK-READER carries them too. Ablating them",
        "  moves nothing; nor do the controls.",
    ]
    order = (["* Choose the node ORDER by deduction. DD's",
              "  own traces are in ascending index order in",
              f"  {O['ascending_frac']:.1%} of rounds ({O['n_rounds']:,} rounds), and a hand",
              "  rule that does no deduction -- 'next unknown",
              f"  index after the last one' -- scores {O['rules']['min_unknown']['max_f1']:.3f} on",
              "  the metric the model's own node features",
              "  are scored on (slide 3). WHICH nodes it",
              "  emits tracks determinacy; the ORDER does not."]
             if O else
             ["* Choose the node ORDER by deduction: the",
              "  order is ascending index in every round,",
              "  which needs no deduction at all."])
    right = [
        "WHAT IT DOES NOT DO",
        "",
        *order,
        "",
        "* Route through the adjacency table: the",
        "  graph rows recover 0.000 at every layer.",
        "",
        "* Concentrate the rule anywhere we can point",
        "  at: ablating the worst of the 48 heads one" if worst is not None else "",
        f"  at a time still leaves {worst:.0%} of the effect. It is" if worst is not None else "",
        "  distributed: this deck says THAT it runs DD,",
        "  not which heads do.",
        "",
        # NOT "never learned the rule": 0.388 against a 0.250 chance rate at
        # n=350 has a Wilson interval clear of chance, so the honest word is
        # "barely", and the deck has to use it or its own table refutes it.
        "* The LEAK-READER barely learned the rule:",
        f"  {lkf:.3f} [{lo:.2f}, {hi:.2f}], above the 0.250 chance rate",
        "  but far below the SOLVER. Shown an edited",
        f"  deduction it keeps its own answer "
        f"{lk['chain']['reverts_to_original']:.1%} of the",
        "  time: it reads the adjacency-order leak.",
    ]
    fig.text(0.5, 0.885,
             "Working picture: sweep nodes in index order; emit one only when its known neighbours "
             "leave it a\nsingle colour, counting the ones it just derived. That is DD's propagation step.",
             ha="center", va="top", size=MED, weight="bold", linespacing=1.5)
    fig.text(0.05, 0.795, "\n".join(left), ha="left", va="top", size=MED - 5,
             family="monospace", linespacing=1.3, color=GREEN)
    fig.text(0.53, 0.795, "\n".join(right), ha="left", va="top", size=MED - 5,
             family="monospace", linespacing=1.3, color=RED)
    # Deliberately NOT smaller than the two columns above it: a limitation set
    # in finer print than the claim it limits is a way of not saying it.
    # Five lines wrapped to the full page width, not seven wrapped to 70% of it:
    # at seven the block ran off the bottom edge and lost its last line, which
    # is the worst possible line of a deck to drop.
    fig.text(0.5, 0.180,
             "LIMITATIONS. The interventions show the model propagating constraints the way DD does; they do NOT pin it to DD's "
             "exact algorithm, and a different propagator with the\nsame one-step behaviour would pass every test on this deck. "
             "One checkpoint per model and no seed replicates, so every leak-free vs leaked contrast is also two\nseparate "
             "training runs. One task configuration (N = 24, degree 5, q = 4), so nothing here is a claim about depth or scale, "
             "and the per-depth bins run as thin as a few dozen\ncases. The guard gap and the solve-rate gap are different "
             "measurements on different samples: their agreement is a consistency check, not a decomposition. The\nsparse-"
             "autoencoder pages are correlational, and the ablation that would make them causal moved no answer at all, "
             "controls included -- inconclusive, not negative.",
             ha="center", va="top", size=MED - 5, color="0.15", linespacing=1.45)
    # no footer() here: the right column's last bullet already carries the
    # comparison to the previous answer, and a third block at the bottom
    # collides with the limitations line
    out.savefig(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    D = load()
    if "r0" not in D["leak-free 1.0000"]:
        raise SystemExit("arm A round-0 results missing -- job 50075339 output not found")
    S = load_sae()
    A = load_ablation()
    O = load_order()
    ex = example_instance(rho=0.60)
    pdf_path = os.path.join(OUT, "slides.pdf")
    have_skip = any("skip" in D[l] for l, _, _, _ in ARMS)
    with PdfPages(pdf_path) as pdf:
        o = PdfAndPng(pdf, OUT, total=11 + 2 * have_skip + bool(S) + ("interp" in A))
        page_title(o, D, ex, O, S)
        page_task(o, ex)
        page_model(o, O, S)
        page_tokenization(o, ex)
        page_trace(o, ex, note="The intervention edits ONE colour token: in this section "
                               "(the CHAINED EDIT) or back in the hints (the HINT EDIT).")
        page_design(o, ex)
        if have_skip:
            page_guard_design(o)
        page_table(o, D)
        page_rates(o, D)
        if have_skip:
            page_guard(o, D)
        page_patching(o, D)
        page_heads(o, D)
        if S:
            page_sae(o, S)
        if "interp" in A:
            page_ablation(o, A)
        page_verdict(o, D, O)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {o.n} pages)")


if __name__ == "__main__":
    main()
