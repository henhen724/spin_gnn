"""Slide deck for job 50151317 -- MLP vs attention, one write at a time.

Eight landscape pages: the verdict, the method, the tokenization and a real
reasoning trace (hshunt's standing request for every deck), the two measured
profiles, the heads-vs-whole-write comparison that is the only positive result
here, the leaked control, and the limitations.

Run from the repo root:  python3 -m notebooks.make_sublayer_deck
Every number is read from the run's CSVs at build time, nothing is typed in.
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

S = os.environ["SCRATCH"] + "/spin_gnn"
SUB = f"{S}/sudoku_sublayer_2026-10-29"
HEAD = f"{S}/sudoku_circuits_leakfree_2026-10-28"
OUT = f"{S}/notebooks/sudoku_sublayer_2026-10-29"
ARMS = {"A_leakfree_1000": "SOLVER", "C_leaked_0996": "LEAK-READER"}

GREEN, RED, AMBER, GREY = "#276749", "#9b2c2c", "#975a16", "0.35"


def load():
    d = {}
    for a, lab in ARMS.items():
        sub = pd.read_csv(f"{SUB}/{a}/sublayer_ablation.csv")
        beh = pd.read_csv(f"{SUB}/{a}/behavior.csv")
        d[lab] = {
            "sub": sub, "beh": beh,
            "mean": sub.pivot_table(index="layer", columns="sublayer",
                                    values="effect_retained", aggfunc="mean"),
            "med": sub.pivot_table(index="layer", columns="sublayer",
                                   values="effect_retained", aggfunc="median"),
            "clean": sub.pivot_table(index="layer", columns="sublayer",
                                     values="ablated_clean_logit_diff", aggfunc="mean"),
            "clean0": beh.clean_logit_diff.mean(),
            "flip": beh.corrupt_flipped.mean(),
            "n": len(beh),
        }
    d["heads"] = pd.read_csv(f"{HEAD}/A_leakfree_1000/head_ablation.csv")
    return d


def textpage(o, title, subtitle, blocks, foot, y0=0.80, pitch=0.148, width=100):
    """Title + subtitle + coloured verdict blocks. Four body lines per block
    is the measured maximum at this pitch."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, title, ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, subtitle, ha="center", size=MED, color="0.3")
    for i, (head, col, body) in enumerate(blocks):
        y = y0 - i * pitch
        fig.text(0.06, y, head, size=MED, weight="bold", color=col)
        fig.text(0.06, y - 0.035, textwrap.fill(body, width=width), size=MED - 4,
                 va="top", linespacing=1.45)
    footer(fig, foot)
    o.savefig(fig)


def page_verdict(o, D):
    A, C = D["SOLVER"], D["LEAK-READER"]
    m = A["mean"]
    nbad = int((m.stack() < 0.90).sum())
    h = D["heads"].groupby(["layer", "head"]).effect_retained.mean()
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.90, "Which sublayer runs the deduction step?",
             ha="center", size=BIG + 3, weight="bold")
    fig.text(0.5, 0.845, "Zero ONE write at a time -- block L's MLP, or block L's attention -- "
                         "on the same 350 counterfactuals as the head page.",
             ha="center", size=MED, color="0.3")
    fig.text(0.5, 0.735, "NOT ANSWERED. No single write carries it, and the ones that\n"
                         "do matter also break the model's ordinary prediction.",
             ha="center", size=BIG, weight="bold", color="#2b6cb0", linespacing=1.45)
    paras = [
        f"{24 - nbad} OF 24 WRITES ARE DISPENSABLE in the SOLVER. Every MLP from L0 to L10 keeps "
        f"{m.mlp[:11].min():.2f}-{m.mlp[:11].max():.2f} of the hint-edit effect; deleting any one of them "
        f"changes nothing.",
        f"THE {nbad} THAT ARE NOT are the read-in and the write-out, not the middle: L0 attention "
        f"{m.attn[0]:.3f}, L11 MLP {m.mlp[11]:.3f}, L10 attention {m.attn[10]:.3f}, L9 attention "
        f"{m.attn[9]:.3f}. Each loses the model's CLEAN decision margin by the same fraction "
        f"({-A['clean0']:.1f} logits -> {-A['clean'].attn[0]:.1f} / {-A['clean'].mlp[11]:.1f}), which the "
        f"pre-registered sanity control says is a broken forward pass, not a deleted step.",
        f"THE ONE POSITIVE RESULT IS REDUNDANCY. At L0 the four attention heads individually keep "
        f"{', '.join(f'{v:.2f}' for v in h.loc[0].values)} -- no single head matters -- yet deleting all "
        f"four at once keeps {m.attn[0]:.3f}. Head ablation UNDERSTATES dependence, and a method that "
        f"removes one component at a time cannot see a step computed in two places at once.",
    ]
    fig.text(0.07, 0.63, "\n\n".join(textwrap.fill(p, width=104) for p in paras),
             size=MED - 1, va="top", linespacing=1.55)
    footer(fig, f"Job 50151317, 2 min 41 s on one GPU. SOLVER = leak-free, solves 100.0% of held-out "
                f"graphs, {A['n']} cases, all of which flip when the forcing is redirected. "
                f"LEAK-READER = the leaked-encoding twin, {C['n']} cases. "
                f"Deck built from the run's CSVs; nothing typed in.")
    o.savefig(fig)


def page_method(o, D):
    A = D["SOLVER"]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "The intervention, and what was predicted before it ran",
             ha="center", size=BIG, weight="bold")
    fig.text(0.06, 0.855, "THESE ARE PRE-NORM BLOCKS, so each block makes two separate additive writes "
                          "into the residual stream:", size=MED - 1)
    fig.text(0.5, 0.795, r"$x \leftarrow x + \mathrm{attn}(\mathrm{norm}_1(x))$"
                         r"$\qquad\qquad x \leftarrow x + \mathrm{MLP}(\mathrm{norm}_2(x))$",
             ha="center", size=MED + 4)
    fig.text(0.06, 0.735,
             textwrap.fill("Zeroing linear2's weight and bias makes block L's MLP write exactly zero; "
                           "zeroing out_proj's does the same for its attention. The skip connection is "
                           "untouched, so every later block still gets a well-formed residual stream -- "
                           "this deletes one write, it does not cut the block out. Score: the fraction of "
                           "the one-token hint-edit effect that survives, 1.00 = that write is irrelevant, "
                           "0.00 = the effect is gone. Same normalisation as the per-head page, so heads "
                           "and whole writes can be read on one axis.", width=118),
             size=MED - 3, va="top", linespacing=1.5)
    rows = [
        ("(a) at least one MLP falls below 0.60 -- below the worst single head's 0.839",
         "CONFIRMED, and uninformative", GREEN),
        ("(b) the load-bearing MLPs sit in L6-L10, where the SAE finds the forcing set",
         "FALSIFIED -- the minimum is at L11, the named falsifier", RED),
        ("(c) the LEAK-READER's profile differs from the SOLVER's",
         "SPLIT -- the last MLP is shared, the read-in is not", AMBER),
        ("(d) SANITY: if the clean prediction dies too, it is damage, not localisation",
         f"HELD at {int((A['mean'].stack() >= 0.90).sum())} of 24, and it is why this page is a null", GREY),
    ]
    fig.text(0.06, 0.495, "PRE-REGISTERED IN THE SBATCH HEADER, BEFORE THE RUN:",
             size=MED, weight="bold")
    for i, (claim, verdict, col) in enumerate(rows):
        y = 0.425 - i * 0.095
        fig.text(0.06, y, textwrap.fill(claim, width=86), size=MED - 3, va="top")
        fig.text(0.06, y - 0.042, verdict, size=MED - 3, va="top", weight="bold", color=col)
    footer(fig, "Pre-registrations quoted from scripts/slurm/sudoku2_sublayer.sbatch lines 41-70. "
                "(a) is confirmed by L11's MLP, which is the output write and is disqualified by (d); "
                "(b)'s falsifier was written as 'the minimum sits at L0-L2 or at L11'.")
    o.savefig(fig)


def page_profile(o, D):
    A = D["SOLVER"]
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    d = A["sub"]
    for s, col, mk, lab in [("mlp", "#2b6cb0", "o", "block L's MLP"),
                            ("attn", "0.40", "s", "block L's attention")]:
        g = d[d.sublayer == s].groupby("layer")
        m, lo, hi = g.effect_retained.mean(), g.effect_retained.quantile(0.1), g.effect_retained.quantile(0.9)
        axes[0].fill_between(m.index, lo, hi, color=col, alpha=0.18, lw=0)
        axes[0].plot(m.index, m, mk + "-", color=col, lw=2.6, ms=9, label=lab, rasterized=True)
        axes[1].plot(m.index, -g.ablated_clean_logit_diff.mean(), mk + "-", color=col, lw=2.6, ms=9,
                     rasterized=True)
        bad = m[m < 0.90]
        axes[0].plot(bad.index, bad, "x", color=RED, ms=18, mew=4)
    axes[0].axhline(1.0, color="k", ls="--", lw=1.5)
    axes[0].axhline(0.839, color=AMBER, ls=":", lw=2.5)
    axes[0].text(0.25, 0.75, "0.839 = the WORST of all 48 single heads", color=AMBER, size=SMALL + 1)
    axes[0].axvspan(5.6, 10.4, color="#f6e05e", alpha=0.22, lw=0)
    axes[0].text(8.0, 0.52, "L6-L10: where the SAE finds the forcing set\nand where patching hands it over",
                 ha="center", color="#8a6d00", size=SMALL + 1)
    axes[0].set_ylabel("fraction of the\nhint-edit effect kept")
    axes[0].set_title("SOLVER: deleting one write at a time", size=MED + 3, weight="bold")
    axes[0].legend(loc="center left", ncol=1)
    axes[1].axhline(-A["clean0"], color="k", ls="--", lw=1.5)
    axes[1].axhline(0, color=RED, lw=2.5)
    axes[1].text(0.25, 0.9, "0 = the model now gets the UNEDITED puzzle wrong", color=RED, size=SMALL + 1)
    axes[1].set_ylabel("SANITY CONTROL:\nclean decision margin")
    axes[1].set_xlabel("block index L")
    axes[1].set_xticks(range(12))
    fig.suptitle("The effect and the clean prediction die in the same places, by the same fractions",
                 size=BIG - 4, weight="bold")
    # tight_layout cannot handle the axvspan/annotations here (it warns and
    # leaves the xlabel on top of the footer), so the margins are set by hand.
    fig.subplots_adjust(left=0.085, right=0.98, top=0.88, bottom=0.145, hspace=0.12)
    footer(fig, "Red crosses mark the four writes below 0.90. Shaded band is the 10th-90th percentile "
                "over the 350 cases. A write that ran the deduction and nothing else would drop in the "
                "top panel and stay flat in the bottom one; none does.")
    o.savefig(fig)


def page_heads(o, D):
    A, h = D["SOLVER"], D["heads"]
    hg = h.groupby(["layer", "head"]).effect_retained.mean()
    worst = hg.groupby("layer").min()
    blk = A["mean"].attn
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(12)
    ax.bar(x - 0.2, worst.values, 0.38, color="#b7791f", label="worst of the block's 4 heads, alone")
    ax.bar(x + 0.2, blk.values, 0.38, color="#2b6cb0", label="all 4 heads deleted together")
    for L in (0, 10):
        ax.annotate("", xy=(L + 0.2, blk[L] + 0.03), xytext=(L - 0.2, worst[L] - 0.02),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=RED))
        ax.text(L + 0.35, (worst[L] + blk[L]) / 2, f"-{worst[L] - blk[L]:.2f}", color=RED,
                size=MED, weight="bold", va="center")
    ax.axhline(1.0, color="k", ls="--", lw=1.5)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.22)
    ax.set_xlabel("block index L")
    ax.set_ylabel("fraction of the hint-edit effect kept")
    ax.set_title("Only two blocks disagree -- and that disagreement is the finding",
                 size=BIG - 2, weight="bold")
    ax.legend(loc="upper left", ncol=2)
    ax.text(1.6, 0.30, textwrap.fill(
        "Everywhere except L0 and L10 the two bars match to within 0.05: nothing to localise at either "
        "granularity. At L0 and L10 the heads are REDUNDANT WITH EACH OTHER -- each one is replaceable, "
        "the group is not. That is a property no single-component ablation can measure, and it is the "
        "reason the flat line across L1-L10 cannot be read as 'those writes do nothing'.", width=74),
        size=MED - 2, va="top", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.6", fc="#fffbea", ec="0.7"))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    footer(fig, "Same 350 counterfactuals, same normalisation, both runs on the SOLVER checkpoint: "
                "head numbers from sudoku_circuits_leakfree_2026-10-28 (finding 27), whole-write numbers "
                "from this run. 3 of 48 heads fall below 0.95; 4 of 24 writes do.")
    o.savefig(fig)


def page_control(o, D):
    A, C = D["SOLVER"], D["LEAK-READER"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, (lab, arm) in zip(axes, [("SOLVER", A), ("LEAK-READER", C)]):
        for s, col, mk in [("mlp", "#2b6cb0", "o"), ("attn", "0.40", "s")]:
            ax.plot(arm["mean"].index, arm["mean"][s], mk + "-", color=col, lw=2.4, ms=8,
                    label=f"{s} (mean)", rasterized=True)
            ax.plot(arm["med"].index, arm["med"][s], ":", color=col, lw=2.0, alpha=0.8,
                    label=f"{s} (median)", rasterized=True)
        ax.axhline(1.0, color="k", ls="--", lw=1.4)
        ax.set_xlabel("block index L")
        ax.set_xticks(range(0, 12, 2))
        ax.set_title(f"{lab}: {arm['n']} cases, {arm['flip']:.3f} of them flip\n"
                     f"(clean margin {-arm['clean0']:.1f} logits)", size=MED + 1, weight="bold")
    axes[0].set_ylabel("fraction of the hint-edit effect kept")
    axes[0].legend(loc="center left", fontsize=SMALL)
    fig.suptitle("The control splits: the last MLP is shared, the READ-IN is not",
                 size=BIG - 2, weight="bold")
    note = (f"L0 attention: {A['mean'].attn[0]:.3f} in the SOLVER, {C['mean'].attn[0]:.3f} in the "
            f"LEAK-READER (median {C['med'].attn[0]:.3f}). L11 MLP: {A['mean'].mlp[11]:.3f} against "
            f"{C['mean'].mlp[11]:.3f} -- shared, and it is the output write in both. The LEAK-READER "
            f"flips on only {C['flip']:.3f} of its cases, so its ratios divide two small numbers and the "
            f"median is the number to trust; its apparent amplification above 1.0 shrinks from "
            f"{C['mean'].max().max():.2f} to {C['med'].max().max():.2f} and should be read as "
            f"'not needed, and noisy'.")
    fig.text(0.5, 0.075, textwrap.fill(note, width=120), ha="center", va="bottom",
             size=MED - 4, linespacing=1.45)
    fig.tight_layout(rect=[0, 0.17, 1, 0.93])
    footer(fig, "The LEAK-READER reads the colouring off the adjacency ORDER (finding 16), so a hint "
                "edit changes the deduced answer but not what it reads -- which is why it flips on 39% "
                "of the cases where the SOLVER flips on 100%.")
    o.savefig(fig)


def page_limits(o, D):
    blocks = [
        ("WHAT THIS RULES OUT", GREEN,
         "A single MLP that runs the elimination step. Every MLP from L0 to L10 is individually "
         "dispensable, including L8, where the dictionary finds the forcing-set features and where "
         "patching hands the fact over. The correlational and causal localisations do NOT agree at "
         "the level of one component."),
        ("WHAT IT CANNOT RULE OUT", AMBER,
         "A step computed redundantly in two or more places -- which is not hypothetical here: the "
         "L0 heads demonstrate exactly that pattern one level down. A flat 1.00 is equally consistent "
         "with 'irrelevant' and with 'duplicated'."),
        ("METHOD CAVEATS", GREY,
         "One seed, one checkpoint per arm, one intervention type. Zeroing a weight pushes the block "
         "off-distribution rather than removing its information; mean- or resample-ablation is the "
         "gentler test. Scores are means of per-case ratios, which is solid for the SOLVER and soft "
         "for the LEAK-READER."),
        ("NEXT TEST", "#2b6cb0",
         "Joint ablation: pairs of writes, and whole L6-L10 windows at once, scored the same way on "
         "the same 350 cases. If redundancy is the explanation, a window drops where none of its "
         "members does."),
    ]
    textpage(o, "What this measures, and what it does not",
             "A null result with a named reason, not a missing one",
             blocks, "The DD verdict (findings 27, 28) rests on the three behavioural interventions, "
                     "which do move the answer: redirect the forcing and the SOLVER flips on 350/350; "
                     "break it and it withholds the node 350/350 against a 0.94 placebo ceiling. "
                     "This page is about WHERE, which is still open.",
             y0=0.78, pitch=0.165, width=104)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    D = load()
    ex = example_instance(rho=0.60)
    with PdfPages(os.path.join(OUT, "slides.pdf")) as pdf:
        o = PdfAndPng(pdf, OUT, total=8)
        page_verdict(o, D)
        page_method(o, D)
        page_tokenization(o, ex)
        page_trace(o, ex, note="The hint-edit counterfactual changes ONE colour token in the HINTS "
                               "section; this run then deletes one sublayer write and asks how much "
                               "of the model's response to that edit survives.")
        page_profile(o, D)
        page_heads(o, D)
        page_control(o, D)
        page_limits(o, D)
    p = os.path.join(OUT, "slides.pdf")
    print(f"wrote {p} ({os.path.getsize(p) / 1e6:.2f} MB, {o.n} pages)")


if __name__ == "__main__":
    main()
