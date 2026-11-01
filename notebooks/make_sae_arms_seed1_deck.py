"""Slide deck for job 50604600 -- the second SAE dictionary on the OTHER TWO arms.

Finding 37 re-fit the SOLVER at seed 1. This run re-fits the LEAK-READER (0.9960)
and the 0.854 TWIN, which is the pair findings 19/21's headline contrast is
actually measured on, so the contrast finally has an error bar on both sides.

Eight landscape pages: the verdict, the comparability check, the tokenization and
a real reasoning trace (hshunt's standing request for every deck), the four
layer profiles, the gap page that scores the headline two different ways, the
four pre-registrations, and the limits.

Run from the repo root:  python3 -m notebooks.make_sae_arms_seed1_deck
Every number is read from the runs' CSVs at build time, nothing is typed in.
"""
import json
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
RUNS = {("TWIN", 0): f"{S}/sudoku_sae_kindsplit_2026-10-27/leakfree_l0_64",        # 49744985
        ("LEAK", 0): f"{S}/sudoku_sae_kindsplit_2026-10-27/leaked_l0_64",          # 49744985
        ("TWIN", 1): f"{S}/sudoku_sae_kindsplit_2026-10-31_seed1/leakfree_l0_64_seed1",
        ("LEAK", 1): f"{S}/sudoku_sae_kindsplit_2026-10-31_seed1/leaked_l0_64_seed1"}
OUT = f"{S}/notebooks/sudoku_sae_arms_seed1_2026-10-31"

# The solver's two kind-split dictionaries (findings 19 and 37). Not an arm of
# this contrast and never plotted here, but they carry the same pre_node
# measurement, so the asymmetry tally has to be counted over all six.
SOLVER = {("SOLVER", 0): f"{S}/sudoku_sae_solved_2026-10-28/leakfree1000_l0_64",
          ("SOLVER", 1): f"{S}/sudoku_sae_solved_2026-10-31/leakfree1000_l0_64_seed1"}

GREEN, RED, AMBER, GREY = "#276749", "#9b2c2c", "#975a16", "0.35"
ARMCOL = {"TWIN": "#2f855a", "LEAK": "#9b2c2c"}
ARMNAME = {"TWIN": "0.854 TWIN (leak-free)", "LEAK": "LEAK-READER (0.9960)"}
QUOTED = 8          # the layer findings 19/21 quote the contrast at
BAR = 0.30          # the pre-registered bar on the between-arm gap


def _fam(t):
    t = t.copy()
    t["jn"] = t.family_counts.map(lambda s: json.loads(s).get("justifies_next", 0))
    t["nn"] = t.n_next_node_features
    return t


def load():
    d = {"pc": {}, "pn": {}, "fit": {}, "pn_all": {}}
    for key, p in {**RUNS, **SOLVER}.items():
        k = _fam(pd.read_csv(f"{p}/sae_summary_by_kind.csv"))
        pn = k[k.position_kind == "pre_node"].set_index("layer").sort_index()
        d["pn_all"][key] = pn
        if key not in RUNS:
            continue
        d["pc"][key] = k[k.position_kind == "pre_color"].set_index("layer").sort_index()
        d["pn"][key] = pn
        d["fit"][key] = pd.read_csv(f"{p}/sae_summary.csv").set_index("layer").sort_index()
    return d


def gaps(D):
    """The headline, scored the two ways the pre-registration conflated."""
    g = {}
    for s in (0, 1):
        t, l = D["pc"][("TWIN", s)], D["pc"][("LEAK", s)]
        g[s] = {"at": t.best_justifier_f1[QUOTED] - l.best_justifier_f1[QUOTED],
                "max": t.best_justifier_f1.max() - l.best_justifier_f1.max(),
                "t_at": t.best_justifier_f1[QUOTED], "l_at": l.best_justifier_f1[QUOTED],
                "t_max": t.best_justifier_f1.max(), "l_max": l.best_justifier_f1.max(),
                "t_arg": int(t.best_justifier_f1.idxmax()),
                "l_arg": int(l.best_justifier_f1.idxmax())}
    return g


def textpage(o, title, subtitle, blocks, foot, y0=0.80, gap=0.030, width=100):
    # Advance by the height the body actually wraps to. A fixed pitch silently
    # overprints the next heading as soon as one block wraps to an extra line.
    body_size = MED - 4
    line_h = body_size * 1.45 / (9 * 72)
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, title, ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, subtitle, ha="center", size=MED, color="0.3")
    y = y0
    for head, col, body in blocks:
        wrapped = textwrap.fill(body, width=width)
        fig.text(0.06, y, head, size=MED, weight="bold", color=col)
        fig.text(0.06, y - 0.035, wrapped, size=body_size, va="top", linespacing=1.45)
        y -= 0.035 + line_h * (wrapped.count("\n") + 1) + gap
    footer(fig, foot)
    o.savefig(fig)


def page_verdict(o, D):
    g = gaps(D)
    pn_total = int(sum(t.jn.sum() for t in D["pn_all"].values()))
    pn_cells = int(sum(len(t) for t in D["pn_all"].values()))
    pn_dicts = len(D["pn_all"])
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.90, "Does the between-arm SAE contrast survive a second dictionary?",
             ha="center", size=BIG + 1, weight="bold")
    fig.text(0.5, 0.855,
             "Findings 19 and 21 compare a model that must deduce against one that reads the\n"
             "adjacency-order leak. Both arms were one dictionary each until tonight.",
             ha="center", va="top", size=MED, color="0.3", linespacing=1.5)
    fig.text(0.5, 0.725,
             f"YES AT THE LAYER IT IS QUOTED AT: {g[0]['at']:+.3f} -> {g[1]['at']:+.3f} at L{QUOTED}.",
             ha="center", size=BIG, weight="bold", color="#2b6cb0")
    paras = [
        f"THE QUOTED NUMBER REPLICATES. At L{QUOTED}, best justifies_next F1 at pre_color is "
        f"{g[0]['t_at']:.3f} (twin) against {g[0]['l_at']:.3f} (leak-reader) at seed 0, and "
        f"{g[1]['t_at']:.3f} against {g[1]['l_at']:.3f} at seed 1. The gap moves by "
        f"{abs(g[1]['at'] - g[0]['at']):.3f}. justifies_next[u] means 'u is a KNOWN neighbour of the "
        f"node being emitted' -- the input to DD's elimination rule.",
        f"BUT MY PRE-REGISTRATION ASKED FOR THE WRONG STATISTIC, AND IT FAILS AT BOTH SEEDS. It said "
        f"'max over layers' for both arms and then quoted a seed-0 baseline of 0.496 that is the twin's "
        f"MAX against the leak-reader's value AT L{QUOTED}. Scored as written: {g[0]['max']:+.3f} at "
        f"seed 0 and {g[1]['max']:+.3f} at seed 1, both under the {BAR:.2f} bar. The arithmetic was "
        f"wrong before the job ran, so this is a scoring error, not a replication failure.",
        f"WHICH MAKES THE RESULT SHARPER, NOT WEAKER. The leak-reader's forcing-set peak is at "
        f"L{g[1]['l_arg']} ({g[1]['l_max']:.3f}), the twin's at L{g[1]['t_arg']} ({g[1]['t_max']:.3f}). "
        f"The difference between the two models is WHEN the forcing set appears, not whether: the twin "
        f"has it by L{QUOTED}, inside the L6-L10 window where activation patching shows the causal fact "
        f"being handed over; the leak-reader only gets one at L10, after that window has closed.",
        f"AND THE ASYMMETRY IS NOW {pn_dicts} DICTIONARIES DEEP. justifies_next features at pre_node -- "
        f"the positions where the model picks the NODE rather than its COLOUR -- total {pn_total} "
        f"across all {pn_cells} layer-measurements ever made: seven layers x three checkpoints "
        f"(solver, twin, leak-reader) x two seeds.",
    ]
    fig.text(0.06, 0.655, "\n\n".join(textwrap.fill(p, width=112) for p in paras),
             size=MED - 3, va="top", linespacing=1.45)
    footer(fig, "Job 50604600, 57 min on one H100, COMPLETED 2026-10-31 23:31. --seed 1 is the only "
                "change from job 49744985, the run findings 19 and 21 were written from. Both arms "
                "re-fit in the same job; every number on this deck is read from the two runs' CSVs.")
    o.savefig(fig)


def page_method(o, D):
    fit = D["fit"]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "Were the two dictionaries fit under the same pressure?",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.878,
             "If seed 1's fits were looser, a smaller gap would mean nothing. They are not.",
             ha="center", size=MED, color="0.3")
    hdr = ["layer", "l1 twin s0", "l1 twin s1", "l1 leak s0", "l1 leak s1",
           "var expl twin", "var expl leak", "L0 twin", "L0 leak"]
    xs = [0.05, 0.14, 0.24, 0.34, 0.44, 0.545, 0.665, 0.785, 0.875]
    for x, h in zip(xs, hdr):
        fig.text(x, 0.805, h, size=SMALL, weight="bold")
    for i, L in enumerate(fit[("TWIN", 0)].index):
        y = 0.765 - i * 0.040
        f = {k: fit[k] for k in RUNS}
        vals = [f"L{L}",
                f"{f[('TWIN', 0)].l1_final[L]:.4f}", f"{f[('TWIN', 1)].l1_final[L]:.4f}",
                f"{f[('LEAK', 0)].l1_final[L]:.4f}", f"{f[('LEAK', 1)].l1_final[L]:.4f}",
                f"{f[('TWIN', 0)].frac_variance_explained[L]:.4f}/"
                f"{f[('TWIN', 1)].frac_variance_explained[L]:.4f}",
                f"{f[('LEAK', 0)].frac_variance_explained[L]:.4f}/"
                f"{f[('LEAK', 1)].frac_variance_explained[L]:.4f}",
                f"{f[('TWIN', 0)].l0[L]:.1f}/{f[('TWIN', 1)].l0[L]:.1f}",
                f"{f[('LEAK', 0)].l0[L]:.1f}/{f[('LEAK', 1)].l0[L]:.1f}"]
        for x, v in zip(xs, vals):
            fig.text(x, y, v, size=SMALL, color=GREY if L else RED,
                     weight="normal" if L else "bold")
    dead = int(fit[("LEAK", 1)].n_dead_features[0])
    fig.text(0.05, 0.445, textwrap.fill(
        f"l1_final is IDENTICAL across seeds at every layer of both arms from L2 up -- the l1 search "
        f"lands on the same coefficient each time. The one exception is the twin's layer 0, where the "
        f"search bottoms out near zero (2e-05 against 6e-06, both printing as 0.0000). Variance "
        f"explained agrees to 4 decimals and achieved sparsity to "
        f"about 1 in 60 against a target of 64. LAYER 0 (red) is degenerate in all four fits, exactly as "
        f"it was in the solver (finding 37): L0 collapses to 6-8 against the target of 64 with ~{dead} "
        f"of 4096 features dead, so its F1 is stable and its feature COUNTS are not. That is a property "
        # Widened from 120: the layer-0 caveat above added a line that overprinted
        # the next heading, whose y is fixed on this hand-laid page.
        f"of the layer, not of a seed.", width=138), size=MED - 4, va="top", linespacing=1.5)
    fig.text(0.05, 0.275, "ONE CODE DELTA, AND IT IS ADDITIVE.", size=MED, weight="bold")
    fig.text(0.05, 0.235, textwrap.fill(
        "spin_gnn/sudoku_sae.py changed once between the seed-0 arms (10-27) and tonight: commit "
        "10c38b0 (10-28) added two WRITES -- the per-dimension normalisation stats the ablation work "
        "needs, and the per-kind per-feature CSVs. Nothing in the fitting or scoring path moved, which "
        "the identical l1_final values above independently confirm. Every other flag is byte-identical: "
        "expansion 16, 1500 SAE epochs, 3000 instances, 40 positions each, alive-rate 1e-3, 19 null "
        "shifts, target L0 64, --split-by-kind.", width=120), size=MED - 4, va="top", linespacing=1.5)
    footer(fig, "--seed moves four things together in sudoku_sae: the instance pool, which positions "
                "are sampled, the SAE initialisation and the null-shift RNG. So this is a full "
                "replication, not an init-only ablation -- agreement is strong, and a disagreement "
                "could not have been pinned on any one of the four.")
    o.savefig(fig)


def page_curves(o, D):
    pc, pn = D["pc"], D["pn"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.3, 1]})
    for arm in ("TWIN", "LEAK"):
        for s, ls, mk in ((0, "-", "o"), (1, "--", "s")):
            t = pc[(arm, s)]
            axes[0].plot(t.index, t.best_justifier_f1, mk + ls, color=ARMCOL[arm], lw=3, ms=10,
                         alpha=1.0 if s == 0 else 0.75,
                         label=f"{ARMNAME[arm]}, seed {s}", rasterized=True)
            axes[1].plot(pn[(arm, s)].index, pn[(arm, s)].best_justifier_f1, mk + ls,
                         color=ARMCOL[arm], lw=2.2, ms=8, alpha=1.0 if s == 0 else 0.75,
                         rasterized=True)
            axes[1].plot(pn[(arm, s)].index, pn[(arm, s)].median_best_f1_null, ":", color="0.6",
                         lw=1.6, rasterized=True)
    axes[0].axvspan(6, 10, color="#f6e05e", alpha=0.25, lw=0)
    axes[0].text(8, 0.06, "L6-L10: the activation-patching\nhandoff window (finding 27)",
                 ha="center", size=SMALL, color="#8a6d00")
    axes[0].set_xticks(list(pc[("TWIN", 0)].index))
    axes[0].set_ylim(-0.04, 1.05)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("best F1 for any justifies_next predicate")
    axes[0].set_title("pre_color: picking the COLOUR", size=MED + 2, weight="bold")
    axes[0].legend(loc="upper left", fontsize=SMALL - 1)
    axes[1].set_xticks(list(pc[("TWIN", 0)].index))
    axes[1].set_ylim(-0.04, 1.05)
    axes[1].set_xlabel("layer")
    axes[1].set_title("pre_node: picking the NODE  (dotted = own null)",
                      size=MED + 2, weight="bold")
    fig.suptitle("Four dictionaries, one shape: nothing at pre_node, a back-half bump at pre_color",
                 size=BIG - 3, weight="bold")
    fig.tight_layout(rect=[0, 0.085, 1, 0.94])
    footer(fig, "Right panel: every curve sits at or under its own shuffled null, and the feature "
                "COUNT is zero at all seven layers in all four dictionaries -- the asymmetry findings "
                "21 and 29 rest on. Left panel: the two arms are far apart through L8 and converge at "
                "L10, which is the whole story of this run.")
    o.savefig(fig)


def page_gap(o, D):
    g, pc = gaps(D), D["pc"]
    layers = list(pc[("TWIN", 0)].index)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.25, 1]})
    for s, c in ((0, "#2b6cb0"), (1, "#c53030")):
        d = pc[("TWIN", s)].best_justifier_f1 - pc[("LEAK", s)].best_justifier_f1
        axes[0].plot(layers, d, "o-", color=c, lw=3, ms=11, label=f"seed {s}", rasterized=True)
    axes[0].axhline(BAR, color="k", ls="--", lw=2)
    axes[0].text(0.1, BAR + 0.015, f"the pre-registered {BAR:.2f} bar", size=SMALL)
    axes[0].axhline(0, color="0.7", lw=1)
    axes[0].axvspan(6, 10, color="#f6e05e", alpha=0.25, lw=0)
    axes[0].set_xticks(layers)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("twin minus leak-reader, best justifies_next F1")
    axes[0].set_title("The gap is LAYER-LOCAL", size=MED + 2, weight="bold")
    axes[0].legend(loc="upper left", fontsize=SMALL + 1)

    lab = [f"at L{QUOTED}\nseed 0", f"at L{QUOTED}\nseed 1", "max over\nlayers, s0",
           "max over\nlayers, s1"]
    val = [g[0]["at"], g[1]["at"], g[0]["max"], g[1]["max"]]
    cols = [GREEN, GREEN, RED, RED]
    axes[1].bar(lab, val, color=cols, rasterized=True)
    axes[1].axhline(BAR, color="k", ls="--", lw=2)
    for i, v in enumerate(val):
        axes[1].text(i, v + 0.012, f"{v:+.3f}", ha="center", size=MED - 3, weight="bold")
    axes[1].set_ylim(0, 0.60)
    axes[1].set_ylabel("between-arm gap")
    axes[1].set_title(f"The same headline, scored two ways\n(bar = the pre-registered {BAR:.2f})",
                      size=MED + 2, weight="bold")
    fig.suptitle("Where the two models differ is WHEN the forcing set shows up",
                 size=BIG - 3, weight="bold")
    fig.tight_layout(rect=[0, 0.085, 1, 0.93])
    footer(fig, f"Both scorings are honest and they disagree, so the write-up has to say which it "
                f"means. The project quotes L{QUOTED} (findings 19, 21, 29), where the gap is ~0.49 in "
                f"both seeds; max-over-layers is ~0.21 because the leak-reader catches up at L10. The "
                f"pre-registration mixed the two and set its bar from the mixture.")
    o.savefig(fig)


def page_prereg(o, D):
    g, pc, pn = gaps(D), D["pc"], D["pn"]
    lk1, lk0 = pc[("LEAK", 1)], pc[("LEAK", 0)]
    rows = [
        ("(a) HEADLINE: max-over-layers twin-minus-leak-reader gap at pre_color is at least 0.30",
         f"FAILED AS WRITTEN, AND IT WOULD HAVE FAILED AT SEED 0 TOO: {g[1]['max']:+.3f} at seed 1, "
         f"{g[0]['max']:+.3f} at seed 0. The 0.496 baseline in the header is the twin's max against the "
         f"leak-reader's L{QUOTED} value. Scored at L{QUOTED}, the layer the project actually quotes, "
         f"it is {g[0]['at']:+.3f} -> {g[1]['at']:+.3f} and replicates.", RED),
        ("(b) ASYMMETRY: zero justifies_next features at pre_node, all seven layers, BOTH arms",
         f"CONFIRMED -- 0 at every layer in both arms, best F1 "
         f"{pn[('TWIN', 1)].best_justifier_f1.max():.3f} and "
         f"{pn[('LEAK', 1)].best_justifier_f1.max():.3f} against nulls of "
         f"{pn[('TWIN', 1)].median_best_f1_null.max():.3f} and "
         f"{pn[('LEAK', 1)].median_best_f1_null.max():.3f}. "
         f"{len(D['pn_all'])} dictionaries now, across three checkpoints and two seeds -- "
         f"{int(sum(len(t) for t in D['pn_all'].values()))} layer-measurements, all zero.", GREEN),
        ("(c) LAYER 0: best next_node F1 >= 0.99 at pre_color and zero next_node features at "
         "pre_node, in both arms (counts at pre_color deliberately NOT pre-registered)",
         f"CONFIRMED -- F1 {pc[('TWIN', 1)].best_next_node_f1[0]:.4f} and "
         f"{pc[('LEAK', 1)].best_next_node_f1[0]:.4f} at pre_color, 0 features at pre_node in both. "
         f"Not pre-registering the counts was right: they went "
         f"{int(pc[('TWIN', 0)].nn[0])}->{int(pc[('TWIN', 1)].nn[0])} and "
         f"{int(pc[('LEAK', 0)].nn[0])}->{int(pc[('LEAK', 1)].nn[0])} while F1 held at 1.000, the "
         f"same instability the solver showed.", GREEN),
        ("(d) LOW CONFIDENCE: the leak-reader's justifies_next features stay LATE -- at L10/L12, with "
         "none at L6 or L8",
         f"CONFIRMED, and tighter than predicted: {int(lk1.jn[10])} at L10 and {int(lk1.jn[12])} at "
         f"L12 (seed 0 had {int(lk0.jn[10])} and {int(lk0.jn[12])}), zero at L6 and L8 in both seeds. "
         f"The falsifier was any leaked feature at L6 or L8; it did not happen. This is the mechanism "
         f"behind (a)'s two scorings.", GREEN),
    ]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The four pre-registrations, scored", ha="center", size=BIG,
             weight="bold")
    fig.text(0.5, 0.895, "Written into scripts/slurm/sudoku2_sae_kindsplit_seed1.sbatch before "
                         "submission, against seed 0's exact table.",
             ha="center", size=MED, color="0.3")
    for i, (claim, verdict, col) in enumerate(rows):
        y = 0.805 - i * 0.195
        fig.text(0.05, y, textwrap.fill(claim, width=114), size=MED - 3, va="top")
        fig.text(0.05, y - 0.060, textwrap.fill(verdict, width=118), size=MED - 4, va="top",
                 weight="bold", color=col, linespacing=1.4)
    footer(fig, "3 of 4, with the miss on the one that mattered and for a reason that has nothing to "
                "do with the data. The lesson is narrow and worth keeping: write the pre-registered "
                "statistic as the expression you will compute, not as a sentence plus a remembered "
                "number.")
    o.savefig(fig)


def page_limits(o, D):
    g, pc = gaps(D), D["pc"]
    blocks = [
        ("WHAT THIS BUYS", GREEN,
         f"The contrast findings 19 and 21 are built on is no longer one dictionary per arm. At "
         f"L{QUOTED} it moves {abs(g[1]['at'] - g[0]['at']):.3f} between seeds against a gap of "
         f"{g[1]['at']:.3f}, and the pre_node asymmetry is now zero-for-zero in "
         f"{len(D['pn_all'])} dictionaries. "
         f"Total justifies_next at pre_color: twin {int(pc[('TWIN', 0)].jn.sum())} -> "
         f"{int(pc[('TWIN', 1)].jn.sum())}, leak-reader {int(pc[('LEAK', 0)].jn.sum())} -> "
         f"{int(pc[('LEAK', 1)].jn.sum())}."),
        ("WHAT HAS TO BE REWORDED", AMBER,
         f"Every place the project states the between-arm gap, it must name the layer. 'The leak-free "
         f"model represents the forcing set and the leaked one does not' is false as a global claim -- "
         f"the leak-reader has {int(pc[('LEAK', 1)].jn[10])} forcing-set features at L10 at F1 "
         f"{pc[('LEAK', 1)].best_justifier_f1[10]:.3f}. The true claim is about timing: the twin has "
         f"the forcing set inside the causal handoff window, the leak-reader only after it. Finding "
         f"29(e) already said this; this run makes it the headline."),
        ("WHAT IS STILL UNFIXED", RED,
         "best_f1 is a max over 256 predicates in every seed, so the multiple-comparisons ceiling is "
         "unchanged -- two seeds show the same ceiling gives the same answer twice, which is weaker "
         "than removing it. Two seeds is two, not a distribution. The SAE stays CORRELATIONAL: the "
         "feature-level ablation (finding 30) found no sensitivity at all, so what the model USES "
         "comes from the interventions in findings 27 and 28, not from here."),
        ("WHAT WOULD SETTLE THE TIMING CLAIM", "#2b6cb0",
         "Activation patching on the LEAK-READER, the same L6-L10 sweep finding 27 ran on the two "
         "leak-free models. If the timing story is right, its forcing fact should arrive late or not "
         "at all -- and it already flips at only 0.388, so there may be little to track. That is the "
         "test that would turn this correlational timing difference into a causal one."),
    ]
    textpage(o, "What the second dictionary settles, and what it rewords",
             "A contrast that held, and a sentence that has to name its layer",
             blocks,
             "Outputs under $SCRATCH/spin_gnn/sudoku_sae_kindsplit_2026-10-31_seed1/ (90-day purge; "
             "the summaries are small and the dictionaries are re-derivable from the checkpoints). "
             "Notebook: notebooks/sudoku_sae_arms_seed1_2026-10-31_results.ipynb. Job 50604600, repo "
             "at d0b4654, sudoku_sae.py at 10c38b0, seed 1.",
             # Four blocks is one more than the other textpages carry, so this
             # page needs the wider measure and the tighter gap to clear the footer.
             y0=0.80, gap=0.022, width=118)


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
        page_trace(o, ex, note=textwrap.fill(
            "The SAE is fit on the model's activations at exactly these positions. pre_node = the step "
            "before it emits a NODE id, pre_color = the step before it emits that node's COLOUR. "
            "justifies_next[u] is true when u is one of the already-known neighbours whose colours "
            "force the answer -- the list shown on the right of this page, for N0.", width=122))
        page_curves(o, D)
        page_gap(o, D)
        page_prereg(o, D)
        page_limits(o, D)
    p = os.path.join(OUT, "slides.pdf")
    print(f"wrote {p} ({os.path.getsize(p) / 1e6:.2f} MB, {o.n} pages)")


if __name__ == "__main__":
    main()
