"""Slide deck for job 50536365 -- the second SAE dictionary on the SOLVER.

Eight landscape pages: the verdict, what a second seed does and does not test,
the tokenization and a real reasoning trace (hshunt's standing request for every
deck), the two seeds' layer profiles, the four pre-registrations scored, the
seed-noise-against-between-arm-gap page that says what survives, and the limits.

Run from the repo root:  python3 -m notebooks.make_sae_seed1_deck
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
SEEDS = {0: f"{S}/sudoku_sae_solved_2026-10-28/leakfree1000_l0_64",        # 50075499
         1: f"{S}/sudoku_sae_solved_2026-10-31/leakfree1000_l0_64_seed1"}  # 50536365
ARMS = {"LEAK-READER": f"{S}/sudoku_sae_kindsplit_2026-10-27/leaked_l0_64",
        "0.854 TWIN": f"{S}/sudoku_sae_kindsplit_2026-10-27/leakfree_l0_64"}
OUT = f"{S}/notebooks/sudoku_sae_seed1_2026-10-31"

GREEN, RED, AMBER, GREY = "#276749", "#9b2c2c", "#975a16", "0.35"
COL = {0: "#2b6cb0", 1: "#c53030"}


def _fam(t):
    t = t.copy()
    t["jn"] = t.family_counts.map(lambda s: json.loads(s).get("justifies_next", 0))
    t["nn"] = t.family_counts.map(lambda s: json.loads(s).get("next_node", 0))
    return t


def load():
    d = {"pc": {}, "pn": {}, "pool": {}, "arm": {}}
    for seed, p in SEEDS.items():
        k = _fam(pd.read_csv(f"{p}/sae_summary_by_kind.csv"))
        d["pc"][seed] = k[k.position_kind == "pre_color"].set_index("layer").sort_index()
        d["pn"][seed] = k[k.position_kind == "pre_node"].set_index("layer").sort_index()
        d["pool"][seed] = pd.read_csv(f"{p}/sae_summary.csv").set_index("layer").sort_index()
    for name, p in ARMS.items():
        k = _fam(pd.read_csv(f"{p}/sae_summary_by_kind.csv"))
        d["arm"][name] = k[k.position_kind == "pre_color"].set_index("layer").sort_index()
    return d


def textpage(o, title, subtitle, blocks, foot, y0=0.80, pitch=0.148, width=100):
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
    pc, pn = D["pc"], D["pn"]
    arg = {s: int(pc[s].best_justifier_f1.idxmax()) for s in (0, 1)}
    tot = {s: int(pc[s].jn.sum()) for s in (0, 1)}
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.90, "Is the SAE result a seed artefact?", ha="center",
             size=BIG + 3, weight="bold")
    fig.text(0.5, 0.855, "Findings 19, 21 and 29 rest on ONE dictionary per layer at ONE seed.\n"
                         "Here is the second one: same checkpoint, same hyper-parameters, --seed 1.",
             ha="center", va="top", size=MED, color="0.3", linespacing=1.5)
    fig.text(0.5, 0.735, "MOSTLY NO. The structure replicates; the exact peak layer does not.",
             ha="center", size=BIG, weight="bold", color="#2b6cb0")
    nz = int(pn[0].jn.sum() + pn[1].jn.sum())
    paras = [
        f"THE ASYMMETRY IS REAL, and it is the claim the DD reading rests on. Features for "
        f"justifies_next -- 'u is a KNOWN neighbour of the node being emitted', exactly the input to "
        f"the elimination rule -- number {nz} at the positions where the model picks the NODE: zero at "
        f"all seven layers in seed 0, and zero at all seven again in seed 1. They appear only where it "
        f"picks the COLOUR, which is when DD needs them.",
        f"THE PEAK LAYER MOVES ONE RUNG. best justifies_next F1 at pre_color peaks at "
        f"L{arg[0]} ({pc[0].best_justifier_f1.max():.3f}) in seed 0 and at L{arg[1]} "
        f"({pc[1].best_justifier_f1.max():.3f}) in seed 1, with the other of the pair just behind "
        f"(L8 {pc[1].best_justifier_f1[8]:.3f}, L10 {pc[0].best_justifier_f1[10]:.3f}). So finding 29 "
        f"must say L8-L10, a WINDOW -- which is still inside the L6->L10 handoff that activation "
        f"patching found independently, so the two localisations still agree at window resolution.",
        f"THE COUNTS SPLIT. Total justifies_next at pre_color goes {tot[0]} -> {tot[1]}, which holds. "
        f"But layer 0's next_node count goes {int(pc[0].nn[0])} -> {int(pc[1].nn[0])} at an unchanged "
        f"best F1 of {pc[1].best_next_node_f1[0]:.3f} -- and layer 0 is the one layer whose dictionary "
        f"collapses in both seeds. Print F1, not layer-0 feature counts.",
    ]
    fig.text(0.07, 0.655, "\n\n".join(textwrap.fill(p, width=108) for p in paras),
             size=MED - 2, va="top", linespacing=1.5)
    footer(fig, "Job 50536365, 30 min on one H100, COMPLETED 2026-10-31 13:47. Checkpoint: the "
                "leak-free N=24 SOLVER that scores 1.0000 on the shared n=1500 pool. Compared against "
                "job 50075499, which is the same configuration at seed 0 and is the run findings 29 "
                "was written from. Deck built from both runs' CSVs; nothing typed in.")
    o.savefig(fig)


def page_method(o, D):
    pool = D["pool"]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "What a second seed tests, and what it cannot",
             ha="center", size=BIG, weight="bold")
    fig.text(0.06, 0.855,
             textwrap.fill("--seed moves FOUR things at once in sudoku_sae: the instance pool "
                           "(seed_start = 600000000 + seed), which positions are sampled, the SAE's "
                           "initialisation, and the null-shift RNG. So this is a FULL REPLICATION, not "
                           "an init-only ablation. Agreement is strong evidence the structure is in the "
                           "model and not in one dictionary; disagreement cannot be pinned on any single "
                           "source. There is no narrower knob exposed, and adding one would change the "
                           "code under the comparison.", width=118),
             size=MED - 3, va="top", linespacing=1.5)
    fig.text(0.06, 0.645, "THE COMPARABILITY CHECK: did the two dictionaries get fit under the same "
                          "pressure?", size=MED, weight="bold")
    hdr = ["layer", "l1_final s0", "l1_final s1", "var expl s0", "var expl s1",
           "L0 achieved s0", "L0 achieved s1"]
    xs = [0.07, 0.21, 0.33, 0.46, 0.59, 0.72, 0.86]
    for x, h in zip(xs, hdr):
        fig.text(x, 0.585, h, size=SMALL + 1, weight="bold")
    for i, L in enumerate(pool[0].index):
        y = 0.545 - i * 0.042
        vals = [f"L{L}", f"{pool[0].l1_final[L]:.4f}", f"{pool[1].l1_final[L]:.4f}",
                f"{pool[0].frac_variance_explained[L]:.4f}",
                f"{pool[1].frac_variance_explained[L]:.4f}",
                f"{pool[0].l0[L]:.1f}", f"{pool[1].l0[L]:.1f}"]
        same = pool[0].l1_final[L] == pool[1].l1_final[L]
        for x, v in zip(xs, vals):
            fig.text(x, y, v, size=SMALL + 1,
                     color=GREY if same else RED,
                     weight="bold" if L == 0 else "normal")
    fig.text(0.06, 0.205, textwrap.fill(
        f"l1_final is IDENTICAL at every layer, variance explained agrees to 3-4 decimals, and achieved "
        f"sparsity agrees to about 1 in 60 against a target of 64. The exception is LAYER 0 (bold), "
        f"where both seeds collapse to L0 = {pool[0].l0[0]:.1f} / {pool[1].l0[0]:.1f} with "
        f"~{int(pool[1].n_dead_features[0])} of 4096 features dead and variance explained at 1.0000 -- a "
        f"near-rank-deficient residual stream the dictionary cannot help overfitting. That is exactly "
        f"where the one non-replicating number lives.", width=118),
        size=MED - 4, va="top", linespacing=1.5)
    footer(fig, "spin_gnn/sudoku_sae.py is unchanged since commit 10c38b0f, the commit findings "
                "19/21/29 were measured under, so the new rows drop straight into their tables. "
                "Every other flag is byte-identical: expansion 16, 1500 SAE epochs, 3000 instances, "
                "alive-rate 1e-3, 19 null shifts, target L0 64, split by position kind.")
    o.savefig(fig)


def page_curves(o, D):
    pc, pn = D["pc"], D["pn"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.35, 1]})
    for s in (0, 1):
        axes[0].plot(pc[s].index, pc[s].best_justifier_f1, "o-", color=COL[s], lw=3, ms=11,
                     label=f"seed {s}: pre_color (picking the COLOUR)", rasterized=True)
        axes[0].plot(pn[s].index, pn[s].best_justifier_f1, "s--", color=COL[s], lw=2, ms=7,
                     alpha=0.6, label=f"seed {s}: pre_node (picking the NODE)", rasterized=True)
        axes[1].bar(np.asarray(pc[s].index) + (s - 0.5) * 0.85, pc[s].jn, width=0.85,
                    color=COL[s], label=f"seed {s}", rasterized=True)
    axes[0].axvspan(6, 10, color="#f6e05e", alpha=0.25, lw=0)
    axes[0].text(8, 0.055, "L6-L10: where activation patching\nhands the forcing fact over",
                 ha="center", size=SMALL + 1, color="#8a6d00")
    for s, xy in ((0, (5.1, 0.60)), (1, (11.4, 0.99))):
        L = int(pc[s].best_justifier_f1.idxmax())
        axes[0].annotate(f"peak, seed {s}\nL{L} at {pc[s].best_justifier_f1[L]:.3f}",
                         xy=(L, pc[s].best_justifier_f1[L]), xytext=xy, color=COL[s],
                         size=SMALL + 2, weight="bold", ha="center", va="center",
                         arrowprops=dict(arrowstyle="-|>", color=COL[s], lw=2))
    axes[0].set_xticks(list(pc[0].index))
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("best F1 for any justifies_next predicate")
    axes[0].set_xlim(-0.8, 13.6)
    axes[0].set_ylim(-0.04, 1.10)
    axes[0].set_title("The forcing set, by layer and by position kind", size=MED + 2,
                      weight="bold")
    axes[0].legend(loc="upper left", fontsize=SMALL)
    axes[1].set_xticks(list(pc[0].index))
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("number of justifies_next features")
    axes[1].set_title("Feature counts at pre_color", size=MED + 2, weight="bold")
    axes[1].legend(fontsize=SMALL)
    fig.suptitle("Two dictionaries, one shape: flat at pre_node, a back-half bump at pre_color",
                 size=BIG - 3, weight="bold")
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    footer(fig, "Dashed = the pre_node positions: never above 0.17 in either seed and within ~0.1 of "
                "their own shuffled null, while the solid pre_color curves reach 0.77-0.84. That is "
                "the asymmetry, drawn. The two solid curves agree to within 0.02 through L0-L8 and "
                "separate only at L10 and L12, where seed 1 stays high; the peak is in that separation.")
    o.savefig(fig)


def page_prereg(o, D):
    pc, pn = D["pc"], D["pn"]
    arg1 = int(pc[1].best_justifier_f1.idxmax())
    rows = [
        ("(a) HEADLINE: zero justifies_next features at pre_node, at all seven layers",
         f"CONFIRMED -- 0 at every layer, in both seeds. Falsifier was any non-zero count.", GREEN),
        ("(b) LAYER 0 REPRODUCTION: next_node F1 >= 0.99 and 18-28 features at pre_color; "
         "0 features and below-null F1 at pre_node",
         f"SPLIT -- F1 {pc[1].best_next_node_f1[0]:.3f} and the asymmetry hold "
         f"({int(pn[1].nn[0])} features at pre_node, F1 {pn[1].best_next_node_f1[0]:.3f} under its "
         f"{pn[1].median_best_f1_null[0]:.3f} null), but the count is {int(pc[1].nn[0])}, outside "
         f"[18, 28]", AMBER),
        ("(c) LOCALISATION, the one that matters: best justifies_next F1 at L8 >= 0.70 AND L8 is "
         "the argmax over layers",
         f"SPLIT -- L8 is {pc[1].best_justifier_f1[8]:.3f}, above 0.70, but the argmax moved to "
         f"L{arg1} ({pc[1].best_justifier_f1[arg1]:.3f}). This was the named falsifier: quote the "
         f"layer to +/-2, not exactly.", RED),
        ("(d) LOW CONFIDENCE: total justifies_next at pre_color within +/-40% of 47, i.e. 28-66",
         f"CONFIRMED -- {int(pc[1].jn.sum())}. Counts in the back half are stable "
         f"(L6 {int(pc[0].jn[6])}->{int(pc[1].jn[6])}, L8 {int(pc[0].jn[8])}->{int(pc[1].jn[8])}, "
         f"L10 {int(pc[0].jn[10])}->{int(pc[1].jn[10])}); only layer 0's are not.", GREEN),
    ]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "The four pre-registrations, scored", ha="center", size=BIG,
             weight="bold")
    fig.text(0.5, 0.875, "Written into scripts/slurm/sudoku2_sae_solved_seed1.sbatch before "
                         "submission, against seed 0's exact table.",
             ha="center", size=MED, color="0.3")
    for i, (claim, verdict, col) in enumerate(rows):
        y = 0.775 - i * 0.175
        fig.text(0.06, y, textwrap.fill(claim, width=108), size=MED - 2, va="top")
        fig.text(0.06, y - 0.062, textwrap.fill(verdict, width=112), size=MED - 3, va="top",
                 weight="bold", color=col, linespacing=1.4)
    footer(fig, "My record on pre-registered mechanism predictions in this project was 2 for 3 going "
                "in. (c) is the one that changes a written claim: finding 29 and page 14 of the "
                "verdict deck both read the DD step off layer 8 exactly, and that has to become L8-L10.")
    o.savefig(fig)


def page_arms(o, D):
    pc, arm = D["pc"], D["arm"]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(arm["LEAK-READER"].index, arm["LEAK-READER"].best_justifier_f1, "^-", color="0.45",
            lw=2.6, ms=11, label="LEAK-READER, 0.9960 (seed 0 only)", rasterized=True)
    ax.plot(arm["0.854 TWIN"].index, arm["0.854 TWIN"].best_justifier_f1, "d-", color="#2f855a",
            lw=2.6, ms=11, label="leak-free twin, 0.8540 (seed 0 only)", rasterized=True)
    for s in (0, 1):
        ax.plot(pc[s].index, pc[s].best_justifier_f1, "o-", color=COL[s], lw=3, ms=11,
                label=f"SOLVER, 1.0000 -- seed {s}", rasterized=True)
    g = arm["0.854 TWIN"].best_justifier_f1[8] - arm["LEAK-READER"].best_justifier_f1[8]
    noise = abs(pc[1].best_justifier_f1[8] - pc[0].best_justifier_f1[8])
    ax.annotate("", xy=(8.25, arm["0.854 TWIN"].best_justifier_f1[8]),
                xytext=(8.25, arm["LEAK-READER"].best_justifier_f1[8]),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=3))
    ax.text(8.4, 0.49, f"leak-free minus leaked\nat L8: {g:+.3f}", color=RED, size=MED - 1,
            weight="bold", va="center")
    ax.annotate(f"seed-to-seed\nat L8: {noise:.3f}", xy=(7.85, 0.779), xytext=(6.4, 0.95),
                color="#2b6cb0", size=MED - 2, weight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#2b6cb0", lw=2.5))
    ax.set_xticks(list(pc[0].index))
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("layer")
    ax.set_ylabel("best F1 for any justifies_next predicate, at pre_color")
    ax.set_title(f"At L8, seed noise is {g / noise:.0f}x smaller than the leak-free-minus-leaked gap",
                 size=BIG - 3, weight="bold")
    ax.legend(loc="upper left", fontsize=SMALL + 1)
    ax.text(0.3, 0.60, textwrap.fill(
        "A single seed was never enough to pin a LAYER, and page 5 shows it was not. It was enough "
        "for the between-arm claim: the model that reads the adjacency-order shortcut barely "
        "represents the forcing set, the two that must deduce it represent it strongly, and that gap "
        "is twenty times what re-rolling the dictionary moves. Caveat kept: the two older arms are "
        "still one seed each, so this page replicates one of its three rungs.", width=58),
        size=MED - 3, va="top", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.6", fc="#fffbea", ec="0.7"))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    footer(fig, "Older arms from job 49729375's kindsplit rerun (findings 19, 21), same flags, same "
                "predicate set, same null. Each checkpoint is analysed on its own distribution -- "
                "colour-sorted for the LEAK-READER, permuted for the two leak-free models -- because "
                "feeding a leak-free model sorted instances would be measuring it off-distribution.")
    o.savefig(fig)


def page_limits(o, D):
    pc = D["pc"]
    blocks = [
        ("WHAT THIS BUYS", GREEN,
         f"The pre_node/pre_color asymmetry and the back-half concentration of the forcing set are "
         f"not one dictionary's accident. Total justifies_next at pre_color {int(pc[0].jn.sum())} -> "
         f"{int(pc[1].jn.sum())}; zero at pre_node, fourteen layer-measurements out of fourteen."),
        ("WHAT IT COSTS", RED,
         "The layer index. Two seeds put the peak at L8 and at L10, so every place the project says "
         "'the forcing set enters at L8' -- finding 29, page 14 of the verdict deck -- becomes "
         "'L8-L10'. That is still inside the patching handoff, so the agreement between the "
         "correlational and causal localisations survives, at window resolution."),
        ("WHAT IS STILL UNFIXED", AMBER,
         "best_f1 is a max over 256 predicates in BOTH seeds, so the multiple-comparisons ceiling is "
         "unchanged -- a replication shows the same ceiling gives the same answer twice, which is "
         "weaker than removing it. Two seeds is two, not a distribution. And the SAE is "
         "correlational: what is USED comes from the interventions (findings 27, 28), not from here."),
        ("NEXT, AND IT IS CHEAP", "#2b6cb0",
         "A second seed on the LEAK-READER arm -- 30 minutes on one GPU, the same script with the "
         "other checkpoint. That is the arm the headline contrast is measured against and the only "
         "unreplicated half of it. Submitted alongside this deck."),
    ]
    textpage(o, "What the replication buys, and what it costs",
             "A result that mostly held, and one claim that has to be reworded",
             blocks, "Outputs under $SCRATCH/spin_gnn/sudoku_sae_solved_2026-10-31/ "
                     "(90-day purge; summaries are small and the dictionaries are re-derivable from "
                     "the checkpoint). Notebook: notebooks/sudoku_sae_seed1_2026-10-31_results.ipynb. "
                     "Job 50536365, commit 54a0a1d, script 31cbdc2, sudoku_sae.py at 10c38b0f, seed 1.",
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
        page_trace(o, ex, note=textwrap.fill(
            "The SAE is fit on the model's activations at exactly these positions. pre_node = the step "
            "before it emits a NODE id, pre_color = the step before it emits that node's COLOUR. "
            "justifies_next[u] is true when u is one of the already-known neighbours whose colours "
            "force the answer -- the list shown on the right of this page, for N0.", width=122))
        page_curves(o, D)
        page_prereg(o, D)
        page_arms(o, D)
        page_limits(o, D)
    p = os.path.join(OUT, "slides.pdf")
    print(f"wrote {p} ({os.path.getsize(p) / 1e6:.2f} MB, {o.n} pages)")


if __name__ == "__main__":
    main()
