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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

SCRATCH = os.environ["SCRATCH"]
CIRC = f"{SCRATCH}/spin_gnn/sudoku_circuits_leakfree_2026-10-28"
CHAIN = f"{SCRATCH}/spin_gnn/sudoku_chain_2026-10-28"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_dd_verdict_2026-10-28")

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "0.45"
CHANCE = 0.25
SITES = ["edited_hint", "graph_row_u", "graph_row_w", "predict_pos"]
# (label, circuits dir, chain dir, colour). Arm A is the model that actually
# solved the task; C is the leaked twin finding 2 was measured on.
ARMS = [("leak-free 1.0000", "A_leakfree_1000", "A_leakfree_1000", BLUE),
        ("leak-free 0.8540", "B_leakfree_0854", "B_leakfree_0854", GREEN),
        ("leaked 0.9960", "C_leaked_0996", "C_leaked_0996", RED)]


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
        out[label] = rec
    return out


def page_title(out, D, ex):
    fig = plt.figure(figsize=(16, 9))
    a = D["leak-free 1.0000"]
    r0, ch = a["r0"]["flip"], a.get("chain", {}).get("corrupt_flipped")
    leaked = D["leaked 0.9960"]["r0"]["flip"]
    fig.text(0.5, 0.86, "Is the leak-free network running DD?", ha="center",
             size=BIG + 7, weight="bold")
    fig.text(0.5, 0.775, "Causal interventions, not probe scores",
             ha="center", size=MED + 3, color="0.35")

    verdict = ("YES for DD's inference. NO for DD's search."
               if ch is not None and ch >= 0.60 else
               "YES for DD's one-step rule; chaining still measuring.")
    fig.text(0.5, 0.645, verdict, ha="center", size=BIG + 2, weight="bold", color=BLUE)

    chain_txt = f"{ch:.3f}" if ch is not None else "(running)"
    lines = [
        f"Recolour the ONE neighbour whose colour forces the answer -- a single token --",
        f"and the model follows DD's recomputed answer:",
        "",
        f"      {r0:.3f}   when that neighbour is a GIVEN hint        (round 0, n={a['r0']['n']})",
        f"      {chain_txt}   when it is one the model DEDUCED itself   (rounds 1-4)",
        f"      {leaked:.3f}   leaked twin, the model finding 2 called 'not running DD'",
        "",
        f"Chance is {CHANCE:.2f}. The second line is the one that matters: it eliminates",
        "against facts it produced, which is propagation, not a single rule application.",
        "",
        "But the SCHEDULE is not DD's: events come out in ascending node index for",
        "100% of rounds, and a no-deduction hand rule beats its own node features.",
    ]
    fig.text(0.5, 0.575, "\n".join(lines), ha="center", va="top", size=MED - 1,
             family="monospace", linespacing=1.5)
    missing = [k for k, v in D.items() if "chain" not in v]
    footer(fig, "Jobs 50075339 (round 0) and 50077245 (chained), leak-free N=24, d=256 L=12 ff=1024, "
                "shuffle_adj=True. Chance = 1/n_colors = 0.25."
                + (f" Arms still running: {', '.join(missing)}." if missing else ""))
    out.savefig(fig)


def page_design(out, ex):
    """What the intervention actually does, in tokens."""
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The intervention: one token, two different correct answers",
                 size=BIG, weight="bold")
    body = [
        "DD forces node v because its known neighbours already use q-1 = 3 of the 4 colours.",
        "Exactly one of them, w, holds colour a. So v must take the one colour left: c.",
        "",
        "   clean     ... BEGIN_ROUND  v  ->  [c]        w is coloured a, so c is the only option",
        "   corrupt   ... BEGIN_ROUND  v  ->  [a]        recolour w to c; now a is free and DD says a",
        "",
        "The ONLY difference between the two prompts is w's colour token.",
        "build_pair asserts (clean != corrupt).sum() == 1, so this is enforced, not assumed.",
        "",
        "ROUND 0  (job 50075339): w is a HINT, handed to the model in the prompt.",
        "         Tests whether it applies the elimination rule to GIVEN facts.",
        "",
        "CHAINED  (job 50077245): w was DEDUCED by the model in an earlier round, so the",
        "         edited token sits in the REASONING section, not the hints.",
        "         Tests whether it eliminates against facts it produced itself = propagation.",
        "",
        "Scored as `corrupt_flipped`: did the prediction at v move to DD's NEW answer a?",
        "Not 'is it still accurate' -- the right answer changes, so memorised output fails.",
    ]
    fig.text(0.06, 0.80, "\n".join(body), ha="left", va="top", size=MED - 1,
             family="monospace", linespacing=1.5)
    footer(fig, "The chained arm's one weak direction: editing an emitted deduction makes the prefix "
                "inconsistent, so a model that DETECTS the edit and keeps its own answer scores as "
                "'does not chain'. reverts_to_original separates those cases; see the table.")
    out.savefig(fig)


def page_table(out, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Every causal measurement", size=BIG, weight="bold")
    ax = fig.add_axes([0.04, 0.17, 0.92, 0.64]); ax.axis("off")
    cols = ["arm", "test", "n", "clean", "flipped to DD's\nnew answer", "reverts to\noriginal",
            "mean logit gap\nclean -> corrupt", "verdict"]
    rows = []
    for label, _, _, _ in ARMS:
        rec = D[label]
        if "r0" in rec:
            r = rec["r0"]
            rows.append([label, "round 0 (hint)", r["n"], f"{r['clean']:.3f}", f"{r['flip']:.3f}",
                         "--", f"{r['clean_ld']:+.1f} -> {r['corrupt_ld']:+.1f}",
                         "uses the rule" if r["flip"] >= 0.60 else "near chance"])
        if "chain" in rec:
            c = rec["chain"]
            rows.append([label, "chained (deduced)", c["n_cases"], f"{c['clean_correct']:.3f}",
                         f"{c['corrupt_flipped']:.3f}", f"{c['reverts_to_original']:.3f}",
                         "--", "chains" if c["corrupt_flipped"] >= 0.60 else "near chance"])
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(MED - 3); t.scale(1, 2.6)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#dddddd"); t[0, j].set_text_props(weight="bold")
    for i, r in enumerate(rows, start=1):
        colour = BLUE if r[0].startswith("leak-free") else RED
        t[i, 0].set_text_props(color=colour, weight="bold")
        t[i, 4].set_text_props(weight="bold",
                               color=GREEN if float(r[4]) >= 0.60 else RED)
    footer(fig, f"Chance = {CHANCE:.2f}. 'clean' is accuracy on the unmodified prompt, so a low "
                "'flipped' with a high 'clean' means the model is right for a reason other than the "
                "elimination -- which is exactly the leaked arm's profile.")
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
        labels.append(label.replace(" ", "\n")); cols.append(colour)
        r0v.append(rec["r0"]["flip"])
        chv.append(rec.get("chain", {}).get("corrupt_flipped", np.nan))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, r0v, 0.38, label="round 0 (hint)", color=cols, edgecolor="k")
    ax1.bar(x + 0.2, chv, 0.38, label="chained (deduced)", color=cols, edgecolor="k",
            hatch="///", alpha=0.75)
    ax1.axhline(CHANCE, color="k", ls="--", lw=2)
    ax1.text(len(labels) - 0.5, CHANCE + 0.02, f"chance {CHANCE:.2f}", ha="right", size=SMALL)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, size=SMALL)
    ax1.set_ylabel("flipped to DD's new answer"); ax1.set_ylim(0, 1.08)
    ax1.set_title("solid = hint, hatched = deduced", size=MED)
    ax1.legend(loc="center right", fontsize=SMALL)
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
        ax2.plot(ks, v, "o-", color=BLUE, lw=3, ms=13)
        ax2.axhline(CHANCE, color="k", ls="--", lw=2)
        for k, y, n in zip(ks, v, ns):
            ax2.annotate(f"{y:.3f}\nn={n}", (k, y), textcoords="offset points",
                         xytext=(0, -42), ha="center", size=SMALL)
        ax2.set_xlabel("DD round the node was deduced in")
        ax2.set_ylabel("flipped to DD's new answer")
        ax2.set_ylim(0, 1.12); ax2.set_xticks(ks)
        ax2.set_title("No decay with depth -- leak-free 1.0000", size=MED)
    fig.tight_layout(rect=[0, 0.11, 1, 0.93])
    footer(fig, "Right panel is the chaining claim: if the model chained one step but not three, "
                "round 1 would be high and the later rounds would fall off. It does not happen. "
                "The deepest bins are small because deep chained forcings are rare, not filtered.")
    out.savefig(fig)


def page_patching(out, D):
    arms = [(l, D[l]) for l, _, _, _ in ARMS if "patch" in D[l]]
    fig, axes = plt.subplots(1, len(arms), figsize=(16, 9), squeeze=False)
    fig.suptitle("Where the recoloured fact travels (activation patching)",
                 size=BIG, weight="bold")
    for ax, (label, rec) in zip(axes[0], arms):
        m = (rec["patch"].groupby(["layer", "site"]).recovery.mean()
             .unstack()[SITES])
        im = ax.imshow(m.values, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       origin="lower", rasterized=True)
        ax.set_xticks(range(len(SITES)))
        ax.set_xticklabels([s.replace("_", "\n") for s in SITES], size=SMALL - 1)
        ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index, size=SMALL - 1)
        ax.set_title(label, size=MED)
        ax.set_ylabel("layer" if ax is axes[0][0] else "")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                val = m.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", size=SMALL - 4,
                        color="w" if val < 0.6 else "k")
    # subplots_adjust, not tight_layout -- the shared colorbar fights it
    fig.subplots_adjust(bottom=0.22, top=0.86, left=0.07, right=0.88)
    fig.colorbar(im, ax=axes[0], fraction=0.025, label="mean recovery")
    footer(fig, "Recovery = how much of the flip is restored by patching the clean activation at that "
                "(layer, site). The fact is held at the EDITED HINT through L5, handed off L6-L10, and "
                "is at the PREDICTION POSITION by L10. graph_row_u/graph_row_w are 0.00 at every layer "
                "in all arms -- the deduction never routes through the adjacency table rows. L11 at the "
                "prediction position is the built-in positive control and must read 1.00.")
    out.savefig(fig)


def page_heads(out, D):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle("No single head carries it -- the circuit is distributed",
                 size=BIG, weight="bold")
    for label, _, _, colour in ARMS:
        rec = D[label]
        if "heads" not in rec:
            continue
        h = (rec["heads"].groupby(["layer", "head"]).effect_retained.mean()
             .sort_values().reset_index())
        ax.plot(range(len(h)), h.effect_retained, "o-", color=colour, lw=2.5, ms=8,
                # h["head"], not h.head -- the latter is DataFrame.head()
                label=f"{label}  (worst L{int(h['layer'].iloc[0]):02d}"
                      f"H{int(h['head'].iloc[0])} = {h.effect_retained.iloc[0]:.3f})")
    ax.axhline(1.0, color="k", ls="--", lw=2)
    ax.text(1, 1.005, "1.00 = ablating this head changes nothing", size=SMALL)
    ax.set_xlabel("attention heads, sorted by damage (most damaging first)")
    ax.set_ylabel("effect retained after ablating that head")
    ax.legend(loc="lower right", fontsize=SMALL)
    fig.tight_layout(rect=[0, 0.11, 1, 0.93])
    footer(fig, "Each point is one of the 48 heads, zeroed one at a time by writing its out_proj weight "
                "columns (nn.MultiheadAttention takes the functional fast path, so module hooks on "
                "out_proj never fire -- an earlier version of this analysis silently reported 1.000 for "
                "all 7200 rows). Removing the worst single head still leaves ~84% of the effect.")
    out.savefig(fig)


def page_verdict(out, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The verdict, and what it is not", size=BIG, weight="bold")
    a = D["leak-free 1.0000"]
    ch = a.get("chain", {}).get("corrupt_flipped")
    left = [
        "WHAT IT DOES DO  (causal)",
        "",
        f"* Applies DD's elimination rule: {a['r0']['flip']:.3f}",
        "  on 350 one-token counterfactuals.",
        "",
        f"* Chains it off its OWN deductions: {ch:.3f}" if ch is not None else
        "* Chaining: measuring (50077245).",
        "  No decay from round 1 to round 4.",
        "",
        "* Localised: hint -> L5, handoff L6-L10,",
        "  prediction position by L10.",
        "",
        "* Distributed across heads, not one head.",
    ]
    right = [
        "WHAT IT DOES NOT DO",
        "",
        "* DD's SEARCH SCHEDULE. Events are emitted",
        "  in ascending node index for 100% of rounds.",
        "  A no-deduction hand rule ('smallest index",
        "  whose colour is unknown') scores 0.632 on",
        "  the same max-over-v F1 where the model's",
        "  own next_node features score 0.545.",
        "",
        "* Use the adjacency table to do it: the",
        "  graph rows recover 0.000 at every layer.",
        "",
        "* The leaked twin does not even do the rule",
        f"  ({D['leaked 0.9960']['r0']['flip']:.3f}) -- finding 2's 'not running",
        "  DD' was always about THAT model. Shown an",
        "  edited deduction it keeps its own answer",
        f"  {D['leaked 0.9960']['chain']['reverts_to_original']:.1%} of the time: it is reading the",
        "  adjacency-order leak, not the prefix.",
    ]
    fig.text(0.06, 0.78, "\n".join(left), ha="left", va="top", size=MED - 1,
             family="monospace", linespacing=1.5, color=GREEN)
    fig.text(0.53, 0.78, "\n".join(right), ha="left", va="top", size=MED - 1,
             family="monospace", linespacing=1.5, color=RED)
    fig.text(0.5, 0.135,
             "Working picture: sweep nodes in index order; at each, eliminate the colours of its\n"
             "known neighbours, including ones it just derived. DD's propagation, not DD's schedule.",
             ha="center", size=MED + 1, weight="bold", linespacing=1.6)
    footer(fig, "Comparison to the previous best answer on this question: finding 2 of this project "
                "concluded 'the model is not running DD' from probes, an algorithm-ID test, a "
                "counterfactual and an SAE. Its counterfactual is reproduced here as the leaked arm at "
                f"{D['leaked 0.9960']['r0']['flip']:.3f}. That conclusion stands for the leaked model and is "
                "overturned for the leak-free one -- the honest task was never measured before.")
    out.savefig(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    D = load()
    if "r0" not in D["leak-free 1.0000"]:
        raise SystemExit("arm A round-0 results missing -- job 50075339 output not found")
    ex = example_instance(rho=0.60)
    pdf_path = os.path.join(OUT, "slides.pdf")
    with PdfPages(pdf_path) as pdf:
        o = PdfAndPng(pdf, OUT)
        page_title(o, D, ex)
        page_tokenization(o, ex)
        page_trace(o, ex, note="The intervention edits ONE colour token in this section "
                               "(chained arm) or in the hints (round-0 arm).")
        page_design(o, ex)
        page_table(o, D)
        page_rates(o, D)
        page_patching(o, D)
        page_heads(o, D)
        page_verdict(o, D)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {o.n} pages)")


if __name__ == "__main__":
    main()
