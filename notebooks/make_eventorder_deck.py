"""Slide deck for job 49840619: does the model lean on the ORDER of the deductions?

Run as `python -m notebooks.make_eventorder_deck` from the repo root.

The trace page is built twice here -- once sorted, once permuted -- because the
within-round event order IS the subject of this run, and a reader looking at a
single sorted trace would read the generator's ascending-index convention as
something the model produced.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from notebooks.deck_tokenization import MONO, example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

SCRATCH = os.environ["SCRATCH"]
RUN = f"{SCRATCH}/spin_gnn/sudoku_n24_eventorder_2026-10-27"
PREV = f"{SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_eventorder_2026-10-28")

BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "0.45"
# Measured on CPU before the job was submitted (scripts/verify_shuffle_events.py,
# 300 instances): log(k!) per k-event round over evaluate()'s own denominator.
FLOOR_NLL, FLOOR_ACC = 0.0270, -0.0163


def load():
    pool = pd.read_csv(f"{RUN}/sharedpool_n1500.csv")
    pool["cell"] = ["sorted (finding 17)", "permuted (shuffle_events)"]
    depth = {"sorted (finding 17)": pd.read_csv(f"{PREV}/depth_n24_leakfree.csv"),
             "permuted (shuffle_events)": pd.read_csv(f"{RUN}/depth_n24_eventorder.csv")}
    for d in depth.values():
        d["k"] = (d.solve_rate * d.n).round().astype(int)
    sweep = {"sorted (finding 17)": pd.read_csv(f"{PREV}/sweep.csv").iloc[0],
             "permuted (shuffle_events)": pd.read_csv(f"{RUN}/sweep.csv").iloc[0]}
    om = pd.read_csv(f"{RUN}/ordering_and_memorization.csv")
    return pool.set_index("cell"), depth, sweep, om


def table(ax, header, rows, colw, fontsize=SMALL, row_colors=None, dy=0.115, rule_dy=0.09):
    ax.axis("off")
    y = 0.92
    for i, txt in enumerate(header):
        ax.text(sum(colw[:i]), y, txt, size=fontsize, weight="bold", va="top")
    ax.plot([0, sum(colw)], [y - rule_dy, y - rule_dy], color="0.6", lw=1.0)
    for r, row in enumerate(rows):
        y -= dy
        c = (row_colors or {}).get(r, "black")
        for i, txt in enumerate(row):
            ax.text(sum(colw[:i]), y, txt, size=fontsize, va="top", color=c)
    ax.set_xlim(0, sum(colw))
    ax.set_ylim(0, 1)


def paired_ci(delta, b, c, n):
    """95% CI on a paired difference of proportions, from the discordant counts."""
    half = 1.96 * np.sqrt(b + c) / n
    return delta - half, delta + half


def page_title(out, pool):
    p = pool.loc["permuted (shuffle_events)"]
    lo, hi = paired_ci(p.delta, p.discordant_lost, p.discordant_won, p.n_instances)
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.82, "The deduction ORDER was decoration.", size=BIG + 6, weight="bold")
    fig.text(0.06, 0.725, "Training without it costs nothing resolvable.",
             size=BIG + 5, weight="bold")
    fig.text(0.06, 0.60,
             f"Retrained with the deductions inside each round PERMUTED, and scored on the\n"
             f"same 1500-instance pool: {p.solve_rate:.4f} against the sorted-order baseline's "
             f"{pool.loc['sorted (finding 17)', 'solve_rate']:.4f}.\n"
             f"Delta {p.delta:+.4f}, {int(p.discordant_won)} discordant instances won and "
             f"{int(p.discordant_lost)} lost,\nMcNemar exact p = {p.mcnemar_exact_p:.4f}.",
             size=MED + 1, va="top")
    fig.text(0.06, 0.40,
             "WHY THIS RUN EXISTED. solve_with_dd_trace emits sorted(...) per round, so a round's\n"
             "deductions arrive in ascending node index for 100% of rounds. A hand rule with no\n"
             "deduction in it -- 'the smallest index above the previous one whose colour is still\n"
             "unknown' -- scores 0.632 on next_node, ABOVE the model's own best SAE feature at 0.545\n"
             "(finding 21). So 'it knows which node comes next' was free, and the open question was\n"
             "whether the model was LEANING on it. Only a retrain can ask: permuting at eval time\n"
             "only puts the model out of distribution.",
             size=SMALL + 1, va="top", color="0.25")
    fig.text(0.06, 0.155,
             f"HOW MUCH COST THIS RULES OUT. The paired 95% interval on the delta is "
             f"[{lo:+.4f}, {hi:+.4f}],\nso this rules out a cost larger than ~{abs(lo) * 100:.1f} "
             f"points -- not all cost. The point estimate does\nfavour the sorted arm, and at this "
             f"n a 1.5-point effect is simply not resolvable.",
             size=SMALL, va="top", color=GREY)
    footer(fig, "job 49840619, COMPLETED 05:50:50, ExitCode 0:0 | commit 77ebb35 | "
                "scripts/slurm/sudoku2_n24_eventorder.sbatch | seed 0 | "
                "shuffle_adj=True, shuffle_events=True")
    out.savefig(fig)


def page_control(out, sweep):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The control, and why it is clean", size=BIG, weight="bold")
    fig.text(0.06, 0.885,
             "A DD round deduces every node whose neighbours already cover q-1 colours\n"
             "SIMULTANEOUSLY, so the order inside it is an arbitrary presentation choice.",
             size=SMALL, va="top")

    fig.text(0.06, 0.805, "VERIFIED ON CPU BEFORE SUBMITTING (scripts/verify_shuffle_events.py, "
                          "300 instances)", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.765,
             "- adj, the planted colouring and the hints are identical instance by instance\n"
             "- every round's membership is identical AS A SET, and T is identical\n"
             "- the encoded sequence LENGTH is identical, so max_len and n_params cannot move\n"
             "- 520 of the 707 multi-event rounds (73.6%) actually come out reordered; a 2-event\n"
             "  round is left sorted half the time by chance, which is why it is not 100%",
             size=SMALL - 1, va="top", **MONO)

    s0, s1 = sweep["sorted (finding 17)"], sweep["permuted (shuffle_events)"]
    rows = [["n_params", f"{int(s0.n_params):,}", f"{int(s1.n_params):,}", "identical, as predicted"],
            ["max_len", f"{int(s0.max_len)}", f"{int(s1.max_len)}", "identical, as predicted"],
            ["final_val_nll", f"{s0.final_val_nll:.4f}", f"{s1.final_val_nll:.4f}",
             f"{s1.final_val_nll - s0.final_val_nll:+.4f} vs floor {FLOOR_NLL:+.4f}"],
            ["final_val_accuracy", f"{s0.final_val_accuracy:.4f}", f"{s1.final_val_accuracy:.4f}",
             f"{s1.final_val_accuracy - s0.final_val_accuracy:+.4f} vs floor {FLOOR_ACC:+.4f}"]]
    ax = fig.add_axes([0.06, 0.345, 0.90, 0.20])
    table(ax, ["", "sorted", "permuted", "change"], rows,
          [0.24, 0.17, 0.17, 0.38], fontsize=SMALL - 1,
          row_colors={0: GREY, 1: GREY, 2: "black", 3: "black"}, dy=0.21, rule_dy=0.14)

    fig.text(0.06, 0.26, "TOKEN METRICS ARE NOT COMPARABLE ACROSS THIS FLAG -- and that is "
                         "measurable, not a hedge.", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.22,
             "Permuting a round of k simultaneous deductions injects log(k!) nats that NO model can\n"
             "predict. Measured over evaluate()'s own denominator (every valid token, graph section\n"
             "included) that floor is +0.0270 nats/token and -0.0163 token accuracy. The permuted arm\n"
             "lands at +0.0323 and -0.0142: one entropy floor away and no further, so the permutation\n"
             "cost no predictive capacity elsewhere in the sequence. That is why solve_rate, which is\n"
             "order-insensitive, is the headline and val_nll is only a check on the control itself.",
             size=SMALL - 1, va="top")
    footer(fig, "floors from scripts/verify_shuffle_events.py; the sorted column is job 49659502 "
                "(finding 17), one change away")
    out.savefig(fig)


def page_pool(out, pool, om):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The pairing is the reason this comparison exists", size=BIG, weight="bold")

    fig.text(0.06, 0.855, "WHY TWO CELLS TRAINED ON DIFFERENT TARGET ORDERS CAN SHARE A POOL",
             size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.815,
             "sudoku_agg_eval prompts with full_tokens[:reasoning_start] -- hints and graph only --\n"
             "and scores FREE generation against the planted colouring through an order-insensitive\n"
             "check_solution. The one thing this flag changes, the order of events inside a round of\n"
             "the TARGET trace, lands entirely in the discarded suffix. So the two cells are handed\n"
             "byte-identical prompts on the same 1500 instances and McNemar applies. shuffle_events\n"
             "is therefore deliberately NOT in INSTANCE_KEYS, unlike shuffle_adj which changes the\n"
             "prompt itself. (Every teacher-forced analysis -- SAE, probes, circuits, depth slots,\n"
             "attention -- is the other way round, and all nine call sites now pass the flag through.)",
             size=SMALL - 1, va="top")

    ords = om[om.part == "ordering"].set_index("ordering")
    rows = []
    for name in ["identity", "within_block", "shuffle", "byid", "bycolor"]:
        r = ords.loc[name]
        rows.append([name, f"{r.heldout_solve_rate:.4f}",
                     "reference" if name == "identity" else f"{r.gap:+.4f}",
                     "--" if name == "identity" else f"{r.p:.4f}"])
    ax = fig.add_axes([0.06, 0.30, 0.42, 0.26])
    table(ax, ["adjacency order", "solve_rate", "delta", "exact p"], rows,
          [0.42, 0.21, 0.19, 0.18], fontsize=SMALL - 1, dy=0.165, rule_dy=0.11)

    fig.text(0.53, 0.555, "THE OTHER ORDER AXIS IS STILL DEAD FLAT",
             size=SMALL, weight="bold", va="top")
    fig.text(0.53, 0.515,
             "The same graphs re-scored under all five adjacency\n"
             "orderings, n=400 each. Nothing resolves, including\n"
             "bycolor -- which RE-INSTATES finding 16's leak and\n"
             "still does not help (-0.0125, p=0.52).\n\n"
             "This replicates finding 17b on a model trained\n"
             "without the event order as well, so removing one\n"
             "order convention did not make the model fall back\n"
             "on the other. Both are decoration for this cell.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.225, "WHAT A NULL HERE IS AND IS NOT", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.185,
             "It is: evidence that the ascending-index convention was not the mechanism producing\n"
             "traces, so finding 17's leak-free numbers stand as deduction scores and the open problem\n"
             "that finding 21 raised closes. It is NOT: proof of no effect. 139 won against 161 lost is\n"
             "a 1.5-point point estimate in the sorted arm's favour that this n cannot resolve, and a\n"
             "cost of up to ~3.7 points is inside the interval.",
             size=SMALL - 1, va="top")
    footer(fig, "sudoku_agg_eval n=1500 seed_start 5e10; orderings from sudoku_memorization_eval "
                "--n-ordering 400")
    out.savefig(fig)


def depth_stats(depth, rho=0.60):
    """Pearson r and the end-to-end drop per arm, so no page hardcodes either."""
    out = {}
    for c, d in depth.items():
        g = d[d.rho_requested == rho].sort_values("T")
        out[c] = {"r": float(np.corrcoef(g["T"], g.solve_rate)[0, 1]),
                  "rates": list(g.solve_rate), "T": list(g["T"]), "n": list(g.n),
                  "drop": float(g.solve_rate.iloc[0] - g.solve_rate.iloc[-1])}
    return out


def page_depth(out, depth, pool):
    st = depth_stats(depth)
    s, p = st["sorted (finding 17)"], st["permuted (shuffle_events)"]
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The depth cost survives, and the drop is bigger",
             size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             f"Pre-registration (c): r at rho=0.60 stays below -0.8. It does, at {p['r']:.3f}.",
             size=SMALL)

    ax = fig.add_axes([0.07, 0.47, 0.39, 0.35])
    for c, col in [("sorted (finding 17)", BLUE), ("permuted (shuffle_events)", RED)]:
        g = depth[c][depth[c].rho_requested == 0.60].sort_values("T")
        lo, hi = [], []
        for _, r in g.iterrows():
            a, b = stats.beta.interval(0.95, r.k + 0.5, r.n - r.k + 0.5)
            lo.append(max(0.0, r.solve_rate - a)); hi.append(max(0.0, b - r.solve_rate))
        ax.errorbar(g["T"], g.solve_rate, yerr=[lo, hi], fmt="-o", color=col, ms=5, lw=1.8,
                    capsize=2, label=c)
    ax.axhline(0.90, color="0.35", ls="--", lw=1.2)
    ax.set_xlabel("DD depth T", size=SMALL); ax.set_ylabel("solve_rate", size=SMALL)
    ax.set_title("rho=0.60, realized density 0.583 in every bin", size=SMALL + 1)
    ax.set_xticks([2, 3, 4, 5]); ax.tick_params(labelsize=SMALL - 2)
    ax.legend(fontsize=SMALL - 3, loc="lower left")

    rows = [["sorted", *[f"{v:.4f}" for v in s["rates"]], f"{s['r']:+.3f}", f"{s['drop']:.3f}"],
            ["permuted", *[f"{v:.4f}" for v in p["rates"]], f"{p['r']:+.3f}", f"{p['drop']:.3f}"]]
    ax2 = fig.add_axes([0.52, 0.60, 0.46, 0.22])
    table(ax2, ["", "T=2", "T=3", "T=4", "T=5", "r", "T2-T5"], rows,
          [0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12], fontsize=SMALL - 1,
          row_colors={0: BLUE, 1: RED}, dy=0.26, rule_dy=0.18)
    fig.text(0.52, 0.555,
             f"n per bin: {', '.join(f'T={t}:{n}' for t, n in zip(s['T'], s['n']))}; "
             f"identical bins, same generator stream", size=SMALL - 3, va="top", color=GREY)

    fig.text(0.52, 0.505, "THE SHAPE, NOT JUST THE SLOPE", size=SMALL, weight="bold", va="top")
    fig.text(0.52, 0.465,
             "The permuted arm is BETTER than sorted at T=2 and\n"
             "T=3 and WORSE at T=4 and T=5 -- the two curves cross.\n"
             "So the 1.5-point pool delta is not a uniform tax: it\n"
             f"is a redistribution towards the shallow end, and the\n"
             f"T=2..5 drop nearly doubles ({p['drop']:.3f} vs {s['drop']:.3f}).\n"
             f"r is nonetheless a hair less negative ({p['r']:.3f} vs\n"
             f"{s['r']:.3f}) because r scores linearity too, and this\n"
             "curve bends -- which is why both columns are shown.\n\n"
             f"The deepest bin is where the order may still be doing\n"
             f"something, and also where n is smallest ({s['n'][-1]} instances\n"
             "at T=5, interval 0.70-0.86). One targeted follow-up,\n"
             "not a claim.",
             size=SMALL - 1, va="top")

    fig.text(0.07, 0.395, "rho=0.75, THE SHALLOW CONTROL", size=SMALL, weight="bold", va="top")
    fig.text(0.07, 0.355,
             "T=1 1.0000 (n=105), T=2 0.9915 (352), T=3 0.9725 (109),\n"
             "against the sorted arm's 1.000 / 0.969 / 0.890. The\n"
             "permuted model is AHEAD in two of three shallow bins,\n"
             "and well_formed_rate is 1.0 with truncated_rate 0.0 in\n"
             "every bin of both runs -- so nothing here is a grammar\n"
             "failure.",
             size=SMALL - 1, va="top")
    footer(fig, "sudoku_depth_eval --rho 0.75 0.60 --n-pool 600 --min-per-bin 40")
    out.savefig(fig)


def page_prereg(out, pool, sweep, om, depth):
    p = pool.loc["permuted (shuffle_events)"]
    m = om[om.part == "memorization"].iloc[0]
    ds = depth_stats(depth)
    st, sorted_st = ds["permuted (shuffle_events)"], ds["sorted (finding 17)"]
    # Difference in differences between the two runs' memorization gaps. The two
    # models share the same 80000 training instances (same seed), so this is
    # conservative: independent binomial SEs on four rates.
    g_prev, g_now = 0.1000, float(m.gap)
    se_prev = np.sqrt(0.9567 * 0.0433 / 600 + 0.8567 * 0.1433 / 600)
    se_now = np.sqrt(m.train_solve_rate * (1 - m.train_solve_rate) / m.n_instances
                     + m.heldout_solve_rate * (1 - m.heldout_solve_rate) / m.n_instances)
    z = (g_now - g_prev) / np.sqrt(se_prev ** 2 + se_now ** 2)
    p_dd = 2 * stats.norm.sf(abs(z))

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Pre-registrations, as resolved", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Written into scripts/slurm/sudoku2_n24_eventorder.sbatch before the job ran.",
             size=SMALL)

    rows = [
        ["(a)", "solve_rate falls RESOLVABLY but stays\n      above 0.75", "FALSIFIED\n   (toward "
         "'no change')", f"{p.solve_rate:.4f} vs 0.8540, {p.delta:+.4f},\n"
                         f"   p={p.mcnemar_exact_p:.4f}. The named falsifier --\n"
                         f"   'the order was pure decoration' -- holds."],
        ["(b)", "val_nll rises by about the irreducible\n      floor and no more", "CONFIRMED",
         f"{sweep['permuted (shuffle_events)'].final_val_nll - sweep['sorted (finding 17)'].final_val_nll:+.4f} "
         f"vs floor {FLOOR_NLL:+.4f};\n   accuracy {sweep['permuted (shuffle_events)'].final_val_accuracy - sweep['sorted (finding 17)'].final_val_accuracy:+.4f} "
         f"vs floor {FLOOR_ACC:+.4f};\n   n_params and max_len unmoved"],
        ["(c)", "The depth cost survives: r at rho=0.60\n      stays below -0.8", "CONFIRMED",
         f"r = {st['r']:.3f}, and the T=2..5 drop is nearly\n   twice the sorted arm's "
         f"({st['drop']:.3f} vs {sorted_st['drop']:.3f})"],
        ["(d)", "The memorization gap does NOT shrink\n      below +0.1000", "NOMINALLY VIOLATED,\n"
         "   not resolvably", f"{g_now:+.4f} (train {m.train_solve_rate:.4f} vs held-out\n"
                              f"   {m.heldout_solve_rate:.4f}, p={m.p:.1e}). Against +0.1000 that\n"
                              f"   is z={z:+.2f}, p={p_dd:.2f} -- a tie."],
    ]
    ax = fig.add_axes([0.06, 0.30, 0.90, 0.53])
    table(ax, ["", "claim", "verdict", "evidence"], rows,
          [0.05, 0.35, 0.19, 0.41], fontsize=SMALL - 1,
          row_colors={0: RED, 1: GREEN, 2: GREEN, 3: ORANGE}, dy=0.20, rule_dy=0.09)

    fig.text(0.06, 0.245, "(a) IS A FALSIFIED PREDICTION I AM GLAD ABOUT, WHICH IS ITS OWN RISK.",
             size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.205,
             "I predicted a resolvable fall, and pre-registered what a null would mean so I could not\n"
             "reinterpret it afterwards: 'the ascending order was pure decoration, the min_unknown\n"
             "shortcut was never being used, and the open problem closes with the leak-free numbers\n"
             "intact.' That is the reading I owe. The honest limit is on the previous page: a null at\n"
             "n=1500 bounds the cost at ~3.7 points, it does not drive it to zero.",
             size=SMALL - 1, va="top")
    footer(fig, "(d)'s difference-in-differences uses independent binomial SEs on four rates, which "
                "is conservative: both cells trained on the same 80000 instances")
    out.savefig(fig)


def page_next(out):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "What this settles", size=BIG, weight="bold")

    fig.text(0.06, 0.855, "SETTLED", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.815,
             "- The open problem finding 21 raised is CLOSED. The ascending-index order inside a round\n"
             "  was available to the model and worth 0.632 to a hand rule, but a model trained without\n"
             "  it solves the task just as well, so it was not the mechanism.\n"
             "- Finding 17's leak-free solve_rates stand as deduction scores, not partly order-decoding\n"
             "  scores -- which was the specific worry this run was built to answer.\n"
             "- Both order conventions in the data are now controlled: shuffle_adj (finding 16) changed\n"
             "  the answer dramatically; shuffle_events does not change it at all.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.555, "STILL OPEN", size=SMALL, weight="bold", va="top", color=RED)
    fig.text(0.06, 0.515,
             "- The curves cross at T=4: the permuted arm is ahead in shallow bins and behind in deep\n"
             "  ones. If real, the order helps exactly where the reasoning is longest, which is the\n"
             "  interesting version of the hypothesis. n=98 in the deepest bin cannot settle it.\n"
             "- The confound stated up front in the sbatch: the permuted arm's training distribution\n"
             "  has strictly higher entropy at the same n_train, so 'lost a crutch' and 'got less\n"
             "  usable supervision per instance' are not separated here. Job 49705064 now shows this\n"
             "  task IS data-bound at 80000, which makes the confound a live alternative, not a formality.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.255, "WHAT I WOULD DO NEXT", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.215,
             "1. Re-run this control at the 240000-instance budget that 49705064 established. If both\n"
             "   arms then sit at 1.000 the question is closed for good; if the permuted arm alone\n"
             "   falls short, the entropy confound was the whole 1.5-point delta.\n"
             "2. A depth-targeted pool (rho=0.55, n>=400 at T>=5) to test the crossing directly.\n"
             "3. Re-score the SAE next_node features on THIS checkpoint: a model trained without the\n"
             "   order should not have them at all, which is a sharp prediction finding 21 can test.",
             size=SMALL - 1, va="top")
    footer(fig, "the checkpoint: d0256_h04_L12_ff01024_lr1e-03_pre_shuf_ev in "
                "sudoku_n24_eventorder_2026-10-27")
    out.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    pool, depth, sweep, om = load()
    # Two traces from the SAME instance: the control is exactly the difference
    # between these two pages, so showing one would hide the subject of the run.
    ex_sorted = example_instance(n_nodes=24, shuffle_adj=True, shuffle_events=False, pick="deep")
    ex_perm = example_instance(n_nodes=24, shuffle_adj=True, shuffle_events=True, pick="deep")
    # If the flag perturbed the generator stream, the two pages would be showing
    # two DIFFERENT instances and the comparison would be a lie. Check, don't trust.
    assert ex_sorted.seed == ex_perm.seed, (ex_sorted.seed, ex_perm.seed)
    assert np.array_equal(ex_sorted.adj, ex_perm.adj)
    assert ex_sorted.trace["hints"] == ex_perm.trace["hints"]
    assert ex_sorted.n_rounds == ex_perm.n_rounds
    assert [sorted(r) for r in ex_sorted.trace["rounds"]] == \
           [sorted(r) for r in ex_perm.trace["rounds"]], "round membership differs as a SET"
    reordered = sum(a != b for a, b in zip(ex_sorted.trace["rounds"], ex_perm.trace["rounds"]))
    multi = sum(len(r) > 1 for r in ex_sorted.trace["rounds"])
    print(f"example seed {ex_sorted.seed}: T={ex_sorted.n_rounds}, {multi} multi-event rounds, "
          f"{reordered} visibly reordered")

    pdf_path = os.path.join(OUT, "eventorder.pdf")
    with PdfPages(pdf_path) as pdf:
        out = PdfAndPng(pdf, OUT)
        page_title(out, pool)
        page_tokenization(out, ex_sorted)
        page_trace(out, ex_sorted, title="One real reasoning trace, as the generator emits it",
                   note="Every round is in ascending node index. That is a convention of\n"
                        "solve_with_dd_trace, not a fact about the graph -- and it is what\n"
                        "this run removed.")
        page_trace(out, ex_perm, title="The SAME instance, events permuted within each round",
                   note="Same graph, same hints, same rounds as a SET, same T, same token count.\n"
                        f"The ONLY difference from the page before: {multi} of these {ex_perm.n_rounds} "
                        f"rounds have more than one\ndeduction, and {reordered} of those came out "
                        "reordered (rounds 2 and 3 above). A\nsingle-deduction round cannot be "
                        "permuted -- which is why the effect is small\nto look at and still removes "
                        "the convention entirely.")
        page_control(out, sweep)
        page_pool(out, pool, om)
        page_depth(out, depth, pool)
        page_prereg(out, pool, sweep, om, depth)
        page_next(out)
    print(f"{out.n} pages -> {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
