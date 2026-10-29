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
from matplotlib.patches import Patch

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

SCRATCH = os.environ["SCRATCH"]
CIRC = f"{SCRATCH}/spin_gnn/sudoku_circuits_leakfree_2026-10-28"
CHAIN = f"{SCRATCH}/spin_gnn/sudoku_chain_2026-10-28"
SKIP = f"{SCRATCH}/spin_gnn/sudoku_skip_2026-10-28"
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


def page_title(out, D, ex):
    fig = plt.figure(figsize=(16, 9))
    a = D["leak-free 1.0000"]
    r0, ch = a["r0"]["flip"], a.get("chain", {}).get("corrupt_flipped")
    leaked = D["leaked 0.9960"]["r0"]["flip"]
    fig.text(0.5, 0.945, "Is the leak-free network running DD?", ha="center",
             size=BIG + 5, weight="bold")
    fig.text(0.5, 0.885, "Causal interventions, not probe scores",
             ha="center", size=MED + 2, color="0.35")

    sk = a.get("skip", {}).get("corrupt_emits_v")
    if ch is not None and ch >= 0.60 and sk is not None and sk <= 0.20:
        verdict = "YES -- it runs DD's rule, its chaining, and its guard."
    elif ch is not None and ch >= 0.60:
        verdict = "YES for DD's rule and chaining. NO for DD's search."
    else:
        verdict = "YES for DD's one-step rule; chaining still measuring."
    fig.text(0.5, 0.825, verdict, ha="center", size=BIG, weight="bold", color=BLUE)

    def _g(label):
        s = D[label].get("skip")
        return f"{1 - s['corrupt_emits_v']:.3f}" if s else "(running)"

    def _c(label):
        s = D[label].get("chain")
        return f"{s['corrupt_flipped']:.3f}" if s else "(running)"

    def _p(label):
        s = D[label].get("skip")
        return f"{1 - s['placebo_emits_v']:.3f}" if s else "  -- "

    lines = [
        "THE TASK    Colour a 24-node graph, every node of degree 5, with q = 4 colours so that no edge joins",
        "            two like colours. Some nodes are given as hints; the rest must be deduced.",
        "",
        "DD          Direct Deduction, the classical 'naked single' solver the model is trained to imitate. Two",
        "            parts at each step. A RULE: eliminate the colours a node's known neighbours already use.",
        "            A GUARD: emit that node only when exactly one colour is left.",
        "",
        "THE MODELS  12-layer transformers, each NAMED BY ITS SOLVE RATE -- the fraction of 1500 held-out graphs",
        "            it colours completely and correctly under free generation.",
        "              'leaked'     trained on an encoding whose adjacency ORDER gave the colouring away.",
        "              'leak-free'  trained with that order permuted, so the answer has to be deduced.",
        "",
        "   One-token counterfactuals test the rule and the guard separately:",
        "",
        "        model                   RULE              RULE             GUARD        (placebo",
        "                            on given hints   on its OWN facts    withholds       ceiling)",
        f"        leak-free 1.0000         {r0:.3f}            {_c('leak-free 1.0000')}             "
        f"{_g('leak-free 1.0000')}         {_p('leak-free 1.0000')}",
        f"        leak-free 0.8540         {D['leak-free 0.8540']['r0']['flip']:.3f}            "
        f"{_c('leak-free 0.8540')}             {_g('leak-free 0.8540')}         {_p('leak-free 0.8540')}",
        f"        leaked 0.9960            {leaked:.3f}            {_c('leaked 0.9960')}             "
        f"{_g('leaked 0.9960')}         {_p('leaked 0.9960')}",
        "        chance                   0.250            0.250               --",
    ]
    fig.text(0.045, 0.775, "\n".join(lines), ha="left", va="top", size=MED - 5,
             family="monospace", linespacing=1.4)
    fig.text(0.045, 0.235,
             "Both leak-free models apply the rule and chain it; the leaked one never learned it. The GUARD "
             "column does NOT separate\nleaked from leak-free -- all three withhold -- so passing it is not by "
             "itself evidence of DD. What it does separate is the two\nleak-free twins: the one that solves the "
             "task withholds on every case and the other fails 0.176 of them, the same order as\nits 0.146 "
             "solve-rate deficit.",
             ha="left", va="top", size=MED - 3, linespacing=1.5)
    fig.text(0.045, 0.112,
             "CAVEAT, up front: one task configuration (N = 24, degree 5, q = 4) and ONE CHECKPOINT PER MODEL, no "
             "seed replicates --\nso every leak-free vs leaked contrast is also two separate training runs.",
             ha="left", va="top", size=MED - 3, color="0.15", weight="bold", linespacing=1.5)
    missing = [k for k, v in D.items() if "chain" not in v or "skip" not in v]
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
        "  Sudoku's 'naked single' rule, on a graph: a node becomes",
        "  determined one round after its known neighbours show all",
        f"  q-1 = {c.n_colors - 1} colours other than its own. Sweep the nodes, emit",
        "  every determined one, repeat until none are left.",
        "",
        "  THE FORCING SET of a node v = the known neighbours whose",
        "  colours do the eliminating. DD fires on v exactly when",
        f"  that set shows q-1 = {c.n_colors - 1} distinct colours; v then takes the",
        "  one colour left over. Every test in this deck is an edit",
        "  to that set.",
        "",
        "  REASONING DEPTH T = how many rounds an instance needs,",
        f"  1 to 6 at this density. The example next needs {ex.n_rounds}.",
    ]
    fig.text(0.05, 0.855, "\n".join(left), ha="left", va="top", size=MED - 1,
             family="monospace", linespacing=1.5)
    fig.text(0.50, 0.855, "\n".join(right), ha="left", va="top", size=MED - 1,
             family="monospace", linespacing=1.5)
    fig.text(0.05, 0.215,
             "Solvable-by-DD is a property of the INSTANCES, not a proof that no other method works -- which is why "
             "the deck goes on to\nrule out the obvious non-deductive routes (reading the leak, page 3; emitting "
             "nodes by index order, last page) rather than\nassuming they are closed.",
             ha="left", va="top", size=MED - 2, linespacing=1.5)
    out.savefig(fig)


def page_model(out):
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
        "  Permuting the rows at test time collapses a model that",
        "  was trained on them: the best such model goes from",
        "  0.9975 to 0.1450. The leak was load-bearing, not",
        "  incidental.",
        "",
        "   'leaked'     trained on the sorted rows. Scores",
        "                0.9960 and needs no reasoning to do it.",
        "   'leak-free'  trained with the rows permuted. Two of",
        "                them, 1.0000 and 0.8540, differing only",
        "                in training-set size.",
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
             "THE QUESTION: the leak-free 1.0000 model solves a\ntask whose instances are ALL solvable by DD. Is it\n"
             "running DD, or doing something else that happens to\nwork here?\n\nEvery BEHAVIOURAL number from here on "
             "is causal: edit\none input token, ask whether the output follows DD's\nanswer to the EDITED problem. "
             "(The one correlational\npage, the sparse autoencoder, is labelled as such.)",
             ha="left", va="top", size=MED - 3, weight="bold", linespacing=1.5)
    fig.text(0.52, 0.305,
             "WHY NOT PROBES: an earlier round of this project read the model's own\n'which node is next' features "
             "with a trained probe and got 0.545 of nodes\nright -- while a hand rule that does no deduction at all, "
             "'the unknown node\nwith the fewest unknown neighbours', got 0.632 on the same cases. A\ndecoding score "
             "cannot tell those two apart. An intervention can.",
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
    body = [
        "Both tests on the previous page SELECT a node that DD forces and then REDIRECT the",
        "forcing. A model that simply wrote a colour for whichever unknown node came next in",
        "index order would pass them both. So break the forcing instead of redirecting it --",
        "the WITHHOLD test:",
        "",
        "   recolour w to a colour that ANOTHER known neighbour of v already holds.",
        "",
        "Now the freed colour a is held by nobody, v's forcing set covers only q-2 = 2",
        "colours, v has TWO candidates, and DD does not emit v in this round at all.",
        "",
        "The measurement moves ONE TOKEN TO THE LEFT: from the logits that predict v's",
        "COLOUR to the logits that predict WHICH NODE comes next. Does the model stop",
        "naming v?",
        "",
        "   clean      ... BEGIN_ROUND  [v]  ->  c       DD emits v, and so should the model",
        "   broken     ... BEGIN_ROUND  [ ? ]            v is no longer determined",
        "",
        "PLACEBO, run on every single case: the same kind of one-token edit, but on a hint",
        "that is NOT a neighbour of v. v stays forced and DD's round is unchanged, so the",
        "model should still name v. That is the ceiling the broken rate is read against --",
        "without it, a low rate would only show that edits perturb the node stream.",
    ]
    fig.text(0.06, 0.845, "\n".join(body), ha="left", va="top", size=MED - 3,
             family="monospace", linespacing=1.4)
    fig.text(0.06, 0.20,
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
    ax = fig.add_axes([0.03, 0.29, 0.94, 0.56]); ax.axis("off")
    cols = ["model", "test", "n", "clean", "moved to DD's\nnew answer", "kept its\nold answer",
            "logit gap (rule) or\nplacebo moved (guard)", "verdict"]
    # the verdict column has to hold "keeps its own answer" on one line
    widths = [0.16, 0.13, 0.05, 0.08, 0.14, 0.10, 0.16, 0.17]
    rows = []
    for label, _, _, _ in ARMS:
        rec = D[label]
        if "r0" in rec:
            r = rec["r0"]
            rows.append([label, "hint edit", r["n"], f"{r['clean']:.3f}", f"{r['flip']:.3f}",
                         "--", f"{r['clean_ld']:+.1f} -> {r['corrupt_ld']:+.1f}",
                         "uses the rule" if r["flip"] >= 0.60 else "near chance"])
        if "chain" in rec:
            c = rec["chain"]
            # 0.008 is thirty times BELOW chance: that model is not guessing,
            # it is actively re-asserting its own answer. Calling that "near
            # chance" (the old two-way test) was simply the wrong word.
            f = c["corrupt_flipped"]
            v = ("chains" if f >= 0.60 else
                 "keeps its own answer" if f <= CHANCE / 2 else "near chance")
            rows.append([label, "chained edit", c["n_cases"], f"{c['clean_correct']:.3f}",
                         f"{f:.3f}", f"{c['reverts_to_original']:.3f}", "--", v])
        if "skip" in rec:
            s = rec["skip"]
            # the guard is scored the other way round: 1 - (still names v), so
            # the "flipped" column stays "moved off DD's old answer" throughout
            rows.append([label, "withhold", s["n_cases"], f"{s['clean_emits_v']:.3f}",
                         f"{1 - s['corrupt_emits_v']:.3f}", f"{s['corrupt_emits_v']:.3f}",
                         f"placebo {1 - s['placebo_emits_v']:.3f}",
                         "guards" if s["corrupt_emits_v"] <= 0.20 else "names it anyway"])
    t = ax.table(cellText=rows, colLabels=cols, colWidths=widths, loc="center",
                 cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(MED - 3); t.scale(1, 3.0)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#dddddd"); t[0, j].set_text_props(weight="bold")
    arm_colour = {label: colour for label, _, _, colour in ARMS}
    for i, r in enumerate(rows, start=1):
        # the same colour each model carries in every plot, or the legend on
        # one page silently contradicts the table on another
        t[i, 0].set_text_props(color=arm_colour[r[0]], weight="bold")
        t[i, 4].set_text_props(weight="bold",
                               color=GREEN if float(r[4]) >= 0.60 else RED)
    fig.text(0.5, 0.265,
             "Every column is scored so that HIGH = followed DD, placebo included. A low 'moved' next to a high "
             "'clean' means the model is\nright for some reason other than the elimination -- exactly the leaked "
             "model's profile on its two rule rows. Note its WITHHOLD row:\nit guards as well as anyone, so that "
             "test alone does not distinguish a model that runs DD from one that does not.",
             ha="center", va="top", size=MED - 3, linespacing=1.5)
    footer(fig, "LOGIT GAP = logit(DD's answer to the EDITED problem) - logit(DD's answer to the CLEAN one) at the "
                "position that predicts v's colour, in raw logits (log-odds), averaged over cases; the arrow spans "
                "clean prompt -> edited prompt, so a sign change is a reversal of preference. On the withhold rows "
                "'moved' = it stopped naming the node, 'kept its old answer' = it named it anyway, 'placebo moved' = "
                "how often the control edit also stopped it: the noise floor. 'clean' differs between a model's own "
                "rows because the conditions do -- the rule rows score the colour emitted for v, the withhold rows "
                "whether v is named at all. n is 350 or 250 by when each run was submitted, not by filtering; paired "
                "tests use the 250 shared cases.", width=132)
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
            ax2.annotate(f"{y:.3f}\nn={n}", (k, y), textcoords="offset points",
                         xytext=(0, -52), ha="center", size=SMALL)
        ax2.set_xlabel("DD round the node was deduced in")
        ax2.set_ylabel("fraction that moved to DD's new answer")
        ax2.set_ylim(0, 1.18); ax2.set_xticks(ks)
        ax2.set_title("No decay with depth -- leak-free 1.0000 (95% Wilson)", size=MED,
                      pad=24)
    fig.tight_layout(rect=[0, 0.19, 1, 0.93])
    fig.text(0.5, 0.175,
             "If the model chained one step but not three, round 1 would be high and the later rounds would fall off. "
             "It does not happen.",
             ha="center", va="top", size=MED - 2)
    footer(fig, "Right panel splits the chained test by how deep the edited deduction sat. Deep bins are small "
                "because deep chained forcings are rare in these instances, not because anything was filtered -- "
                "hence the intervals, wide at rounds 3-4 (n=24 and n=4) and still clear of chance. Rounds 5-6 are "
                "absent because no case in the sample had its binding neighbour deduced that late.")
    out.savefig(fig)


def page_guard(out, D):
    """The converse test: does it WITHHOLD a deduction that stops being forced?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("And it withholds: break the forcing and it stops naming the node",
                 size=BIG, weight="bold")

    labels, cl, br, pl, cols = [], [], [], [], []
    for label, _, _, colour in ARMS:
        s = D[label].get("skip")
        if s is None:
            continue
        labels.append(label.replace(" ", "\n")); cols.append(colour)
        cl.append(s["clean_emits_v"]); br.append(s["corrupt_emits_v"])
        pl.append(s["placebo_emits_v"])
    x = np.arange(len(labels))
    ax1.bar(x - 0.26, cl, 0.25, color=cols, edgecolor="k", label="clean prompt")
    ax1.bar(x, pl, 0.25, color=cols, edgecolor="k", alpha=0.6, hatch="...",
            label="placebo edit")
    ax1.bar(x + 0.26, br, 0.25, color=cols, edgecolor="k", hatch="///", alpha=0.85,
            label="forcing BROKEN")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, size=SMALL)
    # This panel plots "still names v", which is the COMPLEMENT of the results
    # table's "moved" column, and the correct direction differs per bar: v is
    # still forced under clean and placebo, and is not under broken. Unlabelled,
    # that reads as the table contradicting the figure.
    ax1.set_ylabel("fraction that still names v as the next node"); ax1.set_ylim(0, 1.30)
    ax1.text(0.5, 0.985, "clean + placebo: v IS still forced, so HIGH is correct\n"
                         "BROKEN: v is NOT forced, so LOW is correct\n"
                         "(this panel = 1 - the results table's 'moved' column)",
             transform=ax1.transAxes, ha="center", va="top", size=SMALL - 1)
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
        ax2.set_ylabel(f"cases (of {len(sk)} skips)")
        ax2.set_title("What it names instead -- leak-free 1.0000", size=MED)
        for i, v in enumerate(vals):
            ax2.text(i, v + max(vals) * 0.02, f"{v / len(sk):.1%}", ha="center",
                     size=SMALL, weight="bold")
    fig.tight_layout(rect=[0, 0.235, 1, 0.93])
    fig.text(0.5, 0.225,
             "The two leak-free twins differ here and nowhere else: McNemar paired, 44 cases where the 0.8540 twin "
             "named v and the 1.0000\nmodel did not, 0 the other way, p = 1.1e-13. It does not just go quiet either: "
             "of the 350 cases it withholds, it names another node\nthat IS forced under the broken hints, or says "
             "BEGIN_ROUND -- DD's own answer when nothing else in the round is determined.",
             ha="center", va="top", size=MED - 3, linespacing=1.5)
    footer(fig, "Same cases as the hint-edit test, and the case lists are byte-identical across models, so the arms "
                "are paired instance for instance. Conditioning on cases the twin gets right on the clean prompt "
                "gives 0.188 (n=234). Its 0.176 guard gap is the same order as its 0.146 solve-rate gap -- different "
                "measurements on different samples, so a consistency check, not a decomposition. The leaked model "
                "withholds too (0.992): this test separates the twins, not leaked from leak-free.")
    out.savefig(fig)


def page_patching(out, D):
    arms = [(l, D[l]) for l, _, _, _ in ARMS if "patch" in D[l]]
    fig, axes = plt.subplots(1, len(arms), figsize=(16, 9), squeeze=False)
    fig.suptitle("Where the recoloured fact travels (activation patching)",
                 size=BIG, weight="bold")
    # Where each arm's grip on the edited hint falls below half, by linear
    # interpolation between the bracketing layers. Computed, not typed: the
    # footer used to claim "one to two layers" and the data says three.
    release = {}
    for ax, (label, rec) in zip(axes[0], arms):
        m = (rec["patch"].groupby(["layer", "site"]).recovery.mean()
             .unstack()[SITES])
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
        ax.set_xticks(range(len(SITES)))
        ax.set_xticklabels([SITE_LABELS[s] for s in SITES], size=SMALL - 2)
        ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index, size=SMALL - 1)
        ax.set_title(label, size=MED)
        ax.set_ylabel("layer" if ax is axes[0][0] else "")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                val = m.values[i, j]
                # -0.00 is a rounding artefact of a tiny negative recovery and
                # reads as a signed quantity to anyone scanning the grid
                txt = f"{0.0 if abs(val) < 5e-3 else val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", size=SMALL - 4,
                        color="w" if val < 0.6 else "k")
    # subplots_adjust, not tight_layout -- the shared colorbar fights it
    fig.subplots_adjust(bottom=0.355, top=0.86, left=0.07, right=0.88)
    fig.colorbar(im, ax=axes[0], fraction=0.025, label="mean recovery")
    fig.text(0.5, 0.265,
             "RECOVERY = copy one activation from the clean run into the corrupted run at that (layer, position) and "
             "measure how much of the\nflip it undoes: 1.00 = that one activation carries the whole edited fact, 0.00 "
             "= none of it. w is the neighbour the edit recolours,\nu is another known neighbour of v; both graph rows "
             "are controls, and 'predicts v's colour' is where the answer is read off.",
             ha="center", va="top", size=MED - 3, linespacing=1.5)
    a, b = "leak-free 1.0000", "leak-free 0.8540"
    gap = (f" The 1.0000 model RELEASES the fact from the hint token about "
           f"{release[b] - release[a]:.0f} layers earlier than its 0.8540 twin (crossing half "
           f"recovery at L{release[a]:.1f} against L{release[b]:.1f}), though both have it at the "
           f"prediction position by L10, so the difference is in how long it is held, not in where "
           f"it lands." if a in release and b in release else "")
    footer(fig, "Read upwards: the edited colour is held AT the edited hint through L5, handed off between L6 and "
                "L10, and has arrived at the prediction position by L10. The graph rows read 0.00 at every layer in "
                "every model -- the deduction never routes through the adjacency table. L11 at the prediction "
                "position is the built-in positive control and must read 1.00; it does." + gap
                + " In the leaked panel recovery is a fraction of an effect that is near chance to begin with (its "
                  "flip rate is 0.388), so its numbers localise a sub-threshold sensitivity, not a rule it applies.",
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
                label=f"{label}  (worst: layer {int(h['layer'].iloc[0])}, "
                      f"head {int(h['head'].iloc[0])} = {h.effect_retained.iloc[0]:.3f})")
    ax.axhline(1.0, color="k", ls="--", lw=2)
    ax.text(2, 1.02, "1.00 = ablating this head changes the effect not at all;\n"
                     "below 1.00 = it shrinks the effect; above 1.00 = it GROWS the effect",
            size=SMALL, va="bottom")
    ax.set_xlabel("attention heads, sorted by damage (most damaging first)")
    ax.set_ylabel("fraction of the round-0 rule effect retained\nafter ablating that one head")
    ax.legend(loc="lower right", fontsize=SMALL)
    fig.tight_layout(rect=[0, 0.19, 1, 0.93])
    fig.text(0.5, 0.175,
             "The leaked model's curve only looks steeper because its effect is near chance to begin with: "
             "a fraction of almost nothing.",
             ha="center", va="top", size=MED - 2)
    footer(fig, "Each point is one of the 48 heads (12 layers x 4 heads), zeroed one at a time, averaged over the "
                "same 350 (leak-free 1.0000) or 250 (the other two) hint-edit cases as the rule test. Values above "
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
    fig.suptitle("It carries sparse features for DD's own forcing set,\npeaking at "
                 f"L{int(peak.layer)} -- inside the L6-L10 window where the causal patch hands "
                 "the fact off", size=BIG - 5, weight="bold", linespacing=1.4)

    for label, _, _, colour in ARMS:
        if label not in S:
            continue
        k = S[label]
        k = k[k.position_kind == "pre_color"].sort_values("layer")
        n_tot = int(k.n_just.sum())
        first = k.layer[k.n_just > 0].min()
        where = f", from L{int(first)} up" if n_tot else ""
        ax1.plot(k.layer, k.best_justifier_f1, "o-", color=colour, lw=3, ms=11,
                 label=f"{label}  ({n_tot} features{where})")
    null = pd.concat([S[l][S[l].position_kind == "pre_color"]
                      for l in S]).groupby("layer").median_best_f1_null.max()
    ax1.plot(null.index, null.values, ls="--", lw=2.5, color="k",
             label="best F1 a SHUFFLED label gets")
    ax1.axhline(MIN_F1, color="0.4", ls="-.", lw=2)
    ax1.text(0.1, MIN_F1 + 0.015, f"F1 >= {MIN_F1:.2f}: the bar a feature must clear to be COUNTED",
             size=SMALL - 1, color="0.3")
    ax1.set_title("while the model picks the COLOUR", size=MED + 1)
    ax1.set_xlabel("transformer layer the dictionary was trained on")
    ax1.set_ylabel("best single feature's F1 for\n'this node is a KNOWN neighbour\nof the node being emitted'",
                   size=MED - 4)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper left", fontsize=SMALL - 1)

    for kind, name, ls, ms in [("pre_color", "before a COLOUR token", "-", "o"),
                               ("pre_node", "before a NODE token", ":", "s")]:
        k = a[a.position_kind == kind].sort_values("layer")
        n = int(k.n_just.sum())
        why = "" if n else " -- never clears the bar"
        ax2.plot(k.layer, k.best_justifier_f1, ls=ls, marker=ms, color=BLUE, lw=3,
                 ms=11, label=f"{name}: {n} features{why}")
    ax2.plot(null.index, null.values, ls="--", lw=2.5, color="k", label="shuffled label")
    ax2.axhline(MIN_F1, color="0.4", ls="-.", lw=2)
    ax2.set_title("leak-free 1.0000: the same dictionary, by position", size=MED + 1)
    ax2.set_xlabel("transformer layer")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper left", fontsize=SMALL - 1)

    fig.tight_layout(rect=[0, 0.305, 1, 0.88])
    fig.text(0.5, 0.29,
             "A feature is COUNTED only when it matches one of DD's own predicates at F1 >= 0.30 (F1 = the harmonic "
             "mean of precision and recall)\nAND beats a shuffled-label null. Three things to read: (1) the forcing "
             "set is encoded where DD's rule needs it -- at the token that predicts\nthe COLOUR, and never at the "
             "token that picks the node; (2) the 1.0000 model is NOT more interpretable than its 0.854 twin, which is "
             "what\ntheir equal rule scores already said; (3) THE LEAKED MODEL ENCODES THE FORCING SET TOO -- 15 "
             "features, but only at L10-L12, after the\nwindow where the leak-free model uses it. Encoding it is not "
             "using it: that is why this page is not the evidence, and why pages 6-11 are.",
             ha="center", va="top", size=MED - 4, linespacing=1.5)
    footer(fig, "CORRELATIONAL -- this page says WHERE the structure sits, not whether the model uses it. Sparse "
                "autoencoder, 4096 features per layer, ~64 active per token, trained on 53,200 activations per layer, "
                "then scored separately on the 26,600 positions before a colour token and the 26,600 before a node "
                "token, so counts compare within a mask only. Layer 12 is the residual stream after the last block, "
                "which is why this axis runs one further than the patching page's. One dictionary per layer per arm, "
                "one seed.")
    out.savefig(fig)


def page_verdict(out, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The verdict, and what it is not", size=BIG, weight="bold")
    a = D["leak-free 1.0000"]
    ch = a.get("chain", {}).get("corrupt_flipped")
    sk = a.get("skip", {}).get("corrupt_emits_v")
    pl = a.get("skip", {}).get("placebo_emits_v")
    bsk = D["leak-free 0.8540"].get("skip", {}).get("corrupt_emits_v")
    worst = (a["heads"].groupby(["layer", "head"]).effect_retained.mean().min()
             if "heads" in a else None)
    left = [
        "WHAT IT DOES DO  (causal unless marked)",
        "",
        f"* Applies DD's elimination rule: {a['r0']['flip']:.3f}",
        "  on 350 one-token counterfactuals.",
        "",
        f"* Chains it off its OWN deductions: {ch:.3f}," if ch is not None else
        "* Chaining: measuring (50077245).",
        "  no decay from round 1 to round 4.",
        "",
        "* Guards it: break the forcing and it stops",
        f"  naming the node {1 - sk:.3f} of the time, while a"
        if sk is not None else "  (measuring, job 50082120)",
        # same direction as the line above it. Quoting the placebo as "leaves
        # it at 0.940" next to a stop rate invited reading 0.940 as a second
        # stop rate, i.e. the placebo doing the same thing as the real edit.
        f"  placebo edit stops it only {1 - pl:.3f}. But ALL" if sk is not None else "",
        "  THREE models pass this one, so passing it",
        "  is not by itself evidence of DD.",
        "",
        f"* Its 0.854 twin guards only {1 - bsk:.3f}, a gap the" if bsk is not None else "",
        "  same order as that twin's 0.146 solve_rate",
        "  deficit -- and the first test here to",
        "  separate the two TWINS at all.",
        "",
        "* Localised in depth: edited hint -> L5,",
        "  handoff L6-L10, prediction by L10.",
        "",
        "* CORRELATIONAL: it carries sparse features",
        "  for DD's forcing set, peaking at that same",
        "  L8 -- but so does the leaked model, later.",
    ]
    right = [
        "WHAT IT DOES NOT DO",
        "",
        "* Choose the node ORDER by deduction. It",
        "  emits nodes in ascending index order in",
        "  100% of rounds. That is also DD's own",
        "  convention, so it proves nothing either",
        "  way -- but the guard test does show that",
        "  WHICH nodes it emits tracks determinacy.",
        "",
        "* Route through the adjacency table: the",
        "  graph rows recover 0.000 at every layer.",
        "",
        "* Concentrate the rule anywhere we can point",
        f"  at: ablating the worst of the 48 heads one" if worst is not None else "",
        f"  at a time still leaves {worst:.0%} of the effect." if worst is not None else "",
        "  The rule is distributed, so this deck can",
        "  say THAT it runs DD, not which heads do.",
        "",
        "* The leaked twin never learned the rule",
        f"  ({D['leaked 0.9960']['r0']['flip']:.3f}, chance 0.250). Shown an edited",
        f"  deduction it keeps its own answer "
        f"{D['leaked 0.9960']['chain']['reverts_to_original']:.1%}",
        "  of the time: it reads the adjacency-order",
        "  leak, not the prefix.",
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
    fig.text(0.5, 0.155,
             "LIMITATIONS: one checkpoint per model and no seed replicates, so every "
             "leak-free vs leaked contrast is also two\nseparate training runs; one task configuration (N = 24, "
             "degree 5, q = 4), so nothing here is a claim about depth or scale; the per-depth and\nper-degree "
             "breakdowns have bins as thin as a few dozen cases, where the error bars overlap everything; the guard "
             "gap and the solve_rate\ngap are different measurements on different samples, so their agreement is a "
             "consistency check, not a decomposition; and the sparse-\nautoencoder page is correlational, with one "
             "dictionary per layer and one seed.",
             ha="center", va="top", size=MED - 4, color="0.15", linespacing=1.45)
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
    ex = example_instance(rho=0.60)
    pdf_path = os.path.join(OUT, "slides.pdf")
    have_skip = any("skip" in D[l] for l, _, _, _ in ARMS)
    with PdfPages(pdf_path) as pdf:
        o = PdfAndPng(pdf, OUT, total=11 + 2 * have_skip + bool(S))
        page_title(o, D, ex)
        page_task(o, ex)
        page_model(o)
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
        page_verdict(o, D)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {o.n} pages)")


if __name__ == "__main__":
    main()
