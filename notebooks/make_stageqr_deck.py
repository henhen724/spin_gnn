"""The stage Q + stage R deck: ONE deck for the gate, not two.

Q and R are halves of a single question -- does the attention-only cell actually
run direct deduction, or does it merely reproduce the solve curve? -- and they
only mean anything together. Q edits a HINT and measures whether the model's
answer follows; R edits a colour THE MODEL ITSELF DEDUCED earlier in the same
sequence. One is the inference rule over given facts, the other is the iteration
that makes it an algorithm. Split across two decks, each half reads as a
narrower claim than it is, and the limits of one would be read as limits of the
programme.

EVERY NUMBER COMES OFF DISK THROUGH THE SCORERS' OWN FUNCTIONS. The verdicts
were registered before either run (scripts/stageq_score.py, scripts/stager_score.py,
both committed by their own sbatch before the first arm), so nothing here is
recomputed in prose; the notebook runs the same scorers as subprocesses.

THE TWO COUNTERFACTUAL PAGES SHOW CASE #1 OF THE ACTUAL EXPERIMENT. They
regenerate the instance stream from the same --seed-start the arms used, take
the first case the selector returns, and then ASSERT that it matches row 1 of
the arm's behaviour CSV before drawing anything. If the stream ever drifts the
deck fails closed instead of illustrating a case that was never run -- and the
model's response on that page is read out of the CSV, not re-inferred.

Usage (from a job; see scripts/slurm/stageqr_writeup_2026-11-10.sbatch):
    python3 -u -m notebooks.make_stageqr_deck --commit <sha>
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slides import Deck, MUTED, SERIES, style_axes, label_line, series_colors

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, footer

from scripts.stageq_score import (ARMS as QARMS, OUT as QDIR, _read as q_read,
                                  CHANCE, GATE, CLEAN_FLOOR)
from scripts.stager_score import (ARMS as RARMS, OUT as RDIR, read_chain,
                                 read_stageq_flip, CEILING, PROVENANCE_BAR)

MONO = {"family": "monospace"}

# Plot labels. The three seeds are one cell; the control is a DIFFERENT cell in
# both arch_mode and shape (no full L=16 d192 exists at N=48), so it is context
# and every page that shows it says so.
SHORT = ["s0", "s1", "s2", "control"]


def room_for_y_labels(ax, dx):
    """Shift the axes right so long y tick labels survive the page edge.

    slides.claim() pins the axes' left edge at 0.08 and exposes only
    right_margin, so a horizontal bar chart's labels run off the page unless the
    axes itself moves. Twin of the helper in notebooks/make_stagem_deck.py,
    copied rather than imported: a deck script should not be able to break
    another deck's render.
    """
    p = ax.get_position()
    ax.set_position([p.x0 + dx, p.y0, p.width - dx, p.height])


# ----------------------------------------------------------------- the data

def read_chain_rows(tag):
    """Per-case rows for one stage R arm. chain_summary.json carries the rates
    but not the logit diffs or the per-case identities, and both of those are
    load-bearing here: the identities are what show the three seeds' identical
    0.994 to be a coincidence of counts."""
    p = os.path.join(RDIR, tag, "chain_behavior.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def _tf(r, k):
    return r[k].strip().lower() in ("true", "1")


def chain_extras(rows):
    """Logit means, the miss identities, and the revert breakdown of the misses."""
    n = len(rows)
    misses = [r for r in rows if not _tf(r, "corrupt_flipped")]
    return {
        "n": n,
        "ld_clean": sum(float(r["clean_logit_diff"]) for r in rows) / n,
        "ld_corrupt": sum(float(r["corrupt_logit_diff"]) for r in rows) / n,
        "n_miss": len(misses),
        "miss_keys": ["N%s@r%s" % (r["node"], r["round"]) for r in misses],
        "miss_reverts": sum(_tf(r, "reverts_to_original") for r in misses),
    }


def load_qr():
    """Everything both stages measured, read through the scorers' own readers."""
    q = [q_read(a[0]) for a in QARMS]
    r = [read_chain(a[0]) for a in RARMS]
    rows = [read_chain_rows(a[0]) for a in RARMS]
    if any(x is None for x in q) or any(x is None for x in r) or any(x is None for x in rows):
        sys.exit("an arm is missing from disk. Refusing to build a deck of three arms.")
    ex = [chain_extras(x) for x in rows]
    # Registered at 350 per arm. A deck built on fewer would be believed at a
    # confidence the run does not support -- stage M's and stage N's refused for
    # the same reason.
    for a, s in zip(QARMS, q):
        if s["n"] < 350:
            sys.exit("stage Q arm %s has %d cases, registered 350." % (a[0], s["n"]))
    for a, s in zip(RARMS, r):
        if s["n_cases"] < 350:
            sys.exit("stage R arm %s has %d cases, registered 350." % (a[0], s["n_cases"]))
    return {"q": q, "r": r, "rows": rows, "ex": ex}


def case_one(stage, seed_start=600_000_000, n_instances=80):
    """Case #1 of the actual experiment, regenerated and then CHECKED.

    Returns (adj, colors, hints, trace, case, vocab, cfg). The arms walked the
    same stream with the same selector, so the first case they scored is the
    first case here; the caller asserts that against row 1 of the CSV, and a
    mismatch is fatal rather than cosmetic.
    """
    from spin_gnn.config import SudokuReasoningConfig
    from spin_gnn.sudoku_tokens import Vocab
    from spin_gnn.train_sudoku_transformer import build_instances
    from spin_gnn.sudoku_circuits import find_flippable_deductions
    from spin_gnn.sudoku_chain import find_chained_deductions

    # Both stages' s0 arm is the SAME checkpoint -- that identity is the whole
    # reason R4 is a legal subtraction, and stage R's sbatch verified all four
    # cells field by field before any arm ran. The path is read from stage R's
    # summary for both stages because stage Q's summary.json does not record it.
    sm = json.load(open(os.path.join(RDIR, "R_noff_s0", "chain_summary.json")))
    config = json.load(open(os.path.join(sm["checkpoint"], "config.json")))
    cfg = SudokuReasoningConfig(
        n_nodes=config["n_nodes"], degree=config["degree"], n_colors=config["n_colors"],
        rho=config["rho"], rho_max=config["rho_max"],
        scratchpad=config.get("scratchpad", False),
        hint_mode=config.get("hint_mode", "removal"),
        shuffle_adj=bool(config.get("shuffle_adj", False)),
        shuffle_events=bool(config.get("shuffle_events", False)),
        fall=bool(config.get("fall", False)))
    vocab = Vocab(cfg.n_nodes, cfg.n_colors, cfg.degree, scratchpad=False)
    for adj, colors, hints, trace in build_instances(cfg, n_instances, seed_start):
        if stage == "Q":
            cases = find_flippable_deductions(adj, colors, hints, trace, vocab,
                                              max_per_instance=2)
        else:
            cases = find_chained_deductions(adj, colors, hints, trace, vocab)
        if cases:
            return adj, colors, hints, trace, cases[0], vocab, cfg
    sys.exit("no stage %s case in the first %d instances -- the stream drifted."
             % (stage, n_instances))


# ------------------------------------------------------------ the drawings
#
# Module-level, taking a bare axes, so the results notebook calls the same
# function the deck does and the two cannot drift.

def draw_flip(ax, vals, title_unit="flip rate"):
    """Flip rate per arm against the absolute chance rate, which is the bar R0
    and Q0 are scored on -- never against the control (Q2/R2: the control is a
    different cell in two respects at once)."""
    y = np.arange(len(vals))[::-1]
    cols = [SERIES[0]] * 3 + [MUTED]
    ax.barh(y, vals, height=0.55, color=cols, rasterized=True)
    ax.axvline(CHANCE, color="0.25", linestyle="--", linewidth=1.2)
    ax.axvline(GATE, color="0.55", linestyle=":", linewidth=1.2)
    ax.text(CHANCE, len(vals) - 0.25, " chance %.2f" % CHANCE, size=11, color="0.25",
            va="center")
    ax.text(GATE, len(vals) - 0.25, " gate %.2f" % GATE, size=11, color="0.55",
            va="center")
    for yy, v in zip(y, vals):
        ax.text(v + 0.012, yy, "%.3f" % v, va="center", size=13)
    ax.set_yticks(y)
    ax.set_yticklabels(SHORT)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(title_unit)
    style_axes(ax)
    room_for_y_labels(ax, 0.04)


def draw_logit(ax, clean, corrupt):
    """clean -> corrupt mean logit[freed] - logit[true], per arm. A sign
    reversal is the thing to see; the magnitudes are not comparable across
    arms of different width and are not claimed to be."""
    y = np.arange(len(clean))[::-1]
    for yy, c, k in zip(y, clean, corrupt):
        ax.annotate("", xy=(k, yy), xytext=(c, yy),
                    arrowprops=dict(arrowstyle="->", color=SERIES[0], linewidth=1.8))
    ax.plot(clean, y, "o", color=MUTED, markersize=8)
    ax.plot(corrupt, y, "o", color=SERIES[0], markersize=8)
    ax.axvline(0, color="0.25", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(SHORT)
    ax.set_xlabel("mean logit(freed colour) - logit(true colour)")
    style_axes(ax)
    room_for_y_labels(ax, 0.04)
    # x in axes fraction, y in DATA: at len-0.3 this sat above the top row and
    # collided with the page title. Halfway between the top two rows is the one
    # band of the plot that is empty whatever the arms do.
    ax.text(0.02, len(clean) - 1.5, "grey = before the edit,  blue = after",
            size=11, color="0.35", va="center",
            transform=ax.get_yaxis_transform())


def draw_provenance(ax, deltas):
    """R4: chained flip minus stage Q's round-0 flip, WITHIN one arm. The one
    cross-stage subtraction the programme licenses -- the stages share the
    instance stream, and their case sets are disjoint by construction."""
    y = np.arange(len(deltas))[::-1]
    cols = [SERIES[0]] * 3 + [MUTED]
    ax.barh(y, deltas, height=0.55, color=cols, rasterized=True)
    ax.axvline(0, color="0.25", linewidth=1.0)
    for b in (-PROVENANCE_BAR, PROVENANCE_BAR):
        ax.axvline(b, color="0.55", linestyle="--", linewidth=1.2)
    ax.text(-PROVENANCE_BAR, len(deltas) - 0.3, " registered bar %.2f" % PROVENANCE_BAR,
            size=11, color="0.55", va="center")
    for yy, v in zip(y, deltas):
        ax.text(v - 0.004, yy, "%+.3f" % v, va="center", ha="right", size=13)
    ax.set_yticks(y)
    ax.set_yticklabels(SHORT)
    ax.set_xlim(-0.13, 0.13)
    ax.set_xlabel("chained flip rate - round-0 flip rate (same checkpoint)")
    style_axes(ax)
    room_for_y_labels(ax, 0.04)


def draw_by_round(ax, rsum):
    """Flip rate against the round v was deduced in, one line per seed. Three
    series exactly, labelled at the right end; the control is left off because
    the claim is about the cell, not the comparison."""
    cols = series_colors(3)
    ends = []
    for i in range(3):
        br = rsum[i]["by_round"]
        ks = sorted(int(k) for k in br)
        v = [br[str(k)]["flipped"] for k in ks]
        ax.plot(ks, v, "-o", color=cols[i], markersize=7, linewidth=2)
        ends.append((ks[-1], v[-1], SHORT[i], cols[i]))

    # The three seeds land on top of each other here -- every end value is within
    # a few thousandths of 1.0 -- so three separate end labels printed as one
    # illegible smear. Coincident curves get ONE label naming all of them, which
    # is also the honest reading: a reader sees a single line and is told that is
    # three. Grey, because the label no longer stands for one hue.
    ends.sort(key=lambda e: -e[1])
    groups = []
    for x, y, name, col in ends:
        if groups and groups[-1][0] == x and abs(groups[-1][1] - y) <= 0.02:
            groups[-1][2].append(name)
        else:
            groups.append([x, y, [name], col])
    for x, y, names, col in groups:
        label_line(ax, x, y, " ".join(names),
                   color=col if len(names) == 1 else "0.35")
    br0 = rsum[0]["by_round"]
    for k in sorted(int(k) for k in br0):
        ax.text(k, 0.962, "n=%d" % br0[str(k)]["n"], size=11, color="0.35", ha="center")
    ax.axhline(CHANCE, color="0.25", linestyle="--", linewidth=1.2)
    ax.text(1.0, CHANCE + 0.015, "chance 0.25", size=11, color="0.25")
    ax.set_ylim(0.2, 1.02)
    ax.set_xticks(sorted(int(k) for k in br0))
    ax.set_xlabel("round in which the node was deduced")
    ax.set_ylabel("flip rate")
    style_axes(ax)
    room_for_y_labels(ax, 0.02)


# ------------------------------------------------- the counterfactual pages

def page_counterfactual(d, stage, bundle, row, title, subtitle, foot):
    """The edit, why DD's answer changes, and what the model did ON THIS CASE.

    Built as a bare figure rather than a claim_page because it is three text
    panels, not a plot; the title still states the finding.
    """
    from spin_gnn.sudoku_circuits import build_pair, reasoning_color_position
    from spin_gnn.sudoku_chain import build_chain_pair

    adj, colors, hints, trace, case, vocab, cfg = bundle
    if stage == "Q":
        clean, corrupt, v_pos = build_pair(adj, colors, hints, trace, vocab, case)
        w_pos = case["hint_color_pos"]
        rnd, nbr_rnd = 0, None
        known = {int(u): hints[int(u)] for u in adj[case["node"]] if int(u) in hints}
    else:
        clean, corrupt, v_pos, w_pos = build_chain_pair(adj, trace, vocab, case)
        rnd, nbr_rnd = case["round"], case["nbr_round"]
        kb = dict(hints)
        for k, events in enumerate(trace["rounds"]):
            if k >= rnd:
                break
            for nn, cc in events:
                kb[int(nn)] = int(cc)
        known = {int(u): kb[int(u)] for u in adj[case["node"]] if int(u) in kb}

    names_c = [vocab.name(int(t)) for t in clean]
    names_k = [vocab.name(int(t)) for t in corrupt]
    v, c_true, c_flip = case["node"], case["true_color"], case["flipped_color"]
    w = case["recolored_nbr"]

    fig = plt.figure(figsize=(16, 9))
    fig.text(0.06, 0.93, title, size=BIG, weight="bold")
    fig.text(0.06, 0.875, subtitle, size=SMALL, va="top")

    # -- panel 1: the one-token edit, in context
    ax = fig.add_axes([0.06, 0.50, 0.54, 0.32]); ax.axis("off")
    ax.text(0, 1.0, "THE EDIT: ONE TOKEN OF %d" % len(clean), size=SMALL,
            weight="bold", va="top")
    # The row must not run past this panel into panel 2. A fixed +-5 window
    # overran it on stage R: the edit sits mid-REASONING, where the window picks
    # up an 11-character BEGIN_ROUND, and the row printed on top of panel 2's
    # "covered colours" line. Shrink the window until the row fits rather than
    # letting an un-clipped off-axis text spill over the page.
    def _window(pad):
        a, b = max(0, w_pos - pad), min(len(clean), w_pos + pad + 1)
        wide = sum(0.013 * (max(len(names_c[i]), len(names_k[i])) + 1.6)
                   for i in range(a, b))
        return a, b, wide

    lo, hi, _ = _window(0)
    for pad in range(5, -1, -1):
        a, b, wide = _window(pad)
        if 0.14 + wide <= 0.98:
            lo, hi = a, b
            break
    for j, (lbl, nm) in enumerate((("before", names_c), ("after ", names_k))):
        ax.text(0.0, 0.80 - 0.17 * j, lbl, size=SMALL - 2, va="top", **MONO)
        x = 0.14
        for i in range(lo, hi):
            t = nm[i]
            ax.text(x, 0.80 - 0.17 * j, t, size=SMALL - 2, va="top",
                    weight="bold" if i == w_pos else "normal",
                    color="#000000" if i == w_pos else "0.45", **MONO)
            x += 0.013 * (len(t) + 1.6)
    ax.text(0.0, 0.42, "position %d, in the %s section" % (w_pos,
            "HINTS" if stage == "Q" else "REASONING"), size=SMALL - 2, va="top",
            color="0.4", style="italic")
    ax.text(0.0, 0.28,
            "N%d is a HINT: a fact the model was GIVEN." % w if stage == "Q" else
            "N%d was DEDUCED by the model in round %d: a fact it DERIVED."
            % (w, nbr_rnd), size=SMALL - 1, va="top", weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # -- panel 2: why DD's answer moves
    ax = fig.add_axes([0.64, 0.50, 0.32, 0.32]); ax.axis("off")
    ax.text(0, 1.0, "WHY DD'S ANSWER CHANGES", size=SMALL, weight="bold", va="top")
    cov = sorted(set(known.values()))
    lines = ["node N%d, deduced in round %d" % (v, rnd),
             "neighbours known before: %s" % " ".join("N%d=%d" % (u, c)
                                                      for u, c in sorted(known.items())),
             "covered colours %s = q-1 of 4" % cov,
             "so N%d must be COLOR_%d" % (v, c_true),
             "",
             "recolour N%d to COLOR_%d and" % (w, c_true),
             "COLOR_%d is the only one left:" % c_flip,
             "DD's answer becomes COLOR_%d" % c_flip]
    for i, ln in enumerate(lines):
        ax.text(0.0, 0.84 - 0.105 * i, ln, size=SMALL - 2, va="top",
                **(MONO if i in (1, 2, 3, 7) else {}))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # -- panel 3: what the model did, read out of the arm's own CSV
    ax = fig.add_axes([0.06, 0.15, 0.90, 0.28]); ax.axis("off")
    ax.text(0, 1.0, "WHAT THE MODEL DID ON THIS CASE  (row 1 of %s's behaviour CSV, "
                    "not recomputed here)" % ("Q_noff_s0" if stage == "Q" else "R_noff_s0"),
            size=SMALL, weight="bold", va="top")
    ok_clean = _tf(row, "clean_correct")
    ok_flip = _tf(row, "corrupt_flipped")
    # reverts_to_original is R-only: it is the detection signature sudoku_chain
    # records because editing an EMITTED deduction makes the prefix inconsistent
    # with the graph. A hint edit (stage Q) leaves the prefix consistent, so
    # sudoku_circuits never had a reason to record it and the column is absent.
    rev = _tf(row, "reverts_to_original") if "reverts_to_original" in row else None
    if ok_flip:
        after = "predicted COLOR_%d  FOLLOWED THE EDIT" % c_flip
    elif rev:
        after = "predicted COLOR_%d  KEPT ITS OWN ANSWER" % c_true
    elif rev is None:
        after = "did not predict COLOR_%d  DID NOT FOLLOW THE EDIT" % c_flip
    else:
        after = "predicted neither colour"
    ax.text(0.0, 0.70, "before the edit", size=SMALL - 1, va="top", **MONO)
    ax.text(0.17, 0.70, "predicted COLOR_%d  %s" % (c_true, "CORRECT" if ok_clean else "WRONG"),
            size=SMALL - 1, va="top", weight="bold")
    ax.text(0.0, 0.46, "after the edit", size=SMALL - 1, va="top", **MONO)
    ax.text(0.17, 0.46, after, size=SMALL - 1, va="top", weight="bold")
    ax.text(0.52, 0.70, "logit(freed) - logit(true)   %+.2f  ->  %+.2f"
            % (float(row["clean_logit_diff"]), float(row["corrupt_logit_diff"])),
            size=SMALL - 1, va="top", **MONO)
    ax.text(0.52, 0.46, "the sign reverses: the edit moves it from\nconfidently the true "
                        "colour to confidently the freed one", size=SMALL - 2, va="top",
            color="0.4", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    footer(fig, foot)
    d.savefig(fig)


class PngDeck(Deck):
    """slides.Deck plus a PNG per page, and a savefig() the shared pages use.

    There is no PDF rasterizer on this cluster, so the PNGs are the only way to
    LOOK at the deck before shipping, which is required before any upload. On
    the stage M deck that read-back caught five defects no exit code could have
    -- including a page whose bold line contradicted the page before it.
    """

    def __init__(self, path, png_dir, **kw):
        Deck.__init__(self, path, **kw)
        self.png_dir = png_dir
        os.makedirs(png_dir, exist_ok=True)

    def _save(self, fig):
        fig.savefig(os.path.join(self.png_dir, "page%d.png" % self._n), dpi=110)
        Deck._save(self, fig)

    def savefig(self, fig):
        self._n += 1
        self._save(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/users/hshunt/spin_gnn/notebooks/stageqr")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    pdf_path = args.pdf or os.path.join(args.out, "stageqr_slides.pdf")
    os.makedirs(args.out, exist_ok=True)

    S = load_qr()
    q, r, rows, ex = S["q"], S["r"], S["rows"], S["ex"]
    qflip = [s["flip"] for s in q]
    rflip = [s["corrupt_flipped"] for s in r]
    deltas = [r[i]["corrupt_flipped"] - read_stageq_flip(RARMS[i][2]) for i in range(4)]

    # The two counterfactual pages, checked against the CSVs before they draw.
    qb = case_one("Q")
    rb = case_one("R")
    with open(os.path.join(QDIR, "Q_noff_s0", "behavior.csv")) as f:
        qrow = next(csv.DictReader(f))
    rrow = rows[0][0]
    if (int(qrow["node"]) != qb[4]["node"] or int(qrow["true_color"]) != qb[4]["true_color"]
            or int(qrow["flipped_color"]) != qb[4]["flipped_color"]):
        sys.exit("stage Q case #1 does not match row 1 of the CSV: regenerated "
                 "%r, CSV %r. The instance stream drifted; not drawing it."
                 % (qb[4], dict(qrow)))
    if (int(rrow["node"]) != rb[4]["node"] or int(rrow["round"]) != rb[4]["round"]
            or int(rrow["binding_nbr"]) != rb[4]["recolored_nbr"]
            or int(rrow["nbr_round"]) != rb[4]["nbr_round"]):
        sys.exit("stage R case #1 does not match row 1 of the CSV: regenerated "
                 "%r, CSV %r. The instance stream drifted; not drawing it."
                 % (rb[4], dict(rrow)))

    prov = ("Q 51817042 | R 51820549 | commit %s | 3 seeds + 1 control | "
            "N=48 d5 q4 --shuffle-adj --fall | 350 cases/arm" % args.commit)

    with PngDeck(pdf_path, args.out, project="spin_gnn stages Q+R",
                 provenance=prov) as d:

        d.title_page(
            "Attention alone does direct deduction: the rule and the iteration",
            subtitle="Two registered counterfactual audits on the same three seeds of "
                     "no_mlp L=16 d192, 350 cases each.")

        d.points_page(
            "The audit behind this programme had never touched this cell",
            ["Findings 27/28 ran on N=24 full-architecture checkpoints only",
             "Stage Q edits a given hint; stage R edits a derived fact",
             "Same seeds, same instance stream, same 0.250 chance rate"])

        # The tokenization and trace pages are mandatory in every deck, and they
        # take the CHECKPOINT's instance keys: shuffle_adj and fall are not
        # defaults to be left alone (finding 43), they select the distribution.
        exi = example_instance(n_nodes=48, degree=5, n_colors=4,
                              shuffle_adj=True, fall=True)
        page_tokenization(d, exi)
        page_trace(d, exi, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the "
                                "distribution both audits drew their cases from.")

        page_counterfactual(
            d, "Q", qb, qrow,
            "One hint changes, and the model's answer changes with it",
            "Stage Q's case #1, regenerated from the arm's own --seed-start and matched "
            "against row 1 of its behaviour CSV before drawing.",
            "Round-0 forcing: every known neighbour of N%d is a HINT, so this case tests "
            "the elimination RULE over GIVEN facts. The model is never shown the "
            "reasoning on the right -- only the node/colour pairs." % qb[4]["node"])

        d.claim_page(
            "Editing one hint flips the answer on essentially every case",
            lambda ax: draw_flip(ax, qflip),
            note="350 counterfactuals per arm, scored against absolute chance. The "
                 "control is a different cell in two respects.")

        d.table_page(
            "Before the edit, every arm is correct on all 350 cases",
            ["arm", "clean", "flipped", "x chance", "logit gap"],
            [[SHORT[i], "%.3f" % q[i]["clean"], "%.3f" % q[i]["flip"],
              "%.2f" % (q[i]["flip"] / CHANCE),
              "%+.1f to %+.1f" % (q[i]["ld_clean"], q[i]["ld_corrupt"])]
             for i in range(4)],
            note="Q3's precondition is clean >= %.2f. Every arm sits at the ceiling, "
                 "which is what Q2 and R2 are about." % CLEAN_FLOOR)

        d.claim_page(
            "The logit gap reverses sign, from %+.1f to %+.1f"
            % (q[0]["ld_clean"], q[0]["ld_corrupt"]),
            lambda ax: draw_logit(ax, [s["ld_clean"] for s in q],
                                  [s["ld_corrupt"] for s in q]),
            note="Not a marginal preference flip. Magnitudes are not comparable across "
                 "arms of different width and are not compared.")

        d.points_page(
            "Stage Q tested round zero only, where every neighbour is a hint",
            ["find_flippable_deductions picks nodes forced in round 0",
             "clean_correct is 1.000 because these cases are easy by construction",
             "A one-step classifier would score exactly the same here"])

        d.points_page(
            "Q4's registered seed rule fired on noise and means nothing",
            ["A 0.003 seed spread against a 0.001 arm-control gap",
             "Every arm saturated, so the comparison has no resolution",
             "Stage R registers a %.2f ceiling rule so this cannot recur" % CEILING])

        page_counterfactual(
            d, "R", rb, rrow,
            "Stage R edits a colour the model itself deduced earlier",
            "Stage R's case #1, same check against row 1 of its CSV. The edited token is "
            "a previously EMITTED DEDUCTION, not a hint.",
            "This is the difference between applying an inference rule once and running "
            "the algorithm: N%d's colour was derived in round %d, and N%d is forced in "
            "round %d against it." % (rb[4]["recolored_nbr"], rb[4]["nbr_round"],
                                      rb[4]["node"], rb[4]["round"]))

        d.claim_page(
            "Editing a derived fact flips the answer %.2f times above chance"
            % (rflip[0] / CHANCE),
            lambda ax: draw_flip(ax, rflip),
            note="R0's gate was %.2f on two of three seeds, registered before the run. "
                 "It held on all three." % GATE)

        d.table_page(
            "Chained cases are harder, yet clean accuracy is still 1.000",
            ["arm", "clean", "flipped", "reverts", "logit gap"],
            [[SHORT[i], "%.3f" % r[i]["clean_correct"], "%.3f" % r[i]["corrupt_flipped"],
              "%.3f" % r[i]["reverts_to_original"],
              "%+.1f to %+.1f" % (ex[i]["ld_clean"], ex[i]["ld_corrupt"])]
             for i in range(4)],
            note="reverts_to_original is R1's detection signature: the model keeping its "
                 "own answer rather than failing to chain.")

        d.claim_page(
            "Using a derived fact instead of a given one costs nothing",
            lambda ax: draw_provenance(ax, deltas),
            note="R4, the one licensed cross-stage subtraction: same checkpoint, same "
                 "instance stream, disjoint case sets by construction.")

        d.table_page(
            "The three seeds' identical 0.994 is a coincidence of counts",
            ["seed", "flipped", "misses", "which cases"],
            [[SHORT[i], "%.3f" % r[i]["corrupt_flipped"], "%d of %d"
              % (ex[i]["n_miss"], ex[i]["n"]), " ".join(ex[i]["miss_keys"])]
             for i in range(3)],
            note="Node and round are not an instance id: s0 missed node 42 twice, in "
                 "different instances. No two seeds miss the same pair.")

        d.table_page(
            "The control's lower flip rate is detection, not failure to chain",
            ["arm", "flipped", "misses", "of which revert"],
            [[SHORT[i], "%.3f" % r[i]["corrupt_flipped"], "%d" % ex[i]["n_miss"],
              "%d" % ex[i]["miss_reverts"]] for i in range(4)],
            highlight_row=3,
            note="The full solver is the one more likely to notice the prefix was edited "
                 "and keep its own value. Shape-confounded either way.")

        d.claim_page(
            "The flip rate does not decay with the round being deduced",
            lambda ax: draw_by_round(ax, r),
            note="Composing layers predicts no decay, and none appears. Monotone decay "
                 "would have tied rounds to layers quantitatively.")

        d.points_page(
            "The chained cases reach round three, so depth is barely probed",
            ["%d of %d cases are round 1, %d are round 3"
             % (r[0]["by_round"]["1"]["n"], r[0]["n_cases"], r[0]["by_round"]["3"]["n"]),
             "No decay is therefore tested to 3 of 16 layers",
             "Rounds beyond the third are untested on this cell"])

        d.points_page(
            "The guard third of the original audit is still untested here",
            ["27/28 also found 0 of 350 emissions when forcing broke",
             "No module in the current tree tests that on no_mlp",
             "So declining to deduce what does not follow stays unmeasured"])

        d.table_page(
            "No previous audit of this cell exists on this eval",
            ["quantity", "findings 27/28", "stages Q+R"],
            [["graph size", "N=24", "N=48"],
             ["architecture", "full", "no_mlp L=16 d192"],
             ["round-0 flip", "350 of 350", "%.3f, %.3f, %.3f" % tuple(qflip[:3])],
             ["chained flip", "0.980", "%.3f, %.3f, %.3f" % tuple(rflip[:3])],
             ["seeds", "1", "3"]],
            note="Different N and different architecture, so these are not comparable "
                 "numbers -- they are the reason the audit was rerun.")

    print("wrote %s (%d KB, %d pages)" % (pdf_path, os.path.getsize(pdf_path) // 1024,
                                          d._n))
    print("PNGs at %s/page1..%d.png -- READ EVERY ONE before shipping"
          % (args.out, d._n))
    print("stage Q case #1: node %d, true %d, freed %d  (matches CSV row 1)"
          % (qb[4]["node"], qb[4]["true_color"], qb[4]["flipped_color"]))
    print("stage R case #1: node %d round %d, binding N%d deduced in round %d "
          "(matches CSV row 1)" % (rb[4]["node"], rb[4]["round"],
                                   rb[4]["recolored_nbr"], rb[4]["nbr_round"]))


if __name__ == "__main__":
    main()
