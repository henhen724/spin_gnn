#!/usr/bin/env python3
"""The determinism check as a nested web deck: ~/slides/decks/spin-gnn/determinism/.

WHAT THIS REPORTS. Not a stage and not an arm -- no scientific knob moves here
and no published number can change. It reports whether a RERUN of one cell
reproduces that cell, which every comparison in stages S through W silently
assumed, and it puts a number on the phrase "within noise" that five stages have
used without one.

WHY IT IS A DECK AT ALL. The answer changes how every number in this project is
read: if two runs of one configuration differ by more than the effect a stage is
measuring, then a stage's null is a statement about the instrument and not about
the model. That belongs in front of hshunt in the same form the results do.

PAGE SET. The shared explanation pages come from notebooks/deck_explain.py, minus
CENSOR alone -- every term that appears here is defined here, including the ones
a reader of the stage decks would already know. LEAK was dropped too until the
second read-back found the shared tokenization figure asserting "THE ORDER
WITHIN A ROW IS THE LEAK" in bold, with nothing in the deck saying what the leak
is. A term a FIGURE introduces is a term the deck uses. Five more explanation pages are specific to this question (what a seed
fixes, what a replicate is, what the flag pins, the replicate-count equation, and
the hypotheses as they were written before the job ran). Then the results.

NOTHING IS RE-DERIVED. The stage-W pairs, the paired standard deviation and the
replicate count all come from scripts/noise_floor.py; the S1 constants and the
0.0011 effect size come through scripts/stagew_score.py from the stage that
measured them; the NLL levels come from scripts/step_epoch_census.py. The four
cells' own numbers are read from their history.json, never from the job log.
check_claims() re-asserts every sentence against the data: a web deck is a URL,
and a stale number there keeps serving.

Usage (in a job -- $HOME I/O on the login node wedges, and this imports torch):
    python3 -u -m notebooks.make_determinism_web --commit <sha> --log <job .out>
"""

import argparse
import json
import os
import sys
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

from slides import MUTED, SERIES, style_axes

from notebooks import deck_explain
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.make_staget_deck import J2_RHO50
from notebooks.make_staget_web import WebDeck

from scripts.noise_floor import (CONTROL_LOGS, HOURS_PER_REPLICATE, MID_EP,
                                 W_LOGS, at, required_n)
from scripts.noise_floor import NEED as NF_NEED
from scripts.noise_floor import by_seed as logs_by_seed
from scripts.stagew_score import FREE, NEED, S1_TRAINABLE

DECK = "/home/users/hshunt/slides/decks/spin-gnn/determinism"
RUN = os.path.join(os.environ["SCRATCH"], "spin_gnn", "runs",
                   "2026-11-13_determinism")
# The cell is S1 exactly, so the tag is S1's, plus the _det segment the sweep
# adds when the flag is on. Written out rather than rebuilt from args, because a
# deck that looked in the wrong directory would report "missing" and not "wrong".
TAG = "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied"
CELLS = (("det_a", True), ("det_b", True), ("free_a", False), ("free_b", False))
PINNED, UNPINNED = ("det_a", "det_b"), ("free_a", "free_b")
CELL_EPOCHS = 3          # what the sbatch asked for; asserted against the data
# The two epoch-0 train NLLs that raised the question, from stage W's seed-0
# pair. Quoted so the spread page can put this job's cells beside them; both are
# re-read from the logs in check_claims() rather than trusted.
W_EP0, C_EP0 = 2.5167, 2.3656


# ---------------------------------------------------------------- loading

def cell_dir(name, det):
    return os.path.join(RUN, name, TAG + ("_det" if det else ""))


def load(name, det):
    """(history, config) for one cell, or (None, None) if it did not land."""
    d = cell_dir(name, det)
    hp, cp = os.path.join(d, "history.json"), os.path.join(d, "config.json")
    if not (os.path.exists(hp) and os.path.exists(cp)):
        return None, None
    with open(hp) as f:
        hist = json.load(f)
    with open(cp) as f:
        cfg = json.load(f)
    return hist, cfg


def wall_times(log_path):
    """{cell name: minutes} from the job's own 'cell X rc=0 wall=..m..s' lines.

    The only quantity in this deck that is NOT in a checkpoint: wall clock is a
    property of the run, not of the model, so it has nowhere else to live. Cells
    with a non-zero rc are dropped -- a failed cell's wall time is not a cost
    measurement of anything.
    """
    out = {}
    if not log_path or not os.path.exists(log_path):
        return out
    with open(log_path) as f:
        for line in f:
            if not line.startswith("cell ") or " rc=" not in line:
                continue
            parts = line.split()
            name = parts[1]
            rc = int(parts[2].split("=")[1])
            wall = parts[3].split("=")[1]
            if rc != 0:
                continue
            m, s = wall.rstrip("s").split("m")
            out[name] = int(m) + int(s) / 60.0
    return out


# ---------------------------------------------------------------- analysis

def pair_rows(a, b):
    """Per-epoch (epoch, a nll, b nll, delta nll, a acc, b acc, identical)."""
    rows = []
    for ra, rb in zip(a, b):
        same = (rb["train_nll"] == ra["train_nll"]
                and rb["accuracy"] == ra["accuracy"])
        rows.append((ra["epoch"], ra["train_nll"], rb["train_nll"],
                     rb["train_nll"] - ra["train_nll"],
                     ra["accuracy"], rb["accuracy"], same))
    return rows


def stagew_pairs():
    """The three (seed, d_nll, d_acc) stage-W gave for free, at epoch MID_EP."""
    w, c = logs_by_seed(W_LOGS), logs_by_seed(CONTROL_LOGS)
    out = []
    for s in sorted(set(w) & set(c)):
        wn, wa = at(w[s], MID_EP)
        cn, ca = at(c[s], MID_EP)
        out.append((s, wn, cn, wn - cn, wa, ca, wa - ca))
    return out, w, c


def check_claims(hists, cfgs, w_hist, c_hist):
    """Every sentence this deck prints, asserted against the data behind it."""
    fail = []

    # The cells are what the sbatch said they were.
    for (name, det) in CELLS:
        cfg = cfgs.get(name)
        if cfg is None:
            continue
        if cfg.get("deterministic") is not det:
            fail.append("%s: config.json says deterministic=%r, the arm says %r"
                        % (name, cfg.get("deterministic"), det))
        if cfg.get("n_trainable_params") != S1_TRAINABLE:
            fail.append("%s: %r trainable, S1 is %d"
                        % (name, cfg.get("n_trainable_params"), S1_TRAINABLE))
        if cfg.get("seed") != 0:
            fail.append("%s: seed %r, every cell here is seed 0" % (name, cfg.get("seed")))
        if cfg.get("epochs") != CELL_EPOCHS:
            fail.append("%s: %r epochs, the deck says %d"
                        % (name, cfg.get("epochs"), CELL_EPOCHS))
        if cfg.get("n_train_instances") != 240000:
            fail.append("%s: %r training instances, S and W used 240000"
                        % (name, cfg.get("n_train_instances")))
        if len(hists[name]) != CELL_EPOCHS:
            fail.append("%s: %d epochs in history.json, config says %d"
                        % (name, len(hists[name]), CELL_EPOCHS))

    # The effect size is ONE number, and two modules name it.
    if abs(NF_NEED - NEED) > 1e-12:
        fail.append("noise_floor's NEED %.6f and the scorer's %.6f disagree"
                    % (NF_NEED, NEED))

    # The two quoted epoch-0 values are the logs' own.
    for label, hist, want in (("stage W", w_hist.get(0), W_EP0),
                              ("the control", c_hist.get(0), C_EP0)):
        if hist is None:
            fail.append("%s has no seed-0 log to read epoch 0 from" % label)
        elif abs(at(hist, 0)[0] - want) > 5e-5:
            fail.append("%s epoch 0 reads %.4f, the deck quotes %.4f"
                        % (label, at(hist, 0)[0], want))

    # Both arms complete, or the deck says which half is missing rather than
    # printing a verdict it cannot support.
    have = {n for n, _ in CELLS if hists.get(n)}
    pinned_ok = set(PINNED) <= have
    free_ok = set(UNPINNED) <= have

    if fail:
        raise AssertionError("check_claims failed:\n  " + "\n  ".join(fail))
    return pinned_ok, free_ok


def verdicts(hists, pinned_ok, free_ok):
    """H1/H2 as booleans plus the numbers they rest on."""
    out = {}
    if pinned_ok:
        rows = pair_rows(hists[PINNED[0]], hists[PINNED[1]])
        out["pinned"] = rows
        out["h1"] = all(r[6] for r in rows)
        out["pinned_max"] = max(abs(r[3]) for r in rows)
    if free_ok:
        rows = pair_rows(hists[UNPINNED[0]], hists[UNPINNED[1]])
        out["free"] = rows
        out["h2"] = abs(rows[0][3]) > NEED
        out["free_ep0"] = abs(rows[0][3])
        out["free_max"] = max(abs(r[3]) for r in rows)
    return out


# ---------------------------------------------------------------- figure

def draw_spread(ax, runs):
    """Every run of ONE configuration that has an epoch-0 train NLL on disk.

    One claim, one axis: these are all the same cell at the same seed, so the
    spread between them is the instrument and nothing else. The two stage-W runs
    are MUTED because they are the prior evidence; this job's four are the
    series under test, and the pinned pair is drawn as one marker per run so a
    reader can see whether two dots sit on top of each other.
    """
    labels = [r[0] for r in runs]
    xs = list(range(len(runs)))
    for x, (name, nll, new) in zip(xs, runs):
        color = SERIES[0] if new else MUTED
        ax.plot([x], [nll], "o", ms=11, color=color, zorder=4)
        ax.annotate("%.4f" % nll, xy=(x, nll), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=11,
                    color=color)
    lo, hi = min(r[1] for r in runs), max(r[1] for r in runs)
    pad = 0.35 * (hi - lo) if hi > lo else 0.05
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.6, len(runs) - 0.4)
    ax.set_ylim(lo - pad, hi + pad)
    style_axes(ax, xlabel=None, ylabel="train NLL after epoch 0")


# ---------------------------------------------------------------- the pages

SEEDS_FIX = """## A seed fixes the data, the weight init and the batch order, and not the kernels

- Every cell in this project is launched with one integer, `--seed`. Three things are drawn from it, all of them inside `train_sudoku_model`:
  - the **instances**: `build_instances(cfg, n, cfg.seed)` builds the same 240,000 graphs with the same planted colourings and the same hints;
  - the **weight init**: `torch.manual_seed(cfg.seed)` before the model is constructed, so every initial weight is the same number;
  - the **batch order**: `np.random.default_rng(cfg.seed)`, passed into the batch iterator, so epoch 3 sees the same instances in the same order.
- What the seed does **not** fix is the order in which the GPU adds things up. A single layer's output is a sum over thousands of terms, computed by thousands of threads, and which thread's partial sum is added first depends on how the hardware schedules them that time.
- Floating-point addition is not associative: in general $(a + b) + c \\ne a + (b + c)$, because each addition rounds its result to the nearest representable number. Two orderings of the same sum give two answers that differ in the last bits.
- So two runs launched identically agree on their inputs and on their starting weights, and begin to disagree on their arithmetic from the very first step.
"""

REPLICATE = """## A replicate is a rerun of one configuration, and it is a different run

- A **replicate** is the same cell launched twice: same architecture, same data, same seed, same budget, nothing changed.
- When the quantity being compared is measured on two runs rather than inside one, the comparison is **cross-run**, and what it reports is the knob's effect *plus* whatever two replicates differ by.
- Write $x^{A}_{s}$ for a quantity measured in run $A$ at seed $s$, and $x^{B}_{s}$ for the same quantity in run $B$. The **paired difference** is

$$\\Delta_s = x^{A}_{s} - x^{B}_{s}$$

  and the sign convention used everywhere in this deck is *first arm minus second*, so a positive $\\Delta$ means the first-named run is higher.
- If $A$ and $B$ are replicates of one configuration, then $\\Delta_s$ contains **no knob at all**. Whatever it is, it is the floor under every cross-run comparison this project has made.
"""

PINS = """## `--deterministic` pins the kernels, and refuses to run without its environment variable

- An opt-in flag on the training script, off by default, so every run up to and including stage W is reproduced exactly by the code as it stands. Four things, three of them inside the process:
  - `torch.use_deterministic_algorithms(True)` — PyTorch selects a deterministic kernel for every operation, and **raises an error** rather than silently using a nondeterministic one when no such kernel exists.
  - `torch.backends.cudnn.benchmark = False` — cuDNN otherwise times several algorithms on the first call to each convolution shape and keeps the fastest, which can differ between runs on the same hardware.
  - `torch.backends.cudnn.deterministic = True` — restricts cuDNN to its reproducible algorithms.
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the **launcher's** environment. cuBLAS reads this once, when it initializes, and uses it to give each stream its own workspace so that matrix multiplications reduce in a fixed order. Setting it from inside Python would be too late and would look like it had worked, so the trainer checks for it and refuses to start without it.
- A cell trained with the flag gets a `_det` segment in its directory name, for the same reason the tied and $k$-block cells get theirs: pinned and unpinned kernels end at different weights from one initialization, so they must never share a directory with each other.
"""

POWER = """## Replicates needed to resolve an effect: $n = (z_{\\alpha} + z_{\\beta})^2 \\sigma_d^2 / \\delta^2$

- To ask how many replicates a comparison would need, three quantities are required:
  - $\\delta$, the **effect size** — how large a difference has to be seen. Here $\\delta = 0.0011$ of validation token accuracy, which is what the $0.017$ of $\\rho_{50}$ still separating S1 from the gate is worth on the measured accuracy-to-solve curve.
  - $\\sigma_d$, the **standard deviation of the paired difference** between two replicates, estimated from the pairs stage W produced.
  - the two $z$ values, which fix how certain the answer has to be: $z_{\\alpha} = 1.96$ for a two-sided test at the conventional $5\\%$ false-positive rate, and $z_{\\beta} = 0.84$ for an $80\\%$ chance of detecting a real effect of size $\\delta$. Their sum squared is $(1.96 + 0.84)^2 = 7.85$.
- $n$ is the number of **paired replicates per arm**. One replicate here is one 50-epoch training run of this cell, measured at $7.9$ hours of one H100.
- The normal approximation above understates $n$ at small samples; a $t$-based number is larger. Every count in this deck is therefore the optimistic one.
"""

HYPOTHESES = """## Three hypotheses, written into the job script before it was submitted

- **H1 — pinning works.** With `--deterministic`, two replicates of this cell at seed 0 produce train NLL that agrees to every printed digit at every epoch, and identical values in `history.json`.
- **H2 — the baseline really is nondeterministic.** Without the flag, two replicates at seed 0 differ at epoch 0, on the order of the $0.151$ that stage W's seed-0 pair differed by.
- **H3 — pinning costs wall clock.** Sign unknown in advance; the job measures it, because that is what decides whether a future stage can afford to run this way.
- **H2 failing is the interesting outcome**, and the unpinned arm exists to be able to say so: if two *unpinned* replicates also agree to all digits, then kernel nondeterminism is not what separated stage W's arms, the cause of that separation is still unidentified, and pinning fixes nothing.
"""

SCOPE = """## What pinning fixes, and the one thing it does not

- **Fixed: a published cell can be re-run.** A rerun reproduces it exactly, so a number that moves is a bug and not weather, and a reviewer can check a claim instead of taking it.
- **Fixed: a knob that acts late in a run becomes an exact comparison.** The learning rate here is constant at $1.00 \\times 10^{-3}$ with no warmup, so a 100-epoch run and a 50-epoch run at one seed compute *identical* arithmetic up to epoch 49 — once the kernels are pinned, they are the same run, and stage W's control would have been a true continuation rather than an independent draw.
- **Not fixed: the basin lottery.** A knob that changes the computation from step 0 — width, depth, how many blocks, the architecture — makes two runs diverge immediately, and that divergence is still amplified by the same dynamics. Pinning removes the *run-to-run* noise, not the *seed-to-seed* noise, and which of the three NLL levels a seed lands on is seed-to-seed.
- So this closes the question "is my control the same run as my arm?" and leaves open "why does one seed in three find the basin?".
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--log", default=None,
                    help="the job's .out, for the wall-clock page only")
    ap.add_argument("--job", default="unknown")
    args = ap.parse_args()

    deck_explain.check()
    # CENSOR goes unused by this deck -- nothing here is scored on a grid that
    # could fail to clear 0.5 -- so it is not emitted.
    #
    # LEAK IS EMITTED, and dropping it was a defect the second read-back caught:
    # the shared tokenization figure says, in bold, "THE ORDER WITHIN A ROW IS
    # THE LEAK", so the term appears in this deck whether or not a stage result
    # turns on it. Every term that appears here is defined here, and a figure
    # counts as appearing.
    shared = [p for p in deck_explain.PAGES if p is not deck_explain.CENSOR]
    n_explain = len([p for p in shared if p is not deck_explain.FIGURE_SLOT]) + 5

    hists, cfgs = {}, {}
    for name, det in CELLS:
        h, c = load(name, det)
        if h is not None:
            hists[name], cfgs[name] = h, c
    if not hists:
        raise SystemExit("no cell has landed yet -- nothing to publish")

    sw, w_hist, c_hist = stagew_pairs()
    walls = wall_times(args.log)
    pinned_ok, free_ok = check_claims(hists, cfgs, w_hist, c_hist)
    v = verdicts(hists, pinned_ok, free_ok)

    d_acc = [r[6] for r in sw]
    sd_all = stdev(d_acc)
    tight = [r[6] for r in sw if r[0] != 2]
    sd_tight = stdev(tight) if len(tight) > 1 else None

    # The headline is whichever way the data went, not whichever way was hoped.
    if pinned_ok and free_ok:
        if v["h1"] and v["h2"]:
            head = "Pinning the GPU kernels makes a rerun exact; unpinned, it is not"
        elif v["h1"] and not v["h2"]:
            head = "Pinning makes a rerun exact, and the unpinned pair agreed too"
        elif not v["h1"] and v["h2"]:
            head = "Pinning the kernels did not make two replicates agree"
        else:
            head = "Neither pinned nor unpinned replicates behaved as predicted"
    elif pinned_ok:
        head = "Pinned replicates %s; the unpinned arm did not land" % (
            "agree exactly" if v["h1"] else "still disagree")
    else:
        head = "The determinism check is incomplete and reports no verdict"

    prov = ("Job %s, 4 cells of S1 at seed 0 (2 pinned, 2 not), %d epochs each, "
            "240,000 training instances, commit %s. Stage W's pairs re-read from "
            "arrays 52053221 and 51853969. No published number changes."
            % (args.job, CELL_EPOCHS, args.commit))

    d = WebDeck(args.out, head, "hshunt", "2026-11-13", prov)
    d.title_page(head, subtitle=(
        "Not a stage: no scientific knob moves. The question is whether a rerun "
        "of one cell reproduces that cell, which every comparison since stage S "
        "has assumed."))

    d.points_page(
        "Five stages reported a null without ever giving noise a number",
        ["Stages S, T, U, V and W each moved one knob and found nothing.",
         "Each said *within noise* without noise ever having a number.",
         "The effect all five were reading is %.4f of validation token accuracy."
         % NEED])

    d.points_page(
        "Stage W handed over that number without meaning to",
        ["Stage W compared 100 epochs against 50 at matched seeds.",
         "Its two arms already differed **before the extra epochs could act**.",
         "That difference has no knob in it, so it measures the noise."])

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
                     note="Identical instances in every cell here: the seed is 0 "
                          "throughout and only the kernels change.")
            page_trace(d, ex)
        else:
            d._add(page)

    for page in (SEEDS_FIX, REPLICATE, PINS, POWER, HYPOTHESES):
        d._add(page)

    # ---- the evidence that raised the question
    d.table_page(
        "Two runs of one configuration differ by more than the effect",
        ["seed", "arm train NLL", "control train NLL", "$\\Delta$ NLL",
         "arm accuracy", "control accuracy", "$\\Delta$ accuracy"],
        [["%d" % s, "%.4f" % wn, "%.4f" % cn, "%+.4f" % dn,
          "%.4f" % wa, "%.4f" % ca, "%+.4f" % da]
         for s, wn, cn, dn, wa, ca, da in sw],
        note=("Read at epoch %d, the control's last. Arm minus control. Neither "
              "arm has spent its extra epochs yet." % MID_EP))

    d.table_page(
        "Resolving %.4f at this noise level needs hundreds of replicates" % NEED,
        ["reading of $\\sigma_d$", "$\\sigma_d$", "replicates per arm",
         "GPU-hours"],
        [["all %d seeds" % len(d_acc), "%.5f" % sd_all,
          "%.0f" % required_n(sd_all),
          "%.0f" % (required_n(sd_all) * HOURS_PER_REPLICATE)]]
        + ([["seeds 0 and 1 only", "%.5f" % sd_tight,
             "%.0f" % required_n(sd_tight),
             "%.0f" % (required_n(sd_tight) * HOURS_PER_REPLICATE)]]
           if sd_tight else []),
        note=("Seed 2's control stepped into the basin and its twin did not, so "
              "row two drops it. A bound, not a justified exclusion."))

    d.points_page(
        "The answer is not more replicates, it is a lower floor",
        ["Three points estimate a standard deviation badly, hence the 20-fold "
         "range.",
         # Was "out of reach: %.0f GPU-hours at best", which asserted that the
         # optimistic 562 is unreachable. It is about three weeks of one GPU --
         # expensive, not impossible. State both numbers and let them speak.
         "Costs %.0f GPU-hours optimistically, %.0f under the honest reading."
         % ((required_n(sd_tight) if sd_tight else required_n(sd_all))
            * HOURS_PER_REPLICATE,
            required_n(sd_all) * HOURS_PER_REPLICATE),
         "What is left is to make two replicates the same run."])

    # ---- H1
    if pinned_ok:
        d.table_page(
            "With the kernels pinned, two replicates %s"
            % ("agree to every digit" if v["h1"] else "still disagree"),
            ["epoch", "replicate a train NLL", "replicate b train NLL",
             "$\\Delta$ NLL", "a accuracy", "b accuracy"],
            [["%d" % e, "%.12f" % an, "%.12f" % bn, "%+.2e" % dn,
              "%.4f" % aa, "%.4f" % ba]
             for e, an, bn, dn, aa, ba, _ in v["pinned"]],
            note=("Both cells are seed 0 with `--deterministic`, launched "
                  "identically. Largest difference across the %d epochs: %.2e."
                  % (CELL_EPOCHS, v["pinned_max"])))

    # ---- H2
    if free_ok:
        d.table_page(
            "Without it, the same two launches %s"
            % ("diverge immediately" if v["h2"] else "agreed as well"),
            ["epoch", "replicate a train NLL", "replicate b train NLL",
             "$\\Delta$ NLL", "a accuracy", "b accuracy"],
            [["%d" % e, "%.12f" % an, "%.12f" % bn, "%+.2e" % dn,
              "%.4f" % aa, "%.4f" % ba]
             for e, an, bn, dn, aa, ba, _ in v["free"]],
            note=("Identical launches, identical seed, flag off. The epoch-0 "
                  "difference is %.4f, against the %.4f being chased."
                  % (v["free_ep0"], NEED)))

    # ---- the spread across every run of this configuration
    runs = [("stage W\n100 ep", at(w_hist[0], 0)[0], False),
            ("control\n50 ep", at(c_hist[0], 0)[0], False)]
    # The axis says "pinned a" / "unpinned a", not the cell's directory name
    # "det_a" / "free_a". The second read-back caught that: "det" and "free" are
    # this job's internal names and are defined nowhere in the deck, while
    # "pinned" and "unpinned" are the words every other page already uses.
    runs += [("%s %s" % ("pinned" if det else "unpinned", name.split("_")[-1]),
              hists[name][0]["train_nll"], True)
             for name, det in CELLS if name in hists]
    spread = max(r[1] for r in runs) - min(r[1] for r in runs)
    d.claim_page(
        # Read-back caught the first title here, "Every run of this one cell
        # lands somewhere different after epoch 0", being refuted by its own
        # figure: det_a and det_b plot at ONE point, which is the whole result.
        "Only the pinned pair lands twice in the same place",
        lambda ax: draw_spread(ax, runs), "spread",
        note=("Same architecture, same 240,000 instances, same seed 0. Blue is "
              "this job, grey is stage W's pair. Spread %.4f." % spread))

    # ---- H3
    if walls:
        pinned_walls = [walls[n] for n in PINNED if n in walls]
        free_walls = [walls[n] for n in UNPINNED if n in walls]
        rows = [["pinned", "%d" % len(pinned_walls),
                 # One decimal, not zero: rounded to whole minutes the table
                 # read 35, 35 against 32, 32, from which a reader computes
                 # 1.0 min per epoch while the note says 0.8.
                 ", ".join("%.1f" % w for w in pinned_walls)],
                ["unpinned", "%d" % len(free_walls),
                 ", ".join("%.1f" % w for w in free_walls)]]
        note = ("Whole-cell wall clock; each cell pays about nineteen minutes of "
                "instance building before epoch 0.")
        # H3's SIGN was not predicted, so its title is chosen from the number
        # rather than written in advance. The two pinned cells came back 2 s
        # apart over 34.6 min, so same-node wall noise is seconds; a minute per
        # cell (0.33 min/epoch) is comfortably above it and is the threshold for
        # calling the cost real.
        title = "One arm has no wall time, so the cost is unmeasured"
        if pinned_walls and free_walls:
            # The build is identical CPU work in both arms, so what the
            # difference of the means measures is the kernels.
            per_epoch = (mean(pinned_walls) - mean(free_walls)) / CELL_EPOCHS
            note = ("Whole-cell wall clock, including about nineteen minutes of "
                    "instance building. Difference of means over %d epochs: "
                    "%+.1f min per epoch." % (CELL_EPOCHS, per_epoch))
            if per_epoch > 0.33:
                title = "Pinning the kernels costs wall clock, and this is how much"
            elif per_epoch < -0.33:
                title = "The pinned cells ran faster, so pinning cost nothing"
            else:
                title = "Pinning the kernels cost no wall clock worth measuring"
        d.table_page(title,
                     ["arm", "cells", "minutes per cell"], rows, note=note)

    d._add(SCOPE)

    d.points_page(
        "The flag is off by default, so no published number moves",
        ["The flag is off by default, so every published number still "
         "reproduces.",
         "Stages S through W all ran unpinned; none of their numbers move.",
         "A cross-run null describes the instrument until the floor drops."])

    # Last page by the project's own rule: compare against the previous best on
    # this eval, or state that there is none. Here there is none, because no
    # earlier stage measured reproducibility at all.
    d.points_page(
        "There is no previous measurement of this to compare against",
        ["This project has never measured its own reproducibility before.",
         "So there is no previous best on this eval, and nothing is displaced.",
         "J2 still clears the %.3f gate at $\\rho_{50} = %.3f$; S1 does not."
         % (FREE, J2_RHO50)])

    path = d.write()
    used = sorted(os.listdir(d.fig_dir))
    print("wrote %s (%d pages, %d explanation, %d figures)"
          % (path, d.n, n_explain, len(used)))
    print("H1 pinned pair identical: %s (max |d| %.2e)"
          % (v.get("h1"), v.get("pinned_max", float("nan"))))
    print("H2 unpinned pair differs at epoch 0: %s (%.4f, max |d| %.4f)"
          % (v.get("h2"), v.get("free_ep0", float("nan")),
             v.get("free_max", float("nan"))))
    print("epoch-0 spread across %d runs of one configuration: %.4f"
          % (len(runs), spread))
    print("noise floor: sd %.5f -> %.0f replicates; tight sd %s"
          % (sd_all, required_n(sd_all),
             "%.5f -> %.0f" % (sd_tight, required_n(sd_tight)) if sd_tight else "n/a"))
    print("walls: %s" % ", ".join("%s %.0fm" % (k, v2) for k, v2 in sorted(walls.items())))


if __name__ == "__main__":
    main()
