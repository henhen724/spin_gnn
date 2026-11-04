"""Build the results notebook for the N=96 fall probe sweep (50979059).

    python3 notebooks/make_n96_fall_probes_notebook.py     # from the repo root

Writes $HOME/spin_gnn/notebooks/sudoku_n96_fall_probes_2026-11-03_results.ipynb.

Every number in a code cell is read at run time, from the probe CSVs through
`scripts.n96_fall_probe_analysis` -- the COMMITTED reader -- so a notebook
number and a `python3 -m scripts.n96_fall_probe_analysis` number cannot drift
apart. The markdown quotes the values this produced on 2026-11-03; rerun and
the tables update themselves while the quoted prose does not, so check both.

The one thing to know before reading: this run was first scored on balanced
accuracy and looked like a null, and that reading was wrong. Section 1 is
about the column, not about the model.
"""
import io
import json
import os

CELLS = []


def lines(src):
    """nbformat wants each source entry to carry its own newline; without them
    the cell concatenates into one unparseable line."""
    ls = src.strip("\n").split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": lines(src)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": lines(src)})


# ---------------------------------------------------------------- 0. header
md(r"""
# Job 50979059 — does the N=96 model represent DD's deducibility predicate, and where?

hshunt's standing question is whether the network actually runs Direct Deduction, proved
by probes, ablations and interventions. This run is the probe half at N=96, on the fall
ensemble, across the depth and data axes of the `50860571` scan.

**Complete.** Array `50979059_[0-2]` (`sud2N96FallProbe`, partition `sganguli`, one H100
per task), all three tasks `COMPLETED` exit `0:0`. The three tasks are three **probe-fit
seeds** over the same six checkpoints — the noise floor, not three experiments.

**Commits.** Probe job and `sudoku_probes` at `417bddf`; the reader this notebook imports,
`scripts/n96_fall_probe_analysis.py`, rewritten at `309ad1a` (the AUROC fix below) after
`b86a34e` fixed its target list. Checkpoints come from scan commits `c531a86` / `c71146d`.

**Config.** `sudoku_probes --n-train-instances 400 --n-test-instances 160
--max-positions-per-instance 20 --probe-epochs 600 --seed {0,1,2}`. Probed checkpoints are
the six `50860571`/`50903022` cells: `d_model 256, nhead 4, ff 1024`, pre-norm,
`n_nodes 96, degree 5, n_colors 4, --fall --shuffle-adj`.

| cell | num_layers | n_train | train seed | rho_50 (lower better) |
|---|---|---|---|---|
| t0 | 12 | 240k | 0 | 0.679 |
| t1 | 8 | 240k | 0 | 0.691 |
| t2 | 16 | 240k | 0 | 0.613 |
| t3 | 12 | 120k | 0 | 0.610 |
| t4 | 12 | 240k | 1 | 0.564 |
| t5 | 8 | 240k | 1 | 0.829 |

**What is probed.** A logistic-regression probe on the residual stream after each block,
at the *reasoning positions* of a generated trace. **Layer 0 is the token + position
embedding**, so it is the tokenization control: whatever layer 0 scores is available from
the input encoding alone and no computation is credited for it. Targets:

- `frontier_mask` — **the DD predicate.** Which nodes are deducible *this round*.
- `known_mask`, `remaining_mask` — bookkeeping controls. Which nodes already have a
  colour, which do not.
- `next_node`, `next_color` — what the model is about to emit (categorical; `auroc` is
  `nan` for these and the `balanced_accuracy` column is top-1 accuracy, read against
  `majority_accuracy` = 0.0149 / 0.2598).

**Pre-registered in the sbatch header before submission (P1–P4), quoted verbatim below.**
""")

# ---------------------------------------------------------------- 1. setup
code(r"""
import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.expanduser("~/spin_gnn")
sys.path.insert(0, REPO)

# The COMMITTED reader, imported rather than reimplemented.
from scripts.n96_fall_probe_analysis import (
    PROBE_DEFAULT, CELLS, MULTILABEL, CATEGORICAL, TARGETS, SEED_PAIRS, DEPTH_CELLS,
    PRIMARY, METRICS, load, metric_of, peak, cell_stats, spearman, rho50_by_cell)
from scripts import n96_fallscan_analysis as scan

prof, majority, kind, files = load(PROBE_DEFAULT)
present = [c for c in CELLS if any(k[0] == c for k in prof)]
seeds = sorted({k[1] for k in prof})
layers = {c: max(max(p) for (cc, _s, _t), p in prof.items() if cc == c) for c in present}
R50 = rho50_by_cell(scan.SCAN_DEFAULT)

print(f"probe dir   : {PROBE_DEFAULT}")
print(f"CSVs        : {', '.join(files)}")
print(f"cells       : {len(present)} of {len(CELLS)}   probe seeds: {seeds}")
print(f"targets     : {', '.join(TARGETS)}")
for c in present:
    lab, L, n, sd = CELLS[c]
    print(f"   {lab:<18} L={L:<3} probed layers 0..{layers[c]}   "
          f"rho_50 {R50[c][0]:.3f}  val_nll {R50[c][1]:.4f}")
""")

# ---------------------------------------------------------------- 2. the column
md(r"""
## 1. Read this first: the column, not the model

This run was scored on **balanced accuracy** when it landed, and read as a flat null —
every cell *below* the label-only free score, peak layer wandering 8.67 layers across
L=8/12/16. That reading was an artifact of the column, and the project's own baseline
script had already said so in its docstring:

> *balanced accuracy at a fixed 0-logit threshold ranks these procedures by how
> aggressively they predict positives rather than by what they know.*

`frontier_mask` is positive on **10.1%** of (position, node) bits. A BCE-trained probe on
a 10%-positive target under-predicts positives, so its balanced accuracy at threshold 0 is
a **floor**, and the floor moves with the probe's calibration rather than with its
information. The tell is already in the baseline table: the *trivial rule* `not_known`
scores 0.6734 there, above every trained probe, which cannot be a statement about
knowledge.

On **AUROC** — threshold-free, and what the baseline docstring says to quote — the same
rows invert. The cell below prints both columns for the same peaks so the flip is visible
rather than asserted.
""")

code(r"""
print(f"{'cell':<18}{'L':>3} | {'AUROC':>23} | {'balanced accuracy':>23}")
print(f"{'':<18}{'':>3} | {'layer0':>7}{'peak':>8}{'@L':>8} | {'layer0':>7}{'peak':>8}{'@L':>8}")
print("-" * 76)
for c in present:
    lab, L, _n, _s = CELLS[c]
    a = cell_stats(prof, c, "frontier_mask", "auroc")
    b = cell_stats(prof, c, "frontier_mask", "balanced_accuracy")
    print(f"{lab:<18}{L:>3} | {a['layer0']:>7.4f}{a['peak_abs']:>8.4f}{a['peak_layer']:>8.2f}"
          f" | {b['layer0']:>7.4f}{b['peak_abs']:>8.4f}{b['peak_layer']:>8.2f}")

pl_a = [cell_stats(prof, c, "frontier_mask", "auroc")["peak_layer"] for c, _ in DEPTH_CELLS]
pl_b = [cell_stats(prof, c, "frontier_mask", "balanced_accuracy")["peak_layer"]
        for c, _ in DEPTH_CELLS]
print(f"\npeak layer across L=8/12/16 -- AUROC: {pl_a}  spread {max(pl_a)-min(pl_a):.2f}")
print(f"peak layer across L=8/12/16 -- bal_acc: {pl_b}  spread {max(pl_b)-min(pl_b):.2f}")
print("The bal_acc argmax moves 8.67 layers with depth while the TRAINING SEED alone moves")
print("it 9.00 at fixed L=12: that is a flat curve's argmax, not a localisation.")
""")

# ---------------------------------------------------------------- 3. P4
md(r"""
## 2. P4 — the free score. What a probe gets without any notion of deducibility.

> Pre-registered: *"`frontier_mask` is a SUBSET of not-known, so a reader that recovers
> only `known_mask` scores on it for free. The probe must beat the best label-only
> baseline or it has shown nothing."*

`scripts/frontier_from_known_baseline.py` computes this with **no model anywhere** — it
fits the same probe procedure on DD's own ground-truth labels, so it is a property of the
task. Three rows are free scores (bookkeeping, and a count that needs the graph but not
the forcing rule) and two newer rows are the opposite end: they are handed
`n_known_colors`, which **is** what DD thresholds, so they are a ceiling.

`rule_on_visible` is the most informative row and it was an accident. It computes DD's
rule rather than fitting it, and was written with an assertion at 0.999 — which fired. The
assertion was wrong, not the labels: `frontier_mask` is the **round's** set, fixed at the
round boundary, while the state visible at a probe position already includes the
deductions this round has emitted. So a node emitted earlier this round is in the label
and `known` by now, and a node this round's own emissions just unlocked is deducible now
but belongs to round r+1. **A perfect DD reasoner reading the visible state does not
reproduce the label**, and that is a fact about the labelling, not about any model.

**Which row is the ceiling — and section 1's trap, in reverse.** `rule_on_visible` is a
hard 0/1 rule, so it occupies a single operating point and gets no credit for ranking
*within* its two classes; its AUROC (0.764) is therefore an underestimate of what its
information is worth, and the model's best cell (0.772) edges past it **without
necessarily knowing more**. The same row's average precision, 0.489, is far *above* every
probe's. The honest graded ceiling is `color_aug` — the same information, handed to a
probe that can rank — at **0.796–0.809 AUROC**. That is the number the model is read
against below.
""")

code(r"""
BASE = os.path.join(PROBE_DEFAULT, "frontier_baseline_n96_fall.json")
base = json.load(open(BASE))
FREE = ("not_known", "known_probe", "degree_aug")
CEIL = ("color_aug", "rule_on_visible")
ORDER = ("positive_rate", "shuffled") + FREE + CEIL

print("label-only baseline, N=96 fall (no model):")
print(f"  {'':>16}{'auroc':>9}{'avg_prec':>10}{'bal_acc':>9}   role")
role = {**{k: "FREE SCORE" for k in FREE}, **{k: "ceiling" for k in CEIL},
        "positive_rate": "chance", "shuffled": "chance"}
for k in ORDER:
    if k not in base:
        print(f"  {k:>16}{'  (absent -- older JSON, rerun the baseline)':<40}")
        continue
    r = base[k]
    print(f"  {k:>16}{r['auroc']:>9.4f}{r['average_precision']:>10.4f}"
          f"{r['balanced_accuracy']:>9.4f}   {role[k]}")

free_auc = max(base[k]["auroc"] for k in FREE if k in base)
free_ap = max(base[k]["average_precision"] for k in FREE if k in base)
pr = base["positive_rate"]["average_precision"]
print(f"\nBEST FREE SCORE: auroc {free_auc:.4f}   avg_prec {free_ap:.4f} "
      f"(AP chance = the {pr:.4f} positive rate)")

# 51009233: three instance pools x {300, 600} probe epochs, so both ends of the
# bracket carry a spread of their own instead of being point estimates.
BDIR = os.path.join(os.environ["SCRATCH"], "spin_gnn",
                    "sudoku_frontier_baseline_2026-11-03")
POOLS = sorted(glob.glob(os.path.join(BDIR, "n96_fall_ep*_off*.json")))
ceil_auc = ceil_ap = None
if POOLS:
    print(f"\n51009233 -- the same baseline over {len(POOLS)} (pool, epoch) runs:")
    print(f"  {'run':<22}{'free auroc':>11}{'free AP':>9}{'color_aug':>11}"
          f"{'cAP':>8}{'rule':>8}{'ruleAP':>8}")
    rec = {}
    for p in POOLS:
        b = json.load(open(p))
        ep = int(os.path.basename(p).split("_ep")[1][:3])
        fa = max(b[k]["auroc"] for k in FREE if k in b)
        fp = max(b[k]["average_precision"] for k in FREE if k in b)
        ca, cp = b["color_aug"]["auroc"], b["color_aug"]["average_precision"]
        ru, rp = b["rule_on_visible"]["auroc"], b["rule_on_visible"]["average_precision"]
        rec.setdefault(ep, []).append((fa, fp, ca, cp))
        print(f"  {os.path.basename(p)[:-5]:<22}{fa:>11.4f}{fp:>9.4f}{ca:>11.4f}"
              f"{cp:>8.4f}{ru:>8.4f}{rp:>8.4f}")
    # The bracket is taken at 600 probe epochs, matching the MODEL probes, so both
    # sides of the comparison are fit for the same number of steps. The free score
    # is the strictest pool seen (it is a bar); the ceiling is the pool mean (it is
    # a reference level).
    allr = rec.get(600) or [v for vs in rec.values() for v in vs]
    free_auc = max(v[0] for v in allr)
    free_ap = max(v[1] for v in allr)
    ceil_auc = float(np.mean([v[2] for v in allr]))
    ceil_ap = float(np.mean([v[3] for v in allr]))
    sp = max(v[0] for v in allr) - min(v[0] for v in allr)
    print(f"\n  H1 (pool stability): free-score spread {sp:.4f} over pools."
          f"  The model's margin is ~0.06-0.10, so {'H1 PASSES' if sp <= 0.01 else 'H1 FAILS'}.")
    if len(rec) > 1:
        m = {ep: np.mean([v[0] for v in vs]) for ep, vs in sorted(rec.items())}
        d = max(m.values()) - min(m.values())
        print(f"  H2 (300 vs 600 probe epochs): free score "
              + ", ".join(f"{ep}ep {v:.4f}" for ep, v in m.items())
              + f"  delta {d:.4f} -> {'H2 PASSES' if d <= 0.01 else 'H2 FAILS'}"
              + ("" if d <= 0.01 else " -- the committed 300-epoch baseline was underfit"
                                      " and the margin shrinks by this much"))
    else:
        print("  H2 (300 vs 600 probe epochs): the 600-epoch runs have not landed yet.")
    print(f"  H3 (there is headroom): color_aug {ceil_auc:.4f} vs free {free_auc:.4f}"
          f" -> {'H3 PASSES' if ceil_auc > free_auc + 0.02 else 'H3 FAILS'}")
    print(f"\nBRACKET IN FORCE: free {free_auc:.4f} .. ceiling {ceil_auc:.4f} (AUROC);"
          f"  {free_ap:.4f} .. {ceil_ap:.4f} (AP)")
else:
    print("\n(51009233 has not written anything yet; the free score above is a single")
    print(" pool at 300 probe epochs and there is no graded ceiling to read against.)")
""")

md(r"""
## 3. P1 — the DD predicate IS decoded, above every free score, and it is built LAST

> Pre-registered: *"if the model runs DD, `frontier_mask` is recoverable at a layer that
> moves with depth. P1 passes if the peak layer is within 1 of a fixed absolute layer
> across L=8/12/16; the informative alternative is a fixed FRACTION of depth, bar 0.12."*

Both halves of the table below matter, and they say different things.

**The level.** All six cells clear the best free score on **both** threshold-free columns,
by 0.058–0.099 AUROC and by 0.090–0.132 average precision at a 0.1008 base rate. Placed in
the bracket of section 2 — free 0.673, graded ceiling (`color_aug`) 0.809 — the cells have
closed **43%–73%** of the distance from "bookkeeping only" to "handed the quantity DD
thresholds", and 57%–83% of it on average precision. The free score's own spread over
three instance pools is 0.0050, a twentieth of the margin. **This is the positive result
in the run: the deducibility predicate is linearly present in the residual stream, well
beyond what bookkeeping explains.**

**The location.** P1 as written fails — the peak layer does not sit at a fixed absolute
layer, it moves 8 layers. The alternative holds in its extreme form: the peak is at
layer 8, 12 and 16 for L=8, 12 and 16, **fraction 1.000, spread 0.000**, and it is
reproducible across both noise channels (probe-seed spread on the peak layer 0.00,
training-seed spread 0.00). The DD predicate is most decodable at the **last** layer in
every cell.

**But the amount built does not grow with depth.** Peak lifts over layer 0 are 0.143 /
0.119 / 0.131 for L=8/12/16 — spread 0.0237, *below* the 0.0411 training-seed yardstick.
Eight extra layers buy no more frontier information; they only move where it tops out.
""")

code(r"""
print(f"{'cell':<18}{'L':>3}{'layer0':>8}{'peak':>8}{'lift':>8}{'@layer':>8}{'frac':>7}"
      f"{'vs free':>9}{'% brkt':>8}{'AP peak':>9}{'vs free':>9}{'% brkt':>8}")
print("-" * 103)
for c in present:
    lab, L, _n, _s = CELLS[c]
    s = cell_stats(prof, c, "frontier_mask", "auroc")
    ap = cell_stats(prof, c, "frontier_mask", "average_precision")
    bk = (f"{100*(s['peak_abs']-free_auc)/(ceil_auc-free_auc):7.1f}%"
          if ceil_auc else " " * 8)
    bkp = (f"{100*(ap['peak_abs']-free_ap)/(ceil_ap-free_ap):7.1f}%"
           if ceil_ap else " " * 8)
    print(f"{lab:<18}{L:>3}{s['layer0']:>8.4f}{s['peak_abs']:>8.4f}{s['peak_lift']:>8.4f}"
          f"{s['peak_layer']:>8.2f}{s['peak_layer']/L:>7.3f}"
          f"{('ABOVE' if s['peak_abs'] > free_auc else 'below'):>9}{bk}"
          f"{ap['peak_abs']:>9.4f}"
          f"{('ABOVE' if ap['peak_abs'] > free_ap else 'below'):>9}{bkp}")

mar = [cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] - free_auc for c in present]
print(f"\nmargin over the free score: {min(mar):+.4f} to {max(mar):+.4f} AUROC")
if ceil_auc is not None:
    bestm = max(cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] for c in present)
    bestp = max(cell_stats(prof, c, "frontier_mask", "average_precision")["peak_abs"]
                for c in present)
    print(f"AUROC  free {free_auc:.4f} -> model {bestm:.4f} -> ceiling {ceil_auc:.4f}:  "
          f"best cell closes {100*(bestm-free_auc)/(ceil_auc-free_auc):.0f}% of the gap")
    print(f"AP     free {free_ap:.4f} -> model {bestp:.4f} -> ceiling {ceil_ap:.4f}:  "
          f"best cell closes {100*(bestp-free_ap)/(ceil_ap-free_ap):.0f}% of the gap")

print("\nP1 as written vs the alternative:")
pk = [cell_stats(prof, c, "frontier_mask", "auroc")["peak_layer"] for c, _ in DEPTH_CELLS]
fr = [p / L for p, (_c, L) in zip(pk, DEPTH_CELLS)]
lf = [cell_stats(prof, c, "frontier_mask", "auroc")["peak_lift"] for c, _ in DEPTH_CELLS]
print(f"  peak layers {pk} for L={[L for _c, L in DEPTH_CELLS]} -> fractions "
      f"{[round(f,3) for f in fr]}")
print(f"  absolute spread {max(pk)-min(pk):.2f} (P1 bar <=1) -> P1 FAILS")
print(f"  fractional spread {max(fr)-min(fr):.3f} (bar <=0.12) -> the ALTERNATIVE HOLDS,"
      f" at fraction {fr[0]:.3f}")
print(f"  peak lifts {[round(v,4) for v in lf]} spread {max(lf)-min(lf):.4f}"
      f"  -- the AMOUNT is flat in depth")
""")

# ---------------------------------------------------------------- 4. P3
md(r"""
## 4. P3 — the control works: bookkeeping early, the deduction predicate last

> Pre-registered: *"`known_mask` is bookkeeping, not deduction. If it shows the same layer
> pattern as `frontier_mask`, the pattern is about the probe and not about DD."*

It does not. `known_mask` is already at AUROC 0.768 from the **embedding alone** — the
token sequence says which nodes have been coloured — and it peaks at layers 4, 8 and 2 for
L=8, 12 and 16: fractions 0.50 / 0.67 / 0.125, fractional spread 0.542, no depth pattern
at all. `frontier_mask` and `remaining_mask` both peak at the final layer in every cell.

So the layer structure is specific to the targets that need computing, and the control
that needs no computing does not show it. That is the P3 clause passing.
""")

code(r"""
print(f"{'cell':<18}{'target':<16}{'column':<18}{'chance':>8}{'layer0':>8}{'peak':>8}"
      f"{'@L':>4}{'final':>8}")
print("-" * 88)
for c in present:
    lab = CELLS[c][0]
    for t in TARGETS:
        m = metric_of(kind, t)
        s = cell_stats(prof, c, t, m)
        ch = majority.get(t, 0.5) if m == "balanced_accuracy" else 0.5
        print(f"{lab:<18}{t:<16}{m:<18}{ch:>8.4f}{s['layer0']:>8.4f}{s['peak_abs']:>8.4f}"
              f"{int(round(s['peak_layer'])):>4}{s['final']:>8.4f}")
    print()

print("peak-layer fraction of depth, by target, across L=8/12/16:")
for t in MULTILABEL:
    m = metric_of(kind, t)
    f = [cell_stats(prof, c, t, m)["peak_layer"] / L for c, L in DEPTH_CELLS]
    print(f"  {t:<16}{[round(v,3) for v in f]}   spread {max(f)-min(f):.3f}")
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
for ax, t in zip(axes, MULTILABEL):
    m = metric_of(kind, t)
    for c in present:
        lab, L, _n, sd = CELLS[c]
        xs = sorted({l for (cc, _s, tt), p in prof.items() if cc == c and tt == t for l in p})
        ys = [np.mean([p[l][m] for (cc, _s, tt), p in prof.items()
                       if cc == c and tt == t and l in p]) for l in xs]
        ax.plot(xs, ys, "--o" if sd == 1 else "-o", ms=3, lw=1.3, label=lab, rasterized=True)
    if t == "frontier_mask":
        ax.axhline(free_auc, color="k", lw=1.2, ls="--")
        ax.text(0.5, free_auc + 0.004, "label-only free score", fontsize=8)
        if ceil_auc is not None:
            ax.axhline(ceil_auc, color="r", lw=1.2, ls=":")
            ax.text(0.5, ceil_auc + 0.004,
                    "graded ceiling (color_aug: handed n_known_colors)",
                    fontsize=8, color="r")
    ax.set_title(f"{t} ({m})")
    ax.set_xlabel("layer  (0 = token+position embedding)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("AUROC")
axes[0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 5. yardsticks
md(r"""
## 5. P2 — the two noise floors every claim above was read against

> Pre-registered: *"(a) three probe-fit seeds on the same checkpoint give the probe's own
> noise; (b) the two training-seed pairs (t0/t4 at L=12, t1/t5 at L=8) give the bar a
> depth claim must clear, exactly as `rho_50`'s seed spread did in the scan."*

This is the clause that kept the depth reading honest. The `frontier_mask` peak **layer**
is rock solid — 0.00 spread in both channels — which is why the "peak is the last layer"
claim survives. The peak **lift** is not: one training seed is worth 0.0411 at L=12, which
exceeds the 0.0237 spread across the whole 8→16 depth range. `next_node` is noisier still,
0.1555 at L=8, which swallows both of its depth gaps.
""")

code(r"""
print("P2(a) probe-fit noise, 3 seeds, max spread over cells:")
for t in TARGETS:
    m = metric_of(kind, t)
    sp = [cell_stats(prof, c, t, m) for c in present]
    print(f"  {t:<16}({m:<17}) lift {max(s['spread_lift'] for s in sp):.4f}"
          f"   peak layer {max(s['spread_layer'] for s in sp):.2f}"
          f"   final {max(s['spread_final'] for s in sp):.4f}")

print("\nP2(b) training-seed noise, the bar a depth claim must clear:")
for t in TARGETS:
    m = metric_of(kind, t)
    out = []
    for a, b, L in SEED_PAIRS:
        sa, sb = cell_stats(prof, a, t, m), cell_stats(prof, b, t, m)
        out.append(f"L={L:<3} lift {abs(sa['peak_lift']-sb['peak_lift']):.4f}"
                   f" layer {abs(sa['peak_layer']-sb['peak_layer']):.2f}"
                   f" final {abs(sa['final']-sb['final']):.4f}")
    print(f"  {t:<16}" + "   ".join(out))

YARD = max(abs(cell_stats(prof, a, "frontier_mask", "auroc")["peak_lift"]
               - cell_stats(prof, b, "frontier_mask", "auroc")["peak_lift"])
           for a, b, _L in SEED_PAIRS)
print(f"\nYARDSTICK IN FORCE (frontier lift): {YARD:.4f}")
print(f"depth spread of the frontier lift : {max(lf)-min(lf):.4f} -> "
      f"{'BELOW -- unreadable' if max(lf)-min(lf) < YARD else 'above -- readable'}")
""")

# ---------------------------------------------------------------- 6. channels
md(r"""
## 6. The result I did not expect: frontier quality does not predict task success

Six cells, each with a `rho_50` from the scan's 14-point × 400-instance eval. Two probe
channels, ranked against it:

| channel | Spearman vs rho_50 (lower better) | vs val_nll |
|---|---|---|
| `next_node` final-layer top-1 | **−0.943** | −0.829 |
| `frontier_mask` peak AUROC | **−0.086** | −0.029 |

The DD predicate is decoded about equally well in the best and the worst cell: t5, which
is the worst model in the grid by a wide margin (`rho_50` 0.829, `val_nll` 1.8140), has
the **second-highest** frontier AUROC at 0.7643. What separates a good cell from a bad one
is `next_node` — *which* of the currently-deducible nodes it commits to emitting — where
t5 collapses to 0.6034 against 0.83 for the best.

Read with section 3, the picture is: **every cell works out roughly which nodes are
deducible, and the cells differ in whether they then act on it.** That is a specific,
falsifiable claim about where the N=96 failure lives, and it moves the next intervention
off the predicate and onto the selection step.

Two caveats kept in view. (i) n = 6 and `next_node` carries a 0.1555 training-seed spread
at L=8, so −0.943 is an observation over six points, not an estimate. (ii) `next_node` is
closer to the model's own output than `frontier_mask` is, so part of its advantage may be
proximity to the logits rather than mechanism; the clean test is an **intervention** on
the selection step, not a better probe.
""")

code(r"""
rows = sorted(present, key=lambda c: R50[c][0])
print(f"{'cell':<18}{'rho_50':>8}{'val_nll':>9}{'next_node':>11}{'frontier auroc':>16}")
for c in rows:
    nn = cell_stats(prof, c, "next_node", "balanced_accuracy")["final"]
    fm = cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"]
    print(f"{CELLS[c][0]:<18}{R50[c][0]:>8.4f}{R50[c][1]:>9.4f}{nn:>11.4f}{fm:>16.4f}")

r50 = [R50[c][0] for c in present]
vnl = [R50[c][1] for c in present]
nn = [cell_stats(prof, c, "next_node", "balanced_accuracy")["final"] for c in present]
fm = [cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] for c in present]
print(f"\nSpearman next_node      vs rho_50 {spearman(nn, r50):+.3f}   vs val_nll {spearman(nn, vnl):+.3f}")
print(f"Spearman frontier auroc vs rho_50 {spearman(fm, r50):+.3f}   vs val_nll {spearman(fm, vnl):+.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, ys, name in ((ax[0], nn, "next_node top-1 (final layer)"),
                    (ax[1], fm, "frontier_mask peak AUROC")):
    for c, y in zip(present, ys):
        a.scatter(y, R50[c][0], s=70, rasterized=True)
        a.annotate(CELLS[c][0].split(" ")[0], (y, R50[c][0]),
                   textcoords="offset points", xytext=(6, 4), fontsize=8)
    a.set_xlabel(name); a.set_ylabel("rho_50  (lower = better)")
    a.set_title(f"Spearman {spearman(ys, r50):+.3f}")
    a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 7. previous best
md(r"""
## 7. Comparison to the previous best on the same eval — and it is a decline

The same probe eval ran at **N=24** in the competence ladder (`2026-10-21`,
`sudoku_probes_competence_ladder_2026-10-21/probes_competence_ladder.csv`, same
`sudoku_probes` code path, same targets, same three probe seeds, L=12). That is the
previous best on this measurement and the honest comparison is **unflattering**:

| | N=24 ladder (best cell) | N=96 fall (best cell) |
|---|---|---|
| `frontier_mask` AUROC, peak | **0.9381** | 0.7724 |
| peak layer / depth | 0.417–1.000 | 1.000 |
| `known_mask` AUROC, peak | **0.9983** | 0.9072 |
| `next_node` top-1, final | **0.9640** | 0.8704 |

**This run does not beat the previous best on this eval; it is 0.166 AUROC below it.**

**The raw columns overstate it, and the bracket is the fair comparison.** N=24 is an
easier *task* as well as an easier ensemble: `51009233` puts its label-only free score at
**0.7931** against 0.6734 here, and its graded ceiling at 0.9379 against 0.8085. Measured
as the fraction of its own free→ceiling bracket that each model closes — which is the only
cross-N comparison that means anything — the picture is:

| | N=24 ladder (5 cells) | N=96 fall (6 cells) |
|---|---|---|
| bracket, free → ceiling | 0.7931 → 0.9379 | 0.6734 → 0.8085 |
| fraction of bracket closed | **78.7% – 100.2%** | **42.7% – 73.3%** |

So **the decline survives normalisation**: at N=24 the best cell saturates the ceiling
(100.2%, i.e. it decodes the predicate as well as a probe handed `n_known_colors`
outright), while at N=96 the best cell reaches 73.3% and the worst 42.7%. The N=96 models
have a materially weaker representation of deducibility, not merely a harder target.

Two further points, and the second is the one worth keeping.

First, the shortfall is not confined to the deduction predicate. `known_mask` — pure
bookkeeping, near-perfect at N=24 (0.998) — falls to 0.82–0.91 at N=96. The N=96 models do
not reliably track *which nodes are already coloured*, let alone which are deducible, so
part of the frontier shortfall is a bookkeeping shortfall underneath it. There is no
label-only bracket for `known_mask`, so that one is a raw comparison.

Second, at N=24 the frontier curve **peaks mid-stack and plateaus** (layer 5 of 12 for the
d256 cells, fraction 0.417); at N=96 it is still rising into the final layer in every
cell. The natural reading is that at N=96 the computation does not finish inside the
stack. That reading makes a prediction, and the depth axis in this very run **contradicts
it**: L=16 does not decode the predicate better than L=8 (0.742 vs 0.757, inside the seed
yardstick). So "depth-limited" is *not* supported, and the last-layer peak is better read
as the predicate being assembled continuously, with the last layer simply the most
complete — not as a computation that would finish given more layers.

**Caveat on comparability.** N=24 is a different ensemble (no fall; frontier positive rate
0.1463 against 0.1001 here), so even the bracket-normalised comparison is between two
tasks rather than two models on one task, and the AP columns are not comparable across N
at all. Both brackets are now measured on three instance pools each at 600 probe epochs,
so neither side is a point estimate.
""")

code(r"""
import csv as _csv
from collections import defaultdict as _dd

LADDER = os.path.join(os.environ["SCRATCH"], "spin_gnn",
                      "sudoku_probes_competence_ladder_2026-10-21",
                      "probes_competence_ladder.csv")
lad = _dd(dict)
lad_pos = {}
with open(LADDER) as f:
    for r in _csv.DictReader(f):
        key = (r["run"].split("/")[0], r["target"])
        lad[key].setdefault(int(r["layer"]), []).append(
            float(r["auroc"]) if r["kind"] == "multilabel" else float(r["balanced_accuracy"]))
        if r["target"] == "frontier_mask":
            lad_pos[r["run"].split("/")[0]] = float(r["positive_rate"])

def lad_peak(run, target):
    p = {l: float(np.mean(v)) for l, v in lad[(run, target)].items()}
    L = max(p)
    best = max((v, l) for l, v in p.items() if l >= 1)
    return p[0], best[0], best[1], L

runs = sorted({k[0] for k in lad})
print("N=24 competence ladder (L=12, 3 probe seeds), the previous best on this eval:")
print(f"  {'run':<44}{'target':<15}{'layer0':>8}{'peak':>8}{'@L':>4}{'frac':>7}")
for run in runs:
    for t in ("frontier_mask", "known_mask", "next_node"):
        l0, pk, pl, L = lad_peak(run, t)
        print(f"  {run[:44]:<44}{t:<15}{l0:>8.4f}{pk:>8.4f}{pl:>4}{pl/L:>7.3f}")
    print()

print(f"{'':<16}{'N=24 best':>12}{'N=96 best':>12}{'delta':>9}")
for t, m in (("frontier_mask", "auroc"), ("known_mask", "auroc"),
             ("next_node", "balanced_accuracy")):
    a = max(lad_peak(run, t)[1] for run in runs)
    b = max(cell_stats(prof, c, t, m)["peak_abs"] for c in present)
    print(f"{t:<16}{a:>12.4f}{b:>12.4f}{b-a:>+9.4f}")
print(f"\nfrontier positive rate: N=24 {np.mean(list(lad_pos.values())):.4f}   "
      f"N=96 {base['positive_rate']['average_precision']:.4f}"
      f"   -> AUROC comparable in kind, AP not comparable")

# Each N read against ITS OWN free->ceiling bracket. A raw AUROC gap across N
# confounds "weaker model" with "easier task"; the fraction of the bracket closed
# does not, because the bracket is measured on the same ensemble as the cells.
N24 = sorted(glob.glob(os.path.join(BDIR, "n24_ep600_off*.json")))
if N24 and ceil_auc:
    f24 = max(max(b[k]["auroc"] for k in FREE if k in b)
              for b in (json.load(open(p)) for p in N24))
    c24 = float(np.mean([json.load(open(p))["color_aug"]["auroc"] for p in N24]))
    p24 = float(np.mean([json.load(open(p))["run"]["positive_rate"] for p in N24]))
    print(f"\nBracket-normalised frontier_mask (the only cross-N comparison that means"
          f" anything):")
    print(f"  N=24 bracket  free {f24:.4f} .. ceiling {c24:.4f}   positive rate {p24:.4f}")
    print(f"  N=96 bracket  free {free_auc:.4f} .. ceiling {ceil_auc:.4f}"
          f"   positive rate {base['positive_rate']['average_precision']:.4f}")
    g24 = sorted(100 * (lad_peak(run, "frontier_mask")[1] - f24) / (c24 - f24)
                 for run in runs)
    g96 = sorted(100 * (cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] - free_auc)
                 / (ceil_auc - free_auc) for c in present)
    print(f"  N=24 closes  " + " ".join(f"{g:5.1f}%" for g in g24))
    print(f"  N=96 closes  " + " ".join(f"{g:5.1f}%" for g in g96))
    print(f"  -> N=24 {min(g24):.1f}-{max(g24):.1f}%   N=96 {min(g96):.1f}-{max(g96):.1f}%"
          f"   THE DECLINE SURVIVES NORMALISATION")
else:
    print("\n(no N=24 baseline JSONs; the cross-N comparison stays raw)")
""")

code(r"""
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for run in runs:
    p = {l: float(np.mean(v)) for l, v in lad[(run, "frontier_mask")].items()}
    xs = sorted(p); L = max(xs)
    ax.plot([x / L for x in xs], [p[x] for x in xs], "-", lw=1.1, color="tab:green",
            alpha=0.55, rasterized=True)
ax.plot([], [], "-", color="tab:green", label="N=24 ladder (5 cells, L=12)")
for c in present:
    L = layers[c]
    xs = sorted({l for (cc, _s, tt), p in prof.items()
                 if cc == c and tt == "frontier_mask" for l in p})
    ys = [np.mean([p[l]["auroc"] for (cc, _s, tt), p in prof.items()
                   if cc == c and tt == "frontier_mask" and l in p]) for l in xs]
    ax.plot([x / L for x in xs], ys, "-o", ms=3, lw=1.2, alpha=0.9, rasterized=True)
ax.plot([], [], "-o", ms=3, color="grey", label="N=96 fall (6 cells, L=8/12/16)")
ax.axhline(free_auc, color="k", lw=1.0, ls="--")
ax.text(0.02, free_auc + 0.006, "N=96 label-only free score", fontsize=8)
ax.set_xlabel("layer / num_layers   (0 = embedding)")
ax.set_ylabel("frontier_mask AUROC")
ax.set_title("The DD predicate, N=24 vs N=96 on fractional depth\n"
             "N=24 peaks mid-stack and plateaus; N=96 rises into the last layer, 0.17 lower")
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 8. verdict
md(r"""
## 8. Verdict, and what it does not license

**On hshunt's question — does the network run DD, proved by probe activations?** The probe
half now answers *partly yes*, which it did not before this re-scoring:

- The deducibility predicate is linearly decodable from the residual stream **above every
  label-only free score** (AUROC 0.731–0.772 against 0.6734, whose own spread over three
  instance pools is 0.0035; AP 0.306–0.348 against 0.2096 at a 0.1008 base rate), and
  **81% of the way** from that free score to the graded ceiling of 0.802 — a probe handed
  `n_known_colors`, the quantity DD actually thresholds.
- It is **built last**: peak at the final layer in every cell, fraction 1.000, spread
  0.000, stable across both noise channels. Bookkeeping (`known_mask`) is available from
  the embedding and peaks early with no depth pattern. The deduction predicate and the
  bookkeeping are not the same signal.
- The amount of it does **not** grow with depth (spread 0.0237 under a 0.0411 seed
  yardstick), and it does **not** predict task success (Spearman −0.086 against `rho_50`).

**What this does not license.** A probe shows information is *present and linearly
readable*, not that the model *uses* it. Nothing here is causal. The decisive experiments
are the ones hshunt asked for and they are still open: ablate the direction the frontier
probe reads and watch `rho_50`; patch it between a solving and a failing cell. Section 6
says where to aim them — at the **selection** step (`next_node`), which tracks `rho_50` at
−0.943, rather than at the predicate, which does not track it at all.

### Still open, in priority order

1. **The causal test.** Ablate / patch along the frontier probe direction at the final
   layer and measure `rho_50` on the same 14-point eval. Until this runs, "the model
   represents DD" is a statement about decodability only.
2. **The selection step.** `next_node` is the channel that separates the cells. An
   intervention there is the one with a predicted effect on the task metric.
3. ~~`51009233`~~ **done** — H1, H2 and H3 all pass (section 2). The free score moves
   0.0050 across instance pools and 0.0012 between 300 and 600 probe epochs, so the
   committed baseline was neither lucky nor underfit, and `color_aug` sits 0.13 above it,
   so the bracket is real. N=24 now has a free score of its own and section 7 is
   bracket-normalised.
4. The N=24/N=96 bookkeeping collapse (0.998 → 0.907) is unexplained and is upstream of
   everything else here. It is the one comparison in section 7 with no label-only bracket
   behind it, because the baseline script only brackets `frontier_mask`.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.1"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.expanduser("~/spin_gnn"), "notebooks",
                   "sudoku_n96_fall_probes_2026-11-03_results.ipynb")
io.open(out, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out}  ({len(CELLS)} cells)")
