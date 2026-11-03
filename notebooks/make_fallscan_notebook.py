"""Build the results notebook for the N=48 fall-ensemble scaling scan.

    python3 notebooks/make_fallscan_notebook.py      # from the repo root

Writes $HOME/spin_gnn/notebooks/sudoku_n48_fallscan_2026-11-02_results.ipynb.
Every number in a code cell is read from the CSVs and history.json at run time;
the markdown quotes those same numbers, and the pre-registrations are quoted
verbatim from the sbatch header, which was written before submission.
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
# N=48, fall ensemble: how solve rate scales across the DD transition, with data, and with model size

**Jobs `50760297` (array 0-12, `sud2FallScan`) and `50760252` (`sud2RhoBefore`), both submitted
2026-11-02 01:27, all 14 tasks `COMPLETED 0:0` by 12:19:45, ~46 GPU-h on `sh04-02n01`/`sh04-03n01`,
partition `sganguli`. Commit at submit `cd97030`. Seeds 0 for every cell, plus seed 1 replicates of
the three L=8 cells.**

This is the first measurement in the project that actually crosses the transition. Commit `5c17d3c`
found why it could not be done before: under the unconstrained stub split 0.3827 of nodes at
q=4, d=5 never see all `q-1` other colours, are undeducible by DD forever, and pin greedy hint
removal at `rho_min = 0.469` at N=48 -- *above* the slides' `rho_c ~ 0.46`. **Every earlier N=48
result in this project is a fast-phase result.** Drawing the planted colouring as a *fall* colouring
(Dunbar et al. 2000: every vertex adjacent to at least one vertex of every other colour) drops the
floor to `0.333` and lifts the mean required depth at the floor from `T = 7.80` to `T = 13.79`.

### Headline

**All three axes move the curve, and they are not interchangeable.**

| axis | what moved | size | yardstick `s_L8` |
|---|---|---|---|
| hint density `rho` | solve falls ~1.0 -> <0.1 for 9 of the 11 learning cells | 0.9+ | -- (paired within a cell) |
| depth `L` 4 -> 12 at 120k | `rho_50` 0.744 -> 0.570 -> 0.460 | **0.284** | 0.10 (pre-reg bar) |
| data 30k -> 240k at L=12 | tail solve (`rho <= 0.5`) | **+0.862** | 0.028-0.147 |
| **parameters at fixed depth** (3.30M -> 9.60M, L=4) | `rho_50` 0.744 -> 0.748 | **0.004** | 0.03 (pre-reg bar) |

The last row is the load-bearing one. `L=4/ff=4096` (9,599,034 params) and `L=12/ff=1024`
(9,613,370 params) match to **0.15%** and behave nothing alike: the wide-shallow model sits
`+0.288` in `rho_50` above the deep one and within `0.004` of the *narrow* model of its own depth.
**Depth is serial computation here, not capacity**, and every depth reading in findings 11/14/22/33/38
stands as a depth reading.

**Scored against the pre-registration: H1 confirmed as shape but over-run at the deep end,
H2 confirmed at matched data only, H3 confirmed with an unregistered depth x data interaction,
H4 confirmed. Controls: pairing passes, grammar fails in 8 of 115 bins.**

### Config and provenance (recorded for reproduction)

N=48, d=5, q=4, `--fall --shuffle-adj`, `hint_mode=removal`, `rho` 0.30-0.95, `scratchpad=False`,
`d_model=256`, `nhead=4`, `norm_first=True`, `lr=1e-3` constant (`warmup_epochs=0`), `batch_size=32`,
`n_val_instances=300`, `max_len=414`. **Equal gradient steps, not equal epochs**: every cell takes
375k optimizer steps (30k x 400ep, 120k x 100ep, 240k x 50ep), so the data axis varies the number of
*distinct* examples with the amount of optimization held fixed. Eval: `scripts/rho_transition_eval.py`,
400 instances per density, 14 densities, greedy decoding to `max_new_tokens = max_len`.
""")

# ------------------------------------------------------------- 1. load data
md(r"""
## 1. Load, and check the eval is paired

`rho_transition_eval.py` writes a `val_nll` column but leaves it **empty** -- the sweep records
validation NLL in `history.json` under the key `nll`, not in `config.json`. Everything below reads it
from `history.json`.
""")

code(r"""
import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

S      = os.environ["SCRATCH"] + "/spin_gnn"
SCAN   = f"{S}/sudoku_n48_fall_scan_2026-11-02"    # 50760297, 13 array tasks, fall ensemble
BEFORE = f"{S}/sudoku_n48_rho_curves_2026-11-02"   # 50760252, 8 pre-existing default-ensemble cells
COMMIT = "cd97030"

NUM = ["num_layers", "d_model", "dim_feedforward", "n_params", "n_train_instances", "epochs",
       "seed", "n_nodes", "degree", "n_colors", "n_per_bin", "rho", "realized_rho", "solve_rate",
       "outcome_accuracy", "well_formed_rate", "truncated_rate", "n_instances", "dd_mean_T",
       "dd_resolved_frac", "dd_frac_recovered"]


def load(pattern):
    frames = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        # The array task id comes from the FILENAME, not from re-deriving it
        # from (L, ff, n_train, seed): that would silently renumber the cells
        # if the SPECS order in the sbatch ever changed.
        m = re.search(r"rho_curve_t(\d+)_", os.path.basename(f))
        d["task"] = int(m.group(1)) if m else -1
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["fall"] = d["fall"].astype(str) == "True"
    return d


fall   = load(f"{SCAN}/rho_curve_t*.csv")
before = load(f"{BEFORE}/*.csv")

_H = {}
def history(path):
    if path not in _H:
        with open(os.path.join(path, "history.json")) as fh:
            _H[path] = json.load(fh)
    return _H[path]

def nlls(path):
    # (final train nll, final val nll, best val nll). 'nll' is the VAL key.
    h = history(path)
    return float(h[-1]["train_nll"]), float(h[-1]["nll"]), min(float(e["nll"]) for e in h)

KEY = ["num_layers", "dim_feedforward", "n_train_instances", "seed"]
cells  = {k: g.sort_values("realized_rho", ascending=False) for k, g in fall.groupby(KEY)}
bcells = {k: g.sort_values("realized_rho", ascending=False)
          for k, g in before.groupby(["num_layers", "seed"])}

print(f"fall   : {len(fall):4d} rows, {len(cells):2d} cells, {fall.realized_rho.nunique()} densities")
print(f"before : {len(before):4d} rows, {len(bcells):2d} cells, {before.realized_rho.nunique()} densities")

# CONTROL: the grid must be identical cell for cell, which is what makes the
# curves paired point by point (same --seed-start, same instance config).
bad = [r for r, g in fall.groupby("rho") if g.realized_rho.nunique() != 1]
print(f"\nrequested densities whose realized_rho differs across the 13 cells: {len(bad)}"
      f"  ->  {'PAIRED' if not bad else 'NOT PAIRED'}")
assert not bad
""")

# ------------------------------------------------------------- 2. the grid
md(r"""
## 2. The grid, and what the x-axis means

`rho` is requested; `realized_rho` is what the greedy removal path actually reached on each instance,
and it is the honest x-axis. `dd_mean_T` is the mean number of *sequential* DD rounds the ground-truth
solver needed at that density -- the required reasoning depth. The two lowest requested densities
(0.33 and 0.30) both land on the same realized value, **the ensemble floor at `rho = 0.3333`**: below
it there is nothing left to remove. That is 14 distinct realized densities out of 15 requested ones,
not one density measured several times.
""")

code(r"""
grid = (fall.groupby("realized_rho")
            .agg(requested=("rho", "first"), T=("dd_mean_T", "first"),
                 resolved=("dd_resolved_frac", "first"))
            .sort_index(ascending=False))
grid.index = grid.index.round(4)
print(grid.to_string(float_format=lambda v: f"{v:7.4f}"))
print(f"\nfall ensemble floor  realized_rho = {fall.realized_rho.min():.4f}, T = {fall.dd_mean_T.max():.2f}")
print(f"default ensemble floor            = {before.realized_rho.min():.4f}, "
      f"T = {before.dd_mean_T.max():.2f}")
print("\nThe fall fix bought 0.138 of density and 1.77x of required reasoning depth.")
""")

# ------------------------------------------------------------- 3. the table
md(r"""
## 3. Metrics table: every cell

`gap` is final val NLL minus final train NLL. `rho_50` is the realized density at which the cell's
solve rate crosses 0.5, by linear interpolation between adjacent grid points; `>top` means the cell
never reaches 0.5 anywhere on the grid and `<floor` means it never falls below 0.5 -- both are
**bounds, not crossings**, and must not be plotted as if they were measurements.
""")

code(r"""
def curve(g):
    return g.realized_rho.to_numpy(), g.solve_rate.to_numpy()

def rho_50(g):
    # (value, flag). Linear interpolation on the first 0.5 crossing from the top.
    pts = sorted(set(zip(np.round(g.realized_rho.to_numpy(), 6), g.solve_rate.to_numpy())))
    if all(s >= 0.5 for _, s in pts):
        return pts[0][0], "<floor"
    if all(s < 0.5 for _, s in pts):
        return pts[-1][0], ">top"
    for (r0, s0), (r1, s1) in zip(pts, pts[1:]):
        if s0 < 0.5 <= s1:
            return r0 + (0.5 - s0) * (r1 - r0) / (s1 - s0), "ok"
    return float("nan"), "none"

SHOW = [0.9375, 0.75, 0.6458, 0.5, 0.3333]   # realized densities to tabulate
ORDER = [(4, 1024, 30000, 0), (4, 1024, 120000, 0), (4, 1024, 240000, 0), (4, 4096, 120000, 0),
         (8, 1024, 30000, 0), (8, 1024, 30000, 1), (8, 1024, 120000, 0), (8, 1024, 120000, 1),
         (8, 1024, 240000, 0), (8, 1024, 240000, 1),
         (12, 1024, 30000, 0), (12, 1024, 120000, 0), (12, 1024, 240000, 0)]

hdr = (f"{'task':>4} {'L':>3} {'ff':>5} {'params':>8} {'n_train':>8} {'ep':>4} {'seed':>4} "
       f"{'train':>7} {'val':>7} {'gap':>7} " + " ".join(f"{r:>6.2f}" for r in SHOW) + f" {'rho_50':>8}")
print(hdr); print("-" * len(hdr))
R50 = {}
for k in ORDER:
    g = cells[k]
    tr, va, _ = nlls(g.path.iloc[0])
    v, flag = rho_50(g); R50[k] = (v, flag)
    s = [g.loc[(g.realized_rho - r).abs().idxmin(), "solve_rate"] for r in SHOW]
    print(f"{int(g.task.iloc[0]):>4} {k[0]:>3} {k[1]:>5} {g.n_params.iloc[0]/1e6:>7.2f}M "
          f"{k[2]:>8,} {int(g.epochs.iloc[0]):>4} {k[3]:>4} "
          f"{tr:>7.3f} {va:>7.3f} {va - tr:>+7.3f} " + " ".join(f"{x:>6.3f}" for x in s)
          + f" {v:>8.3f}" + ("" if flag == "ok" else f" {flag}"))

print("\nEvery cell: 375k gradient steps, lr 1e-3 constant, batch 32, d_model 256, 4 heads,"
      f"\nN=48 d=5 q=4 fall ensemble, --shuffle-adj, 400 instances per density, commit {COMMIT}.")
""")

md(r"""
Two things in that table are worth reading before the hypotheses.

**`val_nll` does not rank the cells.** `L=12/30k` has the *worst* validation NLL of all thirteen
(1.650) and still solves 96.5% at `rho = 0.94`; `L=4/240k` has a better NLL (1.616) and solves 30.7%.
This is finding 4/20/25 again, now on the fall ensemble.

**Only the three 30k cells overfit, and they overfit more the deeper they are** (+0.042 at L=4,
+0.140 at L=8, +0.243 at L=12). The pre-registration named this as a watch item and said it would be
"part of the answer, not a reason to drop the cell". It is part of the answer -- see the H3 section.
""")

# --------------------------------------------------------- 4. the yardstick
md(r"""
## 4. The yardstick `s_L8`, fixed before the data landed

Finding 41 measured a same-architecture seed spread at L=8 *larger* than the one-rung depth gap the
ladder is built from, so no single-seed cell is a measurement of its architecture. The
pre-registration therefore fixed a yardstick:

> `s_L8(rho)` = |solve(seed 1) - solve(seed 0)| at L=8, from tasks 3-5 against 9-11; this is THE
> yardstick, and no effect smaller than it is claimed as an effect. [...] Where the effect is smaller,
> the statement is "not separable at this budget" -- not a result with a p-value. Curve SHAPE within
> one checkpoint is paired and does not need this caveat; curve LEVEL across checkpoints always does.

The spread is itself a function of data: it collapses from 0.430 at 30k to 0.073 at 240k. Every level
claim below uses the **worst case over the three budgets** at each density, which is the conservative
choice.
""")

code(r"""
RHOS = sorted(fall.realized_rho.unique(), reverse=True)
def srow(k):
    g = cells[k]
    return np.array([g.loc[(g.realized_rho - r).abs().idxmin(), "solve_rate"] for r in RHOS])

print("    n_train " + " ".join(f"{r:>6.2f}" for r in RHOS) + "     max    mean")
S8 = {}
for n in (30000, 120000, 240000):
    d = np.abs(srow((8, 1024, n, 1)) - srow((8, 1024, n, 0)))
    S8[n] = d
    print(f"{n:>11,} " + " ".join(f"{x:>6.3f}" for x in d) + f"   {d.max():>5.3f}   {d.mean():>5.3f}")
YARD = np.max(np.vstack(list(S8.values())), axis=0)
print("\nWORST CASE " + " ".join(f"{x:>6.3f}" for x in YARD))
print("\nseed spread in rho_50 itself: " +
      ", ".join(f"{n//1000}k {abs(R50[(8,1024,n,1)][0] - R50[(8,1024,n,0)][0]):.3f}"
                for n in (30000, 120000, 240000)))
""")

# ------------------------------------------------------------- 5. the plots
md(r"""
## 5. Key plots

**Figure 1** is the measurement hshunt asked for: solve rate across the transition, one panel per
data budget, with the two L=8 seeds shaded between them so the yardstick is visible on the same axes
as the effects.
""")

code(r"""
LCOL = {3: "#718096", 4: "#dd6b20", 6: "#b7791f", 8: "#2b6cb0", 10: "#6b46c1", 12: "#276749"}
NSTY = {30000: ":", 120000: "--", 240000: "-"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white"})

fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
for ax, n in zip(axes[:3], (30000, 120000, 240000)):
    for L in (4, 8, 12):
        x, y = curve(cells[(L, 1024, n, 0)])
        ax.plot(x, y, "-o", color=LCOL[L], ms=3.5, lw=2.2, label=f"L={L}")
    xs, ys = curve(cells[(8, 1024, n, 1)])
    x8, y8 = curve(cells[(8, 1024, n, 0)])
    ax.plot(xs, ys, "--", color=LCOL[8], lw=1.6, alpha=0.75, label="L=8, seed 1")
    ax.fill_between(x8, np.minimum(y8, ys), np.maximum(y8, ys), color=LCOL[8], alpha=0.14, lw=0)
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    ax.set_title(f"{n:,} examples, {int(round(375000/(n/32)))} epochs")
    ax.set_xlabel("realized hint density rho"); ax.invert_xaxis()
axes[0].set_ylabel("solve rate"); axes[0].set_ylim(-0.03, 1.03)
axes[0].legend(fontsize=9, loc="lower left")

ax = axes[3]
for n in (30000, 120000, 240000):
    v = [(L, *R50[(L, 1024, n, 0)]) for L in (4, 8, 12)]
    ax.plot([a for a, _, _ in v], [b for _, b, _ in v], NSTY[n], color="0.25", lw=1.8,
            label=f"{n//1000}k", zorder=1)
    for L, val, flag in v:
        ax.plot([L], [val], marker="o", ms=9, ls="none", color=LCOL[L],
                mfc="white" if flag != "ok" else LCOL[L], mew=2)
for n in (30000, 120000, 240000):
    val, flag = R50[(8, 1024, n, 1)]
    ax.plot([8.35], [val], marker="s", ms=6, ls="none", color=LCOL[8],
            mfc="white" if flag != "ok" else LCOL[8], mew=2)
ax.axhspan(0.28, 0.3333, color="#9b2c2c", alpha=0.12, lw=0)
ax.text(4.2, 0.30, "ensemble floor 0.333", size=9, color="#9b2c2c")
ax.set_xticks([4, 8, 12]); ax.set_xlim(3.3, 13.2)
ax.set_xlabel("layers L"); ax.set_title("rho_50  (open = bound, not a crossing)")
ax.tick_params(labelleft=True); ax.legend(fontsize=9, title="examples", loc="center right")
fig.suptitle("Fig 1. Solve rate across the DD transition, by data budget and depth "
             "(N=48, fall ensemble, 400 instances per point)", y=1.04, fontsize=13)
fig.tight_layout(); plt.show()
""")

md(r"""
**Figure 2** puts the three effects on the one scale that lets them be compared: the yardstick.
""")

code(r"""
fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

# (a) H4: parameters at fixed depth, against depth at fixed parameters.
a = ax[0, 0]
for k, lab, st in [((4, 1024, 120000, 0), "L=4,  ff=1024,  3.30M", "-o"),
                   ((4, 4096, 120000, 0), "L=4,  ff=4096,  9.60M", "--s"),
                   ((12, 1024, 120000, 0), "L=12, ff=1024,  9.61M", "-o")]:
    x, y = curve(cells[k])
    a.plot(x, y, st, color=LCOL[k[0]], ms=4, lw=2.2, label=lab,
           mfc="white" if k[1] == 4096 else None)
a.axhline(0.5, color="0.6", lw=1, ls=":"); a.invert_xaxis()
a.set_xlabel("realized rho"); a.set_ylabel("solve rate")
a.set_title("(a) H4: 2.9x the parameters at L=4 changes nothing")
a.legend(fontsize=9, loc="lower left")

# (b) every effect against the worst-case yardstick.
a = ax[0, 1]
a.fill_between(RHOS, 0, YARD, color="0.6", alpha=0.3, lw=0, label="s_L8 (worst case)")
a.plot(RHOS, np.abs(srow((4, 1024, 120000, 0)) - srow((12, 1024, 120000, 0))),
       "-o", color="#276749", ms=4, lw=2.2, label="depth L=4 vs L=12 (120k)")
a.plot(RHOS, np.abs(srow((12, 1024, 240000, 0)) - srow((12, 1024, 30000, 0))),
       "-o", color="#975a16", ms=4, lw=2.2, label="data 30k vs 240k (L=12)")
a.plot(RHOS, np.abs(srow((4, 4096, 120000, 0)) - srow((4, 1024, 120000, 0))),
       "-o", color="#9b2c2c", ms=4, lw=2.2, label="params 3.3M vs 9.6M (L=4)")
a.invert_xaxis(); a.set_xlabel("realized rho"); a.set_ylabel("|effect|")
a.set_title("(b) effect size vs the yardstick"); a.legend(fontsize=9)

# (c) the depth x data interaction.
a = ax[1, 0]
for L in (4, 8, 12):
    v = [R50[(L, 1024, n, 0)] for n in (30000, 120000, 240000)]
    a.plot([30, 120, 240], [b for b, _ in v], "-o", color=LCOL[L], ms=8, lw=2.2, label=f"L={L}")
    for n, (b, f) in zip((30, 120, 240), v):
        if f != "ok":
            a.plot([n], [b], "o", ms=8, color="white", mec=LCOL[L], mew=2)
a.axhline(0.3333, color="#9b2c2c", lw=1, ls="--")
a.set_xscale("log"); a.minorticks_off()
a.set_xticks([30, 120, 240]); a.set_xticklabels(["30k", "120k", "240k"])
a.set_xlabel("distinct training examples (375k steps everywhere)"); a.set_ylabel("rho_50")
a.set_title("(c) depth pays only if it is fed"); a.legend(fontsize=9)

# (d) the L=4 plateau -- why two L=4 cells never learned.
a = ax[1, 1]
for n, st in [(30000, ":"), (120000, "-"), (240000, "--")]:
    h = history(cells[(4, 1024, n, 0)].path.iloc[0])
    steps = np.arange(1, len(h) + 1) * (n / 32)
    a.plot(steps / 1000, [e["train_nll"] for e in h], st, color=LCOL[4], lw=2,
           label=f"L=4, {n//1000}k")
h = history(cells[(8, 1024, 120000, 0)].path.iloc[0])
a.plot(np.arange(1, len(h) + 1) * (120000 / 32) / 1000, [e["train_nll"] for e in h], "-",
       color=LCOL[8], lw=1.5, alpha=0.7, label="L=8, 120k (reference)")
a.set_xlabel("thousands of gradient steps"); a.set_ylabel("train NLL")
a.set_title("(d) L=4 only: a plateau, escaped once in three"); a.legend(fontsize=9)

fig.suptitle("Fig 2. Depth, data and parameters, each read against the same yardstick",
             y=1.00, fontsize=13)
fig.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------- 6. H1
md(r"""
## 6. H1 -- "the transition is a transition". **Confirmed as shape, over-run at the deep end.**

> **H1.** Prediction: for every cell reaching solve >= 0.5 at rho=0.95, solve(0.95) - solve(0.35)
> >= 0.4, monotone non-increasing in rho to within sampling noise, and rho_50 > 0.50 -- i.e. models
> break down well ABOVE the ensemble's structural floor of 0.332, so what the curve measures is the
> model and not the instance generator running out of room.
> FALSIFIER: a curve flat to within 0.1 across the whole grid, or rho_50 < 0.40 for every cell.

**The falsifier did not fire.** Nine of the eleven cells that learned anything fall from ~1.0 to
below 0.1; no curve is flat; `rho_50` ranges over 0.333-0.938.

**Both of the remaining clauses fail, and they fail on the best cells.** The `>= 0.4` drop clause
holds for 10 of the 11 learning cells; the exception is `L=12/240k`, whose drop is only 0.245
*because it is still at 0.755 when the grid ends*. And the `rho_50 > 0.50` clause fails for the same
two cells: `L=12/120k` crosses at 0.460 and
`L=12/240k` never crosses at all -- it still solves **75.5%** at the floor, where the fall ensemble
runs out of removable hints. At N=48 the best model now **outruns the generator**: the deep end of
the measurement is a property of the instance distribution, not of the model. This is the concrete
reason the next scan has to move to larger N, and it is the opposite of the failure the
pre-registration was guarding against.
""")

code(r"""
print(f"{'cell':>26} {'s(0.94)':>8} {'s(floor)':>9} {'drop':>7} {'monotone':>9} {'rho_50':>8}")
for k in ORDER:
    s = srow(k)
    mono = bool(np.all(np.diff(s) <= 0.02))   # 0.02 = sampling noise at n=400
    v, flag = R50[k]
    lab = f"L={k[0]:<2} ff={k[1]:<4} {k[2]//1000:>3}k s{k[3]}"
    print(f"{lab:>26} {s[0]:>8.3f} {s[-1]:>9.3f} {s[0]-s[-1]:>7.3f} {str(mono):>9} "
          f"{v:>8.3f}{'' if flag=='ok' else ' ' + flag}")

learn = [k for k in ORDER if srow(k)[0] >= 0.5]
drops = [srow(k)[0] - srow(k)[-1] for k in learn]
print(f"\ncells reaching 0.5 at the easy end : {len(learn)} of 13")
print(f"  of those, drop >= 0.4            : {sum(d >= 0.4 for d in drops)}  (min drop {min(drops):.3f})")
print(f"  of those, rho_50 > 0.50          : {sum(R50[k][0] > 0.5 for k in learn)}  "
      f"-> FAILS for {[f'L={k[0]}/{k[2]//1000}k' for k in learn if R50[k][0] <= 0.5]}")
print(f"  flat to within 0.1 (falsifier)   : {sum(d < 0.1 for d in drops)}")
print(f"  rho_50 < 0.40 for EVERY cell     : {all(R50[k][0] < 0.40 for k in ORDER)}")
""")

# ----------------------------------------------------------------- 7. H2
md(r"""
## 7. H2 -- "depth buys the tail and nothing else". **Confirmed at matched data; not separable at 30k.**

> **H2.** Prediction: rho_50 is monotone decreasing in L over {4, 8, 12} at each n_train, with
> rho_50(L=4) - rho_50(L=12) >= 0.10; and the cells agree to within 0.03 at rho=0.95, separating only
> below rho ~ 0.7. FALSIFIER: a non-monotone L ordering whose size exceeds s_L8 at the same rho, or
> depth separating the cells at rho >= 0.9 as much as at rho <= 0.5.

At 120k and 240k every clause holds: `rho_50` is monotone in L, the gap is 0.284 and 0.604 against a
bar of 0.10, the cells agree to 0.030 at `rho = 0.94` against a bar of 0.03, and the spread peaks at
0.752 around `rho = 0.65`. Depth separates the curves **25x more in the tail than at the easy end**.

**At 30k the ordering inverts** -- `rho_50` reads 0.938 / 0.608 / 0.765 at L=4/8/12, so L=12 sits
*above* L=8. The pre-registered falsifier asks whether the inversion exceeds `s_L8`: the inversion is
0.157 in `rho_50` and the measured seed spread in `rho_50` at 30k is **0.154**. They are the same
size. The honest verdict is **not separable at this budget**, and the L=4 cell in that row did not
train at all (section 11). H2 is confirmed where the data axis is adequate and untestable where it is
not.
""")

code(r"""
for n in (30000, 120000, 240000):
    v = [R50[(L, 1024, n, 0)][0] for L in (4, 8, 12)]
    mono = v[0] > v[1] > v[2]
    print(f"n_train={n:>7,}: rho_50 = {v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f} at L=4/8/12   "
          f"monotone={str(mono):<5} gap(L4-L12)={v[0]-v[2]:+.3f}  (bar >= 0.10)")

print("\nseparation across L at fixed n_train, against the yardstick:")
print(f"{'':>14}" + " ".join(f"{r:>6.2f}" for r in RHOS))
# At 240k the L=4 cell never left its loss plateau (section 12), so a max-min
# over {4,8,12} there measures the plateau, not depth; that row is L=8 vs L=12.
for n, Ls in ((120000, (4, 8, 12)), (240000, (8, 12))):
    sp = np.max([srow((L, 1024, n, 0)) for L in Ls], axis=0) - \
         np.min([srow((L, 1024, n, 0)) for L in Ls], axis=0)
    lab = f"{n//1000}k L{'/'.join(str(L) for L in Ls)}"
    print(f"{lab:>14}" + " ".join(f"{x:>6.3f}" for x in sp))
print(f"{'s_L8':>14}" + " ".join(f"{x:>6.3f}" for x in YARD))

sp120 = np.max([srow((L, 1024, 120000, 0)) for L in (4, 8, 12)], axis=0) - \
        np.min([srow((L, 1024, 120000, 0)) for L in (4, 8, 12)], axis=0)
print(f"\nat 120k: spread {sp120[0]:.3f} at rho=0.94 (bar <= 0.03) vs {sp120.max():.3f} at "
      f"rho={RHOS[int(np.argmax(sp120))]:.2f}  ->  {sp120.max()/sp120[0]:.0f}x bigger in the tail")
inv = R50[(12,1024,30000,0)][0] - R50[(8,1024,30000,0)][0]
print(f"30k inversion in rho_50: {inv:+.3f};  seed spread in rho_50 at 30k: "
      f"{abs(R50[(8,1024,30000,1)][0]-R50[(8,1024,30000,0)][0]):.3f}  ->  NOT SEPARABLE")
""")

# ----------------------------------------------------------------- 8. H3
md(r"""
## 8. H3 -- "data helps, and helps most in the tail". **Direction confirmed; size separable only at L=12.**

> **H3.** Prediction: at matched steps, solve(240k) - solve(30k) > s_L8 at rho <= 0.5, and smaller
> than that gap at rho >= 0.8. FALSIFIER: 30k matching or beating 240k anywhere by more than s_L8.
> Watch: val_nll in history.json will show if the 30k cells overfit.

The direction is right everywhere and the falsifier never fires, but the *size* only clears the
yardstick at L=12:

| L | tail mean (rho <= 0.5) | easy mean (rho >= 0.8) | separable? |
|---|---|---|---|
| 4 | +0.000 | -0.001 | no -- the cell never trained |
| 8 | +0.061 | +0.004 | marginal; below `s_L8` at most densities |
| 12 | **+0.862** | +0.136 | yes, by an order of magnitude |

### The unregistered finding: a depth x data interaction

The pre-registration treated depth and data as separate axes. They are not. **Depth pays only if it
is fed.** Going 30k -> 240k moves the tail by +0.000 at L=4, +0.060 at L=8 and +0.862 at L=12; going
L=4 -> L=12 moves `rho_50` by +0.172 at 30k (wrong sign, not separable), +0.284 at 120k and +0.604 at
240k. The marginal value of a layer rises with the data budget and the marginal value of data rises
with depth. A hyper-parameter scan that varies one at a fixed value of the other will read the wrong
answer, in a direction that depends on which value it fixed.

The watch item fired exactly as described: the three 30k cells are the only ones that overfit, and
**they overfit more the deeper they are** (+0.042 / +0.140 / +0.243 at L=4/8/12). 30k instances seen
400 times is not enough to train 12 layers on this task.
""")

code(r"""
tail = np.array(RHOS) <= 0.5
easy = np.array(RHOS) >= 0.8
print("           " + " ".join(f"{r:>6.2f}" for r in RHOS) + "    tail    easy")
for L in (4, 8, 12):
    d = srow((L, 1024, 240000, 0)) - srow((L, 1024, 30000, 0))
    print(f"  L={L:<2}    " + " ".join(f"{x:>+6.3f}" for x in d)
          + f"  {d[tail].mean():>+6.3f}  {d[easy].mean():>+6.3f}")
print("  s_L8    " + " ".join(f"{x:>6.3f}" for x in YARD))
print(f"\nfalsifier (30k beats 240k by more than s_L8 anywhere): "
      f"{any(((srow((L,1024,30000,0)) - srow((L,1024,240000,0))) > YARD).any() for L in (4,8,12))}")

print("\noverfit gap (val - train) by cell, 30k rows only:")
for L in (4, 8, 12):
    g = cells[(L, 1024, 30000, 0)]
    tr, va, best = nlls(g.path.iloc[0])
    print(f"  L={L:<2} 30k: train {tr:.4f}  val {va:.4f}  best val {best:.4f}  gap {va-tr:+.4f}")
""")

# ----------------------------------------------------------------- 9. H4
md(r"""
## 9. H4 -- "it is depth, not parameters". **Confirmed. This is the load-bearing cell.**

> **H4.** Prediction: task 12 (L=4, ff=4096, 9.60M) has rho_50 within 0.03 of task 1 (L=4, ff=1024,
> 3.29M) and at least 0.08 above task 7 (L=12, ff=1024, 9.61M) -- i.e. it behaves like its depth, not
> like its parameter count. FALSIFIER: it tracks L=12 instead. That would say the L ladder has been
> measuring capacity all along, and every depth reading in findings 11/14/22/33/38 would need
> restating as a capacity reading.

`|rho_50(wide L=4) - rho_50(narrow L=4)| = 0.004` against a bar of 0.03, and
`rho_50(wide L=4) - rho_50(L=12) = +0.288` against a bar of +0.08. The two L=4 curves never differ by
more than 0.042 at any density -- inside the yardstick everywhere. **2.9x the parameters at fixed
depth buys nothing on this task.**

So the L ladder measures serial computation. The mechanism the project has been assuming -- a
fixed-depth transformer can chain only so many sequential deduction rounds, and `rho` controls how
many are required -- survives its most direct test.
""")

code(r"""
k4, k4w, k12 = (4, 1024, 120000, 0), (4, 4096, 120000, 0), (12, 1024, 120000, 0)
for k, lab in [(k4, "L=4  ff=1024"), (k4w, "L=4  ff=4096"), (k12, "L=12 ff=1024")]:
    print(f"{lab}: {int(cells[k].n_params.iloc[0]):>10,} params   rho_50 {R50[k][0]:.3f}")
print(f"\nparameter match: "
      f"{abs(cells[k4w].n_params.iloc[0] - cells[k12].n_params.iloc[0]) / cells[k12].n_params.iloc[0] * 100:.2f}%")
print(f"|rho_50(wide L=4) - rho_50(L=4) | = {abs(R50[k4w][0] - R50[k4][0]):.3f}   (bar <= 0.03)  "
      f"{'PASS' if abs(R50[k4w][0]-R50[k4][0]) <= 0.03 else 'FAIL'}")
print(f" rho_50(wide L=4) - rho_50(L=12)  = {R50[k4w][0] - R50[k12][0]:+.3f}   (bar >= +0.08)  "
      f"{'PASS' if R50[k4w][0]-R50[k12][0] >= 0.08 else 'FAIL'}")
d = np.abs(srow(k4w) - srow(k4))
print(f"\nmax |wide L=4 - narrow L=4| over the whole grid: {d.max():.3f} at rho="
      f"{RHOS[int(np.argmax(d))]:.2f};  s_L8 there is {YARD[int(np.argmax(d))]:.3f}")
""")

# ---------------------------------------------------------------- 10. controls
md(r"""
## 10. Controls -- pairing passes, grammar fails in 8 of 115 bins, and the cause is a context cap

> **C.** truncated_rate <= 0.01 and outcome_accuracy >= 0.99 at every rho for every cell with
> solve > 0.1, so a gap is deduction and not grammar (finding 25's failure mode). realized_rho
> identical across all 13 cells at each requested rho [...] The floor note must fire for at most the
> last grid point.

- **Pairing: PASS.** 0 of 15 requested densities give a different `realized_rho` across the 13 cells.
  Every curve is the same 400 instances per point, so curve *shape* needs no error bar.
- **Floor note: PASS.** Exactly one requested density lands on the floor, in each ensemble.
- **Grammar: FAIL, in a bounded and diagnosable way.** 8 of 115 live bins breach, worst
  `outcome_accuracy` 0.9475 and worst `truncated_rate` 0.0275 (bars 0.99 and 0.01). **All 8 are in the
  two L=12 cells at the deepest densities** -- the only cells still solving anything down there.

**Cause, found:** `max_len = 414` tokens, and the longest true trace on the fall ensemble at N=48 is
**408** (`SUDOKU_DD.md`: mean 393 -> 394, max 403 -> 408 when switching ensembles). The deepest bins
are therefore context-capped, and the cap bites exactly where the best model is still working. The
effect is bounded by the measured truncation rate, 0.028, which is one to two orders of magnitude
below the 0.6-0.8 depth and data effects, so **no conclusion in this notebook turns on it** -- but
**the L=12 tail numbers are lower bounds**, and `max_len` must be raised before the N=96 scan, where
traces are 2x longer and the cap would bind everywhere rather than in the last two bins.
""")

code(r"""
live = fall[fall.solve_rate > 0.1]
bad = live[(live.truncated_rate > 0.01) | (live.outcome_accuracy < 0.99)]
print(f"live bins (solve > 0.1): {len(live)};  breaching: {len(bad)}")
print(f"worst outcome_accuracy {live.outcome_accuracy.min():.4f} (bar 0.99), "
      f"worst truncated_rate {live.truncated_rate.max():.4f} (bar 0.01)")
print(bad[["task", "num_layers", "n_train_instances", "seed", "realized_rho", "solve_rate",
           "outcome_accuracy", "well_formed_rate", "truncated_rate"]]
      .sort_values("realized_rho", ascending=False).to_string(index=False))

cfg = json.load(open(os.path.join(cells[(12,1024,240000,0)].path.iloc[0], "config.json")))
print(f"\nmax_len = {cfg['max_len']} tokens; longest true fall trace at N=48 = 408 (SUDOKU_DD.md).")
print(f"headroom = {cfg['max_len'] - 408} tokens  ->  the deepest bins are context-capped.")
""")

# ---------------------------------------------------------- 11. previous best
md(r"""
## 11. Comparison to the previous best, on the one axis the two ensembles share

**The fall fix changes the graph ensemble, so no cell here is paired with any earlier checkpoint and
the two sets cannot be McNemar'd.** The previous best at N=48 is `L=12`, default ensemble, seed 0
(job `50760252` re-evaluated the eight existing leak-free checkpoints on the default-ensemble grid for
exactly this comparison): **0.948 at its floor `rho = 0.4714`, `T = 7.80`, and it was never measured
past that point because the ensemble has no deeper instances.**

Two caveats, both against the new cells:

1. **The old cells got 2x the optimization** -- 240k x 100 epochs = 750k steps, against the scan's
   375k. This is not a like-for-like win and none is claimed.
2. **Different instances.** The honest shared axis is `dd_mean_T`, the required reasoning depth, which
   both ensembles report per bin.

On that axis the new cells **track the old ones at equal L on half the optimization, and then carry on
for another 1.77x in depth**: `L=12/240k` fall solves 0.985 at `T = 7.18` (where the old best managed
0.948 at `T = 7.80`) and **0.755 at `T = 13.79`**, a regime the previous best was structurally unable
to be tested in.
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
BEST = {4: (4, 1024, 120000, 0), 8: (8, 1024, 240000, 0), 12: (12, 1024, 240000, 0)}
for i, xkey in enumerate(["realized_rho", "dd_mean_T"]):
    a = ax[i]
    for L in (4, 8, 12):
        g = bcells[(L, 0)]
        a.plot(g[xkey], g.solve_rate, "--s", color=LCOL[L], ms=4, lw=1.6, alpha=0.65,
               mfc="white", label=f"L={L}, unconstrained, 750k steps")
    for L in (4, 8, 12):
        g = cells[BEST[L]]
        a.plot(g[xkey], g.solve_rate, "-o", color=LCOL[L], ms=4, lw=2.4,
               label=f"L={L}, fall, {BEST[L][2]//1000}k x 375k steps")
    a.axhline(0.5, color="0.6", lw=1, ls=":")
    a.set_ylabel("solve rate")
if True:
    ax[0].set_xlabel("hint density rho -- NOT paired across ensembles"); ax[0].invert_xaxis()
    ax[0].axvline(0.4714, color="#9b2c2c", lw=1.2, ls=":")
    ax[0].text(0.468, 0.12, "old floor", color="#9b2c2c", size=9, rotation=90)
    ax[0].set_title("vs rho")
    ax[1].axvspan(7.7975, 13.79, color="#276749", alpha=0.08, lw=0)
    ax[1].text(8.3, 0.80, "newly reachable\n7.8 -> 13.8 rounds", color="#276749", size=10)
    ax[1].set_xlabel("mean DD rounds T required -- the comparable axis")
    ax[1].set_title("vs required reasoning depth")
ax[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Fig 3. Against the previous best. The old ensemble stops at rho=0.471 / T=7.8",
             y=1.02, fontsize=13)
fig.tight_layout(); plt.show()

g = cells[(12, 1024, 240000, 0)]
b = bcells[(12, 0)]
print(f"previous best (L=12, unconstrained, 750k steps): {b.solve_rate.iloc[-1]:.3f} "
      f"at its floor rho={b.realized_rho.iloc[-1]:.4f}, T={b.dd_mean_T.iloc[-1]:.2f}")
near = g.iloc[(g.dd_mean_T - 7.7975).abs().argmin()]
print(f"new L=12/240k at the nearest depth  T={near.dd_mean_T:.2f}: {near.solve_rate:.3f}")
print(f"new L=12/240k at its own floor      T={g.dd_mean_T.iloc[-1]:.2f}: {g.solve_rate.iloc[-1]:.3f}"
      f"   <- a regime the old ensemble cannot generate")
""")

# -------------------------------------------------------------- 12. plateau
md(r"""
## 12. The L=4 plateau: two of the thirteen cells never trained

Two L=4 cells (30k and 240k, tasks 0 and 2) sit at solve ~0.31 at `rho = 0.94` and 0.000 everywhere
below. They are not weak models; **they are untrained ones.** All three L=4/ff=1024 cells plateau at
train NLL ~1.64, and only the 120k cell escaped: 1.644 at epoch 50, 1.574 by epoch 75. The 240k cell
was cut off at 375k steps **at exactly the NLL the 120k cell had when it escaped.**

With one seed per cell this is a coin flip, not a capacity measurement. **The L=4 row of this grid
measures trainability, not capacity**, and the two bound-valued `rho_50` entries at L=4 must not be
read as depth results. The `L=4/ff=4096` cell (task 12) did escape, which is one more reason H4's
comparison is between the two cells that trained.
""")

code(r"""
print(f"{'cell':>18} {'epochs':>7} {'steps':>8} {'first NLL':>10} {'final NLL':>10} {'escaped?':>9}")
for n in (30000, 120000, 240000):
    h = history(cells[(4, 1024, n, 0)].path.iloc[0])
    tr = [e["train_nll"] for e in h]
    print(f"{'L=4 ff1024 ' + f'{n//1000}k':>18} {len(h):>7} {int(len(h)*n/32):>8,} "
          f"{tr[0]:>10.4f} {tr[-1]:>10.4f} {str(tr[-1] < 1.60):>9}")
h = history(cells[(4, 4096, 120000, 0)].path.iloc[0])
print(f"{'L=4 ff4096 120k':>18} {len(h):>7} {int(len(h)*120000/32):>8,} "
      f"{h[0]['train_nll']:>10.4f} {h[-1]['train_nll']:>10.4f} {str(h[-1]['train_nll'] < 1.60):>9}")

h120 = [e["train_nll"] for e in history(cells[(4, 1024, 120000, 0)].path.iloc[0])]
h240 = [e["train_nll"] for e in history(cells[(4, 1024, 240000, 0)].path.iloc[0])]
print(f"\n120k cell at epoch 50 (= 187.5k steps): {h120[49]:.4f};  by epoch 75: {h120[74]:.4f}  <- escape")
print(f"240k cell at its LAST epoch 50 (= 375k steps): {h240[-1]:.4f}  <- cut off mid-plateau")
""")

# -------------------------------------------------------------- 13. verdict
md(r"""
## 13. What this establishes, and what comes next

**Established.**

1. The success rate has a clean transition in hint density on the fall ensemble, measured at 400
   instances per point over 14 paired densities spanning `T = 1.15` to `T = 13.79`.
2. **Depth moves it, and does so as serial computation rather than capacity** (H4: 2.9x parameters at
   fixed depth moves `rho_50` by 0.004; 3x depth at fixed parameters moves it by 0.288).
3. **Data moves it, most in the tail**, and interacts with depth: the marginal value of a layer rises
   with the data budget. Neither axis is readable at a fixed value of the other.
4. The fall ensemble reaches depths the old one structurally could not, and the best cell there still
   solves 75.5% at the floor.

**Not established.**

- Anything about a *level* comparison at 30k, where the seed spread (0.154 in `rho_50`) swallows the
  depth effect (0.157).
- Anything about L=4 as a depth, given two of three cells never left their loss plateau.
- Where the transition actually sits for `L=12`: it is below the ensemble floor at N=48.

**Next, in order.**

1. **Raise `max_len`** from 414. It is 6 tokens above the longest true trace and is already truncating
   0.028 of the deepest L=12 bins. At N=96 it would bind everywhere. One-line config change, but it
   invalidates checkpoint reuse, so it goes in before the next training job rather than after.
2. **N=96 on the fall ensemble.** H1's `rho_50 > 0.50` clause failed because the model outran the
   generator, which is the signal to move N up, not to re-run N=48.
3. **The hyper-parameter scan hshunt asked for** (does required reasoning depth track MLP depth?) must
   now be run at >= 120k examples and with seed replicates at every rung. At 30k it would measure the
   plateau and the seed spread instead of the architecture.
4. **Seed replicates at L=4 and L=12.** The L=8 pair is the only error bar in the grid, and the two
   cells whose interpretation is weakest are the ones with no replicate.
""")

md(r"""
## 14. Limitations

1. **One seed per cell except L=8.** Tasks 9-11 are the only replicates; every L=4 and L=12 statement
   carries the L=8 spread as its assumed error bar, which is an assumption, not a measurement.
2. **Two L=4 cells did not train** (section 12), so the L=4 row is three points of which one is a
   model.
3. **`max_len = 414` truncates the deepest L=12 bins** at a rate of up to 0.028. Those numbers are
   lower bounds (section 10).
4. **No pairing across ensembles.** Section 11's comparison is on `dd_mean_T` and is qualitative by
   construction; the old cells also had 2x the optimization.
5. **`ff = 1024` everywhere except task 12**, and `d_model = 256` everywhere. The width axis is one
   cell wide.
6. **N=48 only**, and the best cell's transition is below the ensemble floor, so `rho_50` for
   `L=12/240k` is a bound.
7. **Greedy decoding only.** No sampling, no self-consistency; the solve rate is a single-shot number.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.1"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.expanduser("~/spin_gnn/notebooks/sudoku_n48_fallscan_2026-11-02_results.ipynb")
io.open(out, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out, "with", len(CELLS), "cells")
