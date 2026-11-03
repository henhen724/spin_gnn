"""Build the results notebook for the N=96 fall-ensemble scan (50860571, 50903022).

    python3 notebooks/make_n96_fallscan_notebook.py     # from the repo root

Writes $HOME/spin_gnn/notebooks/sudoku_n96_fallscan_2026-11-03_results.ipynb.

Every number in a code cell is read at run time from the scan CSVs and from
`scripts/n96_fallscan_analysis`, the COMMITTED reader -- the notebook imports
rho_50/rho50_ci from it rather than redefining them, so a notebook number and a
reader number cannot drift apart. The markdown quotes the values this produced
on 2026-11-03 at 10:45 with four of six cells in; rerun after t2/t5 land and the
tables update themselves while the quoted prose does not, so check both.

Pre-registrations (H1-H4) are quoted verbatim from the sbatch header, written
before submission.
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
# N=96, fall ensemble: the transition becomes measurable, and the seed swamps the grid

**Jobs.** `50860571` (array 0-4, `sud2N96FallScan`, submitted 2026-11-02 19:24:17) and
`50903022` (array 5-5, `sud2N96FallS1L8`, submitted 2026-11-03 04:22:03), partition
`sganguli`, one H100 80GB per cell, `sh04-02n01`/`sh04-03n01`.

**Commits.** Scan script `c531a86`; the L=8 seed-1 replicate `c71146d`; the reader this
notebook imports, `scripts/n96_fallscan_analysis.py`, committed the same morning.

**Config, identical in every cell except the knob named.** `n_nodes 96, degree 5,
n_colors 4, --fall --shuffle-adj`; `d_model 256, nhead 4, dim_feedforward 1024`, pre-norm;
`lr 1e-3 constant, no warmup, batch 32`; `n_val_instances 300`. The grid is
**equal-gradient-step by construction** -- 240k instances x 50 epochs and 120k x 100
epochs are both 375k steps at batch 32 -- so the data axis is instance *coverage* at
fixed optimization, not more optimization.

**Cells and seeds.**

| task | num_layers | n_train | epochs | seed | role |
|---|---|---|---|---|---|
| t0 | 12 | 240k | 50 | 0 | headline |
| t1 | 8 | 240k | 50 | 0 | depth, shallow |
| t2 | 16 | 240k | 50 | 0 | depth, deep |
| t3 | 12 | 120k | 100 | 0 | data axis, equal steps |
| t4 | 12 | 240k | 50 | **1** | seed replicate at L=12 |
| t5 | 8 | 240k | 50 | **1** | seed replicate at L=8 |

**Why N=96 at all.** At N=48 the fall ensemble drove `rho_min` to 0.332 but the trained
models solved the whole grid: every cell sat above 0.5 at the floor, so `rho_50` was
censored and the scaling question (finding 44) could only be answered in solve-rate
differences, not in a crossing. N=96 pushes the task harder at fixed hint density, and
H1 below is the pre-registered check that this actually bought a crossing.

**Reading direction: `rho_50` is the realized hint density at which solve rate crosses
0.5, and LOWER IS BETTER** -- a lower `rho_50` means the model still solves when given
fewer hints, i.e. it reaches deeper into the slow phase.
""")

# ---------------------------------------------------------------- 1. setup
code(r"""
import os, sys, math, json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

REPO = os.path.expanduser("~/spin_gnn")
sys.path.insert(0, REPO)

# The COMMITTED reader. Importing rho_50/rho50_ci rather than reimplementing them
# is the point: the notebook and `python3 scripts/n96_fallscan_analysis.py` cannot
# disagree, and the N=48 crossing convention (first UPWARD crossing of 0.5, walking
# up in rho) carries over so the two scans' numbers mean the same thing.
from scripts.n96_fallscan_analysis import (
    SCAN_DEFAULT, TASKS, load, rho_50, rho50_ci, history)

rows = load(os.path.join(SCAN_DEFAULT, "rho_curve_t*.csv"))
KEY = ("num_layers", "dim_feedforward", "n_train_instances", "seed")
cells = defaultdict(list)
for r in rows:
    cells[tuple(r[k] for k in KEY)].append(r)
for k in cells:
    cells[k].sort(key=lambda r: r["realized_rho"])

LABEL = {v: k for k, v in TASKS.items()}
present = [k for k in TASKS.values() if k in cells]
missing = [LABEL[k] for k in TASKS.values() if k not in cells]

print(f"scan dir : {SCAN_DEFAULT}")
print(f"cells in : {len(present)} of {len(TASKS)}")
for k in present:
    print(f"   {LABEL[k]:<32} {len(cells[k])} rho bins, "
          f"{cells[k][0]['n_instances']} instances/bin")
if missing:
    print("STILL RUNNING / ABSENT:", "; ".join(missing))
""")

md(r"""
## The eval, stated once so every number below is comparable

`scripts/rho_transition_eval` on a 14-point grid `rho = 0.95 ... 0.32`, **400 fresh
instances per bin**, fall ensemble, `--shuffle-adj`. Every cell is evaluated with the
default `--seed-start`, so all cells see the **same 400 instances at each rho** and the
curves are paired point for point -- a difference between two cells is a paired
difference, not two independent samples.

`rho` is the *requested* density; `realized_rho` is what greedy hint removal actually
achieved, and it is the x-axis everywhere below. The fall ensemble's floor is the lowest
realized density reachable at all.

There is **no earlier N=96 transition curve in this project** -- this eval did not exist
before `50860571`. So "previous best on the same eval" below means the best cell that had
landed when the question was asked this morning (t1), not an older run. The N=48 fall
scan is a different eval (different N, different floor) and is not comparable cell to
cell.
""")

# ---------------------------------------------------------------- 2. table
code(r"""
def summarize(k):
    g = cells[k]
    v, flag = rho_50([(r["realized_rho"], r["solve_rate"]) for r in g])
    lo, hi, dropped = rho50_ci(g)
    fv, bv, tr, ne = history(g[0]["path"])
    scored = [r for r in g if r["solve_rate"] > 0.1]     # where well-formedness matters
    return dict(
        label=LABEL[k], L=k[0], n_train=k[2], seed=k[3],
        n_params=g[0]["n_params"], epochs=ne,
        rho50=v, flag=flag, lo=lo, hi=hi,
        floor=min(r["realized_rho"] for r in g),
        val=fv, best_val=bv, train=tr,
        wf_min=min(r["well_formed_rate"] for r in g),
        wf_min_scored=min((r["well_formed_rate"] for r in scored), default=float("nan")),
        tr_max=max(r["truncated_rate"] for r in g),
        tr_max_scored=max((r["truncated_rate"] for r in scored), default=float("nan")),
        T_max=max(r["dd_mean_T"] for r in g))

S = {k: summarize(k) for k in present}

hdr = (f"{'cell':<32}{'L':>3}{'n_train':>9}{'seed':>5}{'params':>10}"
       f"{'rho_50':>9}{'95% CI':>18}{'val_nll':>9}{'wf_min':>8}{'trunc_max':>10}")
print(hdr); print("-" * len(hdr))
for k in present:
    s = S[k]
    ci = f"[{s['lo']:.3f}, {s['hi']:.3f}]" if not math.isnan(s["lo"]) else "n/a"
    star = "" if s["flag"] == "ok" else f" ({s['flag']})"
    print(f"{s['label']:<32}{s['L']:>3}{s['n_train']:>9}{s['seed']:>5}{s['n_params']:>10,}"
          f"{s['rho50']:>9.3f}{ci:>18}{s['val']:>9.4f}{s['wf_min']:>8.4f}"
          f"{s['tr_max']:>10.4f}{star}")

best = min(present, key=lambda k: S[k]["rho50"])
print(f"\nBEST rho_50 (lower is better): {S[best]['rho50']:.3f} -- {S[best]['label']}")
print(f"grid floor (lowest realized rho reachable): {min(S[k]['floor'] for k in present):.4f}")
print(f"max dd_mean_T anywhere: {max(S[k]['T_max'] for k in present):.2f} DD rounds")
""")

# ---------------------------------------------------------------- 3. H1
md(r"""
## H1 -- PASSES. The transition is measurable at N=96.

> Pre-registered: *"every cell's solve_rate crosses 0.5 strictly inside the rho grid, so
> rho_50 is an interpolation and not a censored bound."*

This is the clause that **failed at N=48**, where every cell sat above 0.5 even at the
floor and `rho_50` could only be reported as `<floor`. At N=96 every landed cell crosses
inside the grid: the flags in the table above are all `ok`, and `dd_mean_T` runs from
about 1.2 rounds at the easy end to >21 rounds at the floor, so the grid really does span
fast phase to deep slow phase.

**This is the result that justifies the move to N=96, and it is the one unambiguous win
in the run.**

## The eval controls, checked before anything is interpreted
""")

code(r"""
# 1. Pairing: every cell must have hit the same realized densities, or the "paired
#    difference" claim is false and the cells are being compared at different tasks.
ref = [round(r["realized_rho"], 6) for r in cells[present[0]]]
bad = {LABEL[k]: [round(r["realized_rho"], 6) for r in cells[k]]
       for k in present if [round(r["realized_rho"], 6) for r in cells[k]] != ref}
print(f"pairing: {len(present)-len(bad)}/{len(present)} cells on the identical rho grid"
      + ("" if not bad else f"  MISMATCHED: {list(bad)}"))

# 2. Truncation. The max_len margin fix (resolve_max_len_margin -> 194 at N=96) is only
#    load-bearing where the model actually solves; a truncated generation at rho where
#    solve_rate ~ 0 is noise, at rho near the crossing it would corrupt rho_50.
print(f"truncated_rate, bins with solve>0.1: max "
      f"{max(S[k]['tr_max_scored'] for k in present):.4f}  (pre-registered bar 0.01)")
print(f"truncated_rate, all bins          : max "
      f"{max(S[k]['tr_max'] for k in present):.4f}")

# 3. Well-formedness.
print(f"well_formed_rate, all bins        : min "
      f"{min(S[k]['wf_min'] for k in present):.4f}")
for k in present:
    print(f"    {LABEL[k]:<32} min wf {S[k]['wf_min']:.4f}")
""")

# ---------------------------------------------------------------- 4. H4
md(r"""
## H4 -- FAILS, by more than a factor of two, and it invalidates the rest of the grid

> Pre-registered: *"the seed term, |rho_50(t0) - rho_50(t4)|, is <= 0.05, small against
> the depth and data gaps H2 and H3 are reading. **IF IT IS LARGER THAN 0.05 HERE, this
> whole grid is one draw and only the ordering survives -- say so rather than quoting the
> levels.**"*

t0 and t4 are the **same configuration** -- L=12, 240k instances, 50 epochs, every
hyper-parameter byte-identical -- differing only in `--seed`. The cell below measures what
that one knob is worth.
""")

code(r"""
def seed_pairs():
    byconf = defaultdict(dict)
    for k in present:
        byconf[(k[0], k[1], k[2])][k[3]] = k
    return {c: d for c, d in byconf.items() if len(d) > 1}

pairs = seed_pairs()
sigma_hat = {}
YARD = float("nan")   # defined here so later cells import cleanly if no pair has landed
for conf, d in sorted(pairs.items()):
    ks = [d[s] for s in sorted(d)]
    vals = [S[k]["rho50"] for k in ks]
    spread = max(vals) - min(vals)
    # Paired per-rho solve-rate difference: the curves share instances, so this is the
    # same model evaluated twice, not sampling noise.
    a, b = cells[ks[0]], cells[ks[1]]
    dsolve = [abs(x["solve_rate"] - y["solve_rate"]) for x, y in zip(a, b)]
    # sd from two draws, bias-corrected for n=2 (c4 = sqrt(2/pi) = 0.7979).
    sigma_hat[conf[0]] = (spread / math.sqrt(2)) / 0.79788
    print(f"L={conf[0]:<3} n_train={conf[2]:<7} seeds {sorted(d)}: "
          f"rho_50 {' vs '.join(f'{v:.3f}' for v in vals)}  "
          f"spread {spread:.3f}   sigma_hat {sigma_hat[conf[0]]:.3f}")
    print(f"      per-rho |delta solve_rate|: max {max(dsolve):.3f}  mean {np.mean(dsolve):.3f}")

if sigma_hat:
    # Read a gap against the NOISIER end it spans: a depth claim is only as sharp as the
    # worse-measured of the two rungs it is a difference of.
    YARD = max(s * math.sqrt(2) * 0.79788 for s in sigma_hat.values())
    print(f"\nSEED YARDSTICK (largest measured spread): {YARD:.3f}   pre-registered bar: 0.050")
    print("H4 " + ("PASSES" if YARD <= 0.05 else "FAILS -- levels are not quotable, "
                                                 "only orderings that exceed this"))
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for conf, d in sorted(pairs.items()):
    for s in sorted(d):
        g = cells[d[s]]
        ax[0].plot([r["realized_rho"] for r in g], [r["solve_rate"] for r in g],
                   "o-", ms=4, label=f"L={conf[0]} seed {s}", rasterized=True)
        v, _ = rho_50([(r["realized_rho"], r["solve_rate"]) for r in g])
        ax[0].axvline(v, ls=":", lw=1, color=ax[0].lines[-1].get_color())
ax[0].axhline(0.5, color="k", lw=0.8, ls="--")
ax[0].set_xlabel("realized rho"); ax[0].set_ylabel("solve rate")
ax[0].set_title("Same config, different seed\n(dotted = rho_50)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

for conf, d in sorted(pairs.items()):
    ks = [d[s] for s in sorted(d)]
    a, b = cells[ks[0]], cells[ks[1]]
    ax[1].plot([x["realized_rho"] for x in a],
               [x["solve_rate"] - y["solve_rate"] for x, y in zip(a, b)],
               "o-", ms=4, label=f"L={conf[0]}: seed0 - seed1", rasterized=True)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("realized rho"); ax[1].set_ylabel("paired delta solve_rate")
ax[1].set_title("Paired seed difference at each rho\n(same 400 instances per point)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 5. H2/H3
md(r"""
## H2 and H3 -- both read SMALLER than the seed term, so neither is resolvable

> H2, pre-registered: *"rho_50 falls monotonically with depth, 8 -> 12 -> 16, and the
> 8 -> 12 gap is at least 0.05."*
>
> H3, pre-registered: *"240k beats 120k at equal gradient steps, i.e. coverage and not
> optimization is binding at N=96. Falsifier: if 120k beats 240k, then optimization and
> not coverage is binding at N=96."*

H3's falsifier fires *nominally* -- 120k lands below 240k -- and H2's depth gap has the
right sign but is about a tenth of the seed yardstick. Neither survives H4.
""")

code(r"""
print("H2, the depth axis at 240k / seed 0:")
dep = sorted((k for k in present if k[2] == 240000 and k[3] == 0), key=lambda k: k[0])
for k in dep:
    print(f"   L={k[0]:<3} rho_50 {S[k]['rho50']:.3f}  [{S[k]['lo']:.3f}, {S[k]['hi']:.3f}]"
          f"   val_nll {S[k]['val']:.4f}   params {S[k]['n_params']:,}")
for a, b in zip(dep, dep[1:]):
    gap = S[a]["rho50"] - S[b]["rho50"]
    print(f"   gap L={a[0]}->L={b[0]}: {gap:+.3f}   "
          f"{'BELOW' if abs(gap) < YARD else 'above'} the {YARD:.3f} seed yardstick"
          f"   ({'helps' if gap > 0 else 'hurts'})")

print("\nH3, the data axis at L=12 / seed 0 (equal 375k gradient steps):")
dat = sorted((k for k in present if k[0] == 12 and k[3] == 0), key=lambda k: k[2])
for k in dat:
    print(f"   n_train={k[2]:<7} epochs {S[k]['epochs']:<4} rho_50 {S[k]['rho50']:.3f}"
          f"  [{S[k]['lo']:.3f}, {S[k]['hi']:.3f}]   val_nll {S[k]['val']:.4f}")
if len(dat) == 2:
    gap = S[dat[1]]["rho50"] - S[dat[0]]["rho50"]
    print(f"   240k - 120k = {gap:+.3f}  ->  "
          f"{'120k wins (falsifier direction)' if gap > 0 else '240k wins (H3 direction)'}")
    print(f"   |{gap:.3f}| is {'BELOW' if abs(gap) < YARD else 'above'} the "
          f"{YARD:.3f} seed yardstick -> {'NOT RESOLVABLE' if abs(gap) < YARD else 'resolvable'}")
""")

code(r"""
fig, ax = plt.subplots(figsize=(7.5, 5))
for k in present:
    g = cells[k]
    ls = "--" if k[3] == 1 else "-"
    ax.plot([r["realized_rho"] for r in g], [r["solve_rate"] for r in g],
            ls, marker="o", ms=4, label=LABEL[k], rasterized=True)
ax.axhline(0.5, color="k", lw=0.8, ls="--")

# The seed band, drawn at the headline cell so depth gaps are read against it.
if pairs:
    hl = (12, 1024, 240000, 0)
    if hl in S:
        c = S[hl]["rho50"]
        ax.axvspan(c - YARD / 2, c + YARD / 2, color="grey", alpha=0.18, zorder=0,
                   label=f"seed yardstick ({YARD:.3f} wide)")
ax.set_xlabel("realized rho"); ax.set_ylabel("solve rate")
ax.set_title("N=96 fall ensemble: every cell, 400 paired instances per point\n"
             "grey band = what one seed is worth")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 6. val_nll
md(r"""
## An unexpected regularity: here, val_nll rank-orders rho_50 exactly

Finding 4 established that val_nll is a poor predictor of solve rate *across* models
(r = +0.33), and findings 20 and 25 had it ranking cells backwards. Inside this grid it
does not misbehave -- it rank-orders the cells perfectly. The cells span only ~0.02 in
val_nll and ~0.13 in rho_50, so a very small difference in how well a run optimized maps
to a large difference in how deep into the slow phase it solves.

That is the mechanism hypothesis for H4's failure: **the seed is not changing what the
network can represent, it is changing how well that particular run converged.** Every
cell is still improving when training stops (last-5-epoch val drop is negative in all of
them, and the "overfit gap" is negative only because train_nll is averaged over the epoch
while weights move -- finding 16).
""")

code(r"""
print(f"{'cell':<32}{'val_nll':>9}{'best_val':>10}{'rho_50':>9}{'epochs':>8}")
for k in sorted(present, key=lambda k: S[k]["val"]):
    s = S[k]
    print(f"{s['label']:<32}{s['val']:>9.4f}{s['best_val']:>10.4f}{s['rho50']:>9.3f}{s['epochs']:>8}")

v = [S[k]["val"] for k in present]
r = [S[k]["rho50"] for k in present]
if len(present) > 2:
    rv = np.argsort(np.argsort(v)); rr = np.argsort(np.argsort(r))
    n = len(present)
    rho_s = 1 - 6 * np.sum((rv - rr) ** 2) / (n * (n ** 2 - 1))
    print(f"\nSpearman rank correlation val_nll vs rho_50 over {n} cells: {rho_s:+.3f}"
          f"   (n is tiny -- this is an observation, not an estimate)")
    print(f"val_nll range {max(v)-min(v):.4f}   rho_50 range {max(r)-min(r):.3f}")

fig, ax = plt.subplots(figsize=(6, 4.2))
for k in present:
    ax.scatter(S[k]["val"], S[k]["rho50"], s=70, rasterized=True)
    ax.annotate(LABEL[k].split(" (")[0], (S[k]["val"], S[k]["rho50"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("final val_nll"); ax.set_ylabel("rho_50  (lower = better)")
ax.set_title("Small val_nll differences, large rho_50 differences")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 7. well-formed
md(r"""
## The well-formedness dip is one cell, and it is the seed again

The grid's minimum `well_formed_rate` is well below the ~1.00 seen when only t1 had
landed. It is **not** spread across cells and it is **not** a depth effect: it is confined
to a single cell, at low rho, and that cell's byte-identical twin at the other seed is
clean everywhere. Truncation is not the cause -- these generations terminate, they just do
not parse.

Two consequences. First, it is a second observable on which one seed changed the model
qualitatively, which corroborates H4 rather than being an independent problem. Second,
malformed output counts as *unsolved*, so it depresses that cell's solve rate -- the
affected cell is the grid's best **despite** the dip, not because of it.
""")

code(r"""
print(f"{'cell':<32}{'min wf':>8}{'at rho':>8}{'wf@crossing':>13}")
for k in present:
    g = cells[k]
    w = min(g, key=lambda r: r["well_formed_rate"])
    near = min(g, key=lambda r: abs(r["realized_rho"] - S[k]["rho50"]))
    print(f"{LABEL[k]:<32}{w['well_formed_rate']:>8.4f}{w['realized_rho']:>8.3f}"
          f"{near['well_formed_rate']:>13.4f}")

worst = min(present, key=lambda k: S[k]["wf_min"])
print(f"\nthe dip, bin by bin -- {LABEL[worst]}:")
for r in cells[worst]:
    if r["well_formed_rate"] < 0.99:
        print(f"   rho {r['realized_rho']:.3f}  well_formed {r['well_formed_rate']:.4f}"
              f"  truncated {r['truncated_rate']:.4f}  solve {r['solve_rate']:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
for k in present:
    g = cells[k]
    ax.plot([r["realized_rho"] for r in g], [r["well_formed_rate"] for r in g],
            "o-", ms=4, label=LABEL[k], rasterized=True)
ax.set_xlabel("realized rho"); ax.set_ylabel("well_formed_rate")
ax.set_title("Grammar holds in every cell but one")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 8. before/after
md(r"""
## Best rho_50, before and after, on the same eval

The morning's reference point was t1 -- the only cell that had landed, `rho_50 = 0.691`.
The table below is the honest before/after against it on the identical eval.

**The improvement is real as a number and worthless as a method.** t4 differs from t1 in
*two* knobs (depth 8 -> 12 **and** seed 0 -> 1). The one-knob control for depth is t0,
which sits essentially on top of t1 -- so almost none of the gain is depth, and almost all
of it is the seed draw. Nothing was tuned to produce it and nothing known would reproduce
it on a fresh seed.
""")

code(r"""
BEFORE = (8, 1024, 240000, 0)        # t1, the only cell in when the question was asked
if BEFORE in S and present:
    b, a = S[BEFORE], S[best]
    print(f"BEFORE  {b['label']:<32} rho_50 {b['rho50']:.3f}  [{b['lo']:.3f}, {b['hi']:.3f}]")
    print(f"AFTER   {a['label']:<32} rho_50 {a['rho50']:.3f}  [{a['lo']:.3f}, {a['hi']:.3f}]")
    print(f"delta   {a['rho50'] - b['rho50']:+.3f}   "
          f"CIs {'disjoint' if a['hi'] < b['lo'] or b['hi'] < a['lo'] else 'OVERLAP'}")
    ctrl = (12, 1024, 240000, 0)     # t0: same depth change as t4, seed held at 0
    if ctrl in S:
        dep_only = S[ctrl]["rho50"] - b["rho50"]
        tot = a["rho50"] - b["rho50"]
        print(f"\n  attributable to DEPTH (t1 -> t0, seed held at 0): {dep_only:+.3f}")
        print(f"  attributable to SEED  (t0 -> t4, depth held at 12): "
              f"{a['rho50'] - S[ctrl]['rho50']:+.3f}")
        print(f"  -> {100*abs(dep_only)/abs(tot):.0f}% of the move is depth, "
              f"{100*abs(a['rho50']-S[ctrl]['rho50'])/abs(tot):.0f}% is the seed")
""")

# ---------------------------------------------------------------- 9. design
md(r"""
## What this costs the planned hyper-parameter scan

The standing plan is a hyper-parameter scan at each N to relate reasoning depth to MLP
depth. H4 sets the price of admission: with a per-cell sd of `sigma_hat`, resolving a gap
`delta` between two rungs at 95% confidence and 80% power needs roughly
`n >= 2 sigma^2 (2.8/delta)^2` seeds **per rung**.
""")

code(r"""
if sigma_hat:
    sg = max(sigma_hat.values())
    print(f"sigma_hat (per cell, from the seed pair) = {sg:.3f}\n")
    print(f"{'gap to resolve':<34}{'seeds/rung':>12}{'cells':>8}{'GPU-h @ 11h':>13}")
    for name, delta in [("H2's pre-registered bar, 0.05", 0.05),
                        ("the seed spread itself, %.3f" % YARD, YARD),
                        ("observed L=8->12 gap", None)]:
        if delta is None:
            d8 = [k for k in present if k[0] == 8 and k[3] == 0]
            d12 = [k for k in present if k[0] == 12 and k[3] == 0]
            if not (d8 and d12):
                continue
            delta = abs(S[d8[0]]["rho50"] - S[d12[0]]["rho50"])
            name = f"observed L=8->12 gap, {delta:.3f}"
        n = math.ceil(2 * sg ** 2 * (2.8 / delta) ** 2)
        print(f"{name:<34}{n:>12}{2*n:>8}{2*n*11:>13,}")
    print("\nAt ~11 GPU-h per cell, averaging the seed away is not affordable for the")
    print("gaps this grid actually shows. The seed term has to be REDUCED, not averaged.")
""")

md(r"""
### The design conclusion, stated plainly

A depth scan at N=96 run the way this grid was run **cannot answer the depth question at
any affordable seed budget**. Averaging is the wrong lever. The val_nll regularity says
why and points at the right one: the cells differ by ~0.02 in val_nll, all of them are
still descending when training stops, and the run that happened to optimize best is the
run that solves deepest. The seed is buying *convergence*, not capacity.

So the next single change should attack convergence, not architecture: this grid trains
at **constant lr 1e-3 with no warmup and no decay**, which is exactly the setting in which
the end-of-training point is a lottery over where the iterate happened to be. A decay
schedule should both lower val_nll and shrink its run-to-run spread, and if the mechanism
hypothesis is right, `rho_50`'s spread collapses with it.

**Next experiment (one knob, two seeds, everything else byte-identical to t0/t4):** add an
lr schedule at L=12 / 240k / 50 epochs, seeds 0 and 1, and read out two numbers --
(1) does best `rho_50` beat 0.564, and (2) **is the seed spread below the 0.115 measured
here?** (2) is the one that matters, because without it no depth scan is worth running.
That is a direct test, a two-cell cost, and it is pre-registered by this paragraph.

### Still open

- t2 (L=16) and t5 (L=8 seed 1) had not landed when this was written. t5 is the one that
  says whether `sigma_hat` is a property of L=12 or of the protocol; if the L=8 spread is
  much smaller, the yardstick above is too conservative and H2 may be partly readable.
- The dip's mechanism -- what the malformed generations at low rho actually emit -- is
  uninspected. It is confined to one cell and does not affect the controls, but it has not
  been explained.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.1"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.expanduser("~/spin_gnn"), "notebooks",
                   "sudoku_n96_fallscan_2026-11-03_results.ipynb")
io.open(out, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out}  ({len(CELLS)} cells)")
