"""Builds notebooks/sudoku_data_scaling_2026-10-22_results.ipynb.

Covers three finished runs, all scored on ONE eval (the shared 1500-instance
pool, seed_start 5e10):
  * 48912548 -- d0256 x n_train=80000, 300 epochs (the 16x data run);
  * 48935109 -- d0128 x n_train=20000 + probes + shared-pool eval;
  * 48999377 -- all seven checkpoints of the width x data grid re-scored on the
    shared pool in one pass, plus probes on the 80000-instance model.

Writes figures next to the notebook as PNGs as well, because nothing on
Sherlock can execute a notebook right now (jupyter_client is broken against
this Python), so a reader who only opens the file still sees the plots.
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np

S = os.environ.get("SCRATCH", "/scratch/users/hshunt") + "/spin_gnn"
GRID_DIR = f"{S}/sudoku_shared_pool_grid_2026-10-22"
D128_DIR = f"{S}/sudoku_d128_data4x_2026-10-21"
NB_DIR = f"{S}/notebooks"
FIG_DIR = f"{NB_DIR}/sudoku_data_scaling_2026-10-22"
COMMIT = "dc3598b"

# (label, d_model, n_train, sweep dir) -- the sweep dir is also where the
# own-validation metrics come from, so the notebook can show own-val and
# shared-pool side by side and make the divergence explicit.
MODELS = [
    ("d128 x 5000", 128, 5000, None),
    ("d256 x 5000", 256, 5000, "sudoku_hparam_sweep"),
    ("d384 x 5000", 384, 5000, "sudoku_hparam_sweep_2026-10-21_d384"),
    ("d128 x 20000", 128, 20000, "sudoku_hparam_sweep_2026-10-21_d128_data4x"),
    ("d256 x 20000", 256, 20000, "sudoku_hparam_sweep_2026-10-21_d256_data4x"),
    ("d384 x 20000", 384, 20000, "sudoku_hparam_sweep_2026-10-21_d384_data4x"),
    ("d256 x 80000", 256, 80000, "sudoku_hparam_sweep_2026-10-21_d256_data16x"),
]


def shared_pool_rows():
    """checkpoint name -> row, keyed by (d_model, n_train) via the parent dir."""
    with open(f"{GRID_DIR}/agg_paired_full_grid.csv") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        name = r["checkpoint"]
        parent = re.search(r"\[(.*)\]", name)
        parent = parent.group(1) if parent else name
        if "epoch00300" in name:
            key = (128, 5000)
        elif parent == "sudoku_hparam_sweep":
            key = (256, 5000)
        else:
            d = int(re.search(r"d0(\d+)", name).group(1))
            n = {"data4x": 20000, "data16x": 80000}.get(parent.split("_")[-1], 5000)
            key = (d, n)
        out[key] = {k: (float(v) if k not in ("checkpoint", "vs_reference") else v)
                    for k, v in r.items()}
    return out


def own_val(sweep_dir, d_model, n_train):
    """Own-validation metrics for ONE cell of the grid.

    sweep.csv holds one row per run, and `sudoku_hparam_sweep` is the original
    28-run scan, so taking row 0 silently returns a d64/L2/ff128 model -- which
    is what it did on the first build of this notebook, putting k0=0.61 against
    d256 x 5000 instead of its real 0.24. Match on the architecture instead.
    """
    if sweep_dir is None:
        # The d0128 x 5000 cell is a checkpoint series, not a sweep: it has a
        # per-epoch own-val solve_rate but no k0 / o1 grid.
        path = f"{S}/sudoku_checkpoint_series_2026-10-19/series.csv"
        if not os.path.exists(path):
            return {}
        with open(path) as fh:
            row = next(r for r in csv.DictReader(fh) if r["epoch"] == "300")
        return {"final_val_nll": float(row["val_nll"]),
                "agg_solve_rate": float(row["solve_rate"])}
    path = f"{S}/{sweep_dir}/sweep.csv"
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    match = [r for r in rows
             if int(r["d_model"]) == d_model and int(r["num_layers"]) == 12
             and int(r["dim_feedforward"]) == 1024 and int(r["epochs"]) == 300
             and int(r["n_train_instances"]) == n_train]
    if len(match) != 1:
        raise SystemExit(f"{path}: {len(match)} rows match d{d_model} x {n_train} "
                         f"at L12/ff1024/300ep -- cannot pick an own-val row")
    r = match[0]
    return {k: float(r[k]) for k in
            ("n_params", "final_val_nll", "agg_solve_rate", "solve_rate_k0",
             "mean_o1_solve_rate") if r.get(k)}


def probe_peaks(csv_path):
    """(run, target) -> (peak layer, mean over seeds, seed spread, layer-0 mean).

    AUROC for the multilabel targets, accuracy for the categorical ones --
    AUROC is undefined for a k-way choice and comes back NaN, which is the
    trap that cost an afternoon on 10-21.
    """
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        metric = float(r["auroc"] if r["kind"] == "multilabel" else r["accuracy"])
        by.setdefault((r["run"], r["target"], int(r["layer"])), []).append(metric)
    out = {}
    runs = sorted({r["run"] for r in rows})
    targets = sorted({r["target"] for r in rows})
    for run in runs:
        for t in targets:
            vals = {L: v for (ru, tt, L), v in by.items() if ru == run and tt == t}
            peak_L = max(vals, key=lambda L: np.mean(vals[L]))
            out[(run, t)] = (peak_L, float(np.mean(vals[peak_L])),
                             float(np.ptp(vals[peak_L])), float(np.mean(vals[0])))
    return out


def spearman(xs, ys):
    """Rank correlation, ties averaged. Seven points, so read it as a sign and
    a rough magnitude, not a precise coefficient."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = rank(list(xs)), rank(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def run_key(run):
    """The probe CSV's `run` column is '<parent dir>/<tag>'.

    Returns None for runs that are not cells of the width x data grid. The
    900-epoch d256 run is the one that matters here: it is also d256 at
    n_train=5000, so mapping it by (width, data) alone would silently
    overwrite the 300-epoch cell with a different training budget.
    """
    parent = run.split("/")[0]
    if "ep900" in parent:
        return None
    if "checkpoint_series" in parent:
        return (128, 5000)
    d = int(re.search(r"d0(\d+)", run.split("/")[-1]).group(1)) if "d0" in run else 128
    n = {"data4x": 20000, "data16x": 80000}.get(parent.split("_")[-1], 5000)
    return (d, n)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    pool = shared_pool_rows()
    own = {(d, n): own_val(sd, d, n) for _, d, n, sd in MODELS}

    peaks = {}
    for path in (f"{D128_DIR}/probes_d128_2x2.csv",
                 f"{GRID_DIR}/probes_d256_data16x.csv",
                 f"{S}/sudoku_probes_competence_ladder_2026-10-21/probes_competence_ladder.csv"):
        if os.path.exists(path):
            for (run, t), v in probe_peaks(path).items():
                key = run_key(run)
                if key is not None:
                    peaks[(key, t)] = v

    # ---------------------------------------------------------------- figures
    widths = sorted({d for _, d, _, _ in MODELS})
    colours = dict(zip(widths, ["#1f77b4", "#d62728", "#2ca02c"]))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for d in widths:
        pts = sorted((n, pool[(d, n)]) for (dd, n) in pool if dd == d)
        if not pts:
            continue
        x = [n for n, _ in pts]
        y = [r["solve_rate"] for _, r in pts]
        lo = [r["solve_rate"] - r["wilson_lo"] for _, r in pts]
        hi = [r["wilson_hi"] - r["solve_rate"] for _, r in pts]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=4,
                    color=colours[d], label=f"d_model={d}")
    ax.set_xscale("log")
    ax.set_xticks([5000, 20000, 80000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Without this the decade minor labels (6x10^3, 3x10^4, ...) print on top
    # of the three ticks that actually exist.
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("n_train_instances")
    ax.set_ylabel("shared-pool solve_rate (1500 instances)")
    ax.set_title("Data scaling on one shared eval pool, 95% Wilson intervals")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_data_scaling.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target, name in zip(axes, ["frontier_mask", "next_node"],
                                ["frontier_mask peak AUROC", "next_node peak accuracy"]):
        for (d, n) in sorted(pool):
            if ((d, n), target) not in peaks:
                continue
            x = pool[(d, n)]["solve_rate"]
            y = peaks[((d, n), target)][1]
            ax.scatter(x, y, s=90, color=colours[d],
                       marker={5000: "o", 20000: "s", 80000: "D"}[n])
            ax.annotate(f"d{d}x{n // 1000}k", (x, y), textcoords="offset points",
                        xytext=(6, -12), fontsize=8)
        ax.set_xlabel("shared-pool solve_rate")
        ax.set_ylabel(name)
        ax.set_title(name.split(" peak")[0])
        ax.grid(alpha=0.3)
    axes[0].set_title("frontier_mask: tracks WIDTH, not competence")
    axes[1].set_title("next_node: tracks competence")
    handles = [plt.Line2D([], [], marker="o", ls="", color=colours[d], label=f"d_model={d}")
               for d in widths]
    handles += [plt.Line2D([], [], marker=m, ls="", color="0.4", label=f"n_train={n}")
                for n, m in ((5000, "o"), (20000, "s"), (80000, "D"))]
    # Upper left is the only empty corner: lower left sits on d128 x 5000.
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_probes_vs_competence.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ns = [5000, 20000, 80000]
    for metric, style in (("solve_rate_k0", "-o"), ("mean_o1_solve_rate", "--s")):
        y = [own.get((256, n), {}).get(metric) for n in ns]
        ax.plot(ns, y, style, label=metric)
    y = [pool[(256, n)]["solve_rate"] if (256, n) in pool else None for n in ns]
    ax.plot(ns, y, "-^", color="black", label="shared-pool solve_rate")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Without this the decade minor labels (6x10^3, 3x10^4, ...) print on top
    # of the three ticks that actually exist.
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("n_train_instances (d_model=256)")
    ax.set_ylabel("solve rate")
    ax.set_title("Full range keeps rising; the easy end peaks at 20000 and collapses")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig3_easy_end.png", dpi=140)
    plt.close(fig)

    # --------------------------------------------------------------- notebook
    nb = nbf.v4.new_notebook()
    c = []
    new = pool[(256, 80000)]
    best = pool[(256, 20000)]
    verdict = ("BEATS" if new["delta"] > 0 and new["mcnemar_exact_p"] < 0.05 else
               "does NOT beat" if new["delta"] < 0 and new["mcnemar_exact_p"] < 0.05 else
               "ties with")

    c.append(nbf.v4.new_markdown_cell(f"""# Does more training data keep helping? d0256 x 80000 on the shared pool

**Runs:** Slurm `48912548` (d0256 x n_train=80000, 300 epochs, 5:33:11),
`48935109` (d0128 x n_train=20000 + probes + eval, 1:07:26),
`48999377` (all seven checkpoints re-scored on the shared pool + probes on the
80000-instance model).

**Git commit:** `{COMMIT}` (branch `sudoku-dd`). Scripts
`scripts/slurm/sudoku2_d256_data16x.sbatch`,
`scripts/slurm/sudoku2_d128_data4x_probe.sbatch`,
`scripts/slurm/sudoku2_shared_pool_full_grid.sbatch`.

**Config** (one variable changed per run; everything else held):
`nhead=4, num_layers=12, dim_feedforward=1024, lr=1e-3, norm_first=True,
n_nodes=24, degree=5, n_colors=4, rho in [0.3, 0.95], hint_mode=removal,
scratchpad=False, n_val_instances=300, batch_size=32, epochs=300`.

**Seeds.** Training runs are single-seed (`sudoku_hparam_sweep` defaults). The
shared eval pool is `seed_start=50_000_000_000`, 1500 instances, disjoint from
every training set (max training seed 80000) and from every other eval module's
base. Probes use seeds 0/1/2, which shift both instance pools
(`500_000_000+seed` train, `900_000_000+seed` test) and the position sampling;
the spread across those three is the measurement noise quoted below.

## Hypothesis (stated before the eval ran)

`48912548`'s own-validation `agg_solve_rate` came back **0.9933**, up from
0.9733 at `n_train=20000`. I predicted the shared pool would put it at or above
the current best of 0.9847, with a positive paired delta, because 4x the data
cut the train/val gap from +0.817 to +0.095. The registered falsifier was a
negative delta, which would mean the collapse in its easy-end metrics
(`solve_rate_k0` 0.60 -> 0.07, `mean_o1` 0.918 -> 0.776) outweighs the
full-range gain and 0.9933 was a val-set artifact.

**Result: {new['solve_rate']:.4f} on the shared pool, delta
{new['delta']:+.4f} against 0.9847, McNemar exact p = {new['mcnemar_exact_p']:.4g}
on {int(new['discordant_won'] + new['discordant_lost'])} discordant instances
({int(new['discordant_won'])} won / {int(new['discordant_lost'])} lost). The
80000-instance model {verdict} the 20000-instance one.**"""))

    c.append(nbf.v4.new_markdown_cell("""## Why own-validation and the shared pool disagree

`train_sudoku_model` seeds its validation instances at
`seed + n_train_instances`, so every `n_train` validates on a *different* 300
instances, and the larger training ranges cover the smaller runs' validation
seeds. At a rate near 0.99 the SE on 300 instances is ~0.005, so nothing above
~0.97 can be ranked there at all. That is what reversed the top two on 10-21
(`48912134`), and it is why the number to quote from this notebook is the
shared-pool column, never the own-val one."""))

    c.append(nbf.v4.new_code_cell("""import csv, os, re
import numpy as np, matplotlib.pyplot as plt

S = os.environ["SCRATCH"] + "/spin_gnn"
GRID = f"{S}/sudoku_shared_pool_grid_2026-10-22"
rows = list(csv.DictReader(open(f"{GRID}/agg_paired_full_grid.csv")))
for r in rows:
    print(f"{float(r['solve_rate']):.4f}  [{float(r['wilson_lo']):.4f}, "
          f"{float(r['wilson_hi']):.4f}]  delta {float(r['delta']):+.4f}  "
          f"p {float(r['mcnemar_exact_p']):.3g}  {r['checkpoint']}")"""))

    hdr = ("| model | params | own-val agg | shared-pool solve_rate | 95% Wilson | "
           "delta vs best | McNemar p | k0 | mean_o1 | frontier AUROC | next_node |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for label, d, n, _ in MODELS:
        r = pool.get((d, n))
        o = own.get((d, n), {})
        fr = peaks.get(((d, n), "frontier_mask"))
        nn = peaks.get(((d, n), "next_node"))
        if r is None:
            continue
        lines.append(
            f"| {label} | {int(o['n_params']) if 'n_params' in o else '--'} | "
            f"{o.get('agg_solve_rate', float('nan')):.4f} | **{r['solve_rate']:.4f}** | "
            f"[{r['wilson_lo']:.4f}, {r['wilson_hi']:.4f}] | {r['delta']:+.4f} | "
            f"{r['mcnemar_exact_p']:.3g} | {o.get('solve_rate_k0', float('nan')):.2f} | "
            f"{o.get('mean_o1_solve_rate', float('nan')):.3f} | "
            + (f"{fr[1]:.4f} (L{fr[0]}) | " if fr else "-- | ")
            + (f"{nn[1]:.4f} (L{nn[0]}) |" if nn else "-- |")
        )

    c.append(nbf.v4.new_markdown_cell(
        "## Metrics table\n\nEvery `shared-pool solve_rate` is the same 1500 "
        "instances; `delta`/`McNemar p` are paired against d256 x 20000 "
        "(0.9847), the previous best on this eval. `own-val agg` is the "
        "300-instance per-run number kept only to show the divergence. Probe "
        "columns are the peak over layers, averaged over seeds 0/1/2.\n\n"
        + "\n".join(lines)))

    c.append(nbf.v4.new_markdown_cell(
        "## Data scaling on one eval\n\n![](sudoku_data_scaling_2026-10-22/fig1_data_scaling.png)"))
    cells_with_probes = [(d, n) for (d, n) in sorted(pool)
                         if ((d, n), "frontier_mask") in peaks]
    sr = [pool[k]["solve_rate"] for k in cells_with_probes]
    rho_f = spearman(sr, [peaks[(k, "frontier_mask")][1] for k in cells_with_probes])
    rho_n = spearman(sr, [peaks[(k, "next_node")][1] for k in cells_with_probes])
    rho_w = spearman([d for d, _ in cells_with_probes],
                     [peaks[(k, "frontier_mask")][1] for k in cells_with_probes])
    c.append(nbf.v4.new_markdown_cell(
        "## What the probes track\n\n"
        "![](sudoku_data_scaling_2026-10-22/fig2_probes_vs_competence.png)\n\n"
        f"Over the {len(cells_with_probes)} cells, Spearman against shared-pool "
        f"solve_rate is **{rho_n:+.2f}** for `next_node` peak accuracy and only "
        f"**{rho_f:+.2f}** for `frontier_mask` peak AUROC, while `frontier_mask` "
        f"against `d_model` is **{rho_w:+.2f}**.\n\n"
        "The individual cells make the point more bluntly than the coefficient "
        "does. d256 x 5000 reaches frontier AUROC 0.907 while solving 0.820; "
        "d128 x 20000 reaches only 0.883 while solving 0.973; and the new best "
        "model, d256 x 80000, *drops* to 0.907 from the 20000-instance run's "
        "0.916 while its solve rate rises 0.985 -> 0.996. Frontier "
        "representation strength is a function of width, and more data at fixed "
        "width can even cost a little of it.\n\n"
        "`next_node` instead steps up with training data at every width "
        "(0.738 -> 0.951 at d128, 0.789 -> 0.964 -> 0.990 at d256, 0.810 -> "
        "0.957 at d384) and reproduces the competence ordering with a single "
        "inversion, between d384 x 20000 and d128 x 20000 -- two models whose "
        "solve rates differ by 0.0013 and are not separable on this pool."))
    c.append(nbf.v4.new_markdown_cell(
        "## The easy end goes backwards\n\n![](sudoku_data_scaling_2026-10-22/fig3_easy_end.png)\n\n"
        "`solve_rate_k0` is the pure-copy case (zero uncolored nodes, no "
        "reasoning required) and `mean_o1` the near-trivial end. Both are "
        "own-val metrics on 100 instances per bin, so they are noisy -- but a "
        "fall from 0.60 to 0.07 is not noise. The shape is non-monotonic: data "
        "5000 -> 20000 *helped* the easy end (k0 0.24 -> 0.60) and 20000 -> "
        "80000 destroyed it (0.60 -> 0.07) while the full range still rose. "
        "Width did the same thing on 10-18 (agg up, k0 collapsing), so this is "
        "not specific to one axis; it looks like the trivial cases are being "
        "traded away for the rest of the distribution rather than bought by "
        "scale of any kind."))

    c.append(nbf.v4.new_code_cell("""# Any pairwise McNemar, from the stored per-instance matrix -- no GPU needed.
import numpy as np, math
z = np.load(f"{GRID}/agg_paired_full_grid_per_instance.npz", allow_pickle=True)
solved, names = z["solved"], list(z["names"])

def mcnemar(i, j):
    b = int((solved[i] & ~solved[j]).sum()); c = int((~solved[i] & solved[j]).sum())
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, t) for t in range(k + 1)) / 2 ** n)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"p={mcnemar(i, j):.4g}  {names[i][:34]} vs {names[j][:34]}")"""))

    nb["cells"] = c
    os.makedirs(NB_DIR, exist_ok=True)
    out = f"{NB_DIR}/sudoku_data_scaling_2026-10-22_results.ipynb"
    with open(out, "w") as fh:
        nbf.write(nb, fh)
    print(f"wrote {out} and 3 figures to {FIG_DIR}")
    print(json.dumps({k: v for k, v in
                      {"shared_pool": {f"d{d}x{n}": round(pool[(d, n)]["solve_rate"], 4)
                                       for (d, n) in sorted(pool)}}.items()}, indent=1))


if __name__ == "__main__":
    main()
