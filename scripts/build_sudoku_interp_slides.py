"""Slide deck for the hint-density transition + interpretability results.

Every page is optional: a missing input becomes a page that says what is
missing rather than a crash or, worse, a silently dropped result. Raster plots
at dpi=150 keep the file well under 2 MB for phone reading.

Usage:
    python -m scripts.build_sudoku_interp_slides --out notebooks/<name>.pdf
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

SCRATCH = os.environ.get("SCRATCH", "")
BIG, MED, SMALL = 26, 19, 15


def _read(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(row, key, default=np.nan):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _page(pdf, title):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=BIG, fontweight="bold", y=0.96)
    return fig


def _missing(pdf, title, what):
    fig = _page(pdf, title)
    fig.text(0.5, 0.5, f"not available yet:\n{what}", ha="center", va="center",
             fontsize=MED, color="#999999")
    pdf.savefig(fig)
    plt.close(fig)


def page_title(pdf, headline, sub):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.70, "DD-Sudoku transformer", ha="center", fontsize=34, fontweight="bold")
    fig.text(0.5, 0.60, "hint-density transition + what algorithm it uses",
             ha="center", fontsize=23, color="#444444")
    fig.text(0.5, 0.42, headline, ha="center", fontsize=MED, wrap=True)
    fig.text(0.5, 0.18, sub, ha="center", fontsize=SMALL, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def page_where_transition(pdf, path):
    rows = _read(path)
    if not rows:
        return _missing(pdf, "Where the transition is", path)
    fig = _page(pdf, "Where the transition is: DD deduction time diverges at rho_min")
    ax1, ax2 = fig.subplots(1, 2)
    sizes = sorted({int(r["n_nodes"]) for r in rows})
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sizes)))
    for c, N in zip(cmap, sizes):
        sub = [r for r in rows if int(r["n_nodes"]) == N and r["mean_T"] != ""
               and _f(r, "frac_paths_reaching_rho") >= 0.8]
        if not sub:
            continue
        ax1.plot([_f(r, "rho") for r in sub], [_f(r, "mean_T") for r in sub],
                 "-", color=c, lw=2.2, label=f"N={N}", rasterized=True)
    ax1.set_xlabel("hint density rho", fontsize=SMALL)
    ax1.set_ylabel("DD deduction time T (rounds)", fontsize=SMALL)
    ax1.set_title("T rises sharply toward the floor", fontsize=MED)
    ax1.invert_xaxis()
    ax1.legend(fontsize=11, ncol=2)
    ax1.grid(alpha=0.3)

    rmin = [(N, np.mean([_f(r, "mean_rho_min") for r in rows if int(r["n_nodes"]) == N]))
            for N in sizes]
    ax2.semilogx([n for n, _ in rmin], [v for _, v in rmin], "o-", lw=2.2, ms=9, color="#c0392b")
    ax2.axhline(0.4777, ls="--", color="#888888", label="large-N rho_min = 0.478")
    ax2.set_xlabel("N (nodes)", fontsize=SMALL)
    ax2.set_ylabel("rho_min (greedy floor)", fontsize=SMALL)
    ax2.set_title("the floor is N-independent", fontsize=MED)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_model_transition(pdf, path):
    rows = _read(path)
    if not rows:
        return _missing(pdf, "Does the model collapse where DD slows down?", path)
    fig = _page(pdf, "Does the model collapse where DD's reasoning depth diverges?")
    axes = fig.subplots(1, len({r["checkpoint"] for r in rows}) or 1, squeeze=False)[0]
    for ax, ck in zip(axes, sorted({r["checkpoint"] for r in rows})):
        sub = sorted([r for r in rows if r["checkpoint"] == ck], key=lambda r: _f(r, "realized_rho"))
        x = [_f(r, "realized_rho") for r in sub]
        ax.plot(x, [_f(r, "solve_rate") for r in sub], "o-", color="#1f77b4", lw=2.2,
                label="solve_rate", rasterized=True)
        ax.plot(x, [_f(r, "truncated_rate") for r in sub], "s--", color="#d62728", lw=1.6,
                ms=4, label="truncated_rate (length confound)", rasterized=True)
        ax.plot(x, [_f(r, "dd_resolved_frac") for r in sub], ":", color="#2ca02c", lw=2,
                label="DD solves (control = 1.0)")
        ax.set_xlabel("realized hint density", fontsize=SMALL)
        ax.set_ylabel("rate", fontsize=SMALL)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(ck.replace("_lr1e-03_pre", ""), fontsize=14)
        ax.grid(alpha=0.3)
        tw = ax.twinx()
        tw.plot(x, [_f(r, "dd_mean_T") for r in sub], "^-", color="#ff7f0e", lw=2,
                ms=5, label="DD mean T", rasterized=True)
        tw.set_ylabel("DD deduction time T", fontsize=SMALL, color="#ff7f0e")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = tw.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="center left")
    fig.text(0.5, 0.025,
             "CAVEAT: rho and T are collinear in this ensemble (removing hints is what lengthens the "
             "chain), so\nr(solve,T) = -0.97 vs r(solve,rho) = +0.94 cannot attribute the collapse to "
             "depth rather than density.",
             ha="center", fontsize=11, color="#b22222")
    fig.tight_layout(rect=[0, 0.09, 1, 0.90])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_probes(pdf, path, title="Probes: bookkeeping yes, DD predicate no"):
    rows = _read(path)
    if not rows:
        return _missing(pdf, title, path)
    cks = sorted({r["checkpoint"] for r in rows})
    cks = [c for c in cks if "d0128" in c] or cks[:1]
    fig = _page(pdf, title)
    ax = fig.subplots()
    styles = {"known_mask": ("#2ca02c", "o", "known_mask (bookkeeping)"),
              "frontier_mask": ("#d62728", "s", "frontier_mask (THE DD predicate)"),
              "remaining_mask": ("#9467bd", "^", "remaining_mask"),
              "next_node": ("#1f77b4", "v", "next_node"),
              "next_color": ("#ff7f0e", "D", "next_color")}
    for target, (color, marker, label) in styles.items():
        sub = sorted([r for r in rows if r["checkpoint"] == cks[0] and r["target"] == target],
                     key=lambda r: int(r["layer"]))
        if not sub:
            continue
        ax.plot([int(r["layer"]) for r in sub], [_f(r, "balanced_accuracy") for r in sub],
                marker=marker, color=color, lw=2.2, ms=6, label=label, rasterized=True)
    ax.axhline(0.5, ls=":", color="#888888", lw=1.5)
    ax.annotate("layer 0 = raw token+position embedding (control)", xy=(0.02, 0.03),
                xycoords="axes fraction", fontsize=12, color="#666666")
    ax.set_xlabel("layer", fontsize=SMALL)
    ax.set_ylabel("balanced accuracy (held-out instances)", fontsize=SMALL)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.02, 1, 0.91])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_algorithm_id(pdf, path):
    """Head-to-head vote, scored ONLY where DD and the baseline disagree.

    This is the page that answers "what algorithm", because agreement rates on
    the full trace are uninformative: solve_with_dd_trace emits each round as
    sorted(...), so node ids ascend within a round and a pure ordering rule
    already matches DD most of the time with no deducibility at all."""
    rows = _read(path)
    if not rows:
        return _missing(pdf, "Which rule wins where DD and a heuristic disagree?", path)
    order = ["ascending_after", "ascending", "most_known_nbrs", "random_uncolored"]
    stats = []
    for b in order:
        sub = [r for r in rows if r["baseline"] == b]
        if not sub:
            continue
        n = len(sub)
        stats.append((b, sum(r["model_follows_dd"] == "True" for r in sub) / n,
                      sum(r["model_follows_baseline"] == "True" for r in sub) / n, n))
    fig = _page(pdf, "Which rule wins where DD and a heuristic disagree?")
    ax = fig.subplots()
    y = np.arange(len(stats))
    ax.barh(y - 0.2, [s[1] for s in stats], height=0.38, color="#d62728", label="follows DD")
    ax.barh(y + 0.2, [s[2] for s in stats], height=0.38, color="#1f77b4", label="follows heuristic")
    for i, s in enumerate(stats):
        ax.text(max(s[1], s[2]) + 0.015, i, f"n={s[3]}", va="center", fontsize=11, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stats], fontsize=13)
    ax.set_xlabel("fraction of disagreement positions", fontsize=SMALL)
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    ax.annotate("ascending_after = 'next uncolored id above the last emitted, wrapping'\n"
                "the only baseline that exploits the sorted-emission convention -- and it beats DD",
                xy=(0.02, 0.02), xycoords="figure fraction", fontsize=12, color="#444444")
    fig.tight_layout(rect=[0, 0.09, 1, 0.91])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_sae(pdf, path):
    rows = _read(path)
    if not rows:
        return _missing(pdf, "SAE: do sparse features mean DD predicates?", path)
    fig = _page(pdf, "SAE: do sparse directions individually mean DD predicates?")
    ax1, ax2 = fig.subplots(1, 2)
    layers = [int(r["layer"]) for r in rows]
    ax1.plot(layers, [_f(r, "l0") for r in rows], "o-", color="#1f77b4", lw=2.2, label="L0 (sparsity)")
    ax1.set_ylabel("mean L0", fontsize=SMALL, color="#1f77b4")
    tw = ax1.twinx()
    tw.plot(layers, [_f(r, "frac_variance_explained") for r in rows], "s--", color="#2ca02c", lw=2)
    tw.set_ylabel("frac variance explained", fontsize=SMALL, color="#2ca02c")
    ax1.set_xlabel("layer", fontsize=SMALL)
    ax1.set_title("dictionary health\n(labels are void if L0 is large)", fontsize=14)
    ax1.grid(alpha=0.3)

    ax2.plot(layers, [_f(r, "n_interpretable") for r in rows], "o-", color="#d62728", lw=2.2,
             label="matched a DD predicate (F1>=0.3)")
    ax2.plot(layers, [_f(r, "n_alive") for r in rows], "s--", color="#888888", lw=1.8, label="alive")
    ax2.set_xlabel("layer", fontsize=SMALL)
    ax2.set_ylabel("features", fontsize=SMALL)
    ax2.set_title("best frontier F1 per layer: "
                  + ", ".join(f"L{int(r['layer'])}={_f(r, 'best_frontier_f1'):.2f}" for r in rows),
                  fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_circuits(pdf, out_dir):
    beh = _read(os.path.join(out_dir, "behavior.csv"))
    patch = _read(os.path.join(out_dir, "patching.csv"))
    if not beh:
        return _missing(pdf, "Causal test: does the model USE the forcing rule?", out_dir)
    fig = _page(pdf, "Causal test: one neighbour-color token flips DD's answer")
    ax1, ax2 = fig.subplots(1, 2)
    n = len(beh)
    clean = sum(r["clean_correct"] == "True" for r in beh) / n
    flip = sum(r["corrupt_flipped"] == "True" for r in beh) / n
    both = sum(r["clean_correct"] == "True" and r["corrupt_flipped"] == "True" for r in beh) / n
    chance = 0.25
    bars = ax1.bar(["clean\ncorrect", "corrupt\nflipped", "both\n(uses rule)"],
                   [clean, flip, both], color=["#2ca02c", "#d62728", "#1f77b4"])
    ax1.axhline(chance, ls="--", color="#888888", label=f"chance = {chance:.2f}")
    for b, v in zip(bars, [clean, flip, both]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=14)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("rate", fontsize=SMALL)
    ax1.set_title(f"behavior over {n} counterfactuals", fontsize=MED)
    ax1.legend(fontsize=12)

    if patch:
        sites = list(dict.fromkeys(r["site"] for r in patch))
        layers = sorted({int(r["layer"]) for r in patch})
        grid = np.full((len(sites), len(layers)), np.nan)
        for i, s in enumerate(sites):
            for j, L in enumerate(layers):
                vals = [_f(r, "recovery") for r in patch
                        if r["site"] == s and int(r["layer"]) == L]
                if vals:
                    grid[i, j] = np.nanmean(vals)
        im = ax2.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(layers)), [str(L) for L in layers], fontsize=11)
        ax2.set_yticks(range(len(sites)), sites, fontsize=12)
        ax2.set_xlabel("layer patched", fontsize=SMALL)
        ax2.set_title("recovery of the clean answer\n(1.0 = deduction carried here)", fontsize=14)
        fig.colorbar(im, ax=ax2)
    else:
        ax2.text(0.5, 0.5, "flip rate at chance:\nno circuit to localize", ha="center",
                 va="center", fontsize=MED, color="#999999")
        ax2.axis("off")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_when(pdf, series_probe_csv, series_csv):
    rows, cap = _read(series_probe_csv), _read(series_csv)
    if not rows:
        return _missing(pdf, "When during training does DD appear?", series_probe_csv)

    def epoch_of(ck):
        digits = "".join(c for c in ck if c.isdigit())
        return int(digits) if digits else -1

    fig = _page(pdf, "When during training does DD structure appear?")
    ax = fig.subplots()
    for target, color, label in [("known_mask", "#2ca02c", "known_mask (bookkeeping)"),
                                 ("frontier_mask", "#d62728", "frontier_mask (DD predicate)")]:
        pts = {}
        for r in rows:
            if r["target"] != target:
                continue
            pts.setdefault(epoch_of(r["checkpoint"]), []).append(_f(r, "balanced_accuracy"))
        # best layer per snapshot: "is it anywhere in the model yet"
        xs = sorted(pts)
        ax.plot(xs, [max(pts[x]) for x in xs], "o-", color=color, lw=2.4, ms=7,
                label=label + " (best layer)", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("training epoch", fontsize=SMALL)
    ax.set_ylabel("probe balanced accuracy", fontsize=SMALL)
    ax.grid(alpha=0.3)
    if cap:
        tw = ax.twinx()
        tw.plot([_f(r, "epoch") for r in cap], [_f(r, "solve_rate") for r in cap],
                "^--", color="#1f77b4", lw=2, label="solve_rate", rasterized=True)
        tw.set_ylabel("solve_rate", fontsize=SMALL, color="#1f77b4")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = tw.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=12, loc="lower right")
    else:
        ax.legend(fontsize=12, loc="lower right")
    fig.tight_layout(rect=[0, 0.02, 1, 0.91])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_summary(pdf, lines):
    fig = _page(pdf, "Summary and what it means")
    fig.text(0.06, 0.78, "\n\n".join(lines), fontsize=15, va="top", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dd-curve", default=f"{SCRATCH}/spin_gnn/sudoku_dd_removal_small_n_2026-10-19/removal_path_T_vs_rho.csv")
    ap.add_argument("--transition", default=f"{SCRATCH}/spin_gnn/sudoku_hint_density_transition_2026-10-19/transition.csv")
    ap.add_argument("--probes", default=f"{SCRATCH}/spin_gnn/sudoku_probes_2026-10-19/probes.csv")
    ap.add_argument("--sae", default=f"{SCRATCH}/spin_gnn/sudoku_sae_2026-10-19_sparse/sae_summary.csv")
    ap.add_argument("--circuits", default=f"{SCRATCH}/spin_gnn/sudoku_circuits_2026-10-19/d0128_L12_v3")
    ap.add_argument("--algorithm-id", default=f"{SCRATCH}/spin_gnn/sudoku_algorithm_id_2026-10-19/d0128_h04_L12_ff01024_lr1e-03_pre.csv")
    ap.add_argument("--series-probes", default=f"{SCRATCH}/spin_gnn/sudoku_series_probes_2026-10-19/series_probes.csv")
    ap.add_argument("--series", default=f"{SCRATCH}/spin_gnn/sudoku_checkpoint_series_2026-10-19/series.csv")
    ap.add_argument("--headline", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary_path = os.path.join(args.circuits, "summary.json")
    circ = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    probes = _read(args.probes)

    def peak(target):
        vals = [_f(r, "balanced_accuracy") for r in probes
                if r["target"] == target and "d0128" in r["checkpoint"]]
        return max(vals) if vals else float("nan")

    headline = args.headline or (
        "The transition IS there: solve_rate 0.99 -> 0.44 as DD's deduction depth goes 1 -> 6.\n"
        "But the model is not running DD -- it tracks which nodes are colored "
        f"(known_mask {peak('known_mask'):.2f}) rather than which are deducible "
        f"(frontier_mask {peak('frontier_mask'):.2f}, control 0.57), and loses the\n"
        "head-to-head vote against a sorted-order heuristic."
    )
    sub = (f"git {os.popen('git rev-parse --short HEAD').read().strip()}  |  "
           f"best checkpoint d0128_h04_L12_ff01024 @900ep (agg_solve_rate=0.767)  |  "
           f"N=24, d=5, q=4, removal-path hints")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        page_title(pdf, headline, sub)
        page_where_transition(pdf, args.dd_curve)
        page_model_transition(pdf, args.transition)
        page_probes(pdf, args.probes)
        page_algorithm_id(pdf, args.algorithm_id)
        page_sae(pdf, args.sae)
        page_circuits(pdf, args.circuits)
        page_when(pdf, args.series_probes, args.series)
        page_summary(pdf, [
            "1. The hint-density transition is NOT in independent random hints -- that ensemble is "
            "subcritical at every density, with recovery identical from N=24 to N=384. It is in the "
            "removal-path ensemble the models already train on, as a divergence of DD's deduction "
            "time T as rho approaches the floor rho_min = 0.478 (N-independent).",
            "2. The gradient exists at N=24, so the transition is testable on existing checkpoints. "
            "This retires the earlier plan to scale N to ~1700 tokens. MEASURED: solve_rate falls "
            "0.99 -> 0.44 monotonically as DD's depth rises T=1.0 -> 6.0, with truncated_rate = 0.000 "
            "in every bin, so it is not a generation-budget artifact. But rho and T are collinear here, "
            "so which of the two controls the collapse is NOT yet resolved -- that needs equal-rho "
            "instances binned by T.",
            f"3. Probes: bookkeeping (known_mask, {peak('known_mask'):.2f}) is computed early and "
            f"strongly; DD's deducibility predicate (frontier_mask, {peak('frontier_mask'):.2f} vs a "
            f"0.57 control) is barely present. next_color is computed late (0.46 at L7 -> 0.65 at L12).",
            "4. Causal test: " + (
                f"the model does NOT apply the forcing rule at decision level (flip rate "
                f"{circ.get('corrupt_flipped', float('nan')):.2f} vs 0.25 chance), but neighbour-color "
                f"information does flow: the corruption moves the logit gap by "
                f"{circ.get('logit_effect', float('nan')):.1f}. Patching localizes that flow to "
                f"edited_hint (L0-5) -> graph row of the RECOLORED NEIGHBOUR (L6-9) -> prediction "
                f"position (L8-11). The row of the node being predicted carries ~0.00 throughout, so "
                f"this is color-to-node binding, not DD's neighbour-list join."
                if circ else "pending."),
            "5. Honest gaps: head ablation in this run was a no-op (fixed, rerunning as 48585025); "
            "'when during training' awaits the checkpoint series; and the scratchpad variant came in "
            "at agg_solve_rate 0.740 vs the 0.767 baseline, i.e. no improvement.",
        ])
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size:.3f} MB)")


if __name__ == "__main__":
    main()
