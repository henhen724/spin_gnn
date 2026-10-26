"""Does the network depth a solver needs track the depth of the deduction?

Results notebook for two runs that finished together on 2026-10-25:

  49523504  the num_layers=3 cell at N=48 that the layer scan (49509466)
            skipped, plus a depth eval over all six N=48 cells and a
            shared-pool pass over all seven N=48 checkpoints at n=1500.
  49524030  a single-cell feasibility probe at N=96 -- the N=48 recipe with
            N as the only change -- written to find out whether L* can be
            measured at N=96 at all before committing 36 h to a scan there.

Provenance. Both runs executed with spin_gnn/ at commit 072f962, which is
byte-identical to f8b15bc for every file under spin_gnn/ (checked with git
diff --name-only); the lr-schedule commit 38d503a came afterwards and touched
none of this. seed=0 for every training run (sudoku_hparam_sweep --seed
default), eval pool seed_start=5e10, depth-eval pool seed 20e9+. Config for
every cell: d_model=256, nhead=4, dim_feedforward=1024, norm_first=True,
lr=1e-3 constant, batch=32, degree=5, n_colors=4, rho=0.3, rho_max=0.95,
hint_mode=removal, 150 epochs and n_train=20000 except where a page says
otherwise (the N=48 baseline 49433453 is 300 epochs).

Metric of record throughout is the shared pool (sudoku_agg_eval, seed_start
5e10, disjoint from every training set, instance i at seed_start+i so a larger
n is a strict superset), with Wilson intervals and paired McNemar exact tests.
Per-run 300-instance agg_solve_rate appears only where the text says so.

Run as:  python -m notebooks.make_depth_vs_n_deck
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style
from spin_gnn.sudoku_agg_eval import mcnemar_exact_p

ROOT = os.environ.get("SCRATCH", "") + "/spin_gnn"
OUT = os.path.join(ROOT, "notebooks", "sudoku_depth_vs_n_2026-10-25")
PDF = os.path.join(OUT, "does_depth_track_deduction.pdf")
THRESH = 0.90

DEPTH = {
    24: f"{ROOT}/sudoku_layer_scan_2026-10-24_n24/depth_by_layers.csv",
    48: f"{ROOT}/sudoku_layer_scan_2026-10-24_n48/depth_by_layers_with_L03.csv",
    96: f"{ROOT}/sudoku_n96_baseline_2026-10-25/depth_n96.csv",
}
POOL = {
    24: f"{ROOT}/sudoku_ff_depth_sharedpool_2026-10-25/sharedpool_n1500.csv",
    48: f"{ROOT}/sudoku_n48_sharedpool_2026-10-25/sharedpool_n1500.csv",
    96: f"{ROOT}/sudoku_n96_baseline_2026-10-25/sharedpool_n1500.csv",
}
N48_NPZ = f"{ROOT}/sudoku_n48_sharedpool_2026-10-25/sharedpool_n1500_per_instance.npz"
# sudoku_agg_eval disambiguates same-tag checkpoints from different runs by
# appending " [run_dir]", and the npz carries those same strings, so the two
# N=48 L=12 cells stay distinguishable in both.
L12_150 = "d0256_h04_L12_ff01024_lr1e-03_pre [sudoku_layer_scan_2026-10-24_n48]"
L12_300 = "d0256_h04_L12_ff01024_lr1e-03_pre [sudoku_n48_baseline_2026-10-24]"
HIST = {
    "N=48 L12, 150 ep": f"{ROOT}/sudoku_layer_scan_2026-10-24_n48/d0256_h04_L12_ff01024_lr1e-03_pre/history.json",
    "N=48 L12, 300 ep": f"{ROOT}/sudoku_n48_baseline_2026-10-24/d0256_h04_L12_ff01024_lr1e-03_pre/history.json",
    "N=96 L12, 150 ep": f"{ROOT}/sudoku_n96_baseline_2026-10-25/d0256_h04_L12_ff01024_lr1e-03_pre/history.json",
    "N=24 L12, 150 ep": f"{ROOT}/sudoku_layer_scan_2026-10-24_n24/d0256_h04_L12_ff01024_lr1e-03_pre/history.json",
}


# ---------------------------------------------------------------- loading

def layers_of(checkpoint):
    return int(re.search(r"_L(\d+)_", checkpoint).group(1))


def load_depth(n_nodes):
    """[(L, rho_requested, T, n, realized_rho, solve_rate)] for one problem size."""
    rows = []
    for r in csv.DictReader(open(DEPTH[n_nodes])):
        rows.append((layers_of(r["checkpoint"]), float(r["rho_requested"]), int(r["T"]),
                     int(r["n"]), float(r["realized_rho"]), float(r["solve_rate"])))
    return rows


def series(rows, rho, L):
    s = sorted((t, sr, n) for l, rq, t, n, _, sr in rows if l == L and abs(rq - rho) < 1e-9
               for sr, n in [(sr, n)])
    return [t for t, _, _ in s], [sr for _, sr, _ in s], [n for _, _, n in s]


def l_star(rows, rho, T):
    """Smallest num_layers reaching THRESH on depth-T instances, or None."""
    ok = [l for l, rq, t, _, _, sr in rows
          if t == T and abs(rq - rho) < 1e-9 and sr >= THRESH]
    return min(ok) if ok else None


def load_pool(n_nodes):
    """{short label: row dict} keyed so N=48's two L=12 cells stay distinguishable."""
    out = {}
    for r in csv.DictReader(open(POOL[n_nodes])):
        ck = r["checkpoint"].split("/")[-1]
        out[ck] = {k: r[k] for k in r}
        out[ck]["L"] = layers_of(ck)
        out[ck]["solve_rate"] = float(r["solve_rate"])
        out[ck]["lo"], out[ck]["hi"] = float(r["wilson_lo"]), float(r["wilson_hi"])
    return out


def load_hist(key):
    h = json.load(open(HIST[key]))
    return ([e["epoch"] for e in h], [e["train_nll"] for e in h], [e["nll"] for e in h])


def paired(masks, names, a, b):
    """(delta, a_won, b_won, exact p) for a against b on the same 1500 instances."""
    x, y = masks[names.index(a)].astype(bool), masks[names.index(b)].astype(bool)
    won, lost = int((x & ~y).sum()), int((y & ~x).sum())
    return x.mean() - y.mean(), won, lost, mcnemar_exact_p(lost, won)


# ---------------------------------------------------------------- pages

def page_title(book, lstar_table):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.80, "Does network depth track deduction depth?", ha="center", size=BIG + 7)
    fig.text(0.5, 0.715, "L*(T) measured at three problem sizes  ·  49523504, 49524030  ·  2026-10-25",
             ha="center", size=MED, color="0.35")

    flat = [f"{tag}: L* = {v}" for tag, v, _ in lstar_table if v is not None]
    n_bins = sum(len(ts) for _, v, ts in lstar_table if v is not None)
    fig.text(0.5, 0.565,
             "No. In every series where L* can be measured, it is FLAT in T.",
             ha="center", size=BIG, color="tab:blue")
    fig.text(0.5, 0.465,
             "   ·   ".join(flat),
             ha="center", size=MED + 1)
    fig.text(0.5, 0.375,
             f"{n_bins} depth bins, {len(flat)} series, 0 of them rising. What moves L* is density\n"
             "and problem size -- never the number of deduction rounds inside a series.",
             ha="center", size=MED, color="0.25")

    fig.text(0.5, 0.185,
             "New this run: the num_layers=3 cell at N=48 (the one the scan skipped), which is what\n"
             "lets L* be compared across N at all; and an N=96 probe that found the recipe's ceiling\n"
             "below the 0.90 threshold, so L* is not measurable there and the 36 h scan stays unsubmitted.",
             ha="center", size=SMALL + 1, color="0.4")
    book.savefig(fig)


def page_lstar_table(book, rows24, rows48, rows96):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("L*(T): the smallest num_layers reaching solve_rate $\\geq$ 0.90 on depth-T instances",
                 pad=24)

    body = [["size", "density", "T bins", "L grid searched", "L*(T)", "flat in T?"]]
    for n, rows in ((24, rows24), (48, rows48), (96, rows96)):
        Ls = sorted({l for l, *_ in rows})
        for rho in sorted({rq for _, rq, *_ in rows}, reverse=True):
            Ts = sorted({t for l, rq, t, *_ in rows if abs(rq - rho) < 1e-9})
            vals = [l_star(rows, rho, t) for t in Ts]
            shown = ", ".join("--" if v is None else str(v) for v in vals)
            uniq = set(vals)
            if uniq == {None}:
                verdict = "not measurable"
            elif len(uniq) > 1:
                verdict = "NO"
            elif len(Ls) == 1:
                # Flat, but vacuously: one L was trained, so there is no knee to find.
                verdict = "flat (grid of 1)"
            else:
                verdict = "yes, flat"
            body.append([f"N={n}", f"rho={rho:g}", f"{Ts[0]}-{Ts[-1]}",
                         "{" + ",".join(str(x) for x in Ls) + "}", shown, verdict])

    tab = ax.table(cellText=body[1:], colLabels=body[0], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(MED)
    tab.scale(1, 2.5)
    for j in range(len(body[0])):
        tab[0, j].set_text_props(weight="bold")
    for i, r in enumerate(body[1:], start=1):
        colour = {"yes, flat": "tab:blue", "flat (grid of 1)": "0.45",
                  "NO": "tab:orange", "not measurable": "tab:red"}[r[-1]]
        tab[i, len(r) - 1].set_text_props(color=colour, weight="bold")

    footer(fig,
           "Read down the L*(T) column: within every row it is a constant. The N=96 rho=0.60 row is the "
           "honest hole -- no configuration measured there reaches 0.90 in any bin, so L* is undefined "
           "rather than large, and the dash is not a small number. The N=96 rows also searched a grid of "
           "one (L=12 only), so N=96's rho=0.75 entry is an upper bound, not a located knee: all it says "
           "is that twelve layers suffice. The cross-N comparison therefore rests on N=24 against N=48, "
           "where the full grid was searched at both sizes.")
    fig.tight_layout(rect=(0, 0.20, 1, 0.97))
    book.savefig(fig)


def page_curves(book, rows24, rows48):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    palette = {1: "0.65", 2: "tab:red", 3: "tab:blue", 4: "tab:green", 6: "tab:purple",
               8: "tab:orange", 12: "tab:brown"}
    bin_n = []
    for ax, (n, rows, rho) in zip(axes, [(24, rows24, 0.60), (48, rows48, 0.60)]):
        for L in sorted({l for l, *_ in rows}):
            T, sr, nn = series(rows, rho, L)
            if not T:
                continue
            bin_n += nn
            ax.plot(T, sr, "o-", color=palette[L], lw=2.4, ms=8, label=f"L={L}")
        ax.axhline(THRESH, color="k", ls=":", lw=2)
        star = {l_star(rows, rho, t) for t in sorted({t for l, rq, t, *_ in rows if abs(rq - rho) < 1e-9})}
        ax.set_title(f"N={n}, rho={rho:g}   ·   L* = {star.pop()} at every T")
        ax.set_xlabel("T, deduction rounds the DD trace takes")
        ax.grid(alpha=0.3)
    axes[0].text(0.015, THRESH + 0.02, "0.90 threshold", transform=axes[0].get_yaxis_transform(),
                 size=SMALL, color="0.25")
    axes[0].set_ylabel("solve_rate at fixed hint density")
    axes[0].set_ylim(-0.03, 1.05)
    axes[1].legend(loc="lower left", ncol=2, framealpha=0.9)

    footer(fig,
           "The curves that matter are the ones straddling the threshold. At N=24 that is L=3 (blue), "
           "which sits above 0.90 at every T from 2 to 5; at N=48 it is L=4 (green), above at every T from "
           "3 to 7, while L=3 has dropped to 0.827-0.875 -- below the line at every depth, and by a nearly "
           "constant margin rather than a widening one. That is the shape of the whole result: when an "
           "extra layer does become necessary, it is necessary UNIFORMLY across T. A serial mechanism "
           "would instead show L=3 clearing the bar at shallow T and failing at deep T, crossing the line "
           f"somewhere inside the plot. Nothing crosses. Caveat: each point is n={min(bin_n)}-{max(bin_n)} "
           "instances, so a single bin is resolved to roughly +-0.06 near 0.90 and no individual "
           "crossing would be trustworthy; what carries the claim is that a whole series sits on one "
           "side of the line, which is 4-5 bins agreeing at once.")
    fig.tight_layout(rect=(0, 0.19, 1, 0.96))
    book.savefig(fig)


def page_n48_ladder(book, pool48, masks, names):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    scan = {v["L"]: v for k, v in pool48.items() if "layer_scan" in v["checkpoint"]}
    Ls = sorted(scan)
    y = [scan[L]["solve_rate"] for L in Ls]
    lo = [scan[L]["solve_rate"] - scan[L]["lo"] for L in Ls]
    hi = [scan[L]["hi"] - scan[L]["solve_rate"] for L in Ls]
    colours = ["tab:blue" if L != 3 else "tab:red" for L in Ls]
    ax.errorbar(Ls, y, yerr=[lo, hi], fmt="none", ecolor="0.4", capsize=6, lw=2, zorder=1)
    ax.scatter(Ls, y, s=150, c=colours, zorder=2)
    for L, v in zip(Ls, y):
        # Labels above only for the cells far below the 0.90 line; the rest would
        # collide with it.
        ax.annotate(f"{v:.4f}", (L, v), textcoords="offset points",
                    xytext=(0, 18 if v < 0.5 else -32), ha="center", size=SMALL + 1)
    ax.axhline(THRESH, color="k", ls=":", lw=2)
    ax.text(1.1, THRESH + 0.02, "0.90 threshold", size=SMALL, color="0.25", ha="left")
    ax.set_xlabel("num_layers")
    ax.set_ylabel("shared-pool solve_rate, n=1500")
    ax.set_title("The N=48 ladder on one pool of 1500 instances  ·  red is the cell added by 49523504")
    ax.set_xticks(Ls)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=0.3)

    d_b, w_b, l_b, p_b = paired(masks, names, L12_150, L12_300)
    footer(fig,
           "Three pre-registrations, all resolved. (a) L=3 scores below 0.90, i.e. L*(48)=4 and the knee "
           f"rises with N: 0.8293 [0.8095, 0.8475], clear of the line with the whole interval to spare -- "
           "and it was a genuine 50/50 written down in advance. (b) the 150-epoch L=12 cell beats the "
           f"300-epoch one, paired p<0.05: {d_b:+.4f}, {w_b}-{l_b} discordant, p={p_b:.2g}. (c) L=8 ties "
           "L=12: +0.0140, p=0.076, inside the 0.02 band and with L=8 the nominally better of the two, "
           "which is consistent with L=12 overfitting. Note what the ladder does past the knee -- it is "
           "flat from L=4 to L=12 to within 0.044, so the N=48 story is a knee, not a slope.")
    fig.tight_layout(rect=(0, 0.19, 1, 0.96))
    book.savefig(fig)


def page_epochs(book, pool48, masks, names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for key, colour in [("N=48 L12, 150 ep", "tab:blue"), ("N=48 L12, 300 ep", "tab:red")]:
        ep, tr, va = load_hist(key)
        mn = int(np.argmin(va))
        axes[0].plot(ep, va, color=colour, lw=2.4, label=f"{key}  (min {va[mn]:.4f} @ep{mn})")
        axes[0].plot(mn, va[mn], "v", color=colour, ms=12)
        axes[0].plot(ep[-1], va[-1], "o", color=colour, ms=11, mfc="none", mew=2.5)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation nll")
    axes[0].set_title("Same minimum, different endpoint")
    axes[0].legend(loc="upper right", fontsize=SMALL)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(1.18, 1.45)

    d, w, l, p = paired(masks, names, L12_150, L12_300)
    cells = [pool48[L12_150], pool48[L12_300]]
    vals = [c["solve_rate"] for c in cells]
    bars = axes[1].bar(["150 epochs", "300 epochs"], vals, color=["tab:blue", "tab:red"], width=0.55)
    for b, c in zip(bars, cells):
        v = c["solve_rate"]
        axes[1].errorbar(b.get_x() + b.get_width() / 2, v,
                         yerr=[[v - c["lo"]], [c["hi"] - v]], ecolor="0.2", capsize=8, lw=2.2)
        axes[1].text(b.get_x() + b.get_width() / 2, 0.80, f"{v:.4f}", ha="center", size=MED)
    axes[1].set_ylim(0.75, 1.0)
    axes[1].set_ylabel("shared-pool solve_rate, n=1500")
    axes[1].set_title(f"Paired on the same instances: {d:+.4f}, p={p:.2g}")
    axes[1].grid(alpha=0.3, axis="y")

    footer(fig,
           "The layer scan's pre-registered gate FAILED: its L=12 cell was required to reproduce the "
           "300-epoch baseline's 0.9033 within [0.865, 0.932] or the scan was to be reported and stopped "
           "as confounded with epoch budget. It came in outside -- but upward, 150 epochs beating 300, "
           f"and on the shared pool the gap is {d:+.4f} with {w}-{l} discordant at p={p:.2g}, so it is "
           "real and not 300-instance noise. The left panel is the cause: both runs reach the same "
           "validation minimum (triangles, 1.2146@ep46 and 1.2164@ep36, so training is reproducible), "
           "then both overfit, and the 300-epoch run ends further from its minimum (circles, +0.1413 "
           "against +0.0497) because save_checkpoint keeps the final epoch, not the best one. That makes "
           "the confound cross-JOB, not within-scan -- every cell in the scan and the whole N=24 arm "
           "share the 150-epoch budget -- so it invalidates using the baseline as a reproducibility "
           "anchor, not the L* comparison. It is also the first time final-epoch checkpointing has been "
           "shown to cost measurable solve_rate rather than just val_nll.")
    fig.tight_layout(rect=(0, 0.30, 1, 0.95))
    book.savefig(fig)


def page_n96_ceiling(book, rows96, pool96):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, rho in zip(axes, [0.75, 0.60]):
        T, sr, nn = series(rows96, rho, 12)
        colours = ["tab:blue" if v >= THRESH else "tab:red" for v in sr]
        ax.bar(T, sr, color=colours, width=0.65)
        # Inside the bars: above them the labels would sit on the 0.90 line, which
        # is the one thing on this page that has to stay legible.
        for t, v in zip(T, sr):
            ax.text(t, v - 0.05, f"{v:.3f}", ha="center", size=SMALL, color="w", weight="bold")
        ax.axhline(THRESH, color="k", ls=":", lw=2.4)
        clears = sum(v >= THRESH for v in sr)
        ax.set_title(f"rho={rho:g}  ·  {clears} of {len(sr)} bins clear 0.90")
        ax.set_xlabel("T, deduction rounds  (n per bin below)")
        ax.set_xticks(T)
        ax.set_xticklabels([f"{t}\nn={m}" for t, m in zip(T, nn)])
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("solve_rate, N=96, L=12")
    axes[0].set_ylim(0, 1.08)

    agg = list(pool96.values())[0]
    footer(fig,
           f"The probe did its job by failing its own gate. It pre-registered that the recipe would clear "
           f"0.90 aggregate at N=96 so that L* would be measurable; it scores {agg['solve_rate']:.4f} "
           f"[{agg['lo']:.4f}, {agg['hi']:.4f}] on the shared pool, and per bin it clears the threshold "
           "everywhere at rho=0.75 and NOWHERE at rho=0.60. Since L*(T) is defined by that threshold, a "
           "36 h layer scan at N=96 would have returned a column of sub-threshold numbers at rho=0.60 "
           "with no L* in it -- and it would have run at half the data and two thirds the epochs, so the "
           "result would have been indistinguishable from its own budget. It stays unsubmitted. The other "
           "two pre-registrations held: the depth window widened again as intended, rho=0.60 now spanning "
           "T=4-9 against 3-7 at N=48 and 2-5 at N=24, which is why N=96 was worth probing at all.")
    fig.tight_layout(rect=(0, 0.24, 1, 0.95))
    book.savefig(fig)


def page_underfit(book):
    # Not sharey: the three sizes sit at different loss scales, and forcing one
    # axis pushed N=24 off the bottom of the panel entirely.
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    for ax, key in zip(axes, ["N=24 L12, 150 ep", "N=48 L12, 150 ep", "N=96 L12, 150 ep"]):
        ep, tr, va = load_hist(key)
        ax.plot(ep, tr, color="tab:green", lw=2.4, label="train nll")
        ax.plot(ep, va, color="tab:purple", lw=2.4, label="val nll")
        mn = int(np.argmin(va))
        ax.plot(mn, va[mn], "v", color="tab:purple", ms=13)
        gap = tr[-1] - va[-1]
        ax.set_title(f"{key.split(',')[0]}\nfinal train-val gap {gap:+.4f}", size=MED)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        floor = min(min(tr), min(va))
        ax.set_ylim(floor - 0.04, floor + 0.45)
        ax.text(0.97, 0.95, f"val min @ep{mn}\nfinal-min {va[-1]-va[mn]:+.4f}",
                transform=ax.transAxes, ha="right", va="top", size=SMALL, color="0.3")
    axes[0].set_ylabel("nll")
    axes[0].legend(loc="center right", fontsize=SMALL)

    footer(fig,
           "Why the N=96 ceiling is probably NOT a data ceiling, and why that contradicts what the probe "
           "itself pre-registered as the next move. At N=24 and N=48 the saved model sits well past its "
           "validation minimum with train pulled below val -- the overfitting signature, where more data "
           "is the classical remedy and finding 6 says it has not saturated even at 80000. At N=96 the "
           "picture inverts: the minimum is at epoch 129 of 150, the last 30 epochs span 0.0043, and "
           "train ends 0.029 BELOW val, i.e. the model never fits its own training set. There is no "
           "generalization gap for more data to close. What there is instead is a constant learning "
           "rate: every run in this project used Adam at a fixed lr with no scheduler anywhere in the "
           "codebase. 49565433 tests cosine decay against 4x data on one pool, so the lever the probe "
           "pre-registered is still measured rather than dropped on an argument.")
    fig.tight_layout(rect=(0, 0.26, 1, 0.94))
    book.savefig(fig)


def page_provenance(book, pool24, pool48, pool96):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("Against the previous best on the same eval, and what is not claimed", pad=22)

    def cell(label, src, key, n_train, epochs, where):
        v = src[key]
        return [label, f"{v['solve_rate']:.4f}", f"[{v['lo']:.4f}, {v['hi']:.4f}]",
                n_train, epochs, where]

    rows = [
        cell("N=24, L=4  (best on that ladder)", pool24, "d0256_h04_L04_ff01024_lr1e-03_pre",
             "20000", "150", "scan n24"),
        cell("N=48, L=8  (best at this size)", pool48,
             "d0256_h04_L08_ff01024_lr1e-03_pre [sudoku_layer_scan_2026-10-24_n48]",
             "20000", "150", "scan n48"),
        cell("N=48, L=3  NEW", pool48,
             "d0256_h04_L03_ff01024_lr1e-03_pre [sudoku_layer_scan_2026-10-24_n48]",
             "20000", "150", "49523504"),
        cell("N=96, L=12  NEW", pool96, "d0256_h04_L12_ff01024_lr1e-03_pre",
             "20000", "150", "49524030"),
        ["project best: N=24, L=12, 4x data", "0.9960", "[0.9913, 0.9982]", "80000", "300",
         "10-22 grid"],
    ]
    tab = ax.table(cellText=rows,
                   colLabels=["cell", "solve_rate", "Wilson 95%", "n_train", "epochs", "source"],
                   colWidths=[0.30, 0.13, 0.19, 0.11, 0.10, 0.13],
                   loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(MED - 2)
    tab.scale(1, 2.0)
    for j in range(6):
        tab[0, j].set_text_props(weight="bold")
    for i in (3, 4):
        tab[i, 0].set_text_props(color="tab:red", weight="bold")

    fig.text(0.5, 0.56,
             "Neither new cell is an improvement, and neither was meant to be: both were measured on "
             "purpose below the project\nbest -- L=3 at N=48 to locate a knee from underneath, L=12 at "
             "N=96 to find out where a ceiling is. The 0.9960 row is a\ndifferent problem size at 4x the "
             "data and 300 epochs, and is listed only so the two new numbers are not read as a "
             "regression.",
             size=SMALL + 2, color="0.2", ha="center", va="top")

    fig.text(0.5, 0.40,
             "commit 072f962 for both runs (identical to f8b15bc under spin_gnn/; 38d503a came after)\n"
             "seed=0 all training  ·  pool seed_start=5e10, n=1500, instance i at seed_start+i\n"
             "d_model=256, nhead=4, ff=1024, pre-norm, lr=1e-3 constant, batch=32\n"
             "degree=5, n_colors=4, rho=0.3, rho_max=0.95, removal hints\n"
             "49523504 ran 00:35:25, 49524030 ran 02:59:15, both ExitCode 0:0",
             size=SMALL, color="0.4", ha="center", va="top", family="monospace")

    footer(fig,
           "One caveat on 'same eval': the pool is instance i at seed_start+i for the CONFIGURED problem "
           "size, so rows at equal N -- including the 0.9960 project best, same seed_start 5e10 and "
           "n=1500, from the 10-22 grid -- are literally the same 1500 instances and are paired-testable, "
           "while rows at different N share only the protocol. Every paired p-value in this deck is "
           "within one N. Open on purpose: N=96's L grid is one point, so its rho=0.75 entry bounds L* "
           "above by 12 rather than locating it, and the cross-N claim rests on N=24 against N=48 alone. "
           "L*(48)=4 at rho=0.60 but 3 at rho=0.75, so the level tracks density as well as size -- "
           "'depth tracks graph size' is one data point at one density, not a law. What is solid is the "
           "negative: 15 bins across 4 fully searched series, none of them rising in T.")
    fig.tight_layout(rect=(0, 0.21, 1, 0.95))
    book.savefig(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    rows24, rows48, rows96 = load_depth(24), load_depth(48), load_depth(96)
    pool24, pool48, pool96 = load_pool(24), load_pool(48), load_pool(96)

    npz = np.load(N48_NPZ, allow_pickle=True)  # names is an object array
    masks = npz["solved"]
    names = [str(s) for s in npz["names"]]

    lstar_table = []
    for n, rows in ((24, rows24), (48, rows48)):
        for rho in sorted({rq for _, rq, *_ in rows}, reverse=True):
            Ts = sorted({t for l, rq, t, *_ in rows if abs(rq - rho) < 1e-9})
            vals = {l_star(rows, rho, t) for t in Ts}
            lstar_table.append((f"N={n} rho={rho:g}", vals.pop() if len(vals) == 1 else None, Ts))

    with PdfPages(PDF) as pdf:
        book = PdfAndPng(pdf, OUT)
        page_title(book, lstar_table)
        page_lstar_table(book, rows24, rows48, rows96)
        page_curves(book, rows24, rows48)
        page_n48_ladder(book, pool48, masks, names)
        page_epochs(book, pool48, masks, names)
        page_n96_ceiling(book, rows96, pool96)
        page_underfit(book)
        page_provenance(book, pool24, pool48, pool96)
    print(f"wrote {PDF} ({os.path.getsize(PDF)/1e6:.2f} MB, {book.n} pages)")


if __name__ == "__main__":
    main()
