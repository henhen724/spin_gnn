"""Every figure in the stage-X deck, from the two arms' own artifacts.

Written while the cells were still training, so that when they land the deck
is one command away and no plot has to be invented under time pressure. It
reads only what the jobs write:

  stage S  <SCAN_S>/t0{0,1,2}_.../<ckpt>/history.json     per-epoch
           <SCAN_S>/rho_curve_t0{0,1,2}_51853969_*.csv    the 14-point grid
  stage X  <SCAN_X>/t0{0,1,2}_.../<ckpt>/history.json     per-epoch
           <SCAN_X>/rho_curve_t0{0,1,2}_52285051_*.csv    the same grid
           <SCAN_X>/determinacy_control_stageS.csv        finding 73
           runs/2026-11-13_stagex_guard/determinacy_both_arms.csv

history.json only exists once a cell finishes. While a cell is still running
the same per-epoch numbers are in its Slurm log, one line per epoch, so
--from-logs reads those instead and the figures can be drawn mid-run. The two
sources are the same numbers printed by the same trainer; the flag is
recorded on every figure it produces so a mid-run plot is never mistaken for
a final one.

PAIRING. Every comparison here is at a MATCHED EPOCH and a MATCHED SEED.
Stage X seed s is only ever compared against stage S seed s, because the
seed fixes the instances and the initialisation; comparing arm means would
hide that the two arms' seed-2 cells leave the plateau at different epochs.

Figures are small multiples rather than six lines on one axis: the deck's
rule is at most three series per plot, and the per-seed pairing is the point.
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from slides import SERIES, MUTED, label_line, series_colors, style_axes

_SCR = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_S = os.path.join(_SCR, "spin_gnn", "sudoku_n48_tiedwidth_S_2026-11-10")
SCAN_X = os.path.join(_SCR, "spin_gnn", "sudoku_n48_determinacy_X_2026-11-13")
GUARD_RUN = os.path.join(_SCR, "spin_gnn", "runs", "2026-11-13_stagex_guard")

S_CELLS = [
    "t00_d0320_ff01024_h04_tied_no_mlp_L16_s0/"
    "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied",
    "t01_d0320_ff01024_h04_tied_no_mlp_L16_s1/"
    "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_s01",
    "t02_d0320_ff01024_h04_tied_no_mlp_L16_s2/"
    "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_s02",
]
X_CELLS = [
    "t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_s0/"
    "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1",
    "t01_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_s1/"
    "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_s01",
    "t02_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_s2/"
    "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_s02",
]
X_LOGS = [os.path.join(_SCR, "spin_gnn", "sud2N48DetermX_52285051_%d.out" % i)
          for i in range(3)]

# One line per epoch, exactly as the trainer prints it. det_bce and
# val_det_auroc are absent from a lambda=0 cell's line and that is not an
# error -- the term was not computed there.
_EPOCH = re.compile(
    r"epoch (\d+): train_nll=([\d.]+) val_nll=([\d.]+) val_accuracy=([\d.]+)"
    r"(?: lr=\S+)?(?: det_bce=([\d.]+))?(?: val_det_auroc=([\d.]+))?")


def history_from_json(path):
    with open(path) as f:
        h = json.load(f)
    return [{"epoch": r["epoch"], "train_nll": r["train_nll"],
             "nll": r["nll"], "accuracy": r["accuracy"],
             "det_bce": r.get("train_determinacy_bce"),
             "auroc": r.get("determinacy_auroc")} for r in h]


def history_from_log(path):
    out = []
    with open(path) as f:
        for line in f:
            m = _EPOCH.match(line.strip())
            if m:
                out.append({
                    "epoch": int(m.group(1)), "train_nll": float(m.group(2)),
                    "nll": float(m.group(3)), "accuracy": float(m.group(4)),
                    "det_bce": float(m.group(5)) if m.group(5) else None,
                    "auroc": float(m.group(6)) if m.group(6) else None})
    return out


def load_arm(scan, cells, logs=None, from_logs=False):
    """Per-seed histories. Falls back to the Slurm log only when asked, so a
    stale log can never silently stand in for a finished cell."""
    arm = []
    for i, c in enumerate(cells):
        j = os.path.join(scan, c, "history.json")
        if os.path.exists(j) and not from_logs:
            arm.append(history_from_json(j))
        elif from_logs and logs:
            arm.append(history_from_log(logs[i]))
        else:
            arm.append([])
    return arm


def load_rho(scan, pattern, column="solve_rate"):
    """The 14-point grid, one list of (realized_rho, <column>) per seed.

    `column` is solve_rate for the rho_50 reading and outcome_accuracy for
    the SOLVED/UNSOLVED call, which is read off the same rows of the same
    file so the two curves can never come from different evals."""
    out = []
    for t in range(3):
        hit = [f for f in sorted(os.listdir(scan))
               if f.startswith("rho_curve_t%02d_" % t) and f.endswith(".csv")
               and (pattern in f)]
        if not hit:
            out.append([])
            continue
        rows = []
        with open(os.path.join(scan, hit[-1])) as f:
            for r in csv.DictReader(f):
                rows.append((float(r["realized_rho"]), float(r[column])))
        out.append(sorted(rows))
    return out


def _tag(fig, partial):
    if partial:
        fig.text(0.99, 0.01, "mid-run: read from the training logs, not from "
                 "a finished history.json", ha="right", va="bottom",
                 fontsize=6, color=MUTED)


def fig_train_nll(xs, ss, out, partial):
    """Three panels, one per seed: the main-term train NLL of both arms. The
    determinacy term is NOT in this column for either arm -- the trainer
    records the main cross-entropy alone -- so the two curves are the same
    quantity and the gap between them is the cost of the extra objective."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for i, ax in enumerate(axes):
        if not xs[i] or not ss[i]:
            continue
        ax.plot([r["epoch"] for r in ss[i]], [r["train_nll"] for r in ss[i]],
                color=SERIES[0], lw=1.6, rasterized=True)
        ax.plot([r["epoch"] for r in xs[i]], [r["train_nll"] for r in xs[i]],
                color=SERIES[1], lw=1.6, rasterized=True)
        style_axes(ax, xlabel="epoch", ylabel="train NLL" if i == 0 else None)
        ax.set_title("seed %d" % i, fontsize=9, color=MUTED)
        if i == 2:
            label_line(ax, ss[i][-1]["epoch"], ss[i][-1]["train_nll"],
                       "no term", color=SERIES[0])
            label_line(ax, xs[i][-1]["epoch"], xs[i][-1]["train_nll"],
                       "determinacy term", color=SERIES[1])
    fig.tight_layout()
    _tag(fig, partial)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_acc_deficit(xs, ss, out, partial):
    """One panel, three lines: stage X minus stage S at the same epoch and
    the same seed. Zero is the control. Below zero is worse."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ends = []
    for i in range(3):
        if not xs[i] or not ss[i]:
            continue
        n = min(len(xs[i]), len(ss[i]))
        ep = [xs[i][e]["epoch"] for e in range(n)]
        d = [xs[i][e]["accuracy"] - ss[i][e]["accuracy"] for e in range(n)]
        ax.plot(ep, d, color=SERIES[i], lw=1.6, rasterized=True)
        ends.append((d[-1], ep[-1], i))
    # All three curves land within about 0.002 of each other, so the three
    # right-hand labels collide unless they are staggered by rank.
    for rank, (y, x, i) in enumerate(sorted(ends, reverse=True)):
        label_line(ax, x, y, "seed %d" % i, color=SERIES[i],
                   dy=11 - 11 * rank)
    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--")
    style_axes(ax, xlabel="epoch",
               ylabel="val token accuracy, stage X minus control")
    fig.tight_layout()
    _tag(fig, partial)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_auroc(xs, control, out, partial):
    """Determinacy AUROC per epoch for the three stage-X cells, against the
    untrained control's band. The control is a flat reference because those
    cells never saw the term; it is measured once, not per epoch.

    The three seeds lie on top of each other to within about 0.001, so they
    are drawn in ONE colour with one label. Three labels here would say
    nothing except that the curves coincide, and would collide while saying
    it."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    last = None
    for i in range(3):
        if not xs[i]:
            continue
        ep = [r["epoch"] for r in xs[i] if r["auroc"] is not None]
        au = [r["auroc"] for r in xs[i] if r["auroc"] is not None]
        ax.plot(ep, au, color=SERIES[0], lw=1.4, rasterized=True)
        last = (ep[-1], au[-1])
    if last:
        label_line(ax, last[0], last[1], "all three seeds", color=SERIES[0])
    if control:
        ax.axhspan(min(control), max(control), color=MUTED, alpha=0.18)
        ax.text(0.02, min(control) - 0.035,
                "the same readout before the term: %.3f-%.3f"
                % (min(control), max(control)),
                transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED)
    ax.set_ylim(0.5, 1.0)
    style_axes(ax, xlabel="epoch", ylabel="determinacy AUROC")
    fig.tight_layout()
    _tag(fig, partial)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_rho(xr, sr, out, partial):
    """Three panels, one per seed: solve rate against realized hint density,
    both arms, with the half-line the crossing is read at."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for i, ax in enumerate(axes):
        if not sr[i]:
            continue
        ax.plot([p[0] for p in sr[i]], [p[1] for p in sr[i]],
                color=SERIES[0], lw=1.6, marker="o", ms=2.5, rasterized=True)
        if xr[i]:
            ax.plot([p[0] for p in xr[i]], [p[1] for p in xr[i]],
                    color=SERIES[1], lw=1.6, marker="o", ms=2.5,
                    rasterized=True)
        ax.axhline(0.5, color=MUTED, lw=1.0, ls="--")
        style_axes(ax, xlabel=r"realized $\rho$",
                   ylabel="solve rate" if i == 0 else None)
        ax.set_title("seed %d" % i, fontsize=9, color=MUTED)
        if i == 2:
            label_line(ax, sr[i][-1][0], sr[i][-1][1], "no term",
                       color=SERIES[0])
            if xr[i]:
                label_line(ax, xr[i][-1][0], xr[i][-1][1], "determinacy term",
                           color=SERIES[1])
    fig.tight_layout()
    _tag(fig, partial)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_outcome(xo, so, out, partial):
    """The SOLVED/UNSOLVED call against density, both arms, three panels.

    This is the mechanism figure. The control is flat near 1.00 at every
    density. The determinacy arm traces a U: right at the top, where almost
    nothing is left to deduce and stopping is correct; right at the bottom,
    where nothing is deducible and "stuck" is correct; and near chance in
    the middle, which is the only region where the call requires carrying
    the deduction on. A term whose whole content is which nodes are still
    deducible breaking exactly the is-anything-still-deducible decision is
    what a guard that fires too eagerly looks like."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for i, ax in enumerate(axes):
        if not so[i]:
            continue
        ax.plot([p[0] for p in so[i]], [p[1] for p in so[i]],
                color=SERIES[0], lw=1.6, marker="o", ms=2.5, rasterized=True)
        if xo[i]:
            ax.plot([p[0] for p in xo[i]], [p[1] for p in xo[i]],
                    color=SERIES[1], lw=1.6, marker="o", ms=2.5,
                    rasterized=True)
        ax.set_ylim(0.0, 1.05)
        style_axes(ax, xlabel=r"realized $\rho$",
                   ylabel="solved/unsolved call" if i == 0 else None)
        ax.set_title("seed %d" % i, fontsize=9, color=MUTED)
        if i == 2:
            # Right-hand end, and staggered: the two curves rejoin at the
            # top of the range to within 0.06, so unstaggered labels overlap.
            label_line(ax, so[i][-1][0], so[i][-1][1], "no term",
                       color=SERIES[0], dy=7)
            if xo[i]:
                label_line(ax, xo[i][-1][0], xo[i][-1][1], "determinacy term",
                           color=SERIES[1], dy=-9)
    fig.tight_layout()
    _tag(fig, partial)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--from-logs", action="store_true",
                    help="read stage X per-epoch numbers from the Slurm logs "
                         "instead of history.json, for a mid-run figure")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    xs = load_arm(SCAN_X, X_CELLS, X_LOGS, from_logs=a.from_logs)
    ss = load_arm(SCAN_S, S_CELLS)
    print("stage X epochs per seed:", [len(h) for h in xs])
    print("stage S epochs per seed:", [len(h) for h in ss])

    control = []
    ctrl_csv = os.path.join(SCAN_X, "determinacy_control_stageS.csv")
    if os.path.exists(ctrl_csv):
        with open(ctrl_csv) as f:
            for r in csv.DictReader(f):
                control.append(float(r["determinacy_auroc"]))
    print("finding 73 control AUROC:", control)

    xr = load_rho(SCAN_X, "52285051") if os.path.isdir(SCAN_X) else [[]] * 3
    sr = load_rho(SCAN_S, "51853969")
    print("rho grid points per seed:  X", [len(c) for c in xr],
          " S", [len(c) for c in sr])

    p = a.from_logs
    fig_train_nll(xs, ss, os.path.join(a.out_dir, "train_nll.png"), p)
    fig_acc_deficit(xs, ss, os.path.join(a.out_dir, "acc_deficit.png"), p)
    fig_auroc(xs, control, os.path.join(a.out_dir, "auroc.png"), p)
    if any(xr):
        fig_rho(xr, sr, os.path.join(a.out_dir, "rho_curve.png"), p)
        xo = load_rho(SCAN_X, "52285051", "outcome_accuracy")
        so = load_rho(SCAN_S, "51853969", "outcome_accuracy")
        fig_outcome(xo, so, os.path.join(a.out_dir, "outcome_call.png"), p)
    else:
        print("no stage-X rho grid yet: rho_curve.png not written")
    print("wrote figures to", a.out_dir)


if __name__ == "__main__":
    main()
