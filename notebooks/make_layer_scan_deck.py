"""Slide deck: does network depth have to grow with reasoning depth?

The question hshunt asked: is there a relationship between the reasoning depth
an instance requires and the depth of the network needed to solve it. The
instrument is L*(T) -- the smallest `num_layers` reaching a solve_rate
threshold on instances whose DD trace takes T rounds, measured at FIXED hint
density so that depth is not standing in for density.

    serial hypothesis   L*(T) rises with T: the model spends layers on
                        deduction rounds, roughly one per round.
    finding 2           L*(T) is flat: whatever it does, it is not chaining
                        rounds through the stack, and extra layers buy
                        accuracy uniformly across T.

Reads whichever `sudoku_layer_scan_2026-10-24_n*` directories exist, so the
same script serves the N=24 arm now and the N=48/N=96 arms later. Run:

    python -m notebooks.make_layer_scan_deck
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

S = os.environ.get("SCRATCH", "/scratch/users/hshunt") + "/spin_gnn"
OUT = os.path.join(S, "notebooks", "sudoku_layer_scan_2026-10-24")
# The threshold defining L*(T). Pre-registered in the sbatch header as 0.90.
THRESH = 0.90
# Bins with fewer instances than this are plotted but excluded from L*(T):
# a 0.90 crossing estimated on 20 instances has a +-0.13 Wilson half-width.
MIN_N = 40
# Mean DD depth over the training range, measured on 20 instances per size.
MEAN_T = {24: 4.25, 48: 6.10, 96: 7.45}


def arms():
    """{N: {"sweep": rows, "depth": rows}} for every scan directory present."""
    out = {}
    for d in sorted(glob.glob(f"{S}/sudoku_layer_scan_2026-10-24_n*")):
        N = int(re.search(r"_n(\d+)$", d).group(1))
        sweep = f"{d}/sweep.csv"
        depth = f"{d}/depth_by_layers.csv"
        if not os.path.exists(sweep):
            continue
        out[N] = {
            "dir": d,
            "sweep": list(csv.DictReader(open(sweep))),
            "depth": list(csv.DictReader(open(depth))) if os.path.exists(depth) else [],
        }
    return out


def layers_of(name):
    """`d0256_h04_L03_ff01024_lr1e-03_pre` -> 3. The tag pads L to two digits
    so the checkpoint glob sorts correctly; that padding is why this is a
    regex and not a split."""
    return int(re.search(r"_L(\d+)_", name).group(1))


def by_layers(sweep):
    """num_layers -> aggregate row, one per trained cell.

    `sweep.csv` carries `num_layers` as its own column, so this does not have
    to go through the tag. The depth CSV does: it only records `checkpoint`,
    hence `layers_of`.
    """
    return {int(r["num_layers"]): r for r in sweep}


def depth_grid(depth, rho):
    """{num_layers: {T: (solve_rate, n, realized_rho)}} at one requested rho."""
    grid = {}
    for r in depth:
        if abs(float(r["rho_requested"]) - rho) > 1e-9:
            continue
        L = layers_of(r["checkpoint"])
        grid.setdefault(L, {})[int(r["T"])] = (
            float(r["solve_rate"]), int(r["n"]), float(r["realized_rho"]))
    return grid


def l_star(grid, T):
    """Smallest num_layers reaching THRESH at depth T, or None if none does."""
    for L in sorted(grid):
        cell = grid[L].get(T)
        if cell and cell[1] >= MIN_N and cell[0] >= THRESH:
            return L
    return None


def rho_drift(grid):
    """max spread in realized_rho across T, the confound check.

    Finding 5 discounted the rho=0.50 series because realized density fell
    with T there (r = -0.82): if deep-T instances are also sparser, a decline
    in solve_rate with T is density, not depth. Any arm whose spread here is
    large has the confound inside its bins and cannot support an L*(T) claim.
    """
    vals = [c[2] for cells in grid.values() for c in cells.values()]
    return (max(vals) - min(vals)) if vals else float("nan")


def page_title(pdf, data):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.82, "Does network depth track reasoning depth?",
             ha="center", size=BIG + 3, weight="bold")
    fig.text(0.5, 0.725, "   ·   ".join(f"N = {N}" for N in sorted(data))
             + "      d_model 256, ff 1024 fixed, only num_layers varies",
             ha="center", size=MED - 2, color="0.35")
    agg = {N: by_layers(d["sweep"]) for N, d in data.items()}
    # Fixed-width columns: a monospace block only reads as a table if the
    # label prefix and every cell are padded to the same width.
    lines, lab, w = [], 10, 8
    for N in sorted(agg):
        got = sorted(agg[N])
        lines.append(f"N = {N}".ljust(lab) + "".join(f"L={L}".rjust(w) for L in got))
        lines.append("solve".rjust(lab) + "".join(
            f"{float(agg[N][L]['agg_solve_rate']):.3f}".rjust(w) for L in got))
        lines.append("params (M)".rjust(lab) + "".join(
            f"{float(agg[N][L]['n_params']) / 1e6:.2f}".rjust(w) for L in got))
        lines.append("")
    fig.text(0.5, 0.56, "aggregate solve_rate by num_layers", ha="center",
             size=MED, color="0.2")
    fig.text(0.5, 0.40, "\n".join(lines), ha="center", va="center",
             size=MED - 3, family="monospace", linespacing=1.9)
    footer(fig, "L*(T) is the smallest num_layers reaching solve_rate >= "
                f"{THRESH:.2f} on depth-T instances at fixed hint density. Bins with "
                f"fewer than {MIN_N} instances are excluded from it.")
    pdf.savefig(fig)


def page_aggregate(pdf, data):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for N, colour in zip(sorted(data), ("tab:blue", "tab:orange", "tab:green")):
        agg = by_layers(data[N]["sweep"])
        Ls = sorted(agg)
        ax.plot(Ls, [float(agg[L]["agg_solve_rate"]) for L in Ls], "o-",
                color=colour, lw=2.5, ms=10, label=f"N = {N}  (mean DD depth {MEAN_T.get(N, float('nan')):.2f})")
    ax.axhline(THRESH, color="0.5", ls=":", lw=1.5)
    # Axes coords, not data: on a log2 x-axis a data-space x of 0.4 lands off
    # the left spine entirely.
    ax.text(0.02, THRESH + 0.015, f"{THRESH:.2f} threshold", size=SMALL,
            color="0.4", transform=ax.get_yaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12])
    ax.set_xticklabels(["1", "2", "3", "4", "6", "8", "12"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("num_layers")
    ax.set_ylabel("aggregate solve_rate")
    ax.set_title("Where the knee in network depth is", size=MED + 5, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    footer(fig, "Aggregate over the whole training density range, so it mixes depths and is NOT the "
                "answer to the depth question -- it locates the knee that the per-depth pages then "
                "test. A knee far below the deepest instance's T is already evidence against one "
                "layer per deduction round.")
    pdf.savefig(fig)


def page_depth_curves(pdf, data, rho):
    present = [N for N in sorted(data) if depth_grid(data[N]["depth"], rho)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(13.33, 7.5),
                             sharey=True, squeeze=False)
    for ax, N in zip(axes[0], present):
        grid = depth_grid(data[N]["depth"], rho)
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(grid)))
        for colour, L in zip(cmap, sorted(grid)):
            Ts = sorted(grid[L])
            ax.plot(Ts, [grid[L][T][0] for T in Ts], "o-", color=colour,
                    label=f"L={L}", lw=2, ms=7)
        ax.axhline(THRESH, color="0.5", ls=":", lw=1.5)
        ax.set_title(f"N = {N}   (realized rho spread {rho_drift(grid):.3f})", size=MED - 1)
        ax.set_xlabel("DD depth T of the instance")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("solve_rate")
    axes[0][0].set_ylim(0, 1.05)
    axes[0][-1].legend(fontsize=SMALL - 1, ncol=2, loc="lower left")
    fig.suptitle(f"solve_rate by reasoning depth, at fixed density rho = {rho:.2f}",
                 size=MED + 4, weight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    footer(fig, "Density is held fixed, so a decline left-to-right is attributable to the length of "
                "the deduction chain rather than to how much was given away. The realized rho spread "
                "in each title is the check on that: if it is large, deeper-T instances are also "
                "sparser and the confound has moved inside the bins. If the curves for different L "
                "are PARALLEL, depth buys accuracy uniformly; if they FAN OUT with T, deeper "
                "networks reach deeper instances and the serial picture is right.")
    pdf.savefig(fig)


def page_lstar(pdf, data, rho):
    present = [N for N in sorted(data) if depth_grid(data[N]["depth"], rho)]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    verdicts = []
    for N, colour in zip(present, ("tab:blue", "tab:orange", "tab:green")):
        grid = depth_grid(data[N]["depth"], rho)
        Ts = sorted({T for cells in grid.values() for T, c in cells.items() if c[1] >= MIN_N})
        xs = [T for T in Ts if l_star(grid, T) is not None]
        ys = [l_star(grid, T) for T in xs]
        unreachable = [T for T in Ts if l_star(grid, T) is None]
        ax.plot(xs, ys, "o-", color=colour, lw=2.5, ms=11, label=f"N = {N}")
        for T in unreachable:
            ax.plot([T], [max(grid) * 1.08], "x", color=colour, ms=12, mew=3)
        if len(ys) >= 2:
            slope = np.polyfit(xs, ys, 1)[0]
            verdicts.append(f"N={N}: L*({xs[0]})={ys[0]}, L*({xs[-1]})={ys[-1]}, slope {slope:+.2f} layers per round")
    ax.set_xlabel("DD depth T of the instance")
    ax.set_ylabel(f"L*(T) = smallest num_layers reaching {THRESH:.2f}")
    ax.set_title("The answer: how network depth scales with reasoning depth",
                 size=MED + 4, weight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    if verdicts:
        ax.text(0.98, 0.04, "\n".join(verdicts), transform=ax.transAxes, ha="right",
                size=SMALL, family="monospace",
                bbox=dict(boxstyle="round", fc="0.95", ec="0.7"))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    footer(fig, "A slope near 0 says the same network depth suffices at every reasoning depth, which "
                "is what finding 2 predicts -- the model is not spending one layer per deduction "
                "round. A slope near +1 is the serial picture. Crosses mark depths no tested "
                f"num_layers reached, i.e. L*(T) is off the top of the grid. Measured at rho = {rho:.2f}.")
    pdf.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    data = arms()
    if not data:
        print("no layer-scan directories yet")
        return
    for N, d in data.items():
        print(f"N={N}: {len(by_layers(d['sweep']))} trained cells, "
              f"{len(d['depth'])} depth rows ({d['dir']})")

    pdf_path = os.path.join(OUT, "layers_vs_reasoning_depth.pdf")
    with PdfPages(pdf_path) as raw:
        pdf = PdfAndPng(raw, OUT)
        page_title(pdf, data)
        page_aggregate(pdf, data)
        for rho in (0.75, 0.60):
            page_depth_curves(pdf, data, rho)
            page_lstar(pdf, data, rho)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {pdf.n} pages)")


if __name__ == "__main__":
    main()
