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


def drift_verdict(grid, N):
    """Spread in realized_rho expressed in HINTS, plus a keep/discount verdict.

    The depth eval reports r(realized_rho, T) and the N=24 sbatch said a large
    value means "this rho must be discounted". Taken literally that discards
    the rho=0.60 series here on r = -0.941. But r measures only whether the
    drift is CONSISTENT in direction, not whether it is big enough to matter,
    and these two can diverge completely: at rho=0.60, N=24 the realized
    density falls monotonically with T (hence |r| ~ 1) across a total range of
    0.0009, which is 0.02 of one hint out of 24. A fifth of a percent of a
    single hint cannot move solve_rate by the 4-5 points seen there.

    So the verdict is taken on the magnitude and the sign of r together, and
    the primary claim is made on the rho with ZERO drift. Reporting the hint
    figure is the point -- it is what makes "r is large" and "the confound is
    negligible" visibly consistent rather than a contradiction to be argued
    away.
    """
    spread = rho_drift(grid)
    hints = spread * N
    if spread == 0:
        return "density identical in every bin -- clean"
    if hints < 0.1:
        return f"drift {spread:.4f} = {hints:.2f} hints -- negligible"
    return f"drift {spread:.4f} = {hints:.2f} hints -- DISCOUNT"


def gain_by_T(grid, lo, hi):
    """{T: solve_rate(hi) - solve_rate(lo)} -- what extra layers buy at each T.

    L*(T) answers "is any depth enough"; this answers "where does the next
    doubling of depth land". The two can disagree, and here they do.
    """
    out = {}
    if lo not in grid or hi not in grid:
        return out
    for T in sorted(set(grid[lo]) & set(grid[hi])):
        if grid[lo][T][1] >= MIN_N and grid[hi][T][1] >= MIN_N:
            out[T] = grid[hi][T][0] - grid[lo][T][0]
    return out


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
    # Headline, computed rather than written, so the N=48/96 arms restate it
    # themselves instead of inheriting a sentence that was true at N=24.
    heads = []
    for N in sorted(data):
        for rho in (0.75, 0.60):
            grid = depth_grid(data[N]["depth"], rho)
            Ts = sorted({T for c in grid.values() for T, v in c.items() if v[1] >= MIN_N})
            stars = [l_star(grid, T) for T in Ts]
            stars = [s for s in stars if s is not None]
            if len(stars) >= 2:
                flat = len(set(stars)) == 1
                heads.append(f"N={N}, rho={rho:.2f}: L*(T) = {stars[0]} at every T in "
                             f"{Ts[0]}..{Ts[-1]}" if flat else
                             f"N={N}, rho={rho:.2f}: L*(T) rises {stars[0]} -> {stars[-1]}"
                             f" over T = {Ts[0]}..{Ts[-1]}")
    if heads:
        allflat = all("at every T" in h for h in heads)
        fig.text(0.5, 0.645, ("Network depth does NOT track reasoning depth"
                              if allflat else "Network depth tracks reasoning depth"),
                 ha="center", size=MED + 2, weight="bold",
                 color="tab:red" if allflat else "tab:green")
        fig.text(0.5, 0.60, "   ·   ".join(heads), ha="center", size=SMALL,
                 color="0.3", family="monospace")
    fig.text(0.5, 0.52, "aggregate solve_rate by num_layers", ha="center",
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
        ax.set_title(f"N = {N}   ({drift_verdict(grid, N)})", size=MED - 1)
        ax.set_xlabel("DD depth T of the instance")
        ax.grid(alpha=0.3)

        # Every L at or above the knee sits in the top few percent, where the
        # whole question lives: whether those curves are parallel or fan out.
        # On the 0-1 axis that structure is a few pixels tall, so it gets an
        # inset. Only curves that clear the threshold everywhere go in it --
        # the failing L values would re-flatten the zoom.
        good = [L for L in sorted(grid)
                if all(v[0] >= THRESH for v in grid[L].values() if v[1] >= MIN_N)]
        if len(good) >= 2:
            # Mid-right: the band between the failing L curves (<= 0.45 here)
            # and the bunched passing ones (>= 0.94) is empty, and the legend
            # holds the lower left.
            inset = ax.inset_axes([0.38, 0.42, 0.58, 0.33])
            vals = [grid[L][T][0] for L in good for T in grid[L]]
            for colour, L in zip(cmap, sorted(grid)):
                if L not in good:
                    continue
                Ts = sorted(grid[L])
                inset.plot(Ts, [grid[L][T][0] for T in Ts], "o-", color=colour,
                           lw=1.8, ms=5)
            inset.set_ylim(min(vals) - 0.01, 1.005)
            inset.tick_params(labelsize=SMALL - 3)
            inset.set_title(f"zoom: L >= {good[0]}", size=SMALL - 2, color="0.35")
            inset.grid(alpha=0.25)
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
        # Flat L*(T) does not mean depth is useless at depth. Report where the
        # next doubling past L* actually lands, because that is the part of the
        # serial picture that survives.
        #
        # Both the BEST and the DEEPEST L are reported when they differ. The
        # best one is the meaningful comparison -- L=12 is past the knee and
        # overfits, so a gain measured against it mixes "more depth" with "more
        # overfitting" and comes out noisy. But picking the best by aggregate
        # and then reading a per-T trend off it is selection on a related
        # metric, so the deepest is printed next to it and the reader can see
        # that the trend claim rests on the pair, not on my choice of one.
        base = ys[0] if ys else None
        agg = by_layers(data[N]["sweep"])
        best = max(agg, key=lambda L: float(agg[L]["agg_solve_rate"]))
        if base is not None:
            targets = [(best, "best")] if best == max(grid) else [
                (best, "best agg"), (max(grid), "deepest")]
            for hi, tag in targets:
                gains = gain_by_T(grid, base, hi)
                if len(gains) >= 2:
                    verdicts.append(f"   L{base}->L{hi} ({tag}): "
                                    + ", ".join(f"T={T}:{g:+.3f}" for T, g in gains.items()))
    ax.set_xlabel("DD depth T of the instance")
    ax.set_ylabel(f"L*(T) = smallest num_layers reaching {THRESH:.2f}")
    ax.set_title("The answer: how network depth scales with reasoning depth",
                 size=MED + 4, weight="bold")
    # Span the whole tested grid. Autoscaling a flat series gives a +-0.15-layer
    # window in which "flat" and "rising" look identical; the point of the page
    # is that the line does not move across the range of depths actually trained.
    deepest = max(max(depth_grid(data[N]["depth"], rho), default=1) for N in present)
    ax.set_ylim(0, deepest * 1.15)
    ax.set_yticks([L for L in (1, 2, 3, 4, 6, 8, 12) if L <= deepest])
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


def score(data, N):
    """(label, verdict, evidence) for each pre-registration in the N=24 sbatch.

    Scored from the CSVs rather than read off the plots so that the N=48 and
    N=96 arms get the same treatment without my re-reading anything.
    """
    agg = by_layers(data[N]["sweep"])
    grids = {rho: depth_grid(data[N]["depth"], rho) for rho in (0.75, 0.60)}
    rows = []

    # (a) one round should not need depth
    t1 = [(L, cells[1][0]) for rho in grids for L, cells in grids[rho].items()
          if 1 in cells and L >= 2 and cells[1][1] >= MIN_N]
    if t1:
        worst = min(t1, key=lambda x: x[1])
        rows.append(("(a) T=1 solve >= 0.95 for all L >= 2",
                     "HOLDS" if worst[1] >= 0.95 else "FAILS",
                     f"worst is L={worst[0]} at {worst[1]:.3f}"))

    # (b) the serial hypothesis
    for rho, grid in grids.items():
        Ts = sorted({T for c in grid.values() for T, v in c.items() if v[1] >= MIN_N})
        stars = [(T, l_star(grid, T)) for T in Ts]
        stars = [(T, L) for T, L in stars if L is not None]
        if len(stars) >= 2:
            vals = [L for _, L in stars]
            rise = vals[-1] - vals[0]
            rows.append((f"(b) L*(T) rises, L*(max) >= L*(min)+3  [rho={rho:.2f}]",
                         "FALSIFIED -> FLAT" if rise < 3 else "HOLDS",
                         f"L*={vals} over T={[T for T, _ in stars]}, rise {rise:+d}"))

    # (c) one layer has no residual stream to iterate on
    deep1 = [v[0] for grid in grids.values() for T, v in grid.get(1, {}).items()
             if T >= 2 and v[1] >= MIN_N]
    if deep1:
        rows.append(("(c) L=1 fails everywhere (< 0.5 at T >= 2)",
                     "HOLDS" if max(deep1) < 0.5 else "FAILS",
                     f"max over T>=2 bins is {max(deep1):.3f}"))

    # (d) saturation past the knee
    if 8 in agg and 12 in agg:
        d = abs(float(agg[12]["agg_solve_rate"]) - float(agg[8]["agg_solve_rate"]))
        rows.append(("(d) aggregate saturates by L=8 (|L12-L8| <= 0.01)",
                     "HOLDS" if d <= 0.01 else "FAILS",
                     f"L8 {float(agg[8]['agg_solve_rate']):.4f}, "
                     f"L12 {float(agg[12]['agg_solve_rate']):.4f}, delta {d:.4f}"))
    return rows


def page_scorecard(pdf, data):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.93, "Pre-registered predictions, scored", ha="center",
             size=BIG, weight="bold")
    y = 0.80
    for N in sorted(data):
        rows = score(data, N)
        if not rows:
            continue
        fig.text(0.06, y, f"N = {N}", size=MED, weight="bold", color="0.2")
        y -= 0.06
        for label, verdict, evidence in rows:
            colour = {"HOLDS": "tab:green", "FAILS": "tab:red"}.get(verdict, "tab:blue")
            fig.text(0.08, y, label, size=SMALL + 1, va="top")
            fig.text(0.70, y, verdict, size=SMALL + 1, va="top", weight="bold",
                     color=colour)
            fig.text(0.08, y - 0.035, evidence, size=SMALL - 1, va="top",
                     color="0.45", family="monospace")
            y -= 0.085
        y -= 0.02
    footer(fig, "(a) failing is the informative one: at T=1 -- a single deduction round -- L=2 still "
                "scores 0.686, so the L=3 knee is not about chaining rounds through the stack at all. "
                "That is the same conclusion (b) reaches from the other side.")
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
        page_scorecard(pdf, data)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {pdf.n} pages)")


if __name__ == "__main__":
    main()
