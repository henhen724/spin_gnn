"""Slide deck: is the L=3 knee serial composition, or just parameters?

Sources, all N=24, d_model=256, nhead=4, lr=1e-3, pre-norm, 150 epochs,
n_train=20000 -- every cell here is trained identically apart from
(num_layers, dim_feedforward), which is the whole point:

  49510197  sudoku_ff_depth_tradeoff_2026-10-25_n24  the four new cells
  49433454  sudoku_layer_scan_2026-10-24_n24         the ff=1024 reference ladder
  49521537  sudoku_ff_depth_sharedpool_2026-10-25    all eight on the shared pool

The per-run 300-instance agg_solve_rate settles the 1.65M pair on its own
(+0.470, far outside its noise) but not the 2.44M pair (+0.023 against a
half-width of 0.03), so every number quoted on the comparison pages is the
shared-pool one at n=1500. The sweep.csv numbers appear only where a page is
explicitly about the training run rather than the comparison.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style
from spin_gnn.sudoku_agg_eval import mcnemar_exact_p

ROOT = os.environ["SCRATCH"] + "/spin_gnn"
POOL = f"{ROOT}/sudoku_ff_depth_sharedpool_2026-10-25/sharedpool_n1500"
TRADE = f"{ROOT}/sudoku_ff_depth_tradeoff_2026-10-25_n24"
SCAN = f"{ROOT}/sudoku_layer_scan_2026-10-24_n24"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "sudoku_ff_depth_2026-10-25")

# The two matched-parameter budgets, each spent wide-shallow and thin-deep.
# Counts are sweep.csv's own, so they are internally consistent; the pairs
# agree to 0.22%, which is what the matched-parameter claims rest on.
BUDGETS = [("1.65M", [("L02_ff01024", "wide, shallow"), ("L04_ff00256", "thin, deep")]),
           ("2.44M", [("L03_ff01024", "wide, shallow  (the knee)"),
                      ("L02_ff01792", "wider, shallow"),
                      ("L06_ff00256", "thin, deep")])]
WIDE = "tab:red"
DEEP = "tab:blue"


def tag(ckpt):
    return ckpt.replace("d0256_h04_", "").replace("_lr1e-03_pre", "")


def load_pool():
    """{tag: dict} from the shared-pool CSV, plus the per-instance masks.

    The CSV only carries McNemar against the first checkpoint, so every
    pairing other than "vs L=2/ff=1024" is recomputed from the npz.
    """
    rows = {}
    with open(POOL + ".csv") as fh:
        for r in csv.DictReader(fh):
            r["tag"] = tag(r["checkpoint"])
            r["solve_rate"] = float(r["solve_rate"])
            r["lo"], r["hi"] = float(r["wilson_lo"]), float(r["wilson_hi"])
            rows[r["tag"]] = r
    z = np.load(POOL + "_per_instance.npz", allow_pickle=True)
    masks = {tag(str(n)): z["solved"][i] for i, n in enumerate(z["names"])}
    return rows, masks


def paired(masks, a, b):
    """(delta, won, lost, exact p) for a against b on the same instances."""
    x, y = masks[a], masks[b]
    won, lost = int((~y & x).sum()), int((y & ~x).sum())
    return x.mean() - y.mean(), won, lost, mcnemar_exact_p(lost, won)


def load_params():
    """{tag: n_params} from both sweeps, so the budgets are not re-derived."""
    out = {}
    for path in (f"{TRADE}/sweep.csv", f"{SCAN}/sweep.csv"):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                t = f"L{int(r['num_layers']):02d}_ff{int(r['dim_feedforward']):05d}"
                out[t] = int(r["n_params"])
    return out


def load_depth():
    """{(rho, tag): {T: (solve_rate, n)}} for the four trade-off cells."""
    out = {}
    with open(f"{TRADE}/depth_by_ff.csv") as fh:
        for r in csv.DictReader(fh):
            out.setdefault((float(r["rho_requested"]), tag(r["checkpoint"])), {})[
                int(r["T"])] = (float(r["solve_rate"]), int(r["n"]))
    return out


def load_history(t):
    """Validation nll per epoch for one cell, from whichever sweep trained it."""
    import json
    for base in (TRADE, SCAN):
        L, ff = int(t[1:3]), int(t[6:])
        p = f"{base}/d0256_h04_L{L:02d}_ff{ff:05d}_lr1e-03_pre/history.json"
        if os.path.exists(p):
            with open(p) as fh:
                h = json.load(fh)
            return [d["epoch"] for d in h], [d["nll"] for d in h]
    return [], []


def label_of(t):
    return f"L={int(t[1:3])}, ff={int(t[6:])}"


def page_title(pdf, pool, masks, params):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.88, "Is the L=3 knee serial composition, or parameters?",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.80, "N=24 (d=5, q=4), 150 epochs, n_train=20000, d_model=256 -- "
                        "only (num_layers, dim_feedforward) varies",
             ha="center", size=SMALL, color="0.35")
    fig.text(0.5, 0.755, "49510197 trained, 49521537 scored on the shared pool at n=1500",
             ha="center", size=SMALL - 1, color="0.5", family="monospace")

    d1, _, _, p1 = paired(masks, "L04_ff00256", "L02_ff01024")
    d2, _, _, p2 = paired(masks, "L06_ff00256", "L03_ff01024")
    fig.text(0.5, 0.645, "SERIAL. At equal parameters, depth wins by a mile.",
             ha="center", size=MED + 5, weight="bold")
    fig.text(0.5, 0.545,
             f"At 1.65M params, L=4/ff=256 scores {pool['L04_ff00256']['solve_rate']:.3f} "
             f"against L=2/ff=1024's {pool['L02_ff01024']['solve_rate']:.3f}",
             ha="center", size=MED)
    fig.text(0.5, 0.495, f"-- a gap of {d1:+.3f}, paired p = {p1:.0e}.",
             ha="center", size=MED)
    fig.text(0.5, 0.405,
             f"At 2.44M, depth still wins: {pool['L06_ff00256']['solve_rate']:.3f} vs "
             f"{pool['L03_ff01024']['solve_rate']:.3f} ({d2:+.3f}, p = {p2:.0e}),",
             ha="center", size=MED)
    fig.text(0.5, 0.355, "and no MLP width tested buys the knee at L=2 at any budget.",
             ha="center", size=MED)
    over = params["L02_ff04096"] / params["L03_ff01024"]
    fig.text(0.5, 0.255,
             f"L=2/ff=4096 carries {params['L02_ff04096'] / 1e6:.1f}M -- {over:.2f}x the knee's "
             f"budget -- and scores {pool['L02_ff04096']['solve_rate']:.3f}.",
             ha="center", size=MED, color="0.25")
    footer(fig, "Finding 11 established that L*(T) is flat at 3 and that L=2 already fails at T=1, "
                "so the third layer is a fixed cost paid before any chaining. That left two readings: "
                "H1, the cost is CAPACITY and where the parameters sit does not matter, so a wider "
                "MLP at L=2 should recover it; H2, the cost is a SEQUENTIAL step and no width buys "
                "it. The two make opposite predictions at matched parameters, which is what this "
                "deck measures. H1 is dead.")
    pdf.savefig(fig)


def page_budgets(pdf, pool, masks, params):
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5),
                             gridspec_kw={"width_ratios": [2, 3]})
    for ax, (budget, cells) in zip(axes, BUDGETS):
        xs = np.arange(len(cells))
        vals = [pool[t]["solve_rate"] for t, _ in cells]
        err = [[v - pool[t]["lo"] for v, (t, _) in zip(vals, cells)],
               [pool[t]["hi"] - v for v, (t, _) in zip(vals, cells)]]
        cols = [DEEP if "ff00256" in t else WIDE for t, _ in cells]
        ax.bar(xs, vals, 0.62, color=cols, yerr=err, capsize=6, ecolor="0.3")
        for x, v, (t, _) in zip(xs, vals, cells):
            ax.text(x, v + 0.035, f"{v:.3f}", ha="center", size=MED, weight="bold")
            ax.text(x, 0.045, f"{params[t] / 1e6:.2f}M", ha="center", size=SMALL,
                    color="white", weight="bold")
        ax.axhline(0.90, color="0.4", ls="--", lw=1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{label_of(t)}\n{how}" for t, how in cells], size=SMALL)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{budget} parameters", size=MED + 1)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("shared-pool solve_rate  (n = 1500)")
    axes[0].text(0.02, 0.915, "0.90", transform=axes[0].get_yaxis_transform(),
                 ha="left", va="bottom", size=SMALL, color="0.4")
    fig.suptitle("The same budget, spent two ways", size=MED + 4, weight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0.20, 1, 0.93))
    d1, w1, l1, p1 = paired(masks, "L04_ff00256", "L02_ff01024")
    d2, w2, l2, p2 = paired(masks, "L06_ff00256", "L03_ff01024")
    footer(fig, f"Each panel holds the parameter count fixed to 0.22% and moves it between depth and "
                f"MLP width. Left is the discriminator: same budget, and the thin-deep cell clears "
                f"0.90 while the wide-shallow one does not clear 0.50 ({d1:+.3f}, {w1} discordant "
                f"instances won to {l1} lost, exact p = {p1:.0e}). Under H1 these two should have "
                f"landed within 0.10 of each other. Right repeats it at the knee's own budget, where "
                f"the wide-shallow cell is the measured knee rather than a cripple, and depth still "
                f"wins ({d2:+.3f}, {w2}-{l2}, p = {p2:.0e}). Spending the knee's budget on width "
                f"instead (L=2/ff=1792) gives {pool['L02_ff01792']['solve_rate']:.3f}: better than "
                f"L=2/ff=1024, nowhere near the knee. Wilson intervals at n=1500; the paired McNemar "
                f"is the test, since all cells are scored on the same instances.")
    pdf.savefig(fig)


def page_pareto(pdf, pool, masks, params):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ladders = [("ff = 1024, varying depth", DEEP, "o-",
                ["L01_ff01024", "L02_ff01024", "L03_ff01024", "L04_ff01024",
                 "L06_ff01024", "L08_ff01024", "L12_ff01024"]),
               ("ff = 256, varying depth", "tab:green", "s-",
                ["L04_ff00256", "L06_ff00256"]),
               ("L = 2, varying width", WIDE, "D-",
                ["L02_ff01024", "L02_ff01792", "L02_ff04096"])]
    for lab, col, style, cells in ladders:
        pts = [(params[t] / 1e6, pool[t]["solve_rate"]) for t in cells if t in pool]
        if not pts:
            continue
        ax.plot(*zip(*sorted(pts)), style, color=col, lw=2.5, ms=11, label=lab)
    # Hand-placed: the three ladders cross, so a uniform offset puts labels on
    # top of other curves.
    for t, off, ha in [("L04_ff00256", (6, -24), "left"), ("L06_ff00256", (16, 4), "left"),
                       ("L12_ff01024", (-6, -30), "right"), ("L02_ff04096", (0, 20), "center")]:
        ax.annotate(label_of(t), (params[t] / 1e6, pool[t]["solve_rate"]),
                    textcoords="offset points", xytext=off, ha=ha,
                    size=SMALL - 1, color="0.3")
    ax.axhline(0.90, color="0.5", ls="--", lw=1.2)
    ax.set_ylim(0.43, 1.04)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (M)")
    ax.set_ylabel("shared-pool solve_rate  (n = 1500)")
    ax.set_title("Where the parameters sit matters more than how many there are",
                 size=MED + 3, weight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.21, 1, 1))
    d3, w3, l3, p3 = paired(masks, "L06_ff00256", "L12_ff01024")
    d4, _, _, p4 = paired(masks, "L06_ff00256", "L04_ff01024")
    footer(fig, f"Three ladders through the same parameter axis. Only the two that add LAYERS climb "
                f"to the ceiling; the one that adds WIDTH at L=2 rises to 0.663 and then falls back, "
                f"never approaching 0.90, despite reaching {params['L02_ff04096'] / 1e6:.1f}M. The "
                f"ff=1024 ladder starts at L=2 here because its L=1 cell (0.067) was not worth a "
                f"shared-pool pass. The green ladder is the efficient one: "
                f"L=6/ff=256 at {params['L06_ff00256'] / 1e6:.2f}M scores "
                f"{pool['L06_ff00256']['solve_rate']:.4f}, statistically tied with L=12/ff=1024 at "
                f"{params['L12_ff01024'] / 1e6:.2f}M ({d3:+.4f}, {w3} to {l3}, p = {p3:.2f}) and "
                f"with L=4/ff=1024 at {params['L04_ff01024'] / 1e6:.2f}M ({d4:+.4f}, p = {p4:.2f}) "
                f"-- the top of the reference ladder reached with {params['L12_ff01024'] / params['L06_ff00256']:.1f}x "
                f"fewer parameters. That tie is the pre-registered (c) and it held. Caveat on reading "
                f"this as a scaling law: ff=256 was tested at two depths only, so the green ladder is "
                f"two points, and nothing here says ff can be cut further.")
    pdf.savefig(fig)


def page_depth_bins(pdf, depth):
    rhos = sorted({r for r, _ in depth})
    fig, axes = plt.subplots(1, len(rhos), figsize=(13.33, 7.5), sharey=True)
    for ax, rho in zip(np.atleast_1d(axes), rhos):
        for t in sorted({c for r, c in depth if r == rho}):
            series = sorted(depth[(rho, t)].items())
            # One colour per cell: the two thin-deep curves sit on top of each
            # other at ~1.0 and the two L=2 curves are 0.3 apart, so a single
            # colour per GROUP makes the legend unreadable.
            col = {"L02_ff01792": WIDE, "L02_ff04096": "tab:orange",
                   "L04_ff00256": DEEP, "L06_ff00256": "tab:green"}[t]
            ls = "-" if "ff00256" in t else "--"
            ax.plot([T for T, _ in series], [v for _, (v, _) in series],
                    "o" + ls, color=col, lw=2.5, ms=10, label=label_of(t))
        ax.axhline(0.90, color="0.4", ls=":", lw=1.5)
        ax.set_xlabel("DD deduction depth T")
        ax.set_title(f"rho = {rho}", size=MED)
        ax.set_xticks(sorted({T for r, c in depth if r == rho for T in depth[(r, c)]}))
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.06)
    np.atleast_1d(axes)[0].set_ylabel("solve_rate")
    # Legend goes in the rho=0.75 panel, whose lower half is empty; in the
    # rho=0.60 panel it lands on the ff=1792 curve.
    np.atleast_1d(axes)[-1].legend(loc="lower left", fontsize=SMALL - 1)
    fig.suptitle("(d) HOLDS: the width axis reintroduces no depth-dependent L*",
                 size=MED + 3, weight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0.28, 1, 0.93))
    footer(fig, "Finding 11 measured L*(T) at one MLP width, so this checks the flatness is not an "
                "artifact of ff=1024. It is not: both thin-deep cells clear 0.90 at EVERY depth bin "
                "at both densities (minimum 0.96, at T=5), and neither L=2 cell clears it at ANY bin. "
                "There is no T at which the ranking flips, so no depth-dependent threshold appears "
                "on this axis either. Each L=2 curve is also flat in T on its own terms -- ff=1792 "
                "runs 0.82-0.86 at rho=0.75 and 0.63-0.70 at rho=0.60, ff=4096 runs 0.68-0.70 and "
                "0.34-0.41 -- so the deficit is a roughly constant offset per cell rather than "
                "something that grows with the number of rounds. That is the same signature finding "
                "11 read as a fixed cost paid before chaining, now seen at ff=1792 and ff=4096 as "
                "well as at the ff=1024 it was measured on. Bin counts are 92-352 instances; these "
                "are the per-run depth bins, not the shared pool, so they are not the metric of "
                "record -- the margins here are far larger than that distinction.")
    pdf.savefig(fig)


def page_width_pathology(pdf, pool, masks, params):
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    ladder = ["L02_ff01024", "L02_ff01792", "L02_ff04096"]
    axes[0].plot([params[t] / 1e6 for t in ladder],
                 [pool[t]["solve_rate"] for t in ladder], "D-", color=WIDE, lw=2.5, ms=12)
    for t, off in zip(ladder, [(6, -20), (0, -24), (0, 18)]):
        axes[0].annotate(f"ff={int(t[6:])}", (params[t] / 1e6, pool[t]["solve_rate"]),
                         textcoords="offset points", xytext=off, ha="center", size=SMALL)
    axes[0].margins(y=0.14)
    axes[0].set_xlabel("parameters (M)")
    axes[0].set_ylabel("shared-pool solve_rate  (n = 1500)")
    axes[0].set_title("MLP width at L = 2 is non-monotone", size=MED)
    axes[0].grid(alpha=0.3)

    # The blue minimum sits at the last epoch, so its label has to go up and to
    # the left or it runs off the panel.
    for t, col, off in [("L02_ff01792", WIDE, (-8, -24)),
                        ("L02_ff04096", "tab:orange", (-8, -24)),
                        ("L06_ff00256", DEEP, (-8, 14))]:
        ep, nll = load_history(t)
        if not ep:
            continue
        axes[1].plot(ep, nll, "-", color=col, lw=2, label=label_of(t))
        best = int(np.argmin(nll))
        axes[1].plot([ep[best]], [nll[best]], "o", color=col, ms=11)
        axes[1].annotate(f"min {nll[best]:.3f} at ep {ep[best]}", (ep[best], nll[best]),
                         textcoords="offset points", xytext=off, ha="right",
                         size=SMALL - 1, color=col)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation nll")
    axes[1].set_title("and it is not an epoch-budget artifact", size=MED)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=SMALL - 1)
    fig.suptitle("Trade-off prediction (c) FAILS -- the one thing here that is unexplained",
                 size=MED + 3, weight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0.30, 1, 0.93))
    d, w, l, p = paired(masks, "L02_ff04096", "L02_ff01792")
    footer(fig, f"(c) pre-registered that solve_rate is monotone in ff at fixed L=2, as a check that "
                f"the wide cells are trained long enough. It fails: ff=1792 scores "
                f"{pool['L02_ff01792']['solve_rate']:.3f} and ff=4096 falls back to "
                f"{pool['L02_ff04096']['solve_rate']:.3f} ({d:+.3f}, {w} to {l}, p = {p:.0e}) -- a "
                f"real drop, not noise, against a 0.02 tolerance. The sbatch said to read that as 150 "
                f"epochs being too few for the wide cells and to stop. The right panel refutes that "
                f"mechanism, and refutes it backwards: ff=4096's validation nll reaches its minimum "
                f"of 1.208 at epoch 67, less than half the budget, and then drifts UP by 0.019 to "
                f"1.227 by epoch 150. A cell that was short of epochs would still be descending at "
                f"the end; this one turned over at the halfway mark, which is mild overfitting rather "
                f"than under-training. The thin-deep cell is the only one of the three whose minimum "
                f"sits at the final epoch. So (c)'s failure is NOT confounded with "
                f"the epoch budget, and the headline does not rest on it: (a) is measured entirely "
                f"between L=2/ff=1024 and L=4/ff=256, neither of which is a wide cell. What remains "
                f"unexplained is why width at L=2 should hurt at all. Reported, not resolved.")
    pdf.savefig(fig)


def page_scorecard(pdf, pool, masks, params, depth):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.93, "Scorecard", ha="center", size=BIG, weight="bold")

    d1, _, _, p1 = paired(masks, "L04_ff00256", "L02_ff01024")
    d2, w2, l2, p2 = paired(masks, "L06_ff00256", "L03_ff01024")
    d3, _, _, p3 = paired(masks, "L06_ff00256", "L12_ff01024")
    # (d) holds if every thin-deep cell clears 0.90 in its worst depth bin.
    worst = [min(v for v, _ in b.values())
             for (_, t), b in depth.items() if "ff00256" in t]
    clears = all(v >= 0.90 for v in worst)
    rows = [
        ("training (a)", "HOLDS",
         f"L=4/ff=256 reaches 0.90 at matched params; measured {pool['L04_ff00256']['solve_rate']:.3f} vs {pool['L02_ff01024']['solve_rate']:.3f}"),
        ("training (b)", "HOLDS",
         f"L=2/ff=4096 stays below 0.90 at {params['L02_ff04096'] / 1e6:.1f}M "
         f"({params['L02_ff04096'] / params['L03_ff01024']:.2f}x the knee): "
         f"{pool['L02_ff04096']['solve_rate']:.3f}"),
        ("training (c)", "FAILS",
         f"ff monotone at L=2: {pool['L02_ff01024']['solve_rate']:.3f} / "
         f"{pool['L02_ff01792']['solve_rate']:.3f} / {pool['L02_ff04096']['solve_rate']:.3f} "
         f"-- drops {abs(paired(masks, 'L02_ff04096', 'L02_ff01792')[0]):.3f} at ff=4096, "
         f"p = {paired(masks, 'L02_ff04096', 'L02_ff01792')[3]:.0e}"),
        ("training (d)", "HOLDS" if clears else "FAILS",
         f"L*(T) flat on the ff axis: ff=256 clears 0.90 at every T (worst {min(worst):.2f}), "
         f"L=2 at no T"),
        ("pool (a)", "HOLDS", f"depth beats width at 1.65M, paired p = {p1:.0e} (<< 0.001)"),
        ("pool (b)", "HOLDS",
         f"L=6/ff=256 >= L=3/ff=1024 at matched 2.44M: {d2:+.4f}, {w2} to {l2}, p = {p2:.0e} (< 0.05)"),
        ("pool (c)", "HOLDS",
         f"L=6/ff=256 within 0.02 of L=12/ff=1024 at 3.9x fewer params: {d3:+.4f}, p = {p3:.2f} (tied)"),
    ]
    y = 0.80
    for name, verdict, why in rows:
        col = {"HOLDS": "tab:green", "FAILS": "tab:red"}[verdict]
        fig.text(0.055, y, name, size=MED, family="monospace", color="0.25")
        fig.text(0.235, y, verdict, size=MED, weight="bold", color=col)
        fig.text(0.355, y, why, size=SMALL - 1, color="0.2")
        y -= 0.088
    footer(fig, "Seven pre-registrations across the two jobs, six held. The one failure, (c), is a "
                "sanity check rather than a hypothesis, and the training curves rule out the "
                "mechanism it was guarding against -- see the previous page. H1 (capacity) predicted "
                "that pool (a) would come in within 0.10 and that training (b) would clear 0.90; "
                "both went the other way, so the honest summary is that the L=3 knee is a serial "
                "requirement rather than a parameter count, with an unexplained width pathology at "
                "L=2 sitting next to it.")
    pdf.savefig(fig)


def page_vs_best(pdf, pool, params):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    cells = ["L02_ff01024", "L03_ff01024", "L04_ff00256", "L06_ff00256",
             "L04_ff01024", "L12_ff01024"]
    xs = np.arange(len(cells) + 1)
    vals = [pool[t]["solve_rate"] for t in cells] + [0.9960]
    cols = [DEEP if "ff00256" in t else WIDE for t in cells] + ["0.45"]
    ax.bar(xs, vals, 0.62, color=cols)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f"{v:.4f}", ha="center", size=SMALL + 1, weight="bold")
    names = [f"{label_of(t)}\n{params[t] / 1e6:.2f}M" for t in cells]
    names.append("L=12, ff=1024\n9.55M, 300 ep\nn_train=80000")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, size=SMALL - 2)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("shared-pool solve_rate")
    ax.set_title("Against the project's previous best on the same eval",
                 size=MED + 3, weight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    footer(fig, "The grey bar is the standing best, 0.9960 at n=1500 (48912548 trained, 48999377 "
                "scored; 0.9955 when re-scored at n=6000), so it is on exactly the same pool and "
                "sample size as the coloured bars. It is NOT beaten here -- it has 4x the data and "
                "2x the epochs, "
                "both of which finding 6 says are the strongest levers available. Nothing in this "
                "deck is a new state of the art, and it was not built to be one: every cell here is "
                "held at 150 epochs and n_train=20000 so that only the allocation moves. The "
                "comparison that matters is within the coloured bars, where L=6/ff=256 at 2.45M "
                "matches L=12/ff=1024 at 9.55M under an identical recipe. The obvious follow-up is "
                "to spend that efficiency rather than bank it: the best recipe so far has never been "
                "run thin-deep, and L=12/ff=256 would cost about 4.8M against the current 9.55M.")
    pdf.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    pool, masks = load_pool()
    params, depth = load_params(), load_depth()
    print(f"source: {len(pool)} cells on the shared pool, "
          f"{len(depth)} (rho, cell) depth series")
    path = os.path.join(OUT, "depth_vs_width_at_matched_params.pdf")
    with PdfPages(path) as raw:
        pdf = PdfAndPng(raw, OUT)
        page_title(pdf, pool, masks, params)
        page_budgets(pdf, pool, masks, params)
        page_pareto(pdf, pool, masks, params)
        page_depth_bins(pdf, depth)
        page_width_pathology(pdf, pool, masks, params)
        page_scorecard(pdf, pool, masks, params, depth)
        page_vs_best(pdf, pool, params)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB, {pdf.n} pages)")


if __name__ == "__main__":
    main()
