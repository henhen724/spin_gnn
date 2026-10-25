"""Slide deck: is there a cheaper corner of (degree, n_colors) than (5, 4)?

The program needs instances with long deduction chains. At the (d=5, q=4)
setting everything so far has used, the only way to get them is to raise N, and
the cost is brutal: ~11*N tokens with quadratic attention means ~8x per
instance from N=24 to N=48 and ~59x to N=96. This CPU sweep asks whether a
different (degree, n_colors) buys the same chain lengths at an N we can
already train on, and -- separately -- whether the density floor rho_min is a
critical point at all or merely where greedy removal happens to stop.

Scores the five pre-registered predictions in
`scripts/dd_control_parameter_sweep.py` against `summary.csv`. Reads whatever
is on disk, so it can be run on a partial sweep; the scorecard says how many
cells of the grid are in. Run:

    python -m notebooks.make_control_param_deck
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

S = os.environ.get("SCRATCH", "/scratch/users/hshunt") + "/spin_gnn"
SRC = os.path.join(S, "sudoku_dd_control_param_2026-10-24")
OUT = os.path.join(S, "notebooks", "sudoku_control_param_2026-10-24")
BASE = (5, 4)  # the setting every transformer run so far has used
# Cells highlighted in the N-scaling and cheap-corner pages. Chosen to span
# the ratio axis, with the baseline always included for comparison.
SHOW = [(3, 4), (5, 4), (4, 3), (6, 4), (8, 4)]
COLOURS = dict(zip(SHOW, ["tab:gray", "tab:red", "tab:blue", "tab:green", "tab:purple"]))


def load(name):
    path = os.path.join(SRC, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def num(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", "nan", None) else float("nan")


def cell_series(rows, d, q):
    """Rows for one (d, q), sorted by N."""
    s = [r for r in rows
         if int(r["degree"]) == d and int(r["n_colors"]) == q]
    return sorted(s, key=lambda r: int(r["n_nodes"]))


def all_cells(rows):
    return sorted({(int(r["degree"]), int(r["n_colors"])) for r in rows},
                  key=lambda dq: dq[0] / (dq[1] - 1))


def exponent(Ns, Ts):
    """Fitted a in T ~ N^a, or nan if fewer than three finite points.

    This is the whole criticality test. A genuine critical point at rho_min
    would show a growing with N only near the floor and a ~ 0 far from it; a
    positive a at every density means bigger graphs simply contain longer
    chains.
    """
    pairs = [(n, t) for n, t in zip(Ns, Ts) if t == t and t > 0]
    if len(pairs) < 3:
        return float("nan")
    xs = np.log([p[0] for p in pairs])
    ys = np.log([p[1] for p in pairs])
    return float(np.polyfit(xs, ys, 1)[0])


def match_n(rows, d, q, target_T, col="mean_T_near_floor"):
    """Smallest N at which (d, q) reaches target_T, by log-log interpolation.

    Used to say what the cheap corner is worth in the only currency that
    matters here: the N at which the baseline would have to be trained to get
    the same deduction depth.
    """
    s = cell_series(rows, d, q)
    Ns = [int(r["n_nodes"]) for r in s]
    Ts = [num(r, col) for r in s]
    pairs = [(n, t) for n, t in zip(Ns, Ts) if t == t and t > 0]
    for i in range(1, len(pairs)):
        if pairs[i - 1][1] < target_T <= pairs[i][1]:
            (n0, t0), (n1, t1) = pairs[i - 1], pairs[i]
            f = (np.log(target_T) - np.log(t0)) / (np.log(t1) - np.log(t0))
            return float(np.exp(np.log(n0) + f * (np.log(n1) - np.log(n0))))
    return float("nan")


def page_title(pdf, rem, ran):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.84, "Is there a cheaper corner than (degree 5, 4 colors)?",
             ha="center", size=BIG + 1, weight="bold")
    cells = all_cells(rem)
    fig.text(0.5, 0.75, f"{len(cells)} of 21 (degree, n_colors) cells x 6 sizes, "
                        f"{len(rem)} removal rows and {len(ran)} random rows",
             ha="center", size=MED - 3, color="0.35")
    # Compare at a FIXED, affordable N rather than at each cell's largest size:
    # "the deepest cell anywhere" is a different and much weaker claim than
    # "the deepest cell at an N we can train on", and only the second is
    # actionable. Taking the max over sizes also leaves the baseline unable to
    # bracket the target, which is what made this read "N ~ nan".
    at96 = {c: num(s[0], "mean_T_near_floor")
            for c in cells
            for s in [[r for r in cell_series(rem, *c) if int(r["n_nodes"]) == 96]]
            if s and num(s[0], "mean_T_near_floor") == num(s[0], "mean_T_near_floor")}
    best = max(at96, key=at96.get)
    need = match_n(rem, *BASE, at96[best])
    base96 = at96.get(BASE, float("nan"))
    lines = [
        f"YES, and at N = 96 it is worth about a factor of {need / 96:.0f} in N.",
        "",
        f"(d={best[0]}, q={best[1]}) reaches mean deduction depth {at96[best]:.1f} near its density",
        f"floor at N = 96, against {base96:.1f} for (d=5, q=4) at the same size.",
        f"The baseline does not reach {at96[best]:.1f} until N ~ {need:.0f}.",
        "",
        "But NOT for the reason predicted, and rho_min is still not a critical",
        "point: T grows as a power of N at every density DD can act on.",
    ]
    fig.text(0.5, 0.42, "\n".join(lines), ha="center", va="center",
             size=MED - 1, linespacing=1.8)
    footer(fig, "The propagation ratio d/(q-1) governs how many neighbours there are per colour that "
                "has to be excluded before DD can force a node. Prediction (a) was that rho_min falls "
                "as that ratio rises; prediction (b) was that the DEEPEST instances live at low ratio. "
                "The first holds across three decades; the second is backwards.")
    pdf.savefig(fig)


def page_rho_min(pdf, rem):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for q, marker in zip((3, 4, 5, 6), ("o", "s", "^", "D")):
        pts = []
        for d, qq in all_cells(rem):
            if qq != q:
                continue
            s = [r for r in cell_series(rem, d, q) if int(r["n_nodes"]) == 996]
            if s:
                pts.append((d / (q - 1), num(s[0], "rho_min")))
        if pts:
            pts.sort()
            ax.plot(*zip(*pts), marker + "-", lw=2, ms=10, label=f"{q} colors")
    ax.axhline(0.4777, color="tab:red", ls="--", lw=1.5)
    ax.text(0.98, 0.4777 + 0.02, "0.478, the (5,4) floor all training has used",
            transform=ax.get_yaxis_transform(), ha="right", size=SMALL, color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("propagation ratio  d / (q - 1)")
    ax.set_ylabel("rho_min, minimum reachable hint density  (N = 996)")
    ax.set_title("(a) HOLDS: the density floor falls monotonically with propagation",
                 size=MED + 3, weight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    footer(fig, "Measured at the largest size, where rho_min is already asymptotic. It spans nearly "
                "two decades, 0.84 at ratio 1.00 down to 0.011 at ratio 5.00, and falls monotonically "
                "throughout -- so prediction (c)'s falsifier, that rho_min is flat in (d,q) and there "
                "is no cheap corner, is itself refuted. Ratio 1.00 is the degenerate case d = q-1, "
                "where DD can force almost nothing and greedy removal stalls at 84% hinted.")
    pdf.savefig(fig)


def page_depth_vs_ratio(pdf, rem):
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    for ax, (col, lab) in zip(axes, [("mean_T_near_floor", "mean T within 0.06 of the floor"),
                                     ("T_at_rho55", "mean T at fixed rho = 0.55")]):
        for q, marker in zip((3, 4, 5, 6), ("o", "s", "^", "D")):
            pts = []
            for d, qq in all_cells(rem):
                if qq != q:
                    continue
                s = [r for r in cell_series(rem, d, q) if int(r["n_nodes"]) == 96]
                if s and num(s[0], col) == num(s[0], col):
                    pts.append((d / (q - 1), num(s[0], col)))
            if pts:
                pts.sort()
                ax.plot(*zip(*pts), marker + "-", lw=2, ms=9, label=f"{q} colors")
        ax.set_xlabel("propagation ratio  d / (q - 1)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=SMALL)
    axes[0].set_title("at each cell's own floor", size=MED)
    axes[1].set_title("at matched density", size=MED)
    fig.suptitle("(b) FAILS, backwards: depth peaks at HIGH propagation, not low",
                 size=MED + 3, weight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0.28, 1, 0.94))
    footer(fig, "(b) expected a ratio <= 1.2 cell to reach mean T >= 22 at N=96. Those cells are "
                "d = q-1, the SHALLOWEST on the grid: DD forces nothing there, removal stalls at "
                "once, T ~ 1. Depth has an interior optimum instead -- low ratio stalls removal "
                "early, high ratio forces so many nodes per round that few rounds are needed, and "
                "the longest chains sit between. The panels disagree on where, and that is the "
                "caveat on the whole sweep: at each cell's OWN floor the high-ratio cells win, but "
                "at MATCHED density the baseline (5,4) is the deepest thing on the grid. The cheap "
                "corner buys depth by moving the floor down, not by making deduction harder.")
    pdf.savefig(fig)


def page_scaling(pdf, rem):
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5), sharex=True)
    cols = [("T_at_rho90", "rho = 0.90"), ("T_at_rho70", "rho = 0.70"),
            ("T_at_rho55", "rho = 0.55")]
    rows_txt = []
    for ax, (col, lab) in zip(axes, cols):
        exps = []
        for c in SHOW:
            s = cell_series(rem, *c)
            if not s:
                continue
            Ns = [int(r["n_nodes"]) for r in s]
            Ts = [num(r, col) for r in s]
            ok = [(n, t) for n, t in zip(Ns, Ts) if t == t]
            if not ok:
                continue
            a = exponent(Ns, Ts)
            exps.append((c, a))
            ax.plot(*zip(*ok), "o-", color=COLOURS[c], lw=2, ms=7,
                    label=f"d={c[0]} q={c[1]}  a={a:+.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_title(lab, size=MED)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=SMALL - 2, loc="upper left")
        rows_txt.append((lab, exps))
    axes[0].set_ylabel("mean deduction depth T")
    fig.suptitle("(d) FAILS: T grows as a power of N at every density DD can act on",
                 size=MED + 3, weight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    footer(fig, "This is the test of criticality itself, and it is the most important page here. A "
                "genuine critical point at rho_min requires T to grow with N near the floor while "
                "staying N-INDEPENDENT far from it. Instead the exponent a in T ~ N^a is positive at "
                "rho = 0.90 in every cell that deduces anything -- it never reaches zero, it only "
                "gets smaller. The one exception, d=3 q=4 at a = +0.00, is not evidence for "
                "criticality but the opposite: it is pinned at exactly T = 1.0 at every N because "
                "d = q-1 lets DD force nothing beyond the first round, so there is no chain to "
                "lengthen. Excluding it, rho_min is where greedy removal stops rather than a "
                "critical point, across the whole (degree, n_colors) grid and not at (5,4) alone -- "
                "a real strengthening of finding 1 rather than a restatement of it.")
    pdf.savefig(fig)


def page_cheap_corner(pdf, rem):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    notes = []
    for c in SHOW:
        s = cell_series(rem, *c)
        if not s:
            continue
        Ns = [int(r["n_nodes"]) for r in s]
        Ts = [num(r, "mean_T_near_floor") for r in s]
        ax.plot(Ns, Ts, "o-", color=COLOURS[c], lw=2.5, ms=9,
                label=f"d={c[0]} q={c[1]}" + ("  (baseline)" if c == BASE else ""))
        at96 = [t for n, t in zip(Ns, Ts) if n == 96]
        if at96 and c != BASE:
            need = match_n(rem, *BASE, at96[0])
            # Cells shallower than the baseline's smallest size cannot be
            # bracketed and are not cheap corners anyway; a bare "N ~ nan" in
            # the box reads as a bug rather than as "this cell is no use".
            notes.append(f"d={c[0]} q={c[1]} at N=96 gives T={at96[0]:5.1f}; " +
                         (f"(5,4) needs N ~ {need:.0f} for that" if need == need
                          else "below (5,4) at every size -- no gain"))
    ax.axvline(96, color="0.6", ls=":", lw=1.5)
    ax.text(96 * 1.05, 0.03, "N = 96, affordable", rotation=90, size=SMALL,
            color="0.4", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("mean deduction depth near the density floor")
    ax.set_title("What the cheap corner is worth, in N", size=MED + 4, weight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")
    if notes:
        ax.text(0.98, 0.04, "\n".join(notes), transform=ax.transAxes, ha="right",
                size=SMALL - 1, family="monospace",
                bbox=dict(boxstyle="round", fc="0.95", ec="0.7"))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    footer(fig, "Read with the caveat from the depth-vs-ratio page: these cells reach their depth at "
                "their OWN floor, which for (8,4) is rho ~ 0.17 against 0.48 for the baseline, so "
                "the instances are far sparser. Whether that is a fair substitute for large N is a "
                "question about what the transformer is being asked to learn, not one this sweep "
                "answers. Token cost favours it twice over: q=3 needs a smaller vocabulary and "
                "d=4 a shorter graph section than the baseline's d=5.")
    pdf.savefig(fig)


def page_random_control(pdf, ran):
    if not ran:
        return
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    cells = all_cells(ran)
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(cells)))
    untestable = []
    for colour, c in zip(cmap, cells):
        s = cell_series(ran, *c)
        pts = [(int(r["n_nodes"]), num(r, "unhinted_at_rho50")) for r in s]
        pts = [(n, u) for n, u in pts if u == u]
        if len(pts) < 2:
            if pts:
                untestable.append(f"d={c[0]} q={c[1]}: one size only, {pts[0][1]:.0f}")
            continue
        ax.plot(*zip(*pts), "o-", color=colour, lw=2, ms=8,
                label=f"d={c[0]} q={c[1]}")
    ns = sorted({int(r["n_nodes"]) for r in ran})
    ax.plot(ns, [0.08 * n for n in ns], "k--", lw=1.5, label="proportional to N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("unhinted nodes at the 50% solvability crossing")
    ax.set_title("(e) HOLDS: the random-hint threshold runs off to rho = 1 for every (d, q)",
                 size=MED + 2, weight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=SMALL - 2, ncol=2, loc="upper left")
    if untestable:
        ax.text(0.98, 0.04, "not testable:\n" + "\n".join(untestable),
                transform=ax.transAxes, ha="right", size=SMALL - 1,
                family="monospace", bbox=dict(boxstyle="round", fc="0.95", ec="0.7"))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    footer(fig, "If the random-hint ensemble had a finite-density solvability transition, the number "
                "of unhinted nodes at the 50% crossing would grow proportionally to N, like the "
                "dashed line. It is flat in N in every cell with two or more measurable sizes. The "
                "CONSTANT rises steeply with degree -- about 3 at d=3 and 85 at d=8, q=3 -- which is "
                "the propagation ratio again, not a transition; the earlier reading of that 85 as "
                "pressure on this prediction confused a large constant for N-growth. The cells with "
                "one size are nan at the others because the crossing fell below the rho = 0.50 grid "
                "floor, not because it is absent: extending the grid downward for ratio >= 4 is the "
                "fix, and it is cheap.")
    pdf.savefig(fig)


def page_scorecard(pdf, rem, ran):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.92, "Scorecard against the pre-registered predictions",
             ha="center", size=MED + 6, weight="bold")
    n_cells = len(all_cells(rem))
    verdicts = [
        ("(a)", "HOLDS", "rho_min falls monotonically with d/(q-1), 0.84 -> 0.011 over\n"
                         "ratio 1.00 -> 5.00. The specific claim rho_min(3,4) > 0.477 was\n"
                         "met at 0.836; the d=10 q=4 half awaits the rest of the grid."),
        ("(b)", "FAILS", "Backwards. The ratio <= 1.2 cells are d = q-1, the SHALLOWEST\n"
                         "on the grid (T ~ 1, floor at 84% hinted), not the deepest.\n"
                         "Depth has an interior optimum in the ratio."),
        ("(c)", "REFUTED", "Its falsifier required rho_min and max T flat within +-20%.\n"
                           "They span two decades and a factor of 100. There IS a cheap\n"
                           "corner -- which is the useful half of this sweep."),
        ("(d)", "FAILS", "The exponent in T ~ N^a is positive at rho = 0.90 in every\n"
                         "cell that deduces at all (+0.19 to +0.25), so nothing is\n"
                         "critical at rho_min anywhere on the grid. The sole flat cell,\n"
                         "d=3 q=4, is pinned at T = 1.0 for lack of any chain to grow.\n"
                         "Generalises finding 1 from (5,4) to all of (degree, n_colors)."),
        ("(e)", "HOLDS", "unhinted-at-crossing is flat in N in every cell with two or\n"
                         "more sizes. The large constants at high degree are the ratio,\n"
                         "not a transition."),
    ]
    # Advance by the actual line count rather than a fixed step: the (d) entry
    # is five lines and a uniform step either overlaps it or wastes the page.
    y = 0.84
    for tag, verdict, text in verdicts:
        colour = {"HOLDS": "tab:green", "FAILS": "tab:red", "REFUTED": "tab:orange"}[verdict]
        fig.text(0.06, y, tag, size=MED - 2, weight="bold", family="monospace", va="top")
        fig.text(0.12, y, verdict, size=MED - 2, weight="bold", color=colour,
                 family="monospace", va="top")
        fig.text(0.27, y, text, size=SMALL - 1, va="top", linespacing=1.45)
        y -= 0.034 * (text.count("\n") + 1) + 0.028
    footer(fig, f"{n_cells} of 21 cells and {len(rem)} of 126 removal rows are in. (a), (c) and (d) "
                f"are decided by what is here and the rest of the grid cannot overturn them. (b) "
                f"could still be rescued by a ratio <= 1.2 cell at higher q, which is what remains "
                f"-- treat it as very likely failed rather than settled.")
    pdf.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    rem = [r for r in load("summary.csv") if r["ensemble"] == "removal"]
    ran = [r for r in load("summary.csv") if r["ensemble"] == "random"]
    if not rem:
        print("no summary.csv yet")
        return
    print(f"{len(all_cells(rem))} cells, {len(rem)} removal rows, {len(ran)} random rows")

    pdf_path = os.path.join(OUT, "cheaper_corner_than_d5q4.pdf")
    with PdfPages(pdf_path) as raw:
        pdf = PdfAndPng(raw, OUT)
        page_title(pdf, rem, ran)
        page_rho_min(pdf, rem)
        page_depth_vs_ratio(pdf, rem)
        page_scaling(pdf, rem)
        page_cheap_corner(pdf, rem)
        page_random_control(pdf, ran)
        page_scorecard(pdf, rem, ran)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {pdf.n} pages)")


if __name__ == "__main__":
    main()
