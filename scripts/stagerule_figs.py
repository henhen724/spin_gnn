#!/usr/bin/env python3
"""Figures for stage RULE. Reuses stagerule_fit's loader so the numbers in the
deck and the numbers in VERDICT.txt cannot drift apart.

Two plots only. The R5 ladder and the per-seed table are markdown in the deck,
because a results page is a plot OR a table and never both.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stagerule_fit import (load, per_cell, OUT, GATE, R1_BAR,  # noqa: E402
                           D1_THREE, J2_LABEL)
from stagegate_fit import spearman  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes  # noqa: E402

FIGS = os.path.join(OUT, "figures")
DPI = 150

# Twelve of the 31 architectures measure h_rule at EXACTLY zero, which has no
# place on a log axis. They are drawn as open markers pinned to this floor and
# the deck body says so on every page that shows it. Pretending a zero is 1e-6
# without saying it is the kind of silent choice this project reads figures
# back to catch.
FLOOR = 1e-6


def shortname(cell):
    """'L=16 h=4 no_mlp d192' -> 'L16 no_mlp d192'. The head count is 4 on
    every rung in this ladder, so printing it wastes label width."""
    return cell.replace("h=4 ", "").replace("L=", "L")


def classify(a, broken):
    if a["cell"] in broken:
        return "broken", SERIES[1]
    if a["rho50"] == a["rho50"] and a["rho50"] <= GATE:
        return "gate", SERIES[0]
    return "other", MUTED


def yval(h):
    return FLOOR if (h != h or h <= 0) else max(h, FLOOR)


def scatter(ax, arch, broken, xkey):
    """One marker per architecture. Open face for a measured zero, so the
    reader can tell 'none observed' from 'small but nonzero' at a glance."""
    for a in arch:
        x = a[xkey]
        if x != x:
            continue
        kind, col = classify(a, broken)
        zero = not (a["h_rule"] > 0)
        ax.plot(x, yval(a["h_rule"]), "o", ms=7 if kind != "other" else 5,
                color=col, mfc="none" if zero else col,
                mew=1.6, zorder=3 if kind == "broken" else 2,
                rasterized=True)


def fig_params(arch, broken):
    f, ax = plt.subplots(figsize=(7.2, 4.3))
    scatter(ax, arch, broken, "params")
    ax.axhline(R1_BAR, ls="--", lw=1.2, color=MUTED, zorder=1)
    ax.annotate("R1 bar, 0.01", xy=(1.02, R1_BAR), xycoords=("axes fraction",
                "data"), va="center", fontsize=9, color=MUTED)
    for a in arch:
        if a["cell"] in broken:
            ax.annotate(shortname(a["cell"]),
                        xy=(a["params"], yval(a["h_rule"])),
                        xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=SERIES[1])
    j2 = next(a for a in arch if a["cell"] == J2_LABEL)
    ax.annotate("J2, the standing minimum",
                xy=(j2["params"], yval(j2["h_rule"])), xytext=(-12, -15), ha="right",
                textcoords="offset points", fontsize=9, color=SERIES[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2, 0.2)
    style_axes(ax, "trainable parameters", "rule hazard  $h^{\\mathrm{rule}}$")
    f.tight_layout()
    p = os.path.join(FIGS, "rule_vs_params.png")
    f.savefig(p, dpi=DPI)
    plt.close(f)
    print("  wrote", os.path.basename(p))


def fig_gate(arch, broken):
    f, ax = plt.subplots(figsize=(7.2, 4.3))
    scatter(ax, arch, broken, "rho50")
    ax.axvline(GATE, ls="--", lw=1.2, color=MUTED, zorder=1)
    ax.annotate("gate 0.740", xy=(GATE, 0.13), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=9,
                color=MUTED)
    xs = [a["h_rule"] for a in arch
          if a["h_rule"] == a["h_rule"] and a["rho50"] == a["rho50"]]
    ys = [a["rho50"] for a in arch
          if a["h_rule"] == a["h_rule"] and a["rho50"] == a["rho50"]]
    ax.annotate(r"Spearman $\rho_s = %+.2f$" % spearman(xs, ys),
                xy=(0.03, 0.93), xycoords="axes fraction", fontsize=10,
                color="#333333")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2, 0.2)
    style_axes(ax, r"published $\rho_{50}$ (a colouring criterion, never a DD one)",
               "rule hazard  $h^{\\mathrm{rule}}$")
    f.tight_layout()
    p = os.path.join(FIGS, "rule_vs_gate.png")
    f.savefig(p, dpi=DPI)
    plt.close(f)
    print("  wrote", os.path.basename(p))


def main():
    os.makedirs(FIGS, exist_ok=True)
    cells, viol, _, _ = load()
    assert viol["in_F_gt_at_risk"] == 0 and viol["err_in_F_gt_err"] == 0, \
        "D2 must pass before any figure is drawn"
    arch = per_cell(cells)
    assert len(arch) == 31, "expected 31 architectures, got %d" % len(arch)

    # Resolve finding 100's three by the counts it published, exactly as the
    # scorer's D1 does -- never by pattern alone, since 'no_mlp d160' matches
    # both the L=12 and the L=16 rung. The counts must be summed over ALL 14
    # densities, because that is the grid 100 pooled over; per_cell() pools the
    # ten FITTED ones and gives 810 where 100 published 977, which is a
    # different (and equally correct) number for a different question.
    tot = {}
    for c in cells:
        tot[c["cell"]] = tot.get(c["cell"], 0) + int(round(c["err_in_F"]))
    broken = set()
    for frag, want in D1_THREE.items():
        hits = [c for c, n in tot.items() if frag in c and n == want]
        assert len(hits) == 1, "cannot resolve %r at %d: %s" % (frag, want, hits)
        broken.add(hits[0])

    fig_params(arch, broken)
    fig_gate(arch, broken)

    with open(os.path.join(FIGS, "SOURCES.txt"), "w") as fh:
        fh.write("stage RULE figures\n")
        fh.write("loader: scripts/stagerule_fit.py load() + per_cell()\n")
        fh.write("data:   runs/stagelogic_2026-11-17/F_hazard_all_*of10.csv\n")
        fh.write("        runs/stagelogic_2026-11-17/LF_summary_all_*of10.csv\n")
        fh.write("broken (finding 100's three): %s\n" % sorted(broken))
        fh.write("zeros drawn as open markers at the floor %g\n" % FLOOR)
    print("  wrote SOURCES.txt")


if __name__ == "__main__":
    sys.exit(main())
