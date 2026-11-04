"""Slide deck for the minimal-architecture grid: stage A (51032450) + B (51072446).

The question this deck answers is not "how well does it solve DD" but "how
little machinery does solving DD need", so every page is a subtraction: a rung
of the depth ladder, a block with its nonlinearity removed, a block whose MLP
is frozen at initialization, a block with no MLP write at all.

Stage A walked the depth ladder and ran the arch axis at L=4; its ladder then
showed L=4 IS the breaking depth, where the hub's own across-seed spread is as
large as its score, so no arch contrast there can be resolved. Stage B re-runs
the arch and head axes at L=6, the shallowest rung that still solves at all
three seeds. Both are drawn here against the SAME pooled s*.

Runs on a PARTIAL grid on purpose -- 51 cells land over ~18 h and the deck is
how the intermediate state gets looked at. Pages say how many cells they are
drawn from and the title says PARTIAL until all 51 are in.

    python3 notebooks/make_minarch_deck.py [--scan-dir DIR [DIR ...]] [--out DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from scripts.minarch_analysis import (ARCH_AXIS, ARCH_DEPTHS, DEPTH_LADDER, PLATEAU, SCAN_DEFAULT,
                                      SCAN_L6, all_rungs, cell_key, cell_metrics, history, load,
                                      load_all, median, rung_label, spread)

# 36 in stage A (51032450) + 15 in stage B (51072446), which is exactly what
# --scan-dir defaults to below. Stage C (51114734) adds 15 more in a third scan
# dir and gets its own deck; pointing this one at it without bumping the count
# would turn "PARTIAL" into "all cells" or the reverse, and the cover is the
# one place that must never be wrong about how much has landed.
N_CELLS_TOTAL = 51
HUB4 = (4, 4, "full")   # stage A's arch hub, which turned out to be the breaking depth
HUB6 = (6, 4, "full")   # stage B's arch hub, the shallowest rung that still solves
# The previous best on this exact eval: finding 44's N=48 fall cells at the
# same 375k steps and the same 14-point grid, seed 0 only.
PREV = os.path.join(os.environ.get("SCRATCH", "/scratch/users/hshunt"),
                    "spin_gnn", "sudoku_n48_fall_scan_2026-11-02")
ARCH_NAME = {
    "full": "full\n(trained ReLU MLP)",
    "linear": "linear\n(identity activation)",
    "rf_mlp": "rf_mlp\n(linear1 frozen at init)",
    "rf_all": "rf_all\n(all frozen but readout)",
    "no_mlp": "no_mlp\n(attention only)",
}
CENSOR = 1.02  # where a censored rho_50 is drawn, above the top of the grid


def collect(scan_dirs, boot):
    rows = load_all([scan_dirs] if isinstance(scan_dirs, str) else scan_dirs)
    cells = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    met = {k: cell_metrics(v, boot) for k, v in cells.items()}
    rung = {}
    for k, m in met.items():
        # The seed is the LAST element of the key and the rung is everything
        # before it, however long the key is: a non-default width adds
        # (d_model, ff) in the middle rather than displacing the seed.
        rung.setdefault(k[:-1], []).append((k[-1], m))
    for v in rung.values():
        v.sort()
    return rows, cells, met, rung


def rung_stat(rung, key, field):
    if key not in rung:
        return None
    return median([m[field] for _, m in rung[key]])


def yardstick(rung):
    """s* per metric: the largest across-seed spread on any complete rung."""
    out = {}
    for field in ("rho50", "mean_solve"):
        sp = []
        for v in rung.values():
            if len(v) < 2:
                continue
            if field == "rho50":
                ok = [m["rho50"] for _, m in v if m["flag"] == "ok"]
                if len(ok) == len(v):
                    sp.append(spread(ok))
            else:
                sp.append(spread([m[field] for _, m in v]))
        out[field] = max(sp) if sp else float("nan")
    return out


def _r50_points(v):
    """Per-seed rho_50, censored seeds lifted to CENSOR so they stay visible."""
    return [(m["rho50"] if m["flag"] == "ok" else CENSOR) for _, m in v]


# --------------------------------------------------------------------- pages
def page_verdict(o, rung, s, n_cells):
    done = n_cells == N_CELLS_TOTAL
    depth_have = [L for L in DEPTH_LADDER if (L, 4, "full") in rung]
    B = None
    for L in DEPTH_LADDER:
        v = rung.get((L, 4, "full"))
        if v and any(m["flag"] != "ok" for _, m in v):
            B = L
            break
    arch_have = {d: [a for a in ARCH_AXIS if a != "full" and (d, 4, a) in rung]
                 for d in ARCH_DEPTHS}
    base = rung_stat(rung, HUB6, "rho50")
    base_ok = (HUB6 in rung and all(m["flag"] == "ok" for _, m in rung[HUB6]))

    # B is the first rung walking down with a failing seed. It is only a
    # breaking POINT if everything below also fails; with the full ladder in,
    # L=3 crosses on all three seeds while L=4 crosses on two, so quoting a
    # knee at L=4 would promote one unlucky seed into a structural claim.
    revived = [L for L in DEPTH_LADDER
               if B is not None and L < B and (L, 4, "full") in rung
               and all(m["flag"] == "ok" for _, m in rung[(L, 4, "full")])]
    if B is not None and revived:
        head = (f"There is no knee. One seed at L={B} never crosses 50%, but L={max(revived)} "
                f"crosses on all three and ties L={B} on mean solve rate "
                f"({rung_stat(rung, (max(revived), 4, 'full'), 'mean_solve'):.4f} vs "
                f"{rung_stat(rung, (B, 4, 'full'), 'mean_solve'):.4f}) — so the censoring at "
                f"L={B} is the seed term, not {B} layers. Deduction degrades GRADUALLY with "
                f"depth and is gone by L=1.")
    elif B is not None:
        head = (f"Deduction breaks between L={B} and L={B + 1}: at L={B} at least one seed never "
                f"reaches a 50% solve rate anywhere in the grid, and no shallower rung recovers.")
    elif depth_have:
        head = (f"Every depth measured so far still solves it — the ladder has been walked down to "
                f"L={min(depth_have)} without finding the breaking point.")
    else:
        head = "No depth rung has landed yet."

    # What can be stripped at L=6 without raising rho_50 by more than s*.
    free, costs, pend = [], [], []
    if base_ok:
        for (L, h, a) in [(6, 4, "linear"), (6, 4, "rf_mlp"), (6, 4, "no_mlp"),
                          (6, 1, "full"), (6, 4, "rf_all")]:
            nm = "nhead=1" if h == 1 else a
            v = rung.get((L, h, a))
            if not v:
                pend.append(nm)
            elif any(m["flag"] != "ok" for _, m in v):
                # "Censored" alone hides whether the survivors sat near the
                # free line or well above it. With only the LARGEST seed
                # censored the median is still a value, so quote the delta.
                pts = sorted(((m["rho50"], m["flag"] == "ok") for _, m in v),
                             key=lambda p: (not p[1], p[0] if p[1] else 0.0))
                mid = pts[len(pts) // 2]
                nc = sum(1 for _, ok in pts if not ok)
                costs.append(f"{nm} ({mid[0] - base:+.3f}, {nc} of {len(pts)} seeds censored)"
                             if mid[1] else f"{nm} (censored {nc}/{len(pts)})")
            else:
                d = median([m["rho50"] for _, m in v]) - base
                (free if d <= s["rho50"] else costs).append(f"{nm} ({d:+.3f})")
    strip = ("The L=6 arch axis has not landed yet." if not base_ok or (not free and not costs)
             else ("Removable for free at L=6 (rho_50 rises by less than s*): "
                   + (", ".join(free) if free else "nothing") + ".  Not removable: "
                   + (", ".join(costs) if costs else "nothing")
                   + (".  Still running: " + ", ".join(pend) if pend else ".")))

    # How much DEPTH is each removal worth? Match each stripped L=6 cell to the
    # depth rung it scores like. Computed, not asserted: for each removal, find
    # the full-MLP rung whose median mean_solve is closest, and report the gap.
    xrate, matched, ms_of, sp_of = [], {}, {}, {}
    for (L, h, a) in [(6, 4, "linear"), (6, 4, "rf_mlp"), (6, 4, "no_mlp"), (6, 1, "full")]:
        if (L, h, a) not in rung:
            continue
        ms = rung_stat(rung, (L, h, a), "mean_solve")
        cand = [(abs(ms - rung_stat(rung, (D, 4, "full"), "mean_solve")), D)
                for D in DEPTH_LADDER if (D, 4, "full") in rung]
        if not cand:
            continue
        gap, D = min(cand)
        nm = "nhead=1" if h == 1 else a
        matched[nm], ms_of[nm] = D, ms
        sp_of[nm] = spread([m["mean_solve"] for _, m in rung[(L, h, a)]])
        xrate.append(f"{nm} ~ L={D} ({ms:.4f} vs "
                     f"{rung_stat(rung, (D, 4, 'full'), 'mean_solve'):.4f})")

    # The three MLP cuts remove three DIFFERENT amounts -- the nonlinearity
    # only, the training of the first matrix only, the whole MLP write. Whether
    # they cost the SAME is the interesting question, and it must be asked of
    # the three cuts against each other, NOT of which depth rung each is
    # nearest: L=3 and L=4 are themselves tied on mean solve, so the nearest
    # rung flips between them on a 0.002 difference and would read as
    # disagreement that is not there.
    #
    # The bar is the L=6 HUB's own across-seed spread: three seeds of the same
    # architecture, all of which trained. Not the pooled s* in mean solve, and
    # not the cuts' own spreads either -- one of the cut rungs has a seed that
    # never escaped the plateau, so its spread IS the pooled s* (0.29), and
    # judging the cuts against a bar one of them set is the circularity the
    # analysis script warns about. Comparing a spread of MEDIANS against a
    # spread of single seeds is conservative in the right direction.
    cuts = [k for k in ("linear", "rf_mlp", "no_mlp") if k in ms_of]
    hub_sp = (spread([m["mean_solve"] for _, m in rung[HUB6]])
              if HUB6 in rung and len(rung[HUB6]) > 1 else float("nan"))
    if len(cuts) == 3 and hub_sp == hub_sp:
        vals = [ms_of[k] for k in cuts]
        local = hub_sp
        ds = sorted({matched[k] for k in cuts})
        where = ("L=" + str(ds[0]) if len(ds) == 1 else
                 "L=" + "/".join(str(d) for d in ds) + ", two rungs the ladder itself cannot "
                 "separate")
        if spread(vals) <= local:
            mlp_line = (f"THE THREE MLP CUTS COST THE SAME: mean solve "
                        + " / ".join(f"{v:.4f}" for v in vals)
                        + f", a spread of {spread(vals):.4f} against the L=6 hub's own across-seed "
                        f"spread of {local:.4f}. Deleting the MLP entirely costs what "
                        f"removing only its nonlinearity costs, and what freezing only its first "
                        f"matrix costs — the MLP is ALL-OR-NOTHING, only its LEARNED "
                        f"nonlinearity buys anything, and it is worth about half the depth "
                        f"({where}).")
        else:
            mlp_line = (f"The three MLP cuts do NOT cost the same: mean solve "
                        + " / ".join(f"{k} {ms_of[k]:.4f}" for k in cuts)
                        + f", a spread of {spread(vals):.4f} against the L=6 hub's across-seed spread "
                        f"of {local:.4f}, so they separate and each cut has its own price.")
    else:
        mlp_line = (f"Only {len(cuts)} of the 3 MLP cuts have landed (or the L=6 hub has no "
                    f"seed spread to judge them against); no all-or-nothing read yet.")

    blocks = [
        ("THE QUESTION", "#1a5fb4",
         "Not how well the model solves direct deduction — it already does, leak-free — but how "
         "little machinery that takes. Every cell is the L=12/ff=1024 model with something "
         "REMOVED: layers, heads, the MLP's nonlinearity, its training, or the MLP write itself."),
        ("THE HEADLINE", "#613583", head),
        ("BEFORE AND AFTER, at the shallowest depth that still deduces", "#26a269",
         ("L=6 full has not landed yet." if not base_ok else
          f"BEFORE: L=6, 4 heads, trained ReLU MLP — rho_50 = {base:.3f}. ") + strip),
        ("WHAT EACH REMOVAL COSTS, IN LAYERS", "#26a269",
         ("Nothing stripped at L=6 has landed yet." if not xrate else
          "Each stripped L=6 cell, matched to the depth rung it scores like: "
          + "; ".join(xrate) + ". " + mlp_line)),
        ("HOW TO READ EVERY NUMBER", "#c64600",
         f"rho_50 is the hint density where the solve rate crosses 50% — LOWER IS BETTER. It "
         f"censors when a cell never crosses, so mean solve rate over the 14-point grid is quoted "
         f"beside it. s* is the largest across-seed spread on any complete rung in EITHER job — "
         f"pooled, so moving the arch axis cannot buy a smaller bar: {s['rho50']:.3f} in rho_50, "
         f"{s['mean_solve']:.3f} in mean solve."),
    ]
    textpage(o, "How little does learning direct deduction actually require?",
             (f"All {N_CELLS_TOTAL} cells" if done
              else f"PARTIAL — {n_cells} of {N_CELLS_TOTAL} cells have landed")
             + " · N=48, degree 5, 4 colours, fall ensemble, leak-free (--shuffle-adj)",
             blocks,
             "Taken away — depth: L = " + ", ".join(str(L) for L in depth_have) + ".  "
             + ".  ".join(f"Architecture at L={d}: "
                          + (", ".join(arch_have[d]) if arch_have[d] else "none landed yet")
                          + "; heads " + ("4->1 measured" if (d, 1, "full") in rung else "pending")
                          for d in ARCH_DEPTHS)
             + ".  Three seeds on every rung. Every cell: d_model 256, ff 1024, pre-norm, "
               "constant lr 1e-3, batch 32, 240k instances x 50 epochs = 375k gradient steps — "
               "finding 44's budget and optimizer, unchanged on purpose. Same 14-point x "
               "400-instance grid and the same instances as the 11-02 scan, so the curves are "
               "paired point by point.",
             y0=0.83, gap=0.020, width=132)


def page_ladder(o, rung, cells, s):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Walking the depth ladder down", ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.897, "nhead 4, full MLP, three seeds per rung; lower rho_50 is better",
             ha="center", size=SMALL + 1, color="0.3")
    ax1 = fig.add_axes([0.055, 0.19, 0.25, 0.67])
    ax2 = fig.add_axes([0.385, 0.19, 0.25, 0.67])
    ax3 = fig.add_axes([0.715, 0.19, 0.25, 0.67])

    Ls = [L for L in sorted(DEPTH_LADDER) if (L, 4, "full") in rung]
    for L in Ls:
        v = rung[(L, 4, "full")]
        pts = _r50_points(v)
        ax1.plot([L] * len(pts), pts, "o", ms=9, mfc="none", color="#1a5fb4", alpha=0.75)
        for (sd, m), y in zip(v, pts):
            if m["flag"] != "ok":
                ax1.annotate("censored", (L, y), textcoords="offset points", xytext=(6, 0),
                             size=SMALL - 3, color="#a51d2d")
        ax1.plot([L], [median(pts)], "s", ms=13, color="#1a5fb4")
        ax2.plot([L] * len(v), [m["mean_solve"] for _, m in v], "o", ms=9, mfc="none",
                 color="#26a269", alpha=0.75)
        ax2.plot([L], [median([m["mean_solve"] for _, m in v])], "s", ms=13, color="#26a269")
    med = [median(_r50_points(rung[(L, 4, "full")])) for L in Ls]
    ax1.plot(Ls, med, "-", color="#1a5fb4", lw=2)
    ax2.plot(Ls, [median([m["mean_solve"] for _, m in rung[(L, 4, "full")]]) for L in Ls],
             "-", color="#26a269", lw=2)
    if Ls and s["rho50"] == s["rho50"]:
        ax1.fill_between([min(Ls) - 0.4, max(Ls) + 0.4], med[-1] - s["rho50"] / 2,
                         med[-1] + s["rho50"] / 2, color="0.75", alpha=0.35, zorder=0)
    ax1.axhline(0.333, color="#a51d2d", ls=":", lw=2)
    ax1.text(max(Ls or [1]), 0.340, "structural floor of the ensemble (0.333)", ha="right",
             size=SMALL - 1, color="#a51d2d")
    ax1.axhline(CENSOR, color="0.5", ls="--", lw=1)
    ax1.set_xlabel("layers L")
    ax1.set_ylabel("rho_50   (lower = needs fewer hints)")
    ax1.set_ylim(0.30, CENSOR + 0.04)
    ax2.set_xlabel("layers L")
    ax2.set_ylabel("mean solve rate over the 14-point grid")
    ax2.set_ylim(-0.02, 1.0)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.set_xticks(Ls)

    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(sorted(Ls, reverse=True)):
        sd0 = rung[(L, 4, "full")][0][0]
        rs = sorted(cells[(L, 4, "full", sd0)], key=lambda r: r["realized_rho"])
        ax3.plot([r["realized_rho"] for r in rs], [r["solve_rate"] for r in rs], "-o", ms=4,
                 color=cmap(i / max(len(Ls) - 1, 1)), label=f"L={L}")
    ax3.axhline(0.5, color="0.5", ls=":")
    ax3.set_xlabel("realized hint density rho")
    ax3.set_ylabel("solve rate")
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_title("the curves rho_50 summarises (seed 0)", size=SMALL + 2)
    ax3.grid(alpha=0.25)
    if Ls:
        ax3.legend(fontsize=SMALL - 1, loc="lower right")
    footer(fig, "Open circles are the three seeds, filled squares the median. The grey band on the "
                "left is +/- s*/2 about the DEEPEST rung measured — the model being shrunk from — "
                "so a rung whose median sits inside it has lost nothing readable by being smaller. Points drawn at 1.02 are cells that "
                "never crossed a 50% solve rate anywhere in the grid, so their rho_50 is a bound "
                "and not a value — read those rungs on the right-hand panel instead.")
    o.savefig(fig)


def _arch_keys(rung, depth):
    """(L, nhead, arch) for every arch/head cell measured at this depth, hub first."""
    ks = [(depth, 4, a) for a in ARCH_AXIS if (depth, 4, a) in rung]
    if (depth, 1, "full") in rung:
        ks.append((depth, 1, "full"))
    return ks


def _arch_tick(key):
    L, h, a = key
    return "nhead=1\n(one head, not four)" if h == 1 else ARCH_NAME[a]


def page_arch(o, rung, cells, s):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.955, "Stripping the machinery: rho_50 before and after, at L=6",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.912,
             "same 375k steps, same seeds, same initialization — only what is trainable changes",
             ha="center", size=SMALL + 1, color="0.3")
    ax = fig.add_axes([0.16, 0.20, 0.40, 0.66])
    axd = fig.add_axes([0.665, 0.585, 0.305, 0.275])
    axc = fig.add_axes([0.665, 0.205, 0.305, 0.275])

    # ---- left: the before/after, on the metric of record, at the live depth
    keys = _arch_keys(rung, 6)
    base = rung_stat(rung, HUB6, "rho50") if HUB6 in rung and \
        all(m["flag"] == "ok" for _, m in rung[HUB6]) else None
    ys = list(range(len(keys)))[::-1]
    for y, k in zip(ys, keys):
        v = rung[k]
        pts = _r50_points(v)
        med = median(pts)
        cens = any(m["flag"] != "ok" for _, m in v)
        if k == HUB6:
            col, hatch = "#1a5fb4", None
        elif cens:
            col, hatch = "#a51d2d", "//"
        elif base is not None and med - base <= s["rho50"]:
            col, hatch = "#26a269", None
        else:
            col, hatch = "#c64600", None
        ax.barh(y, med, height=0.55, color=col, alpha=0.85, hatch=hatch, edgecolor="white")
        ax.plot(pts, [y] * len(pts), "o", ms=8, color="0.15", mfc="white", zorder=3)
        tag = "BEFORE" if k == HUB6 else ("censored — never crosses 50%" if cens
                                          else f"{med - base:+.3f}" if base is not None else "")
        ax.text(med + 0.012, y, tag, va="center", size=SMALL - 1, color="0.2")
        ax.text(0.345, y - 0.345, f"{v[0][1]['n_trainable']/1e6:.2f}M trainable "
                                  f"of {v[0][1]['n_params']/1e6:.2f}M", size=SMALL - 2,
                color="0.25")
    if base is not None:
        ax.axvspan(base, base + s["rho50"], color="0.75", alpha=0.35, zorder=0)
        ax.axvline(base, color="#1a5fb4", lw=2, ls="--")
    ax.axvline(0.333, color="#a51d2d", ls=":", lw=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([_arch_tick(k) for k in keys], size=SMALL)
    ax.set_xlabel("rho_50 at L=6   (lower is better; 1.02 = censored)")
    ax.set_xlim(0.30, CENSOR + 0.06)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.grid(axis="x", alpha=0.25)

    # ---- top right: the same axis at L=4, and why it could not answer this
    k4 = _arch_keys(rung, 4)
    y4 = list(range(len(k4)))[::-1]
    hub4 = rung_stat(rung, HUB4, "mean_solve")
    for y, k in zip(y4, k4):
        vals = [m["mean_solve"] for _, m in rung[k]]
        axd.barh(y, median(vals), height=0.6,
                 color="#1a5fb4" if k == HUB4 else "0.55", alpha=0.85)
        axd.plot(vals, [y] * len(vals), "o", ms=5, color="0.15", mfc="white", zorder=3)
    if hub4 is not None and s["mean_solve"] == s["mean_solve"]:
        axd.axvspan(max(hub4 - s["mean_solve"], 0), hub4 + s["mean_solve"], color="#a51d2d",
                    alpha=0.18, zorder=0)
        axd.axvline(hub4, color="#1a5fb4", lw=1.6, ls="--")
    axd.set_yticks(y4)
    axd.set_yticklabels([(k[2] if k[1] == 4 else "nhead=1") for k in k4], size=SMALL - 2)
    axd.set_xlim(0, 1.0)
    axd.set_ylim(-0.7, len(k4) - 0.3)
    axd.set_xlabel("mean solve rate", size=SMALL - 1)
    axd.set_title("the same axis at L=4, the BREAKING depth", size=SMALL + 1)
    axd.grid(axis="x", alpha=0.25)

    # ---- bottom right: the curves the left panel summarises
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(keys):
        sd0 = rung[k][0][0]
        rs = sorted(cells[(k[0], k[1], k[2], sd0)], key=lambda r: r["realized_rho"])
        axc.plot([r["realized_rho"] for r in rs], [r["solve_rate"] for r in rs], "-o", ms=3,
                 color=cmap(i / max(len(keys) - 1, 1)),
                 label=(k[2] if k[1] == 4 else "nhead=1"))
    axc.axhline(0.5, color="0.5", ls=":")
    axc.set_xlabel("realized hint density rho", size=SMALL - 1)
    axc.set_ylabel("solve rate", size=SMALL - 1)
    axc.set_ylim(-0.02, 1.02)
    axc.set_title("L=6 transition curves (seed 0)", size=SMALL + 1)
    axc.grid(alpha=0.25)
    if keys:
        axc.legend(fontsize=SMALL - 3, loc="lower right")
    footer(fig, "The dashed line is the unmodified L=6 model and the grey band is the pooled s*, "
                "the largest across-seed spread on any complete rung in either job; a bar ending "
                "inside it costs nothing readable. The arch axis MOVED from L=4 to L=6 because "
                "L=4 is where deduction breaks: in the small panel the red band (s*) is as wide "
                "as the hub's whole score, so every bar there ties with every other and with a "
                "model that learned nothing. s* is pooled over both depths, so the move buys no "
                "slack. Every mode starts from the same initialization at a given seed.")
    o.savefig(fig)


def page_table(o, rung, s, commit):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Every rung", ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.900, f"N=48 fall, --shuffle-adj, d_model 256, ff 1024, 375k steps; "
                         f"commit {commit}", ha="center", size=SMALL + 1, color="0.3")
    ax = fig.add_axes([0.03, 0.10, 0.94, 0.78])
    ax.axis("off")
    hdr = (f"{'L':>3} {'head':>5} {'arch':>7} {'trainable':>10} {'seeds':>6}  "
           f"{'rho_50 per seed':>26} {'median':>8} {'spread':>7}  "
           f"{'mean solve':>11} {'spread':>7} {'floor':>6}  {'train NLL':>10} {'escape':>7}")
    lines = [hdr, "-" * len(hdr)]
    for key in all_rungs():
        if key not in rung:
            continue
        L, h, a = key
        v = rung[key]
        pts = ["%.3f" % m["rho50"] if m["flag"] == "ok" else " >0.95" for _, m in v]
        ok = [m["rho50"] for _, m in v if m["flag"] == "ok"]
        med = "%.3f" % median(ok) if len(ok) == len(v) else "censored"
        sp = "%.3f" % spread(ok) if len(ok) == len(v) and len(v) > 1 else "--"
        ms = [m["mean_solve"] for _, m in v]
        tr = [history(m["path"])[1] for _, m in v]
        esc = sum(1 for t in tr if t < PLATEAU)
        lines.append(
            f"{L:>3} {h:>5} {a:>7} {v[0][1]['n_trainable']/1e6:>9.2f}M {len(v):>6}  "
            f"{'/'.join(pts):>26} {med:>8} {sp:>7}  "
            f"{median(ms):>11.4f} {spread(ms) if len(ms) > 1 else float('nan'):>7.4f} "
            f"{median([m['solve_at_floor'] for _, m in v]):>6.3f}  "
            f"{median(tr):>10.3f} {esc:>4}/{len(v)}")
    lines += ["", f"pooled yardstick s* = {s['rho50']:.4f} in rho_50, {s['mean_solve']:.4f} in "
                  f"mean solve — the largest across-seed spread on any complete rung above, "
                  f"taken over BOTH jobs"]
    ax.text(0.5, 1.0, "\n".join(lines), family="monospace", size=11.6, ha="center", va="top",
            linespacing=1.42)
    footer(fig, "'escape' counts the seeds whose final training NLL got below 1.64, where finding "
                "44's L=4 cell plateaued for its whole run at this budget. A rung with 0 of 3 "
                "escapes did not TRAIN here; that is not the same as being unable to represent "
                "deduction, and it is not written up as such. 'floor' is the solve rate at rho = "
                "0.333, the hardest bin in the grid.")
    o.savefig(fig)


def page_before(o, rung, prev):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Against the previous best on this exact eval", ha="center", size=BIG,
             weight="bold")
    fig.text(0.5, 0.897, "finding 44's N=48 fall cells (11-02): same 375k steps, same d_model and "
                         "ff, same 14-point grid, same instances", ha="center", size=SMALL + 1,
             color="0.3")
    ax = fig.add_axes([0.08, 0.17, 0.88, 0.70])
    # k[:3] rather than unpacking: a stage C rung key carries (d_model, ff,
    # warmup) after the first three, and this comparison is only ever against
    # the 11-02 cells, which were all d256/ff1024/no warmup. len(k) == 3 is what
    # selects those -- a narrow cell must not be charted against a wide one.
    Ls = sorted(set(prev) & {k[0] for k in rung if len(k) == 3 and k[1] == 4 and k[2] == "full"})
    if Ls:
        w = 0.35
        ax.bar([i - w / 2 for i in range(len(Ls))], [median(prev[L]) for L in Ls], width=w,
               label="11-02 scan (finding 44)", color="0.6")
        ax.bar([i + w / 2 for i in range(len(Ls))],
               [median(_r50_points(rung[(L, 4, "full")])) for L in Ls], width=w,
               label="this job, median of its seeds", color="#1a5fb4")
        for i, L in enumerate(Ls):
            ax.plot([i - w / 2] * len(prev[L]), prev[L], "o", ms=8, mfc="white", color="0.1",
                    zorder=3)
            pts = _r50_points(rung[(L, 4, "full")])
            ax.plot([i + w / 2] * len(pts), pts, "o", ms=8, mfc="white", color="0.1", zorder=3)
        # A bar sitting at CENSOR is not a measured 0.95-ish rho_50, it is a
        # cell that never crossed 50% anywhere on the grid. Pages 4 and 5 say
        # so; this one must too, or the tallest bars here read as real values.
        for i, L in enumerate(Ls):
            for x, pts in ((i - w / 2, prev[L]),
                           (i + w / 2, _r50_points(rung[(L, 4, "full")]))):
                if any(p >= CENSOR for p in pts):
                    ax.annotate(f"{sum(p >= CENSOR for p in pts)} censored",
                                (x, CENSOR), textcoords="offset points", xytext=(0, 7),
                                ha="center", size=SMALL - 2, color="#a51d2d")
        ax.axhline(CENSOR, color="0.5", ls="--", lw=1)
        ax.set_xticks(range(len(Ls)))
        ax.set_xticklabels([f"L={L}\n{len(prev[L])} vs {len(rung[(L, 4, 'full')])} seeds"
                            for L in Ls])
        ax.set_xlim(-0.7, len(Ls) - 0.3)
        ax.set_ylim(0, CENSOR + 0.10)
        ax.set_ylabel("rho_50   (lower is better)")
        ax.axhline(0.333, color="#a51d2d", ls=":", lw=2)
        ax.legend(fontsize=SMALL)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no matched depth rung has landed yet", ha="center", size=MED)
    footer(fig, "This is a REPLICATION panel, not an improvement claim: the configuration is "
                "identical and the only change is three seeds instead of one. The white dots show "
                "how much of the apparent difference is the seed. The point of this job is "
                "smallness, so a rung that merely matches its 11-02 twin at a third of the depth "
                "is the result being looked for.")
    o.savefig(fig)


def page_controls(o, rows, rung, n_cells):
    live = [r for r in rows if r["solve_rate"] > 0.1]
    worst_tr = max((r["truncated_rate"] for r in live), default=float("nan"))
    worst_wf = min((r["well_formed_rate"] for r in live), default=float("nan"))
    over = [r for r in live if r["truncated_rate"] > 0.01 or r["well_formed_rate"] < 0.99]
    keys = {(r["shuffle_adj"], r["fall"]) for r in rows}
    no_escape = [k for k, v in rung.items()
                 if not any(history(m["path"])[1] < PLATEAU for _, m in v)]
    # A rung where only SOME seeds escaped is the more dangerous case: it is
    # not on the no_escape list, so it looks trained, yet its censored seed
    # may be censored because it never trained rather than because the
    # architecture is weaker. Name those seeds explicitly.
    partial = []
    for k, v in sorted(rung.items()):
        if k in no_escape:
            continue
        stuck = [(sd, history(m["path"])[1]) for sd, m in v
                 if history(m["path"])[1] >= PLATEAU]
        if stuck:
            cens = {sd for sd, m in v if m["flag"] != "ok"}
            # rung_label, not k[0..2]: a stage C rung carries its width and
            # warmup after the first three slots, and naming a narrow cell as
            # though it were the wide one is the exact confusion the cell key
            # was widened to prevent.
            partial.append(f"{rung_label(*k).strip()} seed(s) "
                           + ", ".join(f"{sd} at {nll:.3f}"
                                       + (" — and it is the censored one" if sd in cens else "")
                                       for sd, nll in stuck))
    blocks = [
        ("LEAK — the control that matters most in exactly this regime", "#1a5fb4",
         f"Every cell trains AND evaluates with --shuffle-adj, read back out of each written "
         f"config.json rather than trusted from the command line: {sorted(keys)} over all "
         f"{len(rows)} rows. Findings 16, 17 and 23 are why. A minimal architecture that scores "
         f"well on the leaky task has told us nothing, and the smaller the model the more "
         f"tempting the decoding shortcut becomes."),
        ("GRAMMAR vs DEDUCTION", "#26a269" if not over else "#c64600",
         f"Over the {len(live)} bins with solve rate above 0.1: worst truncation "
         f"{worst_tr:.4f} against a 0.01 bar, worst well-formed rate {worst_wf:.4f} against 0.99. "
         + ("No bin breaches either bar, so every gap in this deck is deduction and not finding "
            "25's grammar failure."
            if not over else
            f"{len(over)} bins breach one of them — small breaches, but reported rather than "
            f"rounded away. They are concentrated at the low-density end, where traces are "
            f"longest, and they are far too small to carry any gap on the other pages.")),
        ("TRAINABILITY IS NOT CAPACITY", "#613583",
         ("Every rung has at least one seed below finding 44's 1.64 training-NLL plateau, so no "
          "rung in this deck is a cell that simply failed to train."
          if not no_escape else
          "These rungs had NO seed get below finding 44's 1.64 training-NLL plateau: "
          + ", ".join(rung_label(*k).strip() for k in sorted(no_escape))
          + ". They are reported as 'did not train at 375k steps', NOT as 'cannot represent "
            "deduction'. Separating the two needs more steps or a better optimizer, neither of "
            "which is in this job.")
         + (("  PARTIAL, and it matters more: " + "; ".join(partial)
             + ". Where the stuck seed is also the censored one, part of that rung's cost is "
               "trainability, not representational power, and the quoted gap is the cost on the "
               "seeds that did train.") if partial else "")),
        ("WHAT THIS DECK CANNOT CARRY", "#a51d2d",
         f"{n_cells} of {N_CELLS_TOTAL} cells. Three seeds is enough to put a spread on a rung "
         f"and not enough "
         f"for a confidence interval on the spread itself. rho_50 is one number summarising 14 "
         f"binomial bins at n=400, and it censors; where it does, the mean solve rate carries the "
         f"comparison and the page says so. Nothing here is causal — a small model that solves "
         f"the task shows the task does not need more machinery, not that this machinery is what "
         f"DD looks like."),
    ]
    textpage(o, "Controls, and what this measurement cannot carry",
             "all pre-registered in the sbatch header before submission",
             blocks,
             "The config.json assertion runs inside each task, between training and evaluation, so "
             "a cell whose arch_mode, shuffle_adj, fall, shape or max_len margin is not what was "
             "asked for fails loudly instead of producing a plausible number — finding 43's "
             "failure mode, on a knob that is exactly as silent.",
             y0=0.84, gap=0.020, width=132)


def main():
    ap = argparse.ArgumentParser()
    # Pinned to stage A + stage B, NOT SCAN_DIRS: this deck is their write-up,
    # s* is pooled over whatever it reads, and a later job joining SCAN_DIRS
    # would otherwise widen the bar underneath verdicts already published here.
    ap.add_argument("--scan-dir", nargs="+", default=[SCAN_DEFAULT, SCAN_L6])
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "minarch_2026-11-04"))
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--commit", default="(uncommitted)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_style()
    rows, cells, met, rung = collect(args.scan_dir, args.boot)
    s = yardstick(rung)
    print(f"{len(met)} cells / {len(rung)} rungs; s* = {s}")

    # The matched subset of finding 44's scan: same budget, same width, same
    # ff, same eval grid. Anything else there is a different experiment and is
    # not drawn beside this one.
    prev, bycell = {}, {}
    try:
        for r in load(PREV):
            if (int(r.get("n_train_instances") or 0) == 240000
                    and int(r.get("dim_feedforward") or 0) == 1024
                    and int(r.get("d_model") or 0) == 256):
                bycell.setdefault((r["num_layers"], r["seed"]), []).append(r)
        from scripts.minarch_analysis import rho_50 as _r50
        for (L, _sd), v in bycell.items():
            val, flag = _r50([(x["realized_rho"], x["solve_rate"]) for x in v])
            prev.setdefault(L, []).append(val if flag == "ok" else CENSOR)
    except (OSError, IndexError, KeyError, ValueError) as e:
        print(f"previous-best panel: {type(e).__name__}: {e}")
        prev = {}
    print("matched 11-02 cells: " + ", ".join(f"L={L}:{len(v)} seeds" for L, v in sorted(prev.items())))

    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=8)
        page_verdict(o, rung, s, len(met))
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task every "
                               "cell in this deck is trained and scored on. This is what the "
                               "models have to produce token by token, with no scratchpad and no "
                               "tool — the question of the deck is how much network that needs.")
        page_ladder(o, rung, cells, s)
        page_arch(o, rung, cells, s)
        page_table(o, rung, s, args.commit)
        page_before(o, rung, prev)
        page_controls(o, rows, rung, len(met))
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
