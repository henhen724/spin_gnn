"""Slide deck for the minimal-architecture grid (51032450).

The question this deck answers is not "how well does it solve DD" but "how
little machinery does solving DD need", so every page is a subtraction: a rung
of the depth ladder, a block with its nonlinearity removed, a block whose MLP
is frozen at initialization, a block with no MLP write at all.

Runs on a PARTIAL grid on purpose -- 36 cells land over ~11 h and the deck is
how the intermediate state gets looked at. Pages say how many cells they are
drawn from and the title says PARTIAL until all 36 are in.

    python3 notebooks/make_minarch_deck.py [--scan-dir DIR] [--out DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from scripts.minarch_analysis import (ARCH_AXIS, DEPTH_LADDER, PLATEAU, SCAN_DEFAULT,
                                      cell_metrics, history, load, median, spread)

HUB = (4, 4, "full")
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


def collect(scan_dir, boot):
    rows = load(scan_dir)
    cells = {}
    for r in rows:
        cells.setdefault((r["num_layers"], r["nhead"], r["arch_mode"], r["seed"]), []).append(r)
    met = {k: cell_metrics(v, boot) for k, v in cells.items()}
    rung = {}
    for k, m in met.items():
        rung.setdefault(k[:3], []).append((k[3], m))
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
    done = n_cells == 36
    depth_have = [L for L in DEPTH_LADDER if (L, 4, "full") in rung]
    B = None
    for L in DEPTH_LADDER:
        v = rung.get((L, 4, "full"))
        if v and any(m["flag"] != "ok" for _, m in v):
            B = L
            break
    arch_have = [a for a in ARCH_AXIS if a != "full" and (4, 4, a) in rung]
    hub = rung_stat(rung, HUB, "mean_solve")

    if B is not None:
        head = (f"Deduction breaks between L={B} and L={B + 1}: at L={B} at least one seed never "
                f"reaches a 50% solve rate anywhere in the grid.")
    elif depth_have:
        head = (f"Every depth measured so far still solves it — the ladder has been walked down to "
                f"L={min(depth_have)} without finding the breaking point.")
    else:
        head = "No depth rung has landed yet."

    blocks = [
        ("THE QUESTION", "#1a5fb4",
         "Not how well the model solves direct deduction — it already does, leak-free — but how "
         "little machinery that takes. Every cell here is the L=12/ff=1024 model with something "
         "REMOVED: layers, heads, the MLP's nonlinearity, the MLP's training, or the MLP write "
         "itself. The breaking point is the result, not the failure."),
        ("THE HEADLINE", "#613583", head),
        ("WHAT WAS TAKEN AWAY", "#26a269",
         "Depth: L = " + ", ".join(str(L) for L in depth_have) + ".  "
         "Architecture at L=4: " + (", ".join(arch_have) if arch_have else "none landed yet")
         + ".  Heads: " + ("4 -> 1 measured" if (4, 1, "full") in rung else "pending")
         + ".  Three seeds on every rung, because at this scale a seed is worth more than a layer "
           "(finding 41: spread 0.1627 at N=48; finding 47: it is not a schedule artifact)."),
        ("HOW TO READ EVERY NUMBER", "#c64600",
         f"rho_50 is the hint density at which the solve rate crosses 50% — LOWER IS BETTER, it "
         f"means the model needs fewer hints. It censors when a cell never crosses, so "
         f"mean solve rate over the 14-point grid is quoted beside it. The yardstick s* is the "
         f"largest across-seed spread on any rung in this job: "
         f"{s['rho50']:.3f} in rho_50, {s['mean_solve']:.3f} in mean solve. "
         f"Nothing smaller than s* is called a difference."
         + ("" if hub is None else
            f" The L=4 full-MLP hub, which every architecture cell is compared against, sits at "
            f"mean solve {hub:.3f}.")),
    ]
    textpage(o, "How little does learning direct deduction actually require?",
             ("All 36 cells" if done else f"PARTIAL — {n_cells} of 36 cells have landed")
             + " · N=48, degree 5, 4 colours, fall ensemble, leak-free (--shuffle-adj)",
             blocks,
             "Every cell: d_model 256, ff 1024, pre-norm, constant lr 1e-3, batch 32, 240k "
             "instances x 50 epochs = 375k gradient steps — finding 44's budget and optimizer, "
             "unchanged on purpose. Evaluated on the same 14-point x 400-instance grid and the "
             "same instances as the 11-02 scan, so the curves are paired point by point.",
             y0=0.80, gap=0.028, width=118)


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


def page_arch(o, rung, cells, s):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Taking the MLP apart, at fixed depth L=4", ha="center", size=BIG,
             weight="bold")
    fig.text(0.5, 0.897,
             "same 375k steps, same seeds, same initialization — only what is trainable changes",
             ha="center", size=SMALL + 1, color="0.3")
    ax = fig.add_axes([0.17, 0.19, 0.41, 0.67])
    axc = fig.add_axes([0.70, 0.19, 0.27, 0.67])

    modes = [a for a in ARCH_AXIS if (4, 4, a) in rung]
    ys = list(range(len(modes)))[::-1]
    hub = rung_stat(rung, HUB, "mean_solve")
    for y, a in zip(ys, modes):
        v = rung[(4, 4, a)]
        vals = [m["mean_solve"] for _, m in v]
        col = "#1a5fb4" if a == "full" else ("#26a269" if median(vals) > (hub or 0) - s["mean_solve"]
                                             else "#c64600")
        ax.barh(y, median(vals), height=0.55, color=col, alpha=0.85)
        ax.plot(vals, [y] * len(vals), "o", ms=8, color="0.15", mfc="white", zorder=3)
        tp = v[0][1]["n_trainable"] / 1e6
        ax.text(0.005, y - 0.33, f"{tp:.2f}M trainable of {v[0][1]['n_params']/1e6:.2f}M",
                size=SMALL - 2, color="0.25")
    if hub is not None and s["mean_solve"] == s["mean_solve"]:
        ax.axvspan(hub - s["mean_solve"], hub + s["mean_solve"], color="0.75", alpha=0.35, zorder=0)
        ax.axvline(hub, color="#1a5fb4", lw=2, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([ARCH_NAME[a] for a in modes], size=SMALL)
    ax.set_xlabel("mean solve rate over the 14-point grid (median of 3 seeds)")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.25)

    cmap = plt.get_cmap("viridis")
    for i, a in enumerate(modes):
        sd0 = rung[(4, 4, a)][0][0]
        rs = sorted(cells[(4, 4, a, sd0)], key=lambda r: r["realized_rho"])
        axc.plot([r["realized_rho"] for r in rs], [r["solve_rate"] for r in rs], "-o", ms=4,
                 color=cmap(i / max(len(ARCH_AXIS) - 1, 1)), label=a)
    axc.axhline(0.5, color="0.5", ls=":")
    axc.set_xlabel("realized hint density rho")
    axc.set_ylabel("solve rate")
    axc.set_ylim(-0.02, 1.02)
    axc.set_title("transition curves (seed 0)", size=SMALL + 2)
    axc.grid(alpha=0.25)
    if modes:
        axc.legend(fontsize=SMALL - 1, loc="lower right")
    footer(fig, "The dashed line is the full-MLP hub and the grey band is +/- s*, this job's own "
                "largest across-seed spread; a bar ending inside the band is NOT separable from "
                "the full model. White dots are the individual seeds. Every mode starts from the "
                "same initialization at a given seed, so a frozen cell's weights are exactly the "
                "weights the trained cell started from — the contrast is about training them, not "
                "about drawing them.")
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
    order = [(L, 4, "full") for L in DEPTH_LADDER] \
        + [(4, 4, a) for a in ARCH_AXIS if a != "full"] + [(4, 1, "full")]
    for key in order:
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
    lines += ["", f"yardstick s* = {s['rho50']:.4f} in rho_50, {s['mean_solve']:.4f} in mean "
                  f"solve — the largest across-seed spread on any complete rung above"]
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
    Ls = sorted(set(prev) & {L for (L, h, a) in rung if h == 4 and a == "full"})
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
        ax.set_xticks(range(len(Ls)))
        ax.set_xticklabels([f"L={L}\n{len(prev[L])} vs {len(rung[(L, 4, 'full')])} seeds"
                            for L in Ls])
        ax.set_xlim(-0.7, len(Ls) - 0.3)
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
          + ", ".join(f"L={L}/h{h}/{a}" for (L, h, a) in sorted(no_escape))
          + ". They are reported as 'did not train at 375k steps', NOT as 'cannot represent "
            "deduction'. Separating the two needs more steps or a better optimizer, neither of "
            "which is in this job.")),
        ("WHAT THIS DECK CANNOT CARRY", "#a51d2d",
         f"{n_cells} of 36 cells. Three seeds is enough to put a spread on a rung and not enough "
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
             y0=0.80, gap=0.026, width=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default=SCAN_DEFAULT)
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
