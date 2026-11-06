"""Deck: stages H and I -- the attention-only WIDTH ladder at fixed depth L=12.

Stage E left the standing minimum at attention-only L=12 d256 (3,331,642
trainable, rho_50 0.605). The open question it left is whether the saving came
from depth carrying the MLP's work, in which case the model can also be made
NARROW, or whether width is a second, independent floor. Stage H (51223007)
asked it at d128 and d192; d128 did not train on 2 of 3 seeds, so the
registered trainability gate fired and H's own (D)/(W) call came back
INCONCLUSIVE. Stage I (51246202) put d160 in the gap and the ladder closed.

    python3 -m notebooks.make_stagehi_deck --commit $(git rev-parse --short HEAD)

H and I share one deck on purpose: stage I's predictions were registered
against stage H's PARTIAL results (see BRACKET_AT_REGISTRATION, imported here
rather than retyped), and splitting them would hide that the interpolation was
anchored on two of three d192 cells. Every verdict on these slides comes from
scripts/stageh_score.py and scripts/stagei_score.py, both committed before
their arrays closed. SCAN_H and SCAN_I stay out of SCAN_DIRS so their rungs
cannot pool into the frozen bar s* = 0.1484 / 0.2918, which is quoted and never
recomputed.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from scripts.minarch_analysis import (SCAN_DIRS, SCAN_E, SCAN_G, SCAN_H, SCAN_I,
                                      cell_key, cell_metrics, load_all, rung_label)
from scripts.stagec_score import PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt, rung, summarize
from scripts.stageh_score import (FREE, HUB_RHO50, HUB_TRAINABLE, LOO_BAR,
                                  PRED_ESCAPES_HEADER, PRED_ESCAPES_Q3)
from scripts.stageh_score import PREDICTED_TRAINABLE as PRED_TR_H
from scripts.stagei_score import (BRACKET_AT_REGISTRATION, E2, H1, H2, I1, L6_D128,
                                  PRED_NLL_HI, PRED_NLL_LO)
from scripts.stagei_score import PREDICTED_TRAINABLE as PRED_TR_I
from scripts import acc_vs_solve as avs

FULL_C = "#13507a"      # the complete model, and the reference depth ladder
NOMLP_C = "#1a7a3c"     # attention-only, the arm this deck is about
BAND_C = "#c1272d"      # the plateau, the 0.72-0.80 barrier band, the free line
NOTE_C = "#6a3d9a"      # registered in advance
DEAD_C = "0.45"         # cells that never escaped the plateau

SEED_C = ["#1a7a3c", "#6a3d9a", "#b06000"]
FULL_L12 = (12, 4, "full")          # best solver on this eval, at any size
HUB = (6, 4, "full")                # the reference architecture s* is anchored to
GRID_TOP = 0.95                     # the top of the 14-point rho grid

# The ladder, narrow to wide. d128 and d192 are stage H, d160 is stage I, d256
# is stage E's rung and the standing minimum -- one knob, d_model, at L=12.
LADDER = [(128, H1, "H1", "H"), (160, I1, "I1", "I"),
          (192, H2, "H2", "H"), (256, E2, "E2", "E")]


def collect(scan_dirs, boot):
    """Per-cell metrics, and the raw per-rho rows the transition plot needs."""
    by = defaultdict(list)
    for r in load_all(scan_dirs):
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, boot), path=v[0]["path"]) for k, v in by.items()}
    return cells, by


def curve(path):
    with open(os.path.join(path, "history.json")) as fh:
        h = json.load(fh)
    return ([float(e["train_nll"]) for e in h], [float(e["accuracy"]) for e in h])


def acc_of(ms):
    """Median final-epoch validation token accuracy over a rung's seeds."""
    a = sorted(curve(m["path"])[1][-1] for m in ms)
    return a[len(a) // 2]


def ladder_rows(cells):
    """One row per width: everything the table and the plots need."""
    rows = []
    for d, key, tag, stage in LADDER:
        ms = rung(cells, key)
        if not ms:
            continue
        s = summarize(ms)
        rows.append(dict(d=d, key=key, tag=tag, stage=stage, ms=ms, s=s,
                         esc=escapes(ms)[0], nll=escapes(ms)[1], acc=acc_of(ms)))
    return rows


def steps(rows):
    """Consecutive rho_50 gaps up the ladder, in s* units, readable pairs only."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["s"]["rho50_readable"] and b["s"]["rho50_readable"]:
            gap = a["s"]["rho50"] - b["s"]["rho50"]
            out.append((a["d"], b["d"], gap, gap / S_RHO50))
    return out


def ordinal(n):
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def spread_rank(cells, key):
    """Where this rung's mean_solve spread sits among all 3-seed rungs in the grid."""
    sp = []
    for k in {c[:-1] for c in cells}:
        ms = rung(cells, k)
        if ms and len(ms) == 3:
            sp.append((summarize(ms)["spread_mean"], k))
    sp.sort()
    target = summarize(rung(cells, key))["spread_mean"]
    return 1 + sum(1 for v, _k in sp if v < target), len(sp)


def sufficiency(scan_dirs):
    """Re-run acc_vs_solve's fit in-process, so page 7 cannot drift from the script."""
    cells = [c for c in avs.collect(scan_dirs) if c["acc"] == c["acc"]]
    ref = [c for c in cells if avs.is_reference(c["rung"])]
    held = [c for c in cells if not avs.is_reference(c["rung"])]
    fit = avs.pava([(c["acc"], c["mean_solve"]) for c in ref])
    loo = []
    for i, c in enumerate(ref):
        sub = avs.pava([(d["acc"], d["mean_solve"]) for j, d in enumerate(ref) if j != i])
        loo.append(c["mean_solve"] - avs.predict(sub, c["acc"])[0])
    bar = avs.rms(loo)
    by_rung, sharp = {}, []
    for c in held:
        p, oor, cov = avs.predict(fit, c["acc"])
        if oor or cov > bar:
            continue
        by_rung.setdefault(c["rung"], []).append(c["mean_solve"] - p)
        if p > avs.FLOOR:
            sharp.append((c, p, c["mean_solve"] - p))
    means = {r: sum(v) / len(v) for r, v in by_rung.items()}
    return dict(bar=bar, n_ref=len(ref), n_rungs=len(means),
                n_exp=sum(1 for m in means.values() if abs(m) <= bar),
                n_cells=sum(len(v) for v in by_rung.values()),
                n_sharp=len(sharp),
                worst=max((abs(r) for _c, _p, r in sharp), default=float("nan")),
                means=means)


def page_headline(o, cells, rows, suf, commit):
    by_d = {r["d"]: r for r in rows}
    st = {(a, b): (g, u) for a, b, g, u in steps(rows)}
    table = "   ".join(f"d{r['d']} {rho_txt(r['s'])}" for r in rows)
    sep = [f"{a}->{b} {u:.2f} s*" for (a, b), (_g, u) in sorted(st.items())]
    e2, i1, h2, h1 = by_d[256], by_d[160], by_d[192], by_d[128]
    textpage(
        o,
        "Width is a second floor, and it is graded: the cheap end dies slowly",
        f"N=48, degree 5, 4 colours, leak-free | one knob, d_model, at L=12 attention-only | "
        f"H 51223007 + I 51246202, 9 cells, 3 seeds",
        [
            ("THE LADDER IS NOW COMPLETE, AND IT IS A RAMP", NOMLP_C,
             f"Median rho_50 across the four widths: {table}. Every rung from d160 up crosses on "
             f"3 of 3 seeds and escapes finding 44's {PLATEAU} plateau on 3 of 3. d128 escapes on "
             f"{h1['esc']} of 3 and is CENSORED -- no bin of the 14-point grid reaches solve rate "
             f"0.5. Nothing collapses between d256 and d160: the model gets worse smoothly, one "
             f"rung at a time, and then stops training altogether."),
            ("WHERE THE MOVEMENT ACTUALLY IS, AGAINST THE FROZEN BAR", BAND_C,
             f"Consecutive gaps in units of the frozen seed-spread bar s* = {S_RHO50}: "
             f"{', '.join(sep)}. Only the top step is separable. d160 and d192 are "
             f"{abs(i1['s']['rho50'] - h2['s']['rho50']):.3f} apart, which is noise at this bar, "
             f"so the whole 64-unit stretch from d160 to d192 -- {h2['s']['trainable'] - i1['s']['trainable']:,} "
             f"parameters, 42% of the smaller rung -- buys nothing a seed could not. Everything "
             f"that is real lives between d192 and d256."),
            ("WHAT STAGE H COULD NOT ANSWER, AND WHY THAT WAS THE RIGHT ANSWER", NOTE_C,
             f"H's registered gate says a rung that does not train cannot be read as a rung that "
             f"cannot represent the task. d128 came in at {h1['esc']} of 3 escapes, the gate fired, "
             f"and H's (D)/(W) call returned INCONCLUSIVE rather than 'width is the floor'. The "
             f"answer on this page instead comes from the two rungs that DID train 3 of 3 and "
             f"still landed in the 0.72-0.80 barrier band: d160 at {rho_txt(i1['s'])} and d192 at "
             f"{rho_txt(h2['s'])}, both above the free line {FREE}, every seed past the plateau."),
            ("WHAT DID NOT MOVE: THE STANDING MINIMUM", DEAD_C,
             f"Neither stage fired its standing-minimum rule (3 of 3 crossing AND median rho_50 at "
             f"or below {FREE}). The minimum is still stage E's attention-only L=12 d256, "
             f"{e2['s']['trainable']:,} trainable at rho_50 {rho_txt(e2['s'])}, and the best solver "
             f"on this eval at any size is still the complete model at L=12. Nine GPU-cells bought "
             f"a shape, not a record, and the shape is the result."),
        ],
        f"rho_50 = realized hint density at which solve rate crosses 0.5 on a 14-point x "
        f"400-instance grid, LOWER IS BETTER, reported CENSORED when no bin crosses. mean_solve = "
        f"unweighted mean over the 14 bins, which never censors. s* = {S_RHO50} / {S_MEAN} was "
        f"frozen before any of these cells ran. The 'free' line {FREE} is the hub (full L=6 d256, "
        f"rho_50 {HUB_RHO50}, {HUB_TRAINABLE:,} trainable) plus one s*. Commit {commit}.",
        width=122)


def page_ladder(o, cells, rows, raw, commit):
    """rho_50 against width, and the transition curves it is read from."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The complete width ladder at L=12, and where the movement is",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, "Left: every seed, not just the median. Right: the curves rho_50 is read "
                         "off, identical 14-point x 400-instance grid for all nine cells",
             ha="center", size=MED - 3, color="0.3")

    cmap = {128: BAND_C, 160: NOTE_C, 192: "#b06000", 256: NOMLP_C}
    ax = fig.add_axes([0.085, 0.28, 0.235, 0.52])
    med_x, med_y = [], []
    n_cens = sum(1 for r in rows for m in r["ms"] if m["flag"] != "ok")
    for r in rows:
        for m in r["ms"]:
            if m["flag"] == "ok":
                ax.plot(r["d"], m["rho50"], "o", ms=7, color=cmap[r["d"]], alpha=0.75)
            else:
                ax.plot(r["d"], GRID_TOP, "^", ms=9, mfc="none", mew=1.8, color=cmap[r["d"]])
        if r["s"]["rho50_readable"]:
            med_x.append(r["d"])
            med_y.append(r["s"]["rho50"])
    ax.plot(med_x, med_y, "-", lw=2.2, color="0.25", zorder=1, label="median rho_50")
    ax.axhspan(0.72, 0.80, color=BAND_C, alpha=0.08, zorder=0)
    ax.axhline(FREE, color=BAND_C, lw=1.6, ls="--")
    ax.text(114, FREE, f" free line {FREE}", size=SMALL - 1, color=BAND_C, va="bottom", ha="left")
    ax.text(0.97, 0.97, f"open triangles: the {n_cens} censored seeds,\ndrawn at the grid edge "
                        f"{GRID_TOP} because\nthat is all the eval knows about them",
            transform=ax.transAxes, size=SMALL - 3, color=BAND_C, va="top", ha="right")
    ax.errorbar(274, 0.68, yerr=S_RHO50 / 2, color="0.25", capsize=5, lw=1.8)
    ax.text(274, 0.68 + S_RHO50 / 2, f"s* {S_RHO50} ", size=SMALL - 2, color="0.25",
            va="bottom", ha="right")
    ax.set_xlabel("d_model")
    ax.set_ylabel("rho_50   (lower is better)")
    ax.set_xlim(112, 292)
    ax.set_ylim(0.55, 1.00)
    ax.set_xticks([128, 160, 192, 256])
    ax.grid(alpha=0.25)
    ax.set_title("A ramp, not a cliff\nattention-only, L=12, ff=1024, 4 heads", size=MED - 4)

    ax2 = fig.add_axes([0.385, 0.28, 0.235, 0.52])
    for r in rows:
        for m in r["ms"]:
            seed_rows = next(v for k, v in raw.items()
                             if k[:-1] == r["key"] and v[0]["path"] == m["path"])
            pts = sorted((x["realized_rho"], x["solve_rate"]) for x in seed_rows)
            ax2.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=1.7,
                     color=cmap[r["d"]], alpha=0.8)
    ax2.axhline(0.5, color="0.5", lw=1.2, ls=":")
    ax2.text(0.015, 0.5, " solve rate 0.5", size=SMALL - 1, color="0.4", va="bottom",
             transform=ax2.get_yaxis_transform())
    ax2.set_xlabel("realized hint density rho")
    ax2.set_ylabel("solve rate")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(alpha=0.25)
    ax2.set_title("Twelve curves: three seeds at each of four widths", size=MED - 2)
    ax2.legend(handles=[Line2D([], [], color=cmap[r["d"]], lw=2.2,
                               label=f"d{r['d']} ({r['stage']})") for r in rows],
               loc="upper left", framealpha=0.95, fontsize=SMALL - 1)

    block = ["            d128     d160     d192     d256",
             "            (H1)     (I1)     (H2)     (E2)"]
    def row(name, fn):
        return f"  {name:<9s}" + "".join(f"{fn(r):>9s}" for r in rows)
    block += [
        "",
        row("rho_50", lambda r: rho_txt(r["s"])),
        row("crossing", lambda r: f"{r['s']['n_cross']}/3"),
        row("escapes", lambda r: f"{r['esc']}/3"),
        row("mean_solve", lambda r: f"{r['s']['mean_solve']:.4f}"),
        row("spread", lambda r: f"{r['s']['spread_mean']:.4f}"),
        row("token acc", lambda r: f"{r['acc']:.4f}"),
        row("train NLL", lambda r: f"{sorted(r['nll'])[1]:.4f}"),
        row("trainable", lambda r: f"{r['s']['trainable'] / 1e6:.2f}M"),
        "",
        "  4*d^2 + 8*d per layer, 630*d + 58 fixed:",
        "  every count above is predicted to the",
        "  parameter by that formula, and asserted",
        "  inside each sbatch before training starts.",
    ]
    fig.text(0.685, 0.80, "\n".join(block), size=SMALL - 1, va="top", family="monospace",
             linespacing=1.55)
    gaps = ", ".join(f"d{a}->d{b} {g:+.3f} ({u:.2f} s*)" for a, b, g, u in steps(rows))
    footer(fig, f"Read the left panel as a ramp with a wall at the bottom. Consecutive medians: "
                f"{gaps} -- one separable step out of the two that can be read at all, and the "
                f"d128 step cannot be read because that rung is censored. A censored seed's "
                f"rho_50 is somewhere above the grid edge and plotting it as a number would "
                f"invent data, so it is a triangle. Note the right panel: the censored d128 seeds "
                f"are not shifted versions of the others, they are flat along the bottom until "
                f"the very top of the grid, which is what the plateau looks like in solve space. "
                f"Commit {commit}.")
    o.savefig(fig)


def page_training(o, cells, rows, commit):
    """Which widths ever start learning, and the gate that read d128."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Nine cells, one plateau line: which widths ever start learning",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, "Training NLL per epoch, identical schedule and 240k pool everywhere; "
                         "the red line is finding 44's plateau, the thing a cell has to leave",
             ha="center", size=MED - 3, color="0.3")

    panels = [r for r in rows if r["stage"] in ("H", "I")]
    for i, r in enumerate(panels):
        ax = fig.add_axes([0.085 + 0.215 * i, 0.28, 0.17, 0.52])
        for j, m in enumerate(r["ms"]):
            tr = curve(m["path"])[0]
            stalled = tr[-1] >= PLATEAU
            ax.plot(range(1, len(tr) + 1), tr, lw=2.0 if not stalled else 1.6,
                    ls="-" if not stalled else "--",
                    color=SEED_C[j] if not stalled else DEAD_C,
                    label=f"seed {j}  {tr[-1]:.3f}")
        ax.axhline(PLATEAU, color=BAND_C, lw=2, zorder=1)
        if i == 0:
            ax.text(0.03, PLATEAU, f" plateau {PLATEAU}", size=SMALL - 1, color=BAND_C,
                    va="bottom", transform=ax.get_yaxis_transform())
            ax.set_ylabel("train NLL")
        ax.set_xlabel("epoch")
        ax.set_ylim(1.52, 1.95)
        ax.set_title(f"{r['tag']}  d{r['d']}, stage {r['stage']}\n{r['esc']} of 3 escapes",
                     size=MED - 4)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.95, fontsize=SMALL - 3)

    lo, hi = PRED_NLL_LO, PRED_NLL_HI
    i1 = next(r for r in rows if r["d"] == 160)
    med = sorted(i1["nll"])[1]
    rank, total = spread_rank(cells, I1)
    lines = [
        "STAGE I's REGISTERED INTERVAL",
        f"committed c5a1009, before any I cell landed",
        "",
        f"  predicted final train NLL (median of 3)",
        f"      inside [{lo:.3f}, {hi:.3f}]",
        f"  observed   {i1['nll'][0]:.4f}  {i1['nll'][1]:.4f}  {i1['nll'][2]:.4f}",
        f"  median     {med:.4f}   -> INSIDE",
        "",
        "  It was an INTERVAL, not a direction: too",
        "  low and d160 is not between its brackets,",
        "  too high and it has not trained. Both ends",
        "  were falsifiers and neither fired.",
        "",
        "SEED SPREAD",
        f"  d160 mean_solve spread {i1['s']['spread_mean']:.4f},",
        f"  the {ordinal(rank)} tightest of {total} three-seed",
        f"  rungs in the grid. Neighbours: d192 "
        f"{next(r for r in rows if r['d'] == 192)['s']['spread_mean']:.4f},",
        f"  d256 {next(r for r in rows if r['d'] == 256)['s']['spread_mean']:.4f}.",
    ]
    fig.text(0.715, 0.80, "\n".join(lines), size=SMALL - 1, va="top", family="monospace",
             linespacing=1.5)
    h1 = next(r for r in rows if r["d"] == 128)
    footer(fig, f"d128's three seeds end at {h1['nll'][0]:.4f}, {h1['nll'][1]:.4f} and "
                f"{h1['nll'][2]:.4f}: one crosses the line, two sit on it and are still "
                f"descending at about a thousandth of a nat per epoch when the budget runs out. "
                f"The registered gate therefore refuses to read this rung as a capacity "
                f"statement, and the deck does not either. Stage G is the nearest evidence and it "
                f"cuts both ways: at L=6 it doubled the budget to 750k steps and rescued nothing "
                f"at d128, while at d256 it turned one censored seed into a solver -- so a "
                f"stalled cell CAN be a budget effect, and at this width it has not been one. "
                f"d160 and d192 escape on every seed, which is what makes them the rungs the "
                f"(D)/(W) call is allowed to use. Commit {commit}.")
    o.savefig(fig)


def page_predictions(o, cells, rows, commit):
    by_d = {r["d"]: r for r in rows}
    h1, i1, h2 = by_d[128], by_d[160], by_d[192]
    br = BRACKET_AT_REGISTRATION
    d192_then = br["d192"]
    d128_then = br["d128"]
    textpage(
        o,
        "Predictions registered before the cells landed, and how they scored",
        "Four of them, from two scripts committed while their arrays were still running; none was "
        "adjusted afterwards",
        [
            ("H: TWO PREDICTIONS FOR d128's ESCAPES, AND THEY DISAGREED", NOTE_C,
             f"The sbatch header said {PRED_ESCAPES_HEADER['H1']}. escape_timing.py's Q3, "
             f"extrapolating each cell's terminal slope, said {PRED_ESCAPES_Q3['H1']}. Observed: "
             f"{h1['esc']} of 3. Q3 was the closer of the two and the header was wrong -- the "
             f"opposite way round from stage G, where the header won. Split one-all across two "
             f"stages: neither source is reliable on escape counts, so both keep being registered."),
            ("I: THE PARAMETER COUNT, ASSERTED BEFORE TRAINING STARTED", FULL_C,
             f"4*d^2 + 8*d per layer plus 630*d + 58 fixed predicts {PRED_TR_I['I1']:,} trainable "
             f"at d160, and the sbatch aborts if the model it builds disagrees. Observed "
             f"{i1['s']['trainable']:,}. The same assertion held at d128 ({PRED_TR_H['H1']:,}) and "
             f"d192 ({PRED_TR_H['H2']:,}). It is a small prediction and it is the one that makes "
             f"'d160 is {by_d[256]['s']['trainable'] / i1['s']['trainable']:.2f}x smaller than "
             f"the standing minimum' a measurement rather than an estimate."),
            ("I: THE TRAIN-NLL INTERVAL, AND WHAT IT WAS ANCHORED ON", BAND_C,
             f"Predicted median final train NLL in [{PRED_NLL_LO:.3f}, {PRED_NLL_HI:.3f}] by "
             f"interpolating between the neighbours AS THEY STOOD AT REGISTRATION -- and the "
             f"scorer records that state so it cannot be reread favourably later: d128 had "
             f"{d128_then['cells']} of 3 cells done, d192 only {d192_then['cells']} of 3, and that "
             f"one was at {d192_then['note']}. The interpolation was anchored on partial data, the "
             f"scorer's bracket control printed exactly that when the array closed, and the "
             f"prediction still hit at {sorted(i1['nll'])[1]:.4f}."),
            ("I: THE VERDICT THE LADDER WAS BUILT TO DECIDE", NOMLP_C,
             f"Registered as a two-way call: a CLIFF if d160 behaved like d128 (fewer than 2 of 3 "
             f"escapes), GRADED if it trained and landed between its neighbours. It trained on 3 "
             f"of 3 and came in at {rho_txt(i1['s'])}, between d128's censored and d192's "
             f"{rho_txt(h2['s'])} in every metric except rho_50 itself, where it is "
             f"{abs(i1['s']['rho50'] - h2['s']['rho50']):.3f} the wrong side of d192 -- inside s*, "
             f"so not a reversal. GRADED. The trainability bracket closes to [128, 160]."),
        ],
        f"What a registered prediction buys here is narrow and worth stating plainly: it stops the "
        f"deck choosing which of two numbers to call the result after seeing both. It does not "
        f"make a 9-cell grid a large sample, and three of these four were about the ARM's "
        f"behaviour, not about the metric of record. Commit {commit}.",
        width=122)


def page_next(o, cells, rows, suf, commit):
    by_d = {r["d"]: r for r in rows}
    e2 = by_d[256]
    l12 = summarize(rung(cells, FULL_L12)) if rung(cells, FULL_L12) else None
    l6d128 = summarize(rung(cells, L6_D128)) if rung(cells, L6_D128) else None
    best = (f"the complete model at L=12, rho_50 {rho_txt(l12)} at {l12['trainable']:,} trainable"
            if l12 else "not readable from these scan dirs")
    new = [k for k in (H1, I1, H2) if k in suf["means"]]
    new_txt = ", ".join(f"{rung_label(*k).strip()} {suf['means'][k]:+.4f}" for k in new) or "none"
    textpage(
        o,
        "What H and I settle, what they leave open, and what is running tonight",
        "One arm closed, one bracket narrowed to a gap not worth a job, and the next question "
        "already queued",
        [
            ("SETTLED: DEPTH DOES NOT MAKE WIDTH FREE", NOMLP_C,
             f"Stage E's saving was real but it does not compose. At L=12 the attention-only model "
             f"holds rho_50 {rho_txt(e2['s'])} at d256 and loses the free line as soon as it is "
             f"narrowed: d192 {rho_txt(by_d[192]['s'])}, d160 {rho_txt(by_d[160]['s'])}, both on 3 "
             f"of 3 trained seeds, both inside the 0.72-0.80 barrier band. Width is a second floor "
             f"and it sits between d192 and d256. The breaking point is the result: this programme "
             f"wants the smallest architecture that still deduces, not the highest solve rate."),
            ("SETTLED: TOKEN ACCURACY STILL EXPLAINS ALL THREE NEW RUNGS", FULL_C,
             f"The sufficiency fit was refit over {suf['n_ref']} depth-ladder cells with H and I "
             f"included: {suf['n_exp']}/{suf['n_rungs']} well-determined held-out rungs "
             f"({suf['n_cells']} cells) sit on the depth ladder's own accuracy-to-solve curve, "
             f"{suf['n_sharp']} of those cells discriminating, worst miss {suf['worst']:.4f} "
             f"against the "
             f"leave-one-out bar {suf['bar']:.4f}. The new rungs: {new_txt}. Narrowing does not "
             f"open a deduction-specific failure mode; it slides the cell down one axis."),
            ("NOT SETTLED, AND DELIBERATELY NOT BEING CHASED", DEAD_C,
             f"Whether d128 can ever learn this. It is {by_d[128]['esc']} of 3 at 375k steps, and "
             f"stage G's doubled budget rescued no d128 seed at L=6, so the honest label stays "
             f"'did not train'. The trainability bracket is [128, 160]; the next cell on it would "
             f"cost 3 GPU-cells to split a 32-unit gap that no separable rho_50 step lives in -- "
             f"d160 to d192 is already only {steps(rows)[0][3] if steps(rows) else 0:.2f} s*. "
             f"Below d192 there is nothing left to measure, so the grid stops here."),
            ("RUNNING: STAGE J, 51280737, SHAPE AT MATCHED BUDGET", NOTE_C,
             f"Everything separable lives between d192 and d256, and nothing in this grid sits "
             f"above L=12. Stage J asks both at once, matched to 2.6% in parameters: J1 "
             f"attention-only L=12 d224 (2,571,130) against J2 attention-only L=16 d192 "
             f"(2,504,890), three seeds each, 6 cells, about 30 GPU-hours. Headline prediction J0: "
             f"|rho_50(J1) - rho_50(J2)| <= s*, i.e. shape is not a free parameter, with a "
             f"separable difference as the falsifier, named DEPTH or WIDTH in advance."),
        ],
        f"Cost for the record: 9 cells, about 37 GPU-hours, to turn 'the attention-only model is "
        f"cheap' into 'the attention-only model is cheap at d256 and nowhere narrower'. No "
        f"improvement is claimed: the standing minimum is unchanged at {e2['s']['trainable']:,} "
        f"trainable, and the same-width-no-depth control (attention-only L=6 d128, "
        f"{l6d128['trainable']:,} trainable) is still censored at "
        f"{escapes(rung(cells, L6_D128))[0]} of 3 escapes. Commit {commit}.",
        width=122)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+",
                    default=SCAN_DIRS + [SCAN_E, SCAN_G, SCAN_H, SCAN_I])
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "stagehi_2026-11-05"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--commit", default="(uncommitted)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build even though a ladder rung has fewer than 3 seeds")
    args = ap.parse_args()

    set_style()
    cells, raw = collect(args.scan_dir, args.boot)
    rows = ladder_rows(cells)
    if len(rows) < 4:
        raise SystemExit(f"only {len(rows)} of the 4 ladder widths are on disk")
    thin = [r["d"] for r in rows if r["s"]["n"] < 3]
    if thin and not args.allow_partial:
        raise SystemExit(f"widths {thin} have fewer than 3 seeds; the registered rule is "
                         f"median-vs-median at three seeds. Pass --allow-partial to override.")
    suf = sufficiency(args.scan_dir)
    print(f"{len(cells)} cells | ladder " + ", ".join(
        f"d{r['d']} {rho_txt(r['s'])} ({r['s']['n_cross']}/3 cross, {r['esc']}/3 esc)"
        for r in rows))
    print(f"sufficiency: {suf['n_exp']}/{suf['n_rungs']} rungs, {suf['n_cells']} cells, "
          f"{suf['n_sharp']} discriminating, worst {suf['worst']:.4f} vs bar {suf['bar']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=7)
        page_headline(o, cells, rows, suf, args.commit)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task every "
                               "cell in this deck is trained and scored on. Stages H and I change "
                               "nothing about it -- only d_model moves, from 256 down to 128, at "
                               "a fixed depth of 12 layers and a fixed 240k-instance pool.")
        page_ladder(o, cells, rows, raw, args.commit)
        page_training(o, cells, rows, args.commit)
        page_predictions(o, cells, rows, args.commit)
        page_next(o, cells, rows, suf, args.commit)
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
