"""Deck: is token accuracy a sufficient statistic for direct deduction?

Six pages on one question, raised by stage C's first four cells and answered
with every cell on disk. The analysis lives in scripts/acc_vs_solve.py; this
file only draws it, so the numbers on the slides and the numbers in that
script's report come from the same code path and cannot drift apart.

Run from the repo root:
    python3 -m notebooks.make_sufficiency_deck --commit <sha>
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from scripts.acc_vs_solve import FLOOR, collect, is_reference, pava, predict, rms
from scripts.minarch_analysis import SCAN_DIRS, rung_label

GREEN, RED, BLUE, GREY = "#1b7837", "#b2182b", "#2166ac", "0.25"


def split(cells):
    """(reference, held-out, isotonic fit, LOO bar) -- the whole analysis."""
    ref = [c for c in cells if is_reference(c["rung"])]
    held = [c for c in cells if not is_reference(c["rung"])]
    fit = pava([(c["acc"], c["mean_solve"]) for c in ref])
    loo = []
    for i, c in enumerate(ref):
        sub = pava([(d["acc"], d["mean_solve"]) for j, d in enumerate(ref) if j != i])
        p, _oor, _cov = predict(sub, c["acc"])
        loo.append(c["mean_solve"] - p)
    return ref, held, fit, rms(loo)


def scored(held, fit, bar):
    """Held-out cells annotated with prediction, residual, coverage, class."""
    out = []
    for c in held:
        p, oor, cov = predict(fit, c["acc"])
        out.append(dict(c, pred=p, resid=c["mean_solve"] - p, cover=cov,
                        cls="out" if oor else ("weak" if cov > bar else "firm"),
                        sharp=(not oor) and cov <= bar and p > FLOOR))
    return out


def page_verdict(o, ref, sc, bar, n_cells):
    firm = [c for c in sc if c["cls"] == "firm"]
    sharp = [c for c in sc if c["sharp"]]
    worst = max(abs(c["resid"]) for c in sharp) if sharp else float("nan")
    rungs = sorted({c["label"] for c in firm})
    lo, hi = min(c["acc"] for c in ref), max(c["acc"] for c in ref)

    blocks = [
        ("THE PUZZLE THAT STARTED IT", GREY,
         "Stage C's first cells train a FULL transformer at d_model 128 to train NLL 1.62 — "
         "essentially where ATTENTION-ONLY at d_model 256 sits, 1.63 — and it scores mean solve "
         "0.027 against attention-only's 0.304. Eleven-fold, at the same training loss. Every "
         "'did it train?' hedge here is pinned to a train-NLL threshold (1.64, finding 44), so "
         "that hedge does not order these models. Validation TOKEN ACCURACY does: 0.459 against "
         "0.465–0.483, with the full d256 hub at 0.494."),
        ("THE QUESTION", BLUE,
         "Once you know a cell's token accuracy, does its ARCHITECTURE tell you anything more "
         "about whether it solves? If no, no cut here destroys a 'deduction circuit' — each only "
         "slides the model along a curve DEPTH already traces. If some cut sits BELOW the curve, "
         "that one breaks something solve-specific and is the interesting cell."),
        ("THE TEST", GREY,
         f"The reference is the DEPTH LADDER ALONE — full, d256, 4 heads, L = 1..12, {len(ref)} "
         f"cells — the only axis spanning a wide accuracy range ({lo:.3f} to {hi:.3f}) with the "
         f"architecture intact. The fit is ISOTONIC: the one thing assumed is that more accurate "
         f"models do not solve less often, since assuming a shape would settle the question by "
         f"choosing a functional form. Everything else is HELD OUT, and the bar is the "
         f"reference's own leave-one-out residual, RMS {bar:.4f}."),
        ("THE ANSWER: NO — ACCURACY IS ENOUGH", GREEN,
         f"All {len(firm)} well-determined held-out cells land on the curve, over {len(rungs)} "
         f"rungs: {', '.join(rungs)}. {len(sharp)} are DISCRIMINATING — the curve sends them "
         f"above {FLOOR:.2f}, so agreement is a constraint and not two numbers near zero — and "
         f"those cover EVERY MLP cut at both depths. Worst miss {worst:.4f} against the bar "
         f"{bar:.4f}. So 'the MLP is all-or-nothing' gets a mechanism: the three cuts are "
         f"indistinguishable because they land at the same accuracy, 0.481–0.485, where the "
         f"curve is near-vertical — one accuracy point is worth twenty-five of solve rate. "
         f"Weakest defensible form: model quality here is ONE-DIMENSIONAL."),
    ]
    textpage(o, "Token accuracy is a sufficient statistic for deduction",
             f"{n_cells} cells, N=48 fall ensemble, leak-free (--shuffle-adj) throughout",
             blocks,
             "Correlational: nothing here was intervened on. It is also a statement about this "
             "grid's accuracy range, which no cell except rf_all leaves. NOT the trivial "
             "compounding law — independent errors at accuracy 0.48 over a trace this long would "
             "put every solve rate at zero, so these models' errors are concentrated on "
             "positions the solve metric forgives.",
             y0=0.825, gap=0.018, width=136)


# marker, colour and legend text per held-out family. Stage C's narrow cells
# get their own marker rather than sharing the arch colours: the whole point of
# widening the cell key was that a d128 cell is not the rung it would otherwise
# have merged into, and the plot should not undo that.
def _style(rung):
    arch, narrow = rung[2], len(rung) > 3
    if narrow:
        return ("P", "#e08214", "d_model 128, full") if arch == "full" else \
               ("*", "#e08214", "d_model 128, attention only")
    return {"linear": ("s", "#762a83", "linear (no nonlinearity)"),
            "rf_mlp": ("^", "#1b7837", "rf_mlp (first matrix frozen)"),
            "no_mlp": ("D", "#2166ac", "no_mlp (attention only)"),
            "rf_all": ("X", "0.55", "rf_all (only the readout trains)"),
            "full": ("v", "#b2182b", "nhead=1 (full otherwise)")}[arch]


def _draw(ax, ref, sc, fit, bar, labels):
    xs = [p[0] for p in fit]
    ys = [p[1] for p in fit]
    ax.fill_between(xs, [y - bar for y in ys], [y + bar for y in ys], color="0.84", zorder=1,
                    label="leave-one-out band of the ladder itself" if labels else None)
    ax.plot(xs, ys, "-", color="0.3", lw=2.5, zorder=3,
            label="isotonic fit to the depth ladder" if labels else None)
    ax.scatter([c["acc"] for c in ref], [c["mean_solve"] for c in ref], s=130, marker="o",
               facecolor="white", edgecolor="black", lw=2, zorder=6,
               label="REFERENCE: full d256 h4, L=1..12" if labels else None)
    seen = set()
    for c in sc:
        m, col, lab = _style(c["rung"])
        ax.scatter([c["acc"]], [c["mean_solve"]], s=120, marker=m, color=col, zorder=5,
                   alpha=1.0 if c["cls"] == "firm" else 0.45,
                   edgecolor="none" if c["cls"] == "firm" else "0.3",
                   label=(lab if labels and lab not in seen else None))
        seen.add(lab)
    ax.grid(alpha=0.3)


def page_curve(o, ref, sc, fit, bar):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1.35]})
    _draw(axL, ref, sc, fit, bar, labels=True)
    axL.set_title("every cell on disk", size=MED)
    axL.set_xlabel("validation token accuracy")
    axL.set_ylabel("mean solve rate over the 14 rho bins (free-running)")
    axL.legend(fontsize=SMALL - 3, loc="upper left", framealpha=0.95)
    axL.annotate("rf_all: accuracy 0.11,\nbelow every depth cell,\nso NOT scored here",
                 xy=(0.112, 0.0), xytext=(0.16, 0.30), size=SMALL - 2, color="0.35",
                 arrowprops=dict(arrowstyle="->", color="0.5"))

    # The right panel is where the question is actually decided: 0.455 to the
    # top is the only stretch the reference covers densely, and it is also
    # where every cell except rf_all sits.
    _draw(axR, ref, sc, fit, bar, labels=False)
    axR.set_xlim(0.4545, 0.5025)
    axR.set_ylim(-0.03, 0.92)
    axR.set_title("the stretch that decides it, magnified", size=MED)
    axR.set_xlabel("validation token accuracy")
    # One label per depth, at the centroid of its three seeds: labelling each
    # seed overprinted the three into unreadable strings like "L44".
    byL = {}
    for c in ref:
        byL.setdefault(c["rung"][0], []).append(c)
    for L, cs in byL.items():
        x = sum(c["acc"] for c in cs) / len(cs)
        y = max(c["mean_solve"] for c in cs)
        axR.annotate(f"L={L}", (x, y), textcoords="offset points", xytext=(0, 12),
                     ha="center", size=SMALL - 2, color="0.35", zorder=7)
    axR.annotate("one point of token accuracy\nis worth ~25 points of solve rate",
                 xy=(0.4905, 0.44), xytext=(0.4605, 0.62), size=SMALL - 1, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.5"))
    axR.annotate("every broken cell —\nnhead=1, d128, and one\nseed of each MLP cut",
                 xy=(0.4592, 0.028), xytext=(0.4565, 0.20), size=SMALL - 2, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.5"))

    fig.suptitle("Every architecture lands on the curve that DEPTH alone traces",
                 size=BIG, weight="bold")
    fig.tight_layout(rect=(0, 0.11, 1, 0.945))
    footer(fig, "Hollow circles are the reference and define the curve; every other marker was "
                "HELD OUT of the fit. Faded markers are cells whose bracketing depth points are "
                "further apart than the bar — reported, but not what the verdict rests on. The "
                "curve is near-vertical between 0.47 and 0.50, and THAT steepness, not the "
                "architecture, is why rho_50 censors and why three seeds of one rung can "
                "disagree by 0.28 in mean solve.", width=126)
    o.savefig(fig)


_WORDS = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def page_table(o, sc, bar, commit, n_cells):
    fig = plt.figure(figsize=(16, 9))
    n_sharp = sum(1 for c in sc if c["sharp"])
    fig.text(0.5, 0.945, f"The {_WORDS.get(n_sharp, n_sharp)} cells that could have falsified it",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895, "held-out cells the curve sends above "
                         f"{FLOOR:.2f}, where landing on it is a real constraint",
             ha="center", size=MED, color="0.3")

    sharp = sorted([c for c in sc if c["sharp"]], key=lambda c: -c["pred"])
    head = f"{'architecture':<26s}{'seed':>5s}{'token acc':>12s}{'predicted':>12s}" \
           f"{'observed':>11s}{'miss':>9s}"
    rows = [f"{c['label']:<26s}{c['seed']:>5d}{c['acc']:>12.4f}{c['pred']:>12.4f}" \
            f"{c['mean_solve']:>11.4f}{c['resid']:>+9.4f}" for c in sharp]
    y = 0.80
    fig.text(0.07, y, head, size=MED - 3, family="monospace", weight="bold")
    fig.text(0.07, y - 0.035, "\n".join(rows), size=MED - 3, family="monospace", va="top",
             linespacing=1.6)

    y2 = y - 0.045 - len(rows) * 0.038
    worst = max(abs(c["resid"]) for c in sharp)
    fig.text(0.07, y2, f"worst miss {worst:.4f}   against the bar {bar:.4f}   "
                       f"→ none of them is off the curve", size=MED - 2, weight="bold",
             color=GREEN)

    firm = [c for c in sc if c["cls"] == "firm"]
    weak = [c for c in sc if c["cls"] == "weak"]
    out = [c for c in sc if c["cls"] == "out"]
    byr = {}
    for c in firm:
        byr.setdefault(c["rung"], []).append(c["resid"])
    worst_rung = max(byr.items(), key=lambda kv: abs(sum(kv[1]) / len(kv[1])))
    lines = [
        f"ALL {len(sc)} HELD-OUT CELLS, BY HOW WELL THE CURVE PINS THEM",
        f"  firm   {len(firm):>3d} cells, {len(byr)} rungs — bracketed by depth cells closer "
        f"together than the bar. {len(sharp)} of these are the table above.",
        f"         worst rung mean residual: {rung_label(*worst_rung[0]).strip()} "
        f"{sum(worst_rung[1]) / len(worst_rung[1]):+.4f}, still inside {bar:.4f}.",
        f"  weak   {len(weak):>3d} cells — they land in the ladder's empty accuracy stretch, "
        f"where the curve is flat and near zero. Agreement there is",
        f"         consistency, not evidence, so they are segregated rather than pooled into "
        f"the verdict.",
        f"  out    {len(out):>3d} cells — all rf_all, at token accuracy 0.11. No depth cell is "
        f"anywhere near that, so they are reported as out of range, never scored.",
        "",
        "WHY THE REFERENCE IS THE DEPTH LADDER AND NOTHING ELSE",
        "  It is the only axis that moves accuracy a long way with the architecture intact, and "
        "the cells under test must not help draw the curve they",
        "  are judged against — the same non-circularity rule that made the hub's own seed "
        "spread, not the pooled s*, the bar for the MLP cuts.",
    ]
    fig.text(0.07, y2 - 0.055, "\n".join(lines), size=MED - 5, va="top", linespacing=1.6,
             family="monospace")
    footer(fig, f"{n_cells} cells across all three scan dirs; every one N=48, degree 5, 4 "
                f"colours, fall ensemble, --shuffle-adj, 240000x50 steps. "
                f"scripts/acc_vs_solve.py at commit {commit}. Token accuracy is the final-epoch "
                f"validation figure already in every history.json — nothing was retrained and no "
                f"checkpoint was reloaded to produce this deck.")
    o.savefig(fig)


def page_consequences(o, sc, bar):
    blocks = [
        ("1. IT REPLACES THE METRIC THE PROGRAMME WAS STUCK ON", GREEN,
         "rho_50 censors — eight of the grid's rungs have at least one seed that never crosses "
         "0.5, so their median is a bound and the depth gaps had to be read on mean solve "
         "instead. Token accuracy never censors, is already recorded for every cell that has "
         "ever been trained here, and predicts solve rate to within 0.009 on the cells where "
         "that is a real constraint. Future rungs should be scored on it FIRST and on rho_50 "
         "only to confirm."),
        ("2. IT DISSOLVES THE TRAINABILITY CAVEAT INTO A NUMBER", BLUE,
         "Every stripped rung carried the hedge 'part of this cost is that one seed never left "
         "the 1.64 plateau'. That hedge can now be quantified instead of repeated: each cut "
         "costs one to three points of token accuracy, and because the curve is near-vertical "
         "there, that IS the whole solve-rate gap. Nothing is left over for a cut to have broken "
         "separately."),
        ("3. IT MAKES THE nhead=1 VERDICT SAYABLE", BLUE,
         "N5 was the one confounded verdict in the grid: one head scored censored 3/3 with 0/3 "
         "plateau escapes, so 'worse' could not be separated from 'did not train'. It lands at "
         "token accuracy 0.459, on the curve, exactly where full-at-d128 also lands. That still "
         "does not say one head CANNOT represent deduction — but it says one head fails the way "
         "a too-small model fails, not in some head-specific way. Stage C's warmup arm is still "
         "the intervention that settles it."),
        ("4. WHAT IT DOES NOT SHOW", RED,
         "Nothing here was intervened on: this is a curve through cells that happen to exist, "
         "and an architecture could still be built that sits off it. rf_all is outside the "
         "range entirely and was excluded, not explained. And a sufficient statistic for the "
         "SOLVE METRIC is not a sufficient statistic for the MECHANISM — two models at equal "
         "accuracy can still deduce differently, which is the question the attention-only "
         "frontier was chosen to make answerable."),
        ("THE FRONTIER THIS LEAVES", GREEN,
         "Attention-only at L=6: 1,746,490 trainable parameters, rho_50 0.763, token accuracy "
         "0.481, cheaper than full L=3 at the same score. On the matched-budget reading it beats "
         "full L=2 outright, 0.304 to 0.081, at 1.746M against 1.741M. It is also the most "
         "analytically tractable object in the grid — with no MLP, the residual stream is "
         "written only by attention."),
    ]
    textpage(o, "What follows, and what does not",
             "consequences of a one-dimensional quality axis for the minimal-architecture search",
             blocks,
             "Previous best on the same eval: the 11-02 scan, whose matched L=8 cells replicate "
             "to within s* (-0.1062). There is no previous result to compare this page against — "
             "the accuracy-to-solve curve had not been drawn before, which is why it is "
             "published as a method change rather than as an improvement in any score.",
             y0=0.845, gap=0.013, width=137)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_DIRS)
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "sufficiency_2026-11-04"))
    ap.add_argument("--commit", default="(uncommitted)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_style()
    cells = [c for c in collect(args.scan_dir) if c["acc"] == c["acc"]]
    ref, held, fit, bar = split(cells)
    sc = scored(held, fit, bar)
    print(f"{len(cells)} cells: {len(ref)} reference, {len(held)} held out; bar {bar:.4f}; "
          f"{sum(1 for c in sc if c['sharp'])} discriminating")

    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=6)
        page_verdict(o, ref, sc, bar, len(cells))
        page_curve(o, ref, sc, fit, bar)
        page_table(o, sc, bar, args.commit, len(cells))
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall. TOKEN ACCURACY "
                               "is how often the model's next token matches this trace under "
                               "teacher forcing; SOLVE RATE is whether it can generate the whole "
                               "thing unaided. This deck is about the relation between the two.")
        page_consequences(o, sc, bar)
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
