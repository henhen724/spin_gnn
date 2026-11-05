"""Deck: stage C closed, and what happened when width was handed back.

Stage C (51114734, 15/15) finished the width ladder at d_model 256/192/128/64,
and stage D (51151957) asked the question no earlier cell in this grid could
answer -- give an ABLATED model MORE of a resource than the model it is scored
against, and does it cross? D2 is attention-only at d384, 1.5x the hub's
residual stream with no MLP anywhere.

    python3 -m notebooks.make_widtharm_deck --commit $(git rev-parse --short HEAD)

Three rules this file follows on purpose:

  * EVERY VERDICT IS barrier_call's, not the deck's. scripts/capacity_back.py
    was committed before the deciding cells landed; this deck imports it and
    renders what it returns. A deck that restates the conclusion in its own
    words is a deck that can drift away from the registered rule.
  * STAGE E IS NOT IN HERE. 51171824 is still running, SCAN_E stays out of
    SCAN_DIRS (its rungs carry default hyperparameters and would pool into the
    s* every verdict here is scored against), and the last page says what each
    of its outcomes would mean and nothing more.
  * It refuses to build on an incomplete stage D unless asked, because the
    registered rule is median-vs-median at three seeds and a two-seed median
    on a slide is a number people quote.

s* is quoted frozen at 0.1484 / 0.2918 and never recomputed here.
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from scripts.capacity_back import (BARRIER_HI, BARRIER_LO, D1, D2, E2_CAPACITY, FREE,
                                   FULL_L8, FULL_L12, barrier_call)
from scripts.minarch_analysis import (SCAN_C, SCAN_D, SCAN_DEFAULT, SCAN_L6, cell_key,
                                      cell_metrics, load_all)
from scripts.stagec_score import (HUB, PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt, rung,
                                  summarize)

NO_MLP_D256 = (6, 4, "no_mlp")
FULL_C = "#13507a"      # the complete model
NOMLP_C = "#1a7a3c"     # attention-only
BAND_C = "#c1272d"      # the band, and anything that lands on it
NOTE_C = "#6a3d9a"      # things registered in advance
DEAD_C = "0.45"         # cells that never escaped the plateau

# The two width ladders, complete and attention-only, as (d_model, rung key).
FULL_WIDTHS = [(256, HUB), (192, D1), (128, (6, 4, "full", 128, 1024, 0)),
               (64, (6, 4, "full", 64, 1024, 0))]
NOMLP_WIDTHS = [(384, D2), (256, NO_MLP_D256), (128, (6, 4, "no_mlp", 128, 1024, 0))]


def collect(scan_dirs, boot):
    by = defaultdict(list)
    for r in load_all(scan_dirs):
        by[cell_key(r)].append(r)
    return {k: dict(cell_metrics(v, boot), path=v[0]["path"]) for k, v in by.items()}


def ladder_rows(cells, widths):
    """[(d_model, summary, per-seed mean_solve, n_escapes)] for rungs on disk."""
    rows = []
    for d, key in widths:
        ms = rung(cells, key)
        if not ms:
            continue
        rows.append((d, summarize(ms), [m["mean_solve"] for m in ms], escapes(ms)[0]))
    return sorted(rows, key=lambda r: -r[0])


def call(cells, key, s):
    """barrier_call on a rung, with the trainability count it reads."""
    return barrier_call(s, escapes(rung(cells, key))[0])


def page_headline(o, cells, hub, d2, n_cells, commit):
    _, d2_status = call(cells, D2, d2)
    nomlp = summarize(rung(cells, NO_MLP_D256))
    n_esc = escapes(rung(cells, D2))[0]
    crossed = d2_status == "falsified"
    # A one-seed rung has no spread; print nothing rather than "[spread nan]".
    sm = d2["spread_mean"]
    spr = f" [spread {sm:.4f}]" if sm == sm else ""
    textpage(
        o,
        "Width bought the MLP back" if crossed else "1.5x the width does not buy the MLP back",
        f"N=48, degree 5, 4 colours, leak-free | stage C closed 15/15, stage D's width "
        f"arm at {d2['n']}/3 seeds | {n_cells} cells",
        [
            ("THE TEST NO EARLIER CELL IN THIS GRID PERFORMED", FULL_C,
             f"Every ablated rung measured before stage D ran at AT MOST the hub's depth and at "
             f"most its width, so no ablated model had ever been handed MORE of a resource than "
             f"the model it is scored against. D2 is attention-only -- no MLP anywhere -- at "
             f"d_model 384, 1.5x the hub's residual stream. At {d2['trainable']:,} trainable "
             f"against the hub's {hub['trainable']:,} it is still the smaller model; what it has "
             f"more of is WIDTH."),
            ("WHAT THE PRE-REGISTERED SCORER RETURNS", BAND_C,
             f"rho_50 median {rho_txt(d2)} against the hub's {hub['rho50']:.3f}, mean_solve "
             f"{d2['mean_solve']:.4f}{spr} against {hub['mean_solve']:.4f}, {d2['n_cross']} of "
             f"{d2['n']} seeds crossing, plateau escapes {n_esc}/{d2['n']}. Attention-only at "
             f"the hub's own width sits at {nomlp['rho50']:.3f} / {nomlp['mean_solve']:.4f}. "
             f"D2's registered prediction was rho_50 <= {FREE:.3f} -- that 1.5x width buys the "
             f"MLP back. The scorer's call on it is {d2_status.upper()}, printed in full on "
             f"page 6."),
            ("AND THE CALL WAS ON RECORD BEFORE THE CELL EXISTED", NOTE_C,
             f"Stage E's header, committed while stage D was still running, explicitly backed "
             f"D2's FALSIFIER over D2's own prediction: if the barrier reading is right then "
             f"capacity in any form is not the currency, so d384 attention-only should land "
             f"beside d256 attention-only rather than reaching the hub. That is a registered "
             f"prediction about another job's unfinished cell, and it is scored here exactly as "
             f"it was written."),
        ],
        f"rho_50 = realized hint density at which solve rate crosses 0.5 on a 14-point x "
        f"400-instance grid, LOWER IS BETTER, reported CENSORED when no bin crosses. mean_solve = "
        f"unweighted mean over the 14 bins, which never censors. The seed-spread bar s* = "
        f"{S_RHO50} / {S_MEAN} was frozen before any of these cells ran. Commit {commit}.",
        width=122)


def page_ladders(o, cells, commit):
    """The two width ladders side by side: complete, and attention-only."""
    full_rows = ladder_rows(cells, FULL_WIDTHS)
    nomlp_rows = ladder_rows(cells, NOMLP_WIDTHS)

    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "Two width ladders: the complete model, and the same model with no MLP",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, "L=6, 4 heads, ff=1024 wherever an MLP exists; everything else at the "
                         "hub's settings, 3 seeds per rung",
             ha="center", size=MED - 2, color="0.3")

    ax = fig.add_axes([0.07, 0.28, 0.52, 0.49])
    # Different marker shapes, not just colours: the two ladders land within
    # 0.002 of each other at d128 and circles alone overprinted there.
    for rows, colour, mk, name in ((full_rows, FULL_C, "o", "full (MLP intact)"),
                                   (nomlp_rows, NOMLP_C, "s", "attention-only")):
        ax.plot([r[0] for r in rows], [r[1]["mean_solve"] for r in rows], "-",
                marker=mk, color=colour, lw=2.5, ms=11, zorder=3, label=name)
        for d, s, seeds, n_esc in rows:
            ax.plot([d] * len(seeds), seeds, "o", color="0.72", ms=6, zorder=2)
            # Hollow = the rung never got below the plateau, so its number is an
            # optimization failure and says nothing about width.
            if n_esc == 0:
                ax.plot([d], [s["mean_solve"]], mk, mfc="white", mec=colour, mew=2.5,
                        ms=12, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 192, 256, 384])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("d_model (residual stream width, log scale)")
    ax.set_ylabel("mean_solve   (higher is better)")
    ax.set_title("Width is a cliff; the MLP is a step off it", size=MED)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95)

    lines = ["HOLLOW MARKER = never escaped the",
             f"{PLATEAU} plateau, so the cell is an",
             "optimization failure and its number",
             "is not a statement about width.",
             "", "        d  mean_solve   trainable  esc"]
    for rows, name in ((full_rows, "full"), (nomlp_rows, "no_mlp")):
        lines.append(f"  {name}")
        for d, s, _, n_esc in rows:
            lines.append(f"     {d:>4}      {s['mean_solve']:.4f}  {s['trainable']:>10,}  "
                         f"{n_esc}/{s['n']}")
    fig.text(0.635, 0.79, "\n".join(lines), size=SMALL - 1, va="top", family="monospace",
             linespacing=1.6)
    footer(fig, f"Per-seed values in grey. Per trainable parameter, width buys more than the MLP "
                f"does: attention-only at d256 has fewer parameters than the complete model at "
                f"d128 and solves, while the complete model at d128 never crosses on any seed. "
                f"The d64 rung is drawn hollow because all three of its seeds finished ABOVE the "
                f"plateau -- it did not train in 375k steps, which is a different claim from "
                f"'too narrow to deduce'. Commit {commit}.")
    o.savefig(fig)


def page_band(o, cells, d1, d2, hub, commit):
    """Where the two stage D rungs landed, against the band and both predictions."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "Where the two stage D rungs landed, against predictions made first",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, "Every prediction below was written into a committed sbatch header "
                         "before the cell it describes finished",
             ha="center", size=MED - 2, color="0.3")

    ax = fig.add_axes([0.07, 0.26, 0.55, 0.51])
    ax.axhspan(BARRIER_LO, BARRIER_HI, color=BAND_C, alpha=0.10, zorder=0)
    ax.axhline(FREE, color=NOTE_C, lw=2, ls="--", zorder=1)
    # Right-aligned: the left end of this line runs through the D1 label.
    ax.text(0.985, FREE, f"(B) CAPACITY predicts at or below {FREE:.3f}", size=SMALL,
            color=NOTE_C, va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(hub["rho50"], color=FULL_C, lw=2, ls=":", zorder=1)
    ax.text(0.015, hub["rho50"], f"the hub, {hub['rho50']:.3f}", size=SMALL, color=FULL_C,
            va="bottom", transform=ax.get_yaxis_transform())

    # Label offsets are given per point rather than computed: the six rungs sit
    # in two tight clusters and anything automatic overprinted them.
    for key, name, colour, off, ha in (
            (HUB, "hub: full d256", FULL_C, (0, -32), "center"),
            (D1, "D1: full d192", FULL_C, (-16, 18), "right"),
            (NO_MLP_D256, "no_mlp d256", NOMLP_C, (16, -6), "left"),
            (D2, "D2: no_mlp d384", NOMLP_C, (16, 18), "left"),
            (FULL_L8, "full L=8", FULL_C, (0, -32), "center"),
            (FULL_L12, "full L=12", FULL_C, (-18, -30), "right")):
        ms = rung(cells, key)
        if not ms:
            continue
        s = summarize(ms)
        if not s["rho50_readable"]:
            continue
        stage_d = key in (D1, D2)
        ax.scatter([s["trainable"]], [s["rho50"]], s=240 if stage_d else 130,
                   marker="D" if stage_d else "o", color=colour, zorder=4)
        # Only seeds that actually crossed 0.5 have a readable rho_50.
        seeds = [m["rho50"] for m in ms if m["flag"] == "ok"]
        ax.plot([s["trainable"]] * len(seeds), seeds, "_", color="0.6", ms=18, zorder=3)
        ax.annotate(name, (s["trainable"], s["rho50"]), textcoords="offset points",
                    xytext=off, ha=ha, size=SMALL, color=colour,
                    weight="bold" if stage_d else "normal")
    ax.margins(x=0.10, y=0.12)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("rho_50   (lower is better)")
    ax.set_title(f"Shaded: the [{BARRIER_LO:.2f}, {BARRIER_HI:.2f}] band. "
                 f"Tick marks: crossing seeds", size=MED)
    ax.grid(alpha=0.25)

    d1_in = 0.10 <= d1["mean_solve"] <= 0.45
    block = [
        "REGISTERED, THEN SCORED",
        "",
        f"D1   full d192, {d1['trainable']:,}",
        "  asked  mean_solve in [0.10, 0.45],",
        "         and strictly between its two",
        "         neighbours on the ladder",
        f"  got    {d1['mean_solve']:.4f}, rho_50 {rho_txt(d1)}",
        f"         -> {'INSIDE the window' if d1_in else 'OUTSIDE the window'}",
        "",
        f"D2   no_mlp d384, {d2['trainable']:,}",
        f"  asked  rho_50 <= {FREE:.3f}",
        f"  got    {rho_txt(d2)}",
        "",
        "Censored rungs cannot be placed on",
        "this axis and are left out; all of",
        "them sit BELOW the band, so leaving",
        "them out understates the pile-up",
        "rather than flattering it.",
    ]
    fig.text(0.655, 0.79, "\n".join(block), size=SMALL - 1, va="top", family="monospace",
             linespacing=1.55)
    footer(fig, f"The hub and the complete model at L=8 and L=12 are the paired controls: same "
                f"14-point grid, same 375k-step budget, same eval. s* = {S_RHO50} is the "
                f"seed-to-seed spread measured at fixed architecture, which is why the band is a "
                f"band and not a line. Commit {commit}.")
    o.savefig(fig)


def page_verdicts(o, cells, d1, d2, commit):
    d1_text, d1_status = call(cells, D1, d1)
    d2_text, d2_status = call(cells, D2, d2)
    textpage(
        o,
        "The rule, and the one place it contradicts itself",
        "Transcribed from the sbatch headers into scripts/capacity_back.py, "
        "pinned by 15 tests before the cells landed",
        [
            (f"D1 -- {d1_status.upper()}   |   D2 -- {d2_status.upper()}", FULL_C,
             f"D1: {d1_text}   D2: {d2_text}"),
            ("THE ORDER THE BRANCHES FIRE IN, WHICH IS ITSELF REGISTERED", DEAD_C,
             f"Trainability first: 0 plateau escapes means 'did not train in 375k steps' and is "
             f"NEVER evidence in either direction -- the trap stage C's P2 fell into. Then fewer "
             f"than three seeds, where nothing can be scored. Then fewer than two crossing seeds, "
             f"where the falsifier cannot fire. Then a median below {BARRIER_LO}, which ends the "
             f"barrier reading on that rung alone. Then the tie-break region up to {FREE:.3f}, "
             f"then the band's upper edge at {BARRIER_HI:.2f}. A rung that never got the chance "
             f"to break "
             f"the reading is not counted as one that held it."),
            ("AND THE CONTRADICTION, REPORTED RATHER THAN RESOLVED QUIETLY", BAND_C,
             f"The stage E header states a tie-break sending any median in [{BARRIER_LO:.2f}, "
             f"{FREE:.3f}] to the barrier reading, and nine lines later a falsifier killing that "
             f"same reading for any median below {FREE:.3f} on 2 of 3 crossing seeds. They "
             f"disagree on the whole interval [{BARRIER_LO:.2f}, {FREE:.3f}). The scorer follows "
             f"the TIE-BREAK -- written for "
             f"exactly that overlap, and the conservative choice, since the capacity reading is "
             f"the claim needing evidence -- and PRINTS BOTH readings whenever a median lands "
             f"there. Choosing one silently is how a pre-registration becomes a post-hoc one."),
        ],
        f"tests/test_capacity_back.py pins every threshold to the header both as literal text and "
        f"as the arithmetic that produced it (hub + s* = {FREE:.3f}, full L=8 + s* = "
        f"{E2_CAPACITY:.3f}), "
        f"and pins the branch order. Moving one threshold by 0.02 turns three quarters of the "
        f"tie-break region into a falsification, and nothing else in the repo would notice. "
        f"Commit {commit}.",
        width=122)


def page_next(o, cells, hub, commit):
    nomlp = summarize(rung(cells, NO_MLP_D256))
    l12 = summarize(rung(cells, FULL_L12))
    textpage(
        o,
        "What this does not settle, and what is still running",
        "Stage E (51171824) hands the same ablation one more DEPTH instead of "
        "one more width -- the second axis",
        [
            ("THE DEPTH ARM, DELIBERATELY NOT IN THIS DECK", NOMLP_C,
             f"Attention-only at L=8 and L=12, three seeds each, everything else at the hub's "
             f"settings. Both depths were chosen because the complete model is already measured "
             f"there, so each rung has a paired partner at identical depth. Registered falsifier: "
             f"either rung with a median below the tie-break floor on at least 2 of 3 crossing "
             f"seeds ends the barrier reading, with no partial credit and no 'but the other rung "
             f"held'."),
            ("THE WAY IT PRODUCES NOTHING, NAMED IN ADVANCE", BAND_C,
             f"Trainability, not capacity. Attention-only escapes the {PLATEAU} plateau on 2 of 3 "
             f"seeds at L=6 and 0 of 3 at L=4, so a DEEPER attention-only stack failing to escape "
             f"is the expected failure mode and will be written 'did not train in 375k steps'. "
             f"The budget is deliberately not raised to rescue it: a rung trained on a different "
             f"budget cannot be read against the ladder it is compared to."),
            ("NO IMPROVEMENT IS CLAIMED HERE, AND NONE WAS SOUGHT", NOTE_C,
             f"Previous best on this eval, same grid and same budget: the complete model at L=12, "
             f"rho_50 {l12['rho50']:.3f} / mean_solve {l12['mean_solve']:.4f} at "
             f"{l12['trainable']:,} trainable. Nothing in this deck approaches it. Since 11-04 "
             f"the brief is the SMALLEST architecture that still learns direct deduction, so the "
             f"number that matters is the standing minimum: attention-only at d256, "
             f"{nomlp['trainable']:,} trainable, rho_50 {nomlp['rho50']:.3f} -- "
             f"{hub['trainable'] / nomlp['trainable']:.2f}x fewer parameters than the hub and "
             f"{(nomlp['rho50'] - hub['rho50']) / S_RHO50:.1f} s* worse. Next, once the depth arm "
             f"lands: a 3-seed replicate of whichever rung survives at N=96."),
        ],
        f"SCAN_E is absent from SCAN_DIRS and from this deck on purpose: stage E's rungs carry "
        f"entirely default hyperparameters and would pool into the s* every verdict here is "
        f"scored against. When it lands it gets its own write-up, not a silent edit to this one. "
        f"Commit {commit}.",
        width=122)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_DEFAULT, SCAN_L6, SCAN_C, SCAN_D])
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "widtharm_2026-11-05"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--commit", default="(uncommitted)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build even though a stage D rung has fewer than 3 seeds")
    args = ap.parse_args()

    set_style()
    cells = collect(args.scan_dir, args.boot)
    d1_ms, d2_ms = rung(cells, D1), rung(cells, D2)
    if not d1_ms or not d2_ms:
        raise SystemExit("stage D has nothing on disk for D1 or D2 yet")
    hub, d1, d2 = summarize(rung(cells, HUB)), summarize(d1_ms), summarize(d2_ms)
    if (d1["n"] < 3 or d2["n"] < 3) and not args.allow_partial:
        raise SystemExit(
            f"stage D is at D1 {d1['n']}/3, D2 {d2['n']}/3 seeds. The registered rule is "
            f"median-vs-median at three seeds, and a two-seed median on a slide is a number "
            f"people quote. Pass --allow-partial only if that is really what you want.")
    print(f"{len(cells)} cells | D1 {d1['n']}/3 seeds, D2 {d2['n']}/3 seeds")

    os.makedirs(args.out, exist_ok=True)
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=7)
        page_headline(o, cells, hub, d2, len(cells), args.commit)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task every "
                               "cell in this deck is trained and scored on. The attention-only "
                               "models have to produce this with no MLP anywhere in the stack.")
        page_ladders(o, cells, args.commit)
        page_band(o, cells, d1, d2, hub, args.commit)
        page_verdicts(o, cells, d1, d2, args.commit)
        page_next(o, cells, hub, args.commit)
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
