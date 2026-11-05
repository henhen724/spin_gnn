"""Deck: stage E, the DEPTH arm of "can an ablated model be handed capacity back?"

Stage D gave an attention-only model more WIDTH than the hub (d384 against
d256) and the deck for it is notebooks/make_widtharm_deck.py. Stage E
(51171824) gives it more DEPTH instead: attention-only at L=8 and L=12, three
seeds each, everything else at the hub's settings. Both depths were chosen
because the COMPLETE model is already measured there, so every rung here has a
paired partner at identical depth, identical budget and the identical 14-point
grid.

    python3 -m notebooks.make_deptharm_deck --commit $(git rev-parse --short HEAD)

Written and dry-run while 51171824 was still at E1 2/3 and E2 1/3 seeds, so
the layout and the wording could not be chosen to suit the answer. The same
three rules the width deck follows apply here:

  * EVERY VERDICT IS barrier_call's. scripts/capacity_back.py was committed
    before any stage E cell landed; this deck imports it and renders what it
    returns, including the roll-up text, which main() and page 6 now share
    rather than paraphrase.
  * SCAN_E IS IN HERE, and it has to be -- it is the subject. It stays out of
    SCAN_DIRS, and therefore out of every other deck, because its rungs carry
    entirely default hyperparameters and would pool into the s* that stages
    A-D were scored against. s* is quoted frozen at 0.1484 / 0.2918 and is
    never recomputed here either.
  * It refuses to build on an incomplete arm unless asked, because the
    registered rule is median-vs-median at three seeds and a two-seed median
    on a slide is a number people quote.
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from scripts.capacity_back import (BARRIER_HI, BARRIER_LO, D2, E1, E2, E2_CAPACITY, FREE,
                                   FULL_L8, FULL_L12, NOMLP_L6, arm_rollup, barrier_call)
from scripts.minarch_analysis import (SCAN_C, SCAN_D, SCAN_DEFAULT, SCAN_E, SCAN_L6,
                                      cell_key, cell_metrics, load_all)
from scripts.stagec_score import (HUB, PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt, rung,
                                  summarize)

FULL_C = "#13507a"      # the complete model
NOMLP_C = "#1a7a3c"     # attention-only
BAND_C = "#c1272d"      # the band, and anything that lands on it
NOTE_C = "#6a3d9a"      # things registered in advance
DEAD_C = "0.45"         # cells that never escaped the plateau

# The two depth ladders, as (num_layers, rung key). The complete model is
# measured at seven depths; attention-only existed at two before stage E.
FULL_DEPTHS = [(L, (L, 4, "full")) for L in (1, 2, 3, 4, 6, 8, 12)]
NOMLP_DEPTHS = [(L, (L, 4, "no_mlp")) for L in (4, 6, 8, 12)]


def collect(scan_dirs, boot):
    by = defaultdict(list)
    for r in load_all(scan_dirs):
        by[cell_key(r)].append(r)
    return {k: dict(cell_metrics(v, boot), path=v[0]["path"]) for k, v in by.items()}


def ladder_rows(cells, depths):
    """[(L, summary, per-seed mean_solve, n_escapes)] for rungs on disk."""
    rows = []
    for L, key in depths:
        ms = rung(cells, key)
        if not ms:
            continue
        rows.append((L, summarize(ms), [m["mean_solve"] for m in ms], escapes(ms)[0]))
    return sorted(rows)


def call(cells, key, s):
    """barrier_call on a rung, with the trainability count it reads."""
    return barrier_call(s, escapes(rung(cells, key))[0])


def pair_txt(s, partner, what):
    """One line of rung-vs-depth-partner, the comparison stage E exists for."""
    return (f"{what} {rho_txt(s)} / {s['mean_solve']:.4f} at {s['trainable']:,} trainable, "
            f"against the complete model at the SAME depth, {rho_txt(partner)} / "
            f"{partner['mean_solve']:.4f} at {partner['trainable']:,}")


def page_headline(o, cells, hub, e1, e2, st, n_cells, commit):
    l8, l12 = summarize(rung(cells, FULL_L8)), summarize(rung(cells, FULL_L12))
    nomlp6 = summarize(rung(cells, NOMLP_L6))
    if "falsified" in st.values():
        title = "Depth bought the MLP back"
    elif set(st.values()) <= {"held", "worse"}:
        title = "Two more depths, and the attention-only rung does not move"
    else:
        title = "The depth arm did not produce a rung that could be scored"
    e1_esc, e2_esc = escapes(rung(cells, E1))[0], escapes(rung(cells, E2))[0]
    textpage(
        o,
        title,
        f"N=48, degree 5, 4 colours, leak-free | stage E at E1 {e1['n']}/3, E2 {e2['n']}/3 "
        f"seeds | {n_cells} cells",
        [
            ("THE SECOND AXIS, WITH A PAIRED CONTROL AT IDENTICAL DEPTH", NOMLP_C,
             f"Stage D handed an attention-only model more WIDTH than the hub and it did not "
             f"cross. Stage E hands it more DEPTH: no MLP anywhere, at L=8 and at L=12, against "
             f"a hub at L=6. Both depths were picked because the COMPLETE model is already "
             f"measured there on this exact grid and budget, so neither rung is compared to an "
             f"extrapolation -- full L=8 sits at {rho_txt(l8)} and full L=12 at {rho_txt(l12)}, "
             f"and attention-only at the hub's own depth sits at {rho_txt(nomlp6)}."),
            ("WHAT THE PRE-REGISTERED SCORER RETURNS", BAND_C,
             f"E1, {pair_txt(e1, l8, 'no_mlp L=8:')}; plateau escapes {e1_esc}/{e1['n']}, "
             f"{e1['n_cross']} of {e1['n']} seeds crossing -- {st['E1'].upper()}. "
             f"E2, {pair_txt(e2, l12, 'no_mlp L=12:')}; plateau escapes {e2_esc}/{e2['n']}, "
             f"{e2['n_cross']} of {e2['n']} seeds crossing -- {st['E2'].upper()}. The two calls "
             f"are printed in full on page 6, in the scorer's own words."),
            ("AND THE WAY IT PRODUCES NOTHING WAS NAMED FIRST", NOTE_C,
             f"Attention-only escapes finding 44's {PLATEAU} plateau on 2 of 3 seeds at L=6 and 0 "
             f"of 3 at L=4, so a DEEPER attention-only stack failing to escape is the EXPECTED "
             f"failure mode, written into the header before the job ran and scored here as 'did "
             f"not train in 375k steps' -- never as evidence that the barrier holds. The budget "
             f"was deliberately not raised to rescue it: a rung trained on a different budget "
             f"cannot be read against the ladder it is compared to. Stage G (51194118) moves "
             f"that knob on purpose, as its own arm."),
        ],
        f"rho_50 = realized hint density at which solve rate crosses 0.5 on a 14-point x "
        f"400-instance grid, LOWER IS BETTER, reported CENSORED when no bin crosses. mean_solve = "
        f"unweighted mean over the 14 bins, which never censors. The seed-spread bar s* = "
        f"{S_RHO50} / {S_MEAN} was frozen before any of these cells ran. Commit {commit}.",
        width=122)


def page_ladders(o, cells, commit):
    """The two depth ladders: complete, and attention-only."""
    full_rows = ladder_rows(cells, FULL_DEPTHS)
    nomlp_rows = ladder_rows(cells, NOMLP_DEPTHS)

    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "Two depth ladders: the complete model, and the same model with no MLP",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, "d_model 256, 4 heads, ff=1024 wherever an MLP exists; everything else "
                         "at the hub's settings, 3 seeds per rung",
             ha="center", size=MED - 2, color="0.3")

    ax = fig.add_axes([0.07, 0.28, 0.52, 0.49])
    # Markers are drawn per rung, not by the line, so a hollow one can have NO
    # face: the two ladders run close together at the shallow end and a white
    # face hid the other ladder's point on the width deck.
    for rows, colour, mk in ((nomlp_rows, NOMLP_C, "s"), (full_rows, FULL_C, "o")):
        ax.plot([r[0] for r in rows], [r[1]["mean_solve"] for r in rows], "-",
                color=colour, lw=2.5, zorder=3)
        for L, s, seeds, n_esc in rows:
            ax.plot([L] * len(seeds), seeds, "o", color="0.72", ms=6, zorder=2)
            # Hollow = the rung never got below the plateau, so its number is an
            # optimization failure and says nothing about depth.
            hollow = n_esc == 0
            ax.plot([L], [s["mean_solve"]], mk, mfc="none" if hollow else colour,
                    mec=colour, mew=2.5 if hollow else 1.0, ms=14 if hollow else 11,
                    zorder=5 if colour == FULL_C else 4)
    ax.axvline(6, color="0.6", lw=1.2, ls=":", zorder=1)
    ax.text(6, 0.02, " the hub's depth", size=SMALL, color="0.45",
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([L for L, _ in FULL_DEPTHS])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("num_layers (log scale)")
    ax.set_ylabel("mean_solve   (higher is better)")
    ax.set_title("Everything right of the dotted line is capacity the hub does not have",
                 size=MED)
    ax.grid(alpha=0.25)
    ax.legend(handles=[Line2D([], [], color=FULL_C, marker="o", lw=2.5, ms=11,
                              label="full (MLP intact)"),
                       Line2D([], [], color=NOMLP_C, marker="s", lw=2.5, ms=11,
                              label="attention-only")],
              loc="upper left", framealpha=0.95)

    lines = ["HOLLOW MARKER = never escaped the",
             f"{PLATEAU} plateau, so the cell is an",
             "optimization failure and its number",
             "is not a statement about depth.",
             "", "        L  mean_solve   trainable  esc"]
    for rows, name in ((full_rows, "full"), (nomlp_rows, "no_mlp")):
        lines.append(f"  {name}")
        for L, s, _, n_esc in rows:
            lines.append(f"     {L:>4}      {s['mean_solve']:.4f}  {s['trainable']:>10,}  "
                         f"{n_esc}/{s['n']}")
    fig.text(0.635, 0.79, "\n".join(lines), size=SMALL - 1, va="top", family="monospace",
             linespacing=1.6)
    footer(fig, f"Per-seed values in grey. The complete model's ladder is the reference the whole "
                f"grid is built on: it descends with depth and keeps descending past L=6. The "
                f"question stage E asks is whether the attention-only ladder does the same thing "
                f"when it is given the same two extra depths, or whether it flattens wherever the "
                f"pile-up at 0.76 sits. A hollow marker at the deep end answers neither question "
                f"-- it says the cell did not train. Commit {commit}.")
    o.savefig(fig)


def page_band(o, cells, e1, e2, hub, commit):
    """Where the two stage E rungs landed, against the band and both readings."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "Where the two depth rungs landed, against predictions made first",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, "Every line below was written into a committed sbatch header before the "
                         "cell it describes finished",
             ha="center", size=MED - 2, color="0.3")

    ax = fig.add_axes([0.07, 0.26, 0.55, 0.51])
    ax.axhspan(BARRIER_LO, BARRIER_HI, color=BAND_C, alpha=0.10, zorder=0)
    ax.axhline(FREE, color=NOTE_C, lw=2, ls="--", zorder=1)
    ax.text(0.985, FREE, f"(B) for E1: at or below {FREE:.3f}", size=SMALL, color=NOTE_C,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(E2_CAPACITY, color=NOTE_C, lw=2, ls="-.", zorder=1)
    ax.text(0.985, E2_CAPACITY, f"(B) for E2: at or below {E2_CAPACITY:.3f}", size=SMALL,
            color=NOTE_C, va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(hub["rho50"], color=FULL_C, lw=2, ls=":", zorder=1)
    ax.text(0.015, hub["rho50"], f"the hub, {hub['rho50']:.3f}", size=SMALL, color=FULL_C,
            va="bottom", transform=ax.get_yaxis_transform())

    # Label offsets are given per point rather than computed: the complete
    # model's deep rungs sit in one tight cluster and anything automatic
    # overprinted them on the width deck.
    for key, name, colour, off, ha in (
            (HUB, "hub: full L=6", FULL_C, (0, -32), "center"),
            # Up-left, not right: E1's diamond lands within a few hundred
            # thousand parameters of this marker and sat on top of the label.
            (NOMLP_L6, "no_mlp L=6", NOMLP_C, (-14, 10), "right"),
            # BOTH paired controls go UP. With L=12's label sent downward the
            # two ran into each other in the middle of the plot, because a
            # right-aligned label at 9.6M parameters reaches back past 5M.
            (FULL_L8, "full L=8 (E1's partner)", FULL_C, (-16, 18), "right"),
            (FULL_L12, "full L=12 (E2's partner)", FULL_C, (-12, 18), "right"),
            (D2, "D2: no_mlp d384 (width arm)", NOMLP_C, (16, -26), "left"),
            # Below the marker: above it is the (B) threshold line, and the
            # label was printing straight through the dashes.
            (E1, "E1: no_mlp L=8", NOMLP_C, (0, -30), "center"),
            (E2, "E2: no_mlp L=12", NOMLP_C, (14, -28), "left")):
        ms = rung(cells, key)
        if not ms:
            continue
        s = summarize(ms)
        if not s["rho50_readable"]:
            continue
        stage_e = key in (E1, E2)
        ax.scatter([s["trainable"]], [s["rho50"]], s=240 if stage_e else 130,
                   marker="D" if stage_e else "o", color=colour, zorder=4)
        # Only seeds that actually crossed 0.5 have a readable rho_50.
        seeds = [m["rho50"] for m in ms if m["flag"] == "ok"]
        ax.plot([s["trainable"]] * len(seeds), seeds, "_", color="0.6", ms=18, zorder=3)
        ax.annotate(name, (s["trainable"], s["rho50"]), textcoords="offset points",
                    xytext=off, ha=ha, size=SMALL, color=colour,
                    weight="bold" if stage_e else "normal")
    ax.margins(x=0.10, y=0.12)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("rho_50   (lower is better)")
    ax.set_title(f"Shaded: the [{BARRIER_LO:.2f}, {BARRIER_HI:.2f}] band. "
                 f"Tick marks: crossing seeds", size=MED)
    ax.grid(alpha=0.25)

    block = [
        "REGISTERED, THEN SCORED",
        "",
        f"E1   no_mlp L=8, {e1['trainable']:,}",
        f"  (A)    rho_50 in [{BARRIER_LO}, {BARRIER_HI}],",
        "         mean_solve in [0.26, 0.38]",
        f"  (B)    rho_50 <= {FREE} -- joins the hub",
        f"  got    {rho_txt(e1)} / {e1['mean_solve']:.4f}",
        "",
        f"E2   no_mlp L=12, {e2['trainable']:,}",
        f"  (A)    rho_50 in [{BARRIER_LO}, {BARRIER_HI}] again",
        f"  (B)    rho_50 <= {E2_CAPACITY} -- full L=8's rung",
        f"  got    {rho_txt(e2)} / {e2['mean_solve']:.4f}",
        "",
        "FALSIFIER OF (A), from the header:",
        f"EITHER rung with a median below {FREE}",
        "on >=2 of 3 crossing seeds ends the",
        "barrier reading. No partial credit, no",
        "'but the other rung held'. The overlap",
        "with the tie-break clause is a conflict",
        "IN THE REGISTRATION; page 6 reports it",
        "rather than resolving it quietly.",
        "",
        "Censored rungs cannot be placed on this",
        "axis and are left out; all of them sit",
        "BELOW the band, so leaving them out",
        "understates the pile-up.",
    ]
    fig.text(0.655, 0.80, "\n".join(block), size=SMALL - 2, va="top", family="monospace",
             linespacing=1.5)
    footer(fig, f"The two complete-model rungs at L=8 and L=12 are the paired controls: same "
                f"14-point grid, same 375k-step budget, same eval, same seeds. s* = {S_RHO50} is "
                f"the seed-to-seed spread measured at fixed architecture, which is why (A) is a "
                f"band and not a line, and why a gap smaller than one s* is not a result. "
                f"Commit {commit}.")
    o.savefig(fig)


def page_verdicts(o, cells, e1, e2, st, commit):
    e1_text, _ = call(cells, E1, e1)
    e2_text, _ = call(cells, E2, e2)
    roll = " ".join(arm_rollup(st).split())
    textpage(
        o,
        "The two calls, and the roll-up across both axes",
        "Transcribed from the sbatch headers into scripts/capacity_back.py, "
        "pinned by 15 tests before the cells landed",
        [
            (f"E1 -- {st['E1'].upper()}", NOMLP_C, e1_text),
            (f"E2 -- {st['E2'].upper()}", BAND_C, e2_text),
            (f"ACROSS D2, E1 AND E2 -- the scorer's own roll-up, not the deck's", FULL_C, roll),
        ],
        f"Trainability is checked first and outranks everything: 0 plateau escapes means 'did not "
        f"train in 375k steps' and is NEVER evidence in either direction -- the trap stage C's P2 "
        f"fell into. Then fewer than three seeds, then fewer than two crossing seeds, then a "
        f"median below {BARRIER_LO}, then the tie-break region up to {FREE:.3f}, then the band's "
        f"upper edge at {BARRIER_HI:.2f}. A rung that never got the chance to break the barrier "
        f"reading is not counted as one that held it. Commit {commit}.",
        width=122)


def page_next(o, cells, hub, commit):
    nomlp = summarize(rung(cells, NOMLP_L6))
    l12 = summarize(rung(cells, FULL_L12))
    textpage(
        o,
        "What this does not settle, and what is still running",
        "Stage G (51194118) moves the one knob stage E deliberately held fixed: "
        "the training budget",
        [
            ("THE BUDGET ARM, REGISTERED BEFORE ITS FIRST CELL FINISHED", NOTE_C,
             f"Every rung in this deck ran 50 epochs. Stage G reruns attention-only at d128 and "
             f"d256 for 100, three seeds each, with an epoch-50 reproduction gate: the first 50 "
             f"epochs must match the frozen 50-epoch twins to within 0.01 nats, or the cells are "
             f"not twins and nothing is comparable. All three in-flight cells passed that gate at "
             f"epoch 16-23 with a worst deviation of 0.008. It is the only way a rung written off "
             f"here as 'did not train' can come back."),
            ("AND THE PREDICTION FOR IT IS THAT IT FAILS", DEAD_C,
             f"scripts/escape_timing.py Q3 measured where the 14 late plateau escapes actually "
             f"came from: all 14 crossed from a train NLL of 1.640-1.652, and 13 of the 14 were "
             f"within five epochs of the plateau at their own prior rate. There is no knee. "
             f"Extrapolating the 39 never-escaping cells at their own last-20% slope puts the "
             f"median 87 epochs short, and only 10 of 39 within another 50 -- none of them a "
             f"stage G cell. So doubling the budget is registered to NOT rescue the d128 rung, "
             f"which contradicts stage G's own header, and that contradiction is on record."),
            ("NO IMPROVEMENT IS CLAIMED HERE, AND NONE WAS SOUGHT", FULL_C,
             f"Previous best on this eval, same grid and same budget: the complete model at L=12, "
             f"rho_50 {rho_txt(l12)} / mean_solve {l12['mean_solve']:.4f} at {l12['trainable']:,} "
             f"trainable. Nothing in this deck approaches it, and nothing in it was trying to. "
             f"Since 11-04 the brief is the SMALLEST architecture that still learns direct "
             f"deduction, so the number that matters is the standing minimum: attention-only at "
             f"d256, {nomlp['trainable']:,} trainable, rho_50 {rho_txt(nomlp)} -- "
             f"{hub['trainable'] / nomlp['trainable']:.2f}x fewer parameters than the hub and "
             f"{(nomlp['rho50'] - hub['rho50']) / S_RHO50:.1f} s* worse. Stage F, a 3-seed "
             f"replicate at N=96 of whichever rung survives, is still unwritten on purpose."),
        ],
        f"The resolution limit stands whatever the calls on page 6 say: rung_collapse found no "
        f"DETECTABLE separation between architecture labels inside the band (p = 0.569, positive "
        f"control p = 0.045 on a separation 26x larger), which is not the same claim as 'the "
        f"architectures are equivalent'. Commit {commit}.",
        width=122)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_DEFAULT, SCAN_L6, SCAN_C, SCAN_D, SCAN_E])
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "deptharm_2026-11-05"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--commit", default="(uncommitted)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build even though a stage E rung has fewer than 3 seeds")
    args = ap.parse_args()

    set_style()
    cells = collect(args.scan_dir, args.boot)
    e1_ms, e2_ms = rung(cells, E1), rung(cells, E2)
    if not e1_ms or not e2_ms:
        raise SystemExit("stage E has nothing on disk for E1 or E2 yet")
    hub, e1, e2 = summarize(rung(cells, HUB)), summarize(e1_ms), summarize(e2_ms)
    if (e1["n"] < 3 or e2["n"] < 3) and not args.allow_partial:
        raise SystemExit(
            f"stage E is at E1 {e1['n']}/3, E2 {e2['n']}/3 seeds. The registered rule is "
            f"median-vs-median at three seeds, and a two-seed median on a slide is a number "
            f"people quote. Pass --allow-partial only if that is really what you want.")
    st = {"D2": call(cells, D2, summarize(rung(cells, D2)))[1],
          "E1": call(cells, E1, e1)[1], "E2": call(cells, E2, e2)[1]}
    print(f"{len(cells)} cells | E1 {e1['n']}/3 seeds, E2 {e2['n']}/3 seeds | "
          + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))

    os.makedirs(args.out, exist_ok=True)
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=7)
        page_headline(o, cells, hub, e1, e2, st, len(cells), args.commit)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task every "
                               "cell in this deck is trained and scored on. The attention-only "
                               "models have to produce this with no MLP anywhere in the stack, "
                               "at L=8 and L=12 instead of the hub's L=6.")
        page_ladders(o, cells, args.commit)
        page_band(o, cells, e1, e2, hub, args.commit)
        page_verdicts(o, cells, e1, e2, st, args.commit)
        page_next(o, cells, hub, args.commit)
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
