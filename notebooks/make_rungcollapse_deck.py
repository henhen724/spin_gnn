"""Deck: seven architectures land on one rung, and what that does and does not mean.

The write-up of scripts/rung_collapse.py plus stage C's four registered rows.
Deliberately NOT a write-up of stages D and E, which are still running: the
last page says what each of their outcomes would mean and nothing else.

    python3 -m notebooks.make_rungcollapse_deck --commit $(git rev-parse --short HEAD)

Scan directories are pinned to stages A-D exactly as the cells stood when this
was written. s* is NOT recomputed here -- it is quoted frozen at 0.1484 /
0.2918, the bar every verdict on these pages was scored against.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from scripts.minarch_analysis import (
    SCAN_DEFAULT, SCAN_L6, SCAN_C, SCAN_D, load_all, cell_key, cell_metrics,
)
from scripts.rung_collapse import BAND_LO, BAND_HI, describe, permutation_p, rungs

S_RHO50, S_MEAN = 0.1484, 0.2918
HUB_RHO50, HUB_MEAN, HUB_PARAMS = 0.592, 0.5389, 4899898
BAND_C = "#c1272d"
FULL_C = "#13507a"


def collect(scan_dirs, boot):
    by = defaultdict(list)
    for r in load_all(scan_dirs):
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, boot), path=v[0]["path"]) for k, v in by.items()}
    complete = {k: ms for k, ms in rungs(cells).items() if len(ms) == 3}
    return cells, complete


def _band(desc):
    return [r for r in desc if r["readable"] and BAND_LO <= r["rho50"] <= BAND_HI]


_WORDS = {5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def page_headline(o, desc, band, p_band, p_ctrl, obs_band, obs_ctrl, n_cells):
    params = [r["trainable"] for r in band]
    textpage(
        o,
        f"{_WORDS.get(len(band), len(band))} architectures, one rung",
        f"N=48, degree 5, 4 colours, leak-free (--shuffle-adj --fall) "
        f"| {n_cells} cells, {len(desc)} complete rungs, 3 seeds each",
        [
            ("THE OBSERVATION", BAND_C,
             f"{len(band)} complete rungs sit at rho_50 {min(r['rho50'] for r in band):.3f}"
             f"-{max(r['rho50'] for r in band):.3f}, spanning {min(params):,} to "
             f"{max(params):,} trainable parameters -- a factor of "
             f"{max(params) / min(params):.1f} -- across six different interventions: shallower "
             f"depth, a narrower residual stream, a narrower MLP, a linearised MLP, a frozen "
             f"random MLP, and no MLP at all. Removing the entire MLP and halving the depth put "
             f"the model in the same place."),
            ("IT IS NOT A SQUINT", "#333333",
             f"Across-rung spread is {obs_band:.4f} in mean_solve; the worst ACROSS-SEED spread "
             f"within a single one of these rungs is {S_MEAN:.4f} -- exactly the pooled s* this "
             f"programme froze as its yardstick, because s* was defined as the largest "
             f"across-seed spread in the grid. The pile-up spans one tenth of the bar every "
             f"verdict here is scored against. Shuffle which rung each cell belongs to, 20,000 "
             f"times: p = {p_band:.3f}."),
            ("AND THE HONEST LIMIT, WHICH IS THE POINT OF THE PAGE", "#6a3d9a",
             f"p = {p_band:.3f} is also what a test with no power returns. The same test on the "
             f"full model's depth ladder rejects at p = {p_ctrl:.3f} -- but it took that "
             f"ladder's entire dynamic range, a {obs_ctrl:.3f} separation, to get there at 3 "
             f"seeds, and the band's is {obs_ctrl / obs_band:.0f}x smaller. The claim is NO "
             f"DETECTABLE SEPARATION, resolution limit between {obs_band:.3f} and "
             f"{obs_ctrl:.3f}. Not equivalence; a 3-seed grid cannot support that."),
            ("WHAT IT CANNOT SETTLE, AND WHAT IS RUNNING", "#1a7a3c",
             "(A) BARRIER: 0.76 is a floor any incomplete architecture hits however much "
             "capacity it is given back. (B) CAPACITY: the ablations merely cost about the same "
             "at this budget. The grid cannot decide, for a reason invisible row by row -- every "
             "ablated rung in it was run at AT MOST the hub's depth. Stages D (width) and E "
             "(depth) give one more; last page."),
        ],
        "rho_50 = realized hint density at which solve rate crosses 0.5 on a 14-point x 400-instance "
        "grid, LOWER IS BETTER, reported as >0.95 when it never crosses. mean_solve = unweighted mean "
        "over the 14 bins, which never censors. Every cell trained and evaluated with --shuffle-adj.",
        width=125)


def page_scatter(o, desc, band, commit):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.07, 0.17, 0.60, 0.68])

    ladder = sorted((r for r in desc if r["key"] == (r["key"][0], 4, "full") and len(r["key"]) == 3),
                    key=lambda r: r["key"][0])
    lx = [r["trainable"] for r in ladder if r["readable"]]
    ly = [r["rho50"] for r in ladder if r["readable"]]
    ax.plot(lx, ly, "-o", color=FULL_C, lw=2.5, ms=10, zorder=3,
            label="full model, depth ladder (L=2..12)")
    # Ladder labels go ABOVE the line and ablated numbers BELOW their marker:
    # four of the five ablated rungs sit inside the band at the same height as
    # full L=3 and L=4, and side-by-side text labels overprinted each other.
    for r in ladder:
        if r["readable"]:
            # L=3 and L=4 pass THROUGH the band, so their labels would sit on
            # top of an ablated diamond; push those two out to the left.
            inband = BAND_LO <= r["rho50"] <= BAND_HI
            ax.annotate(f"L={r['key'][0]}", (r["trainable"], r["rho50"]),
                        textcoords="offset points",
                        xytext=(-16, 2) if inband else (0, 13),
                        ha="right" if inband else "center",
                        size=SMALL, color=FULL_C)

    ax.axhspan(BAND_LO, BAND_HI, color=BAND_C, alpha=0.08, zorder=0)
    ax.axhline(min(r["rho50"] for r in band), color=BAND_C, lw=1, ls=":", zorder=1)
    ax.axhline(max(r["rho50"] for r in band), color=BAND_C, lw=1, ls=":", zorder=1)

    ablations = [r for r in band if not (len(r["key"]) == 3 and r["key"][2] == "full")]
    ax.scatter([r["trainable"] for r in ablations], [r["rho50"] for r in ablations],
               s=190, marker="D", color=BAND_C, zorder=4, label="ablated, 3 seeds complete")
    abl = sorted(ablations, key=lambda r: r["trainable"])
    for i, r in enumerate(abl, 1):
        # rf_mlp L=4 and linear L=4 differ by 512 parameters out of 3.32M, so
        # their markers are the same point on a log axis; stagger the numbers.
        ax.annotate(str(i), (r["trainable"], r["rho50"]), textcoords="offset points",
                    xytext=(0, -27 if i % 2 else -43), ha="center",
                    size=SMALL, weight="bold", color=BAND_C)

    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("rho_50     (lower is better)")
    ax.set_title("Capacity buys deduction only for the complete model", size=MED + 2)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95)

    lines = [
        "THE ABLATED RUNGS, numbered by size",
        "     rho_50   trainable",
    ]
    for i, r in enumerate(abl, 1):
        lines.append(f" {i}   {r['rho50']:.3f}  {r['trainable']:>9,}  "
                     + r["label"].replace("h=4 ", "").replace("L=", "L").strip())
    lines += [
        "",
        f"band width {max(r['rho50'] for r in band) - min(r['rho50'] for r in band):.4f}"
        f" in rho_50, over",
        f"a {max(r['trainable'] for r in band) / min(r['trainable'] for r in band):.1f}x"
        f" range in parameters",
        "",
        "THE FULL MODEL passes straight",
        "through the band at L=3-4 and keeps",
        "descending: 0.592 at L=6, 0.471 at",
        "L=8, 0.385 at L=12.",
        "",
        "THE ABLATED MODELS stop there, incl.",
        "ones with MORE parameters than full",
        "L=3. And no_mlp d256 has 1,746,490",
        "and SOLVES at 0.763, while full d128",
        "has 2,059,834 -- 18% MORE -- and",
        "never crosses 0.5 at any of 14",
        "densities on any of 3 seeds.",
    ]
    fig.text(0.695, 0.84, "\n".join(lines), size=SMALL - 1, va="top", family="monospace",
             linespacing=1.5)
    footer(fig, f"Censored rungs (rho_50 never crosses 0.5) cannot be placed on this axis and are "
                f"omitted: full d128, no_mlp d128, rf_mlp L=4, rf_all, nhead=1 at both depths, "
                f"full L=1, full d64. All of them sit BELOW the band, so omitting them understates "
                f"the collapse rather than flattering it. Commit {commit}.")
    o.savefig(fig)


def page_stats(o, band, desc, obs_band, p_band, obs_ctrl, p_ctrl):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.93, "The same test, run twice: on the band and on something known to differ",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.875, "Permutation test on mean_solve: shuffle which rung each cell belongs "
                         "to, keep group sizes, recompute the spread of rung medians",
             ha="center", size=MED - 2, color="0.3")

    ax = fig.add_axes([0.155, 0.30, 0.335, 0.48])
    rows = sorted(band, key=lambda r: r["mean_solve"])
    y = range(len(rows))
    for i, r in zip(y, rows):
        vals = r["_cells"]
        ax.plot(vals, [i] * len(vals), "o", color="0.65", ms=9, zorder=2)
        ax.plot([r["mean_solve"]], [i], "D", color=BAND_C, ms=13, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["label"].replace("h=4 ", "").strip() for r in rows], size=SMALL)
    ax.set_xlabel("mean_solve   (grey = each seed, red = rung median)", size=MED - 3)
    ax.set_title("Seeds overlap; medians do not separate", size=MED)
    ax.grid(alpha=0.25, axis="x")

    ax2 = fig.add_axes([0.605, 0.30, 0.345, 0.48])
    labels = ["observed\nband", "worst\nwithin-rung\nseed spread", "observed\nfull-model\nladder"]
    vals = [obs_band, S_MEAN, obs_ctrl]
    cols = [BAND_C, "0.55", FULL_C]
    ax2.bar(labels, vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.015, f"{v:.4f}", ha="center", size=MED - 3, weight="bold")
    ax2.set_ylabel("spread of rung medians, mean_solve")
    ax2.set_ylim(0, max(vals) * 1.18)
    ax2.set_title("Scale: the band against its own noise floor", size=MED)
    ax2.grid(alpha=0.25, axis="y")

    txt = (f"band: p = {p_band:.3f}  (not rejected)        "
           f"full-model depth ladder: p = {p_ctrl:.3f}  (rejected)")
    fig.text(0.5, 0.155, txt, ha="center", size=MED - 1, weight="bold")
    footer(fig, f"The positive control is not decoration. p = {p_band:.3f} on the band is also what a "
                f"test with no power returns, so the identical test runs on rungs the whole programme "
                f"treats as real. It rejects -- but only at p = {p_ctrl:.3f}, and only on a "
                f"{obs_ctrl:.3f} separation, {obs_ctrl / obs_band:.0f}x the band's. The honest reading "
                f"is a resolution limit, not an equivalence. 20,000 shuffles, RNG seed 0.")
    o.savefig(fig)


def page_stagec(o, commit):
    textpage(
        o,
        "Stage C: four registered rows, scored against a bar frozen before they ran",
        "51114734, 13 of 15 cells done. Paired control = stage A's L=6 d256 ff1024 h4 "
        "full: rho_50 0.592, mean_solve 0.5389",
        [
            ("P1 FALSIFIED -- the residual stream width is load-bearing, and this one is clean",
             BAND_C,
             "full at d_model 128 is censored 3/3 at every one of 14 hint densities, mean_solve "
             "0.0273 [spread 0.0036] at 2,059,834 trainable. 2 of 3 seeds got BELOW the 1.64 "
             "plateau, so unlike every other decisive failure in this grid the trainability "
             "caveat does not touch it. Findings 13 and 26 said depth beats width; at N=48 they "
             "are wrong about width. Removing the ENTIRE MLP at d256 costs 0.235 of mean_solve; "
             "halving the stream costs 0.512."),
            ("P2 UNSCOREABLE AS REGISTERED, and recorded as that rather than as a tie", "#333333",
             "Its quantity is a difference of differences and both d128 terms are censored, so "
             "it does not exist. The mean_solve fallback reads 0.0291 against 0.0273 -- two "
             "cells on the floor, the 'both cells failed' trap -- and all three no_mlp d128 "
             "seeds are above the plateau too. d128 is below the width this task needs, and "
             "nothing about the MLP can be read at a width where the full model fails as well."),
            ("P3 PASSES THE REGISTERED RULE BY 0.0004 AND NOTHING MORE", "#6a3d9a",
             "ff 1024->256: 3/3 crossed, rho_50 median 0.740, where the free/not-free threshold "
             "is 0.740. That is a tie with the bar, not a demonstration that quartering ff is "
             "free. Read mean_solve, where there is no boundary: 0.3320 against 0.5389 is "
             "-0.2070, essentially what removing the entire MLP costs (-0.2350). A 4x narrower "
             "MLP is worth about what no MLP is worth -- so the 'all' in 'the MLP is "
             "all-or-nothing' is its WIDTH."),
            ("P4 IS A RESULT, NOT A FAILED CELL -- and it closes stage B's open question",
             "#1a7a3c",
             "nhead=1 with warmup 5 escapes the plateau 0/3 (1.648/1.648/1.647), mean_solve "
             "0.0257 [0.0048] -- matching stage B's N5 at warmup 0 (also 0/3, "
             "1.657/1.648/1.650, 0.0270). Two independent optimizer settings failing the same "
             "way is enough to write N5's 'WORSE' as a property of having ONE HEAD rather than "
             "of the schedule, which is precisely what stage B could not say. P5 (d64) is at "
             "1 of 3 seeds: censored, mean_solve 0.0011."),
        ],
        f"'Free' means rho_50 stays at or below 0.592 + s* = 0.740, with s* = 0.1484 frozen before "
        f"stage C ran and not recomputed since. Stage C's rungs carry a non-default d_model, ff or "
        f"warmup, so they keep their own keys and stay OUT of the pooled bar stages A and B were "
        f"scored against. Commit {commit}.",
        width=125)


def page_next(o, commit):
    textpage(
        o,
        "What is running, and what each outcome would mean",
        "Both jobs hand an ablated model MORE of a resource than the hub it is "
        "compared against -- which no cell in the grid has done",
        [
            ("STAGE D, 51151957, THE WIDTH ARM -- 2 of 6 cells complete", FULL_C,
             "D1 (full at d192, 3,381,562 trainable) has two seeds at rho_50 0.759 and 0.752, "
             "mean_solve 0.3138 and 0.3266, both escaping the plateau. Its registered "
             "mean_solve window [0.10, 0.45] holds, but its 'lands strictly between its "
             "neighbours' reading does not: d192 landed IN the band with everything else. "
             "Bisecting the width cliff produced another member of the pile-up. D2 "
             "(attention-only at d384, 3,799,354 trainable, 22% under the hub) has no "
             "completed cell yet."),
            ("STAGE E, 51171824, THE DEPTH ARM -- submitted today, 3 of 6 running", "#1a7a3c",
             "Attention-only at L=8 (2,274,874 trainable) and L=12 (3,331,642), three seeds "
             "each, everything else at the hub's settings. L=8 and L=12 were chosen because the "
             "full model is already measured at both, so each rung has a paired partner at "
             "identical depth and identical budget. Registered falsifier of the barrier reading: "
             "either rung with a median rho_50 below 0.740 on at least 2 of 3 crossing seeds. "
             "No partial credit, and the 0.740 tie-break is written down in advance this time."),
            ("THE PREDICTION THIS DECK IS ON RECORD FOR, MADE BEFORE D2 FINISHED", "#6a3d9a",
             "Stage E's header backs D2's FALSIFIER over D2's own prediction: under the barrier "
             "reading, width is not the currency either, so d384 attention-only should land "
             "beside d256 attention-only at 0.76 rather than reaching the hub. If D2 and both E "
             "rungs all stay at 0.76, the barrier is demonstrated on two independent axes. If "
             "D2 clears it while E does not, the currency is width specifically -- a sharper "
             "result than either job registered."),
            ("THE WAY THIS PRODUCES NOTHING, NAMED IN ADVANCE", BAND_C,
             "Trainability, not capacity. Attention-only escapes the 1.64 plateau 2/3 at L=6 and "
             "0/3 at L=4, so a DEEPER stack failing to escape is the EXPECTED failure mode and "
             "gets written as 'did not train in 375k steps', never as evidence for a barrier "
             "-- the trap P2 fell into. The budget is deliberately not raised, since a rung on "
             "a different budget cannot be read against this ladder."),
        ],
        f"Previous best on this eval, same 14-point grid and same budget: the hub at rho_50 0.592 / "
        f"mean_solve 0.5389 and 4,899,898 trainable, and below it only full L=8 (0.471) and L=12 "
        f"(0.385) at 6.5M and 9.6M. Nothing in this deck beats it, and nothing was expected to -- the "
        f"brief is the smallest architecture that still learns direct deduction, so the standing "
        f"minimum remains attention-only at d256, 1,746,490 trainable, rho_50 0.763. "
        f"Commit {commit}.",
        width=125)


def main():
    ap = argparse.ArgumentParser()
    # Pinned to stages A-D as they stood. Stage E is deliberately absent: its
    # rungs carry default hyperparameters and would pool into the s* this deck
    # quotes frozen. When E lands it gets its own deck, not a silent edit here.
    ap.add_argument("--scan-dir", nargs="+",
                    default=[SCAN_DEFAULT, SCAN_L6, SCAN_C, SCAN_D])
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "rungcollapse_2026-11-04"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--commit", default="(uncommitted)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_style()
    cells, complete = collect(args.scan_dir, args.boot)
    desc = sorted((describe(k, ms) for k, ms in complete.items()),
                  key=lambda r: (not r["readable"], r["rho50"]))
    for r in desc:
        r["_cells"] = [m["mean_solve"] for m in complete[r["key"]]]
    band = _band(desc)
    print(f"{len(cells)} cells, {len(complete)} complete rungs, {len(band)} in band")

    obs_band, p_band, _ = permutation_p({r["label"]: r["_cells"] for r in band})
    ladder = {f"L={L}": [m["mean_solve"] for m in complete[(L, 4, "full")]]
              for L in (12, 8, 6, 4, 3, 2) if (L, 4, "full") in complete}
    obs_ctrl, p_ctrl, _ = permutation_p(ladder)
    print(f"band p={p_band:.3f} obs={obs_band:.4f} | control p={p_ctrl:.3f} obs={obs_ctrl:.4f}")

    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=7)
        page_headline(o, desc, band, p_band, p_ctrl, obs_band, obs_ctrl, len(cells))
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task every "
                               "cell in this deck is trained and scored on. Every architecture "
                               "on page 1 has to produce this token by token, with no scratchpad "
                               "and no tool. rho_50 asks how few hints it can be given and still "
                               "finish.")
        page_scatter(o, desc, band, args.commit)
        page_stats(o, band, desc, obs_band, p_band, obs_ctrl, p_ctrl)
        page_stagec(o, args.commit)
        page_next(o, args.commit)
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
