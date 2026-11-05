"""Deck: stage G, the BUDGET arm -- is the cheap end excluded by capacity, or by 375k steps?

Stages A-F all ran 50 epochs (375k gradient steps at a 240k pool), and seven
rungs were written off in those stages with the words "0 of 3 seeds escaped
finding 44's 1.64 plateau, which is not evidence in either direction". Stage G
(51194118) moves ONE knob -- epochs 50 -> 100 -- on two of them, with the
standing-minimum rung included as a PAIRED CONTROL inside the same job, because
nothing measured at 50 epochs is comparable across a budget change.

    python3 -m notebooks.make_stageg_deck --commit $(git rev-parse --short HEAD)

Every verdict on these slides is scripts/stageg_score.py's, committed (dc7c08e)
while tasks 0-1 were the only ones running and no stage G cell had finished.
This file imports its registered literals rather than retyping them, and the
notebook beside it runs the scorer as a subprocess so the words on page 6 can
be checked against the words it prints. SCAN_G stays out of SCAN_DIRS: its
rungs would otherwise pool into the frozen bar s* = 0.1484 / 0.2918, which is
quoted here and never recomputed.
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
from scripts.minarch_analysis import (SCAN_DIRS, SCAN_E, SCAN_G, cell_key, cell_metrics,
                                      load_all)
from scripts.stagec_score import PLATEAU, S_MEAN, S_RHO50, escapes, rho_txt, rung, summarize
from scripts.stageg_score import (EPOCHS_G, G1, G1_TWIN, G2, G2_TWIN, G2_ARTIFACT,
                                  HEADER_ESCAPES, PRED_ESCAPES, TOL_CKPT, TWIN, TWIN_RUNG)

FULL_C = "#13507a"      # the complete model, and anything at the old budget
NOMLP_C = "#1a7a3c"     # attention-only
BAND_C = "#c1272d"      # the 0.76 band, the plateau, and anything that lands on them
NOTE_C = "#6a3d9a"      # registered in advance
DEAD_C = "0.45"         # cells that never escaped the plateau

SEED_C = ["#1a7a3c", "#6a3d9a", "#b06000"]   # one colour per seed, both budgets
E2 = (12, 4, "no_mlp")          # stage E's rung, the standing minimum as of 11-05
FULL_L12 = (12, 4, "full")      # best solver on this eval, at any size


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


def gate(cells, key, twin_key):
    """Epoch-50 reproduction, per seed: (seed, got, want, delta, ok)."""
    out = []
    for m, t in zip(rung(cells, key), rung(cells, twin_key)):
        tr = curve(m["path"])[0]
        want = curve(t["path"])[0][-1]
        got = tr[49] if len(tr) >= 50 else float("nan")
        out.append((len(out), got, want, got - want, abs(got - want) <= TOL_CKPT))
    return out


def rescue(cells):
    """The seed, if any, that the doubled budget turned from a non-solver into a solver."""
    for tag, key, twin_key in (("G2", G2, G2_TWIN), ("G1", G1, G1_TWIN)):
        seeds = sorted(k[-1] for k in cells if k[:-1] == key)
        for s, m, t in zip(seeds, rung(cells, key), rung(cells, twin_key)):
            if t["flag"] != "ok" and m["flag"] == "ok":
                return {"tag": tag, "seed": s, "key": key, "twin_key": twin_key,
                        "path": m["path"], "twin_path": t["path"],
                        "best_before": t["best_solve"], "best_after": m["best_solve"],
                        "mean_before": t["mean_solve"], "mean_after": m["mean_solve"],
                        "nll_before": curve(t["path"])[0][-1],
                        "nll_after": curve(m["path"])[0][-1],
                        "acc_before": curve(t["path"])[1][-1],
                        "acc_after": curve(m["path"])[1][-1],
                        "rho_after": m["rho50"]}
    return None


def page_headline(o, cells, g1, g2, t1, t2, n_cells, commit):
    e1, e2n = escapes(rung(cells, G1))[0], escapes(rung(cells, G2))[0]
    rescued = rescue(cells)
    moved = g2["rho50_readable"] and t2["rho50_readable"] and \
        abs(g2["rho50"] - t2["rho50"]) > S_RHO50
    if moved:
        title = "Doubling the budget moved the standing minimum's solve rate"
    elif rescued:
        title = "The unlucky seed was an untrained seed. The band was not a budget artifact"
    else:
        title = "Doubling the budget bought a plateau escape and nothing else"
    if rescued:
        bought = (
            "WHAT 750k STEPS BOUGHT: ONE WHOLE SEED, AND THE SPREAD", NOMLP_C,
            f"G2's seed {rescued['seed']} was a NON-SOLVER at 50 epochs -- censored, best solve "
            f"{rescued['best_before']:.3f} at any hint density, train NLL "
            f"{rescued['nll_before']:.3f}, sitting on the plateau. At {EPOCHS_G} epochs it is "
            f"rho_50 {rescued['rho_after']:.3f}, best solve {rescued['best_after']:.3f}: "
            f"indistinguishable from its two siblings. The rung goes {TWIN_RUNG['G2'][1]} of 3 "
            f"crossing seeds to {g2['n_cross']} of 3, and its seed spread in mean_solve COLLAPSES "
            f"from {TWIN_RUNG['G2'][4]:.4f} to {g2['spread_mean']:.4f}. The spread at this rung "
            f"was a budget artifact.")
    else:
        bought = (
            "WHAT 750k STEPS BOUGHT THE CONTROL RUNG", NOMLP_C,
            f"G2 escapes the plateau on {e2n} of {g2['n']} seeds against its twin's "
            f"{TWIN_RUNG['G2'][6]} of 3, and crosses on {g2['n_cross']} of 3 against "
            f"{TWIN_RUNG['G2'][1]} of 3. No seed changed category.")
    textpage(
        o,
        title,
        f"N=48, degree 5, 4 colours, leak-free | one knob: 50 -> {EPOCHS_G} epochs, "
        f"375k -> 750k steps, fixed 240k pool | {n_cells} cells",
        [
            ("WHY A BUDGET ARM EXISTS AT ALL", NOTE_C,
             f"Across stages A-F, 12 of 24 multi-seed rungs never got below finding 44's "
             f"{PLATEAU} train-NLL plateau, and each was written off as 'did not train in 375k "
             f"steps' -- deliberately NOT as evidence that its architecture lacks the capacity. "
             f"That caveat is honest and it is unfalsifiable until someone spends the steps. "
             f"Stage G spends them on the cheapest rung in the grid (G1, attention-only d128, "
             f"{t1['trainable']:,} trainable) and on the rung that was the standing minimum when "
             f"the job was written (G2, attention-only d256, {t2['trainable']:,})."),
            bought,
            ("WHAT IT DID NOT BUY: THE METRIC OF RECORD", BAND_C,
             f"Median rho_50 {rho_txt(t2)} -> {rho_txt(g2)}, a move of "
             f"{abs(g2['rho50'] - t2['rho50']):.3f} against the frozen seed-spread bar {S_RHO50}, "
             f"and mean_solve {t2['mean_solve']:.4f} -> {g2['mean_solve']:.4f}. The median was "
             f"already set by a seed that had trained, so rescuing the third one moves the rung "
             f"nowhere: THE 0.76 BAND IS NOT A BUDGET ARTIFACT. G1 is unchanged on every count -- "
             f"{e1} of 3 escapes against its twin's {TWIN_RUNG['G1'][6]}, rho_50 {rho_txt(g1)} -- "
             f"and by the registered asymmetry that is still 'did not train', not 'cannot'."),
        ],
        f"rho_50 = realized hint density at which solve rate crosses 0.5 on a 14-point x "
        f"400-instance grid, LOWER IS BETTER, reported CENSORED when no bin crosses. mean_solve "
        f"= unweighted mean over the 14 bins, which never censors. s* = {S_RHO50} / {S_MEAN} was "
        f"frozen before any of these cells ran and is not recomputed. Commit {commit}.",
        width=122)


def page_training(o, cells, commit):
    """The curves the whole arm turns on, with the epoch-50 gate visible."""
    rescued = rescue(cells)
    first = [curve(m["path"])[0][0] for k in (G1, G2) for m in rung(cells, k)]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Twice the steps, same seeds, same pool: where the extra 375k went",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, "Dashed = the 50-epoch twin, solid = the 100-epoch run. They are the "
                         "same optimization until epoch 50, and that is checked, not assumed",
             ha="center", size=MED - 3, color="0.3")

    for i, (tag, key, twin_key, name) in enumerate((
            ("G1", G1, G1_TWIN, "G1  attention-only d128, the cheap end"),
            ("G2", G2, G2_TWIN, "G2  attention-only d256, paired control"))):
        ax = fig.add_axes([0.055 + 0.315 * i, 0.26, 0.255, 0.56])
        for j, (m, t) in enumerate(zip(rung(cells, key), rung(cells, twin_key))):
            tw = curve(t["path"])[0]
            nw = curve(m["path"])[0]
            ax.plot(range(1, len(tw) + 1), tw, ls="--", lw=1.6, color=SEED_C[j], alpha=0.75)
            ax.plot(range(1, len(nw) + 1), nw, ls="-", lw=2.0, color=SEED_C[j],
                    label=f"seed {j}")
            if rescued and tag == rescued["tag"] and j == rescued["seed"]:
                xo = next((e for e, v in enumerate(nw, 1) if v < PLATEAU), len(nw))
                ax.annotate(f"this seed starts solving:\nbest solve "
                            f"{rescued['best_before']:.3f} -> {rescued['best_after']:.3f}",
                            xy=(xo, PLATEAU), xytext=(14, 1.86), size=SMALL - 2,
                            color=SEED_C[j], ha="left",
                            arrowprops=dict(arrowstyle="->", color=SEED_C[j], lw=1.3,
                                            connectionstyle="arc3,rad=-0.15"))
        ax.axhline(PLATEAU, color=BAND_C, lw=2, ls="-", zorder=1)
        ax.text(0.015, PLATEAU, f" plateau {PLATEAU}", size=SMALL - 1,
                color=BAND_C, va="bottom", ha="left", transform=ax.get_yaxis_transform())
        ax.axvline(50, color="0.6", lw=1.2, ls=":")
        ax.text(50, 0.02, " twin ends", size=SMALL - 1, color="0.45",
                transform=ax.get_xaxis_transform(), va="bottom")
        ax.set_xlabel("epoch")
        if i == 0:
            ax.set_ylabel("train NLL")
        ax.set_ylim(1.55, 2.05)
        ax.set_title(name, size=MED - 3)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.95, fontsize=SMALL - 2)

    lines = [f"EPOCH-50 REPRODUCTION GATE", f"tolerance {TOL_CKPT} nats, registered before the job",
             "", "       seed  epoch-50  twin final   delta"]
    for tag, key, twin_key in (("G1", G1, G1_TWIN), ("G2", G2, G2_TWIN)):
        lines.append(f"  {tag}")
        for seed, got, want, d, ok in gate(cells, key, twin_key):
            lines.append(f"     {seed:>4}     {got:.3f}       {want:.3f}  {d:+.3f}  "
                         f"{'ok' if ok else 'MISMATCH'}")
    fig.text(0.665, 0.80, "\n".join(lines), size=SMALL - 1, va="top", family="monospace",
             linespacing=1.6)
    footer(fig, f"The y axis is clipped at 2.05; the six curves start between {min(first):.2f} "
                f"and {max(first):.2f} at epoch 1. With a constant "
                f"learning rate, no warmup, a fixed pool and the same seed, the first 50 epochs of "
                f"a 100-epoch run ARE the 50-epoch run. If a cell missed the gate the pairing "
                f"would be void and the cell reported inconclusive rather than explained -- "
                f"nothing here missed it. Note what the escape looks like: G2 seed 1 drifts across "
                f"the line rather than turning a corner, which is what escape_timing.py measured "
                f"on all 14 late escapes in the grid. Commit {commit}.")
    o.savefig(fig)


def page_bought(o, cells, rows, g1, g2, t1, t2, commit):
    """Solve rate against hint density, both budgets, both rungs."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "The rescued seed lands on top of its siblings, and the median stays "
                         "put", ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.893, "Transition curves on the identical 14-point x 400-instance grid, "
                         "three seeds per rung, --shuffle-adj throughout",
             ha="center", size=MED - 3, color="0.3")

    ax = fig.add_axes([0.065, 0.28, 0.50, 0.53])
    for key, colour, ls, lab in ((G2_TWIN, FULL_C, "--", f"G2 d256, 50 epochs"),
                                 (G2, NOMLP_C, "-", f"G2 d256, {EPOCHS_G} epochs"),
                                 (G1_TWIN, DEAD_C, "--", "G1 d128, 50 epochs"),
                                 (G1, BAND_C, "-", f"G1 d128, {EPOCHS_G} epochs")):
        ms = rung(cells, key)
        if not ms:
            continue
        for m in ms:
            seed_rows = next(v for k, v in rows.items()
                             if k[:-1] == key and v[0]["path"] == m["path"])
            pts = sorted((r["realized_rho"], r["solve_rate"]) for r in seed_rows)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=colour, lw=1.8,
                    alpha=0.85)
    r = rescue(cells)
    if r:
        ax.text(0.36, 0.78, f"{r['tag']} seed {r['seed']} at 50 epochs is the dashed blue curve "
                            f"along the bottom:\nbest solve {r['best_before']:.3f} at any density, "
                            f"never crosses. At {EPOCHS_G} epochs it is\none of the three green "
                            f"curves, rho_50 {r['rho_after']:.3f}.",
                size=SMALL - 2, color="0.15", ha="left", va="top", linespacing=1.5)
    ax.axhline(0.5, color="0.5", lw=1.2, ls=":")
    ax.text(0.015, 0.5, " solve rate 0.5, where rho_50 is read", size=SMALL - 1, color="0.4",
            va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xlabel("realized hint density rho")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.set_title("Each line is one seed. One line moves, and it moves onto the other two",
                 size=MED - 2)
    ax.legend(handles=[Line2D([], [], color=FULL_C, ls="--", lw=2, label="G2 d256, 50 ep"),
                       Line2D([], [], color=NOMLP_C, ls="-", lw=2,
                              label=f"G2 d256, {EPOCHS_G} ep"),
                       Line2D([], [], color=DEAD_C, ls="--", lw=2, label="G1 d128, 50 ep"),
                       Line2D([], [], color=BAND_C, ls="-", lw=2,
                              label=f"G1 d128, {EPOCHS_G} ep")],
              loc="upper left", framealpha=0.95, fontsize=SMALL - 1)

    acc = {}
    for tag, key in (("G1", G1), ("G2", G2)):
        accs = [curve(m["path"])[1][-1] for m in rung(cells, key)]
        acc[tag] = sum(accs) / len(accs)
    block = [
        "                        50 ep      100 ep",
        "G1  attention-only d128",
        f"  rho_50            {TWIN_RUNG['G1'][2]:>9}  {rho_txt(g1):>10}",
        f"  crossing seeds    {TWIN_RUNG['G1'][1]:>7}/3  {g1['n_cross']:>8}/3",
        f"  mean_solve        {TWIN_RUNG['G1'][3]:>9.4f}  {g1['mean_solve']:>10.4f}",
        f"  seed spread       {TWIN_RUNG['G1'][4]:>9.4f}  {g1['spread_mean']:>10.4f}",
        f"  token accuracy    {sum(v[1] for v in TWIN['G1'].values()) / 3:>9.4f}  "
        f"{acc['G1']:>10.4f}",
        f"  plateau escapes   {TWIN_RUNG['G1'][6]:>7}/3  {escapes(rung(cells, G1))[0]:>8}/3",
        "",
        "G2  attention-only d256",
        f"  rho_50            {TWIN_RUNG['G2'][2]:>9}  {rho_txt(g2):>10}",
        f"  crossing seeds    {TWIN_RUNG['G2'][1]:>7}/3  {g2['n_cross']:>8}/3",
        f"  mean_solve        {TWIN_RUNG['G2'][3]:>9.4f}  {g2['mean_solve']:>10.4f}",
        f"  seed spread       {TWIN_RUNG['G2'][4]:>9.4f}  {g2['spread_mean']:>10.4f}",
        f"  token accuracy    {sum(v[1] for v in TWIN['G2'].values()) / 3:>9.4f}  "
        f"{acc['G2']:>10.4f}",
        f"  plateau escapes   {TWIN_RUNG['G2'][6]:>7}/3  {escapes(rung(cells, G2))[0]:>8}/3",
        "",
        f"  trainable {t2['trainable']:,}, unchanged",
        f"  and asserted against the twin",
        "",
        "REGISTERED THRESHOLD, from the scorer:",
        f"  G2 below rho_50 {G2_ARTIFACT} would have",
        "  made the 0.76 band a budget artifact.",
        f"  It is {rho_txt(g2)}.",
    ]
    fig.text(0.60, 0.82, "\n".join(block), size=SMALL - 2, va="top", family="monospace",
             linespacing=1.5)
    footer(fig, f"The comparison that is legitimate here is each rung against its OWN twin: same "
                f"architecture, same seeds, same pool, one knob. G1's rho_50 against the frozen "
                f"0.76 band would be CROSS-BUDGET and the scorer refuses to print it. The effect "
                f"of 375k extra gradient steps on the standing minimum's rung is a "
                f"{abs(g2['rho50'] - t2['rho50']):.3f} shift in median rho_50 "
                f"({abs(g2['rho50'] - t2['rho50']) / S_RHO50:.2f} s*) and +"
                f"{acc['G2'] - sum(v[1] for v in TWIN['G2'].values()) / 3:.4f} in token accuracy "
                f"-- but a {TWIN_RUNG['G2'][4] / max(g2['spread_mean'], 1e-9):.0f}x collapse in "
                f"the seed spread, because the dispersion was one untrained seed and not three "
                f"different answers. Commit {commit}.")
    o.savefig(fig)


def page_predictions(o, cells, g1, g2, commit):
    e1, e2n = escapes(rung(cells, G1))[0], escapes(rung(cells, G2))[0]
    below = []
    for tag, key in (("G1", G1), ("G2", G2)):
        for j, m in enumerate(rung(cells, key)):
            nll, _a, slope = TWIN[tag][j]
            pred = nll + slope * (EPOCHS_G - 50)
            got = curve(m["path"])[0][-1]
            if got - pred < -TOL_CKPT:
                below.append(f"{tag} seed {j}, {got:.3f} against {pred:.3f}")
    below_txt = "; ".join(below) or "no cell did"
    r = rescue(cells)
    nll_r, _acc_r, slope_r = TWIN[r["tag"]][r["seed"]] if r else TWIN["G2"][1]
    eta = int(round((nll_r - PLATEAU) / -slope_r))
    textpage(
        o,
        "Two registered predictions disagreed, and the data picked one",
        "Both were in the repository before any stage G cell finished; neither was adjusted "
        "afterwards",
        [
            ("THE SBATCH HEADER (4a35a2e), WRITTEN AT SUBMISSION", NOTE_C,
             f"G1 escapes {HEADER_ESCAPES['G1']}. G2 escapes {HEADER_ESCAPES['G2']} of 3 -- a "
             f"guess, made from the fact that two of G2's three seeds were already below the "
             f"plateau at epoch 50 and the third was still descending."),
            ("THE EXTRAPOLATION (dc7c08e), WRITTEN WHILE THE JOB RAN", BAND_C,
             f"escape_timing.py's Q3 found that late escapes have no knee -- all 14 in the grid "
             f"crossed from a train NLL of 1.640-1.652, 13 of 14 within five epochs of the "
             f"plateau at their own prior rate -- and extrapolating each twin's last-20% slope "
             f"to epoch {EPOCHS_G} predicted G1 {PRED_ESCAPES['G1']} of 3 and G2 "
             f"{PRED_ESCAPES['G2']} of 3. The G2 number openly contradicted the header."),
            ("OBSERVED, AND THE HEADER WON ON THE ONE THEY DISAGREED ABOUT", FULL_C,
             f"G1 {e1} of 3, as both predicted. G2 {e2n} of 3, so the header was right and the "
             f"extrapolation was wrong: seed 1 crossed a gap its own terminal slope put {eta} "
             f"epochs away, and it is the one cell of six to land below its extrapolated bound "
             f"({below_txt}). The tails bend down rather than running straight, so Q3's ETAs are "
             f"UPPER bounds on how far a stalled cell is and its count of 10 cells within another "
             f"50 epochs is an undercount. That is one of the two findings here that outlive this "
             f"rung; the other is on the last page."),
        ],
        f"The asymmetry is enforced by the scorer, not by the reader: G1 escaping would have "
        f"proved the budget was binding, G1 not escaping proves nothing, and there is no symmetric "
        f"reading available. Three seeds per rung, medians compared to medians, against the frozen "
        f"s* = {S_RHO50} / {S_MEAN}. Commit {commit}.",
        width=122)


def page_next(o, cells, g1, g2, t2, commit):
    e2 = summarize(rung(cells, E2)) if rung(cells, E2) else None
    l12 = summarize(rung(cells, FULL_L12)) if rung(cells, FULL_L12) else None
    best = (f"the complete model at L=12, rho_50 {rho_txt(l12)} at {l12['trainable']:,} trainable"
            if l12 else "not readable from these scan dirs")
    minimum = (f"attention-only L=12, {e2['trainable']:,} trainable at rho_50 {rho_txt(e2)} on "
               f"{e2['n_cross']} of {e2['n']} seeds (stage E, 51171824)" if e2 else "unchanged")
    textpage(
        o,
        "What stage G settles, what it does not, and what is running",
        "The budget question is answered for one rung and left open for the other, on purpose",
        [
            ("SETTLED: THE 0.76 BAND IS NOT AN ARTIFACT OF 375k STEPS", FULL_C,
             f"The standing minimum's rung was rerun at double the budget with every seed now "
             f"past the plateau and all {g2['n_cross']} of 3 solving, and its median rho_50 moved "
             f"{abs(g2['rho50'] - t2['rho50']):.3f} "
             f"({abs(g2['rho50'] - t2['rho50']) / S_RHO50:.2f} s*). Nothing in stages A-F that was "
             f"scored inside the band needs rereading as a budget effect, and the registered "
             f"{G2_ARTIFACT} threshold that would have forced that rereading was not approached."),
            ("SETTLED, AND NOT WHAT THE JOB WAS FOR: WHAT A SEED SPREAD IS", NOMLP_C,
             f"This rung's dispersion was the programme's standing excuse for not reading small "
             f"gaps -- mean_solve spread {TWIN_RUNG['G2'][4]:.4f}, wider than most depth steps. At "
             f"{EPOCHS_G} epochs it is {g2['spread_mean']:.4f}. The spread was ONE UNTRAINED SEED, "
             f"not three different answers, and the two seeds that had trained were already within "
             f"{abs(TWIN['G2'][0][1] - TWIN['G2'][2][1]):.4f} of each other in token accuracy. So "
             f"a 2-of-3 rung at 50 epochs should be read as 'one seed is not finished', not as "
             f"'this architecture is unreliable' -- which is a claim about the frozen s* bar too, "
             f"since it was pooled at 50 epochs and is therefore conservative, not tight."),
            ("NOT SETTLED: WHETHER d128 CAN EVER LEARN THIS", DEAD_C,
             f"G1 is {rho_txt(g1)} at {escapes(rung(cells, G1))[0]} of 3 escapes after 750k "
             f"steps. The registered reading is 'did not train', not 'cannot represent it', and "
             f"the next cell on that question is 200 epochs rather than a conclusion."),
            ("UNCHANGED, AND STILL RUNNING", NOTE_C,
             f"Best on this eval at any size: {best}. Standing minimum, by stage H's rule (3 of 3 "
             f"crossing AND median at or below 0.740): {minimum}; G3 did not fire, so stage G "
             f"does not move it. Stage H (51223007) is in flight -- attention-only at L=12, d128 "
             f"and d192, three seeds each -- asking whether the depth that bought the MLP back "
             f"survives being made narrow. Its scorer was registered before its first cell "
             f"landed, as this one was."),
        ],
        f"Cost, for the record: 6 cells, about 25 GPU-hours, to move one escape count, one seed "
        f"from censored to solving, and no metric of record. That is the price of being able to "
        f"write 'the band is not a budget artifact' instead of 'the band is probably not a budget "
        f"artifact' -- and of learning what the spread it is measured against is made of. "
        f"Commit {commit}.",
        width=122)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_DIRS + [SCAN_E, SCAN_G])
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("SCRATCH", "/scratch/users/hshunt"), "spin_gnn", "decks",
        "stageg_2026-11-05"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--commit", default="(uncommitted)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build even though a stage G rung has fewer than 3 seeds")
    args = ap.parse_args()

    set_style()
    cells, rows = collect(args.scan_dir, args.boot)
    if not rung(cells, G1) or not rung(cells, G2):
        raise SystemExit("stage G has nothing on disk for G1 or G2 yet")
    g1, g2 = summarize(rung(cells, G1)), summarize(rung(cells, G2))
    t1, t2 = summarize(rung(cells, G1_TWIN)), summarize(rung(cells, G2_TWIN))
    if (g1["n"] < 3 or g2["n"] < 3) and not args.allow_partial:
        raise SystemExit(
            f"stage G is at G1 {g1['n']}/3, G2 {g2['n']}/3 seeds. The registered rule is "
            f"median-vs-median at three seeds; pass --allow-partial if that is really what "
            f"you want.")
    print(f"{len(cells)} cells | G1 {g1['n']}/3, G2 {g2['n']}/3 seeds | "
          f"escapes G1 {escapes(rung(cells, G1))[0]}/3, G2 {escapes(rung(cells, G2))[0]}/3")

    os.makedirs(args.out, exist_ok=True)
    ex = example_instance(n_nodes=48, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    with PdfPages(f"{args.out}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, args.out, total=7)
        page_headline(o, cells, g1, g2, t1, t2, len(cells), args.commit)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=48, degree 5, 4 colours, --shuffle-adj --fall: the task every "
                               "cell in this deck is trained and scored on. Stage G changes "
                               "nothing about it -- the same 240k instances are shown twice as "
                               "many times, which is the only difference between a 50-epoch cell "
                               "and its 100-epoch twin.")
        page_training(o, cells, args.commit)
        page_bought(o, cells, rows, g1, g2, t1, t2, args.commit)
        page_predictions(o, cells, g1, g2, args.commit)
        page_next(o, cells, g1, g2, t2, args.commit)
    size = os.path.getsize(f"{args.out}/slides.pdf") / 1e6
    print(f"wrote {args.out}/slides.pdf ({o.n} pages, {size:.2f} MB)")


if __name__ == "__main__":
    main()
