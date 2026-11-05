"""Is the grid continuous in capacity, or does it come in rungs?

WHAT THIS IS FOR. Stages A-D were each scored row by row: one ablation against
the hub, against a frozen bar. Read that way, every row produced the same
shrug -- "worse than the hub, better than the floor". Read as one table, the
rows do something the individual verdicts never reported: they PILE UP. Nine
structurally distinct rungs land at rho_50 0.740-0.765, spanning a factor of
2.8 in trainable parameters and six different interventions (shallower depth,
a narrower stream, a narrower MLP, a linearised MLP, a frozen random MLP, no
MLP at all).

THE STATISTIC THAT MAKES THAT A CLAIM RATHER THAN A SQUINT, and the only one
this script exists to compute: compare the spread of median rho_50 ACROSS the
rungs in a band against the largest across-SEED spread WITHIN any single one of
them. If the across-rung spread is smaller, the rungs are not merely similar,
they are UNRESOLVABLE FROM EACH OTHER at this budget -- changing the
architecture moved the model less than changing the seed does.

WHAT THIS IS NOT. It is a descriptive statistic on rungs that already exist, it
has no null hypothesis, and the band edges below are a reading convention and
not a fitted boundary. It cannot distinguish "every ablation costs the same
thing" from "every ablation happened to cost about the same at this budget";
that is what stage E (51171824, depth) and stage D's D2 (51151957, width) are
for, both of which give an ablated model capacity back and ask whether it
crosses. Nothing here is scored against s*, and nothing here is a verdict.

A rung enters the table only with all three seeds on disk. Censored rungs are
listed with their mean_solve and kept out of every rho_50 statistic, since
">0.95" is an ordering fact and not a number.

    python3 -m scripts.rung_collapse
    python3 -m scripts.rung_collapse --scan-dir <dir>   # repeatable, adds dirs
"""
import argparse
import random
from collections import defaultdict

from scripts.minarch_analysis import (
    SCAN_DIRS, PLATEAU, D_DEFAULT, FF_DEFAULT, W_DEFAULT,
    load_all, cell_key, cell_metrics, rung_label, history, median, spread,
    _bound_median,
)

# The reading convention, fixed here so it is not chosen to suit the answer.
# 0.70-0.82 brackets the pile-up with room on both sides: the nearest rung
# below it is the hub at 0.592 and the nearest above is L=2 full at 0.919, so
# no rung sits near either edge and widening or narrowing the window by 0.05
# changes nothing about which rungs are inside it.
BAND_LO, BAND_HI = 0.70, 0.82
N_PERM = 20000
PERM_SEED = 0


def permutation_p(groups, n_perm=N_PERM, rng_seed=PERM_SEED):
    """Is the spread of rung medians larger than label-shuffling alone gives?

    WHY THIS IS HERE. The headline comparison -- across-rung spread of MEDIANS
    against the worst within-rung spread of SEEDS -- is not a fair contest, and
    saying so is the point of this function. A median of three draws is less
    variable than a single draw, so across-rung < within-rung is partly
    guaranteed by arithmetic and would appear even if the rungs genuinely
    differed. The honest question is whether the rung LABELS carry any
    information at all about the metric.

    Null: within the band, the cells are exchangeable -- which rung a cell came
    from tells you nothing about its score. Statistic: the spread of the per-
    rung medians, exactly the quantity reported above. Shuffle the labels, keep
    the group sizes, recompute.

    p is the fraction of shuffles whose spread is at least the observed one, so
    a LARGE p means the observed separation is no more than relabelling noise
    and the rungs are not distinguishable. That is the direction this test is
    being used in, which makes it a test that can only FAIL to reject: it can
    never prove the rungs identical, only show the data has no power to tell
    them apart. Reported with n and the group sizes so that is visible.

    Run on mean_solve and not rho_50: mean_solve never censors, so every cell
    in the band contributes, whereas four of the band's rungs have a censored
    seed whose rho_50 is an inequality and cannot be shuffled as a number.
    """
    labels = [k for k, vs in groups.items() for _ in vs]
    vals = [v for vs in groups.values() for v in vs]
    sizes = [len(vs) for vs in groups.values()]

    def stat(lab, val):
        g = defaultdict(list)
        for k, v in zip(lab, val):
            g[k].append(v)
        return spread([median(v) for v in g.values()])

    obs = stat(labels, vals)
    rng = random.Random(rng_seed)
    shuffled = list(vals)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        if stat(labels, shuffled) >= obs:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1), sizes


def rungs(cells):
    """Group cells into rungs keyed by everything except the seed."""
    out = defaultdict(list)
    for k, m in cells.items():
        out[k[:-1]].append(m)
    return out


def describe(key, ms):
    """One rung's median metrics, plus the within-rung across-seed spreads."""
    crossed = [m["rho50"] for m in ms if m["flag"] == "ok"]
    bm = _bound_median([(m["rho50"], m["flag"] == "ok") for m in ms])
    nll = [history(m["path"])[1] for m in ms]
    return {
        "key": key,
        "label": rung_label(*key) if len(key) == 3 else rung_label(*key[:3], *key[3:]),
        "n": len(ms),
        "n_cross": len(crossed),
        "rho50": bm["rho50_bm"],
        "readable": bm["bm_readable"],
        # Only defined when every seed crossed: a censored seed contributes no
        # distance, so a spread over the survivors alone understates the rung.
        "rho50_spread": spread(crossed) if len(crossed) == len(ms) else float("nan"),
        "mean_solve": median([m["mean_solve"] for m in ms]),
        "mean_spread": spread([m["mean_solve"] for m in ms]),
        "trainable": ms[0]["n_trainable"],
        "escapes": sum(1 for v in nll if v < PLATEAU),
    }


def row(r):
    rho = f"{r['rho50']:.3f}" if r["readable"] else "CENSORED"
    sp = "    -  " if r["rho50_spread"] != r["rho50_spread"] else f"{r['rho50_spread']:.3f}  "
    return (f"  {r['label']:<24}{rho:>9} {sp:>8}{r['mean_solve']:>8.4f} "
            f"{r['mean_spread']:>8.4f} {r['trainable']:>11,}  {r['escapes']}/{r['n']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-dir", action="append", default=[],
                    help="extra scan directory; repeatable. Stage E lives "
                         "outside SCAN_DIRS on purpose -- see minarch_analysis.")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    dirs = SCAN_DIRS + args.scan_dir
    by = defaultdict(list)
    for r in load_all(dirs):
        by[cell_key(r)].append(r)
    cells = {k: dict(cell_metrics(v, args.boot), path=v[0]["path"])
             for k, v in by.items()}

    complete = {k: ms for k, ms in rungs(cells).items() if len(ms) == 3}
    partial = {k: ms for k, ms in rungs(cells).items() if len(ms) != 3}
    described = sorted((describe(k, ms) for k, ms in complete.items()),
                       key=lambda r: (not r["readable"], r["rho50"]))

    print("RUNG COLLAPSE: every rung with 3 seeds on disk, ordered by rho_50")
    print(f"  scanned {len(dirs)} directories, {len(cells)} cells, "
          f"{len(complete)} complete rungs ({len(partial)} partial, excluded)")
    print()
    print(f"  {'rung':<24}{'rho_50':>9} {'spread':>8}{'mean':>8} {'spread':>8} "
          f"{'trainable':>11}  esc")
    print("  " + "-" * 76)

    band = [r for r in described if r["readable"] and BAND_LO <= r["rho50"] <= BAND_HI]
    shown_band = False
    for r in described:
        in_band = r in band
        if in_band and not shown_band:
            print(f"  --- the pile-up, rho_50 in [{BAND_LO}, {BAND_HI}] " + "-" * 32)
            shown_band = True
        if shown_band and not in_band and r["rho50"] > BAND_HI:
            print("  " + "-" * 76)
            shown_band = False
        print(row(r))
    print()

    if len(band) < 3:
        print(f"  only {len(band)} complete rung(s) in the band -- nothing to collapse.")
        return

    across = spread([r["rho50"] for r in band])
    within = [r["rho50_spread"] for r in band if r["rho50_spread"] == r["rho50_spread"]]
    worst_within = max(within) if within else float("nan")
    params = [r["trainable"] for r in band]
    across_mean = spread([r["mean_solve"] for r in band])
    worst_mean_within = max(r["mean_spread"] for r in band)

    print(f"THE BAND: {len(band)} rungs at rho_50 {min(r['rho50'] for r in band):.3f}"
          f"-{max(r['rho50'] for r in band):.3f}")
    print(f"  trainable parameters      {min(params):,} to {max(params):,} "
          f"({max(params) / min(params):.1f}x)")
    print(f"  ACROSS-RUNG spread, rho_50          {across:.4f}")
    print(f"  worst WITHIN-RUNG seed spread       {worst_within:.4f}"
          f"   ({len(within)} of {len(band)} rungs have all 3 seeds crossing)")
    print(f"  ACROSS-RUNG spread, mean_solve      {across_mean:.4f}")
    print(f"  worst WITHIN-RUNG seed spread       {worst_mean_within:.4f}")
    print()
    if worst_within == worst_within and across < worst_within:
        print(f"  >>> ACROSS-RUNG < WITHIN-RUNG in rho_50 ({across:.4f} < "
              f"{worst_within:.4f}).")
        print("      These rungs are not resolvable from each other at this budget:")
        print("      six different interventions over a "
              f"{max(params) / min(params):.1f}x parameter range moved the model")
        print("      LESS than changing the random seed moves one of them.")
    else:
        print(f"  >>> across-rung {across:.4f} is NOT smaller than the worst "
              f"within-rung {worst_within:.4f};")
        print("      the rungs are close but separable. Do not call this a collapse.")
    print()

    # The comparison above is a spread of medians against a spread of seeds,
    # which arithmetic alone tilts one way. This is the version that is not
    # tilted: does the rung label predict the score at all?
    band_cells = {}
    for r in band:
        band_cells[r["label"]] = [m["mean_solve"] for m in complete[r["key"]]]
    obs, p, sizes = permutation_p(band_cells)
    print(f"PERMUTATION TEST on mean_solve, {len(band_cells)} rungs x "
          f"{'/'.join(str(s) for s in sizes)} seeds, {N_PERM:,} shuffles (seed {PERM_SEED})")
    print("  null: inside the band the rung label carries no information")
    print(f"  observed spread of rung medians  {obs:.4f}")
    print(f"  p(shuffled spread >= observed)   {p:.3f}")
    if p > 0.10:
        print(f"  >>> NOT REJECTED at p={p:.3f}. Shuffling which rung a cell belongs to")
        print("      reproduces the observed separation easily, so the architecture")
        print("      label inside this band predicts nothing the seed does not.")
        print("      This is a failure to reject and NOT proof the rungs are equal:")
        print("      3 seeds per rung has little power. It is the weaker, true claim.")
    else:
        print(f"  >>> REJECTED at p={p:.3f}. The rungs in the band ARE separable")
        print("      and the pile-up above must not be written as a collapse.")
    print()
    # POSITIVE CONTROL. A p of 0.569 above means nothing on its own: a test
    # with no power never rejects, and "we cannot tell these apart" and "this
    # test cannot tell anything apart" produce the same number. So run the
    # identical test, same n, same shuffle count, on rungs that are KNOWN to
    # differ -- the full model's own depth ladder, which descends 0.8163 to
    # 0.0807 and which every stage of this programme has treated as real. If
    # that also fails to reject, the band result is an artifact of 3 seeds and
    # must be withdrawn.
    ladder = {}
    for L in (12, 8, 6, 4, 3, 2):
        key = (L, 4, "full")
        if key in complete:
            ladder[f"L={L} full"] = [m["mean_solve"] for m in complete[key]]
    if len(ladder) >= 3:
        c_obs, c_p, c_sizes = permutation_p(ladder)
        print(f"POSITIVE CONTROL, same test on the full-model depth ladder "
              f"({len(ladder)} rungs x {'/'.join(str(s) for s in c_sizes)})")
        print(f"  observed spread of rung medians  {c_obs:.4f}")
        print(f"  p(shuffled spread >= observed)   {c_p:.3f}")
        if c_p <= 0.05:
            print(f"  >>> REJECTED at p={c_p:.3f}, as it must be. The test HAS power at")
            print("      this n, so the band's p is a real failure to separate and not")
            print("      an empty one.")
            print(f"      BUT READ THE MARGIN, NOT JUST THE SIGN: it took a {c_obs:.3f}")
            print(f"      separation -- the ladder's entire dynamic range -- to reach")
            print(f"      p={c_p:.3f}. The band's separation is {obs:.4f}, "
                  f"{c_obs / obs:.0f}x smaller.")
            print("      So the defensible claim is NO DETECTABLE SEPARATION at 3 seeds,")
            print("      with a resolution limit somewhere between those two numbers.")
            print("      'These architectures are equivalent' is NOT what this shows and")
            print("      would need seeds this grid does not have.")
        else:
            print(f"  >>> NOT REJECTED at p={c_p:.3f}. THE TEST HAS NO POWER AT THIS n.")
            print("      It cannot separate rungs the whole programme treats as")
            print("      different, so the band result above says nothing and must")
            print("      not be reported. More seeds, or a different statistic.")
        print()

    print("  WHAT IT DOES NOT SHOW: whether an ablated model given capacity back")
    print("  crosses the band. Every rung above was run at AT MOST the hub's depth.")
    print("  Stage E (51171824) gives the attention-only stack 1.3x and 2x the depth;")
    print("  stage D's D2 (51151957) gives it 1.5x the width. Those decide it.")

    if partial:
        print()
        print(f"  --- {len(partial)} partial rung(s), excluded from everything above ---")
        for k, ms in sorted(partial.items(), key=lambda kv: str(kv[0])):
            r = describe(k, ms)
            print(f"  {r['label']:<24}{r['n']}/3 seeds   mean_solve {r['mean_solve']:.4f}")


if __name__ == "__main__":
    main()
