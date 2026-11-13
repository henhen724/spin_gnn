#!/usr/bin/env python3
"""When does a tied d320 cell leave the plateau? A census of the logs on disk.

WHY THIS EXISTS. Stage W's seed 2 is still on the plateau at epoch 35 while its
own 50-epoch control -- same seed, same config, only the budget differs --
stepped at epochs 21-24. Before calling that seed broken, the honest question is
what the step epoch does across every tied d320 cell already trained, because
this project has twice found the step, not the architecture, to be what moves a
number (finding 64: which basin the optimizer finds is the binding constraint;
stage V's one untrained seed carried the whole failure).

Reads only this project's own *.out training logs. No GPU, no new cell, no
re-training: text parsing of files already on disk.

DEFINITION, stated before the numbers. The step is the first epoch e >= MIN_EP
with THREE properties: a flat run-in (train_nll fell less than FLAT per epoch
over the preceding RUNIN epochs), a one-epoch fall of at least DROP at e, and
persistence (it keeps falling by at least PERSIST over the following WINDOW
epochs). All three matter. The first version of this script had only the last
two and reported epoch 5 for 12 of 15 cells -- that is the TAIL OF THE INITIAL
COLLAPSE, where a 0.02 fall over five epochs is unremarkable, and it is the same
bug stage U's deck builder had when breakaway() reported epoch 0. A step is only
a step if the cell had stopped moving first.
"""

import os
import re
import sys
from statistics import median

MIN_EP = 8          # skip the initial collapse out of init
RUNIN = 5           # the cell must have been flat for this many epochs...
FLAT = 0.0025       # ...falling less than this per epoch
DROP = 0.005        # one-epoch fall that counts as a step onset
PERSIST = 0.020     # it must keep going...
WINDOW = 5          # ...over this many epochs

BASIN = 1.561       # W-d's line, between stage U's two populations
FLOOR = 1.600       # the recipe's own floor; the tight seeds land 1.585-1.592

ROOT = "/scratch/users/hshunt/spin_gnn"
EPOCH = re.compile(r"^epoch (\d+): train_nll=([0-9.]+).*val_accuracy=([0-9.]+)")


def level(nll):
    return "BASIN" if nll <= BASIN else ("FLOOR" if nll <= FLOOR else "STUCK")


def config_of(path):
    """The cell's config line, whose format differs by stage."""
    cfg = {}
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 400:
                break
            # Knobs come ONLY from the one config echo, the line carrying seed=.
            # Scanning every line picked up an unrelated k=0 elsewhere and
            # silently dropped stage S out of the comparable set.
            if "seed=" in line:
                for key in ("d_model", "seed", "ff", "nhead", "layers",
                            "applications", "epochs", "k"):
                    m = re.search(r"\b%s=(\d+)\b" % key, line)
                    if m and key not in cfg:
                        cfg[key] = int(m.group(1))
            if "arch=tied_no_mlp" in line or "arch_mode=tied_no_mlp" in line:
                cfg["tied"] = True
    return cfg


def history(path):
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = EPOCH.match(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def step_epoch(h):
    """First persistent fall out of a FLAT run-in, per the docstring."""
    nll = [x[1] for x in h]
    for e in range(MIN_EP, len(nll) - 1):
        if nll[e - 1] - nll[e] < DROP:
            continue
        if (nll[e - 1 - RUNIN] - nll[e - 1]) >= FLAT * RUNIN:
            continue                      # still descending; no plateau behind it
        far = min(e + WINDOW, len(nll) - 1)
        if nll[e - 1] - nll[far] >= PERSIST:
            return e
    return None


def onset(h):
    """The largest fall out of a FLAT run-in, with its persistence reported.

    step_epoch() answers the registered question -- did a step happen -- and
    returns the first epoch passing ALL THREE tests. This answers a different
    one, asked by a deck page that has to describe what a plotted curve visibly
    does: where is the biggest one-epoch fall out of a flat stretch, and did it
    persist? A fall that clears DROP out of a flat run-in and then misses
    PERSIST is invisible to step_epoch() and plainly visible to a reader, and a
    page claiming "no step anywhere" over a figure showing one is the defect
    this exists to prevent.

    Returns None when no epoch clears DROP out of a flat run-in. Changes no
    registered verdict: step_epoch() above is untouched and stays the
    definition; `step` in the returned dict is its criterion, not a new one.
    """
    nll = [x[1] for x in h]
    best = None
    for e in range(MIN_EP, len(nll) - 1):
        drop = nll[e - 1] - nll[e]
        if drop < DROP:
            continue
        runin = nll[e - 1 - RUNIN] - nll[e - 1]
        if runin >= FLAT * RUNIN:
            continue
        far = min(e + WINDOW, len(nll) - 1)
        persist = nll[e - 1] - nll[far]
        if best is None or drop > best["drop"]:
            best = {"epoch": e, "drop": drop, "runin": runin,
                    "persist": persist, "step": persist >= PERSIST,
                    # the two endpoints of the fall, so a caller can ask which
                    # level it crossed without re-indexing the history itself
                    "before": nll[e - 1], "after": nll[e]}
    return best


def main():
    rows = []
    for name in sorted(os.listdir(ROOT)):
        if not name.endswith(".out"):
            continue
        path = os.path.join(ROOT, name)
        try:
            cfg = config_of(path)
        except (IOError, OSError):
            continue
        if not cfg.get("tied") or cfg.get("d_model") != 320:
            continue
        h = history(path)
        if len(h) < MIN_EP + WINDOW + 2:
            continue
        rows.append((name, cfg, h, step_epoch(h)))

    if not rows:
        sys.exit("no tied d320 training logs matched -- check ROOT")

    print("TIED d320 CELLS ON DISK: when does train_nll leave the plateau?")
    print("step = first epoch with a %.3f one-epoch fall that keeps falling "
          "%.3f over %d epochs,\n       scanned from epoch %d so the initial "
          "collapse cannot be mistaken for it.\n" % (DROP, PERSIST, WINDOW, MIN_EP))
    hdr = ("%-34s %4s %4s %4s %5s %6s %8s %8s %7s"
           % ("log", "seed", "appl", "k", "last", "step", "pre_nll", "end_nll", "end_acc"))
    print(hdr)
    print("-" * len(hdr))
    for name, cfg, h, se in rows:
        pre = "%8.4f" % h[se - 1][1] if se else "       -"
        print("%-34s %4s %4s %4s %5d %6s %s %8.4f %7.4f"
              % (name[:34], cfg.get("seed", "?"), cfg.get("applications",
                 cfg.get("layers", "?")), cfg.get("k", 1), h[-1][0],
                 se if se is not None else "NONE", pre, h[-1][1], h[-1][2]))

    comparable = [(n, c, h, s) for n, c, h, s in rows
                  if c.get("k", 1) == 1
                  and c.get("applications", c.get("layers")) == 16]
    stepped = [s for _, _, _, s in comparable if s is not None]
    print("\nCOMPARABLE SET (k=1, 16 applications, d320 tied): %d cells, "
          "%d stepped, %d did not."
          % (len(comparable), len(stepped), len(comparable) - len(stepped)))
    if len(stepped) >= 3:
        print("step epoch: median %.0f, range %d-%d."
              % (median(stepped), min(stepped), max(stepped)))
    elif stepped:
        print("step epochs: %s -- too few for a median; quote them, do not "
              "summarise them." % ", ".join(str(s) for s in sorted(stepped)))
    print("\nBUT 'NONE' LUMPS TWO OPPOSITE STATES, so split them by endpoint. The")
    print("project's three levels: BASIN <= %.3f (stage U's two outliers), FLOOR"
          % BASIN)
    print("<= %.3f (the recipe's own floor, where the tight seeds land), STUCK"
          % FLOOR)
    print("above it (stage V's seed 1 ended at 1.6334 and never left).\n")
    for name, cfg, h, se in rows:
        running = h[-1][0] < int(cfg.get("epochs", 50)) - 1
        print("  %-34s %-6s step %-5s -> %s"
              % (name[:34], level(h[-1][1]), se if se is not None else "none",
                 "SO FAR, still running at epoch %d" % h[-1][0] if running
                 else "finished"))
    print("\nTWO KINDS OF STEP, and they are not the same event:")
    print("  plateau -> floor: the control's seed 2 at epoch 22, 1.6399 -> 1.5864,")
    print("      catching up to where its two siblings already were.")
    print("  floor -> basin:   stage U k=2 s2 at 15 and k=4 s1 at 21, ~1.590 ->")
    print("      ~1.53, going BELOW the floor -- the rho_50 0.603 / 0.579 seeds.")
    print("\nA cell that is still running shows its CURRENT epoch as 'last': "
          "'none' there means not yet, not never.")


if __name__ == "__main__":
    main()
