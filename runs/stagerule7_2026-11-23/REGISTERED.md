# Stage RULE7 — registered before the scorer exists

Written 2026-11-23 after stage RULE6 (findings 129/130) and before any
per-`(run, rho)` value of `u` has been read. The probe that preceded this file
read the **function signatures** of `stagerule5_fit.slice0`,
`stagerule6_fit.cells_j0` and `stagerule6_fit.uniform_by_rho` and nothing else;
no stopping-condition number below the per-density level exists in this
session's transcript except the ones RULE6 already published, which are quoted
here as the stage's basis.

Seventh refit of `runs/stagelogic_2026-11-17`. **Nothing is trained and nothing
is submitted.** No closed knob is reopened; `seed` appears as an *axis of
description* only, and the training it would imply stays hshunt's call.

## What RULE6 left open, and why it matters within hours of publishing it

Finding 130 is the newest claim on this line and the weakest supported:

> THE 79 RUNS SIT IN TWO CLUMPS, NOT ON A CONTINUUM — 21 runs at `u <= 0.0716`,
> 58 at `u >= 0.2061`, gap 0.1345 — AND THAT, NOT ARCHITECTURE, IS WHAT POOLED
> `u` MEASURES.

It is post-hoc, its boundary was read off the data after seeing it, and it is
the stage's argument that **seed** is the next axis. Before any of that is acted
on, it has to survive one question the same shards answer: **`u` was pooled over
fourteen densities whose chance level varies 16-fold**, and the pooled number
cannot tell a run that keeps DD's stopping condition everywhere from a run that
keeps it only where keeping it is trivial.

RULE6 measured the chance level and published it:

| rho | 0.95 | 0.90 | 0.85 | 0.80 | 0.75 | 0.70 | 0.65 | 0.60 | 0.55 | 0.50 | 0.45 | 0.40 | 0.35 | 0.33 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `u_unif` | .0492 | .0925 | .1616 | .2245 | .2762 | .3610 | .4246 | .5234 | .5769 | .6338 | .7114 | .7545 | .8015 | .8095 |

`u_unif(rho) = 1 - E|F| / n_blank` is what `u` would be if the first emitted
cell were drawn uniformly from the blanks: at `rho = 0.95` there are 3 blank
cells of which 2.85 are forced, so **doing nothing scores 0.0492**. At
`rho = 0.33` there are 32.0 blanks of which 6.10 are forced and the same
procedure scores 0.8095. Equal weight over the grid it is 0.4572.

So a pooled `u` of 0.0716 is 0.157x the pooled chance level, but that single
ratio is a ratio of means and says nothing about *where on the grid* the
failures sit. A low-clump run could be uniformly excellent, or it could be at
`u = 0` on the nine easiest bins and near chance on the hardest — the arithmetic
permits `u(0.33)` as large as `14 x 0.0716 = 1.00` for a run at the boundary.

## The quantity

Per `(run_id, rho)` at **`j = 0` only**, with no pooling over density:

$$u(r,\rho) \;=\; 1 - \frac{n_{\text{in }F}(0)}{n_{\text{at risk}}(0)}, \qquad n_{\text{at risk}}(0) = 400 \ \text{by construction}$$

and its **chance-normalised** form, which is the stage's object:

$$\tilde u(r,\rho) \;=\; \frac{u(r,\rho)}{u_{\text{unif}}(\rho)}, \qquad u_{\text{unif}}(\rho) = 1 - \frac{\mathbb{E}|F|}{n_{\text{blank}}(\rho)}$$

`u_tilde = 0` is DD's stopping condition exactly; `u_tilde = 1` is
indistinguishable from drawing a blank cell at random. `u_tilde > 1` is worse
than random and is permitted by the definition.

`E|F|` at `j = 0` is a property of the **instance and the hints**, not of the
model — no token has been emitted yet — so the denominator is the same for every
run at a given density. **D2 tests exactly that** and the stage is unreadable if
it fails.

The clump membership under test is RULE6's: the **21** runs at `u <= 0.0716`
and the **58** at `u >= 0.2061`, boundary `401/5600 = 0.0716071`, computed by
`stagerule6_fit.clump_split` with `MIN_SIDE = 5`.

## Hypothesis

The low clump is a *run-level* property that holds across the grid, and the
chance normalisation leaves the two clumps separated. If instead `u_tilde`
converges as `rho` falls, then "keeps the stopping condition" is a statement
about the easy end of the grid and finding 129's two-arm table needs a density
qualifier in public.

## Predictions, each with an explicit size floor

- **E1 — the low clump is low at every density, not only where the task is
  easy.** For **all 21** low-clump runs, `max over the 14 densities of u_tilde
  <= 0.50`. **MISS** if any low-clump run reaches `u_tilde >= 0.50` at any bin:
  the pooled `u` would then be hiding a collapse, and finding 129's "0 of 8
  objects keep the stopping condition" becomes a claim that needs a *where*.
- **E2 — normalised failure rises as hints are removed.** Pooling the 21
  low-clump runs by density, `Spearman(u_tilde, rho) <= -0.60` **and**
  `u_tilde(0.33) / u_tilde(0.95) >= 2.0`. **MISS** if either fails. The
  normalisation itself pushes the other way — dividing by 0.0492 at `rho = 0.95`
  amplifies a small `u` twenty-fold — so this is not a foregone sign.
- **E3 — the clump split is reproduced inside a single density.** Rank the 79
  runs by `u(r, rho = 0.80)` **alone** (10 blank cells, chance 0.2245, a bin
  inside the fitted ten) and split at the largest gap with at least 5 runs a
  side, the same rule `clump_split` uses. The low side's membership agrees with
  RULE6's 21-run low clump at **Jaccard >= 0.80**. **MISS** if `< 0.80`: the
  clumps would be a property of the pooled statistic rather than of the runs,
  and finding 130's headline sentence is wrong as written.
- **E4 — the high clump is a constant fraction of chance, not a constant
  rate.** For the 58 high-clump runs pooled by density,
  `max u / min u >= 3.0` while `max u_tilde / min u_tilde < 3.0`. **MISS** if
  the raw spread is itself `< 3.0` (nothing to flatten) or the normalised
  spread is `>= 3.0` (the normalisation does not explain the high clump).
- **E5 — reproduction.** Summing the per-`(run, rho)` `j = 0` counts over
  density reproduces stage RULE6's 31-cell `u` table **and** its clump split
  (21 / 58, gap 0.1345) to **`0.000e+00`**. **MISS** on any non-zero residual:
  the slice would not be the same slice and nothing else in the stage is
  readable.

## Diagnostics

- **D1** — `n_at_risk(0) = 400` on every one of the `79 x 14 = 1,106`
  `(run, rho)` pairs, total 442,400, no pair missing. Zero violations required.
- **D2** — **the chance baseline is a property of the instances.** At each
  density, `F_sum(0) / n_at_risk(0)` is identical across all 79 runs to
  `< 1e-9`. Zero violations required. If this fails the runs do not share an
  eval set, and no cross-run comparison anywhere in 121–130 is valid.
- **D3** — `n_in_F(0) <= n_at_risk(0)` on every row, so `u` lies in `[0, 1]`.
- **D4** — the recomputed `u_unif(rho)` reproduces the fourteen values tabled
  above to 4 decimal places, and `n_blank(rho)` reproduces the j-independent
  ladder 3, 5, 8, 10, 12, 15, 17, 20, 22, 24, 27, 29, 31.635, 32.0025.
- **D5** — keyed on `run_id` for everything about clumps and on `cell` for
  anything per-architecture; **never** on `(arch_mode, num_layers, d_model)`,
  which pools `no_mlp L=6` with `no_mlp L=6 ep100`.

## The interpretive limit

`u` measures **agreement with DD's stopping condition**, not logical validity; a
model that names a cell propagation has not yet forced, and names it correctly,
is doing something defensible and is simply not following DD's trace. RULE6's
limit carries over unchanged. What is new here is only *where on the density
grid* that agreement sits.

`u_tilde`'s denominator is a **uniform draw over blank cells**, which is the
weakest possible comparison. Beating it is necessary, not sufficient, and
`u_tilde = 0.5` should never be read as "half right" — it is "half as often
off-trace as a procedure with no model in it".

Everything is `j = 0`. Nothing here speaks to later indices, where censoring
returns and RULE3's `lambda* = 1.19` applies.

## What a null licenses

Every outcome is a sentence, and **no training follows from any of them.** If E1
misses, findings 129 and 130 each get a correction page and the two-arm table
gains a density column. If E3 misses, finding 130's headline is rewritten in
public. If everything hits, the case that **seed** is the next axis is
strengthened — and it still stays hshunt's call, because seeds are training. The
width sweep between `d192` and `d320` stays closed, the two single-seed seeding
candidates stay unsubmitted, and stage DQ4 stays unsubmitted.

## Traps

1. **RULE6's `u` and this stage's `u(r, rho)` are the same slice at different
   granularity.** E5 exists to prove that. Never quote a per-density `u` as if
   it were the pooled one.
2. **Name the densities, never "the fourteen."** RULE5's E4 quoted a Spearman
   over a different ten bins than it registered. Every scored set here prints
   its own bins, and E3 names `rho = 0.80` in advance.
3. **The clump boundary is descriptive and is never a test.** It is reused here
   only to carry RULE6's membership forward; `clump_split` is imported, not
   retyped, so the rounded-literal defect fixed in `f2e47f3` cannot come back.
4. **A ratio of means is not a mean of ratios.** The pooled `u / u_unif` of
   0.157x is not `u_tilde` and may not be quoted as it.
5. **Six single-seed cells**, and the clump work is per *run*, so a cell with
   one run contributes one point. Any sentence about a cell states its seed
   count.
6. **The talk deck `transition-and-minimal-models` is not touched**, whatever
   this stage finds. hshunt is editing a local copy.
7. `u_tilde > 1` is not an error in the code. It means worse than a random blank
   pick, and must be reported as such rather than clipped.
