#!/usr/bin/env python3
"""The stage-independent explanation pages, written once and shared by every deck.

WHY THIS EXISTS. ~/CLAUDE.md was revised on 2026-11-12 to require that a deck
explain itself to a reader who has seen no earlier deck from the project and has
not read the code: every technical term used anywhere in the deck is defined in
the deck, and every computed quantity is defined with an equation whose symbols
are all introduced. Stage V's deck predates that rule. It has four explanation
pages (the knob, the question, tokenization, the trace) and uses rho_50, solve
rate, the gate, the tied block, the three NLL levels and the two baselines
without ever defining them.

None of these pages depend on which stage is being reported, which is the point:
the definitions are the same whether the arm is a schedule, a budget or a width,
so they are written once here and imported, rather than retyped per deck and
drifting. Only the RESULTS pages belong in a stage's own builder.

ORDER THE BUILDER SHOULD EMIT. The task first, then what the model sees, then
what is measured, then what counts as a pass -- a reader needs the quantity
before the number:

    TASK, HINTS, DEDUCTION, [tokenization figure], [trace figure], GENERATION,
    ARCHITECTURE, SOLVE_RATE, RHO50, CENSOR, GATE, ACCURACY, SUFFICIENCY,
    LEVELS, LEAK, BASELINES

The two figure pages stay in the stage builder because the PNGs are rendered
per deck; everything else is here. PAGES is that list in that order, with a
single FIGURE_SLOT entry standing for BOTH figure pages -- the builder replaces
that one element with its tokenization page followed by its trace page.

PAGE RULES THESE OBEY. Titles are sentences carrying the page's content, <= 12
words. The <= 25-word body cap does NOT apply here -- the definitions are the
page, and ~/CLAUDE.md names that cap as the likeliest reason these pages have
been getting skipped. Math is $...$ or $$...$$; \\(...\\) is not math here.
`\\mathbb{1}` is NOT in KaTeX, so indicator functions use \\mathbf{1}.

EVERY NUMBER BELOW IS CHECKED, not retyped: check() asserts the parameter count,
the gate and the accuracy bar against scripts.stagew_score, which imports them
in turn from the stage that measured them. A deck that contradicts the scorer
fails to build instead of publishing.
"""

FIGURE_SLOT = "<<FIGURE>>"

TASK = """## Colour a 48-node graph so no edge joins two equal colours

- $N = 48$ nodes, each with exactly $d = 5$ neighbours (a 5-regular graph), and $q = 4$ available colours.
- A *proper colouring* assigns one of the $q$ colours to every node so that no edge has the same colour at both ends.
- Instances are built by **planting**: a proper colouring is drawn first and the graph is constructed around it, so a valid answer is guaranteed to exist. A failure is always the model's, never an unsolvable instance.
- This is the graph-colouring form of the constraint problem; nothing here is a Sudoku grid, despite the `sudoku` in the script names, which are inherited from the earlier 9x9 version of the same code.
"""

HINTS = """## Hint density $\\rho$ is the fraction of nodes revealed in the prompt

- A **hint** is a node whose planted colour is given away in the prompt. The **hint density** is

$$\\rho = \\frac{n_{\\text{hint}}}{N}$$

  with $n_{\\text{hint}}$ the number of revealed nodes and $N = 48$.
- $\\rho$ near 1 is easy: almost everything is given. $\\rho$ near 0 is hard: almost nothing is, and the answer has to be derived through long chains of inference.
- Every figure uses the **realized** density, not the requested one. The generator cannot always hit a requested $\\rho$ exactly -- it is constrained by what it can build -- so the honest $x$-axis is the density the instance actually ended up with.
"""

DEDUCTION = """## A colour is forced when its neighbours show the other three

- Each unknown node carries a **candidate set**: the colours not yet ruled out for it. It starts as all $q = 4$.
- A neighbour of known colour **eliminates** that colour from the set. When $q - 1 = 3$ of the four have been eliminated, one candidate remains and the node's colour is **forced**.
- **Direct deduction (DD)** is this rule and nothing else: no guessing, no backtracking, no trying a colour to see if it fails later.
- One **round** applies the rule to every node at once, then updates. An instance is DD-solvable if repeated rounds resolve every node; the number of rounds needed is the depth of its longest inference chain.
- Draws from this generator reach **19 rounds**, so the deepest chains need 19 successive applications before the last node is pinned.
"""

GENERATION = """## The model is prompted with the problem and writes the rest itself

- The prompt is the whole problem and nothing more: the graph's edges, the hint colours, and a `BEGIN_REASONING` marker.
- From there the model generates every remaining token on its own -- the deduction steps in order, then a terminal `SOLVED` with a full colouring, or `UNRESOLVED` with the nodes it could not pin.
- Decoding is greedy (temperature 0), so a cell's output on an instance is deterministic and two cells can be compared on the same instance list.
- This is the harder of the two things measured here. It is **not** next-token prediction given a correct history: nothing after the marker is given to the model, so an early mistake is one the model then has to keep reasoning from.
"""

ARCHITECTURE = """## One block applied sixteen times: $h_{t+1} = F(h_t)$, at tied weights

- The network is a single transformer block $F$ applied $T = 16$ times to the token sequence, not a stack of 16 different blocks. **Tied** means the same weights at every application, so depth costs no parameters.
- $h_t$ is the sequence of $d_{\\text{model}} = 320$-dimensional vectors, one per token, after $t$ applications; $h_0$ is the token plus position embedding.
- $F$ is attention only. In `no_mlp` mode the feed-forward sublayer's second linear is zeroed and both of its linears are frozen, so that sublayer writes identically zero for every input. The `ff=1024` in the config is a vestigial field: the module still exists so checkpoints load, but it is not part of the trained function.
- What remains trainable is attention (4 heads), two LayerNorms, the embeddings and the readout: **613,818 parameters**.
- Why this object: one iterated map is the thing a theory could describe. The goal of the project is the simplest architecture that still learns direct deduction -- analytically tractable, not merely small.
"""

SOLVE_RATE = """## Solve rate counts instances solved from a blank reasoning section

- At each hint density, $M = 400$ fresh instances are generated and the model answers each one from the prompt alone:

$$\\text{solve}(\\rho) = \\frac{1}{M}\\sum_{i=1}^{M} \\mathbf{1}\\big[\\hat{o}_i = o_i \\;\\wedge\\; \\text{valid}(\\hat{c}_i)\\big]$$

- $\\hat{o}_i$ is the outcome the model called (`SOLVED` or `UNRESOLVED`) and $o_i$ the true one; $\\hat{c}_i$ is the colouring it listed, and $\\text{valid}$ checks that colouring edge by edge against the graph.
- **Both** conditions are required. Calling `SOLVED` and then listing a colouring with a violated edge counts as a failure, as does calling `UNRESOLVED` on an instance that direct deduction does resolve.
- $\\mathbf{1}[\\cdot]$ is 1 when the condition inside holds and 0 otherwise.
"""

RHO50 = """## $\\rho_{50}$ is the hint density where solve rate first reaches one half

- Solve rate is measured on a grid of **14 densities** from 0.95 down to 0.33, at 400 instances each. It is high at dense hints and falls as hints are removed.
- $\\rho_{50}$ is where that curve crosses one half, scanning from sparse to dense and interpolating linearly between the two bins that straddle it:

$$\\rho_{50} = \\rho_{i-1} + (0.5 - s_{i-1})\\,\\frac{\\rho_i - \\rho_{i-1}}{s_i - s_{i-1}}$$

  where $s_i = \\text{solve}(\\rho_i)$ and $i$ is the first bin with $s_{i-1} < 0.5 \\le s_i$.
- **Lower is better.** A smaller $\\rho_{50}$ means the model still solves half the instances when fewer hints are given, i.e. it sustains longer chains of deduction.
- One number per seed. Three seeds are run per configuration and all three are quoted, because the spread between them has repeatedly been larger than the effect being tested.
"""

CENSOR = """## A censored $\\rho_{50}$ is a result, not a missing number

- If no density bin on the grid reaches a solve rate of 0.5, the curve never crosses and $\\rho_{50}$ does not exist for that seed. That seed is **censored**.
- Censoring is reported, never dropped and never filled in. A cell that fails to reach one half anywhere is worse than any cell that crosses, and silently averaging over only the crossing seeds would hide exactly that.
- Two estimators appear in this project and are never mixed inside one figure: a **bound-aware** summary, which carries the censoring through, and a **median over crossing seeds only**, which is the headline number. Which one produced a figure is stated on that figure's page.
"""

GATE = """## 0.740 is a gate to pass, not a score to beat

- A **gate** is a threshold registered before a run, which an arm must clear for the line of work to continue. It is not the objective. Clearing it means "keep going", not "this is good".
- The standing gate is $\\rho_{50} < 0.740$ in all three seeds for the simple tied model, whose control sits at 0.757.
- The thing actually being chased is the **standing minimum**, 0.580, held by a model four times larger. The gate is the near-term step toward it.
- Registering thresholds in advance is deliberate: every bar quoted in this deck was written into the scorer before the cells landed, so a result cannot be graded against a bar chosen after seeing it.
"""

ACCURACY = """## Token accuracy is teacher-forced, so it is read against a matched control

- The second measurement is next-token accuracy with the **correct** history supplied at every position -- the opposite regime from free generation:

$$\\text{acc} = \\frac{1}{|\\mathcal{V}|}\\sum_{p \\in \\mathcal{V}} \\mathbf{1}\\big[\\arg\\max_{v} z_{p,v} = y_p\\big]$$

- $\\mathcal{V}$ is the set of non-padding positions, $y_p$ the true token at position $p$, and $z_{p,v}$ the model's logit for symbol $v$ there. Train NLL is the same sum with $-\\log \\text{softmax}(z_p)_{y_p}$ in place of the indicator.
- It is **dense**: every token in every sequence contributes, so it moves while solve rate is still flat, which makes it the sensitive instrument of the two.
- Its absolute value is not interpretable on its own -- most positions are structural tokens that any trained cell gets right. So it is only ever read as a difference against a matched control, which is why the registered bar is the control's own accuracy plus a margin rather than a round number.
"""

SUFFICIENCY = """## Solve rate follows token accuracy, to within a measured bar

- Across every cell this project has trained, mean solve rate over the grid is an increasing function of validation token accuracy: a cell that predicts more tokens correctly solves more instances, largely regardless of how it was built.
- The **reference ladder** is the family of cells that differ only in depth, at one width and one recipe. Fitting a monotone (isotonic) curve through their (accuracy, mean solve) pairs gives $\\hat{s}(\\text{acc})$, the mean solve rate a cell of that accuracy is expected to reach.
- A cell's **residual** is how far it beats or misses that expectation:

$$r = \\bar{s} - \\hat{s}(\\text{acc})$$

  with $\\bar{s}$ the cell's own mean solve rate, averaged over the 14 densities.
- The **leave-one-out bar** is how well the curve predicts a ladder cell it was not fitted on: refit without cell $i$, predict that cell, and take the root mean square of those errors across the ladder. It is **0.0423**, and it is the yardstick a residual is read against.
- Why this page exists: a residual inside the bar means an arm changed *where* a cell sits on the existing relation, not the relation itself. Only a residual outside the bar is evidence that something architectural happened.
"""

LEVELS = """## Train NLL settles at one of three levels: plateau, floor, basin

- Cells in this family do not land on a continuum. Training ends at one of three places, and which one is reached matters more than any architectural knob tested so far:
- **Plateau**, above 1.600: the cell left its initial collapse and then stopped. One cell on disk ended at 1.6334 and never moved again.
- **Floor**, at or below 1.600, typically 1.585-1.592: where the recipe's healthy seeds land. Tight and reproducible.
- **Basin**, at or below 1.561: below the floor, and rare. Two of the cells that reached it are the best $\\rho_{50}$ seeds the tied family has produced, which is why the level is tracked at all. Reaching it is not by itself a better model: the association is what makes the level worth watching, and it is not a guarantee.
- A **step** is a move between levels: a one-epoch fall out of a flat stretch that keeps falling. It happens at one epoch rather than gradually, and it has been the event most associated with a cell ending up good.
"""

LEAK = """## Every eval shuffles the adjacency so the answer cannot leak

- The graph is written into the prompt as a list of edges. If node ids appeared in a fixed order correlated with the planted colouring, a model could read the answer off the token order instead of deducing it.
- Every evaluation here runs with `shuffle_adj=True`, which permutes the adjacency presentation per instance, so that channel is closed.
- The flag is taken from each checkpoint's own stored config rather than passed in at eval time, and the scoring job asserts it is `True` before reporting a number -- so a cell evaluated the wrong way fails the run instead of quietly producing a better-looking score.
- All numbers in this deck are on the same leak-free grid, which is what makes the before/after comparison a comparison at all.
"""

BASELINES = """## The two reference points: J2 at 0.580, S1 at 0.757

| cell | architecture | trainable | $\\rho_{50}$ |
|---|---|---|---|
| **J2** | untied, 16 distinct blocks, $d_{\\text{model}}=192$ | 2,504,890 | **0.580** |
| **S1** | tied, one block applied 16 times, $d_{\\text{model}}=320$ | 613,818 | 0.757 |

- **J2** is the standing minimum: the best $\\rho_{50}$ any cell in the project has reached. It is four times the parameters and, being 16 different maps composed, is the thing a theory would struggle with.
- **S1** is the control every stage is compared against: the simple tied object, one quarter the size, sitting 0.017 above the 0.740 gate and 0.177 above J2.
- Closing that distance **without** untying the block is the whole point. An arm that improves the number by reintroducing distinct per-layer weights would not be progress toward what this project is for.
"""

PAGES = [TASK, HINTS, DEDUCTION, FIGURE_SLOT, GENERATION, ARCHITECTURE,
         SOLVE_RATE, RHO50, CENSOR, GATE, ACCURACY, SUFFICIENCY, LEVELS, LEAK,
         BASELINES]


def check():
    """Assert every number in the prose against the scorer that owns it.

    Imports lazily: the login node's python3 is 3.6.8 without numpy, so a
    py_compile of this module has to work without the scorer being importable.
    The deck job calls this with the project's modules loaded.
    """
    from scripts.stagew_score import FREE, LOO_BAR, S1_RHO50, S1_TRAINABLE

    body = "".join(PAGES)
    failures = []
    for text, where in (("613,818", "ARCHITECTURE and BASELINES"),
                        ("0.740", "GATE"),
                        ("0.757", "GATE and BASELINES"),
                        ("0.580", "GATE and BASELINES"),
                        ("0.0423", "SUFFICIENCY")):
        if text not in body:
            failures.append("%s missing from %s" % (text, where))
    if "{:,}".format(S1_TRAINABLE) != "613,818":
        failures.append("scorer says S1 is %d trainable, prose says 613,818"
                        % S1_TRAINABLE)
    if "%.3f" % FREE != "0.740":
        failures.append("scorer gate is %.4f, prose says 0.740" % FREE)
    # S1_RHO50 is carried as a STRING ("0.757") in stagev_score, so that a
    # censored cell can be reported in the same field as a number. Coerce here
    # rather than assuming either type.
    s1 = float(S1_RHO50)
    if "%.3f" % s1 != "0.757":
        failures.append("scorer says S1 rho_50 is %.4f, prose says 0.757" % s1)
    if "%.3f" % (s1 - FREE) != "0.017":
        failures.append("S1 is %.4f above the gate, prose says 0.017" % (s1 - FREE))
    if "%.4f" % LOO_BAR != "0.0423":
        failures.append("the leave-one-out bar is %.4f, prose says 0.0423" % LOO_BAR)
    if "\\mathbb{1}" in body:
        failures.append("\\mathbb{1} is not in KaTeX; use \\mathbf{1}")
    for bad in ("\\(", "\\[", "\\bm", "\\mathscr", "\\mathbbm", "\\eqref"):
        if bad in body:
            failures.append("%s does not render in this pipeline" % bad)
    for page in PAGES:
        if page is FIGURE_SLOT:
            continue
        title = page.splitlines()[0]
        if not title.startswith("## "):
            failures.append("page does not open with a '## ' title: %r" % title[:40])
        elif len(title[3:].split()) > 12:
            failures.append("title over 12 words: %s" % title)
    if failures:
        raise AssertionError("deck_explain.check() failed:\n  "
                             + "\n  ".join(failures))
    return len([p for p in PAGES if p is not FIGURE_SLOT])


if __name__ == "__main__":
    print("%d explanation pages, %d figure slot(s)"
          % (len([p for p in PAGES if p is not FIGURE_SLOT]),
             PAGES.count(FIGURE_SLOT)))
    try:
        n = check()
        print("check() passed on %d pages" % n)
    except ImportError as exc:
        print("check() skipped, scorer not importable here: %s" % exc)
