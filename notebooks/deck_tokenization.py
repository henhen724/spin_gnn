"""Deck pages that show WHAT THE MODEL IS ACTUALLY FED: the tokenization and a
real reasoning trace.

hshunt reads results as PDFs with no notebook and no code to hand, so a deck of
solve_rate tables is unauditable on its own -- and on this project that is not
hypothetical. The headline 0.9960 stood for weeks before 49655447 showed it was
decoding the generator's TOKEN ORDER (adj[v] arrived sorted by the neighbour's
true colour), a fact about the tokenization that no metrics page could ever
have revealed. Every deck from 2026-10-26 on carries these two pages.

Both pages take a real instance from the same generator and config as the run
being reported, never a schematic: `example_instance(shuffle_adj=...)` must be
passed the SAME shuffle_adj as the checkpoint, or the deck will illustrate a
distribution the model was not trained on. The same goes for `shuffle_events`,
which permutes the order of the deductions WITHIN a round: it leaves the token
multiset of the reasoning section alone, so only the trace page shows it, but
that page is exactly where a reader would otherwise read the generator's
ascending-index convention as something the model produced.

Usage, from a deck script:

    from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
    ex = example_instance(n_nodes=24, degree=5, n_colors=4, shuffle_adj=True)
    page_tokenization(out, ex)
    page_trace(out, ex)
"""
import matplotlib.pyplot as plt
import numpy as np

from notebooks.deck_utils import BIG, MED, SMALL, footer
from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.sudoku_tokens import Vocab, encode_instance, evidence_rounds
from spin_gnn.train_sudoku_transformer import build_instances

MONO = {"family": "monospace"}


class Example:
    """One instance plus everything needed to narrate it."""

    def __init__(self, adj, colors, hints, trace, vocab, tokens, cfg, seed):
        self.adj, self.colors, self.hints, self.trace = adj, colors, hints, trace
        self.vocab, self.tokens, self.cfg, self.seed = vocab, tokens, cfg, seed
        self.names = [vocab.name(int(t)) for t in tokens]
        self.evidence = evidence_rounds(adj, trace)

    @property
    def n_rounds(self):
        return len(self.trace["rounds"])

    @property
    def n_deductions(self):
        return sum(len(r) for r in self.trace["rounds"])

    def section_bounds(self):
        """(name, start, end) per grammar section, found from the stream itself
        rather than recomputed from the config, so a grammar change shows up
        here instead of being silently mis-annotated."""
        v, out = self.vocab, []
        marks = [("HINTS", v.BEGIN_HINTS), ("GRAPH", v.BEGIN_GRAPH),
                 ("REASONING", v.BEGIN_REASONING)]
        idx = {name: int(np.where(self.tokens == tok)[0][0]) for name, tok in marks}
        order = sorted(idx.items(), key=lambda kv: kv[1])
        for i, (name, start) in enumerate(order):
            end = order[i + 1][1] if i + 1 < len(order) else len(self.tokens)
            out.append((name, start, end))
        return out


def example_instance(n_nodes=24, degree=5, n_colors=4, rho=0.3, rho_max=0.95,
                     shuffle_adj=True, shuffle_events=False,
                     seed=50_000_000_000, pick="median"):
    """One instance from the eval generator. `pick` selects among 40 draws:
    "median" takes the median number of DD rounds, so the deck shows a typical
    instance rather than a cherry-picked shallow one; "deep" takes the 90th
    percentile, for decks whose subject IS reasoning depth (a T=3 trace does
    not illustrate a depth cost). Either way the depth and the pool it came
    from are printed on the page, so the choice is visible."""
    cfg = SudokuReasoningConfig(n_nodes=n_nodes, degree=degree, n_colors=n_colors,
                                rho=rho, rho_max=rho_max, hint_mode="removal",
                                scratchpad=False, shuffle_adj=shuffle_adj,
                                shuffle_events=shuffle_events)
    insts = build_instances(cfg, 40, seed_start=seed)
    depths = [len(t["rounds"]) for _, _, _, t in insts]
    order = np.argsort(depths)
    i = int(order[{"median": len(order) // 2, "deep": int(0.9 * (len(order) - 1)),
                   "first": 0}[pick]]) if pick != "first" else 0
    adj, colors, hints, trace = insts[i]
    vocab = Vocab(n_nodes, n_colors, degree, scratchpad=False)
    tokens = encode_instance(adj, trace, vocab)
    ex = Example(adj, colors, hints, trace, vocab, tokens, cfg, seed + i)
    ex.pool_depths, ex.pick = depths, pick
    return ex


def page_tokenization(out, ex):
    """The vocabulary, the grammar, and the real stream segmented by section."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.06, 0.93, "What the model actually sees", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             f"Every token is one categorical id into a {ex.vocab.size}-symbol vocabulary "
             f"(N={ex.cfg.n_nodes}, q={ex.cfg.n_colors}, degree={ex.cfg.degree}). "
             f"No text, no digits-as-characters.", size=SMALL)

    v = ex.vocab
    rows = [("0-5", "BEGIN_HINTS BEGIN_GRAPH BEGIN_REASONING", ""),
            ("", "BEGIN_ROUND SOLVED UNRESOLVED", "markers (UNRESOLVED = DD got stuck; unused here)"),
            (f"{v.COLOR_0}-{v.COLOR_0 + v.n_colors - 1}",
             " ".join(f"COLOR_{c}" for c in range(v.n_colors)),
             "the q colours (token name COLOR_n, prose 'colour')"),
            (f"{v.node_offset}-{v.size - 1}",
             f"N0 N1 ... N{v.n_nodes - 1}", "one id per node, no structure in the id")]
    ax = fig.add_axes([0.06, 0.50, 0.42, 0.31]); ax.axis("off")
    ax.text(0, 1.0, "VOCABULARY", size=SMALL, weight="bold", va="top")
    y = 0.80
    for ids, syms, note in rows:
        ax.text(0.0, y, ids, size=SMALL - 1, va="top", **MONO)
        ax.text(0.17, y, syms, size=SMALL - 1, va="top", **MONO)
        if note:
            ax.text(0.17, y - 0.1, note, size=SMALL - 2, va="top", color="0.4", style="italic")
        y -= 0.26 if note else 0.14
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = fig.add_axes([0.54, 0.50, 0.42, 0.31]); ax.axis("off")
    ax.text(0, 1.0, "GRAMMAR", size=SMALL, weight="bold", va="top")
    grammar = ["BEGIN_HINTS      (NODE COLOR)*",
               "BEGIN_GRAPH      (NODE (NODE){degree})*",
               "BEGIN_REASONING  (BEGIN_ROUND (NODE COLOR)+)*",
               "SOLVED"]
    for i, g in enumerate(grammar):
        ax.text(0.0, 0.84 - 0.155 * i, g, size=SMALL - 1, va="top", **MONO)
    ax.text(0.0, 0.26, "regex: * = zero or more, + = one or more, {degree} = exactly "
                       f"{ex.cfg.degree}",
            size=SMALL - 2, va="top", color="0.4", style="italic")
    ax.text(0.0, 0.15,
            "The graph section is a neighbour table, not an image:\nnode, then its "
            f"{ex.cfg.degree} neighbours.",
            size=SMALL - 1, va="top")
    # Black bold, not red: red keys MODEL IDENTITY on every other page of the
    # decks this module feeds, and prose emphasis in the same red made "red"
    # mean both "the leaked model" and "look here".
    ax.text(0.0, -0.02, "THE ORDER WITHIN A ROW IS THE LEAK.",
            size=SMALL + 1, va="top", weight="bold", color="0.0")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = fig.add_axes([0.06, 0.11, 0.90, 0.31]); ax.axis("off")
    # The seed is provenance, not content: it moved to the footer.
    ax.text(0, 1.0, f"THIS INSTANCE, {len(ex.tokens)} TOKENS",
            size=SMALL, weight="bold", va="top")
    y = 0.80
    for name, start, end in ex.section_bounds():
        shown = ex.names[start:min(end, start + 13)]
        tail = f"   ... ({end - start} tokens)" if end - start > 13 else ""
        ax.text(0.0, y, name, size=SMALL - 1, weight="bold", va="top")
        ax.text(0.13, y, " ".join(shown) + tail, size=SMALL - 2, va="top", **MONO)
        y -= 0.20
    # NOT "loss is on the REASONING section only". sat_dataset2.__getitem__
    # returns (seq[:-1], seq[1:]) and the collate mask is a PADDING mask
    # (mask[i, :length] = True) -- there is no section mask anywhere, so the
    # loss covers the problem statement too. It is the EVAL that is sectioned:
    # train_sudoku_transformer prompts through BEGIN_REASONING and generates.
    # This also settles why page 10's BEGIN_ROUND outcome means anything: it is
    # a trained target like every other token.
    ax.text(0.0, y - 0.02,
            "TRAINING LOSS is next-token cross-entropy on EVERY token, the problem statement included -- "
            "there is no section mask.\n"
            "AT EVAL the model is PROMPTED with everything through BEGIN_REASONING and generates the rest: "
            f"{ex.n_rounds} rounds, {ex.n_deductions} deductions,\n"
            # There is no END_REASONING token: `grep -rn END_REASONING spin_gnn/*.py`
            # finds nothing, and SPECIAL_BASE_NAMES in sudoku_tokens.py is exactly
            # the six this page's VOCABULARY block lists. The sequence terminates
            # with SOLVED or UNRESOLVED, as the GRAMMAR block beside it shows.
            f"{2 * ex.n_deductions} node+colour tokens, plus the BEGIN_ROUND markers and the SOLVED/UNRESOLVED "
            "terminator, which are trained targets like any other token.",
            size=SMALL - 1, va="top", linespacing=1.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    footer(fig, "DD is single-candidate elimination -- sudoku's 'naked single' rule transplanted to graph "
                "colouring, which is where the code's sudoku names come from; the task itself is the graph, "
                f"not a 9x9 grid. Instance shown: seed {ex.seed}."
                + ("" if ex.cfg.shuffle_adj else " These rows are the LEAKED (colour-sorted) order."))
    out.savefig(fig)


def page_trace(out, ex, max_rounds=4, title="One real reasoning trace", note=None):
    """A real trace, with one deduction worked through so the task is legible.

    `title` and `note` exist for decks that show this page TWICE, once per event
    order, where the subject of the run is the difference between the two pages:
    two identically titled traces would be unreadable as a comparison."""
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.06, 0.93, title, size=BIG, weight="bold")
    hints = sorted(ex.trace["hints"].items())
    hint_str = " ".join(f"N{n}=COLOR_{c}" for n, c in hints[:6]) + (" ..." if len(hints) > 6 else "")
    fig.text(0.06, 0.875,
             f"{len(hints)} hints of {ex.cfg.n_nodes} nodes, then "
             f"{ex.n_rounds} rounds of simultaneous deduction. Hints: {hint_str}", size=SMALL)

    ax = fig.add_axes([0.06, 0.44, 0.52, 0.38]); ax.axis("off")
    ax.text(0, 1.0, "THE TRACE, AS EMITTED", size=SMALL, weight="bold", va="top")
    # 4 rounds + the elision line + SOLVED is 6 rows, so the spacing has to
    # divide the axes rather than be chosen per row, or SOLVED lands below the
    # axes and collides with the paragraph underneath.
    y, dy = 0.88, 0.145
    for r, events in enumerate(ex.trace["rounds"][:max_rounds]):
        # Count goes in its own left column: at 6 deductions the token string is
        # wide enough to run under a right-hand annotation.
        ax.text(0.0, y, "BEGIN_ROUND", size=SMALL - 2, va="top", color="#1f77b4", **MONO)
        ax.text(0.0, y - 0.075, f"  {len(events)} deduced", size=SMALL - 3, va="top",
                color="0.45")
        ax.text(0.26, y, " ".join(f"N{n} COLOR_{c}" for n, c in events[:4])
                + (" ..." if len(events) > 4 else ""), size=SMALL - 2, va="top", **MONO)
        y -= dy
    if ex.n_rounds > max_rounds:
        ax.text(0.0, y, f"... {ex.n_rounds - max_rounds} further rounds", size=SMALL - 2,
                va="top", color="0.45")
        y -= dy
    ax.text(0.0, y, "SOLVED", size=SMALL - 2, va="top", color="#2ca02c", **MONO)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # One deduction, worked. Pick the first whose evidence is a full q-1 set, which
    # is the DD firing rule itself and the thing the model has to learn.
    node, nbr_cols, ded = None, None, None
    for events in ex.evidence:
        for n, cols, c in events:
            if len({x for x in cols if x is not None}) == ex.cfg.n_colors - 1:
                node, nbr_cols, ded = n, cols, c
                break
        if node is not None:
            break

    ax = fig.add_axes([0.63, 0.44, 0.33, 0.38]); ax.axis("off")
    ax.text(0, 1.0, "WHY ONE DEDUCTION FIRES", size=SMALL, weight="bold", va="top")
    if node is None:
        ax.text(0, 0.8, "no q-1 evidence row in this instance", size=SMALL - 2, va="top")
    else:
        known = {c for c in nbr_cols if c is not None}
        missing = sorted(set(range(ex.cfg.n_colors)) - known)
        # Name the still-unknown neighbours explicitly. Listing five neighbours
        # and then three colours leaves the audience counting to work out which
        # two were blank, which is exactly where a room stalls.
        unk = [x for x, c in zip(ex.adj[node], nbr_cols) if c is None]
        n_known_nbrs = sum(c is not None for c in nbr_cols)
        # Why the counts do not match: with degree 5 and q = 4 the known
        # neighbours usually repeat a colour, and a room that is counting
        # "5 neighbours, 3 colours" stalls on exactly that line.
        dup = (f"   ({n_known_nbrs} known neighbours, {len(known)} DISTINCT colours"
               + (" -- some repeat)" if n_known_nbrs > len(known) else ")"))
        lines = [
            f"node N{node}, neighbours "
            + " ".join(f"N{x}" for x in ex.adj[node]),
            f"their colours known before this round:",
            f"   {sorted(known)}  ({len(known)} = q-1 of the {ex.cfg.n_colors})",
            dup,
            "   still unknown: " + (" ".join(f"N{x}" for x in unk) if unk else "none"),
            f"only colour left: {missing[0]}",
            f"so the trace emits  N{node} COLOR_{ded}",
        ]
        # 0.115, not 0.13: seven lines at 0.13 push the italic note below the
        # axes and into the paragraph underneath.
        for i, line in enumerate(lines):
            ax.text(0.0, 0.84 - 0.115 * i, line, size=SMALL - 2, va="top",
                    **(MONO if i in (2, 3, 4, 6) else {}))
        ax.text(0.0, 0.84 - 0.115 * len(lines) - 0.02,
                "The model is never shown this reasoning --\nonly the node/colour pair.",
                size=SMALL - 3, va="top", color="0.4", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.text(0.06, 0.36,
             "A round deduces every node whose neighbours already cover q-1 colours, all at once. "
             "Reasoning depth T is the\nnumber of rounds: a deduction in round 3 depends on ones made "
             "in rounds 1 and 2, which is the serial computation\nthat network depth has to supply.",
             size=SMALL, va="top")
    if note:
        # black bold, not red -- see the note on prose emphasis above
        fig.text(0.06, 0.24, note, size=SMALL - 1, va="top", weight="bold")
    pool = getattr(ex, "pool_depths", None)
    how = {"median": "the median", "deep": "the 90th percentile",
           "first": "the first"}.get(getattr(ex, "pick", ""), "one")
    prov = (f"T={ex.n_rounds} rounds, {how} of 40 draws from this config "
            f"(range {min(pool)}-{max(pool)}) -- one instance, shown to make the task legible; no "
            "result in this deck is computed from it. " if pool else "")
    footer(fig, f"{prov}The worked deduction on the right uses the state BEFORE this round, which is "
                f"what DD itself saw; it is recomputed from the graph, not read out of the model.")
    out.savefig(fig)
