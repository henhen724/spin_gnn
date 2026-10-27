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
distribution the model was not trained on.

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
                     shuffle_adj=True, seed=50_000_000_000, pick="median"):
    """One instance from the eval generator. `pick` selects among 40 draws:
    "median" takes the median number of DD rounds, so the deck shows a typical
    instance rather than a cherry-picked shallow one; "deep" takes the 90th
    percentile, for decks whose subject IS reasoning depth (a T=3 trace does
    not illustrate a depth cost). Either way the depth and the pool it came
    from are printed on the page, so the choice is visible."""
    cfg = SudokuReasoningConfig(n_nodes=n_nodes, degree=degree, n_colors=n_colors,
                                rho=rho, rho_max=rho_max, hint_mode="removal",
                                scratchpad=False, shuffle_adj=shuffle_adj)
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
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "What the model actually sees", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             f"Every token is one categorical id into a {ex.vocab.size}-symbol vocabulary "
             f"(N={ex.cfg.n_nodes}, q={ex.cfg.n_colors}, degree={ex.cfg.degree}). "
             f"No text, no digits-as-characters.", size=SMALL)

    v = ex.vocab
    rows = [("0-5", "BEGIN_HINTS BEGIN_GRAPH BEGIN_REASONING", ""),
            ("", "BEGIN_ROUND SOLVED UNRESOLVED", "section + round markers"),
            (f"{v.COLOR_0}-{v.COLOR_0 + v.n_colors - 1}",
             " ".join(f"COLOR_{c}" for c in range(v.n_colors)), "the q colours"),
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
        ax.text(0.0, 0.80 - 0.17 * i, g, size=SMALL - 1, va="top", **MONO)
    ax.text(0.0, 0.08,
            "The graph section is a neighbour table, not an image: node, then its\n"
            "degree neighbours. THE ORDER WITHIN A ROW IS THE LEAK of finding 16.",
            size=SMALL - 2, va="top", color="0.4", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = fig.add_axes([0.06, 0.11, 0.90, 0.31]); ax.axis("off")
    ax.text(0, 1.0, f"THIS INSTANCE, {len(ex.tokens)} TOKENS (seed {ex.seed})",
            size=SMALL, weight="bold", va="top")
    y = 0.80
    for name, start, end in ex.section_bounds():
        shown = ex.names[start:min(end, start + 13)]
        tail = f"   ... ({end - start} tokens)" if end - start > 13 else ""
        ax.text(0.0, y, name, size=SMALL - 1, weight="bold", va="top")
        ax.text(0.13, y, " ".join(shown) + tail, size=SMALL - 2, va="top", **MONO)
        y -= 0.20
    ax.text(0.0, y - 0.02,
            f"Loss is on the REASONING section only: {ex.n_rounds} rounds, "
            f"{ex.n_deductions} deductions, {2 * ex.n_deductions} target tokens. "
            f"Everything before BEGIN_REASONING is the problem statement.",
            size=SMALL - 1, va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    footer(fig, f"spin_gnn.sudoku_tokens.Vocab + encode_instance; "
                f"shuffle_adj={ex.cfg.shuffle_adj}")
    out.savefig(fig)


def page_trace(out, ex, max_rounds=4):
    """A real trace, with one deduction worked through so the task is legible."""
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "One real reasoning trace", size=BIG, weight="bold")
    hints = sorted(ex.trace["hints"].items())
    hint_str = " ".join(f"N{n}={c}" for n, c in hints[:8]) + (" ..." if len(hints) > 8 else "")
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
        lines = [
            f"node N{node}, neighbours {list(ex.adj[node])}",
            f"colours known before this round:",
            f"   {sorted(known)}  ({len(known)} = q-1)",
            f"only colour left: {missing[0]}",
            f"so the trace emits  N{node} COLOR_{ded}",
        ]
        for i, line in enumerate(lines):
            ax.text(0.0, 0.84 - 0.15 * i, line, size=SMALL - 2, va="top",
                    **(MONO if i in (2, 4) else {}))
        ax.text(0.0, 0.84 - 0.15 * len(lines) - 0.02,
                "The model is never shown this reasoning --\nonly the node/colour pair.",
                size=SMALL - 3, va="top", color="0.4", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.text(0.06, 0.36,
             "A round deduces every node whose neighbours already cover q-1 colours, all at once. "
             "Reasoning depth T is the\nnumber of rounds: a deduction in round 3 depends on ones made "
             "in rounds 1 and 2, which is the serial computation\nthat network depth has to supply.",
             size=SMALL, va="top")
    pool = getattr(ex, "pool_depths", None)
    how = {"median": "the median", "deep": "the 90th percentile",
           "first": "the first"}.get(getattr(ex, "pick", ""), "one")
    prov = (f"T={ex.n_rounds} rounds, {how} of 40 draws from this config "
            f"(range {min(pool)}-{max(pool)}); " if pool else "")
    footer(fig, f"{prov}seed {ex.seed}, shuffle_adj={ex.cfg.shuffle_adj}; "
                f"evidence from sudoku_tokens.evidence_rounds (pre-round state, as DD saw it)")
    out.savefig(fig)
