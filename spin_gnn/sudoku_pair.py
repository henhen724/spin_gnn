"""Stage PAIR's per-instance edit: the four integers the counterfactual margin
needs, and nothing else.

WHAT THIS IS FOR. Finding 78 measured that stage Y's determinacy channel holds a
partial predicate that is not causally wired: within a stratum of fixed
revealed-neighbour count it ranks nodes by determinacy at AUROC 0.8059/0.8021/
0.8019, and on a MATCHED pair -- one prompt and the same prompt with v's forcing
destroyed by a single hint recolour -- it orders the two at 0.4457/0.5714/0.4543
against a placebo's 0.4800/0.4686/0.4714. Chance, both. The median |delta g|
under the corruption is 0.0103 where the placebo's is 0.0102 and g's own spread
is 3.5. Stage X and Y only ever supervised a ranking over nodes on CLEAN inputs;
neither objective ever asked g to move when the property it names stops holding.

This module supplies the index for a term that asks exactly that:

    L_pair = max(0, m - (g_clean[p, v] - g_broken[p, v]))

THE STORAGE, and why it is four integers. build_triple (sudoku_skip.py:175) is
clean.copy() with ONE index written -- no solver call, no second instance
generated -- so a corrupted prompt costs one token write at batch time and does
not need to be stored. What has to be stored per instance is

    (edit position, edit TOKEN, predict position, node INDEX)

THE TWO ID SPACES ARE DIFFERENT AND MIXING THEM IS SILENT. The edit is written
into the token stream, so it is a vocabulary id (vocab.color_token). The node is
read out of g, whose last axis is the node block ALREADY selected -- n_nodes
wide, no node_offset, see determinacy_loss_from_channel -- so it is a node index
in [0, n_nodes). A vocabulary id put here is < n_nodes for the low nodes and
would score the WRONG NODE at identical cost rather than raising. That mistake
was made once already, in the price script, and caught before the job reported.

CASE SELECTION IS STAGE Z'S, DELIBERATELY. find_unforceable_deductions is called
unchanged, including its requirement that a placebo exist -- a hint that is not
v's neighbour and whose colour can be changed while keeping the revealed
colouring proper. Training never uses the placebo. The requirement is kept
because it is what makes the training population and the probe's population the
same construction, and because the measured yield (0.8034 of instances, 16,068
of 20,000 at seed 9e8) was measured with it in place.

WHAT A MISS LOOKS LIKE. About a fifth of instances carry no case at all: the
round-0 deduction may be forced by a colour no single hint holds, or every hint
that holds it is itself a deduction, or no proper placebo exists. Those
instances get PAIR_NONE and contribute to the main loss alone. A sentinel rather
than a filtered subset, so the training set, its batch order and its NLL stay
exactly the control's.
"""
import numpy as np

# edit position -1 means "this instance carries no pair". The trainer selects on
# it; the collate pads with it, so a padded row is a miss for free.
PAIR_NONE = (-1, -1, -1, -1)
PAIR_WIDTH = 4


def pair_entry(adj, colors, hints, trace, vocab, tokens, rng):
    """The four integers for one instance, or PAIR_NONE.

    Every assertion here is a convention being pinned against the actual token
    stream rather than trusted from a position formula: hint_color_pos is
    computed as 1 + 2*rank + 1 from the hint ORDER, and a vocabulary whose
    header grows by one token would move every hint colour without changing the
    arithmetic.
    """
    # Imported here, not at module scope: sudoku_skip imports
    # train_sudoku_transformer for build_instances, and the trainer imports this
    # module, so a top-level import would close the cycle.
    from .sudoku_skip import find_unforceable_deductions, node_token_position

    cases = find_unforceable_deductions(adj, colors, hints, trace, vocab, rng,
                                        max_per_instance=1)
    if not cases:
        return PAIR_NONE
    c = cases[0]
    epos = int(c["hint_color_pos"])
    etok = int(vocab.color_token(c["new_color"]))
    node = int(c["node"])
    ppos = int(node_token_position(tokens, vocab, node))

    assert vocab.is_color_token(tokens[epos]), (
        "edit position %d is not a hint colour token (%s); the hint header's "
        "layout moved" % (epos, vocab.name(int(tokens[epos]))))
    assert int(vocab.color_index(tokens[epos])) == int(c["freed_color"]), (
        "the token at the edit position is not the colour the case frees",
        int(vocab.color_index(tokens[epos])), c["freed_color"])
    # A no-op edit would make g_broken == g_clean identically and the hinge a
    # constant the model cannot reduce. b != a by construction; asserted because
    # a silent no-op is indistinguishable from an unlearnable contrast.
    assert int(tokens[epos]) != etok, ("the edit does not change the token",
                                       epos, etok)
    assert 0 <= node < vocab.n_nodes, (node, vocab.n_nodes)
    assert vocab.is_node_token(tokens[ppos + 1]), (
        "the token after the predict position is not a node token", ppos)
    assert int(vocab.node_index(tokens[ppos + 1])) == node, (
        "the predict position names a different node", ppos, node)
    # SATTokenDataset.__getitem__ returns seq[:-1], seq[1:], so x is one token
    # shorter than the sequence and a position at the very end would index past
    # it. Both are near the front in practice; checked because an out-of-range
    # advanced index on a GPU tensor is not always loud.
    n_x = len(tokens) - 1
    assert 0 <= epos < n_x and 0 <= ppos < n_x, (epos, ppos, n_x)
    return (epos, etok, ppos, node)


def build_pair_index(instances, token_seqs, vocab, seed):
    """One entry per instance, parallel to token_seqs, PAIR_NONE where there is
    no case.

    `seed` drives only the placebo search's permutation, which training never
    reads; it is passed so the scan is reproducible rather than because the
    choice matters.
    """
    if len(instances) != len(token_seqs):
        raise ValueError("%d instances for %d sequences"
                         % (len(instances), len(token_seqs)))
    rng = np.random.default_rng(seed)
    return [pair_entry(adj, colors, hints, trace, vocab, tok, rng)
            for (adj, colors, hints, trace), tok in zip(instances, token_seqs)]


def summarize_pairs(pairs):
    """Yield and the spread of the two positions, for the run's own log."""
    n = len(pairs)
    hit = [p for p in pairs if p[0] >= 0]
    if not hit:
        return {"n": n, "n_pairs": 0, "yield": 0.0}
    epos = np.array([p[0] for p in hit])
    ppos = np.array([p[2] for p in hit])
    return {
        "n": n, "n_pairs": len(hit), "yield": len(hit) / n,
        "edit_pos_min": int(epos.min()), "edit_pos_max": int(epos.max()),
        "predict_pos_min": int(ppos.min()), "predict_pos_max": int(ppos.max()),
        "predict_pos_mean": float(ppos.mean()),
    }
