"""Directional ablation in the residual stream, scored on rho_50.

WHY THIS EXISTS. Finding 46 established, with linear probes, that the N=96
fall models do represent DD's deducibility predicate (`frontier_mask` AUROC
0.731-0.772 against a label-only free score of 0.6734) and that the strength
of that representation is uncorrelated with how well the cell actually solves
(Spearman -0.086 with rho_50), while `next_node` top-1 is almost perfectly
anti-correlated with it (-0.943). A probe only shows that information is
present and linearly readable. It cannot show the model USES it. This module
is the causal half: delete a subspace from the stream and re-measure rho_50
on the same eval.

WHERE THE DELETION HAPPENS, AND WHY NOT JUST THE LAST LAYER. In a causal
decoder the last block's output is read only by the unembedding -- no later
layer attends to it -- so ablating there changes logits and nothing else. It
would answer "is this direction part of the readout", not "does the model use
this information while computing". So the default is to ablate at EVERY block
output from --ablate-from onward, each layer with its OWN probe's subspace, so
the information is removed from the stream persistently and the downstream
layers have to work without it. The readback then reports whether the model
recomputes it anyway, which is the standard incompleteness caveat and worth
measuring rather than assuming.

THE DIRECTION. fit_linear_probe standardizes its inputs, so a probe row acts
on (x - mu) / sigma; the direction it reads in the RAW residual stream is
therefore `weight / sigma`, which is what raw_directions returns. The probe
weight is a matrix, not a vector -- (n_nodes, d_model) for the per-node
targets -- so "the direction" is a subspace: the top-k right singular vectors
of that matrix, which is the k-dimensional subspace the probe's outputs are
most sensitive to.

THE CONTROLS ARE THE EXPERIMENT. A single ablation that hurts proves nothing:
removing any k dimensions from a 256-dimensional stream hurts. Every condition
removes the SAME k dimensions at the SAME layers, and the comparison is
between conditions:

  ablate_next_node   the selection step. Expected to be catastrophic; it is
                     the positive control that the hook bites at all.
  ablate_frontier    the DD predicate. THE TEST.
  ablate_next_color  a second output-adjacent target, so "it was near the
                     readout" cannot by itself explain a next_node effect.
  ablate_random      a count-matched random orthonormal subspace, resampled
                     per layer. The magnitude floor: a condition that does not
                     beat this did nothing causal.
"""
import numpy as np
import torch

from .sudoku_activations import collect_dataset
from .sudoku_probes import auroc, average_precision, fit_linear_probe

# The targets worth ablating. remaining_mask is left out: it is the complement
# of known_mask up to the hint set, so it is not an independent condition.
ABLATION_TARGETS = [
    ("next_node", "categorical"),
    ("frontier_mask", "multilabel"),
    ("next_color", "categorical"),
]


def raw_directions(pack):
    """The probe's read directions in the raw residual stream: (out_dim, d)."""
    return pack["weight"] / pack["sigma"][None, :]


def top_k_basis(w_raw, k):
    """Orthonormal (k, d) basis for the top-k right singular subspace.

    The right singular vectors of the read matrix are the directions in
    activation space that move the probe's outputs most, ordered. k is capped
    at the matrix rank, so a target with fewer outputs than k (there are none
    at present, but next_color has only n_colors rows) cannot silently pad the
    basis with numerical noise.
    """
    k = min(k, min(w_raw.shape))
    _u, _s, vt = np.linalg.svd(np.asarray(w_raw, dtype=np.float64), full_matrices=False)
    return vt[:k]


def random_basis(d, k, rng):
    """A random orthonormal (k, d) basis -- the count-matched floor."""
    q, _r = np.linalg.qr(rng.standard_normal((d, k)))
    return q.T[:k]


def spectrum_fraction(w_raw, k):
    """Fraction of the read matrix's squared Frobenius norm inside the top-k.

    How much of the probe this ablation actually removes. A k that captures
    30% of the spectrum and a k that captures 95% are different experiments,
    and the number belongs in the output rather than in a footnote.
    """
    s = np.linalg.svd(np.asarray(w_raw, dtype=np.float64), compute_uv=False)
    k = min(k, len(s))
    return float((s[:k] ** 2).sum() / max(1e-30, (s ** 2).sum()))


class SubspaceAblation:
    """Project a per-layer subspace out of each block's output.

    bases: {layer_index -> (k, d) orthonormal array}, with the SAME 1-based
    indexing the probes use -- layer i is the output of encoder block i-1, and
    layer 0 (the embedding) cannot be hooked here, so a key of 0 is rejected
    rather than silently ignored.

    Entering with an empty dict is the `clean` condition and installs no
    hooks, so the control path runs the identical code.
    """

    def __init__(self, model, bases, device="cpu"):
        self.model = model
        n_layers = len(model.encoder.layers)
        bad = [i for i in bases if not 1 <= i <= n_layers]
        if bad:
            raise ValueError(f"layer indices {bad} outside 1..{n_layers}; layer 0 is the "
                             f"embedding and is not a block output")
        self.bases = {i: torch.as_tensor(b, dtype=torch.float32, device=device)
                      for i, b in bases.items()}
        self._handles = []

    def __enter__(self):
        def mk(basis):
            def hook(_mod, _inp, out):
                # out: (B, T, d). Remove the component inside span(basis).
                return out - (out @ basis.T) @ basis
            return hook

        for i, block in enumerate(self.model.encoder.layers):
            basis = self.bases.get(i + 1)
            if basis is not None:
                self._handles.append(block.register_forward_hook(mk(basis)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def score_stored_probe(pack, x, y, kind):
    """Score a probe fit on CLEAN activations against (possibly ablated) x.

    This is the validity read-back: an ablation that does not move its own
    probe's score did not remove the thing it was aimed at, and whatever the
    rho_50 curve then shows is about something else.
    """
    xs = (np.asarray(x, dtype=np.float64) - pack["mu"][None, :]) / pack["sigma"][None, :]
    logits = xs @ pack["weight"].T + pack["bias"][None, :]
    if kind == "multilabel":
        s, yy = logits.ravel(), np.asarray(y, dtype=np.float64).ravel()
        return {"auroc": auroc(s, yy), "average_precision": average_precision(s, yy),
                "top1": float("nan")}
    pred = logits.argmax(axis=1)
    return {"auroc": float("nan"), "average_precision": float("nan"),
            "top1": float((pred == np.asarray(y)).mean())}


def fit_probes_all_layers(model, vocab, sudoku_cfg, device, n_train, n_test, max_pos,
                          epochs, seed, targets=ABLATION_TARGETS):
    """Clean-run probes at every layer, keeping the weights.

    Returns (packs, metrics, test_inst, n_layers) where packs[(layer, target)]
    is the {weight, bias, mu, sigma} dict and metrics[(layer, target)] the
    held-out score on clean activations. test_inst goes back out so the
    read-back re-runs the SAME instances with the ablation live.
    """
    from .train_sudoku_transformer import build_instances

    train_inst = build_instances(sudoku_cfg, n_train, seed_start=500_000_000 + seed)
    test_inst = build_instances(sudoku_cfg, n_test, seed_start=900_000_000 + seed)
    acts_tr, lab_tr = collect_dataset(model, train_inst, vocab, device, max_pos,
                                      np.random.default_rng(seed))
    acts_te, lab_te = collect_dataset(model, test_inst, vocab, device, max_pos,
                                      np.random.default_rng(seed + 1))
    n_layers = acts_tr.shape[0] - 1
    print(f"  probes: {acts_tr.shape[1]} train / {acts_te.shape[1]} test positions, "
          f"{n_layers} blocks, d_model={acts_tr.shape[2]}")

    packs, metrics = {}, {}
    for layer in range(1, n_layers + 1):
        for target, kind in targets:
            m = fit_linear_probe(acts_tr[layer], lab_tr[target], acts_te[layer], lab_te[target],
                                 kind, epochs=epochs, device=device, return_probe=True)
            packs[(layer, target)] = m.pop("probe")
            metrics[(layer, target)] = m
        print(f"  clean layer {layer:2d}: frontier AUROC "
              f"{metrics[(layer, 'frontier_mask')]['auroc']:.4f}   next_node top1 "
              f"{metrics[(layer, 'next_node')]['balanced_accuracy']:.4f}   next_color top1 "
              f"{metrics[(layer, 'next_color')]['balanced_accuracy']:.4f}")
    return packs, metrics, test_inst, n_layers


def build_bases(packs, condition, n_layers, k, ablate_from, d_model, rng):
    """{layer -> (k, d)} for one condition; {} for `clean`."""
    if condition == "clean":
        return {}
    layers = range(max(1, ablate_from), n_layers + 1)
    if condition == "ablate_random":
        return {i: random_basis(d_model, k, rng) for i in layers}
    target = condition[len("ablate_"):]
    return {i: top_k_basis(raw_directions(packs[(i, target)]), k) for i in layers}


def readback(model, vocab, device, instances, packs, bases, n_layers, max_pos, seed,
             targets=ABLATION_TARGETS):
    """Re-collect activations with the ablation live and score the clean probes.

    Scored at the FINAL layer for every target, which is where frontier_mask
    and next_node both peak (finding 46), so one row per target says both
    "did the aimed-at thing go" and "did the others survive" -- selectivity,
    not just strength.
    """
    out = {}
    with SubspaceAblation(model, bases, device=device):
        acts, lab = collect_dataset(model, instances, vocab, device, max_pos,
                                    np.random.default_rng(seed + 1))
    for target, kind in targets:
        out[target] = score_stored_probe(packs[(n_layers, target)], acts[n_layers].numpy(),
                                         lab[target], kind)
    return out
