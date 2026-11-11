"""Dry check for stage U: k distinct blocks cycled, before any cell reaches a GPU.

Stage S's check (scripts/_tiedwidth_dry_check.py) proved the SHARED block at
d320. This one checks only what the k knob can break, at the one width stage U
uses, and it is run by scripts/slurm/stageu_register_2026-11-11.sbatch with the
training array depending on it --dependency=afterok.

The claims the stage rests on:
  1. k=1 at d320 is still stage S's cell, 613,818 trainable. If it moved, the
     control moved, and the ladder has no baseline.
  2. k=2 (1,025,978) and k=4 (1,850,298) are both ABOVE H1's censoring floor
     879,418, so a censor at either is about the architecture and not about
     size -- the same discipline stage S applied to d448.
  3. The cycle really is 0,1,...,k-1 repeated T/k times, so the network is the
     composition G = F_{k-1} o ... o F_0 applied T/k times. If the order were
     0,0,1,1 or if the loop ran k times instead of T, the cell would still
     train and still report a number.
"""

import torch

from spin_gnn.sat_transformer2 import (
    SATReasoningTransformer, applied_blocks, model_depth, n_blocks,
)

L, NHEAD, FF, V, MAXLEN, D = 16, 4, 1024, 58, 512, 320
KS = (1, 2, 4)
CENSOR_FLOOR = 879418           # H1 no_mlp L=12 d128, the only rung that censored
S1_TRAINABLE = 613818           # stage S's shared-block d320 cell, rho_50 0.757
WANT = {1: 613818, 2: 1025978, 4: 1850298}


def formula(d, k):
    """Embeddings and readout once, the block k times -- the tied arithmetic
    with the per-block term paid k times instead of once."""
    return (630 * d + 58) + (4 * d * d + 8 * d) * k


def build(k, arch_mode="tied_no_mlp", depth=L, seed=0):
    torch.manual_seed(seed)
    return SATReasoningTransformer(
        vocab_size=V, d_model=D, num_layers=depth, nhead=NHEAD, max_len=MAXLEN,
        dim_feedforward=FF, norm_first=True, arch_mode=arch_mode, n_blocks=k)


def main():
    ok = True
    n = 0
    counts = {}

    for k in KS:
        print("=== k=%d (%d distinct block(s), each applied %d times) ==="
              % (k, k, L // k))
        m = build(k)
        got = m.trainable_parameter_count()
        counts[k] = got
        want = formula(D, k)
        n += 1
        hit = got == want == WANT[k]
        ok &= hit
        print("  1. trainable=%d  formula=%d  registered=%d  %s"
              % (got, want, WANT[k], "OK" if hit else "MISMATCH"))

        n += 1
        shaped = len(m.encoder.layers) == k and n_blocks(m) == k and model_depth(m) == L
        ok &= shaped
        print("  2. len(encoder.layers)=%d  n_blocks=%d  depth=%d  %s"
              % (len(m.encoder.layers), n_blocks(m), model_depth(m),
                 "OK" if shaped else "WRONG -- k changed the APPLICATION count"))

        n += 1
        # By identity, not by equality: nn.Module has no __eq__ and ModuleList
        # has no .index, so the block that was APPLIED is matched to the block
        # that was BUILT the only way that means anything here.
        ids = [id(b) for b in m.encoder.layers]
        order = [ids.index(id(b)) if id(b) in ids else -1
                 for b in applied_blocks(m)]
        want_order = [t % k for t in range(L)]
        cycles = order == want_order
        ok &= cycles
        print("  3. application order %s  %s"
              % ("".join(str(i) for i in order),
                 "OK" if cycles else "NOT THE CYCLE %s" % want_order))

        n += 1
        ident = all(
            torch.equal(v, m.encoder.layers[0].state_dict()[key])
            for i in range(1, k)
            for key, v in m.encoder.layers[i].state_dict().items())
        ok &= ident
        print("  4. all %d blocks identical at init  %s"
              % (k, "OK -- the ladder is about training, not about the draw"
                 if ident else "DIFFERENT DRAWS -- k confounds init with capacity"))

        n += 1
        zero = max(max(blk.linear2.weight.abs().max().item(),
                       blk.linear2.bias.abs().max().item())
                   for blk in m.encoder.layers)
        frozen = all(not blk.linear1.weight.requires_grad
                     and not blk.linear2.weight.requires_grad
                     for blk in m.encoder.layers)
        hit = zero == 0.0 and frozen
        ok &= hit
        print("  5. feedforward zero and frozen in EVERY block: max|w2|=%.1e "
              "frozen=%s  %s" % (zero, frozen, "OK" if hit else "LIVE MLP"))

        n += 1
        x = torch.randint(0, V, (2, 32))
        m.eval()
        with torch.no_grad():
            deep = m(x)
            m.n_iters = k            # one turn of the cycle instead of L/k
            shallow = m(x)
            m.n_iters = L
        gap = (deep - shallow).abs().max().item()
        ok &= gap > 1e-3
        print("  6. iterating is real: max|G^%d(x) - G^1(x)| = %.4f  %s"
              % (L // k, gap, "OK" if gap > 1e-3 else "COLLAPSED"))

    print("=== the ladder as a whole ===")
    n += 1
    base = counts[1] == S1_TRAINABLE
    ok &= base
    print("  7. k=1 is still stage S's cell: %d vs %d  %s"
          % (counts[1], S1_TRAINABLE,
             "OK" if base else "THE CONTROL MOVED -- stage S is no longer the baseline"))

    n += 1
    above = counts[2] > CENSOR_FLOOR and counts[4] > CENSOR_FLOOR
    ok &= above
    print("  8. k=2 (%d) and k=4 (%d) are both above H1's floor %d  %s"
          % (counts[2], counts[4], CENSOR_FLOOR,
             "OK -- a censor at either is about the architecture"
             if above else "BELOW THE FLOOR: a censor could not be read"))

    n += 1
    mono = counts[1] < counts[2] < counts[4] < formula(D, L)
    ok &= mono
    print("  9. the ladder interpolates: %d < %d < %d < %d (untied)  %s"
          % (counts[1], counts[2], counts[4], formula(D, L),
             "OK" if mono else "NOT AN INTERPOLATION"))

    print()
    print("%d checks run, all passed" % n if ok
          else "%d checks run, AT LEAST ONE FAILED (see the lines above)" % n)
    if not ok:
        print("FAILING CLOSED: something stage U's registration asserts is not "
              "true of the code. No cell may start.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
