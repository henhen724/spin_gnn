"""Dry check for stage S: the two NEW tied widths, before either reaches a GPU.

scripts/_tied_dry_check.py already proved the tied mode itself at d192 (7/7,
stage P). It hardcodes D=192 and is left alone -- it is stage P's committed
registration artifact, and editing it would change what that stage asserted.
This file checks only what a WIDTH change can break, at the two widths stage S
adds, and it is run by scripts/slurm/stages_register_2026-11-10.sbatch with the
array depending on it --dependency=afterok.

The one claim the whole stage rests on is that H1's censoring floor 879,418
sits BETWEEN the two widths: d320 at 613,818 is below it and cannot discriminate
tying from size, d448 at 1,088,698 is above it and can. If either count is
wrong, the stage has no discriminating cell and must not start.
"""

import torch

from spin_gnn.sat_transformer2 import SATReasoningTransformer

L, NHEAD, FF, V, MAXLEN = 16, 4, 1024, 58, 512
WIDTHS = (320, 448)
CENSOR_FLOOR = 879418           # H1 no_mlp L=12 d128, the only rung that censored
WANT = {320: 613818, 448: 1088698}


def formula(d, n_blocks):
    return (630 * d + 58) + (4 * d * d + 8 * d) * n_blocks


def build(arch_mode, d, seed=0):
    torch.manual_seed(seed)
    return SATReasoningTransformer(
        vocab_size=V, d_model=d, num_layers=L, nhead=NHEAD, max_len=MAXLEN,
        dim_feedforward=FF, norm_first=True, arch_mode=arch_mode)


def main():
    ok = True
    n = 0

    print("=== 1. d192 is unchanged -- the trend's own baseline ===")
    tied192 = build("tied_no_mlp", 192)
    got = tied192.trainable_parameter_count()
    n += 1
    ok &= got == 270010
    print("  tied_no_mlp L=16 d192  trainable=%d  want=270010  %s"
          % (got, "OK" if got == 270010 else "REGRESSION -- stage P's cell moved"))

    for d in WIDTHS:
        print("=== d_model=%d ===" % d)
        print("  2. d_model divides by nhead")
        n += 1
        ok &= d % NHEAD == 0
        print("     %d %% %d = %d  %s" % (d, NHEAD, d % NHEAD,
                                          "OK" if d % NHEAD == 0 else "INVALID"))

        tied = build("tied_no_mlp", d)
        got = tied.trainable_parameter_count()
        want = formula(d, 1)
        print("  3. the tied count is ONE block")
        n += 1
        ok &= got == want == WANT[d]
        print("     trainable=%d  formula=%d  registered=%d  %s"
              % (got, want, WANT[d],
                 "OK" if got == want == WANT[d] else "MISMATCH"))

        untied = formula(d, L)
        print("  4. tied and untied differ at this width")
        n += 1
        ok &= got != untied
        print("     untied at d%d would be %d (%.2fx)  %s"
              % (d, untied, untied / got, "OK" if got != untied else "VOID"))

        print("  5. the block is shared")
        n_blocks, n_iters = len(tied.encoder.layers), tied.n_iters
        n += 1
        shared = n_blocks == 1 and n_iters == L and tied.tied
        ok &= shared
        print("     len(encoder.layers)=%d  n_iters=%d  tied=%s  %s"
              % (n_blocks, n_iters, tied.tied, "OK" if shared else "WRONG"))

        print("  6. depth is real (iterating != applying once)")
        x = torch.randint(0, V, (2, 32))
        tied.eval()
        with torch.no_grad():
            deep = tied(x)
            tied.n_iters = 1
            shallow = tied(x)
            tied.n_iters = L
        gap = (deep - shallow).abs().max().item()
        n += 1
        ok &= gap > 1e-3
        print("     max|f_16(x) - f_1(x)| = %.4f  %s"
              % (gap, "OK" if gap > 1e-3 else "COLLAPSED"))

        print("  7. the feedforward writes identically zero")
        blk = tied.encoder.layers[0]
        probe = torch.randn(2, 32, d)
        ff = blk.linear2(blk.dropout(blk.activation(blk.linear1(probe))))
        z = max(blk.linear2.weight.abs().max().item(),
                blk.linear2.bias.abs().max().item(),
                ff.abs().max().item())
        n += 1
        ok &= z == 0.0
        print("     max|ff| = %.1e  %s" % (z, "OK" if z == 0.0 else "NONZERO"))

    print("=== 8. the censoring floor sits BETWEEN the two widths ===")
    lo, hi = WANT[320], WANT[448]
    n += 1
    straddles = lo < CENSOR_FLOOR < hi
    ok &= straddles
    print("  %d (d320) < %d (H1) < %d (d448)  %s"
          % (lo, CENSOR_FLOOR, hi,
             "OK -- d448 is the discriminating cell" if straddles
             else "NO DISCRIMINATING CELL: the stage has no S0 to call"))

    print()
    print("%d checks run, all passed" % n if ok
          else "%d checks run, AT LEAST ONE FAILED (see the lines above)" % n)
    if not ok:
        print("FAILING CLOSED: a width stage S depends on is not what the "
              "registration says it is. No cell may start.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
