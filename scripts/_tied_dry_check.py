"""Dry check for arch_mode="tied_no_mlp" (stage P).

Run before the first tied cell is submitted. Verifies, in order:

  1. the untied path is UNCHANGED -- `no_mlp L=16 d192` still counts 2,504,890,
     so adding the mode regressed nothing the ladder is built on;
  2. a tied cell at the same shape counts (630d+58) + (4d^2+8d) exactly, i.e.
     one block's worth of attention plus the fixed embedding/readout term;
  3. the block really is SHARED -- one entry in encoder.layers, n_iters=16;
  4. depth is still real -- the tied forward differs from the same weights
     applied once, so the iteration is not silently collapsing;
  5. the feedforward writes identically ZERO, as in plain no_mlp;
  6. gradient reaches the shared block, and the frozen MLP gets none;
  7. the checkpoint round-trips: a reloaded tied cell rebuilds with
     num_layers=16 and reproduces the logits bit for bit.

CPU, a few seconds, no data and no GPU. Usage:
  python3 -u -m scripts._tied_dry_check
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/home/users/hshunt/spin_gnn")

from spin_gnn.sat_transformer2 import ARCH_MODES, SATReasoningTransformer  # noqa: E402

# V and MAXLEN are NOT free: the fixed term in the trainable count is
# d*(2V + max_len + 2) + V, so the whole ladder's "630d + 58" is only
# reproduced at vocab_size=58 (read off a stage N config.json) and
# max_len=512. Check 1 failing is how a wrong value here announces itself.
D, L, NHEAD, FF, V, MAXLEN = 192, 16, 4, 1024, 58, 512


def formula(d, n_blocks):
    return (630 * d + 58) + (4 * d * d + 8 * d) * n_blocks


def build(arch_mode, seed=0):
    torch.manual_seed(seed)
    return SATReasoningTransformer(
        vocab_size=V, d_model=D, num_layers=L, nhead=NHEAD, max_len=MAXLEN,
        dim_feedforward=FF, norm_first=True, arch_mode=arch_mode)


def main():
    ok = True

    print("=== 1. the untied path is unchanged ===")
    untied = build("no_mlp")
    got = untied.trainable_parameter_count()
    want = 2504890
    print("  no_mlp      L=16 d192  trainable=%d  want=%d  %s"
          % (got, want, "OK" if got == want else "REGRESSION"))
    ok &= got == want

    print("=== 2. the tied count is one block ===")
    tied = build("tied_no_mlp")
    got = tied.trainable_parameter_count()
    want = formula(D, 1)
    print("  tied_no_mlp L=16 d192  trainable=%d  want=%d  %s"
          % (got, want, "OK" if got == want else "MISMATCH"))
    print("  ratio untied/tied = %.2fx" % (untied.trainable_parameter_count() / got))
    ok &= got == want

    print("=== 3. the block is shared ===")
    n_blocks, n_iters = len(tied.encoder.layers), tied.n_iters
    print("  len(encoder.layers)=%d  n_iters=%d  tied=%s  %s"
          % (n_blocks, n_iters, tied.tied,
             "OK" if (n_blocks == 1 and n_iters == L and tied.tied) else "WRONG"))
    ok &= n_blocks == 1 and n_iters == L and tied.tied

    x = torch.randint(0, V, (2, 32))

    print("=== 4. depth is real (iterating != applying once) ===")
    tied.eval()
    with torch.no_grad():
        deep = tied(x)
        tied.n_iters = 1
        shallow = tied(x)
        tied.n_iters = L
    gap = (deep - shallow).abs().max().item()
    print("  max|f_16(x) - f_1(x)| = %.4f  %s" % (gap, "OK" if gap > 1e-3 else "COLLAPSED"))
    ok &= gap > 1e-3

    print("=== 5. the feedforward writes zero ===")
    blk = tied.encoder.layers[0]
    w2 = blk.linear2.weight.abs().max().item()
    b2 = blk.linear2.bias.abs().max().item()
    probe = torch.randn(2, 32, D)
    ff_out = blk.linear2(blk.dropout(blk.activation(blk.linear1(probe))))
    print("  max|linear2.weight|=%.1e  max|linear2.bias|=%.1e  max|ff(probe)|=%.1e  %s"
          % (w2, b2, ff_out.abs().max().item(),
             "OK" if max(w2, b2, ff_out.abs().max().item()) == 0.0 else "NONZERO"))
    ok &= max(w2, b2, ff_out.abs().max().item()) == 0.0

    print("=== 6. gradient reaches the shared block, not the MLP ===")
    tied.train()
    tied.zero_grad()
    tied(x).sum().backward()
    attn_g = blk.self_attn.in_proj_weight.grad
    attn_norm = attn_g.abs().max().item() if attn_g is not None else 0.0
    mlp_g = blk.linear1.weight.grad
    print("  max|grad in_proj| = %.3e  %s" % (attn_norm, "OK" if attn_norm > 0 else "NO GRAD"))
    print("  grad on frozen linear1 = %s  %s"
          % ("None" if mlp_g is None else "PRESENT",
             "OK" if mlp_g is None else "LEAKED"))
    ok &= attn_norm > 0 and mlp_g is None

    print("=== 7. checkpoint round-trip ===")
    tied.eval()
    with torch.no_grad():
        before = tied(x)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model.pt")
        torch.save(tied.state_dict(), p)
        again = build("tied_no_mlp", seed=123)       # different init on purpose
        missing = again.load_state_dict(torch.load(p, map_location="cpu"))
        again.eval()
        with torch.no_grad():
            after = again(x)
    same = torch.equal(before, after)
    print("  load_state_dict: %s" % (missing,))
    print("  num_layers restored=%d  logits identical=%s  %s"
          % (again.n_iters, same, "OK" if (same and again.n_iters == L) else "DRIFT"))
    ok &= same and again.n_iters == L

    print()
    print("ARCH_MODES = %s" % (ARCH_MODES,))
    print("DRY CHECK: %s" % ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
