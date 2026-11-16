"""Shapes and invariants for stage LOOP's feedback term. CPU, seconds, no data.

The three things that have to be true before a GPU hour is spent on it:

  1. A model built WITHOUT the flag is unchanged -- same parameter count, no
     `loop_scale` key, and forward() returns what it returned before.
  2. With the flag on and `loop_scale = 0`, the logits are EXACTLY the
     no-feedback logits at the same weights, with a NON-ZERO `det_scale`. This
     is the warm-start guarantee: the arm begins at the stage Y body, not near
     it. `log sigma(g) + log 2` is what buys it and a dropped `+ log 2` would
     show up here and nowhere else.
  3. A non-zero `loop_scale` does change them, and the gradient at c = 0 is
     non-zero -- otherwise training could never open the loop and the whole arm
     would be a null measurement dressed as one.

Also checks that a stage-Y state dict loads into the feedback model with
`loop_scale` as the ONLY missing key, which is how the probe warm-starts.
"""
import sys
import torch

sys.path.insert(0, "/home/users/hshunt/spin_gnn")
from spin_gnn.sat_transformer2 import SATReasoningTransformer  # noqa: E402

V, D, T, NODES, OFF = 58, 64, 4, 12, 10
kw = dict(vocab_size=V, d_model=D, num_layers=T, nhead=4, max_len=64,
          dim_feedforward=128, norm_first=True, arch_mode="tied_no_mlp",
          n_blocks=1, determinacy_channel=True, node_offset=OFF, n_nodes=NODES)

torch.manual_seed(0)
plain = SATReasoningTransformer(**kw)
torch.manual_seed(0)
loop = SATReasoningTransformer(**kw, determinacy_feedback=True)

n_plain = sum(p.numel() for p in plain.parameters() if p.requires_grad)
n_loop = sum(p.numel() for p in loop.parameters() if p.requires_grad)
print(f"trainable: plain {n_plain:,}   loop {n_loop:,}   diff {n_loop - n_plain}")
assert n_loop - n_plain == 1, "the loop must cost exactly one scalar"
assert "loop_scale" not in plain.state_dict(), "the flag leaked into the control"
assert plain.loop_scale is None and loop.loop_scale is not None

# a TRAINED channel, so g is not identically zero and only c = 0 can make the
# feedback inert -- checking this at det_scale = 0 would prove nothing
with torch.no_grad():
    s = torch.randn(D) * 0.3
    plain.det_scale.copy_(s)
    loop.det_scale.copy_(s)
assert float(loop.det_scale.norm()) > 0.05

x = torch.randint(0, V, (3, 17))
plain.eval(), loop.eval()
with torch.no_grad():
    z_plain = plain(x)
    z_loop = loop(x)
    g_plain = plain(x, return_determinacy=True)[1]
    g_loop = loop(x, return_determinacy=True)[1]
d_logit = float((z_plain - z_loop).abs().max())
print(f"max |z_plain - z_loop| at loop_scale=0: {d_logit:.3e}  "
      f"(g: {float((g_plain - g_loop).abs().max()):.3e})")
assert torch.equal(z_plain, z_loop), "c=0 is not inert -- the centering is wrong"
assert torch.equal(g_plain, g_loop)

with torch.no_grad():
    loop.loop_scale.fill_(0.5)
    z_open = loop(x)
rel = float((z_open - z_plain).abs().mean() / z_plain.abs().mean())
print(f"mean relative logit change at loop_scale=0.5: {rel:.4f}  "
      f"fb_rel {float(loop._fb_rel):.4f}")
assert rel > 1e-4, "an open loop does nothing -- the delta never reached h"

# the gradient at c = 0: if it vanishes, training can never open the loop
with torch.no_grad():
    loop.loop_scale.zero_()
loop.train()
z = loop(x)
z.square().mean().backward()
gc = float(loop.loop_scale.grad)
print(f"dL/dc at c=0: {gc:+.3e}")
assert abs(gc) > 1e-10, "c=0 is a saddle -- the loop could never open"

missing, unexpected = loop.load_state_dict(plain.state_dict(), strict=False)
print(f"warm start: missing={list(missing)}  unexpected={list(unexpected)}")
assert list(missing) == ["loop_scale"] and not list(unexpected)

# and the untied / channel-less guards, which exist so a mis-specified arm
# cannot run silently
for bad, why in (
        (dict(arch_mode="no_mlp"), "untied"),
        (dict(determinacy_channel=False, node_offset=None, n_nodes=None),
         "no channel")):
    try:
        SATReasoningTransformer(**{**kw, **bad}, determinacy_feedback=True)
    except ValueError as e:
        print(f"refused ({why}): {str(e).splitlines()[0][:60]}...")
    else:
        raise AssertionError(f"{why} was not refused")

print("\nall invariants hold")
