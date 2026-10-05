"""Forward/backward sanity + causal-mask + generation-stopping checks for
SATReasoningTransformer, mirroring test_sat_transformer_shapes.py."""
import numpy as np
import torch

from spin_gnn.ksat2_tokens import Vocab
from spin_gnn.sat_transformer2 import SATReasoningTransformer, generate, sat2_accuracy, sat2_nll_loss


def test_forward_shape_and_backward():
    b, l, v = 3, 7, 20
    model = SATReasoningTransformer(vocab_size=v, d_model=16, num_layers=2, nhead=2, max_len=32)
    x = torch.randint(0, v, (b, l))
    target = torch.randint(0, v, (b, l))
    mask = torch.ones(b, l, dtype=torch.bool)

    logits = model(x, valid_mask=mask)
    assert logits.shape == (b, l, v)

    loss = sat2_nll_loss(logits, target, mask)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

    acc = sat2_accuracy(logits, target, mask)
    assert 0.0 <= acc <= 1.0


def test_causal_mask_blocks_future_positions():
    v, l = 10, 6
    model = SATReasoningTransformer(vocab_size=v, d_model=16, num_layers=2, nhead=2, max_len=32)
    model.eval()

    torch.manual_seed(0)
    x = torch.randint(0, v, (1, l))
    x_perturbed = x.clone()
    x_perturbed[0, -1] = (x_perturbed[0, -1] + 1) % v

    with torch.no_grad():
        logits = model(x)
        logits_perturbed = model(x_perturbed)

    torch.testing.assert_close(logits[:, :-1], logits_perturbed[:, :-1])
    assert not torch.allclose(logits[:, -1], logits_perturbed[:, -1])


def test_generate_stops_right_after_unsat():
    vocab = Vocab(n_vars=4, m_clauses=3)
    model = SATReasoningTransformer(vocab_size=vocab.size, d_model=8, num_layers=1, nhead=1, max_len=64)
    model.eval()
    # Force the model to always predict UNSAT via a huge output bias.
    with torch.no_grad():
        model.readout.bias.zero_()
        model.readout.bias[vocab.UNSAT] = 100.0
    prompt = np.array([vocab.BEGIN_REASONING], dtype=np.int64)
    out = generate(model, prompt, vocab, max_new_tokens=30)
    assert out[-1] == vocab.UNSAT
    assert len(out) == 2  # prompt token + exactly one generated (UNSAT) token


def test_generate_stops_within_2n_tokens_after_sat():
    vocab = Vocab(n_vars=4, m_clauses=3)
    model = SATReasoningTransformer(vocab_size=vocab.size, d_model=8, num_layers=1, nhead=1, max_len=200)
    model.eval()
    with torch.no_grad():
        model.readout.bias.zero_()
        model.readout.bias[vocab.SAT] = 100.0  # every step, SAT is the argmax
    prompt = np.array([vocab.BEGIN_REASONING], dtype=np.int64)
    out = generate(model, prompt, vocab, max_new_tokens=100)
    sat_positions = np.flatnonzero(out == vocab.SAT)
    assert sat_positions.size >= 1
    first_sat = sat_positions[0]
    assert len(out) - first_sat - 1 <= 2 * vocab.n_vars + 1  # a little slack for the boundary check


def test_norm_first_forward_shape_and_backward():
    """Pre-norm path (norm_first=True): builds with a final LayerNorm and
    runs forward/backward cleanly, same as the post-norm default."""
    b, l, v = 3, 7, 20
    model = SATReasoningTransformer(vocab_size=v, d_model=16, num_layers=3, nhead=2, max_len=32, norm_first=True)
    assert model.encoder.norm is not None
    x = torch.randint(0, v, (b, l))
    target = torch.randint(0, v, (b, l))
    mask = torch.ones(b, l, dtype=torch.bool)

    logits = model(x, valid_mask=mask)
    assert logits.shape == (b, l, v)
    loss = sat2_nll_loss(logits, target, mask)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_norm_first_default_has_no_final_norm():
    model = SATReasoningTransformer(vocab_size=20, d_model=16, num_layers=2, nhead=2, max_len=32)
    assert model.encoder.norm is None


def test_norm_first_changes_output_vs_post_norm():
    """Same weights would differ in shape anyway (pre-norm adds a final
    LayerNorm's params), but more to the point: forward passes under
    otherwise-identical random init are not numerically identical -- this
    is actually a different computation, not just a relabeling."""
    torch.manual_seed(0)
    post = SATReasoningTransformer(vocab_size=15, d_model=16, num_layers=4, nhead=2, max_len=16, norm_first=False)
    torch.manual_seed(0)
    pre = SATReasoningTransformer(vocab_size=15, d_model=16, num_layers=4, nhead=2, max_len=16, norm_first=True)
    x = torch.randint(0, 15, (1, 5))
    with torch.no_grad():
        out_post = post(x)
        out_pre = pre(x)
    assert not torch.allclose(out_post, out_pre)


def test_generate_respects_max_new_tokens_when_never_terminal():
    vocab = Vocab(n_vars=4, m_clauses=3)
    model = SATReasoningTransformer(vocab_size=vocab.size, d_model=8, num_layers=1, nhead=1, max_len=200)
    model.eval()
    with torch.no_grad():
        model.readout.bias.zero_()
        model.readout.bias[vocab.NO_UNIT] = 100.0  # never emits SAT/UNSAT
    prompt = np.array([vocab.BEGIN_REASONING], dtype=np.int64)
    out = generate(model, prompt, vocab, max_new_tokens=25)
    assert len(out) == 26
