"""SpinGNN: a graph neural network whose nodes are the spins of the kinetic
Ising model (spin_gnn.simulate), trained self-supervised for next-spin
("next-token") prediction: given the K-lag spin history at every node plus
the (possibly randomly-sparsified) coupling graph, predict the distribution
over the next spin at every node.

Each node's working representation has dimension `2 + hidden_dim`: 2 fixed
"visible" directions carrying the physical up/down spin information, plus
`hidden_dim` additional latent directions the network is free to use for
anything not captured by the bare linear mean field (this is exactly where
sparsification-induced information loss has to be compensated, if it can be).

Architecture rationale / exact-recovery sanity check
------------------------------------------------------
The physical local field is itself a linear graph convolution of the K-lag
history:  h_t,i = sum_lag sum_j J[i,j,lag] * s_{t-lag,j}  (simulate.local_field).
The first layer (LagCoupledInputConv) computes exactly this operation, one
lag-weighted edge sum at a time. With `exact=True` (readin fixed rather than
learned, hidden_dim=0) and a single layer, SpinGNN's logits reduce exactly to
[-beta*h, beta*h] (down, up), i.e. softmax gives the analytic Bayes-optimal
next-spin posterior on the *full* graph (keep_prob=1). See tests/test_exact_recovery.py.
Learned parameters, extra layers, and the hidden directions only start to
matter once the graph is randomly sparsified (information about the true
field must then be reconstructed/compensated) or once hidden_dim > 0 gives the
network room to model corrections beyond the bare linear field.
"""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class LagCoupledInputConv(MessagePassing):
    """First layer: raw (K, 2)-one-hot history -> (state_dim,) node state.

    message_{j->i} = sum_{lag=1}^K edge_attr[e, lag-1] * readin[lag] @ x_j[lag]
    """

    def __init__(self, context_length, state_dim, exact=False):
        # node_dim=0: x has shape (N, K, 2), and MessagePassing's default
        # node_dim=-2 would treat the K axis as "nodes" instead of N -- with
        # small K that silently indexes the wrong axis instead of erroring.
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.context_length = context_length
        self.state_dim = state_dim
        if exact:
            # onehot index 0 = down (s=-1), index 1 = up (s=+1) (dataset.spin_to_onehot);
            # this recovers the raw signed spin exactly: onehot1 - onehot0 = s.
            w = torch.zeros(context_length, 2, state_dim)
            w[:, 0, 0] = -1.0
            w[:, 1, 0] = 1.0
            self.register_buffer("readin", w)
        else:
            self.readin = nn.Parameter(0.1 * torch.randn(context_length, 2, state_dim))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        embedded = torch.einsum("ekc,kcd->ekd", x_j, self.readin)
        return torch.einsum("ek,ekd->ed", edge_attr, embedded)


class GatedGraphBlock(MessagePassing):
    """Generic residual message-passing block operating on (state_dim,) node
    states for layers beyond the first. Edge weight is a learned scalar
    function of the K-dim lag-resolved edge_attr (it no longer needs
    lag-resolution once history has already been folded into node state)."""

    def __init__(self, context_length, state_dim):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.edge_gate = nn.Sequential(nn.Linear(context_length, state_dim), nn.Tanh())
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + self.update_mlp(torch.cat([x, agg], dim=-1)))

    def message(self, x_j, edge_attr):
        return self.edge_gate(edge_attr) * x_j


class SpinGNN(nn.Module):
    def __init__(self, context_length, hidden_dim=32, num_layers=2, exact=False, beta_init=0.5):
        super().__init__()
        state_dim = 2 + hidden_dim
        self.state_dim = state_dim
        self.context_length = context_length
        self.exact = exact

        self.input_conv = LagCoupledInputConv(context_length, state_dim, exact=exact)
        self.blocks = nn.ModuleList(
            GatedGraphBlock(context_length, state_dim) for _ in range(max(num_layers - 1, 0))
        )
        if exact:
            # agg[:, 0] == h_t exactly (see LagCoupledInputConv); softmax of
            # [-beta*h, beta*h] gives p(up) = sigmoid(2*beta*h), the analytic posterior.
            readout = torch.zeros(state_dim, 2)
            readout[0, 0] = -beta_init
            readout[0, 1] = beta_init
            self.register_buffer("readout_weight", readout)
        else:
            self.readout = nn.Linear(state_dim, 2)

    def forward(self, x, edge_index, edge_attr):
        """x: (N, K, 2) one-hot spin history, oldest-first per simulate.rollout
        convention. Returns logits: (N, 2) over {down, up}."""
        h = self.input_conv(x, edge_index, edge_attr)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
        if self.exact:
            return h @ self.readout_weight
        return self.readout(h)
