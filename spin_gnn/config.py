from dataclasses import dataclass, field


@dataclass
class DisorderConfig:
    """Quenched-disorder / dynamics parameters for the kinetic Ising model of
    Zavatone-Veth & Pehlevan, "A note on the dynamics of extended-context
    disordered kinetic spin models" (2026), https://doi.org/10.1088/1751-8121/ae38a4.

    E(s_t | s_{t-1..t-K}) = -sum_k s_t^T J_k s_{t-k}, J_{ij,k} jointly Gaussian,
    zero mean, E[J_{ij,k} J_{i'j',k'}] = delta_ii' delta_jj' Gamma_{k,k'}.

    The paper states this covariance without an explicit N-scaling. To get an
    O(1) local field as N -> infinity (required for the N -> infinity, fixed
    (T, K) DMFT limit they analyze) we scale entries by 1/N, i.e.
    Var(J_ij,k) = Gamma_kk / N. This is the standard Sherrington-Kirkpatrick
    convention and is the one implemented here; set `coupling_scale` to 1.0 to
    recover the paper's literal (non-extensive) convention for small N.
    """

    n_spins: int = 200
    context_length: int = 4
    beta: float = 0.5
    gamma_kind: str = "identity"  # "identity" | "alternating" | "power_law", see simulate.build_gamma
    gamma_param: float = 0.0
    coupling_scale: float = 1.0
    seed: int = 0


@dataclass
class SparsificationConfig:
    keep_prob: float = 1.0
    compensate: bool = True
    seed: int = 0


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    message_dim: int = 32
    num_layers: int = 2


@dataclass
class TrainConfig:
    n_realizations: int = 8
    trajectory_length: int = 512
    val_fraction: float = 0.1
    batch_size: int = 4
    lr: float = 1e-3
    epochs: int = 50
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"


@dataclass
class ExperimentConfig:
    disorder: DisorderConfig = field(default_factory=DisorderConfig)
    sparsification: SparsificationConfig = field(default_factory=SparsificationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class SATConfig:
    """Random k-SAT instance generation, see spin_gnn/ksat.py.

    m_clauses = round(alpha * n_vars); alpha=4.2 sits near the classic
    random-3-SAT satisfiability threshold (~4.267), so the default samples
    a mix of SAT and UNSAT instances rather than the easy regime far from it.
    """

    n_vars: int = 24
    k: int = 3
    alpha: float = 4.2
    seed: int = 0


@dataclass
class SATModelConfig:
    d_model: int = 64
    num_layers: int = 2
    nhead: int = 1
    dim_feedforward: int = 0  # 0 = SATTransformer's own default (4 * d_model)
    qk_dim: int = 0  # 0 = full rank (= d_model); see sat_transformer.LowRankSelfAttention


@dataclass
class SATTrainConfig:
    n_train_instances: int = 256
    n_val_instances: int = 32
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 50
    device: str = "cuda"
    max_trace_tokens: int = 0  # 0 = no cap; see sat_dataset.SATTraceDataset's docstring


@dataclass
class SATReasoningConfig:
    """Random k-SAT instance generation for the v2 event-based encoding, see
    spin_gnn/ksat2.py. Unlike SATConfig's default alpha=4.2 (near the
    satisfiability threshold, deliberately hard), the default here is
    deliberately easy/underconstrained -- this encoding's own tractability
    is still being established, so start where backtracking is rare."""

    n_vars: int = 16
    k: int = 3
    alpha: float = 1.25  # m_clauses = round(alpha * n_vars) = 20 at n_vars=16
    n_hints: int = 0
    seed: int = 0


@dataclass
class SATReasoningModelConfig:
    d_model: int = 64
    num_layers: int = 2
    nhead: int = 1
    dim_feedforward: int = 0  # 0 = 4 * d_model, same convention as SATModelConfig
    norm_first: bool = False  # True = pre-norm (stable at depth); see sat_transformer2.py


@dataclass
class SATReasoningTrainConfig:
    n_train_instances: int = 2000
    n_val_instances: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 50
    device: str = "cuda"
    max_len: int = 0  # 0 = no cap (sequences are short in this encoding, unlike ksat.py's)


@dataclass
class SudokuReasoningConfig:
    """Random DD-Sudoku instance generation for the event-based
    reasoning-trace encoding, see spin_gnn/sudoku_trace.py. n_nodes/degree/
    n_colors must satisfy spin_gnn.sudoku_dd.planted_regular_graph's
    constraints (n_colors | n_nodes, n_colors*(n_colors-1) | degree*n_nodes).
    Deliberately small-N (same scale as SATReasoningConfig.n_vars=16, not
    sudoku_sweep.py's N~10^3-10^5 phase-transition regime) -- the point here
    is whether a transformer can learn the DD process at all, not
    large-N statistics.

    rho is the target hint fraction, drawn once per instance uniformly in
    [rho, rho_max] (rho_max == rho => fixed target density). Hints
    themselves are sampled via the slides' own protocol (see
    train_sudoku_transformer.build_instances / sudoku_dd.removal_path):
    greedy random-order hint removal, only accepting a removal that still
    leaves the coloring fully DD-recoverable. This guarantees every instance
    is fully resolvable by construction -- there is no "unresolved" case,
    matching the slides' own methodology exactly -- so rho/rho_max here only
    controls the mix of reasoning depths (shallow/high-rho vs. deep/low-rho,
    down to each path's own minimal density), not resolved-vs-unresolved."""

    n_nodes: int = 24
    degree: int = 5
    n_colors: int = 4
    rho: float = 0.3
    rho_max: float = 0.95
    seed: int = 0
    # "removal" (default): the slides' greedy hint-removal protocol described
    # above -- every instance resolvable by construction, rho sets reasoning
    # depth only. "random": independent uniform hints at exactly
    # round(rho * n_nodes) nodes, so DD fails below its threshold and the
    # distribution actually contains a hint-density solvability transition.
    hint_mode: str = "removal"
    # Restate each deduction's evidence (its neighbors' already-known colors)
    # immediately before the deduction, making every step locally determined --
    # see sudoku_tokens.py's module docstring. Off by default so existing
    # checkpoints and their token ids are unaffected.
    scratchpad: bool = False
    # Permute each node's neighbor list before encoding. MEASURED 2026-10-26:
    # planted_regular_graph appends edges in class-pair block order, so adj[v]
    # arrives sorted by the neighbor's TRUE COLOR for 100.0% of nodes at
    # N=24/48/96, and encode_instance emits that order verbatim -- the planted
    # coloring is partly readable off token position. Job 49655447 showed every
    # checkpoint in the project collapses (0.94-1.00 -> 0.00-0.15) when adj[v]
    # is permuted at eval time, so the models are decoding that order rather
    # than deducing. False reproduces the generator stream of every run before
    # 2026-10-26 bit-for-bit (the permutation draws from its own rng stream);
    # True is what any run measuring *deduction* should use.
    shuffle_adj: bool = False
    # Permute the events WITHIN each DD round before encoding. The second
    # order-based shortcut, and the last uncontrolled one. solve_with_dd_trace
    # emits `sorted(...)` per round (sudoku_trace.py:83), so round events arrive
    # in ascending node index for 100.0% of rounds, and `shuffle_adj` does not
    # touch that -- it permutes neighbour lists only. MEASURED 2026-10-27
    # (scripts/order_baseline_next_node.py): the hand rule "smallest index above
    # the previous one whose colour is still unknown" -- bookkeeping, no
    # frontier, no deduction -- scores max-over-v F1 0.632 on `next_node` at
    # pre_node positions, ABOVE the leak-free model's own best SAE feature
    # (0.545), and taking the frontier into account makes the rule exact
    # (1.000). So "it knows which node comes next" is free in both arms, and no
    # eval-time permutation can say whether the model DEPENDS on the order:
    # that needs a retrain, which is what this flag is for.
    #
    # Permuting within a round is semantics-preserving. A DD round deduces
    # every node whose neighbours already cover q-1 colours SIMULTANEOUSLY, so
    # the order inside the round is an arbitrary presentation choice: the round
    # partition, its membership, the reasoning depth T and every DD predicate
    # are unchanged, and the target colours travel with their nodes. That makes
    # this a strictly weaker intervention than shuffle_adj, which changes which
    # tokens appear where in the graph section.
    #
    # False reproduces the generator stream of every run before 2026-10-27
    # bit-for-bit (the permutation draws from its own rng stream, offset 50M,
    # following the 10/20/30/40M convention); True is what a run measuring
    # order-independent deduction should use.
    #
    # UNLIKE shuffle_adj it is NOT an instance key for sudoku_agg_eval, and
    # that is what makes the control cheap: solve_rate prompts with hints and
    # graph only and scores free generation against the planted colouring, so
    # a permuted-round cell and a sorted-round cell get byte-identical prompts
    # and an order-insensitive verdict. They POOL, so the comparison is paired
    # McNemar rather than two overlapping Wilson intervals. Every teacher-
    # forced analysis (SAE, probes, circuits, depth slots, attention) is the
    # other way round and must rebuild its sequences with the flag the
    # checkpoint was trained with, or it reads the model off-distribution.
    # Token-level metrics are NOT comparable across the flag: permuting the
    # order injects log(k!) nats per k-event round that no model can predict,
    # so a higher val_nll in the True arm is the task changing, not the model
    # getting worse.
    shuffle_events: bool = False
    # Draw the planted colouring as a FALL colouring: every node has at least
    # one neighbour of every other colour (sudoku_dd.planted_regular_graph's
    # `fall`). DD can only ever deduce a colourful node, so under the default
    # ensemble the 0.3827 of nodes (q=4, d=5) born missing a colour are
    # permanent hints and pin greedy removal at rho_min ~ 0.47 -- above the
    # slides' rho_c ~ 0.46, which is why every depth result in this project
    # so far is fast-phase (finding 34, SUDOKU_DD.md's open issue).
    # MEASURED 2026-11-02: the floor is exactly the non-colourful fraction,
    # and removing it drops rho_min to 0.308 at large N with T ~ N^0.727 and
    # <m> ~ 0.21, i.e. the slides' d=5 slow branch. At the sizes we train on
    # it roughly doubles the reachable depth (N=48: rho_min 0.469 -> 0.332,
    # T_end 7.9 -> 13.8) and halves the seed spread in rho_min.
    #
    # An INSTANCE KEY for sudoku_agg_eval: it changes the graph the prompt
    # describes, so a fall cell and a default cell must never pool. False
    # reproduces every pre-2026-11-02 generator stream bit-for-bit (verified
    # against HEAD over 3 sizes x 3 seeds).
    fall: bool = False


@dataclass
class SudokuReasoningModelConfig:
    d_model: int = 64
    num_layers: int = 2
    nhead: int = 1
    dim_feedforward: int = 0  # 0 = 4 * d_model, same convention as SATReasoningModelConfig
    # True (unlike SATReasoningModelConfig's norm_first=False default): the
    # k-SAT v2 work found post-norm collapses at depth (0% well-formed output
    # at num_layers in {12,16}) and pre-norm is the standard fix -- applying
    # that lesson from the start here rather than waiting to rediscover it.
    norm_first: bool = True
    # How much MACHINERY the block keeps. "full" reproduces every run before
    # 2026-11-04 (a trained ReLU MLP in every block); the other four take
    # something away, one kind at a time, for the minimal-architecture
    # question -- how little does learning DD actually require?
    #   "linear"  identity activation, so each block's MLP collapses to a
    #             single linear map and the attention softmax is the only
    #             nonlinearity left in the network.
    #   "rf_mlp"  random features, literal reading: linear1 (d_model -> ff) is
    #             frozen at its initialization in every block and linear2
    #             trains. Attention and the readout still train, so the
    #             training problem is NOT convex.
    #   "rf_all"  random features, convex reading: everything is frozen at
    #             initialization except the final readout. Softmax
    #             cross-entropy on fixed features is convex in those weights,
    #             so this is the one cell that can be written down and
    #             analyzed rather than only measured.
    #   "no_mlp"  attention only: the MLP write is zeroed and frozen, leaving
    #             x = x + attn(norm1(x)) exactly. The floor of the axis.
    # Load-bearing at EVAL time in the same way `fall` is (finding 43): a
    # reader that defaults this wrong runs a ReLU where the model was trained
    # linear, with no error and a plausible number, so it is written into
    # config.json and read back by load_checkpoint.
    arch_mode: str = "full"
    # k: how many DISTINCT blocks a tied stack cycles through, with each one
    # applied num_layers/k times. 1 (the default) is the weight-tied cell every
    # run before 2026-11-11 trained; num_layers would be the untied stack. Only
    # defined for arch_mode="tied_no_mlp" -- SATReasoningTransformer raises
    # otherwise rather than quietly ignoring it. Load-bearing at EVAL time the
    # way arch_mode is: a reader that defaults it to 1 for a k=2 checkpoint
    # builds half the blocks and load_state_dict fails -- loudly, which is the
    # one mercy here, but it is still written into config.json and read back.
    n_blocks: int = 1


@dataclass
class SudokuReasoningTrainConfig:
    n_train_instances: int = 2000
    n_val_instances: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 50
    device: str = "cuda"
    max_len: int = 0  # 0 = no cap
    # Generation headroom, in tokens, added to the positional table on top of
    # the longest TRAINING sequence. `generate` hard-stops at
    # pos_embed.shape[0], so with no margin a model that emits one event more
    # than the longest trace it ever saw is guillotined mid-sequence and scored
    # as a failure. Finding 44 measured exactly that: at N=48 max_len came out
    # 414 against a longest true fall trace of 408, and the two L=12 cells
    # truncated in up to 0.0275 of the deepest-density bins. Sized in the trace
    # grammar's own units -- a deduction costs 2 tokens and a round marker 1 --
    # so the auto value is room to re-emit every node once more, plus a round
    # marker and the terminal SOLVED. 0 reproduces every run before 2026-11-02.
    max_len_margin: int = -1  # -1 = auto: 2 * n_nodes + 2
    # "constant" reproduces every run before 2026-10-25, which held lr fixed for
    # the whole budget. "cosine" decays lr -> lr*lr_min_factor across epochs,
    # stepped once per epoch. Default stays "constant" so no earlier comparison
    # is silently redefined.
    lr_schedule: str = "constant"
    lr_min_factor: float = 0.01
    # Linear ramp from lr/(warmup_epochs+1) to lr over the first
    # `warmup_epochs` epochs, then whatever lr_schedule says for the rest.
    # 0 (the default) means no ramp, which is every run before 2026-10-28.
    # Motivated by finding 25: at N=96 the deep-thin cell (L=12, ff=256) ends
    # with a well_formed_rate of 0.853 while its CONDITIONAL solve rate sits at
    # the wide knee's, i.e. it never learned the output grammar rather than
    # never learning to deduce -- the signature of an early-training failure,
    # which is what warmup exists to fix.
    warmup_epochs: int = 0
    # Pin the GPU kernels' reduction order, so that two runs of ONE
    # configuration at one seed are the same run. A seed already fixes the
    # instances, the weight init and the batch order (train_sudoku_model seeds
    # all three), and finding 66b is what is left over after that: at the
    # 50-epoch cell's last epoch the 100-epoch cell at the SAME seed differed
    # by up to 0.0602 train NLL and 0.0246 accuracy, against the 0.0011 of
    # accuracy that stage S's gap is worth. That is the measurement instrument,
    # not the model. False reproduces every run before 2026-11-13 exactly --
    # stages S through W were all run without it and none of their numbers
    # move -- and True costs speed, which is what runs/2026-11-13_determinism
    # measures.
    deterministic: bool = False
    # Stage X. Weight on a multi-label BCE that trains the readout's NODE
    # logits, at every position where a node is about to be emitted, to say
    # which still-uncoloured nodes are deducible right now (the target is
    # spin_gnn.sudoku_determinacy). Zero new parameters: the loss attaches to
    # the readout the model already has.
    #
    # Motivated by findings 68 and 69, and by a gauge argument. Cross-entropy
    # sees the logits only through a softmax, so it fixes them only up to an
    # additive constant and can express "this node comes next" but never "that
    # node is not emittable at all" -- which is exactly the guard the tied
    # model lacks while demonstrably computing the determinacy it would need
    # (finding 69: p_freed ~0.52 under an edit that destroys determinacy, and
    # the node readout does not move to four decimals). This term is the
    # smallest change that makes the absolute logit height mean something.
    #
    # 0.0 reproduces every run before 2026-11-13 exactly, including the batch
    # order: at 0 the determinacy targets are never even built.
    determinacy_weight: float = 0.0
