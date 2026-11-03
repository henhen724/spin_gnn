"""One command-line surface for "which instance ensemble is this run about".

Every training entry point grew its own copy of --n-nodes/--degree/--rho/...,
and only sudoku_hparam_sweep ever grew --shuffle-adj, --shuffle-events and
--fall. The consequence was not cosmetic: sudoku_checkpoint_series,
sudoku_transition_sweep and sudoku_attention_evolution could not address the
fall ensemble at all, so none of the probe, SAE or attention tooling could be
pointed at the N=96 fall cells of 50860571 -- the only cells in the project
that actually sit in the slow phase (finding 45). sudoku_attention_evolution
could not even set n_nodes, so it was pinned to the N=24 default.

Scripts that already declare the instance-shape flags pass shape=False and get
just the three ensemble switches; the switches are what was missing everywhere.

sudoku_hparam_sweep keeps its own inline copies on purpose for now: job
50933500 is mid-flight against that file and its eval step has not started, so
it is left untouched until that lands.
"""
from .config import SudokuReasoningConfig

#: Every SudokuReasoningConfig field these flags cover, so callers can splat
#: `ensemble_kwargs(args)` without having to remember the list.
FIELDS = ("n_nodes", "degree", "n_colors", "rho", "rho_max", "seed", "hint_mode",
          "scratchpad", "shuffle_adj", "shuffle_events", "fall")


def add_ensemble_args(ap, *, shape=True, n_nodes=24, degree=5, n_colors=4,
                      rho=0.3, rho_max=0.95, seed=0):
    """Add the ensemble switches, and optionally the instance-shape flags.

    Defaults are parameters rather than constants because the existing scripts
    disagree about them and silently changing a script's default would change
    what its old command lines mean.
    """
    if shape:
        ap.add_argument("--n-nodes", type=int, default=n_nodes)
        ap.add_argument("--degree", type=int, default=degree)
        ap.add_argument("--n-colors", type=int, default=n_colors)
        ap.add_argument("--rho", type=float, default=rho)
        ap.add_argument("--rho-max", type=float, default=rho_max)
        ap.add_argument("--seed", type=int, default=seed)
        ap.add_argument("--hint-mode", type=str, default="removal",
                        choices=["removal", "random"],
                        help="'random' samples hints independently so DD can fail, which is "
                             "required for a hint-density solvability transition to exist in "
                             "the data at all")
        ap.add_argument("--scratchpad", action="store_true",
                        help="restate each deduction's evidence before it (sudoku_tokens "
                             "scratchpad grammar)")
    ap.add_argument("--shuffle-adj", action="store_true",
                    help="permute each node's neighbor list, removing the generator's "
                         "color-sorted adjacency order (see SudokuReasoningConfig.shuffle_adj); "
                         "required for any run that is measuring deduction rather than decoding")
    ap.add_argument("--shuffle-events", action="store_true",
                    help="permute the events within each DD round, removing the generator's "
                         "ascending-node-index order inside a round (see "
                         "SudokuReasoningConfig.shuffle_events); the round partition, T and "
                         "every DD predicate are unchanged, only the presentation order is")
    ap.add_argument("--fall", action="store_true",
                    help="draw the planted colouring as a FALL colouring -- every node has a "
                         "neighbour of every other colour (SudokuReasoningConfig.fall). Removes "
                         "the structural floor under rho_min (0.47 -> 0.33 at N=48) that "
                         "otherwise keeps every run in the fast phase.")
    return ap


def ensemble_kwargs(args):
    """The SudokuReasoningConfig kwargs these flags set, skipping any the
    caller's parser does not declare."""
    return {f: getattr(args, f) for f in FIELDS if hasattr(args, f)}


def ensemble_config(args, **overrides):
    return SudokuReasoningConfig(**{**ensemble_kwargs(args), **overrides})
