"""CPU check on the max_len generation margin added 2026-11-02.

Run it from a job, not a login node -- see
scripts/slurm/sudoku2_maxlen_margin_check_2026-11-02.sbatch for why, and for
the defect it guards (finding 44: max_len 414 against a longest true fall
trace of 408, truncating up to 0.0275 of the deepest-density L=12 bins).

Asserts, on a tiny N=12 instance set:
  1. margin 0 reproduces the pre-change table size exactly (= longest sequence)
  2. the auto margin is 2*n_nodes+2 above that, and pos_embed agrees
  3. an explicit margin is honoured
  4. save_checkpoint records the RESOLVED margin, not the -1 sentinel
  5. generation can run past the longest training sequence, which is the whole
     point -- under margin 0 it is hard-stopped there
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningModelConfig, SudokuReasoningTrainConfig
from spin_gnn.sudoku_tokens import encode_instance
# The sudoku generate(), not sat_transformer2's: same loop, but it stops on
# SOLVED/UNRESOLVED and takes (model, prompt, vocab, max_new_tokens, device).
from spin_gnn.train_sudoku_transformer import (
    build_instances, generate, resolve_max_len_margin, save_checkpoint, train_sudoku_model,
)

N = 12
sudoku_cfg = SudokuReasoningConfig(n_nodes=N, degree=5, n_colors=4, rho=0.3, rho_max=0.95, seed=0)
model_cfg = SudokuReasoningModelConfig(d_model=32, num_layers=2, nhead=2, dim_feedforward=64)


def run(margin):
    cfg = SudokuReasoningTrainConfig(
        n_train_instances=40, n_val_instances=10, batch_size=8, epochs=1,
        device="cpu", max_len_margin=margin,
    )
    return cfg, train_sudoku_model(sudoku_cfg, model_cfg, cfg)


cfg0, r0 = run(0)
longest = r0["model_max_len"]
assert r0["model"].pos_embed.shape[0] == longest, "margin 0 must size pos_embed to the longest sequence"
print(f"[1] margin 0 -> max_len {longest} (the pre-change value)")

cfg_auto, r_auto = run(-1)
expected = longest + 2 * N + 2
assert resolve_max_len_margin(cfg_auto, sudoku_cfg) == 2 * N + 2
assert r_auto["model_max_len"] == expected, (r_auto["model_max_len"], expected)
assert r_auto["model"].pos_embed.shape[0] == expected
print(f"[2] auto   -> max_len {expected} = {longest} + {2 * N + 2}")

cfg_ex, r_ex = run(7)
assert r_ex["model_max_len"] == longest + 7
print(f"[3] margin 7 -> max_len {longest + 7}")

with tempfile.TemporaryDirectory() as d:
    save_checkpoint(d, sudoku_cfg, model_cfg, r_auto["model"], r_auto["vocab"],
                    r_auto["model_max_len"], r_auto["baseline_accuracy"],
                    r_auto["solve_metrics"], r_auto["history"], train_cfg=cfg_auto)
    cfg_json = json.load(open(os.path.join(d, "config.json")))
assert cfg_json["max_len_margin"] == 2 * N + 2, cfg_json["max_len_margin"]
assert cfg_json["max_len"] == expected
print(f"[4] config.json records max_len={cfg_json['max_len']} margin={cfg_json['max_len_margin']}")

# (5) The behavioural claim. Prompt both models with the same held-out instance
# and let them generate freely; only the margined table can emit past
# `longest`. Both models are barely trained, so they babble to their cap --
# which is exactly the condition the margin exists for.
inst = build_instances(sudoku_cfg, 1, 999_000)[0]
toks = encode_instance(inst[0], inst[3], r0["vocab"])
begin = int(np.flatnonzero(toks == r0["vocab"].BEGIN_REASONING)[0])
prompt = toks[: begin + 1]
len0 = len(generate(r0["model"], prompt, r0["vocab"], 10_000, device="cpu"))
lena = len(generate(r_auto["model"], prompt, r_auto["vocab"], 10_000, device="cpu"))
print(f"[5] free generation: margin 0 stops at {len0} (cap {longest}), "
      f"auto reaches {lena} (cap {expected})")
assert len0 <= longest and lena <= expected
assert lena > longest, "margined model never used the headroom -- check the cap is what bound it"
print("ALL CHECKS PASSED")
