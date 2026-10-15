# LOG

Dated history for the k-SAT reasoning-trace project. Do not read this file
in full from the start of a session -- search it (`grep`/`grep -n`) for
whatever you need. At most 3 lines appended per session.

- 2026-10-14: Pre-norm depth/nhead scan results written up properly
  (`notebooks/results_prenorm_scan.ipynb`, commit `12d5d71`): confirms
  pre-norm fixes the post-norm depth collapse and best solve_rate is
  12.3% (noisy, repeat run got 4.3%) at d=128,h=2,L=16,lr=3e-4.
- 2026-10-14: Wired `[SATISFIED]`/`[REACTIVATED]` clause-status tokens
  (opt-in, backward-compatible) and tested them (Slurm `47896056`, 2
  seeds) against the baseline, same config/eval -- NEGATIVE RESULT, both
  runs (0.0%, 0.67%) below both baseline runs (12.3%, 4.3%); not adopted.
- 2026-10-14: Introduced STATUS.md/LOG.md tracking convention for this
  project (first time either file existed).
- 2026-10-15: Seeded model init (`torch.manual_seed`, was never called) --
  bit-identical on CPU, but GPU run-to-run variance only shrank (2.9x ->
  2.1x spread at the best config), not eliminated (cuDNN nondeterminism
  unaddressed); not a solve_rate improvement, just partial repro fix.
