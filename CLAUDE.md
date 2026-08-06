# PSVPP — working notes for Claude Code sessions

This repo implements ALNS (Rust core, PyO3 bindings) enhanced with RL (Python/Gymnasium/
Stable-Baselines3) for the Platform Supply Vessel Pickup and Delivery Problem, for a PhD thesis.
Environment: conda env `py312`. Build the Rust extension with `maturin develop --release` in
`rust_alns/` before running anything. See `README.md` and `docs/PROJECT_STRUCTURE.md` for full
repo layout, `docs/DEVELOPMENT.md` for setup, `docs/EXPERIMENTS.md` for the experiment log.

## Active plan

The current work is scoped by **`docs/PAPER1_ACTION_SPACE_PLAN.md`** — read that file first in any
new session on this project. It's the experiment plan for paper 1 ("RL enforced ALNS for PSVPP",
tracked in Notion), which is limited in scope to RL action-space design (destroy/repair/improvement
operator selection). Status and the same plan are also mirrored in Notion under "RL enforced ALNS
for PSVPP" — search Notion for "Paper 1 — Action Space Experiment Plan" if Notion access is
available and more recent status is needed than what's written here.

As of 2026-07-30, we're at the start of Phase 1/2 of that plan: establishing solid baseline vs.
Kisialiou-ALNS benchmark results (no RL yet) on a grown test set before retraining RL on
medium/large instances. Two scripts were extended to support this and should already have the
flags described below — if they don't, something regressed:

- `scripts/generate_datasets.py` gained `--only-split {train,test}` so test-set growth doesn't
  silently inflate train data (which would invalidate already-trained models' training set).
- `scripts/run_experiment_kisialiou_modes.py` gained `--modes` (comma-separated subset of its 6
  mode labels) so a `baseline,kisialiou_original`-only run doesn't waste time on the other 4
  Kisialiou variants.

Next concrete step (unless already done): generate 15 test datasets per size and run baseline +
kisialiou_original across small/medium/large — see `docs/PAPER1_ACTION_SPACE_PLAN.md` section 5
for exact commands and the reasoning behind them (including a note that the adaptive-weight update
has no weight floor, which is a candidate explanation if Kisialiou collapses to Shaw-removal-only
again). After that benchmark completes, check for operator collapse with
`scripts/operator_usage_analysis.py` before moving on to Phase 2 (RL retraining).

## Long-running jobs

Benchmark/training runs here can take hours to days. If a run needs to survive terminal/session
restarts, detach it (`nohup ... &`, or tmux) rather than relying on a single Claude Code session
to keep it alive — session background tasks get killed on session end. Progress/results land under
`output/<run_id>/` and `runs/benchmarks/<run_id>/manifest.json`.
