# Paper 1 — RL Action-Space Design for ALNS (PSVPP): Experiment Plan

Drafted 2026-07-30. Scope: the first target paper ("RL enforced ALNS for PSVPP" in Notion) is
restricted to **action-space design** — how much of the ALNS decision (destroy operator, repair
operator, improvement operator, or all three) should be handed to the RL policy. The claim to
prove is that RL-controlled operator selection beats the adaptive-weighted ALNS baseline and
Kisialiou's fixed pipeline, primarily on **completion time** (time/iterations to reach a given
solution quality, and total wall-clock under a matched time budget), with solution cost as a
secondary metric. Everything below builds on what's already in `docs/EXPERIMENTS.md`, the
`Experiment Investigation Tasks` and `Research diary` pages in Notion, and the current state of
the repo (branch `feature/rl-alns-improvements`, last commit `ada50ff`).

## 1. Where things stand

Four experiments are already defined and have been run once, on the **small** dataset only:

- **Exp 0** (`op_pair_v1`) — RL picks destroy × repair only (9 actions), no improvement operator.
- **Exp 1** (`op_triple_v1`) — RL picks destroy × repair × improvement jointly (36 actions).
- **Exp 2** (`improvement_chain_v1`) — adaptive backbone handles destroy/repair; RL only decides
  which improvement operator to chain in (up to 3 in a row) before a forced backbone step.
- **Exp 3** — fixed-pair benchmark (no RL), sanity check that adaptive selection beats any single
  fixed destroy/repair pair.

Baselines are `backbone_v1` (adaptive weighted ALNS, no improvement) and `kisialiou` (fixed
improvement sequence after every step). As of `ada50ff` (2026-06-08), baseline and Kisialiou are
restricted to the same 3×3 destroy/repair pool as the RL experiments, so the comparison is
apples-to-apples on operator choice.

The June 9 diary entry and the "Experiment Investigation Tasks" Notion page recorded the blocking
problem: **every mode (RL and baselines) collapses to a narrow operator subset**, and the RL
models were confirmed to have been trained only on the small dataset. On small instances there is
rarely room to cut a vessel, so voyage-number-reduction never gets a strong reward signal and the
policy over-weights swap/relocation instead. Five follow-up tasks were logged; only #1 (confirm
training-data size) was marked done.

## 2. What I found in the code that changes the plan

Two things from `scripts/run_experiment_rl_improvement_mvp.py` (the script actually used for
exp0–2) make the retraining task cheaper than the Notion notes assumed:

- `data/generated/{medium,large,mixed}` and the matching `data/processed/alns/{medium,large,mixed}`
  splits **already exist** — no dataset generation work needed, only a size argument change.
- The script's `--sizes` flag already supports **joint multi-size train/eval** (comma-separated,
  overrides `--size`), and there's an existing but currently-unused `--anti-repeat-penalty` flag —
  a reward penalty for repeating the same improvement operator without finding a new best. That is
  exactly the mechanism task #5 (Exp 2 repeating swap 3× in a row) asked for; it just needs to be
  turned on and tuned, not built.

This means Phase 2 below is a parameter change plus reruns, not new engineering.

## 3. Phased plan

### Phase 0 — Housekeeping (before any new runs)
The working tree currently has uncommitted deletions (old `output/benchmark_small5_tb120s_*`
artifacts) and an untracked `output/alns_sequence_grid_20260316_195103__...` folder left over from
March. Commit or discard these first so new run folders aren't mixed in with stale ones, and so
`git status` stays a reliable log of what's actually part of the paper's evidence base.

### Phase 1 — Cheap fixes on the small dataset (fast iteration before committing to big runs)
1. Turn on `--anti-repeat-penalty` for Exp 2 and sweep 2–3 values on the small dataset (minutes
   per run at ~1,400 steps/s). Confirm the chain stops repeating the same improvement operator
   3× before moving to Phase 2 — no point retraining on medium/large with a reward bug still in
   Exp 2.
2. Reuse the existing `operator_usage_analysis.py` / `plot_sequence_grid_dots.py` tooling to plot
   weight evolution for Kisialiou and the baseline over iterations (tasks #3 and #4). This is
   diagnostic, not a fix — the goal is to confirm whether Shaw-removal/random-visit-removal
   dominance is legitimate adaptive convergence or a weight-update bug (e.g. reaction factor too
   high, no weight floor). If it's legitimate, document it as expected ALNS behavior and move on;
   only fix it if it's actually distorting the baseline's fairness as a comparison point.

### Phase 2 — Retrain Exp 0, 1, 2 on medium and large instances
Rerun the same script with `--sizes medium,large` (and probably a `--sizes small,medium,large`
joint run too, since the paper's claim should hold across scales, not just on the size where the
bug was found). Key question carried over from the investigation tasks: does voyage-number-
reduction come back as a preferred operator once vessel-count reduction is actually achievable?
If yes, the operator-collapse hypothesis is confirmed and the fix is "train on realistic instance
sizes." If not, the reward/state design needs a deeper look before this paper can claim RL adds
value.

### Phase 3 — Bigger-proof evaluation
This is the part that turns "preliminary result" into "proof" for the paper. Concretely:
- Expand past the current 5 datasets × 5 seeds = 25 samples per experiment — at minimum keep that
  density per size (small/medium/large) rather than small-only, so the headline result isn't
  resting on 25 samples from one instance size.
- Report results **stratified by instance size**, not just pooled, since the small-dataset bias
  is exactly what caused the collapse — pooling could hide a regression at one scale.
- On completion time specifically: use the existing `time_to_best_solver_seconds` /
  `wall_runtime_seconds` fields already logged per restart, under the existing `match_kisialiou`
  time-fair eval mode, so RL and baselines get identical wall-clock budgets per dataset/seed.
- Add paired statistical testing (paired by dataset+seed) — Wilcoxon signed-rank or paired t-test
  on both cost-gap and time-to-best deltas, plus a simple win-rate count and a confidence interval
  on the mean gap. "RL wins on average" and "RL wins on 22/25 paired instances with p<0.05" are
  different strengths of evidence, and the second is what a reviewer will ask for.

### Phase 4 — Write-up
Update `docs/EXPERIMENTS.md` with final results across sizes, regenerate the operator-selection
and runtime-progress plots for the paper figures, and draft the results section once Phase 3
numbers are in.

## 4. Open decisions

- ~~Test dataset count for this stage~~ — **resolved 2026-07-30: 15 test datasets per size**
  (small/medium/large), generated via `--only-split test` so train data (and therefore already-
  trained models' training set) is untouched.
- **Training regime**: one model per size (small/medium/large separately) vs. one joint model
  trained on `--sizes small,medium,large` together. Separate models test the hypothesis more
  cleanly; a joint model is closer to what you'd actually deploy and is cheaper to produce (one
  training run instead of three). Still open — decide before Phase 2 (RL retraining) starts.
- **Kisialiou/Baseline collapse (tasks #3, #4)**: **updated 2026-08-05.** Code check (2026-07-30)
  confirmed γ=0.2 reaction factor and σ=[33,9,3] are implemented correctly per the Kisialiou
  formula but flagged a missing weight floor as a possible collapse mechanism. The Phase 1
  destroy/repair usage-evolution check on `benchmark_modes_matrix_20260805_085142` found **no
  collapse at all** in the current 3×3-restricted pool (baseline and kisialiou_original both stay
  flat at ~33% per operator, no within-restart drift) — explained by signal-to-noise: per-restart
  operator sample size (~150-170 applications) is too small for the real quality gap between
  operators (~2.7% of one std dev) to move the weights. Full writeup:
  `output/operator_usage_analysis_20260805_085142/FINDINGS.md`. This looks like it resolves tasks
  #3/#4 for the *current* restricted-pool setup — no fix needed, document as expected behavior —
  but it doesn't yet explain the original June 9 observations (Kisialiou→Shaw-removal-only,
  baseline→random-visit-removal), which most likely came from the pre-`ada50ff` *unrestricted*
  30-destroy/7-repair pool where real quality gaps would be easier to detect. Not yet confirmed
  against the actual June 9 run configs — worth a quick check before fully closing #3/#4.
- **Sample size target for the later RL "bigger proof" eval (Phase 3)**: still open — revisit once
  Phase 1 baseline/Kisialiou results are in.

## 5. Immediate next action (as of 2026-07-30)

Phase 1 of this plan starts now: grow test sets and benchmark baseline vs. Kisialiou only (no RL
yet), across all three sizes, saving metrics for later comparison and checking whether operator
selection collapses on the way. Exact commands:

```bash
# 1. Grow test sets to 15 per size (train untouched — deterministic seeds keep samples 1-5 identical)
python scripts/generate_datasets.py --samples 15 --sizes small medium large --only-split test

# 2. Run baseline vs Kisialiou only, across all three sizes, 15 datasets each
python scripts/run_experiment_kisialiou_modes.py --modes baseline,kisialiou_original \
  --sizes small,medium,large --datasets-per-size 15
```

Both `--only-split` and `--modes` were added to these scripts on 2026-07-30 — they didn't exist
before. Results land in `output/benchmark_modes_matrix_<timestamp>/` and
`runs/benchmarks/<run_id>/manifest.json`. Afterward, run `scripts/operator_usage_analysis.py`
against that run to check for operator collapse (expected: baseline should be fast but weaker on
large instances; Kisialiou should be slower but higher quality — if either instead collapses to a
1-2 operator policy, check the weight-floor hypothesis above before trusting the numbers).

Once this benchmark is done and saved, proceed to Phase 1's remaining diagnostic step (Exp 2
anti-repeat-penalty / weight-evolution plots) and then Phase 2 (RL retraining on medium/large).

**Update 2026-08-06: benchmark complete (90/90 result files), Phase 1 headline result in.** See
`output/benchmark_modes_matrix_20260805_085142/PER_SIZE_ANALYSIS.md` (mirrored in Notion). No
operator collapse (confirmed 2026-08-05). Runtime hypothesis holds — kisialiou_original is
~19-20x slower per restart than baseline at every size. Quality hypothesis is partially wrong:
the cost gap in Kisialiou's favor *shrinks* with size (12.68%→8.55%→2.29% mean gap, small→large)
and Kisialiou loses outright on 1/15 large datasets — the opposite of "Kisialiou pulls further
ahead at scale." Leading explanation: within the fixed time budget, Kisialiou gets far fewer
restarts at large size (~83 total vs baseline's ~713), which may be offsetting its per-restart
quality edge. Also: mean gap >> median gap at every size — a handful of large-outlier datasets
(likely vessel-count-saving cases, consistent with the 2026-03-05 diary note) drive most of the
average advantage, not a uniform improvement. Remaining Phase 1 items before Phase 2:
1. Exp 2 anti-repeat-penalty check (still not done — see Phase 1 step 1 above).
2. Phase 0 housekeeping is still not done — `git status` still shows the stale deleted
   `output/benchmark_small5_tb120s_*` files and other uncommitted changes.
3. Consider whether to re-run the large-size benchmark with a longer time budget to test the
   "fewer restarts" explanation before concluding the quality-at-scale finding is real.
4. Resolve the training-regime open decision (below) before starting Phase 2.
