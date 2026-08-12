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
- **Kisialiou/Baseline collapse (tasks #3, #4)**: **resolved 2026-08-07, but not the way the
  2026-08-05/06 entries below claimed.** Those entries (signal-to-noise explanation, "reconfirmed
  on two independent runs") are **retracted** — see section 7. Root cause was a real bug:
  `alns_mode_benchmark.py` was calling `op_pair_v1` (random pair selection) instead of
  `backbone_v1` (adaptive weighted) for "baseline" mode, so there was never any real weight update
  happening to observe. With that fixed (plus the separate `BackboneActionSpace` pool-restriction
  bug in section 6, plus a full Kisialiou-paper calibration pass), real adaptive weighting shows
  strong divergence — this actually answers the original June 9 question directly: yes, the
  adaptive mechanism does produce real, non-uniform operator preference, and on large instances it
  strongly favors `random_visit_removal` (59.4%) over `shaw_removal` (8.1%), which is closer in
  spirit to the June 9 "Shaw removal comes to dominate" observation than anything found in the
  buggy 08-05/06 runs. Whether it's the *same* mechanism as June 9 is still unconfirmed, but the
  question "is there real divergence" is now settled: yes. Superseded material below kept for the
  record only — see the `⚠️ SUPERSEDED` banners in
  `output/operator_usage_analysis_20260805_085142/FINDINGS.md` and
  `output/benchmark_modes_matrix_20260805_085142/PER_SIZE_ANALYSIS.md`.

  <details><summary>Original (retracted) 2026-08-05/06 text, kept for audit trail</summary>

  Code check (2026-07-30) confirmed γ=0.2 reaction factor and σ=[33,9,3] are implemented correctly
  per the Kisialiou formula but flagged a missing weight floor as a possible collapse mechanism.
  The Phase 1 destroy/repair usage-evolution check on `benchmark_modes_matrix_20260805_085142`
  found no collapse at all in the current 3×3-restricted pool — explained by signal-to-noise:
  per-restart operator sample size too small for the real quality gap between operators to move
  the weights. "Reconfirmed" 2026-08-06 on an independent re-run. **Both readings were wrong** —
  see above.

  </details>
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

**Update 2026-08-06 (⚠️ SUPERSEDED 2026-08-07 — kept for audit trail, see section 7 for
trustworthy numbers):** benchmark complete (90/90 result files). At the time this looked like the
Phase 1 headline result, but "baseline" in that run was actually running `op_pair_v1` (random pair
selection) under an unrelated bug, and Kisialiou's parameters were pre-calibration — so the
12.68%→8.55%→2.29% shrinking-gap finding and the "Kisialiou loses on 1/15 large datasets" finding
below describe a different, buggy/uncalibrated experiment, not the real baseline-vs-Kisialiou
comparison. Do not cite these numbers. Original text: <details><summary>expand</summary>

No operator collapse (confirmed 2026-08-05). Runtime hypothesis holds — kisialiou_original is
~19-20x slower per restart than baseline at every size. Quality hypothesis is partially wrong:
the cost gap in Kisialiou's favor *shrinks* with size (12.68%→8.55%→2.29% mean gap, small→large)
and Kisialiou loses outright on 1/15 large datasets — the opposite of "Kisialiou pulls further
ahead at scale." Leading explanation: within the fixed time budget, Kisialiou gets far fewer
restarts at large size (~83 total vs baseline's ~713), which may be offsetting its per-restart
quality edge. Also: mean gap >> median gap at every size — a handful of large-outlier datasets
(likely vessel-count-saving cases, consistent with the 2026-03-05 diary note) drive most of the
average advantage, not a uniform improvement.

</details>

**Update 2026-08-07 — see section 7 for the trustworthy replacement of all of the above.** The
immediate next actions are now the open follow-ups listed at the end of section 7, not the list
that used to be here. Phase 0 housekeeping (`git status` cleanup) is still outstanding and still
worth doing before more runs pile up.

## 6. Known issue — BackboneActionSpace pool-restriction bug (found 2026-08-07)

`rl_alns_environment.py` restricts the adaptive engine's operator pool to the same 3×3 set used by
`op_pair_v1`/RL/Exp 3 by reading `action_impl.destroy_operators`/`.repair_operators` and calling
`restrict_operators()`. `BackboneActionSpace` (`backbone_v1`) never filters those lists down,
though — it stores the full registry as-is (~30 destroy variants, 7 repair variants), unlike
`OpPairV1ActionSpace`, which correctly calls the `_base_destroy_repair()` helper
(`rl/modules/actions.py`) to select just the 3 canonical destroy / 3 canonical repair operators. So
`restrict_operators()` got called with the entire pool — a no-op restriction.

**Affected run:** `benchmark_modes_matrix_20260806_213701` (baseline + kisialiou_original,
overnight run). Both modes used genuine adaptive-weighted selection (roulette-wheel, real weight
updates — that mechanism itself is correct) but over the full ~30×7 operator registry, not the
intended restricted 3×3 set. This is a real, valid experiment — just not the one intended.
**Logged as a separate experiment**: "adaptive ALNS over the full operator catalog," not
comparable to the restricted-pool runs (`benchmark_modes_matrix_20260805_085142`,
`benchmark_modes_matrix_20260806_134054`) or to Exp 3. Full note:
`output/benchmark_modes_matrix_20260806_213701/PROVENANCE_NOTE.md`.

**Exp 3 unaffected** — uses `op_pair_v1`'s explicit-pair mechanism directly (verified separately),
which does correctly restrict to the 3×3 pool. That data is clean and stands as-is.

**Resolved answer to the open question above:** it wasn't this bug that caused the flat
`085142`/`134054` operator-usage pattern — it was a *different* bug, `alns_mode_benchmark.py`
using `op_pair_v1` (random selection) instead of `backbone_v1` for "baseline" mode entirely (see
section 7). Both bugs are now fixed.

**Fix applied 2026-08-07:** `BackboneActionSpace` now calls `_base_destroy_repair()` the same way
`OpPairV1ActionSpace` does. The old unrestricted-pool behavior is preserved deliberately as a
separate action module, `backbone_full_v1`, for anyone who wants it.

**Status: ✅ resolved.** See section 7 for the calibrated results this enabled.

## 7. Trustworthy results — calibrated baseline vs Kisialiou + Exp 3 (2026-08-07)

**This section supersedes sections 4-6's 2026-08-05/06 material.** Full source:
`docs/experiments/diary_entry_calibrated_baseline_and_exp3.md`. Three bugs were fixed together to
get here: (1) `alns_mode_benchmark.py` was using `op_pair_v1` (random pair selection) for
"baseline"/"kisialiou" instead of adaptive-weighted `backbone_v1`; (2) `BackboneActionSpace` never
restricted the operator pool to 3×3 (section 6); (3) a full calibration pass against Kisialiou,
Gribkovskaia & Laporte (2018) Table 1 — fixed a cooling-schedule formula bug, corrected γ
(0.2→0.6), σ (33/9/3→25/15/20), δ (10→50/100 by ρ), added paper-calibrated Shaw/Worst/Random-
removal variants, fixed fleet-and-cost-reduction's λmax (8→35), and fixed Kisialiou's improvement
sequence to target the calibrated λmax=35 variant with unlimited (not capped) relocation/swap.
Runs: `benchmark_modes_matrix_20260807_112815` (main, all 3 sizes, 15 datasets/size) and
`benchmark_modes_matrix_20260807_180302` (Exp 3, small only, 15 datasets). Both clean, zero errors.

**Finding 1 — adaptive weighting shows real divergence.** Mean spread across 60 baseline/large
restarts: 62.0pp (median 61.4, max 80.8, 60/60 restarts exceed 20pp) — completely different from
the ~4pp/0-of-60 reported under the pre-fix bug. Destroy-operator share on large:
`random_visit_removal` 59.4%, `worst_removal` 32.6%, `shaw_removal` 8.1%. Open follow-up: why
random removal dominates Shaw removal on large instances (hypothesis: Shaw's now-correct p=100
near-deterministic selection may be too narrow/slow to diversify on large, many-voyage instances).

**Finding 2 — Kisialiou beats baseline ~10-11% on small/medium; loses on large once the
comparison is actually valid (resolved 2026-08-11, see follow-up 1 below for the full story).**
Small: baseline 329,250 vs kisialiou 294,473 (**10.78%** gap, trustworthy, comparable restart
counts 1,643 vs 493). Medium: 676,156 vs 607,586 (**10.09%**, trustworthy, 2,962 vs 104 restarts).
Large at the original 600s budget: -4.67%, **not trustworthy** — only 6/15 datasets even produced
a kisialiou result (the other 9 timed out mid-restart), and those 6 got exactly 1 restart each.
Large at 1800s (3x budget, all 15/15 complete for both modes,
`benchmark_modes_matrix_20260811_095643`): **-2.44%, this time trustworthy** — kisialiou is
marginally *worse* than baseline on large, not better. Root cause: kisialiou's paper-specified
"run improvement operators to full convergence" design makes `deep_swap_v10`/`deep_relocation_v10`
consume 98.9% of its search time on large (many-visit) instances, capping it to 2-3 restarts vs
baseline's ~261 — a structural property of the algorithm at scale, not a bug or a budget artifact.
The "Kisialiou is slow but better, baseline is fast but worse" framing holds on small/medium and
**breaks down completely on large**, where Kisialiou is both slower and not better. See follow-up
1 below and `output/benchmark_modes_matrix_20260811_095643/FINDINGS.md` for full detail.

**Finding 3 — Exp 3: no fixed pair meaningfully beats or loses to the adaptive baseline on small
instances, but the *which* pair wins inverts the plan's prior expectation.** Every one of the 9
fixed D×R pairs lands within ±0.33% of the adaptive baseline; the best, `worst_removal` +
`k_regret_3`, is a statistical tie (+0.05%, nominally better). This supports the original Exp 3
hypothesis ("does switching operators matter, or does picking one good pair suffice?") — switching
buys essentially nothing on small instances. But contrary to the plan's a priori expectation
(`docs/EXPERIMENTS.md`, now corrected) that `shaw_removal` + `k_regret` would win, the data says
the opposite: `worst_removal` pairs rank best, `shaw_removal` pairs rank worst, of all 9. Destroy
operator choice drives ~9x more variation than repair operator choice. Caveat: small-size only —
not yet run on medium/large, and given Finding 1's large-specific destroy-operator preference,
"pick one good pair" should not yet be generalized past small instances. Also: Exp 3's
`analysis.txt` "improvement per second" section undercounts samples for fast-converging pairs
(runtime rounds to 0.0s for up to ~99.8% of `fixed_random_greedy`'s restarts) — a measurement
quirk in that one metric only; `best_cost`/`improvement_pct`, used for all comparisons above, are
unaffected.

**Open follow-ups (the actual next steps for this plan, replacing the stale list in section 5):**
1. Re-test whether Kisialiou's single-restart-per-dataset behavior on `large` is actually fine
   (one-shot-sufficient hypothesis) rather than assuming the time budget needs to change.
   **✅ Resolved 2026-08-11**, both modes, 1800s budget, all 15/15 datasets complete for both —
   `benchmark_modes_matrix_20260811_095643`. Answer: **no**, not one-shot-sufficient in a good
   way — kisialiou is structurally restart-starved on large by design, not by budget accident.
   `deep_swap_v10`/`deep_relocation_v10` (unlimited, run-to-convergence, per the paper's actual
   spec) consume 98.9% of kisialiou's search time (947ms/202ms per call vs sub-4ms for everything
   else), capping it to 2-3 restarts/dataset (37 total) vs baseline's ~261/dataset (3,920 total),
   even with 3x the time budget and zero truncation. Net result: **kisialiou reverses from ~+10-11%
   ahead (small/medium) to -2.44% behind on large** (wins 9/15 datasets ~+2.6% avg, loses 6/15
   ~-10.0% avg). Not a bug — inherent to the algorithm as specified. Full writeup:
   `output/benchmark_modes_matrix_20260811_095643/FINDINGS.md`. This is now the final, trustworthy
   large-size number, replacing both the 600s-budget -4.67%/n=6 figure and the earlier
   "not necessarily a problem to fix" speculation below.
2. Run Exp 3 on medium and large to check whether "pick one pair" still holds at scale.
3. Investigate why `random_visit_removal` dominates adaptive selection on large (59% share) — real
   operator-quality signal or a remaining calibration artifact.
4. `docs/EXPERIMENTS.md`'s Exp 3 "expected finding" text has been updated to match the actual
   result (done 2026-08-07).
5. Phase 0 housekeeping (`git status` cleanup) is still outstanding, independent of the above.
6. Training-regime open decision (section 4) is still open and still blocks Phase 2 (RL
   retraining) — none of today's fixes touch that question.
