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
- **Exp 4** (planned, 2026-08-19) — RL gets full control of the search: destroy/repair pair choice,
  improvement operator choice, and when to switch between them, with no scripted backbone engine
  and no fixed forced-return cap. See section 8 for the full design.
- **Exp 5** (planned, 2026-08-19) — not an RL experiment: numerically quantify whether a given
  operator application explored or exploited, using partition-distance (Adjusted Rand Index /
  Variation of Information on voyage membership) and per-operator fitness-landscape/autocorrelation
  characterization. Feeds into Exp 4's state/reward design. See section 9.

Baselines are `adaptive_weighted_v1` (adaptive weighted ALNS, no improvement) and `kisialiou` (fixed
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
  `adaptive_weighted_v1` (adaptive weighted) for "baseline" mode, so there was never any real weight update
  happening to observe. With that fixed (plus the separate `AdaptiveWeightedActionSpace` pool-restriction
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
- **Sample size target for the later RL "bigger proof" eval (Phase 3)**: **direction resolved
  2026-08-18** (during Exp 0 rerun discussion), exact numbers still open. Relative to the current
  5-datasets × 5-seeds convention, prefer **more test instances per size, fewer eval seeds** —
  not the reverse. Reasoning:
  - Each eval episode already runs 100+ restarts internally for baseline/RL, each with a fully
    distinct RNG seed (`restart_seed = eval_seed + restart_idx * 1009`, see
    `scripts/run_experiment_rl_improvement_mvp.py`'s `_restart_seed()`). A single eval seed's
    "best of restarts" is already a near-converged per-instance estimate — extra eval seeds
    mostly re-explore the same territory and tighten noise around an already-stable number.
  - What eval seeds do *not* cover is structural variation between problem instances (topology,
    demand pattern, vessel mix) — that's what the paper's "generalizes across the instance
    population" claims actually depend on, and it's exactly what more distinct test instances add.
  - **Asymmetry to flag**: this reasoning is strongest for baseline/RL. Kisialiou only completes
    2-5 restarts per episode (each restart is expensive — full improvement chain every iteration),
    so its per-(dataset, seed) result is much noisier and benefits more from seed replication than
    baseline/RL's. The eval script currently shares one `--eval-seeds` list across all three modes
    (baseline, kisialiou, RL), so seed count can't be cut per-mode without a script change — don't
    reduce seed count all the way to 1 for this reason; keep a floor of ~2-3 seeds.
  - Exact final numbers (e.g. "12 datasets × 3 seeds") are intentionally not picked yet — wait
    until Exp 0's current rerun (small/medium/large, 5 datasets × 5 seeds each) finishes and those
    results are in hand, per this bullet's original "revisit once Phase 1 results are in"
    condition. This entry captures the *direction* only, so it isn't lost in the meantime.

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
that used to be here. Phase 0 housekeeping is now done (`git status` no longer shows the stale
`benchmark_small5_tb120s_*` deletions as of 2026-08-14).

## 6. Known issue — AdaptiveWeightedActionSpace pool-restriction bug (found 2026-08-07)

`rl_alns_environment.py` restricts the adaptive engine's operator pool to the same 3×3 set used by
`op_pair_v1`/RL/Exp 3 by reading `action_impl.destroy_operators`/`.repair_operators` and calling
`restrict_operators()`. `AdaptiveWeightedActionSpace` (`adaptive_weighted_v1`) never filters those lists down,
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
using `op_pair_v1` (random selection) instead of `adaptive_weighted_v1` for "baseline" mode entirely (see
section 7). Both bugs are now fixed.

**Fix applied 2026-08-07:** `AdaptiveWeightedActionSpace` now calls `_base_destroy_repair()` the same way
`OpPairV1ActionSpace` does. The old unrestricted-pool behavior is preserved deliberately as a
separate action module, `adaptive_weighted_full_v1`, for anyone who wants it.

**Status: ✅ resolved.** See section 7 for the calibrated results this enabled.

## 7. Trustworthy results — calibrated baseline vs Kisialiou + Exp 3 (2026-08-07)

**This section supersedes sections 4-6's 2026-08-05/06 material.** Full source:
`docs/experiments/diary_entry_calibrated_baseline_and_exp3.md`. Three bugs were fixed together to
get here: (1) `alns_mode_benchmark.py` was using `op_pair_v1` (random pair selection) for
"baseline"/"kisialiou" instead of adaptive-weighted `adaptive_weighted_v1`; (2) `AdaptiveWeightedActionSpace` never
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
5. ✅ Phase 0 housekeeping done (git tree clean of the old stale deletions as of 2026-08-14).
6. Training-regime open decision (section 4) is still open and still blocks Phase 2 (RL
   retraining) — none of today's fixes touch that question.

## 8. Exp 4 — full RL control of explore/exploit (planned 2026-08-19)

**Motivation.** Exp 2 (`improvement_chain_v1`) is being kept exactly as implemented — its D×R
phase is always the fixed adaptive-weighted backbone, and the forced-return-to-backbone trigger is
a flat count cap (`chain_max_consecutive_improvements=3`), not stagnation-driven. Exp 4 is the
natural next step after Exp 2: give RL full control of the entire decision — which destroy/repair
pair to explore with, which improvement operator to exploit with, and when to switch between the
two — instead of delegating the "explore" side to a scripted engine. Framing (2026-08-19
discussion): ALNS destroy/repair = diversification, `voyage_number_reduction`/`deep_relocation`/
`deep_swap` = intensification, `fleet_and_cost_reduction` = dual-purpose (both); a well-trained
policy shouldn't converge to a single operator or pair, it should alternate
explore→exploit→escape-on-stagnation→exploit, mirroring the diversification/intensification
balance in classical metaheuristics — structurally close to Variable Neighborhood Search's
shake/local-search loop.

**Risk flagged up front.** A naive reward (raw cost-reduction / time) is expected to collapse to a
single operator or fixed sequence — not hypothetical, it's what already happened in Exp 0
(`op_pair_v1` converged to `random_visit_removal_kisialiou_*` on small/medium and
`worst_visit_removal_kisialiou_large_v1` on large; see the 2026-08-19 Exp0 findings in
`docs/experiments/diary_entry_sa_reward_decoupling_exp0_lock.md` and the matching Notion page).
Exp 4's design has to actively counter this, not just hope RL finds the right balance on its own.

**Diagnostics to run first (cheap, no new training needed):**
1. `PPO(...)` in `rl/train_alns_rl.py:388` is called with no `ent_coef` — Stable-Baselines3 defaults
   this to `0.0`, i.e. zero entropy regularization during training currently. Sweep `ent_coef` in
   `{0.01, 0.02, 0.05}` — cheapest, highest-leverage lever against collapse, and applicable to
   existing experiments too, not just Exp 4.
2. Eval defaults to `deterministic=True` everywhere (`rl/cli/common.py`, `evaluate.py`, `train.py`).
   `evaluate.py` already supports `--stochastic`. Rerun eval on an existing Exp0/1/2 model with
   `--stochastic` to check whether "collapse" is a training-time reality or an eval-time argmax
   artifact, before building anything new.
3. `rl_alns_environment.py` already tracks `_latest_policy_entropy` / `set_policy_statistics()` —
   check what's already logged (training callback CSV/TensorBoard) before building new
   instrumentation; the collapse diagnosis may already be sitting in old run logs.

**Architecture options (increasing effort):**
1. *Flat unified action space* — union `op_pair_v1`'s 9 D×R actions with the 4 improvement
   operators into one 13-way discrete action, no forced backbone, no cap. Minimal code change
   (combine two existing action registries). No explicit "phase" signal — inferred after the fact
   from which of the 13 actions got picked.
2. *Hierarchical/two-level policy* — a high-level head picks a mode (explore/exploit, or a
   continuous intensity in [0,1]) each step, a low-level head picks the operator conditioned on
   that mode (options framework, Sutton/Precup/Singh). More engineering, but the high-level head is
   effectively a *learned* exploration/exploitation score, rather than one hand-defined from
   cost/time (which is expected to collapse, per Exp 0).

**Reward/credit fixes (needed regardless of architecture — this is the core of what prevents
collapse):**
- *Segment-level credit assignment* — mirrors the classical RW/Kisialiou segment scoring (σ1/σ2/σ3,
  Ropke & Pisinger; Eq 1 in Johnn et al. 2024, already reviewed this session) already familiar from
  this project's calibration work: accumulate credit over a K-step window and attribute a fraction
  of the segment's best improvement back to the step that opened it, including explore steps that
  themselves show zero immediate ΔF. Fixes the structural bias against explore actions under a
  purely myopic per-step reward.
- *Sliding-window diversity/entropy bonus* — `reward += λ · H(action distribution over last N
  steps)`. Not a new idea — already proposed once in this project's early Notion brainstorm
  ("Diversity bonus: entropy of recent actions"), just not yet built.
- *Stagnation-coupled shaping* — let the marginal value of another exploit step decay with
  `stagnation_count` (already computed in the env) while an explore step's value rises
  correspondingly. Middle ground between "hope raw reward discovers switching on its own" (the
  flagged risk above) and "hand-script the switch" (Exp 2's fixed cap, which stays as-is).

**Proposed build order (cheapest first):**
0. Diagnostics above — `ent_coef` sweep + `--stochastic` eval rerun on an existing model. No new
   experiment infrastructure needed; may already reframe or resolve the collapse concern.
1. Flat 13-way action space + entropy bonus + segment-level credit — two independent,
   well-understood levers, still simple to ablate separately.
2. Hierarchical mode/operator split — only if (1) still collapses despite both fixes.
3. Intrinsic novelty bonus on solution-structure (RND/count-based novelty) — research-grade, likely
   out of scope for Paper 1 unless (1)/(2) both fail.

**Status:** planned, not started. Exp 2 stays exactly as currently implemented — Exp 4 is
additive, not a replacement.

## 9. Exp 5 — quantifying exploration vs. exploitation numerically (planned 2026-08-19)

**Quick recap — read this first if you've forgotten the details.** Two separate tools, answering
two different questions, plus 3 ways to use either one:

- **Tool A — "how different are two solutions?"** Look at which voyage each installation is
  assigned to, before vs. after a move (ignore visit *order* within a voyage — see why below).
  Count how much that assignment changed, using a standard metric from cluster comparison
  (Adjusted Rand Index). Big change = the operator explored (reshuffled the packing). Tiny/no
  change = the operator exploited (polished within the existing packing).
- **Tool B — "how disruptive is an operator, on average?"** Apply one operator over and over from
  the same starting solution and watch the cost value: does it jump around a lot, or settle into a
  smooth trend? Jumpy cost = explorative operator. Smooth cost = exploitative operator. Only needs
  cost numbers, nothing about solution structure — simpler to build, and not affected by the
  sequence issue below.
- **Why not also compare visit order within a voyage:** there's an always-on step that
  re-optimizes order after every change (a mini TSP solve), so order ends up near-optimal no matter
  which operator caused the change — comparing order wouldn't tell you anything new beyond what
  Tool A already shows.
- **3 ways to use either tool, easiest first:**
  1. *Just look* — run it on data you already have (Exp0/1/2 results). No coding into RL. Purest
     sanity check: does it confirm the 4 operator groups (explore / dual / exploit) from the
     2026-08-19 discussion?
  2. *Feed it to RL as input* — give the policy a number like "how far am I from my best solution
     right now," replacing the current time-based `stagnation_count`.
  3. *Reward RL for it* — small bonus when the policy makes a genuinely different move (per Tool A),
     instead of small, useless tweaks.

Full detail below.

**Motivation.** Exp 4 wants RL to balance explore/exploit itself, but "which operator got picked"
(action-choice entropy, as floated in Exp 4's build order) is only a weak proxy for whether the
*solution* actually moved through the search space — an agent could pick one operator index every
time and still explore heavily if that operator is inherently disruptive, or vary its operator
choice while producing near-identical solutions. Exp 5 is a numeric, literature-grounded way to
measure the real thing: how far a solution moved, and how disruptive a given operator is on
average. Two complementary approaches, both adjusted below after a same-day follow-up discussion
caught a real issue with the first one.

**Correction to the initial framing.** The project's voyage-improvement step re-optimizes visit
order within a voyage (a small TSP) after every membership change — confirmed in code
(`rust_alns/src/utils/tsp_solver.rs`) — pushing in-voyage sequence to near-optimal regardless of
which operator caused the change. Consequence: sequence/order-based distance metrics (Hamming
distance, Positional Jaccard distance, Kendall's Tau distance — the family that would normally
cover the "within-voyage ordering" layer) are likely **not an independent signal** in this system —
sequence distance is mostly a downstream, near-deterministic consequence of the partition (which
installations ended up in which voyage), not something operators independently fight over.
**Demoted to an optional diagnostic only** (e.g. to check whether the TSP sub-solver itself lands in
different local optima for the same customer set — a narrow, separate question, not the main point
of Exp 5).

**Primary metric: partition/assignment distance.** The real independent degree of freedom in this
solution representation is installation→voyage(+vessel) membership — unaffected by the
sequence-reoptimization issue above. Compare two solutions via Adjusted Rand Index (Hubert &
Arabie, 1985, "Comparing Partitions," *Journal of Classification*) or Variation of Information
(Meilă, 2003, "Comparing Clusterings by the Variation of Information") on that membership
assignment. Directly measures "did the packing structure change" — the testable prediction is that
`voyage_number_reduction`/`fleet_and_cost_reduction` should move this a lot and
`deep_swap`/`deep_relocation` should barely touch it; not yet verified.

**Secondary approach: fitness-landscape/autocorrelation characterization — robust to the whole
representation question above.** Weinberger (1990, "Correlated and Uncorrelated Fitness Landscapes
and How to Tell the Difference," *Biological Cybernetics*); Reeves (1999, "Landscapes, Operators
and Heuristic Search," *Annals of Operations Research* — does essentially this exact study, six
operators/four distance metrics, for a different scheduling problem); Merz & Freisleben
(2000/2001, fitness-landscape + memetic-algorithm papers for TSP/QAP). Method: apply one operator
repeatedly from a fixed starting solution, record only the resulting objective-value (cost) trace
— never touches solution structure directly — and compute its autocorrelation / correlation
length. Short correlation length = disruptive/explorative operator; long correlation length =
smooth/exploitative operator. Because this only needs the cost trace, the sequence-reoptimization
concern above doesn't apply to it at all — safe to build first, and needs no RL/training
infrastructure, just the existing Rust engine called in a loop.

**Three ways to actually use either metric — additive, not a replacement for existing state
features, and not mutually exclusive:**
1. *Pure offline analysis on already-completed runs* — no training/pipeline changes at all. If
   per-iteration solution snapshots (voyage membership) are already logged for Exp0/1/2 runs,
   compute ARI-distance-per-step and correlation-length-per-operator directly on existing data, to
   test the 4-group operator taxonomy (from the 2026-08-19 explore/exploit discussion) empirically
   before deciding whether to wire anything into training. **Cheapest starting point** — check
   first whether existing logs capture full voyage membership per iteration or only scalar
   cost/operator-id metrics (per-iteration logging infrastructure exists —
   `scripts/run_experiment_rl_improvement_mvp.py`, `rl/rl_alns_environment.py` — but whether it
   stores full solution snapshots needs confirming before assuming this is free).
2. *State feature* — e.g. "ARI-distance from the last accepted-best solution" as a richer,
   structurally-grounded replacement/supplement for the purely iteration-count-based
   `stagnation_count` already in `features_improvement_v1`/`v3`.
3. *Reward shaping* — a bonus proportional to genuine structural novelty (ARI-distance-based),
   refining the cruder action-choice-entropy bonus floated in Exp 4; or a stagnation-triggered
   bonus toward operators with a known-short correlation length (from the offline landscape
   characterization) once the search is stuck.

**Proposed order:** (1) offline analysis on existing Exp0/1/2 data first — cheapest, answers the
taxonomy question, no new runs, contingent on confirming solution snapshots are actually logged;
(2) standalone per-operator correlation-length characterization — new but training-free; (3) only
if (1)-(2) show a strong, useful signal, wire ARI-distance into Exp 4's state/reward design.

**Status:** planned, not started.

## 10. Known issue — train/eval iteration-scaling mismatch in `progress`/`normalized_stagnation`
(found 2026-08-20, during Exp1 analysis)

`rl/modules/state.py:77,217` compute `progress` and `normalized_stagnation` as
`raw_count / env.max_iterations`. Training runs use `max_iterations=50`; evaluation runs
(`scripts/run_experiment_rl_improvement_mvp.py`, `match_kisialiou` mode) use `300` per restart —
a 6x different denominator for the same raw iteration count. Concretely: at eval iteration 25,
`progress=0.083`, but at training iteration 25 (out of a 50-step episode), `progress` was already
0.5 — the same raw count means something completely different at train vs. eval time. Past
training's 50-iteration horizon, eval states fall entirely outside anything the policy was ever
trained on for these two features.

**Found via**: Exp1 (`op_triple_v1`) operator-usage deep dive (2026-08-20) — diagnosed as the root
cause of why the trained policy makes one early decision (destroy/repair/improvement bundle) and
then repeats it mechanically for the rest of a 300-iteration restart regardless of whether it's
working. Full mechanism and evidence in
`docs/experiments/diary_entry_exp1_op_triple_results_and_operator_usage.md`, section 8 and Overall
conclusion.

**Not Exp1-specific.** Affects every experiment using `features_improvement_v1`/`v3` state
encoders whenever train `max_iterations` ≠ eval `restart_max_iterations` — i.e. Exp0-2 as trained
so far, and would affect Exp 4/5 too if built on the same state modules without a fix.

**Status: ✅ fixed 2026-08-20.** Added an explicit `progress_horizon` parameter to
`ALNSEnvironment`, independent of `max_iterations` (which still legitimately controls actual
episode/restart length); defaults to `max_iterations` when unset, so nothing changes for any
existing caller that doesn't set it. Threaded a single shared value (new `--progress-horizon`
flag in `run_experiment_rl_improvement_mvp.py`, defaulting to `--eval-restart-max-iterations`)
through the real training env, the diagnostic/reference env, and all three eval episode functions
(RL, baseline, Kisialiou). Verified: training-shaped (`max_iterations=50`) and eval-shaped
(`max_iterations=300`) envs with the same `progress_horizon` now produce identical
`progress`/`normalized_stagnation` values for the same raw iteration/stagnation count; old mismatch
reproduces exactly when `progress_horizon` is left unset. Full train+eval smoke test passed.

**Does not retroactively fix Exp0/Exp1's already-trained models** — their learned behavior is
calibrated to the old mismatched scale regardless of this code change. Retraining is required to
actually realize the fix; this only makes new training/eval runs consistent going forward. Next
retraining pass (Exp1 large, any Exp2 retrain, Exp 4/5) should use this flag.

## 11. Known issue — Kisialiou episode logic duplicated across two scripts, one copy stale
(found & fixed 2026-08-20)

`scripts/run_experiment_rl_improvement_mvp.py`'s `run_kisialiou_episode` and
`scripts/alns_mode_benchmark.py`'s `run_episode` each independently built the `ALNSEnvironment`
for a Kisialiou episode. They drifted: the mvp script's copy passed `action_module=args.
action_module` (whichever RL model it was being compared against) into Kisialiou's env
construction, so Kisialiou's destroy/repair pair got fixed to whatever index 0 of the RL model's
action space decoded to, for the whole episode — non-adaptive, silently wrong. `alns_mode_
benchmark.py`'s copy always used `action_module="adaptive_weighted_v1"` regardless of any RL
model, and was never affected. This is exactly the failure mode of having the same logic
implemented twice: one copy can regress while the other stays correct, with nothing forcing them
back in sync.

**Fix**: extracted the shared baseline/Kisialiou episode-running logic (`EpisodeSummary`,
`run_baseline_episode`, `run_kisialiou_episode`, cache helpers, restart-stat helpers) into
`rl/episode_runner.py`. `run_experiment_rl_improvement_mvp.py` now imports from there instead of
defining its own copies; its call site was also fixed to pass `args.baseline_action_module`/
`args.baseline_state_module` into `run_kisialiou_episode`, decoupling Kisialiou's operator
selection from whichever RL model it's being compared against. `alns_mode_benchmark.py` was left
as-is (different single-run, no-restart-loop design feeding the actively-running Phase 1
benchmark/`operator_usage_analysis.py` pipeline with its own established log-filename
convention) — it was already correct and never had this bug, so unifying it too would add risk
without fixing a live defect; worth revisiting if it drifts in the future.

**Also changed, per explicit instruction**: cache-only comparison is now the *default* for
`--run-eval`, not an opt-in flag. `run_experiment_rl_improvement_mvp.py` gained:
- `--baseline-kisialiou-only` — runs just the baseline+Kisialiou loop over the resolved test
  datasets × `--eval-seeds` (no PPO training/model at all) and writes/updates the cache. This is
  the "run it once" entry point.
- A canonical shared cache path (`runs/baseline_kisialiou_cache/cache.json`) used whenever
  `--baseline-kisialiou-cache` isn't explicitly given.
- `--run-eval` now requires a cache to already exist at that path and errors out immediately (before
  training starts, not after) if it's missing or missing a needed dataset/seed entry, instead of
  silently recomputing baseline/Kisialiou per experiment.
- `--recompute-baseline-kisialiou` — explicit escape hatch to compute missing entries on the fly
  and merge them back into the canonical cache, for when cache-only isn't wanted.

Practical effect: baseline/Kisialiou must be benchmarked once via `--baseline-kisialiou-only`
across every size/dataset/seed the RL experiments will use; every subsequent Exp0/1/2/4/5
`--run-eval` invocation reads from that single cache rather than re-running Kisialiou/baseline
per model, so numbers stay directly comparable across experiments and can't silently diverge from
a code change landing in only one of two implementations.

## 12. `op_triple_v2`/`op_pair_v2` (first-fit) + entropy-collapse diagnosis and fix (2026-08-21/24)

Full detail in `docs/experiments/diary_entry_ph300_fix_and_op_triple_v2_firstfit_results.md`
(four dated sections). Summary:

- Added a new cheap, non-cost-ranked repair operator, `first_fit_insertion_v1`
  (`rust_alns/src/operators/repair/first_fit_insertion.rs`), tested in two new action modules:
  `op_triple_v2` (full D×R×I + first-fit) and `op_pair_v2` (flat D×R only + first-fit, no
  improvement phase).
- Found nearly every `ph300`-trained model (progress_horizon-fixed) totally collapses to one fixed
  destroy/repair/improvement triple for every iteration of every restart, **with or without**
  first-fit — so first-fit isn't the root cause. Root cause: PPO's entropy collapses to near-zero
  during training (`entropy_loss` → -0.02 to -0.3) regardless of action-space size or contents,
  well before training ends.
- The flat `op_pair_v2` control (no improvement phase) performs statistically identically to
  uniform-random baseline on both sizes (0/25 wins vs Kisialiou, matching baseline's mean/stdev) —
  confirms the improvement-operator dimension is where 100% of RL's value over random search comes
  from; take it away and RL is just expensive random search.
- **Fix**: `delta_cost_v6` reward (bounded 1x→2x late-restart weighting, using the same
  `progress_horizon`-normalized `progress` value as the state features) + a new `--ent-coef` flag
  (SB3 PPO entropy-bonus coefficient, threaded through `train_ppo_agent`), tested together at
  `ent_coef=0.01` on `op_triple_v2`. Entropy now plateaus (~-0.8 to -1.8) instead of collapsing.
  Win rate vs Kisialiou: small 4%→40%, medium 32%→**56%** (RL now wins outright more often than
  not on medium). Best result of the investigation so far.
- Still open: whether higher `ent_coef` produces genuine multi-switch adaptive behavior (current
  fix still converges to one or two fixed choices per restart, just better ones); no large-size
  run yet for any of these variants.

## 13. Exp 6/7 — weight-collapse mechanism confirmed + RL's large-size policy matches Exp 7's
weakest fixed sequence (2026-09-02)

Full detail: `docs/experiments/diary_entry_fixed_seq_ablation_and_weight_evolution.md`.

**Exp 6 (adaptive weight evolution).** Directly instrumented `destroy_weights`/`repair_weights`
per iteration (30 restarts, small/medium, baseline+kisialiou). Confirms the no-weight-floor EMA
update (`update_operator_weights`, `rust_alns/src/alns/context.rs`) really does let a destroy
operator's weight collapse near zero within a single restart (`shaw_removal`/`worst_visit_removal`
per-restart minimums as low as 0.004-0.03) — the weight-floor hypothesis flagged back in Phase 1
(section 3) was real, not just theoretical. `random_visit_removal` dominates consistently
(mean ~18-20, low variance) across every combo tested. Large not yet run.

**Exp 7 (fixed-sequence improvement ablation).** Isolated the improvement-chain cost problem
(deep_swap eating 81.56% of Kisialiou's large-instance budget, section 7 above) by testing 4 fixed
single/double-operator improvement sequences (VNR-only, FCR-only, relocation-only, VNR+relocation)
against baseline/kisialiou_original, same protocol as Exp0 (15 datasets/size, matched time
budgets). **Result: on large, kisialiou_original is the worst of all 6 modes tested** (mean
ratio-to-best 1.1325, 2/15 wins) — beaten by baseline and 3 of the 4 trivial fixed sequences.
`FCR-only` wins on mean (1.0051) purely via restart volume (37,048 restarts vs. kisialiou's 37 —
the same "quantity beats quality" mechanism as section 7's Finding 2, pushed further by dropping
adaptive-weighting overhead too). `VNR+relocation` wins the most datasets outright (9/15, median
1.0000) with far fewer restarts (369) — clearing structure with VNR before relocating makes each
attempt meaningfully stronger, the first evidence in this project of two operators in sequence
beating either alone. On small/medium, kisialiou_original still leads (small: clear; medium: still
ahead but by a much smaller margin than at small).

**The finding that matters for the paper.** RL's large-size policy (from the widened 15-dataset/
1-seed re-eval of the existing `op_triple_v2`+`delta_cost_v6`+`ent_coef` checkpoint — same
recipe as section 12, run ids `rl_improvement_mvp_20260902_{213050,213730,223958}__
exp_reward_v6_entcoef_op_triple_v2_{small,medium,large}_ph300_15ds1seed`) is, on large,
100% `deep_relocation_v10` with random-removal+first-fit destroy/repair — **exactly** Exp 7's
`fixed_random_ff_relocation_only` configuration. Merging RL's actual per-dataset `best_cost` into
the same 15-dataset ratio-to-best field as Exp 7's table gives a genuine 7-way comparison:

| Mode | Mean | Median | Wins/15 |
|---|---|---|---|
| baseline | 1.1069 | 1.1310 | 0 |
| kisialiou_original | 1.1325 | 1.1237 | 2 |
| fixed VNR-only | 1.0219 | 1.0056 | 0 |
| **fixed FCR-only** | **1.0051** | 1.0062 | 4 |
| fixed relocation-only | 1.1072 | 1.1186 | 0 |
| **fixed VNR+relocation** | 1.0324 | **1.0000** | **9** |
| **RL (learned policy)** | **1.0825** | 1.1170 | 0 |

RL's mean (1.0825) sits almost on top of `fixed_random_ff_relocation_only`'s (1.1072) — RL did not
discover a clever operator combination through training on large instances; it converged on
precisely the configuration Exp 7 tested explicitly and found tied for weakest of six. Two cheap,
already-available alternatives in the same action space (FCR-only, VNR+relocation) score
meaningfully better. On small/medium, by contrast, the widened re-eval shows RL's policy
strengthening past both baselines (small: mean 1.0001, 11/15 wins; medium: mean 1.0018, 10/15
wins) — the large-size gap is a scale-specific problem, not a general failure of the approach.

**Read as headroom, not a ceiling.** Exp 7 shows a policy that used VNR+relocation (or even
FCR-only) on large would land in a materially stronger tier. Leading hypothesis: the reward signal
rewards `deep_relocation`'s larger raw per-call ΔF more than VNR's (which mostly enables later
moves rather than banking cost itself), actively steering the policy toward relocation-only rather
than training simply running out of budget on large's expensive iterations — see section 14 for
the proposed decisive test of this hypothesis.

## 14. Decisive test — time-scaled reward as a falsification test for "RL learns adaptive
sequencing" (proposed 2026-09-04, approved for logging by user same day)

**Context.** Responding to section 13's finding, the user's read: RL's large-size policy is not a
weak version of adaptive search — it's the same greedy single-operator collapse RL has shown at
every scale since Exp 0/1, and it "wins" over kisialiou_original only because kisialiou_original is
itself a poorly-tuned fixed sequence (deep_swap alone eating 81.56% of its budget, section 7/13) —
a comparison that says more about Kisialiou's construction than about RL's search quality. Per the
user directly: "RL is doing just some job, but pretty badly... it performs kinda better than
Kisialiou just because Kisialiou is poorly constructed, few modifications would improve it greatly
and it can be done without RL."

**Why this isn't fully settled yet, and what would settle it.** The reward signal has never been
made aware of wall-clock/solver-time cost per operator call (flagged as a possible confound in
section 13's "read as headroom" note) — `deep_relocation`'s bigger raw ΔF-per-call could be
winning the policy over `voyage_number_reduction`'s smaller-but-structure-enabling ΔF purely
because the reward doesn't discount for how expensive each call is, not because relocation is
actually the better choice once restart economics (the dominant effect established in section 7's
Finding 2 and Exp 7) are accounted for. A **time-scaled reward** (ΔF per unit of solver-time spent
on that call, rather than raw ΔF) directly tests this.

**Why time-scaling is a legitimate correction, not overfitting to the known answer.** The user's
own worry, raised in the same discussion, was that scaling reward by time is just "pushing RL
towards picking FCR/VNR" because those happen to look good in Exp 7 — i.e., hard-coding the
answer under a different name. Counter-argument logged here as the reasoning to revisit if this
gets challenged later: eval is already wall-clock/restart-budget-fair (`match_kisialiou` mode,
section 7), so *ignoring* time in the reward while being scored on a time-fair metric is the actual
mis-specification — the policy is being optimized for a quantity (raw ΔF per call) that the eval
protocol doesn't reward. Time-scaling aligns training-objective and eval-objective; it does not
tell the policy which operator to prefer, only that a call's cost matters, same as it already does
at eval time. This is corroborated independently by section 7's Finding 2 (restart count, not
per-restart quality, is established as the dominant driver of large-scale performance for
kisialiou vs. baseline) — the reward not seeing this at all is a real, structural gap, discovered
before the argument for fixing it.

**The decisive test — pre-committed success criterion, checked in this order:**
1. **Converges toward `voyage_number_reduction` + `deep_relocation` switching mid-restart**
   (VNR to clear structure, relocation to exploit it — matching Exp 7's actual best-median mode,
   `fixed_random_ff_vnr_relocation`, 9/15 wins) → strong positive evidence: this requires the
   policy to alternate between two different operators depending on search state, something RL has
   never done in any run to date (Exp 0 through Exp 7-follow-up all show single-operator collapse
   at large). This is the first result that would count as genuine adaptive sequencing, not just a
   better fixed point.
2. **Converges to `fleet_and_cost_reduction`-only** (matching Exp 7's best-mean mode) → still a
   single-operator collapse, just a relocated, better one (mean 1.0051 vs. relocation-only's
   1.1072) — evidence that the reward mis-specification explains *which* fixed point training finds
   (consistent with section 12's ent_coef finding: "what changed is which fixed point training
   converges to," not whether it's still a fixed point), but not evidence that RL is capable of
   adaptive, multi-operator behavior. Weak/mixed result.
3. **No material change, or converges to something worse** → the reward mis-specification wasn't
   masking real adaptive capability; strong evidence for the user's negative read.

**Two candidate paper framings, contingent on outcome:**
- **If (1):** "RL adds real value once the reward is properly time-aware" — RL's action-space
  design (Paper 1's core scope) is validated once a genuine training/eval-objective mismatch is
  fixed; contrast with Johnn et al. 2024's positive RL results on classic VRP variants, framed as
  "the same operator-selection RL approach transfers to PSVPP, but needed a domain-specific reward
  correction (time-awareness) that a generic ΔF reward misses once operator cost is heterogeneous."
- **If (2) or (3):** an honest negative-result framing — RL is highly sensitive to action-space
  granularity and reward shape, consistently learns a static/greedy per-scale choice rather than
  genuinely adaptive search even after the entropy-collapse fix (section 12) and this reward fix,
  and Exp 7's near-free enumeration of fixed sequences matches or beats both Kisialiou and RL at
  the scale (large) that matters most for the real-world problem — contrasted explicitly with
  Johnn et al. 2024's positive findings on classic VRP variants as a question of *why* PSVPP's
  structure (restart economics dominating at scale, section 7 Finding 2; intra-voyage sequence
  always TSP-reoptimized so RL's real decision space is narrower than raw action-count suggests)
  makes this problem less RL-friendly than the classic VRP settings Johnn et al. studied.

**Status: superseded by section 15 (2026-09-04, same day).** The "implement a time-scaled
reward variant" next step below turned out to already be done — `delta_cost_v9` (found on
disk, undocumented until now) is exactly this, just built and calibrated on 2026-08-25/26
against a different action module (`improvement_chain_v2`, never run at large). Section 15
has the finalized launch plan (dataset scope, exact commands) reusing `delta_cost_v9` on
`op_triple_v2` at large, which is what this section's criteria are actually about.

~~Next concrete step: implement a time-scaled reward variant (ΔF ÷ solver-time-for-that-call,
bounded/normalized the same deliberate way `delta_cost_v6` bounds its progress ramp — see
section 12 — rather than a raw division, to avoid blow-up on very cheap calls) and rerun the
large-size `op_triple_v2` training/eval with it, then check outcome 1 vs. 2 vs. 3 above before
touching anything else.~~

## 15. Decisive-test launch — dataset scope, config additions, exact commands (2026-09-04)

Full detail: `docs/experiments/diary_entry_decisive_time_scaled_reward_test_launch.md`.

**Reward**: reuse `delta_cost_v9` (already exists, see above) — no new reward code needed.
Point it at `op_triple_v2` (not `improvement_chain_v2`, which is what its only prior runs
used) so the result is directly comparable to Exp 7's 7-way table (section 13).

**Dataset scope, decided this session**:
- Test sets grown 15 → **50/size** (`--only-split test`, deterministic seeds keep 1-15
  identical) — biggest lever on statistical power for the paper's headline claim (section 3
  Phase 3).
- Train sets: **superseded same day** — replaced the initial 2-config-per-size add
  (`TRAIN_4`/`_5`) with a systematic 10-config-per-size scheme after the user asked for
  "much more configs for variety": a **TW axis** (`{SIZE}_TRAIN_TW1..TW5`, time-window
  density 0%→100% of installations) and a **DF axis** (`{SIZE}_TRAIN_DF1..DF5`,
  demand+frequency spectrum from small-demand+frequent to large-demand+infrequent, 3 mixes
  between), each axis's 5 configs spread across the size tier's requested install-count
  range (small 6-12, medium 12-20, large 20-40). 13 train configs/size total (was 3). Full
  parameter tables in the diary entry linked below. This is the "retrain on more variable
  training data" idea from earlier this session, previously flagged as not yet run.

**Execution note**: the Cowork sandbox this was authored in has no `conda`/Rust
toolchain/built extension — code/config changes were made directly against the real repo
files, but the actual generation/training/eval commands (in the diary entry above) need to
run in a real `py312` environment (Claude Code session or terminal), detached via
`nohup`/`tmux` per the "Long-running jobs" note at the top of this file — a 50-dataset large
run will take hours.

**Status**: config/code changes complete and verified 2026-09-04. Commands not yet launched.

## 16. Planned — test-side TW-density sweep, eval-only, no retraining (planned 2026-09-11, target: next week)

**Motivation**: while grouping Experiment 8's results (`docs/EXPERIMENTS.md`) by dataset
characteristics, found the current test sets (`{SIZE}_TEST_1`) have **zero time-window
variation to group by** — every one of the 50 generated instances per size has *exactly* the
same tight-window fraction (measured: 0.300 for every single large instance, no exceptions).
This isn't sampling noise: `time_windows: N` in `generation_config.yaml` is a literal count of
how many installations (out of that block's `num`) get assigned a tight window, not a
probability — so every draw from one scenario config produces the same tight/loose
composition by construction. Demand and visit-frequency, by contrast, are drawn from
distributions and do vary instance-to-instance (see Experiment 8's demand/frequency grouped
breakdown).

The train side already has exactly the density axis we're missing on test:
`{SIZE}_TRAIN_TW1..TW5` (section 15's dataset-scope note; configs at
`generation_config.yaml` lines ~835-965 for large, similarly for medium/small) sweep tight-window
density from 0% (`TW1`) through ~50% (`TW3`) to 100% (`TW5`) by setting `time_windows: N` as a
fraction of each block's `num`. No test-side equivalent exists yet.

**Plan**: mirror that mechanism on the test side — new `{SIZE}_TEST_TW0`, `_TW50`, `_TW75`,
`_TW100` scenario configs (density points as requested; could instead reuse train's exact
5-point 0/~25/50/~75/100 scheme for consistency — open decision, pick when implementing),
generate datasets from them (`scripts/generate_datasets.py --only-split test`), and then run
**eval-only** (`--skip-training --model-path ...`, no retraining) using the **already-trained
Experiment 8 models** (`op_triple_v2` / `features_improvement_v3` / `delta_cost_v9` / ph300 — one
model per size, same ones used for Exp 8) against these new TW-density-controlled test sets.
Goal: see whether RL's operator collapse pattern and its win rate vs. Kisialiou/baseline shift as
TW tightness moves from 0% to 100%, holding demand/frequency distribution fixed. Cheap relative to
retraining — this only needs new datasets + eval runs, not new PPO training.

**Status**: planned only, not implemented. Target: next week.
