# RL-ALNS Experiments

## Overview

Each experiment tests a different RL action-space design on top of the ALNS solver.
All experiments are evaluated on the **small** dataset (5 instances, 5 seeds = 25 samples)
using **time-fair** mode: RL/baseline receive the same wall-clock budget as Kisialiou's
reference restarts.

### Common setup
- Training: 200 000 PPO timesteps, 50 iterations/episode, seed 42, small dataset
- Eval: `match_kisialiou` runtime mode, 5 eval seeds (42–46), 5 datasets
- **Baseline**: adaptive weighted ALNS (`adaptive_weighted_v1`) — engine selects destroy/repair
  via roulette-wheel with weights updated every 10 iterations (σ1=33, σ2=9, σ3=3, γ=0.2).
  No improvement operators.
- **Kisialiou**: full Kisialiou ALNS — adaptive backbone + fixed improvement sequence
  (voyage_number_reduction → fleet_and_cost_reduction → deep_relocation →
  fleet_and_cost_reduction → deep_swap → fleet_and_cost_reduction) after every step.

### Operator set
- **Destroy** (30): `shaw_removal_v1..v10`, `random_visit_removal_v1..v10`, `worst_visit_removal_v1..v10`
- **Repair** (7): `deep_greedy_rank_1..5`, `k_regret_2`, `k_regret_3`
- **Improvement** (35): `voyage_number_reduction_v1..v5` (search_k 1–5),
  `fleet_and_cost_reduction_v1..v10` (lambda_max 8–80),
  `deep_relocation_v1..v10` (max_moves 1–9, then unlimited),
  `deep_swap_v1..v10` (max_moves 1–9, then unlimited)

---

## Experiment 0 — Pair selection only

**Hypothesis**: Can RL learn to pick better destroy/repair pairs than the adaptive ALNS baseline?

| Field | Value |
|---|---|
| Action module | `op_pair_v1` |
| Action space | 9 = 3 destroy × 3 repair (base v1 operators, no improvement) |
| Compare against | Baseline (adaptive ALNS, no improvement) |
| Folder tag | `exp0_id0_op_pair_v1` |
| Training timesteps | 200 000 |

**Action details**: RL picks one of 9 D×R pairs each step. Engine does not apply
any improvement operator. Baseline also runs without improvement (adaptive_weighted_v1).

**Expected finding**: RL should learn to prefer sophisticated operators (shaw + k_regret)
over simpler ones (random + deep_greedy). Win rate vs baseline indicates whether RL
pair selection adds value beyond adaptive weight selection.

---

## Experiment 1 — Full triple selection (D×R×I)

**Hypothesis**: Adding an improvement operator to every destroy/repair step improves
solution quality. Can RL learn the right combination?

| Field | Value |
|---|---|
| Action module | `op_triple_v1` |
| Action space | 36 = 3 destroy × 3 repair × 4 improvement |
| Compare against | Baseline (adaptive ALNS, no improvement), Kisialiou |
| Folder tag | `exp1_id1_op_triple_v1` |
| Training timesteps | 200 000 |

**Action details**: Each RL step selects destroy + repair + improvement simultaneously.
Base operators used: `shaw_removal_v1`, `random_visit_removal_v1`, `worst_visit_removal_v1`,
`deep_greedy_rank_1`, `k_regret_2`, `k_regret_3`, `voyage_number_reduction_v1`,
`fleet_and_cost_reduction_v9` (λ_max=72), `deep_relocation_v10` (unlimited),
`deep_swap_v10` (unlimited).

**Expected finding**: Solution quality substantially better than exp 0 baseline;
gap vs Kisialiou measures how much value the RL triple selection adds vs
Kisialiou's fixed improvement sequence.

---

## Experiment 2 — Kisialiou backbone + RL improvement chain

**Hypothesis**: Keeping the adaptive ALNS backbone and letting RL control only
the improvement phase (in chains of up to 3) can approach or exceed Kisialiou quality.

| Field | Value |
|---|---|
| Action module | `improvement_chain_v1` |
| Action space | 5 = 1 backbone + 4 improvement operators |
| Compare against | Baseline (adaptive ALNS, no improvement), Kisialiou |
| Chain max length | 3 consecutive improvement steps before forced backbone |
| Folder tag | `exp2_id3_improvement_chain_v1` |
| Training timesteps | 200 000 |

**Action details**:
- Action 0: run backbone step (adaptive weighted destroy+repair, no improvement)
- Actions 1–4: apply one improvement operator (`voyage_number_reduction_v1`,
  `fleet_and_cost_reduction_v9`, `deep_relocation_v10`, `deep_swap_v10`)

RL can chain up to 3 improvements before backbone is forced. The engine's adaptive
weights evolve across backbone steps.

**Expected finding**: RL learns to sequence improvement operators effectively,
potentially discovering non-trivial chains (e.g. relocation then voyage reduction).
Tighter gap to Kisialiou than experiments without improvement chain control.

---

## Experiment 3 — Fixed pair benchmark (no RL training)

**Hypothesis**: More sophisticated destroy/repair operators (shaw removal + k-regret)
outperform simpler ones (random removal + greedy insertion) regardless of learning.

| Field | Value |
|---|---|
| Type | Benchmark (no RL training) |
| Pairs tested | All 9 = 3 destroy × 3 repair fixed combinations |
| Compare against | Baseline (adaptive weighted ALNS) |
| Script | `scripts/run_experiment_pair_benchmark.py` (TODO) |
| Folder tag | `exp3_pair_benchmark` |

**Method**: Each of the 9 D×R pairs is run as a fixed non-adaptive policy
(always same pair, no weight updates used for selection). Results ranked by
solution quality. Confirms which pair performs best individually and shows
whether adaptive ALNS outperforms any fixed pair.

**Original expectation (superseded 2026-08-07 — see below)**: shaw_removal + k_regret achieves
highest quality among fixed pairs. Adaptive ALNS baseline outperforms all fixed pairs due to
dynamic operator selection learning over the run.

**Actual result (2026-08-07, first fully-calibrated run, small size only —
`benchmark_modes_matrix_20260807_180302`, full details in
`docs/experiments/diary_entry_calibrated_baseline_and_exp3.md`)**: the opposite pattern on both
counts. `worst_removal` + `k_regret_3` ranks *best* of all 9 pairs (+0.05% vs adaptive baseline —
essentially a tie, nominally beating it), and `shaw_removal` pairs rank *worst* of the 9. Every
fixed pair lands within ±0.33% of the full adaptive baseline on small instances — switching
operators buys essentially nothing here, which does still support the experiment's original
question ("is switching necessary, or does picking the best pair suffice?") even though it
guessed wrong about which pair. Destroy-operator choice drives ~9x more variation than
repair-operator choice. Not yet run on medium/large — this conclusion is small-instance-only for
now; operator effectiveness may not be scale-invariant (see Finding 1 in the diary entry, which
found a strong, size-specific destroy-operator preference on large instances in the adaptive
baseline).

---

## Experiment 6 — Adaptive weight evolution (done for small/medium, 2026-09-02)

**Motivation**: baseline and Kisialiou both use `adaptive_weighted_v1`, whose destroy/repair
weights update every `weight_update_interval` iterations via
`w_new = (1 - reaction_factor) * w + reaction_factor * (score / count)` (`rust_alns/src/alns/
context.rs::update_operator_weights`) — an EMA with **no weight floor**, already flagged as a
candidate explanation if Kisialiou collapses to Shaw-removal-only. We didn't know what the
weights actually evolve to by the end of a restart, or how they trend over the course of search.

| Field | Value |
|---|---|
| Type | Instrumentation + benchmark (no RL training) |
| Modes | baseline, kisialiou (both use adaptive_weighted_v1 / mode="weighted") |
| Sizes | small + medium done; large deferred (no CPU contention concern left, just not yet run) |
| Script | `scripts/capture_adaptive_weights.py` (new), 30 restarts x 200/300 fixed iterations |

**Implementation**: added `info["destroy_weights"]`/`info["repair_weights"]` to
`ALNSEnvironment.step()` (`rl_alns_environment.py`) — `execute_iteration` already computed and
returned both every iteration on the Rust side, just never surfaced to Python before this.

**Result**: only 3 of ~36 registered destroy operators and 3 of 8 repair operators are ever
nonzero (the rest permanently locked at weight 0 — Kisialiou et al. 2018 Table 1's 3x3
calibration). Final (end-of-restart) weight mean/std across 30 restarts, all 4 (size, mode)
combos: `random_visit_removal` is consistently strong and low-variance (mean ~18-20); `shaw_removal`
and `worst_visit_removal` show high variance, with per-restart minimums as low as 0.004-0.03 —
**confirms the no-weight-floor mechanism does let an operator collapse near zero within a single
restart**, not just a theoretical risk. Repair operators (`deep_greedy_rank_1`, `k_regret_2`,
`k_regret_3`) stay within a similar range of each other in every combo — no clear repair-operator
dominance the way there is for destroy. Full details and per-iteration time series (ready to
plot): `docs/experiments/diary_entry_fixed_seq_ablation_and_weight_evolution.md`, data in
`output/exp6_adaptive_weights/`.

**Status**: small/medium done. Large not yet run — no longer blocked (Experiment 7 below has
completed), just not prioritized yet.

---

## Experiment 7 — Fixed-sequence improvement ablation (done, 2026-09-02)

**Question**: Experiment 0 (Section 4/5 of the calibration report) found `deep_swap_v10` eats
81.56% of Kisialiou-original's large-instance budget while being its least time-efficient
improvement operator. Does removing it (and the rest of the fixed 6-step sequence) free up
enough restarts to change the large-scale result?

| Field | Value |
|---|---|
| Type | Benchmark (no RL training) |
| Modes | 4 new: fixed destroy/repair (random-removal + first-fit-insertion, `fixed_pair_idx=7` under `op_pair_v2`) x improvement-sequence variants: VNR-only, FCR-only, deep_relocation-only, VNR+deep_relocation |
| Sizes/budgets | Same as Exp0: 15 datasets/size, 60s/180s/1800s (small/medium/large) |
| Script | `scripts/run_experiment_kisialiou_modes.py` (4 new `ModeSpec`s), `scripts/alns_mode_benchmark.py` (new `--fixed-pair-action-module` flag) |
| Run | `output/benchmark_modes_matrix_20260901_113248__fixed_seq_ablation` |

**Result**: on large, **Kisialiou-original is the worst-mean mode of all 6 tested** (baseline,
kisialiou_original, and the 4 new fixed-sequence modes), beaten by baseline and 3 of 4 fixed
variants. FCR-only wins on mean ratio-to-best (1.0051 vs. kisialiou_original's 1.1325) purely via
restart volume (37,048 restarts vs. Kisialiou's 37 — same "quantity beats quality" mechanism as
baseline, pushed further). VNR+relocation wins the most datasets outright (9/15) with only 369
restarts. On small/medium, kisialiou_original still wins clearly — restart cost isn't the
bottleneck at those scales. Full six-way tables and the falsified FCR-weakness prediction (an
earlier same-session guess, corrected rather than dropped):
`docs/experiments/diary_entry_fixed_seq_ablation_and_weight_evolution.md`.

**Follow-up (2026-09-04)**: re-evaluated the trained RL policy on all 15 datasets (1 seed, no
retraining) and merged it into the same large-scale ratio-to-best field as this experiment's 6
modes. RL's large-size policy (random-removal + first-fit + 100% `deep_relocation_v10`) is
operator-for-operator identical to `fixed_random_ff_relocation_only` above, and scores almost
exactly the same (mean 1.0825 vs. 1.1072) — RL rediscovered this experiment's *weakest* fixed
sequence, not FCR-only or VNR+relocation. Read as headroom for large-size RL training, not a
ceiling. Full 7-way table and discussion in the same diary entry.

**Status**: done, all 3 sizes.

---

## Experiment 8 — Full-scale RL (op_triple_v2, reward_v9, ph300) vs. baseline vs. Kisialiou, 50 datasets/size (done, 2026-09-08)

**Hypothesis**: with the `delta_cost_v9` (time-penalty) reward and a 300-iteration progress
horizon (ph300), does RL still beat Kisialiou at medium/large scale on a much larger test set
(50 datasets/size, vs. Exp7's 15), and does Exp7's large-scale operator collapse persist under
this training config?

| Field | Value |
|---|---|
| Action module | `op_triple_v2` |
| Action space | 48 = 3 destroy × 4 repair × 4 improvement |
| State module | `features_improvement_v3` |
| Reward module | `delta_cost_v9` |
| Progress horizon | 300 (ph300) |
| Compare against | Baseline (`adaptive_weighted_v1`), Kisialiou (`kisialiou_original`, cached) |
| Sizes | small, medium, large — 50 datasets each, 1 eval seed (42) |
| Eval mode | `match_kisialiou` — RL/baseline restart until their own wall time matches Kisialiou's 5-restart reference |
| Run tags | `exp_reward_v9_timepenalty_op_triple_v2_{small,medium,large}_ph300_50ds1seed[_evalonly]` |
| Output dirs | `output/rl_improvement_mvp_20260906_063629__..._small_ph300_50ds1seed`, `..._070345__..._medium_ph300_50ds1seed`, `..._125355__..._large_ph300_50ds1seed_evalonly` |

**Infra note**: the large-size eval crashed twice (2026-09-06/07) before completing — not a
result-affecting issue, but worth recording since it explains the `_evalonly` output dir. (1) OOM:
the eval loop deep-cloned a full engine snapshot per restart so baseline could start from RL's
matched initial solution, but built the whole snapshot list even on cache-hit datasets where
baseline was never going to run and consume it. Fixed by skipping snapshot collection whenever
baseline is cached, and interleaving RL/baseline restarts (instead of running RL's full sweep
first) when a snapshot is genuinely needed, so at most one is ever live in memory — see
`rl/episode_runner.py`'s `run_policy_and_baseline_episodes`. (2) Rust panic: the constructive
initial-solution heuristic (`rust_alns/src/operators/initial_solution.rs`) gave up after a single
bad vessel-assignment shuffle instead of retrying its other 9 attempts (`break 'outer` where the
sibling failure path correctly used `continue 'outer` — a typo). Measured failure rate was ~1/5000
restarts on the crashing instance, not a real capacity issue (fleet averages ~78% slack) — fixed
the typo and rebuilt. Both fixes verified numerically equivalent to the original code path on a
controlled before/after comparison; no effect on the reported results below.

**Result — ratio-to-best (1.0000 = best of the 3 modes on that instance; n=50 datasets/size):**

*Small*

| mode | mean | median | max | # instance-best |
|---|---|---|---|---|
| RL | 1.0003 | 1.0000 | 1.0021 | 30 |
| Baseline | 1.0513 | 1.0162 | 1.8978 | 0 |
| Kisialiou | 1.0364 | 1.0000 | 1.8878 | 20 |

*Medium*

| mode | mean | median | max | # instance-best |
|---|---|---|---|---|
| RL | 1.0040 | 1.0045 | 1.0110 | 20 |
| Baseline | 1.1019 | 1.0175 | 1.3011 | 0 |
| Kisialiou | 1.1025 | 1.0000 | 1.2927 | 30 |

*Large*

| mode | mean | median | max | # instance-best |
|---|---|---|---|---|
| RL | 1.0029 | 1.0000 | 1.0110 | 35 |
| Baseline | 1.0961 | 1.1285 | 1.1647 | 0 |
| Kisialiou | 1.0799 | 1.1091 | 1.1379 | 15 |

RL never loses to baseline at any size (0 instance-bests for baseline, mean ratio always ≥1.05).
RL-vs-Kisialiou instance-win rate is 30/50 → 20/50 → 35/50 (small→medium→large) — RL's edge over
Kisialiou does not shrink with instance size; it's strongest on large. RL's worst case stays tight
everywhere (max ratio 1.002–1.011); baseline and Kisialiou both have long tails (up to 1.90x on
small).

**Operator collapse** (checked per the Phase 1 gate before Phase 2, per CLAUDE.md): at all 3
sizes the trained policy collapses onto a single fixed destroy/repair/improvement triple — 100%
of decisions, every restart, every dataset (confirmed by sampling multiple datasets per size,
including the first, last, and one mid-run). Destroy and repair are identical across sizes
(`random_visit_removal_kisialiou_{size}_v1` + `first_fit_insertion_v1`); the improvement operator
splits cleanly along size:

| size | improvement operator chosen |
|---|---|
| small | `deep_relocation_v10` |
| medium | `fleet_and_cost_reduction_kisialiou_v1` |
| large | `fleet_and_cost_reduction_kisialiou_v1` |

Read as a domain-sensible split, not noise: FCR (fleet-and-cost-reduction) consolidates vessels
and voyages, which only pays off when there's fleet slack to consolidate from — small instances
have little of that, so per-visit relocation is where the savings are; medium/large have enough
fleet redundancy for FCR to find room to work. The same split appears independently in 3 separately
trained models (small/medium/large), which is a real signal, not chance agreement.

Consistent with Exp7's "quantity beats quality" mechanism: RL relies on far more restarts than
Kisialiou, not adaptive per-state operator switching. RL restart counts on dataset 1 of each size:
small 17 vs. Kisialiou's 5 (3.4x); medium 958 vs. 5 (192x); large 7,110 vs. 5 (1,422x).

**Grouped by instance characteristics**: tried to also group by time-window tightness, but found
the current test sets have **zero TW variation to group by** — every one of the 50 instances per
size has *exactly* the same tight-window fraction (measured 0.300 for all 50 large instances, no
exceptions; `time_windows: N` in `generation_config.yaml` is a fixed count, not a probability, so
every draw from one scenario config has identical TW composition by construction). Follow-up
planned to fix this gap on the test side — see plan section 16 (test-side TW-density sweep,
eval-only, next week).

Demand and visit-frequency *do* vary instance-to-instance (drawn from distributions), and are
essentially uncorrelated with each other (Pearson r = 0.01–0.12 across all 3 sizes) so they're
independent axes. Median-split ratio-to-best (mean ratio, #instance-best in parens):

| size | mean_demand: high half | mean_demand: low half | mean_freq: high half | mean_freq: low half |
|---|---|---|---|---|
| small | RL 1.0003(14) Base 1.0504(0) Kis **1.0340(13)** | RL 1.0002(16) Base 1.0524(0) Kis **1.0392(7)** | RL 1.0001(19) Base 1.0488(0) Kis **1.0321(10)** | RL 1.0004(11) Base 1.0548(0) Kis **1.0424(10)** |
| medium | RL 1.0047(8) Base 1.1128(0) Kis **1.0782(17)** | RL 1.0033(12) Base 1.0910(0) Kis **1.1269(13)** | RL 1.0048(8) Base 1.0832(0) Kis **1.0671(19)** | RL 1.0031(12) Base 1.1239(0) Kis **1.1441(11)** |
| large | RL 1.0028(18) Base 1.0909(0) Kis **1.0772(7)** | RL 1.0030(17) Base 1.1014(0) Kis **1.0826(8)** | RL 1.0025(19) Base 1.0846(0) Kis **1.0796(7)** | RL 1.0034(16) Base 1.1086(0) Kis **1.0803(8)** |

Because demand and frequency are uncorrelated, also checked the joint condition directly (low
frequency **and** high demand together — the "harder to route" case: fewer, bigger visits — vs.
high frequency **and** low demand together — "easier": many, smaller visits):

| size | harder (low-freq & high-demand) | easier (high-freq & low-demand) |
|---|---|---|
| small (n=9 / n=11) | RL 1.0006 Base 1.0136 Kis 1.0000 | RL 1.0001 Base 1.0160 Kis 1.0010 |
| medium (n=13 / n=15) | RL 1.0047 Base 1.1021 Kis **1.0858** | RL 1.0048 Base 1.0502 Kis **1.0648** |
| large (n=13 / n=14) | RL 1.0039 Base 1.1054 Kis 1.0688 | RL 1.0032 Base 1.0926 Kis 1.0739 |

**Reading it**: RL's mean ratio is flat everywhere (1.0001–1.0048 across every slice, every size)
— its quality doesn't depend on instance load. On **medium**, the harder/easier split shows
exactly the pattern you'd expect: baseline and Kisialiou both degrade on the harder case (Kisialiou
1.0648→1.0858, baseline 1.0502→1.1021) while RL barely moves (1.0048→1.0047) — RL holds steady
while the other two get worse as the routing problem gets harder. Small and large show the same
direction much more weakly or not at all (e.g. large's Kisialiou is *marginally better* on the
"harder" case, 1.0688 vs. 1.0739) — so treat the harder/easier degradation as a clear, real effect
at medium scale specifically, not yet a confirmed all-sizes trend.

**Status**: done, all 3 sizes, 50 datasets each, numbers locked for the paper. No further
experiments planned on this run/config.

---

## Experiment 9 — State / action / reward design ablation for NORM 2026 (running, launched 2026-10-09)

**Question**: for the talk "State, Action, and Reward Design for RL-Enhanced ALNS in Periodic
Supply Vessel Planning" — which design choices actually matter? One factor at a time around the
Experiment 8 reference (`op_triple_v2` / `features_improvement_v3` / `delta_cost_v9`,
ent_coef 0.01, ph300, 200k PPO steps), plus non-learned controls on the same action spaces.
Small + medium only (large skipped for time).

| Axis | Variants (reference in bold) |
|---|---|
| Action | `improvement_chain_v2` (adaptive D/R, RL picks the improvement; 5) · **`op_triple_v2` (48)** · `op_triple_many_v1` (288 = 6 destroy × 4 repair × 12 improvement) |
| State | `features_instance_v1` (11, static) · `features_search_v1` (34) · `features_solution_v1` (15) · `features_all_v1` (60, all three) · `features_null_v1` (constant) · **`features_improvement_v3`** |
| Reward | `alns_score_v1` (Ropke & Pisinger σ 33/9/3 → 1 / 0.27 / 0.09 / 0) · `alns_score_time_v1` (+ 0.002/ms time tax, cap 0.5) · `new_best_v1` (sparse +1) · `best_gain_v1` (% best-cost gain) · **`delta_cost_v9`** |
| Controls (no training) | uniform random over triple / many; fixed triples random-removal + first-fit + FCR (action 29) and + relocation (action 30) — the triples Exp 8's medium/small policies collapsed to |

**Many-operators action space** (`op_triple_many_v1`, recommendation from the previous
conference): destroy = Shaw / random / worst × removal ratio {Kisialiou-calibrated, high
ξ∈[0.30,0.40]} (only ξ differs; other parameters at the Kisialiou calibration); repair as in the
triple; improvement = 3 intensity levels each: VNR search_k 1/3/5, FCR λmax 8/35/80, relocation
and swap max_moves 1/10/unlimited. Five new Rust operators (`*_high_v1`, `deep_relocation_m10_v1`,
`deep_swap_m10_v1`) were appended after all existing ones; a fixed-seed trajectory check confirmed
byte-identical behaviour of existing operators (triple, chain, adaptive baseline; small + medium),
so trained models and the baseline/Kisialiou cache stay valid.

**State groups**: *instance* — installations, visits, weekly deck demand, demand in full loads of
the smallest / largest vessel (the vessel list is only an upper bound on the fleet, so fleet size
itself is excluded), average / max distance, visits per installation, demand per visit, share of
installations and of visits with a time window (< 24 h). *Search process* — progress, stagnation,
temperature ratio, last step accepted / acceptance type / new best / reward, recent acceptance and
new-best rates, **last used destroy / repair / improvement family** (+ removal-ratio and intensity
level), repeat streak, last step duration, per-family improvement track record. *Solution* —
current and best cost vs initial, current-best gap, fixed-cost share, voyages per visit, empty
voyages, vessels used / fleet, voyage and vessel load / time utilisation, feasibility. All
normalised (the v3 reference feeds raw costs).

**Protocol**: 50 test instances per size, eval seed 42, `match_kisialiou` wall-time budget against
the cached Kisialiou/baseline results (per-size slices `runs/baseline_kisialiou_cache/
cache_{size}_seed42.json`). Training seeds: small 42/43/44 for every trained variant; medium 42
(reference 42/43/44). The Exp 8 seed-42 models are re-evaluated, not reused, so every variant is
scored under the same machine load (4 concurrent jobs on the M1's 4 performance cores).
Evaluation now checkpoints after every instance (`partial_rows.jsonl`, `--resume`) and records
per-restart action statistics (distinct actions, switches, action histogram).

**Scripts**: `scripts/run_norm2026_ablation.py` (queue, 58 jobs, status in `NORM2026_STATUS.md`),
`scripts/analyze_norm2026_ablation.py` (ratio-to-best tables, paired Wilcoxon + bootstrap CIs vs
reference and Kisialiou, policy-behaviour tables). Outputs: `output/norm26__<axis>__<variant>__
<size>__s<seed>/`, analysis in `output/norm26_analysis/`.

**Data provenance note (found 2026-10-08)**: `data/processed/alns/*` (what the solver, training and
the cache read) is created once from `data/generated/*` and never refreshed. Test datasets 01–05
of every size (and large 08), plus 15 medium/large training sets, differ between the two because
`data/generated` was regenerated on 2026-09-04. All experiments are internally consistent (they
all read the processed files), but any analysis reading instance properties from
`data/generated` describes the wrong instances for those datasets — read `data/processed` instead.

**Status**: running.

---

## Historical experiments (pre-refactor)

Earlier experiments ran with single non-parameterized operators. Bugs fixed
before the current series:
- Kisialiou improvement sequence used exact bare-name matching → found nothing
  after operators were renamed to `_v{i}` format (fixed: prefix matching)
- `VoyageNumberReduction.search_k` was a no-op (called `top_k_visit_insertion_costs`
  per-voyage in an outer loop; fixed: single call ranks all voyages)
- `KRegretInsertion.search_k` was a no-op for regret computation (fixed: removed
  redundant variants, kept single k=2 and k=3)
- Baseline used random explicit pair selection (fixed: `adaptive_weighted_v1` uses adaptive
  weighted engine selection)
- Improvement operators missing from logs for D×R×I triple actions (fixed:
  `log_improvement` now true whenever `improvement_idx_result is not None`)
