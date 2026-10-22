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

## Timing protocol and RL decision overhead (analysis for the paper, 2026-10-09)

**Why this matters for the paper.** Exp 8 matched methods on wall-clock time; Exp 9 matches them
on *search CPU time*. The obvious reviewer question is whether search time is chosen to hide
RL's overhead. Answer, with the numbers below: the overhead is an implementation artefact of the
Python prototype; in an optimised implementation RL's per-decision cost is a few microseconds,
about 1–3% of its search time per decision (median; at most ~5%).

**1. How Exp 8 measured time.** Budget per instance = Kisialiou's *wall-clock* time for its 5
reference restarts × 300 iterations (`time.perf_counter()` around each restart: environment
creation → all iterations → close), taken from the cached reference run. RL and the baseline ran
300-iteration restarts until their own cumulative wall-clock time reached that budget (checked
between restarts, so the last restart always completed). The engine's own per-iteration timer
("solver time", wall clock, ns) was reported but not used for budgets; `delta_cost_v9`'s time
penalty used it per step.

Engine (search) share of wall-clock time in Exp 8, summed over the 50 instances per size
(from each run's `results.csv`):

| size | Kisialiou | RL | Baseline |
|---|---|---|---|
| small | 93% (8.6 of 9.2 min) | 65% (6.2 of 9.5 min) | 54% (5.0 of 9.3 min) |
| medium | 99% (105.7 of 106.5 min) | **21%** (22.6 of 106.6 min) | 83% (88.9 of 106.7 min) |
| large | 100% (1681 of 1684 min) | 40% (680 of 1684 min) | 98% (1653 of 1687 min) |

So at equal wall-clock time RL searched only ~1/5 (medium) to ~2/3 (small) as long as Kisialiou
and still beat it on average — Exp 8's protocol was conservative *against* RL.

**2. Why RL's overhead is the largest.** Per step, medium instance, one job alone, Exp 8 policy:

| method | search | Python in the env step (state, reward, log row, result conversion) | policy inference (SB3 `predict`) | search share |
|---|---|---|---|---|
| RL | 0.17 ms | 0.31 ms | 0.22 ms | 24% |
| Baseline | 1.88 ms | 0.24 ms | — | 89% |
| Kisialiou | 196 ms | 1.24 ms | — | 99% |

Python costs per step are roughly the same for every method; what differs is how much search one
step contains. RL's learned medium policy (random removal + first-fit + FCR) is very cheap per
step — the time-aware reward and restart economics (Exp 7) favour many fast steps — so fixed
per-step costs dominate exactly for RL.

**3. RL's essential decision cost in an optimised implementation.** At evaluation time only two
things are needed per decision: building the state vector and the policy forward pass (reward
computation, operator logging and per-restart environment creation are not).
- Policy: actor MLP 45 → 64 → 64 → 48 with tanh (~10k multiply-adds). Per decision: SB3
  `predict` 199 µs; lean PyTorch forward 44 µs; NumPy 12 µs; compiled Rust (arm64, plain scalar
  loops, M1) **4.3 µs** — all measured.
- State features: mostly values the engine already computes; native cost estimated at ~2 µs.
  Conservative upper bound: the engine's current solution-metrics routine called from Python,
  17 / 25 / 38 µs (small / medium / large) — it clones the solution and re-runs a full
  feasibility check on every call, which a lean state does not need.
- Realistic total ≈ **6.3 µs per decision**; conservative bound ≈ 21 / 30 / 43 µs.

**4. Overhead relative to RL's search per decision** (per instance, Exp 8 policies, 150
instances; RL's search per decision: median 1,120 µs small / 197 µs medium / 431 µs large):

| size | realistic (~6.3 µs/decision) | conservative bound |
|---|---|---|
| small | **0.6%** median, 1.9% worst instance | 1.9% median, 6.5% worst |
| medium | **3.2%** median, 4.9% worst | 15% median, 23% worst |
| large | **1.5%** median, 1.8% worst | 9.9% median, 12% worst |

For comparison, the current Python implementation spends ~76% of RL's time on medium in overhead.

**Caveats to state.** (a) The ratio depends on the operators the policy chooses: cheaper
operators mean relatively larger overhead (medium is the worst case here). (b) It depends on the
network: the 288-action head of `op_triple_many_v1` adds ~50% to the forward pass (still a few
µs); graph or attention networks over the solution would cost tens to hundreds of µs per decision.
(c) The comparison methods' own selection logic (roulette wheel, weight updates) runs inside the
engine and is counted in their search time, whereas RL's decision is not unless it is charged
explicitly (adding a fixed per-decision charge of ~6 µs to RL's search time is a one-line change).
(d) Decision frequency vs. operator cost is itself an action-space design consideration: with
cheap operators, deciding at every step is relatively expensive.

**5. Measurement facts behind the search-CPU protocol (Exp 9).**
- Search CPU time is measured inside the Rust engine (`search_cpu_ns`, process CPU clock, ns
  resolution) over exactly the spans the engine already timed: each iteration's destroy, repair,
  improvement, acceptance and weight update, plus each restart's constructive initial solution
  (0.13 ms small / 0.31 ms medium). It agrees with the engine's wall-clock timer within 1–3% on an
  idle machine, and adding it left fixed-seed search trajectories byte-identical.
- Under Rosetta 2 (x86_64 Python on Apple Silicon) the per-thread CPU clock returns 24 MHz ticks
  instead of nanoseconds (~41.7× too small); the process clock is correct and equals the thread's
  CPU time for the single-threaded engine.
- On the M1 Air, the CPU time a fixed amount of work costs depends on machine load: identical,
  deterministic Kisialiou work took 2.37 s alone, 3.0–3.3 s with 3 concurrent jobs and 3.8–4.0 s
  with 4 (with 7 background CPU hogs: +40–48%; marking the process "user-interactive" did not
  help). Fair budgets therefore need the reference and all evaluations to run under the same load.

**Suggested wording for the paper.** "Methods are compared at equal search time, i.e. CPU time
spent inside the search engine (operator applications, acceptance and adaptive bookkeeping),
excluding the interpreter overhead of our Python prototype. In an optimised implementation the
RL agent's per-decision cost — constructing the state and one forward pass of a 45-64-64-48
network — is about 6 µs, i.e. 0.6–3.2% of its search time per decision (median over test
instances; at most 4.9%), measured with a compiled implementation of the policy network."

*Sources: Exp 8 output folders (`results.csv`, `restart_results.csv`); measurement scripts and
what each one produced: `scripts/timing/` (see its README).*

---

## Experiment 9 — State / action / reward design ablation for NORM 2026 (prepared 2026-10-09)

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

**Protocol** (revised 2026-10-09 — first launch stopped after 1 job, see below): first 30 small /
20 medium test instances, eval seed 42. Training seeds: small 42/43 for every trained variant;
medium 42; the reference (= Exp 8 configuration) is *retrained* at medium 42/43/44 and small
42/43, because Exp 8's models were trained with the old wall-clock time penalty.

*Time budget = search CPU time.* RL and the adaptive baseline get exactly the search CPU time
Kisialiou needs for its 5 reference restarts on each instance. Search CPU time is measured inside
the Rust engine (`search_cpu_ns`, process CPU clock, ns) over exactly the spans the engine already
timed — every iteration's destroy / repair / improvement / acceptance / weight update, plus the
constructive initial solution of each restart — so Python overhead (policy inference, state
encoding, operator logging, object creation, GC) no longer counts against any method. Under the
old wall-clock rule it did: Exp 8's medium RL spent only ~21% of its wall time in the engine.
The time-aware rewards (`delta_cost_v9`, `alns_score_time_v1`) and the search encoder's
step-duration feature use the same CPU measurement. The Kisialiou/baseline reference is
regenerated under this rule inside the run (`runs/baseline_kisialiou_cache/search_cpu_v1/`);
Kisialiou is deterministic and reproduces the old cache's best costs exactly.

*Constant machine load.* Two measurement findings on this M1 Air (x86_64 Python under Rosetta):
(1) the per-thread CPU clock is wrong under Rosetta (returns 24 MHz ticks, ~41.7x too small),
hence the process clock — the engine is single-threaded; (2) the CPU time a fixed amount of
search costs depends on machine load — identical deterministic Kisialiou work took 2.37 s alone,
~3.1 s with 3 and ~3.9 s with 4 concurrent jobs (QoS "user-interactive" did not help). So every
time-budgeted run uses the same load: exactly 4 busy processes whenever a reference or evaluation
task runs (filler processes cover the tail), on an otherwise idle machine.

*RL decision charge.* RL is charged its essential per-decision cost in an optimised
implementation, added to its engine search CPU time for the budget and time-to-best: native
forward pass of its actual network (3.2–9.8 µs depending on input size and action head;
`scripts/timing/mlp_decision_cost.rs`) + 2 µs state construction (0 for the no-state variant);
random/fixed control policies are charged 0. This makes the budgets symmetric with the
comparison methods, whose selection logic runs inside the engine. Recorded in every output:
`decision_cost_us`, `decision_charge_seconds` and `engine_search_cpu_seconds` in results /
restart CSVs (`search_cpu_seconds` = engine + charge), `rl_decision_charge` in the run manifest,
the charge's share of search in `summary.json`, and `decision_charge_ns_per_iteration` on every
operator-usage log row.

Evaluation checkpoints after every instance (`partial_rows.jsonl`, `--resume`) and records
per-restart action statistics (distinct actions, switches, action histogram); per-iteration
operator-usage logs stay on.

**Scripts**: `scripts/run_norm2026_ablation.py` (queue: 7 reference chunks, 38 trainings, 46
evaluations; status in `NORM2026_STATUS.md`),
`scripts/analyze_norm2026_ablation.py` (ratio-to-best tables, paired Wilcoxon + bootstrap CIs vs
reference and Kisialiou, policy-behaviour tables). Outputs: `output/norm26__<axis>__<variant>__
<size>__s<seed>/`, analysis in `output/norm26_analysis/`.

**Data provenance note (found 2026-10-08)**: `data/processed/alns/*` (what the solver, training and
the cache read) is created once from `data/generated/*` and never refreshed. Test datasets 01–05
of every size (and large 08), plus 15 medium/large training sets, differ between the two because
`data/generated` was regenerated on 2026-09-04. All experiments are internally consistent (they
all read the processed files), but any analysis reading instance properties from
`data/generated` describes the wrong instances for those datasets — read `data/processed` instead.

**Run**: launched 2026-10-09 15:01, finished 2026-10-10 13:54 (22.9 h), 46/46 evaluations, 0
failures. Outputs `output/norm26__*`, analysis `output/norm26_analysis/` (`summary.md`, tables,
`figures/`). Exp 8 stays locked as it was; its numbers come from the old wall-clock rule and are
not directly comparable to Exp 9's.

**Results — ratio to best-known cost** (1.0000 = best of every method and seed on that instance;
multi-seed variants scored on the per-instance mean cost over training seeds; W/L = per-instance
wins / losses vs Kisialiou):

*Medium (20 instances)*

| method | mean | median | max | W/L vs Kisialiou |
|---|---|---|---|---|
| Reward: ALNS score + time | **1.0037** | 1.0040 | 1.0094 | 16/4 |
| Reference (Exp 8 config), seeds 42/43/44 | 1.0355 | 1.0078 | 1.2915 | 13/7 |
| State: none / instance / search / solution / all | 1.0354–1.0355 | 1.0078 | 1.2915 | 13/7 |
| Control: fixed random + first-fit + FCR | 1.0355 | 1.0078 | 1.2915 | 13/7 |
| Action: improvement chain | 1.0466 | 1.0077 | 1.2864 | 13/7 |
| Control: uniform random (triple / many) | 1.0717 | 1.0010–1.0013 | 1.2934–1.2969 | 17/3 |
| Action: many (288 parameterised) | 1.1248 | 1.0047 | 1.2988 | 11/9 |
| Control: fixed random + first-fit + relocation | 1.1250 | 1.0048 | 1.2988 | 11/9 |
| Adaptive ALNS (no improvement) | 1.1843 | 1.2664 | 1.3166 | 4/16 |
| Reward: best-cost gain | 1.1851 | 1.2552 | 1.3046 | 12/8 |
| Reward: ALNS score / new best (sparse) | 1.2079 | 1.2803 | 1.3034 | 14/6 |
| Kisialiou ALNS | 1.2125 | 1.2834 | 1.5711 | — |

*Small (30 instances; trained variants averaged over seeds 42/43, per-seed means in brackets)*

| method | mean | median | max | W/L vs Kisialiou |
|---|---|---|---|---|
| Control: uniform random (many / triple) | **1.0289** | 1.0000 | 1.8616 | 22/3, 23/4 |
| Reward: best-cost gain | 1.0448 (1.0304–1.0592) | 1.0011 | 1.8616 | 15/15 |
| Action: improvement chain | 1.0457 (1.0316–1.0598) | 1.0018 | 1.8624 | 13/17 |
| State: all three groups | 1.0583 (1.0582–1.0583) | 1.0003 | 1.8708 | 18/10 |
| State: instance only | 1.0592 (1.0299–1.0884) | 1.0006 | 1.8705 | 21/8 |
| State: search process only | 1.0733 (1.0582–1.0884) | 1.0003 | 1.8705 | 21/6 |
| Reference (Exp 8 config) = control fixed random + first-fit + relocation | 1.0884 | 1.0003 | 1.9063 | 18/8 |
| State: solution only / none | 1.1072 (1.0884–1.1261) | 1.0047 | 1.8749 | 6/24, 5/25 |
| Reward: ALNS score | 1.1188 (1.0886–1.1490) | 1.0005 | 1.8989 | 18/9 |
| Action: many (288 parameterised) | 1.1218 (1.0291–1.2145) | 1.0102 | 1.8761 | 5/25 |
| Control: fixed random + first-fit + FCR | 1.1257 | 1.0084 | 1.9194 | 2/28 |
| Adaptive ALNS (no improvement) | 1.1626 | 1.0158 | 1.9293 | 1/29 |
| Kisialiou ALNS | 1.1635 | 1.0009 | 1.9063 | — |
| Reward: ALNS score + time | 1.1749 (1.0889–1.2609) | 1.0125 | 1.9308 | 3/27 |
| Reward: new best (sparse) | 1.2511 (1.1795–1.3228) | 1.1253 | 1.9384 | 1/29 |

Paired Wilcoxon tests and bootstrap CIs: `output/norm26_analysis/tests_{small,medium}.csv`.

**Findings.**
1. *Policies mostly collapse to one fixed configuration; some switch once per restart.* On medium
   the reference (all three seeds) and all five state variants learned *the same* action, random
   removal + first-fit + FCR, every step, and are numerically identical to the fixed-configuration
   control (same seeded search; 19/20 instances identical best cost) — automatic *algorithm
   configuration* rather than adaptive control. But several policies switch exactly once within a
   restart, always from a fleet/voyage-reduction phase to another operator
   (`scripts/analyze_norm2026_switching.py`, output `output/norm26_analysis/switching.txt`; the
   per-step switch rates of ~0.003 hid this — one switch per 300-step restart):
   - ALNS score + time, medium (the best medium method): 74% of restarts run Shaw + first-fit +
     FCR for the first ~11 iterations (median), then random + first-fit + VNR to the end; switching
     restarts improve 20.5% (median) vs 2.2% for restarts that never switch — 19.2% before the
     switch, 1.0% after it (medians over switch-once restarts).
   - All-three-groups state, small, seed 43: 99.9% of restarts run random + first-fit + VNR for
     ~42 iterations, then random + first-fit + relocation. Only 1 of 1,522 restarts stayed, so
     there is no "stayed" group; median restart gain 32.4%, of which 26.0% comes in the VNR phase
     and 0.4% after the switch (94% of restarts gain something after it). For comparison, the
     small reference policy (relocation throughout, never switches; seeds 42 and 43) gains a median
     3.6% per restart (analysed 2026-10-15).
   - Many operators, small, seed 43: 66% run random + greedy + VNR (k=3) on iteration 1, then
     random + first-fit + relocation (33.3% vs 2.9% improvement; 32.7% before the switch, i.e. in
     the first iteration, 0.7% after); similar one-switch patterns for
     best-cost gain, ALNS score and new best on small.
   Reduce first, then local search — Kisialiou's hand-designed chain order. Association, not
   causation: a deterministic policy switches when early progress changes its state (for the
   ALNS score + time policy the trigger is the removal of a vessel — see finding 7). The same
   pattern appeared in the Experiment 1 follow-up (every small restart started with FCR; restarts
   that moved on to relocation never stalled).
2. *State design does not matter on medium* (the policy ignores the state); on small it changes
   which configuration is learned but with large seed-to-seed variation (e.g. instance-only
   1.0299 vs 1.0884), so state effects on small are within seed noise.
3. *Reward design matters most, through time-awareness.* Rewards without a time term (ALNS score,
   new best) learn Shaw + first-fit + unlimited swap on medium: the swap improves almost every step
   (high per-step reward) but costs so much that only ~6 steps per search-second fit — 1.2079,
   worse than the adaptive baseline. Time-aware rewards learn cheap FCR / VNR policies with
   thousands of steps per second. The best medium result is ALNS score + time (1.0037, 19/20
   better than the reference, Holm p = 0.011; 16/4 vs Kisialiou): it mixes VNR (67%) and FCR and
   finds the smallest fleet on all 20 instances. On small the same reward is unstable across seeds
   (1.0889 vs 1.2609).
4. *Medium results are decided by fleet size.* One vessel more costs ~22–30%. Kisialiou uses one
   more vessel than the best RL policy on 13/20 instances (where its ratio is ~1.28) and wins only
   where it finds the same fleet; the reference misses the minimum fleet by one vessel on 2
   instances, ALNS score + time never.
5. *Uniform random operator choice is a strong baseline.* On small it is the best method (1.0289);
   on medium (1.0717) it beats Kisialiou on 17/20 instances but loses to the best fixed
   configurations. A richer action space without better learning (many, 288 actions) helps neither.
6. *The RL decision charge took 0–6.2% of RL's search budget* (`summary.json`,
   `decision_charge_share_of_search`): highest for policies on the cheapest operators (FCR, VNR) —
   6.2% small solution-only, 2.2–2.6% for the medium FCR policies — and at most 1.2% (mostly
   0.1–0.5%) for policies on relocation or swap.
7. *The best medium policy learned a state-dependent phase rule: cut cost with FCR until it is
   ≈18.5% below the restart's start — usually one vessel — then switch to VNR* (analysed
   2026-10-15). This is the one clear case of adaptive control in Experiment 9 — the switch is
   triggered by an event in the search, not by the iteration count or the instance size.
   - **Evidence** (`ALNS score + time`, medium, seed 42; 20,985 restarts on 20 test instances; from
     the per-iteration operator logs `logs/norm26__reward__alns_score_time_v1__medium__s42/` and the
     per-restart fleet sizes in `output/norm26__reward__alns_score_time_v1__medium__s42/restart_results.csv`;
     scripts `output/presentations/NORM2026_HiMolde_files/build/{switch_trigger,fleet_switch}_analysis.py`):
     - Every restart starts with Shaw + first-fit + FCR. 74% switch once to random + first-fit +
       VNR and keep it to the end of the restart; 25% never switch; ~1% switch more than once.
     - **The trigger is a cost ratio, not the fleet.** The rule "switch once the cost is ≥ 18.5%
       below the restart's starting cost" classifies **100%** of restarts correctly (switched vs
       stayed); "switch after a vessel is cut" only 88%. The step that crosses 18.5% saves a median
       **151k** (IQR 149–153k) — one 150k charter — so on most instances the threshold means
       "a vessel is gone".
     - Where one vessel is only ~15% of the starting cost (6 of 20 instances: test 01, 09, 12, 16,
       18, 19), one vessel does not reach the threshold: 46% of the restarts that never switched
       *did* cut a vessel (their drop stalls at 15.8–17.2%), and the restarts that switched there
       had cut two. Those instances have the lowest switch shares (0.31–0.84). 37% of all switching
       restarts switched only after cutting 2+ vessels. Plausibly the policy learned "a vessel ≈
       18–19% of cost" from the training instances and uses the cost ratio as its proxy.
     - The reaction is immediate: **67%** switch on the very next step after crossing, 75% within
       4 steps (95th percentile 28).
     - Not "switch because no further vessel can be cut": 58% of switching restarts end one vessel
       above the best-known fleet (minimum over every medium Exp 9 run and method), 41% at it. The
       non-switching restarts keep running FCR for the whole restart (54% never cut a vessel; 87%
       end one vessel above the best-known fleet).
     - Fleet results per restart vs. other methods (same instances): cut ≥ 1 vessel — ALNS score +
       time 86%, reference RL (always FCR) 67%, Kisialiou 46% (100 long restarts), classical ALNS 6%;
       reached the best-known fleet — 31%, 11%, 9%, 0.3%.
     - Consequence for interpretation: the 20.5% vs 2.2% restart-gain gap in finding 1 is mostly the
       reduction itself — the gain comes before the switch (19.2% before, 1.0% after; medians over
       switch-once restarts); the switch is a learned reaction, not the cause of the gain.
   - **How it was achieved** — no hand-coded rule; it emerged from the following setup (only the
     reward differs from the Exp 8 reference configuration):
     - *Reward* `alns_score_time_v1`: the classic ALNS score rescaled to new best = 1, better than
       current = 9/33, accepted worse = 3/33, rejected = 0, minus a search-CPU-time tax of 0.002 per
       ms (capped at 0.5). The score makes finding a new best (such as a removed vessel) by far the
       most valuable outcome; the time tax makes cheap operators (FCR ~0.3 ms, VNR, Shaw/random
       removal, first-fit) preferable to expensive ones (unlimited swap ~160 ms), so the policy
       can afford thousands of steps and many restarts per search-second.
     - *State* `features_improvement_v3` (45 features): includes `num_vessels_used`, number of
       voyages / empty voyages, current and best cost relative to the restart's initial cost,
       progress, stagnation, SA acceptance signals and the last improvement operator — so a removed
       vessel is directly visible to the policy (vessel count drops, cost ratio falls by ~19%).
     - *Action* `op_triple_v2` (48 destroy × repair × improvement triples), *training* PPO
       (Stable-Baselines3), 200k timesteps, entropy coefficient 0.01, 300-iteration restarts,
       medium training instances, seed 42; evaluated greedily (deterministic policy) on the 20
       medium test instances under the equal search-CPU-time budget with the RL decision charge.
       Commands: `output/norm26__reward__alns_score_time_v1__medium__s42{__train,}/run_command.txt`.
     - The same state and action with the reference reward (`delta_cost_v9`) collapsed to a single
       triple (random + first-fit + FCR) for the whole restart, so the reward is what produced the
       phase behaviour.
   - **Not yet known:** which state feature(s) the policy actually keys on. The 100% vs 88% fit
     points to the cost ratio ((initial − best) / initial or best / initial, both in the state)
     rather than `num_vessels_used`, but the operator logs do not record the observation vector. A
     replay of the saved model recording the 45-dim state per step plus a small decision tree
     predicting FCR vs VNR would pin it down. Only one training seed exists for this variant, and
     the small-instance switches (VNR → relocation) have not been analysed this way.
   - **Why this does not contradict finding 2** (state does not matter on medium): the state
     variants were all trained with the reference reward, under which the policy collapses to one
     triple and ignores its input. Under ALNS score + time the policy *does* use the state — a
     deterministic policy can only change its action within a restart if its observation changes —
     so the reward decides whether the state is used at all. State × reward interactions were not
     tested (one-factor-at-a-time design); state variants under `alns_score_time_v1` are the
     obvious follow-up.
8. *The improvement-chain policies rebuilt the reference configuration with idle repeats*
   (analysed 2026-10-15, `improvement_chain_v2`; RL picks backbone vs. one of 4 improvement
   operators, the engine's adaptive roulette picks destroy/repair, at most 3 consecutive
   improvement steps — `--chain-max-consecutive-improvements` default 3).
   - Medium (seed 42, 12,758 restarts): backbone steps are exactly 25.0% of steps — the policy
     never chooses the backbone voluntarily, every chain runs to the cap: backbone → X → X → X.
     Chains repeat one operator: FCR×3 70%, VNR×3 30% (VNR chains 23% of chains in the first half
     of a restart, 37% in the second). The engine's roulette mostly picks first-fit repair
     (random + first-fit 26%, worst + first-fit 20%, Shaw + first-fit 9%).
   - Almost all improvement steps are idle: the solution changes on 1.3% of chain position 1,
     0.1% of position 2 and 0% of position 3 (FCR/VNR converge in one call). Effective algorithm:
     adaptive destroy + mostly first-fit, then FCR or VNR once, plus two wasted calls — essentially
     the reference triple random + first-fit + FCR, hence the close cost (1.0466 vs 1.0355). The
     repeats are cheap (FCR ~0.3 ms), so they cost throughput, not quality.
   - Small: seed 42 chains FCR (39.5% of steps) and unlimited swap (32.8%, expensive yet still
     repeated); seed 43 chains relocation (44.3%) and FCR (30.7%). Best action variant on small
     (1.0457 vs reference 1.0884).
   - The reference reward's −1 repeat penalty (`delta_cost_v9`, inherited from v7/v8) *does* fire
     — but a backbone step is punished harder. Logged rewards (medium, one test instance, ~63k
     chains): chain position 1 mean +0.006 (median −0.05, the improvement-step tax), positions 2
     and 3 ≈ −1.05 (repeat penalty), backbone (destroy + repair) steps mean **−1.38** (median
     −0.12, heavy negative tail: a destroyed-and-repaired candidate is usually worse than the
     current solution and `delta_cost_v9` charges 30% of that loss, ×10 vs current and ×30 vs
     best, up to ×2 late in the restart). An idle repeat (−1.05) beats a perturbation (−1.38) on
     average, so the policy idles until the cap forces the backbone. The time term plays no role
     here (FCR ≈ 0.3 ms → −0.0015). This is the per-step-reward problem in its clearest form: the
     perturbation ALNS needs is penalised for its immediate cost, with no credit for the
     improvement it enables later.

**Caveats.** One training seed per medium variant (the reference's three seeds collapsed
identically); small results vary strongly between seeds; one eval seed; 20 / 30 instances.

---

## Experiment 8 rerun — setup and launch (2026-10-22)

**Why.** Experiment 8's numbers (locked 2026-09-08) were produced with (1) wall-clock time
budgets that counted the Python prototype's overhead against RL, (2) baselines whose repair pool
lacked the first-fit insertion operator RL's action space has, (3) a cost model with a 150,000 NOK
weekly charter and an unrealistically cheap fuel price, (4) test sets that were 50 draws from one
configuration per size. The rerun fixes all four and uses one seed (training and evaluation 42).
Everything is retrained and the baseline/Kisialiou reference regenerated; nothing from Exp 8/9 is
reused (Exp 8 stays locked as it was).

**What changed (and where).**
1. *Search-CPU-time budgets, RL charged a native decision cost* — the Exp 9 protocol (section
   "Timing protocol and RL decision overhead"): every method gets the search CPU time Kisialiou's 5
   reference restarts needed on that instance; RL is charged 4.64 µs (native forward pass of its
   45→64→64→48 network, measured) + 2.0 µs (state construction, an estimate; conservative bound 17–38 µs)
   = 6.64 µs per decision, instead of its measured Python overhead.
2. *First-fit in both baselines* — new action module `adaptive_weighted_v2` (rl/modules/actions.py):
   3 Kisialiou-calibrated destroy × 4 repair (deep greedy rank 1, k-regret 2, k-regret 3,
   `first_fit_insertion_v1`). Passed as `--baseline-action-module`, it restricts the adaptive
   baseline and Kisialiou's ALNS alike. `adaptive_weighted_v1` is untouched.
3. *Cost model in NOK* — charter 7 × 250,000 = **1,750,000 per vessel and week**
   (py_alns/resource/generation_config.yaml, all fleet entries; only vessels used are paid);
   fuel price **13,500 NOK per tonne** (rust_alns/src/structs/constants.rs; MGO 1,425 USD/t, Ship & Bunker
   Rotterdam 21 Oct 2026, 2-week range 1,333–1,437, at ≈ 9.5 NOK/USD) instead of 650; consumption
   unchanged (0.43 / 0.26 t/h). Because the simulated-annealing start temperature is in cost units, it
   is scaled with the fuel price: 500 × 13,500 / 650 = **10,384.6** (`default_initial_temperature()`,
   read from the engine by the Python env). Rewards, states and operators use relative quantities
   (checked), so nothing else depends on the cost scale. Recorded in every run manifest
   (`resolved.cost_model`).
4. *Diversified test sets* — scripts/generate_test_set_v2.py: per size a full factorial of
   **time-window density {0, 25, 50, 75, 100} % × demand/frequency level DF1–DF5 × 2 replicates =
   50 instances**, each its own configuration (DF1–5 = the parameter points of the train DF
   configs: small demand + frequent visits → large demand + infrequent; weekly demand per
   installation ≈ constant). Installation counts spread over the tier's range (small 6–12,
   medium 12–20, large 20–40) by a Latin-square rank assignment, balanced over both factors (mean
   count by TW level 28.8–30.8 and by DF level 28.8–30.8 for large); tight window 7–19 throughout. Design
   table with realised properties: `data/rerun26/generated/test_set_v2_design.csv` (tracked copy: `py_alns/resource/test_set_v2_design.csv`). Train sets are
   the unchanged 13 configs × 5 per size, regenerated (`generate_datasets.py`). All data lives in a
   separate tree (`data/rerun26/{generated,processed/alns}`, selected through the
   `PSVPP_GENERATED_ROOT` / `PSVPP_PROCESSED_ROOT` environment variables) so earlier experiments'
   data and caches are untouched; `DATA_MANIFEST.json` stores counts and checksums.
5. *Robustness fixes found while preparing* — (a) the all-tight-window cells with frequent visits are
   hard for the constructive heuristic on large instances (up to 63% of attempts failed with 14
   vessels; ~1% of restarts panicked with the 10-attempt limit). **Large fleet raised 14 → 18 vessels
   for train and test** (zero failed attempts afterwards; on instances where 14 sufficed the search
   is bit-identical and costs the same CPU time, since unused vessels cost nothing), and the attempt
   limit raised 10 → 500 as a safety net (the 585 sampled initial solutions are bit-identical to the
   old build). Medium/small keep 9 vessels. (b) Per-iteration operator logs would be ~130 GB per
   large RL evaluation under CPU-time budgets; decision 2026-10-22: log **every restart of every
   size** (expected ≈ 300 GB in total, ≈ 200 GB of it large; range 220–400 GB; SSD has ~700 GB free).
   Only the fixed-configuration controls on medium/large log their first 20 restarts per instance
   (a fixed triple carries no information in the per-iteration rows; their restart summaries are
   complete). `--operator-log-restarts N` is the switch. The queue starts no new task while the SSD has
   less than 120 GB free (running tasks continue, the queue resumes by itself when space is freed).

**Jobs** (`scripts/run_exp8_rerun.py`; one process each; seed 42; PPO 200k steps, ent_coef 0.01,
ph300, `op_triple_v2` / `features_improvement_v3`):
reference chunks (Kisialiou + adaptive baseline, cost-balanced slices of the 50 test instances) →
trainings (`main` = `delta_cost_v9`, the Exp 8 reward; `alt` = `alns_score_time_v1`, Exp 9's best
medium reward) → evaluations of every model on all 50 instances (cost-balanced shards, merged) →
fixed-configuration controls (random removal + first-fit + FCR on every size, + relocation on
small; RL's collapsed triples without RL). The queue keeps exactly 4 CPU-bound processes busy
whenever a reference/evaluation task runs (CPU time of fixed work depends on load: 2.37 s alone vs
~3.9 s with 4 jobs), and logs the CPU time of a fixed probe workload every 30 min
(`logs/rerun26_queue/speed_probe.csv`) to document machine-speed drift.

**Outputs and manifests.** Every task writes the usual `run_manifest.json` (git state incl.
untracked files, CLI args, resolved paths, cost model, data roots, baseline operator pool, logging
policy, timestamps), `run_command.txt`, results/summary/report and its operator-usage and training
logs (`output/rerun26__<tag>[__train|__shNN]/`, `logs/rerun26__<tag>.../`); merged evaluations get
`merge manifest` + combined results; the queue writes `output/rerun26__QUEUE_MANIFEST.json` (full
plan, commands, predictions, outcomes), `RERUN26_STATUS.md`, `logs/rerun26_queue/events.log`.

**Progress is written steadily, not at the end.** Reference chunks append every finished instance to
their cache part. Evaluation shards write, after **every finished instance**, `partial_rows.jsonl`,
`partial_restart_rows.jsonl`, `progress.json`, `results_partial.csv` and `summary_partial.json`
(`summary.json` appears only when the shard is complete); operator-usage logs stream per restart.
Trainings write SB3's `progress.csv` / TensorBoard events continuously, a model checkpoint every 10k
steps (`output/.../model/checkpoints/`) and refresh `training_summary.txt`, `training_costs.csv` and
`training_progress.json` every 10k steps. The queue rewrites `RERUN26_STATUS.md` every minute: running
tasks with progress, SSD free space, and a **results-so-far table per evaluation job** (mean
ratio-to-best of policy / baseline / Kisialiou over the instances finished so far, W/T/L).

**Cost-model consistency check** (1 Kisialiou restart on 14 of the new instances, all three sizes,
after the cost change): charter is 81–87% of total cost both in the initial and the best solutions
(fuel 13–19%; before the change ≈ 89%). By construction fuel can never rival the charter per vessel:
a vessel sailing non-stop for a week burns 0.43 t/h × 168 h × 13,500 = 0.98 M NOK, 56% of the
1.75 M charter, and real solutions use a vessel only part of the week. One vessel more or less
(1.75 M) is still worth ≈ 300 h of sailing fuel, so fleet size dominates the objective as before.

**Predicted runtime** (`python scripts/run_exp8_rerun.py --list`; model calibrated on Exp 8 caches,
Exp 9's queue timings and Kisialiou runs on 10 of the new instances under the 4-process load; the
measured/predicted ratio was 0.7–1.6 per instance, mean ≈ 1.3 for medium/large, which is included):

| process-hours, 4-process load | references | train main | eval main | controls | train ALNS-score+time | eval ALNS-score+time | total |
|---|---|---|---|---|---|---|---|
| small | 2 | 0.3 | 1 | 5 | 0.6 | 1 | 10 |
| medium | 11 | 1 | 20 | 13 | 1 | 12 | 58 |
| large | 64 | 8 | 87 | 61 | 10 | 56 | 286 |
| all | 77 | 9 | 109 | 79 | 11 | 69 | 354 |

Wall-clock on 4 workers (list scheduler, required work first, optional work last; range =
predictions ×0.8 / ×1.4): required part (references, main models, controls) **≈ 2.9 days**
(2.3–4.0); with the ALNS-score+time model on small and medium ≈ 3.0 days; with it on all three
sizes **≈ 4.0 days** (3.2–5.6), the large ALNS-score+time evaluation being the last, cuttable tail.
The large test instances dominate (DF1/DF2 cells with 80–100 visits ≈ 70% of the large time); the
installation count of large DF1/DF2 was capped at 28 for this reason (uncapped: visits up to 140,
≈ 2× the large time).

**Launch command** (as run on 2026-10-22, with the ALNS-score+time model on all sizes):

```bash
nohup caffeinate -is bash -c 'for i in 1 2 3 4 5; do python scripts/run_exp8_rerun.py --workers 4 \
    --alt-sizes small,medium,large && break; sleep 120; done' > logs/rerun26_queue/queue.log 2>&1 &
python scripts/run_exp8_rerun.py --status      # or read RERUN26_STATUS.md
```
(the runner is resumable and adopts tasks that are still running, so the retry loop and a manual
re-launch are safe; to stop everything: `pkill -f run_exp8_rerun.py; pkill -f run_experiment_rl_improvement_mvp.py`).


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

---

## Design notes (open questions, not bugs)

### Repeat penalty keys on the improvement operator only (noted 2026-10-12)

`delta_cost_v7`/`v8`/`v9` subtract `NO_OP_IMPROVEMENT_PENALTY = 1.0` when the selected
improvement operator equals the previous iteration's (`improvement_operator_idx ==
env.last_improvement_idx`) **and** the step was not accepted. The destroy and repair
choices are not part of the condition, so the penalty fires even when the policy picked a
different D/R pair and only reused the improvement operator — and it does not fire when
the policy repeats the whole triple but switches improvement operator.

This made sense where it originated (`improvement_chain_v2`, where consecutive
improvement-only steps act on an unchanged schedule), but it is a weaker argument for
`op_triple_v1/v2`, where every step also runs destroy+repair. Worth keeping in mind when
reading operator-usage results.

Related observation from the same check: all three destroy operators draw randomness
(the removal count is always `xi_min + U(0,1)*(xi_max-xi_min)`; Shaw also picks its seed
visit at random), while `deep_greedy_insertion` and `k_regret_insertion` use no RNG at all
and `first_fit_insertion` shuffles voyages. So repeating a triple does not reproduce an
identical candidate, only a similar one — unlike Kisialiou et al. (2018), where the
destroy operators are deterministic given their input parameters.

Status: logged only, no change made. Low priority for paper 1 results.

### Feasibility guards added to the engine (2026-10-12)

While documenting the solver for paper 1 it turned out that (a) `run_iteration` accepted
candidates on cost alone, with no completeness/feasibility check, even though an
incomplete schedule is *cheaper* (unassigned visits cost nothing), and (b) the Python env's
infeasibility branch read `result["feasible"]`, a key the Rust side never sets (the real
keys are `solution_metrics.is_complete` / `is_feasible`), so it was dead code.

Fixed in `rust_alns/src/alns/engine.rs`: the improvement sequence is skipped when repair
leaves visits unassigned, and a candidate is accepted only if it is complete and fully
feasible. Also `python_interface.rs` now defaults the log filter to `warn` (env_logger's
default is error-only, which hid every operator warning in Python-driven runs), and the
12 `warn!` calls inside `Solution::is_fully_feasible` were downgraded to `debug!` because
operators call that method to evaluate trial moves.

Measured effect: zero discards over 1,200 iterations (small baseline; small/medium/large
in RL mode, 300 iterations each), and every iteration was complete and feasible both
before and after the change. The guards are a safety net; existing results stand.

### Candidate action: early restart break (idea, 2026-10-12)

Add an extra action to the action space that ends the current restart early, so the search
continues from a freshly constructed initial solution instead of spending the remaining
iterations on a schedule the policy judges to be stuck. Currently a restart always runs its
full iteration budget (`max_iterations`, 300 in the ph300 runs), and restarts are triggered
only by the eval loop, so the policy has no way to abandon an unpromising basin.

Reward-hacking risk to design against: the reward pays for improvements relative to the
*current* schedule (weight 10) as well as relative to the best known one (weight 30). A
fresh initial solution is poor, so the early iterations after a break produce large
improvements over the current schedule at no cost in quality. A policy could therefore
learn to break often and farm those easy gains while rarely improving the best known
schedule. Possible counters: pay only for improvements over the best known schedule after a
break, charge an explicit cost for the break (it discards the iterations spent so far and
pays for constructing a new initial solution), require a minimum number of iterations or a
minimum stagnation count before a break is allowed, or evaluate breaks on the time-fair
budget so their restart cost is visible in the metric the policy is scored on.

Status: idea only, not implemented.
