# RL-ALNS Experiments

## Overview

Each experiment tests a different RL action-space design on top of the ALNS solver.
All experiments are evaluated on the **small** dataset (5 instances, 5 seeds = 25 samples)
using **time-fair** mode: RL/baseline receive the same wall-clock budget as Kisialiou's
reference restarts.

### Common setup
- Training: 200 000 PPO timesteps, 50 iterations/episode, seed 42, small dataset
- Eval: `match_kisialiou` runtime mode, 5 eval seeds (42–46), 5 datasets
- **Baseline**: adaptive weighted ALNS (`backbone_v1`) — engine selects destroy/repair
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
any improvement operator. Baseline also runs without improvement (backbone_v1).

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

## Historical experiments (pre-refactor)

Earlier experiments ran with single non-parameterized operators. Bugs fixed
before the current series:
- Kisialiou improvement sequence used exact bare-name matching → found nothing
  after operators were renamed to `_v{i}` format (fixed: prefix matching)
- `VoyageNumberReduction.search_k` was a no-op (called `top_k_visit_insertion_costs`
  per-voyage in an outer loop; fixed: single call ranks all voyages)
- `KRegretInsertion.search_k` was a no-op for regret computation (fixed: removed
  redundant variants, kept single k=2 and k=3)
- Baseline used random explicit pair selection (fixed: `backbone_v1` uses adaptive
  weighted engine selection)
- Improvement operators missing from logs for D×R×I triple actions (fixed:
  `log_improvement` now true whenever `improvement_idx_result is not None`)
