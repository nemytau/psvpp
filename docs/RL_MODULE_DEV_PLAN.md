# RL Module Requirements & Development Plan

## Project Context
Project context: ALNS + RL for PSVPP (Rust core + Python RL wrapper). Aligns with roadmap phases 3 through 5.

---

## 1. Goals & Scope

### Primary goals
- Train RL agents that select ALNS operators to improve convergence speed and final cost.
- Provide a repeatable experiment workflow: train -> test -> evaluate -> solve.
- Make action, state, and reward modules pluggable to enable rapid iteration.

### Out of scope (for now)
- End-to-end fleet simulation beyond the PSVPP instance model.
- Distributed RL training at scale (candidate for future work).

---

## 2. Core Commands (done)

All commands operate on a YAML or JSON config with optional overrides. Suggested entrypoints:

```
# Train on the training split, save model, logs, and metadata
python -m rl.train --config configs/ppo_default.yaml --exp-name ppo_v1 --seed 42

# Evaluate a saved model on the test split and emit summary artefacts
python -m rl.test --model runs/ppo_v1/model.zip --config configs/ppo_default.yaml --include-baseline

# Generate comparison plots against the random baseline and export CSVs
python -m rl.evaluate --model runs/ppo_v1/model.zip --config configs/ppo_default.yaml --output-dir reports/ppo_v1_eval

# Solve a single processed instance with a given model
python -m rl.solve --model runs/ppo_v1/model.zip --instance data/processed/alns/small/test/small_test_1_01 --config configs/ppo_default.yaml
```

**Notes**
- Each command reads the experiment manifest (Section 6) when the model path lives under `runs/<exp_id>/`. Dataset splits, seeds, and iteration limits remain aligned across commands.

---

## 3. Functional Requirements

### 3.1 Train
- Load the training split via `GeneratedDatasetManager`.
- Construct `ALNSEnvironment` with the chosen action, state, and reward modules.
- Train PPO (or another algorithm) while logging:
  - Policy and value losses, entropy, approximate KL, explained variance.
  - ALNS specifics: operator usage histograms, acceptance rate, best-cost trace, feasibility ratio, temperature trends.
- Save artefacts:
  - `model.zip`, `manifest.json`, `config.yaml`, per-episode CSVs, TensorBoard logs.
  - Comparison outputs under dedicated subdirectories (Section 6).
- Tag outputs with the generated experiment ID (Section 6) and copy the resolved config.

### 3.2 Test
- Load the test split (manifest-specified if available, otherwise via the dataset manager).
- Run one evaluation episode per instance with the trained policy (no learning).
- Emit `evaluation_summary.json`, per-instance details, convergence plots, and optional baseline statistics.
- Compare against training metrics to detect overfitting (delta in final cost, convergence rate, action entropy).

### 3.3 Evaluate (baselines and comparisons)
- Run the trained policy and a random baseline on each test instance across configured seeds.
- Produce artefacts:
  - Percent-gap convergence curves for best and current cost.
  - Combined comparisons (mean +/- one standard deviation shading).
  - Per-instance CSV summaries with best-cost deltas and iteration counts.
- Persist outputs under `runs/<exp_id>/model_vs_baseline/` (or a custom `--output-dir`).

### 3.4 Solve (single instance)
- Load a single processed instance and execute the chosen policy.
- Return final schedule statistics and write a JSON summary compatible with visualisation scripts.

---

## 4. Modularity: Action, State, Reward (hot-swappable) (done)

### 4.1 Registries
Implement a lightweight registry pattern so new variants can be added without touching core loops.

```
# rl/registries.py
ACTION_REGISTRY: dict[str, type[Any]] = {}
STATE_REGISTRY: dict[str, type[Any]] = {}
REWARD_REGISTRY: dict[str, type[Any]] = {}

def register_action(name: str):
	def decorator(cls):
		ACTION_REGISTRY[name.strip()] = cls
		return cls
	return decorator

# Equivalent decorators exist for state and reward registries.
```

Configuration selects implementations by key:

```
# configs/ppo_default.yaml
modules:
  action: op_pair_v1
  state: features_v2
  reward: delta_cost_v3
```

### 4.2 Interfaces
```
class ActionSpace:
	def n(self) -> int: ...
	def id_to_action(self, action_id: int) -> tuple[int, int]

class StateEncoder:
	def space(self) -> spaces.Space: ...
	def encode(self, result: dict[str, Any] | None, env: Any) -> np.ndarray

class RewardFn:
	def compute(self, result: dict[str, Any], env: Any) -> float
```

Each implementation carries a version string that is logged into the manifest for traceability.

---

## 5. Metrics & Diagnostics

### 5.1 Convergence and quality

---

## 6. Experiment Management & Reproducibility (done)

### 6.1 Experiment ID and layout
Experiment IDs follow the pattern:

```
<timestamp>__<dataset>__<algo>__A-<action>__S-<state>__R-<reward>__seed<k>
# Example: 20241112_153045__small__ppo__A-op_pair_v1__S-features_v2__R-delta_cost_v3__seed42
```

Directory layout under `runs/<exp_id>/`:

```
config.yaml                 # Resolved configuration snapshot
manifest.json               # Git metadata, environment info, dataset hashes
config/                     # Optional copy of the source config file
model/                      # Stable-Baselines3 save directory
model.zip                   # Convenience export of the policy
artifacts/                  # Ad hoc artefacts (solution JSON, traces)
convergence/                # Training cost curves and CSVs
cfg = load_cfg(args.config)
baseline_random/            # Random baseline results
model_vs_baseline/          # RL vs baseline comparisons
tb/                         # TensorBoard logs
```

### 6.2 Manifest contents
- Experiment ID and creation timestamp.
- Git commit hash, branch, and dirty flag.
- Python, Rust, Stable-Baselines3, Gymnasium, Torch, and NumPy versions.
- Dataset hashes for train and test splits (with combined digests).
- Action, state, and reward module identifiers with versions.
- Training hyperparameters (timesteps, learning rate, sampling strategy, iterations).
- Evaluation settings (deterministic flag, seeds, summary statistics).
- Artefact locations (model paths, TensorBoard directory, evaluation directories).

### 6.3 Reproducibility switches
- Centralised seeding for environment, NumPy, and torch.
- Deterministic and cuDNN flags when using GPU backends.
- Serialise observation normalisers if VecNormalize is introduced.
- Manifest integrity checks to confirm dataset splits have not drifted.

---

## 7. Dataset Management
- Source datasets live under `data/generated/<size>/<split>/` and are converted into `data/processed/alns/...` on demand.
- `GeneratedDatasetManager` prepares processed directories (installations, vessels, base CSVs) and caches conversions.
- Maintain a split manifest with hashes to guarantee integrity and enable manifest verification.

---

## 8. Configuration System
Use a minimal YAML schema (Hydra optional). Example:

```
# configs/ppo_default.yaml
exp_id = build_exp_id(cfg, args)
	size: small
train:
	algo: ppo
	total_timesteps: 50000
	max_iterations: 100
	sampling_strategy: random
	seed: 42
	learning_rate: 3e-4
logger = RunLogger(exp_id, cfg)
	deterministic: true
	seeds: [42]
logging:
	base_dir: runs
modules:
	action: op_pair_v1
	state: features_v2
	reward: delta_cost_v3
```

At runtime, resolve the config and write a frozen snapshot to `runs/<exp_id>/config.yaml`.

---

## 9. Evaluation Outputs
- `evaluation_summary.json`: mean and standard deviation of rewards, per-instance details, manifest reference.
- Convergence plots for current and best cost (percent gap vs best-known).
- Baseline summaries: CSVs listing best-cost deltas, iterations, and plot locations.
- Optional report bundles can be written to `reports/<exp_id>/` when using a custom output directory.

---

## 10. Interfaces with Rust ALNS
- `rust_alns_py.RustALNSInterface.initialize_alns(...)` sets up an instance and returns initial metrics.
- `execute_iteration(...)` advances one iteration given destroy and repair operator indices.
- Returned payload includes `current_cost`, `best_cost`, feasibility flags, operator identifiers, acceptance status, delta cost, temperature, and iteration counters.
- The Gymnasium environment wraps these calls and exposes observations, rewards, and info dictionaries.
- Initial implementation targets single-environment training; vectorised wrappers can be explored later.

---

## 11. Error Handling & Validation
- Validate config keys and raise actionable errors when required fields are missing.
- Before training: confirm dataset hashes match manifest expectations and registry entries exist.
- After training: verify manifests reference existing artefacts and that evaluation directories are populated.
- Provide a `--resume` path that reuses existing artefacts when a run directory already contains checkpoints.

---

## 12. Future Extensions
- Support alternative algorithms (A2C, DQN, SAC) behind a `make_algo` factory.
- Parallel evaluation across instances to reduce wall-clock time.
- Optional Weights and Biases or MLflow integration for experiment tracking.
- Curriculum or instance-mix training strategies.
- Operator-level learning (for example, learned destroy heuristics or adaptive operator sets).

---

## 13. Checklists

### Before training
- Dataset hashes verified.
- Config frozen and copied to run directory.
- Action, state, and reward versions logged.
- Seeds set for environment and libraries.

### After training
- Model artefacts saved (`model.zip` and SB3 directory).
- Operator statistics exported.
- Convergence CSVs generated.
- Manifest written with evaluation summary.

### Evaluation
- Baselines executed (if requested).
- Convergence and anytime plots created.
- Diversity metrics computed or scheduled.
- Summary CSV and README added to report directory when publishing results.

---

## 14. Minimal Code Stubs
```
# rl/manifest.py
@dataclass
class Manifest:
		exp_id: str
		created_at: str
		commit: str
		dataset_hash: str
		split_hashes: dict[str, str]
		modules: dict[str, str]
		hyperparams: dict[str, Any]
		seeds: dict[str, Any]

# rl/cli/train.py (skeleton)
args = parse_args()

exp_id = build_exp_id(cfg, args)
manager = ExperimentManager(exp_id, base_dir(cfg))

env = make_env(cfg)
model = make_algo(cfg, env)
model.learn(total_timesteps=cfg.train.total_timesteps, callback=manager.callback())
manager.save_artifacts(model, env)
```

---

## 15. Acceptance Criteria
- A single command produces a self-contained run directory with model, metrics, convergence CSVs, operator stats, and a manifest.
- Evaluation commands output side-by-side comparisons versus baselines with plots, CSVs, and manifest references.
- Swapping action, state, or reward modules in the config requires no code changes outside their dedicated modules.

---

End of document.
model = make_algo(cfg, env)
model.learn(total_timesteps=cfg.train.total_timesteps, callback=logger.callback())
logger.save_artifacts(model, env)
```

---

15) Acceptance Criteria
	-	Single command produces a self-contained run directory with: model, metrics, convergence CSVs, operator stats, and a manifest.
	-	Evaluation command outputs side-by-side comparisons vs baselines with plots and CSVs.
	-	Swapping action/state/reward in config requires no code changes outside their modules.

---

End of document.
