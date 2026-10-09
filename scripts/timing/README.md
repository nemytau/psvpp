# Timing measurements

Scripts behind the timing protocol of Experiment 9 and the RL-overhead argument for the paper.
Results and their interpretation: `docs/EXPERIMENTS.md`, section "Timing protocol and RL decision
overhead". All numbers below were measured on 2026-10-09 on the M1 MacBook Air (Python env
`py312` is x86_64 under Rosetta 2).

| Script | What it is for | Result it produced |
|---|---|---|
| `mlp_decision_cost.rs` | Native (compiled, arm64) cost of one RL decision's forward pass + argmax for a given network size. Gives the per-decision charge RL pays in Exp 9 (`NATIVE_FORWARD_US` in `scripts/run_experiment_rl_improvement_mvp.py`). | 45→64→64→48: 4.3–4.6 µs; 45→5: 3.6 µs; 45→288: 9.8 µs; 1/11/34/15/60 inputs →48: 3.2 / 3.6 / 3.9 / 3.8 / 5.0 µs |
| `rl_step_overhead.py` | Splits one step's time into engine search, Python inside `env.step`, policy inference and environment setup, for RL, baseline and Kisialiou; also times the policy decision in SB3 / lean PyTorch / NumPy and the engine's solution-metrics call (upper bound for state construction). | Medium, per step: RL 0.17 ms search vs 0.31 + 0.22 ms Python (24% search); baseline 1.88 vs 0.24 ms (89%); Kisialiou 196 vs 1.24 ms (99%). Decision: SB3 199 µs, PyTorch 44 µs, NumPy 12 µs. Metrics call: 17 / 25 / 38 µs (small / medium / large). |
| `exp8_timing_analysis.py` | From Exp 8's outputs: each method's search share of its wall-clock budget, and RL's decision overhead relative to its search per decision (realistic and conservative). | Search share (small / medium / large): RL 65% / 21% / 40%, Kisialiou 93% / 99% / 100%. Overhead: realistic 0.6% / 3.2% / 1.5% median (≤ 4.9%), conservative 1.9% / 15% / 9.9% median. |
| `search_cpu_vs_load.py` | How the search CPU time of a fixed, deterministic amount of work changes with machine load (run alone, as several simultaneous copies, or next to CPU hogs). Motivates the constant-load rule. | 2.37 s alone; 3.0–3.3 s with 3 copies; 3.8–4.0 s with 4 copies; 3.3–3.5 s next to 7 hogs; QoS "user-interactive" did not help. |
| `clock_check.py` | Which CPU clock is trustworthy in this Python. | Under Rosetta the thread clock reads ~41.7× too small (24 MHz ticks, not ns); the process clock is correct, so the engine uses `CLOCK_PROCESS_CPUTIME_ID`. |

Running them (from the repo root):

```bash
rustc -C opt-level=3 -o /tmp/mlp_decision_cost scripts/timing/mlp_decision_cost.rs && /tmp/mlp_decision_cost 45 48
PYTHONPATH=. python scripts/timing/rl_step_overhead.py --size medium --model output/<run>/model/ppo_rl_improvement_mvp.zip
python scripts/timing/exp8_timing_analysis.py
PYTHONPATH=. python scripts/timing/search_cpu_vs_load.py
python scripts/timing/clock_check.py
```

Run timing measurements on an otherwise idle machine (no experiment queue running): the
measured values depend on machine load (see `search_cpu_vs_load.py`).
