"""Experiment 8 timing analysis: search share of wall time, and RL's decision overhead ratio.

Reads Experiment 8's results.csv / restart_results.csv (small, medium, large) and prints
  1. per method, the engine (search) time as a share of the wall-clock budget it was given, and
  2. RL's search time per decision per instance, and the decision overhead of an optimised
     implementation relative to it -- realistic (native forward pass + ~2 us state) and a
     conservative bound (forward pass + the engine's current solution-metrics call).
Produced tables 1 and 4 of docs/EXPERIMENTS.md, "Timing protocol and RL decision overhead".

    python scripts/timing/exp8_timing_analysis.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXP8_RUNS = {
    "small": "output/rl_improvement_mvp_20260906_063629__*",
    "medium": "output/rl_improvement_mvp_20260906_070345__*",
    "large": "output/rl_improvement_mvp_20260907_125355__*",
}
FORWARD_US = 4.3                 # native forward pass, 45 -> 64 -> 64 -> 48 (mlp_decision_cost.rs)
STATE_REALISTIC_US = 2.0         # estimated native state construction
STATE_UPPER_US = {"small": 16.8, "medium": 25.2, "large": 38.3}  # engine metrics call (rl_step_overhead.py)


def main() -> None:
    print("1. Search (engine) share of the wall-clock budget, summed over instances")
    for size, pattern in EXP8_RUNS.items():
        run = Path(glob.glob(str(REPO / pattern))[0])
        rows = pd.read_csv(run / "results.csv")
        for mode in ("kisialiou", "rl_improvement_mvp", "baseline"):
            r = rows[rows["mode"] == mode]
            wall, solver = r["wall_runtime_seconds"].sum(), r["solver_runtime_seconds"].sum()
            print(f"   {size:6s} {mode:20s} wall {wall / 60:7.1f} min  search {solver / 60:7.1f} min  share {solver / wall:5.0%}")

    print("\n2. RL decision overhead relative to RL's search per decision (per instance)")
    for size, pattern in EXP8_RUNS.items():
        run = Path(glob.glob(str(REPO / pattern))[0])
        r = pd.read_csv(run / "restart_results.csv", low_memory=False)
        r = r[r["mode"] == "rl_improvement_mvp"]
        g = r.groupby("dataset").agg(search=("solver_runtime_seconds", "sum"), steps=("iterations", "sum"))
        us_per_decision = (g["search"] / g["steps"] * 1e6).to_numpy()
        realistic = FORWARD_US + STATE_REALISTIC_US
        upper = FORWARD_US + STATE_UPPER_US[size]
        print(f"   {size:6s} search/decision median {np.median(us_per_decision):6.0f} us | "
              f"realistic {realistic:.1f} us: {np.median(realistic / us_per_decision):5.1%} median, "
              f"{np.max(realistic / us_per_decision):5.1%} worst | upper bound {upper:.0f} us: "
              f"{np.median(upper / us_per_decision):5.1%} median, {np.max(upper / us_per_decision):5.1%} worst")


if __name__ == "__main__":
    main()
