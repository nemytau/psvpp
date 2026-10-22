"""Within-restart operator switching of the NORM 2026 (Experiment 9) policies.

For each evaluated policy, reads the per-iteration operator-usage logs, rebuilds every
restart's action sequence (destroy + repair + improvement per iteration; improvement-only or
backbone steps for the chain action space) and classifies the restart:

  constant     one action for the whole restart
  switch once  action A for the first k iterations, then action B until the end
  multi        more than one switch

For "switch once" restarts it reports the most common (A -> B) patterns and the switch point,
and for every class the restart's improvement (initial cost -> restart best). This is the
evidence for "probe with a fleet-reduction operator first, then local search" patterns.

    python scripts/analyze_norm2026_switching.py [--runs tag1,tag2,...]
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    "reward__alns_score_time_v1__medium__s42",
    "action__improvement_chain_v2__medium__s42",
    "reward__best_gain_v1__medium__s42",
    "state__search__small__s42",
    "state__all__small__s42",
    "state__all__small__s43",
    "action__improvement_chain_v2__small__s42",
    "action__improvement_chain_v2__small__s43",
    "reward__best_gain_v1__small__s42",
    "reward__best_gain_v1__small__s43",
    "reward__alns_score_v1__small__s42",
    "reward__alns_score_time_v1__small__s42",
    "action__op_triple_many_v1__small__s43",
    "reward__new_best_v1__small__s43",
]


def short(name: str) -> str:
    name = re.sub(r"_kisialiou_(small|large)_v1|_insertion_v1|_removal_in_voyages", "", name)
    return (name.replace("random_visit_removal", "random").replace("worst_visit_removal", "worst")
            .replace("shaw_removal", "shaw").replace("first_fit", "first-fit")
            .replace("deep_greedy_rank_1", "greedy").replace("voyage_number_reduction", "VNR")
            .replace("fleet_and_cost_reduction_kisialiou_v1", "FCR").replace("fleet_and_cost_reduction", "FCR")
            .replace("deep_relocation", "reloc").replace("deep_swap", "swap"))


def restart_sequences(csv_path: str, chain: bool):
    cols = ["restart_id", "iteration", "operator_type", "operator_name", "cost_best", "cost_current"]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    initial = df[df["operator_type"] == "initial"].groupby("restart_id")["cost_current"].first()
    steps = df[df["operator_type"] != "initial"]
    if chain:
        # Chain action space: on backbone steps the *engine* picks destroy/repair adaptively, so
        # they are labelled "D/R" -- only the policy's own choice (backbone vs. which
        # improvement operator) counts as an action.
        steps = steps[steps["operator_type"].isin(["destroy", "improvement"])]
        names = steps["operator_type"].where(steps["operator_type"] == "improvement", "D/R")
        names = names.where(names == "D/R", steps["operator_name"].map(short))
        steps = steps.assign(name=names)
    else:
        steps = steps.assign(name=steps["operator_name"].map(short))
    labels = steps.groupby(["restart_id", "iteration"], sort=True)["name"].agg(" + ".join)
    best = steps.groupby("restart_id")["cost_best"].min()
    for restart_id, seq in labels.groupby(level=0):
        actions = seq.tolist()
        runs = []  # run-length encoding
        for a in actions:
            if runs and runs[-1][0] == a:
                runs[-1][1] += 1
            else:
                runs.append([a, 1])
        start = float(initial.get(restart_id, np.nan))
        gain = (start - float(best.get(restart_id, start))) / start * 100 if start == start and start > 0 else np.nan
        yield runs, gain


def analyse(tag: str) -> None:
    files = sorted(glob.glob(str(REPO / "logs" / f"norm26__{tag}" / "operator_usage_*.csv")))
    classes = Counter()
    gains = {"constant": [], "switch once": [], "multi": []}
    patterns = Counter()
    switch_points = {}
    for f in files:
        for runs, gain in restart_sequences(f, chain="improvement_chain" in tag):
            cls = "constant" if len(runs) == 1 else ("switch once" if len(runs) == 2 else "multi")
            classes[cls] += 1
            gains[cls].append(gain)
            if cls == "switch once":
                key = f"{runs[0][0]}  ->  {runs[1][0]}"
                patterns[key] += 1
                switch_points.setdefault(key, []).append(runs[0][1])
    total = sum(classes.values())
    print(f"\n=== {tag}: {total} restarts over {len(files)} instances")
    for cls in ("constant", "switch once", "multi"):
        g = [x for x in gains[cls] if x == x]
        if classes[cls]:
            print(f"   {cls:12s} {classes[cls] / total:6.1%}   restart improvement: median {np.median(g):5.1f}%")
    for key, n in patterns.most_common(4):
        sp = switch_points[key]
        print(f"     {n / total:6.1%}  {key}   (switch after iteration {int(np.median(sp))} median, "
              f"{min(sp)}-{max(sp)} range)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    args = ap.parse_args()
    for tag in [t for t in args.runs.split(",") if t]:
        analyse(tag)


if __name__ == "__main__":
    main()
