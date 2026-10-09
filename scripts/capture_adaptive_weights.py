#!/usr/bin/env python3
"""Capture adaptive-weighted ALNS destroy/repair weight evolution (Experiment 6).

Runs N restarts x M fixed iterations (not time-budgeted -- this doesn't need
Exp0's time-budget convention) for baseline and/or kisialiou mode, recording
the destroy_weights/repair_weights vector returned every iteration by
rust_alns's execute_iteration (surfaced into ALNSEnvironment.step()'s info
dict as of this experiment). Outputs:

  - <out>__final_weights.json: mean/std of the LAST iteration's weight per
    operator, across restarts -- "what did it evolve to by the end".
  - <out>__timeseries.csv: mean weight per (iteration, operator) across
    restarts -- one row per iteration x operator, ready to plot.

Only operators that are ever nonzero across the whole run are included in
the timeseries CSV (the underlying registry has ~36 destroy / 8 repair
operators; Kisialiou's calibration keeps all but 3 of each permanently
locked at weight 0 -- see docs/EXPERIMENTS.md Experiment 6).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.dataset_manager import GeneratedDatasetManager
from rl.rl_alns_environment import ALNSEnvironment


def run_one_restart(
    *, dataset_path: str, size_bucket: str, algorithm_mode: str, max_iterations: int, seed: int
) -> Dict[str, List[List[float]]]:
    env = ALNSEnvironment(
        problem_instance=dataset_path,
        max_iterations=max_iterations,
        seed=seed,
        problem_instance_paths=[dataset_path],
        problem_sampling_strategy="round_robin",
        enable_operator_logging=False,
        action_module="adaptive_weighted_v1",
        algorithm_mode=algorithm_mode,
        size_bucket=size_bucket,
    )
    env.reset(seed=seed)
    destroy_names = list(env.destroy_operators)
    repair_names = list(env.repair_operators)

    destroy_history: List[List[float]] = []
    repair_history: List[List[float]] = []
    done = False
    truncated = False
    steps = 0
    while not done and not truncated and steps < max_iterations:
        obs, reward, done, truncated, info = env.step(0)
        dw = info.get("destroy_weights") or []
        rw = info.get("repair_weights") or []
        destroy_history.append(list(dw))
        repair_history.append(list(rw))
        steps += 1
    env.close()
    return {
        "destroy_names": destroy_names,
        "repair_names": repair_names,
        "destroy_history": destroy_history,
        "repair_history": repair_history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture adaptive-weighted ALNS weight evolution (Experiment 6)")
    parser.add_argument("--size", required=True, choices=["small", "medium", "large"])
    parser.add_argument("--modes", default="baseline,kisialiou", help="Comma-separated: baseline,kisialiou")
    parser.add_argument("--dataset-index", type=int, default=0, help="Index into the size's test split")
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=None, help="Default: 200 (small), 300 (medium/large)")
    parser.add_argument("--base-seed", type=int, default=4242)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "output" / "exp6_adaptive_weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    size_bucket = "small" if args.size == "small" else "large"
    max_iterations = args.max_iterations or (200 if args.size == "small" else 300)

    manager = GeneratedDatasetManager()
    splits = manager.prepare_size(args.size)
    test_paths = splits.get("test", [])
    if args.dataset_index >= len(test_paths):
        raise RuntimeError(f"dataset_index {args.dataset_index} out of range (only {len(test_paths)} test datasets)")
    dataset_path = str(test_paths[args.dataset_index])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        print(f"=== size={args.size} mode={mode} dataset={Path(dataset_path).name} "
              f"restarts={args.restarts} max_iterations={max_iterations} ===", flush=True)

        all_destroy: List[List[List[float]]] = []  # [restart][iteration][operator]
        all_repair: List[List[List[float]]] = []
        destroy_names: List[str] = []
        repair_names: List[str] = []

        for r in range(args.restarts):
            seed = args.base_seed + r
            result = run_one_restart(
                dataset_path=dataset_path,
                size_bucket=size_bucket,
                algorithm_mode=mode,
                max_iterations=max_iterations,
                seed=seed,
            )
            destroy_names = result["destroy_names"]
            repair_names = result["repair_names"]
            all_destroy.append(result["destroy_history"])
            all_repair.append(result["repair_history"])
            print(f"  restart {r + 1}/{args.restarts} done (seed={seed})", flush=True)

        def nonzero_indices(histories: List[List[List[float]]]) -> List[int]:
            n_ops = len(histories[0][0]) if histories and histories[0] else 0
            active = set()
            for restart_hist in histories:
                for iter_weights in restart_hist:
                    for i, w in enumerate(iter_weights):
                        if w != 0.0:
                            active.add(i)
            return sorted(active) if active else list(range(n_ops))

        destroy_active = nonzero_indices(all_destroy)
        repair_active = nonzero_indices(all_repair)

        # Final (last-iteration) weights per restart -> mean/std per operator
        final_weights: Dict[str, Dict[str, float]] = {}
        for label, histories, names, active in (
            ("destroy", all_destroy, destroy_names, destroy_active),
            ("repair", all_repair, repair_names, repair_active),
        ):
            for idx in active:
                name = names[idx]
                vals = [hist[-1][idx] for hist in histories if hist]
                if not vals:
                    continue
                final_weights[f"{label}:{name}"] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                    "n_restarts": len(vals),
                }

        final_path = args.out_dir / f"{args.size}__{mode}__final_weights.json"
        final_path.write_text(json.dumps(final_weights, indent=2), encoding="utf-8")
        print(f"  wrote {final_path}")

        # Per-iteration mean across restarts -> timeseries CSV
        ts_path = args.out_dir / f"{args.size}__{mode}__timeseries.csv"
        with ts_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["iteration", "operator_type", "operator_name", "mean_weight", "std_weight", "n_restarts"])
            for label, histories, names, active in (
                ("destroy", all_destroy, destroy_names, destroy_active),
                ("repair", all_repair, repair_names, repair_active),
            ):
                n_iters = min(len(h) for h in histories) if histories else 0
                for it in range(n_iters):
                    for idx in active:
                        vals = [hist[it][idx] for hist in histories]
                        mean_w = statistics.mean(vals)
                        std_w = statistics.stdev(vals) if len(vals) > 1 else 0.0
                        writer.writerow([it, label, names[idx], f"{mean_w:.6f}", f"{std_w:.6f}", len(vals)])
        print(f"  wrote {ts_path}")

    print("Done.")


if __name__ == "__main__":
    main()
