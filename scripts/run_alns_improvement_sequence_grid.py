#!/usr/bin/env python3
"""Run ALNS Kisialiou improvement-sequence grid with time fairness vs default Kisialiou."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.dataset_manager import GeneratedDatasetManager
from rl.rl_alns_environment import ALNSEnvironment


def log(message: str) -> None:
    print(message, flush=True)


def parse_sizes(raw: str) -> List[str]:
    sizes = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not sizes:
        raise ValueError("No sizes provided")
    return sizes


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for item in str(raw).split(","):
        value = item.strip()
        if not value:
            continue
        seeds.append(int(value))
    if not seeds:
        raise ValueError("No seeds provided")
    return seeds


def parse_lengths(raw: str) -> List[int]:
    lengths: List[int] = []
    for item in str(raw).split(","):
        value = item.strip()
        if not value:
            continue
        lengths.append(int(value))
    lengths = sorted(set(lengths))
    if not lengths:
        raise ValueError("No sequence lengths provided")
    if any(length <= 0 for length in lengths):
        raise ValueError("Sequence lengths must be positive")
    return lengths


def infer_size_label(dataset_path: str) -> str:
    parts = {part.lower() for part in Path(dataset_path).parts}
    if "small" in parts:
        return "small"
    if "medium" in parts:
        return "medium"
    if "large" in parts:
        return "large"
    return "unknown"


def prepare_datasets(sizes: Sequence[str], datasets_per_size: int) -> List[Tuple[str, str]]:
    manager = GeneratedDatasetManager()
    entries: List[Tuple[str, str]] = []
    for size in sizes:
        splits = manager.prepare_size(size)
        test_paths = splits.get("test", [])[: max(1, int(datasets_per_size))]
        if len(test_paths) < max(1, int(datasets_per_size)):
            raise RuntimeError(
                f"Expected at least {datasets_per_size} test datasets for size '{size}', found {len(test_paths)}"
            )
        entries.extend((size, path) for path in test_paths)
    return entries


def discover_improvement_operators(
    dataset_path: str,
    seed: int,
    max_iterations: int,
    action_module: str,
    state_module: str,
) -> List[str]:
    env = ALNSEnvironment(
        problem_instance=dataset_path,
        max_iterations=max(1, int(max_iterations)),
        seed=int(seed),
        problem_instance_paths=[dataset_path],
        problem_sampling_strategy="round_robin",
        enable_operator_logging=False,
        action_module=action_module,
        state_module=state_module,
        algorithm_mode="kisialiou",
    )
    operators = list(env.improvement_operators)
    env.close()
    if not operators:
        raise RuntimeError("No improvement operators discovered")
    return operators


def generate_sequences(
    operators: Sequence[str],
    lengths: Sequence[int],
    allow_repetition: bool,
) -> List[Tuple[str, ...]]:
    sequences: List[Tuple[str, ...]] = []
    for length in lengths:
        if allow_repetition:
            sequences.extend(itertools.product(operators, repeat=length))
        else:
            if length > len(operators):
                continue
            sequences.extend(itertools.permutations(operators, length))
    return sequences


def run_kisialiou_episode(
    *,
    dataset_path: str,
    seed: int,
    max_iterations: int,
    action_module: str,
    state_module: str,
    kisialiou_improvement_sequence: Optional[List[str]],
    target_runtime_seconds: Optional[float],
    enable_operator_logging: bool,
    log_dir: Path,
    mode_label: str,
) -> Dict[str, Any]:
    env = ALNSEnvironment(
        problem_instance=dataset_path,
        max_iterations=max(1, int(max_iterations)),
        seed=int(seed),
        problem_instance_paths=[dataset_path],
        problem_sampling_strategy="round_robin",
        enable_operator_logging=enable_operator_logging,
        operator_logging_mode="kisialiou",
        operator_logging_mode_label=mode_label,
        operator_logging_size=infer_size_label(dataset_path),
        operator_logging_dataset_name=f"{Path(dataset_path).name}__seed{seed}",
        operator_logging_stable_filename=True,
        operator_logging_dir=str(log_dir),
        action_module=action_module,
        state_module=state_module,
        algorithm_mode="kisialiou",
        force_baseline_improvement=False,
        baseline_improvement_idx=None,
        kisialiou_improvement_sequence=kisialiou_improvement_sequence,
    )

    env.reset(seed=int(seed), options={"problem_path": dataset_path})

    done = False
    truncated = False
    steps = 0
    elapsed_ms_total = 0.0
    stopped_on_budget = False

    runtime_limit_ms = None
    if target_runtime_seconds is not None:
        runtime_limit_ms = max(0.0, float(target_runtime_seconds)) * 1000.0

    while not done and not truncated and steps < int(max_iterations):
        if runtime_limit_ms is not None and steps > 0 and elapsed_ms_total >= runtime_limit_ms:
            stopped_on_budget = True
            break

        _, _, done, truncated, info = env.step(0)
        step_info = info.get("step_info", {}) if isinstance(info, dict) else {}
        step_ms = float(step_info.get("operator_elapsed_ms", step_info.get("elapsed_ms", 0.0)) or 0.0)
        elapsed_ms_total += step_ms
        steps += 1

    stats = env.get_episode_statistics()
    env.close()

    initial_cost = float(stats.get("initial_cost") or 0.0)
    best_cost = float(stats.get("best_cost") or initial_cost)
    final_cost = float(stats.get("final_cost") or best_cost)
    elapsed_seq = stats.get("elapsed_seconds") or []
    runtime_seconds = float(elapsed_seq[-1]) if elapsed_seq else elapsed_ms_total / 1000.0

    return {
        "initial_cost": initial_cost,
        "best_cost": best_cost,
        "final_cost": final_cost,
        "iterations": int(steps),
        "runtime_seconds": float(runtime_seconds),
        "stopped_on_budget": bool(stopped_on_budget),
    }


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_sequence: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row["sequence_key"])
        by_sequence.setdefault(key, []).append(row)

    sequence_summary: List[Dict[str, Any]] = []
    for key, items in by_sequence.items():
        costs = [float(item["best_cost"]) for item in items]
        improvements = [float(item["improvement_pct"]) for item in items]
        runtimes = [float(item["runtime_seconds"]) for item in items]
        gaps = [float(item["gap_vs_kisialiou_pct"]) for item in items]
        budget_stops = sum(1 for item in items if item.get("stopped_on_budget"))

        sequence_summary.append(
            {
                "sequence_key": key,
                "sequence": items[0]["sequence"],
                "sequence_length": int(items[0]["sequence_length"]),
                "n": len(items),
                "mean_best_cost": float(statistics.fmean(costs)) if costs else 0.0,
                "mean_improvement_pct": float(statistics.fmean(improvements)) if improvements else 0.0,
                "mean_runtime_seconds": float(statistics.fmean(runtimes)) if runtimes else 0.0,
                "mean_gap_vs_kisialiou_pct": float(statistics.fmean(gaps)) if gaps else 0.0,
                "wins_vs_kisialiou": int(sum(1 for g in gaps if g > 0.0)),
                "losses_vs_kisialiou": int(sum(1 for g in gaps if g < 0.0)),
                "ties_vs_kisialiou": int(sum(1 for g in gaps if abs(g) <= 1e-12)),
                "budget_limited_runs": int(budget_stops),
            }
        )

    sequence_summary.sort(key=lambda item: (item["mean_gap_vs_kisialiou_pct"], item["mean_improvement_pct"]), reverse=True)

    return {
        "num_rows": len(rows),
        "num_sequences": len(sequence_summary),
        "sequence_summary": sequence_summary,
    }


def write_rows_files(out_dir: Path, rows: Sequence[Dict[str, Any]], filename_prefix: str = "results") -> Tuple[Path, Path]:
    results_json = out_dir / f"{filename_prefix}.json"
    with results_json.open("w", encoding="utf-8") as fh:
        json.dump(list(rows), fh, indent=2)

    results_csv = out_dir / f"{filename_prefix}.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "size",
            "dataset",
            "dataset_path",
            "seed",
            "sequence",
            "sequence_key",
            "sequence_length",
            "iterations",
            "runtime_seconds",
            "runtime_budget_seconds",
            "stopped_on_budget",
            "initial_cost",
            "best_cost",
            "final_cost",
            "improvement_pct",
            "reference_kisialiou_best_cost",
            "gap_vs_kisialiou_pct",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_csv = dict(row)
            row_csv["sequence"] = ",".join(row["sequence"])
            writer.writerow(row_csv)

    return results_json, results_csv


def write_progress_checkpoint(
    *,
    out_dir: Path,
    rows: Sequence[Dict[str, Any]],
    run_idx: int,
    total_runs: int,
    start_ts: float,
) -> None:
    write_rows_files(out_dir=out_dir, rows=rows, filename_prefix="results_partial")
    elapsed = max(1e-9, time.perf_counter() - start_ts)
    rate = run_idx / elapsed
    eta_seconds = (total_runs - run_idx) / rate if rate > 1e-9 else None
    progress = {
        "completed_runs": int(run_idx),
        "total_runs": int(total_runs),
        "completion_ratio": float(run_idx / total_runs) if total_runs > 0 else 0.0,
        "elapsed_seconds": float(elapsed),
        "runs_per_second": float(rate),
        "eta_seconds": float(eta_seconds) if eta_seconds is not None else None,
        "updated_at": datetime.now().isoformat(),
    }
    (out_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ALNS over all improvement sequences with Kisialiou time budget matching"
    )
    parser.add_argument("--sizes", type=str, default="small,medium,large")
    parser.add_argument("--datasets-per-size", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--sequence-lengths", type=str, default="2,3")
    parser.add_argument(
        "--allow-repetition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow repeated operators within a sequence (default: true)",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--match-kisialiou-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each sequence with per-dataset-seed runtime budget from default Kisialiou",
    )
    parser.add_argument("--action-module", type=str, default="op_pair_v1")
    parser.add_argument("--state-module", type=str, default="features_v2")
    parser.add_argument("--run-tag", type=str, default="alns_seq_grid_l2_l3_i300_s5")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed runs (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write partial results/progress every N completed runs (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-operator-logging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable detailed operator logs for every run (large output)",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional cap for quick testing/debugging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_ts = time.perf_counter()

    sizes = parse_sizes(args.sizes)
    seeds = parse_seeds(args.seeds)
    lengths = parse_lengths(args.sequence_lengths)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"alns_sequence_grid_{timestamp}__{args.run_tag}"
    out_dir = REPO_ROOT / "output" / run_id
    log_dir = REPO_ROOT / "logs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset_entries = prepare_datasets(sizes=sizes, datasets_per_size=args.datasets_per_size)
    if not dataset_entries:
        raise RuntimeError("No datasets selected")

    sample_dataset = dataset_entries[0][1]
    operators = discover_improvement_operators(
        dataset_path=sample_dataset,
        seed=seeds[0],
        max_iterations=max(1, min(10, int(args.iterations))),
        action_module=args.action_module,
        state_module=args.state_module,
    )

    sequences = generate_sequences(
        operators=operators,
        lengths=lengths,
        allow_repetition=bool(args.allow_repetition),
    )
    if args.max_sequences is not None:
        sequences = sequences[: max(0, int(args.max_sequences))]

    if not sequences:
        raise RuntimeError("No sequences generated (check operators/lengths settings)")

    log(f"run_id={run_id}")
    log(f"sizes={sizes} datasets={len(dataset_entries)} seeds={len(seeds)}")
    log(f"operators={operators}")
    log(f"sequence_lengths={lengths} allow_repetition={args.allow_repetition} total_sequences={len(sequences)}")

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "sizes": sizes,
        "datasets": [
            {"size": size, "dataset_path": dataset_path, "dataset_name": Path(dataset_path).name}
            for size, dataset_path in dataset_entries
        ],
        "seeds": seeds,
        "iterations": int(args.iterations),
        "action_module": args.action_module,
        "state_module": args.state_module,
        "match_kisialiou_runtime": bool(args.match_kisialiou_runtime),
        "sequence_lengths": lengths,
        "allow_repetition": bool(args.allow_repetition),
        "operators": operators,
        "num_sequences": len(sequences),
    }
    (out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reference_runtime: Dict[Tuple[str, int], float] = {}
    reference_best: Dict[Tuple[str, int], float] = {}

    if args.match_kisialiou_runtime:
        total_refs = len(dataset_entries) * len(seeds)
        ref_idx = 0
        for _, dataset_path in dataset_entries:
            for seed in seeds:
                ref_idx += 1
                stats = run_kisialiou_episode(
                    dataset_path=dataset_path,
                    seed=seed,
                    max_iterations=args.iterations,
                    action_module=args.action_module,
                    state_module=args.state_module,
                    kisialiou_improvement_sequence=None,
                    target_runtime_seconds=None,
                    enable_operator_logging=bool(args.enable_operator_logging),
                    log_dir=log_dir,
                    mode_label="kisialiou_reference",
                )
                key = (dataset_path, int(seed))
                reference_runtime[key] = max(1e-6, float(stats["runtime_seconds"]))
                reference_best[key] = float(stats["best_cost"])
                log(
                    f"[ref {ref_idx}/{total_refs}] dataset={Path(dataset_path).name} seed={seed} "
                    f"runtime={reference_runtime[key]:.4f}s best={reference_best[key]:.6f}"
                )
    else:
        for _, dataset_path in dataset_entries:
            for seed in seeds:
                reference_runtime[(dataset_path, int(seed))] = 0.0
                reference_best[(dataset_path, int(seed))] = 0.0

    total_runs = len(sequences) * len(dataset_entries) * len(seeds)
    run_idx = 0
    rows: List[Dict[str, Any]] = []

    for sequence in sequences:
        sequence_list = list(sequence)
        sequence_key = "|".join(sequence_list)
        for size_label, dataset_path in dataset_entries:
            dataset_name = Path(dataset_path).name
            for seed in seeds:
                run_idx += 1
                key = (dataset_path, int(seed))
                runtime_budget = reference_runtime.get(key) if args.match_kisialiou_runtime else None

                stats = run_kisialiou_episode(
                    dataset_path=dataset_path,
                    seed=seed,
                    max_iterations=args.iterations,
                    action_module=args.action_module,
                    state_module=args.state_module,
                    kisialiou_improvement_sequence=sequence_list,
                    target_runtime_seconds=runtime_budget,
                    enable_operator_logging=bool(args.enable_operator_logging),
                    log_dir=log_dir,
                    mode_label=f"seq_{len(sequence_list)}",
                )

                initial_cost = float(stats["initial_cost"])
                best_cost = float(stats["best_cost"])
                improvement_pct = (
                    ((initial_cost - best_cost) / initial_cost) * 100.0 if initial_cost > 1e-12 else 0.0
                )
                ref_best = float(reference_best.get(key, 0.0))
                gap_vs_kisialiou_pct = (
                    ((ref_best - best_cost) / ref_best) * 100.0 if ref_best > 1e-12 else 0.0
                )

                rows.append(
                    {
                        "size": size_label,
                        "dataset": dataset_name,
                        "dataset_path": dataset_path,
                        "seed": int(seed),
                        "sequence": sequence_list,
                        "sequence_key": sequence_key,
                        "sequence_length": len(sequence_list),
                        "iterations": int(stats["iterations"]),
                        "runtime_seconds": float(stats["runtime_seconds"]),
                        "runtime_budget_seconds": float(runtime_budget) if runtime_budget is not None else None,
                        "stopped_on_budget": bool(stats["stopped_on_budget"]),
                        "initial_cost": initial_cost,
                        "best_cost": best_cost,
                        "final_cost": float(stats["final_cost"]),
                        "improvement_pct": float(improvement_pct),
                        "reference_kisialiou_best_cost": ref_best,
                        "gap_vs_kisialiou_pct": float(gap_vs_kisialiou_pct),
                    }
                )

                progress_every = max(1, int(args.progress_every))
                checkpoint_every = max(1, int(args.checkpoint_every))
                if run_idx % progress_every == 0 or run_idx == total_runs:
                    elapsed = max(1e-9, time.perf_counter() - start_ts)
                    rate = run_idx / elapsed
                    eta_seconds = (total_runs - run_idx) / rate if rate > 1e-9 else None
                    eta_display = f"{eta_seconds/60.0:.1f}m" if eta_seconds is not None else "n/a"
                    log(
                        f"[run {run_idx}/{total_runs}] seq={sequence_key} dataset={dataset_name} seed={seed} "
                        f"best={best_cost:.6f} gap_vs_ref={gap_vs_kisialiou_pct:.4f}% eta={eta_display}"
                    )
                if run_idx % checkpoint_every == 0:
                    write_progress_checkpoint(
                        out_dir=out_dir,
                        rows=rows,
                        run_idx=run_idx,
                        total_runs=total_runs,
                        start_ts=start_ts,
                    )

    results_json, results_csv = write_rows_files(out_dir=out_dir, rows=rows, filename_prefix="results")

    summary = aggregate(rows)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top_path = out_dir / "top_sequences.txt"
    lines = [
        f"run_id={run_id}",
        f"rows={summary['num_rows']}",
        f"sequences={summary['num_sequences']}",
        "top_by_mean_gap_vs_kisialiou_pct:",
    ]
    for idx, item in enumerate(summary["sequence_summary"][:20], start=1):
        lines.append(
            f"{idx:02d}. {item['sequence_key']} | gap={item['mean_gap_vs_kisialiou_pct']:.6f}% "
            f"impr={item['mean_improvement_pct']:.6f}% runtime={item['mean_runtime_seconds']:.6f}s"
        )
    top_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(f"done run_id={run_id}")
    log(f"results_json={results_json}")
    log(f"results_csv={results_csv}")
    log(f"summary={summary_path}")
    log(f"top_sequences={top_path}")


if __name__ == "__main__":
    main()
