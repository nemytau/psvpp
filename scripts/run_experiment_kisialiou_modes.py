#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rl.dataset_manager import GeneratedDatasetManager


@dataclass(frozen=True)
class ModeSpec:
    mode_label: str
    algorithm_mode: str
    kisialiou_improvement_sequence: Optional[List[str]] = None


def safe_float(value) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_size_int_mapping(raw: str, expected_keys: List[str], *, name: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid {name} entry '{item}'. Expected format size:value")
        key, value_raw = item.split(":", 1)
        key = key.strip().lower()
        if key not in expected_keys:
            raise ValueError(f"Unknown size '{key}' in {name}")
        mapping[key] = int(value_raw.strip())

    missing = [size for size in expected_keys if size not in mapping]
    if missing:
        raise ValueError(f"Missing {name} values for sizes: {', '.join(missing)}")
    return mapping


def parse_size_float_mapping(raw: str, expected_keys: List[str], *, name: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid {name} entry '{item}'. Expected format size:value")
        key, value_raw = item.split(":", 1)
        key = key.strip().lower()
        if key not in expected_keys:
            raise ValueError(f"Unknown size '{key}' in {name}")
        mapping[key] = float(value_raw.strip())

    missing = [size for size in expected_keys if size not in mapping]
    if missing:
        raise ValueError(f"Missing {name} values for sizes: {', '.join(missing)}")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run six-mode ALNS experiment matrix with per-size time budgets and restart loops"
    )
    parser.add_argument(
        "--sizes",
        default="small,medium,large",
        help="Dataset sizes to run (comma-separated)",
    )
    parser.add_argument(
        "--datasets-per-size",
        type=int,
        default=5,
        help="Number of datasets per size",
    )
    parser.add_argument(
        "--iterations-by-size",
        default="small:200,medium:300,large:500",
        help="Iterations per run by size",
    )
    parser.add_argument(
        "--time-budget-by-size-seconds",
        default="small:60,medium:180,large:600",
        help="Time budget per dataset (per mode) in seconds",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=4242,
        help="Base seed for restart loops",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1000,
        help="Seed increment per restart",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional tag appended to run_id",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help=(
            "Comma-separated subset of mode labels to run (default: all 6). "
            "Example: baseline,kisialiou_original"
        ),
    )
    return parser.parse_args()


def get_mode_specs() -> List[ModeSpec]:
    return [
        ModeSpec(mode_label="baseline", algorithm_mode="baseline", kisialiou_improvement_sequence=None),
        ModeSpec(mode_label="kisialiou_original", algorithm_mode="kisialiou", kisialiou_improvement_sequence=None),
        ModeSpec(
            mode_label="kisialiou_fleet_only",
            algorithm_mode="kisialiou",
            kisialiou_improvement_sequence=["fleet_and_cost_reduction"],
        ),
        ModeSpec(
            mode_label="kisialiou_deep_relocation_only",
            algorithm_mode="kisialiou",
            kisialiou_improvement_sequence=["deep_relocation"],
        ),
        ModeSpec(
            mode_label="kisialiou_vnr_deep_relocation",
            algorithm_mode="kisialiou",
            kisialiou_improvement_sequence=["voyage_number_reduction", "deep_relocation"],
        ),
        ModeSpec(
            mode_label="kisialiou_vnr_fleet_vnr",
            algorithm_mode="kisialiou",
            kisialiou_improvement_sequence=[
                "voyage_number_reduction",
                "fleet_and_cost_reduction",
                "voyage_number_reduction",
            ],
        ),
    ]


def run_mode_size_with_time_budget(
    *,
    mode: ModeSpec,
    size: str,
    dataset_path: Path,
    dataset_index: int,
    time_budget: float,
    iterations: int,
    repo: Path,
    out_dir: Path,
    log_dir: Path,
    base_seed: int,
    seed_step: int,
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    all_results: List[Dict] = []
    restart_records: List[Dict[str, object]] = []

    restart_count = 0
    start_time = time.time()
    recent_runtimes: List[float] = []  # Track recent restart times for adaptive scheduling
    safety_margin = 5.0  # Buffer before giving up on a restart

    combined_results_path = out_dir / f"{mode.mode_label}__{size}__{dataset_path.stem}__results.json"
    combined_benchmark_log_path = out_dir / f"{mode.mode_label}__{size}__{dataset_path.stem}__benchmark.log"
    combined_console_path = out_dir / f"{mode.mode_label}__{size}__{dataset_path.stem}__console.log"

    def append_file(source: Path, destination: Path) -> None:
        if not source.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("r", encoding="utf-8", errors="ignore") as src:
            content = src.read()
        if not content:
            return
        with destination.open("a", encoding="utf-8") as dst:
            dst.write(content)
            if not content.endswith("\n"):
                dst.write("\n")

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            print(
                f"Time budget exhausted for {mode.mode_label} size={size} dataset={dataset_path.name} after {restart_count} restarts ({elapsed:.1f}s)",
                flush=True,
            )
            break

        seed = base_seed + (dataset_index * 100) + restart_count * seed_step
        with tempfile.NamedTemporaryFile(suffix="_results.json", delete=False) as tmp_results:
            results_path = Path(tmp_results.name)
        with tempfile.NamedTemporaryFile(suffix="_benchmark.log", delete=False) as tmp_log:
            benchmark_log_path = Path(tmp_log.name)

        cmd = [
            sys.executable,
            "scripts/alns_mode_benchmark.py",
            "--sizes",
            size,
            "--mode",
            mode.algorithm_mode,
            "--mode-label",
            mode.mode_label,
            "--iterations",
            str(iterations),
            "--base-seed",
            str(seed),
            "--results-path",
            str(results_path),
            "--log-path",
            str(benchmark_log_path),
            "--log-dir",
            str(log_dir),
            "--dataset-paths",
            str(dataset_path),
            "--size-label",
            size,
        ]

        if mode.kisialiou_improvement_sequence is not None:
            cmd.extend(
                [
                    "--kisialiou-improvement-sequence",
                    ",".join(mode.kisialiou_improvement_sequence),
                ]
            )

        remaining = time_budget - elapsed
        
        # Adaptive scheduling: check if we have enough time for another restart
        if recent_runtimes:
            estimated_runtime = statistics.mean(recent_runtimes)
            required_time = estimated_runtime + safety_margin
            if required_time > remaining:
                print(
                    f"Stopping early: estimated next restart would take ~{estimated_runtime:.1f}s + {safety_margin:.1f}s buffer = {required_time:.1f}s, but only {remaining:.1f}s remaining for {mode.mode_label} size={size} dataset={dataset_path.name}",
                    flush=True,
                )
                break
        
        print(
            f"\n=== {mode.mode_label} | size={size} | dataset={dataset_path.name} | restart={restart_count + 1} | remaining={remaining:.1f}s | seed={seed} ===",
            flush=True,
        )

        restart_start = time.time()
        timed_out = False
        with combined_console_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n=== {mode.mode_label} | size={size} | dataset={dataset_path.name} | restart={restart_count + 1} | remaining={remaining:.1f}s | seed={seed} ===\n"
            )
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=repo,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                proc = None
        restart_elapsed = time.time() - restart_start

        append_file(benchmark_log_path, combined_benchmark_log_path)
        benchmark_log_path.unlink(missing_ok=True)

        if timed_out:
            results_path.unlink(missing_ok=True)
            restart_records.append(
                {
                    "mode": mode.mode_label,
                    "size": size,
                    "dataset": dataset_path.name,
                    "dataset_path": str(dataset_path),
                    "restart_id": restart_count,
                    "seed": seed,
                    "elapsed_seconds": restart_elapsed,
                    "results_path": str(combined_results_path),
                    "benchmark_log_path": str(combined_benchmark_log_path),
                    "console_log_path": str(combined_console_path),
                    "status": "timeout",
                }
            )
            break

        if proc is not None and proc.returncode != 0:
            raise RuntimeError(
                f"{mode.mode_label} size={size} dataset={dataset_path.name} restart={restart_count} failed with exit code {proc.returncode}; see {combined_console_path}"
            )

        if results_path.exists():
            restart_results = json.loads(results_path.read_text(encoding="utf-8"))
            for row in restart_results:
                row["restart_id"] = restart_count
                row["mode_label"] = mode.mode_label
                row["algorithm_mode"] = mode.algorithm_mode
                row["size"] = size
                row["dataset"] = dataset_path.name
                row["dataset_path"] = str(dataset_path)
                row["iterations_config"] = iterations
                row["time_budget_seconds"] = time_budget
                row["kisialiou_improvement_sequence"] = mode.kisialiou_improvement_sequence
                all_results.append(row)
            results_path.unlink(missing_ok=True)

        if all_results:
            combined_results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

        restart_records.append(
            {
                "mode": mode.mode_label,
                "size": size,
                "dataset": dataset_path.name,
                "dataset_path": str(dataset_path),
                "restart_id": restart_count,
                "seed": seed,
                "elapsed_seconds": restart_elapsed,
                "results_path": str(combined_results_path),
                "benchmark_log_path": str(combined_benchmark_log_path),
                "console_log_path": str(combined_console_path),
                "status": "ok",
            }
        )

        # Track successful restart time for adaptive scheduling
        recent_runtimes.append(restart_elapsed)
        # Keep only last 5 runtimes for better estimate
        if len(recent_runtimes) > 5:
            recent_runtimes.pop(0)

        restart_count += 1

    return all_results, restart_records


def build_analysis(all_results: List[Dict], log_dir: Path) -> Dict:
    by_mode: Dict[str, List[Dict]] = {}
    for row in all_results:
        mode = str(row.get("mode"))
        by_mode.setdefault(mode, []).append(row)

    improvement_per_time: Dict[str, Dict] = {}
    for mode, rows in by_mode.items():
        total_improvement_pct = 0.0
        total_runtime = 0.0
        num_valid_samples = 0
        for row in rows:
            improvement_pct = safe_float(row.get("improvement_pct"))
            runtime = safe_float(row.get("runtime_seconds"))
            if runtime == runtime and runtime > 1e-12 and improvement_pct == improvement_pct:
                total_improvement_pct += improvement_pct
                total_runtime += runtime
                num_valid_samples += 1

        improvement_rate = (
            total_improvement_pct / total_runtime
            if num_valid_samples > 0 and total_runtime > 1e-12
            else float("nan")
        )
        improvement_per_time[mode] = {
            "total_improvement_pct": total_improvement_pct,
            "total_runtime_seconds": total_runtime,
            "improvement_pct_per_second": improvement_rate,
            "num_samples": num_valid_samples,
        }

    per_operator_values: Dict[Tuple[str, str], List[float]] = {}
    for csv_path in sorted(log_dir.glob("operator_usage_*.csv")):
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                operator_name = str(row.get("operator_name") or "")
                operator_type = str(row.get("operator_type") or "")
                if not operator_name:
                    continue
                value = safe_float(row.get("best_cost_delta"))
                if value != value:
                    continue
                per_operator_values.setdefault((operator_type, operator_name), []).append(value)

    avg_best_cost_delta_per_operator = []
    for (operator_type, operator_name), values in sorted(per_operator_values.items()):
        avg_best_cost_delta_per_operator.append(
            {
                "operator_type": operator_type,
                "operator_name": operator_name,
                "mean_best_cost_delta": statistics.mean(values),
                "num_rows": len(values),
            }
        )

    grouped: Dict[Tuple[int, str, str], Dict[str, float]] = {}
    for row in all_results:
        restart_id = int(row.get("restart_id", -1))
        dataset = str(row.get("dataset"))
        size = str(row.get("size"))
        mode = str(row.get("mode"))
        best_cost = safe_float(row.get("best_cost"))
        grouped.setdefault((restart_id, size, dataset), {})[mode] = best_cost

    baseline_vs_mode: Dict[str, Dict[str, float]] = {}
    mode_names = sorted(by_mode.keys())
    for mode in mode_names:
        if mode == "baseline":
            continue
        diffs_pct: List[float] = []
        for _, modes in grouped.items():
            if "baseline" in modes and mode in modes:
                b = modes["baseline"]
                m = modes[mode]
                if b == b and m == m and b > 1e-12:
                    diffs_pct.append(((b - m) / b) * 100.0)
        baseline_vs_mode[mode] = {
            "mean_diff_pct": statistics.mean(diffs_pct) if diffs_pct else float("nan"),
            "num_pairs": len(diffs_pct),
        }

    return {
        "average_improvement_by_method_per_time": improvement_per_time,
        "average_best_cost_delta_per_operator": avg_best_cost_delta_per_operator,
        "baseline_minus_mode_best_cost_pct": baseline_vs_mode,
    }


def render_analysis_text(analysis: Dict) -> str:
    lines: List[str] = []
    lines.append("Benchmark analysis")
    lines.append("==================")

    lines.append("\n1) Average improvement by method per time")
    for mode, stats in sorted(analysis["average_improvement_by_method_per_time"].items()):
        lines.append(f"- {mode}:")
        lines.append(
            f"    improvement_pct_per_second: {stats['improvement_pct_per_second']:.6f}"
        )
        lines.append(
            f"    (total_improvement_pct={stats['total_improvement_pct']:.2f}%, total_runtime={stats['total_runtime_seconds']:.1f}s, samples={stats['num_samples']})"
        )

    lines.append("\n2) Average best_cost_delta per operator")
    for item in analysis["average_best_cost_delta_per_operator"]:
        lines.append(
            f"- {item['operator_type']}/{item['operator_name']}: "
            f"mean_best_cost_delta={item['mean_best_cost_delta']:.6f} (rows={item['num_rows']})"
        )

    lines.append("\n3) Baseline minus mode best-cost difference (%)")
    for mode, stats in sorted(analysis["baseline_minus_mode_best_cost_pct"].items()):
        lines.append(
            f"- {mode}: mean((baseline_best - mode_best)/baseline_best*100)={stats['mean_diff_pct']:.6f}% "
            f"(pairs={stats['num_pairs']})"
        )

    return "\n".join(lines) + "\n"


def write_experiment_registry(
    *,
    repo: Path,
    run_id: str,
    run_dir: Path,
    config: Dict,
    combined_results_path: Path,
    analysis_json_path: Path,
    analysis_txt_path: Path,
    restart_manifest_path: Path,
) -> None:
    benchmark_root = repo / "runs" / "benchmarks"
    experiment_dir = benchmark_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "references": {
            "output_run_dir": str(run_dir),
            "combined_results": str(combined_results_path),
            "analysis_json": str(analysis_json_path),
            "analysis_txt": str(analysis_txt_path),
            "restart_manifest": str(restart_manifest_path),
            "operator_logs_dir": str(run_dir.parent.parent / "logs" / run_id),
        },
    }

    (experiment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    description_lines = [
        f"# Experiment {run_id}",
        "",
        "## Description",
        "Comparison of six ALNS modes (baseline and Kisialiou variants with custom improvement sequences)",
        "under per-size time budgets and restart loops.",
        "",
        "## Configuration",
        f"- sizes: {', '.join(config['sizes'])}",
        f"- datasets_per_size: {config['datasets_per_size']}",
        f"- iterations_by_size: {config['iterations_by_size']}",
        f"- time_budget_per_dataset_seconds: {config['time_budget_per_dataset_seconds']}",
        "",
        "## References",
        f"- output run dir: {run_dir}",
        f"- combined results: {combined_results_path}",
        f"- analysis json: {analysis_json_path}",
        f"- analysis text: {analysis_txt_path}",
        f"- restart manifest: {restart_manifest_path}",
        f"- logs dir: {run_dir.parent.parent / 'logs' / run_id}",
    ]
    (experiment_dir / "README.md").write_text("\n".join(description_lines) + "\n", encoding="utf-8")


def run() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]

    sizes = [item.strip().lower() for item in str(args.sizes).split(",") if item.strip()]
    if not sizes:
        raise ValueError("At least one dataset size must be provided")

    iterations_by_size = parse_size_int_mapping(
        args.iterations_by_size,
        sizes,
        name="iterations-by-size",
    )
    time_budget_by_size_seconds = parse_size_float_mapping(
        args.time_budget_by_size_seconds,
        sizes,
        name="time-budget-by-size-seconds",
    )

    mode_specs = get_mode_specs()
    if args.modes:
        requested = [m.strip() for m in args.modes.split(",") if m.strip()]
        available = {spec.mode_label: spec for spec in mode_specs}
        unknown = [m for m in requested if m not in available]
        if unknown:
            raise ValueError(
                f"Unknown mode label(s): {', '.join(unknown)}. "
                f"Available: {', '.join(available)}"
            )
        mode_specs = [available[m] for m in requested]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"__{args.run_tag}" if args.run_tag else ""
    run_id = f"benchmark_modes_matrix_{ts}{suffix}"

    out_dir = repo / "output" / run_id
    log_dir = repo / "logs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    restart_records: List[Dict[str, object]] = []

    for size in sizes:
        iter_count = int(iterations_by_size[size])
        budget_seconds = float(time_budget_by_size_seconds[size])

        manager = GeneratedDatasetManager()
        splits = manager.prepare_size(size)
        test_paths = splits.get("test", [])[: args.datasets_per_size]
        if len(test_paths) < args.datasets_per_size:
            raise RuntimeError(
                f"Expected at least {args.datasets_per_size} test dataset(s) for size '{size}', found {len(test_paths)}"
            )

        for dataset_index, dataset_path in enumerate(test_paths):
            dataset_path = Path(dataset_path)
            for mode in mode_specs:
                print("\n" + "=" * 70)
                print(
                    f"Starting mode={mode.mode_label} | size={size} | dataset={dataset_path.name} | iterations={iter_count} | budget={budget_seconds:.1f}s"
                )
                print("=" * 70, flush=True)

                mode_results, mode_restarts = run_mode_size_with_time_budget(
                    mode=mode,
                    size=size,
                    dataset_path=dataset_path,
                    dataset_index=dataset_index,
                    time_budget=budget_seconds,
                    iterations=iter_count,
                    repo=repo,
                    out_dir=out_dir,
                    log_dir=log_dir,
                    base_seed=args.base_seed,
                    seed_step=args.seed_step,
                )
                all_results.extend(mode_results)
                restart_records.extend(mode_restarts)

    combined_path = out_dir / "combined_results.json"
    combined_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    restart_manifest_path = out_dir / "restart_manifest.json"
    restart_manifest_path.write_text(json.dumps(restart_records, indent=2), encoding="utf-8")

    analysis = build_analysis(all_results, log_dir)
    analysis_json_path = out_dir / "analysis.json"
    analysis_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    analysis_txt_path = out_dir / "analysis.txt"
    analysis_txt_path.write_text(render_analysis_text(analysis), encoding="utf-8")

    mode_manifest = [asdict(mode) for mode in mode_specs]
    config = {
        "sizes": sizes,
        "datasets_per_size": args.datasets_per_size,
        "iterations_by_size": iterations_by_size,
        "time_budget_per_dataset_seconds": time_budget_by_size_seconds,
        "base_seed": args.base_seed,
        "seed_step": args.seed_step,
        "modes": mode_manifest,
    }

    out_manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "output_dir": str(out_dir),
        "log_dir": str(log_dir),
        "combined_results": str(combined_path),
        "analysis_json": str(analysis_json_path),
        "analysis_txt": str(analysis_txt_path),
        "restart_manifest": str(restart_manifest_path),
        "num_result_rows": len(all_results),
        "num_restart_rows": len(restart_records),
    }
    (out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")

    write_experiment_registry(
        repo=repo,
        run_id=run_id,
        run_dir=out_dir,
        config=config,
        combined_results_path=combined_path,
        analysis_json_path=analysis_json_path,
        analysis_txt_path=analysis_txt_path,
        restart_manifest_path=restart_manifest_path,
    )

    print("\n=== Benchmark complete ===")
    print(f"Run ID: {run_id}")
    print(f"Combined results: {combined_path}")
    print(f"Analysis: {analysis_txt_path}")
    print(f"Restart manifest: {restart_manifest_path}")
    print(f"Operator logs dir: {log_dir}")
    print(f"Experiment registry: {repo / 'runs' / 'benchmarks' / run_id}")


if __name__ == "__main__":
    run()
