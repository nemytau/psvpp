#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def safe_float(value) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-budget constrained ALNS benchmark")
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="Total time budget for all restarts (seconds). If None, use fixed restart count.",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=10,
        help="Number of restarts (used if --time-budget-seconds not set)",
    )
    parser.add_argument(
        "--sizes",
        default="small",
        help="Dataset sizes to run (comma-separated or single)",
    )
    parser.add_argument(
        "--datasets-per-size",
        type=int,
        default=5,
        help="Number of datasets per size",
    )
    parser.add_argument(
        "--iterations-per-run",
        type=int,
        default=200,
        help="Iterations per benchmark run",
    )
    return parser.parse_args()


def run_mode_with_time_budget(
    mode: str,
    time_budget: float,
    repo: Path,
    out_dir: Path,
    log_dir: Path,
    args: argparse.Namespace,
) -> List[Dict]:
    """Run a single mode (baseline or kisialiou) for a fixed time budget."""
    all_results: List[Dict] = []
    base_seed_start = 4242
    seed_step = 1000
    restart_count = 0
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            print(f"Time budget exhausted for {mode} after {restart_count} restarts ({elapsed:.1f}s)", flush=True)
            break
        
        seed = base_seed_start + restart_count * seed_step
        results_path = out_dir / f"{mode}_restart_{restart_count:02d}_results.json"
        benchmark_log_path = out_dir / f"{mode}_restart_{restart_count:02d}_benchmark.log"
        console_path = out_dir / f"{mode}_restart_{restart_count:02d}_console.log"

        cmd = [
            sys.executable,
            "scripts/alns_mode_benchmark.py",
            "--sizes",
            args.sizes,
            "--mode",
            mode,
            "--iterations",
            str(args.iterations_per_run),
            "--datasets-per-size",
            str(args.datasets_per_size),
            "--base-seed",
            str(seed),
            "--results-path",
            str(results_path),
            "--log-path",
            str(benchmark_log_path),
            "--log-dir",
            str(log_dir),
        ]

        remaining = time_budget - elapsed
        print(f"\n=== {mode.upper()} Restart {restart_count + 1} (time remaining: {remaining:.1f}s) | base_seed={seed} ===", flush=True)
        restart_start = time.time()
        
        with console_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=repo, stdout=handle, stderr=subprocess.STDOUT)
        
        restart_elapsed = time.time() - restart_start
        if proc.returncode != 0:
            raise RuntimeError(
                f"{mode} restart {restart_count} failed with exit code {proc.returncode}; see {console_path}"
            )

        restart_results = json.loads(results_path.read_text(encoding="utf-8"))
        for row in restart_results:
            row["restart_id"] = restart_count
            all_results.append(row)
        
        print(f"{mode.upper()} restart {restart_count} completed in {restart_elapsed:.1f}s", flush=True)
        restart_count += 1
    
    return all_results


def run() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    
    time_budget_prefix = ""
    if args.time_budget_seconds is not None:
        time_budget_prefix = f"tb{int(args.time_budget_seconds)}s"
    else:
        time_budget_prefix = f"r{args.num_restarts}"
    
    run_id = f"benchmark_small5_{time_budget_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = repo / "output" / run_id
    log_dir = repo / "logs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    if args.time_budget_seconds is not None:
        # Time-budget mode: split budget equally between baseline and kisialiou
        # Each method gets time_budget_seconds to run independently
        per_mode_budget = args.time_budget_seconds
        
        print(f"\n=== Time-budget mode: {args.time_budget_seconds}s per method ===", flush=True)
        
        for mode in ["baseline", "kisialiou"]:
            print(f"\n{'='*60}")
            print(f"Starting {mode.upper()} with {per_mode_budget}s budget")
            print(f"{'='*60}", flush=True)
            
            mode_results = run_mode_with_time_budget(
                mode=mode,
                time_budget=per_mode_budget,
                repo=repo,
                out_dir=out_dir,
                log_dir=log_dir,
                args=args,
            )
            all_results.extend(mode_results)
    else:
        # Fixed restart mode: loop for fixed number of restarts per mode
        print(f"\n=== Fixed restart mode: {args.num_restarts} restarts per mode ===", flush=True)
        
        for mode in ["baseline", "kisialiou"]:
            print(f"\n{'='*60}")
            print(f"Starting {mode.upper()} with {args.num_restarts} restarts")
            print(f"{'='*60}", flush=True)
            
            base_seed_start = 4242
            seed_step = 1000
            
            for restart in range(args.num_restarts):
                seed = base_seed_start + restart * seed_step
                results_path = out_dir / f"{mode}_restart_{restart:02d}_results.json"
                benchmark_log_path = out_dir / f"{mode}_restart_{restart:02d}_benchmark.log"
                console_path = out_dir / f"{mode}_restart_{restart:02d}_console.log"

                cmd = [
                    sys.executable,
                    "scripts/alns_mode_benchmark.py",
                    "--sizes",
                    args.sizes,
                    "--mode",
                    mode,
                    "--iterations",
                    str(args.iterations_per_run),
                    "--datasets-per-size",
                    str(args.datasets_per_size),
                    "--base-seed",
                    str(seed),
                    "--results-path",
                    str(results_path),
                    "--log-path",
                    str(benchmark_log_path),
                    "--log-dir",
                    str(log_dir),
                ]

                print(f"\n=== {mode.upper()} Restart {restart + 1}/{args.num_restarts} | base_seed={seed} ===", flush=True)
                with console_path.open("w", encoding="utf-8") as handle:
                    proc = subprocess.run(cmd, cwd=repo, stdout=handle, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{mode} restart {restart} failed with exit code {proc.returncode}; see {console_path}"
                    )

                restart_results = json.loads(results_path.read_text(encoding="utf-8"))
                for i, row in enumerate(restart_results):
                    row["restart_id"] = restart
                    all_results.append(row)

    combined_path = out_dir / "combined_results.json"
    combined_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    analysis = build_analysis(all_results, log_dir)
    analysis_json_path = out_dir / "analysis.json"
    analysis_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    analysis_txt_path = out_dir / "analysis.txt"
    analysis_txt_path.write_text(render_analysis_text(analysis), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "log_dir": str(log_dir),
        "combined_results": str(combined_path),
        "analysis_json": str(analysis_json_path),
        "analysis_txt": str(analysis_txt_path),
        "num_result_rows": len(all_results),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Benchmark complete ===")
    print(f"Run ID: {run_id}")
    print(f"Combined results: {combined_path}")
    print(f"Analysis: {analysis_txt_path}")
    print(f"Operator logs dir: {log_dir}")


def build_analysis(all_results: List[Dict], log_dir: Path) -> Dict:
    by_mode: Dict[str, List[Dict]] = {}
    for row in all_results:
        mode = str(row.get("mode"))
        by_mode.setdefault(mode, []).append(row)

    # 1) average improvement by method per time (percentage points per second)
    # Properly computed as: total_improvement_pct / total_runtime across all valid samples
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
        
        if num_valid_samples > 0 and total_runtime > 1e-12:
            improvement_rate = total_improvement_pct / total_runtime
        else:
            improvement_rate = float("nan")
        
        improvement_per_time[mode] = {
            "total_improvement_pct": total_improvement_pct,
            "total_runtime_seconds": total_runtime,
            "improvement_pct_per_second": improvement_rate,
            "num_samples": num_valid_samples,
        }

    # 2) average best_cost_delta per operator (from all CSV logs)
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

    # 3) average diff between best found for both approaches (percentage)
    grouped: Dict[Tuple[int, str], Dict[str, float]] = {}
    for row in all_results:
        restart_id = int(row.get("restart_id", -1))
        dataset = str(row.get("dataset"))
        mode = str(row.get("mode"))
        best_cost = safe_float(row.get("best_cost"))
        grouped.setdefault((restart_id, dataset), {})[mode] = best_cost

    diffs_pct = []
    for (_, _), modes in grouped.items():
        if "baseline" in modes and "kisialiou" in modes:
            b = modes["baseline"]
            k = modes["kisialiou"]
            if b == b and k == k and b > 1e-12:
                diffs_pct.append(((b - k) / b) * 100.0)

    avg_diff_pct = statistics.mean(diffs_pct) if diffs_pct else float("nan")

    return {
        "average_improvement_by_method_per_time": improvement_per_time,
        "average_best_cost_delta_per_operator": avg_best_cost_delta_per_operator,
        "average_diff_best_found_baseline_minus_kisialiou_pct": {
            "mean_diff_pct": avg_diff_pct,
            "num_pairs": len(diffs_pct),
        },
    }


def render_analysis_text(analysis: Dict) -> str:
    lines: List[str] = []
    lines.append("Benchmark analysis")
    lines.append("==================")

    lines.append("\n1) Average improvement by method per time")
    for mode, stats in sorted(analysis["average_improvement_by_method_per_time"].items()):
        lines.append(
            f"- {mode}:"
        )
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

    diff = analysis["average_diff_best_found_baseline_minus_kisialiou_pct"]
    lines.append("\n3) Average diff between best found for both approaches")
    lines.append(
        f"- mean((baseline_best - kisialiou_best)/baseline_best*100)={diff['mean_diff_pct']:.6f}% "
        f"(pairs={diff['num_pairs']})"
    )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run()
