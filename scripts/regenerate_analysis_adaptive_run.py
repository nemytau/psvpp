import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def safe_float(value) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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
        lines.append(f"    improvement_pct_per_second: {stats['improvement_pct_per_second']:.6f}")
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


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    run_dir = repo / "output" / "benchmark_modes_matrix_20260224_174919__adaptive_full_run_improved_budgets"
    log_dir = repo / "logs" / "benchmark_modes_matrix_20260224_174919__adaptive_full_run_improved_budgets"

    all_results: List[Dict] = []
    for path in sorted(run_dir.glob("*__results.json")):
        if path.name == "combined_results.json":
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(rows, list):
            all_results.extend(rows)

    combined_path = run_dir / "combined_results.json"
    analysis_json_path = run_dir / "analysis.json"
    analysis_txt_path = run_dir / "analysis.txt"

    combined_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    analysis = build_analysis(all_results, log_dir)
    analysis_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    analysis_txt_path.write_text(render_analysis_text(analysis), encoding="utf-8")

    print(f"Rebuilt combined results rows: {len(all_results)}")
    print(f"Wrote: {combined_path}")
    print(f"Wrote: {analysis_json_path}")
    print(f"Wrote: {analysis_txt_path}")


if __name__ == "__main__":
    main()
