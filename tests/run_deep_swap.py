#!/usr/bin/env python3
"""Execute destroy+repair followed by deep swap across multiple datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

try:
    from rust_alns_py import RustALNSInterface  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - diagnostic path
    print("rust_alns_py module not found: {}".format(exc))
    print("Build the extension first: cd rust_alns && maturin develop --release")
    sys.exit(1)

# Reuse plotting helper lazily to keep dependency optional during CLI usage.
_SYS_PATH_APPENDED = False


def import_plot_helpers() -> Callable:
    global _SYS_PATH_APPENDED
    if not _SYS_PATH_APPENDED:
        sys.path.append(str(Path(__file__).resolve().parent))
        _SYS_PATH_APPENDED = True
    from plot_fleet_reduction import build_comparison  # type: ignore

    return build_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        help=(
            "Optional explicit dataset directories or identifiers. When omitted, the script "
            "collects the first --limit datasets under data/processed/alns/."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of datasets to process when --datasets is not provided (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed forwarded to the Rust ALNS engine (default: 0).",
    )
    parser.add_argument(
        "--destroy",
        default="random_visit_removal",
        help="Destroy operator name applied before deep swap (default: random_visit_removal).",
    )
    parser.add_argument(
        "--repair",
        default="deep_greedy_insertion",
        help="Repair operator name applied before deep swap (default: deep_greedy_insertion).",
    )
    parser.add_argument(
        "--improvement",
        default="deep_swap",
        help="Improvement operator name invoked for snapshots (default: deep_swap).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rust_alns/output/python/deep_swap"),
        help="Directory where per-dataset snapshots and plots are written.",
    )
    parser.add_argument(
        "--max-degrade-iterations",
        type=int,
        default=5,
        help="Maximum destroy+repair iterations attempted to obtain an accepted degradation (default: 5).",
    )
    parser.add_argument(
        "--require-cost-increase",
        action="store_true",
        help=(
            "When set, the script insists on finding an accepted iteration that worsens cost "
            "before applying the deep swap operator."
        ),
    )
    parser.add_argument(
        "--export-summary",
        type=Path,
        help="Optional path to a JSON summary report aggregating costs per dataset.",
    )
    return parser.parse_args()


def discover_datasets(limit: int) -> List[str]:
    root = Path("data/processed/alns")
    if not root.exists():
        return ["SMALL_1"]

    candidates: List[Path] = []
    for directory in sorted(root.rglob("*")):
        if not directory.is_dir():
            continue
        install_csv = directory / "installations.csv"
        vessels_csv = directory / "vessels.csv"
        base_csv = directory / "base.csv"
        if install_csv.exists() and vessels_csv.exists() and base_csv.exists():
            candidates.append(directory)
        if len(candidates) >= limit:
            break

    if not candidates:
        return ["SMALL_1"]
    return [str(path) for path in candidates]


def resolve_operator_index(operators: Sequence[str], target: str, kind: str) -> int:
    if not operators:
        raise RuntimeError(f"No {kind} operators registered in the Rust engine")
    try:
        return operators.index(target)
    except ValueError as exc:
        available = ", ".join(operators)
        raise RuntimeError(
            f"{kind.capitalize()} operator '{target}' not found. Available: {available}"
        ) from exc


def slugify_dataset(dataset: str) -> str:
    normalized = dataset.replace(os.sep, "__")
    normalized = normalized.replace("/", "__")
    return normalized


def ensure_moves_dir(base_dir: Path) -> Path:
    moves_dir = base_dir / "moves"
    moves_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ALNS_MOVES_DIR"] = str(moves_dir)
    return moves_dir


def process_dataset(
    dataset: str,
    seed: int,
    destroy_name: str,
    repair_name: str,
    improvement_name: str,
    output_dir: Path,
    max_degrade_iterations: int,
    require_cost_increase: bool,
) -> dict:
    interface = RustALNSInterface()
    init_result = interface.initialize_alns(dataset, seed)

    operator_info = interface.get_operator_info()
    destroy_idx = resolve_operator_index(
        list(operator_info.get("destroy_operators", [])), destroy_name, "destroy"
    )
    repair_idx = resolve_operator_index(
        list(operator_info.get("repair_operators", [])), repair_name, "repair"
    )
    improvement_idx = resolve_operator_index(
        list(operator_info.get("improvement_operators", [])), improvement_name, "improvement"
    )

    baseline_cost = float(init_result.get("total_cost", 0.0))
    degrade_metrics: Optional[dict] = None
    current_cost = baseline_cost

    for attempt in range(max_degrade_iterations):
        metrics = interface.execute_iteration(
            iteration=attempt,
            destroy_operator_idx=destroy_idx,
            repair_operator_idx=repair_idx,
            improvement_operator_idx=None,
            mode="explicit",
        )
        degrade_metrics = metrics
        current_cost = float(metrics.get("current_cost", current_cost))
        accepted = bool(metrics.get("accepted", False))
        if not accepted:
            continue
        if not require_cost_increase:
            break
        if current_cost > baseline_cost + 1e-6:
            break
    else:
        if require_cost_increase and current_cost <= baseline_cost + 1e-6:
            print(
                f"[WARN] Dataset {dataset}: unable to obtain an accepted degradation after {max_degrade_iterations} attempts.",
                file=sys.stderr,
            )

    dataset_dir = output_dir / slugify_dataset(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ensure_moves_dir(dataset_dir)

    before_path = dataset_dir / "before.json"
    after_path = dataset_dir / "after.json"

    improvement_metrics = interface.execute_improvement_with_snapshots(
        iteration=max_degrade_iterations,
        improvement_operator_idx=improvement_idx,
        before_path=str(before_path),
        after_path=str(after_path),
    )

    build_comparison = import_plot_helpers()
    figure = build_comparison(before_path, after_path, title_suffix=Path(dataset).name)
    html_path = dataset_dir / "comparison.html"
    figure.write_html(html_path, auto_open=False)

    result = {
        "dataset": dataset,
        "baseline_cost": baseline_cost,
        "degraded_cost": current_cost,
        "improved_cost": float(improvement_metrics.get("current_cost", current_cost)),
        "accepted_degradation": bool(degrade_metrics.get("accepted", False)) if degrade_metrics else False,
        "degradation_metrics": degrade_metrics,
        "improvement_metrics": improvement_metrics,
        "before_snapshot": str(before_path),
        "after_snapshot": str(after_path),
        "plot": str(html_path),
    }
    summary_path = dataset_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    args = parse_args()
    datasets = args.datasets or discover_datasets(args.limit)
    if not datasets:
        print("No datasets found to process.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        summary = process_dataset(
            dataset=dataset,
            seed=args.seed,
            destroy_name=args.destroy,
            repair_name=args.repair,
            improvement_name=args.improvement,
            output_dir=args.output_dir,
            max_degrade_iterations=args.max_degrade_iterations,
            require_cost_increase=args.require_cost_increase,
        )
        print(
            "  baseline={baseline_cost:.2f} degraded={degraded_cost:.2f} improved={improved_cost:.2f}".format(
                **summary
            )
        )
        summaries.append(summary)

    if args.export_summary:
        args.export_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.export_summary.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)
        print(f"Summary written to {args.export_summary}")
if __name__ == "__main__":
    main()
