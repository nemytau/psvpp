#!/usr/bin/env python3
"""Run the FleetAndCostReduction operator via the PyO3 bindings and export before/after snapshots."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from rust_alns_py import RustALNSInterface  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - diagnostic path
    print("rust_alns_py module not found: {}".format(exc))
    print("Build the extension first: cd rust_alns && maturin develop --release")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="SMALL_1",
        help=(
            "Problem instance identifier or path to a dataset folder. "
            "The default uses the bundled SMALL_1 sample."
        ),
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the ALNS engine")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional initial temperature override (defaults to interface default)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        help="Optional temperature decay override (defaults to interface default)",
    )
    parser.add_argument(
        "--weight-update-interval",
        type=int,
        help="Optional weight update interval override (defaults to interface default)",
    )
    parser.add_argument(
        "--improvement",
        default="fleet_and_cost_reduction",
        help="Improvement operator name to execute (must match Rust registry)",
    )
    parser.add_argument(
        "--destroy",
        default="random_visit_removal",
        help="Destroy operator name applied before improvement",
    )
    parser.add_argument(
        "--repair",
        default="deep_greedy_insertion",
        help="Repair operator name applied after destroy",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration index passed to the improvement run (informational only)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rust_alns/output/python"),
        help="Directory used when --before/--after are not provided",
    )
    parser.add_argument("--before", type=Path, help="Explicit path for the pre-operator snapshot")
    parser.add_argument("--after", type=Path, help="Explicit path for the post-operator snapshot")
    parser.add_argument(
        "--moves-dir",
        type=Path,
        help=(
            "Directory used to store per-iteration snapshots. Defaults to a subdirectory next to the"
            " before/after snapshots. Overrides ALNS_MOVES_DIR when provided."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help=(
            "Optional log file written via RustALNSInterface.enable_file_logging. Parent directories"
            " are created automatically."
        ),
    )
    return parser.parse_args()


def resolve_operator_index(operators: list[str], target: str, kind: str) -> int:
    if not operators:
        raise RuntimeError(f"No {kind} operators registered in the Rust engine")
    try:
        return operators.index(target)
    except ValueError as exc:
        available = ", ".join(operators)
        raise RuntimeError(
            f"{kind.capitalize()} operator '{target}' not found. Available: {available}"
        ) from exc


def materialize_snapshot_path(candidate: Optional[Path], fallback: Path) -> Path:
    path = candidate if candidate is not None else fallback
    return path.expanduser().resolve()


def main() -> None:
    args = parse_args()

    if args.log_file is not None:
        log_path = args.log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        RustALNSInterface.enable_file_logging(str(log_path))
    interface = RustALNSInterface()

    init_kwargs = {
        "temperature": args.temperature,
        "theta": args.theta,
        "weight_update_interval": args.weight_update_interval,
    }
    init_result = interface.initialize_alns(args.dataset, args.seed, **init_kwargs)

    operator_info = interface.get_operator_info()
    destroy_idx = resolve_operator_index(
        list(operator_info.get("destroy_operators", [])), args.destroy, "destroy"
    )
    repair_idx = resolve_operator_index(
        list(operator_info.get("repair_operators", [])), args.repair, "repair"
    )
    improvement_idx = resolve_operator_index(
        list(operator_info.get("improvement_operators", [])), args.improvement, "improvement"
    )

    pre_iteration_snapshot = interface.create_snapshot()

    output_dir = args.output_dir.expanduser().resolve()
    before_path = materialize_snapshot_path(
        args.before, output_dir / "{}__seed{}__before.json".format(args.improvement, args.seed)
    )
    after_path = materialize_snapshot_path(
        args.after, output_dir / "{}__seed{}__after.json".format(args.improvement, args.seed)
    )
    moves_dir: Optional[Path]
    env_moves_dir = os.environ.get("ALNS_MOVES_DIR")
    if args.moves_dir is not None or not env_moves_dir:
        moves_dir = materialize_snapshot_path(
            args.moves_dir,
            output_dir / "{}__seed{}__moves".format(args.improvement, args.seed),
        )
        moves_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ALNS_MOVES_DIR"] = str(moves_dir)
    else:
        moves_dir = Path(env_moves_dir).expanduser().resolve()
        moves_dir.mkdir(parents=True, exist_ok=True)

    iteration_metrics = interface.execute_iteration(
        iteration=args.iteration,
        destroy_operator_idx=destroy_idx,
        repair_operator_idx=repair_idx,
        improvement_operator_idx=improvement_idx,
        mode="explicit",
    )

    post_iteration_snapshot = interface.create_snapshot()

    interface.apply_snapshot(pre_iteration_snapshot)
    interface.dump_current_solution(str(before_path))
    interface.apply_snapshot(post_iteration_snapshot)
    interface.dump_current_solution(str(after_path))

    summary = {
        "accepted": bool(iteration_metrics.get("accepted", False)),
        "total_cost": float(iteration_metrics.get("current_cost", 0.0)),
        "best_cost": float(iteration_metrics.get("best_cost", 0.0)),
        "improvement_operator": args.improvement,
    }

    print("Initial cost: {:.2f}".format(float(init_result.get("total_cost", 0.0))))
    print(
        "Post-iteration cost: {:.2f} (accepted={})".format(
            summary["total_cost"], summary["accepted"],
        )
    )
    print("Snapshots: {} (before), {} (after)".format(before_path, after_path))
    if moves_dir:
        print("Iteration snapshots: {}".format(moves_dir))


if __name__ == "__main__":
    main()
