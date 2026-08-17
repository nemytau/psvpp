#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.dataset_manager import GeneratedDatasetManager
from rl.rl_alns_environment import ALNSEnvironment
from rl.train_alns_rl import train_ppo_agent


@dataclass
class EpisodeSummary:
    initial_cost: float
    best_cost: float
    solver_runtime_seconds: float
    wall_runtime_seconds: float

    @property
    def improvement_pct(self) -> float:
        if self.initial_cost <= 1e-12:
            return 0.0
        return max(0.0, (self.initial_cost - self.best_cost) / self.initial_cost * 100.0)

    @property
    def runtime_seconds(self) -> float:
        return self.wall_runtime_seconds


def _restart_seed(base_seed: int, restart_idx: int) -> int:
    return int(base_seed + restart_idx * 1009)


def _time_to_best(stats: Dict) -> tuple:
    """Return (solver_seconds_to_best, iterations_to_best) within a single restart.

    Uses the cumulative elapsed_seconds from episode_history.  If the episode
    produced no usable data both values are 0.
    """
    best_cost_history = stats.get("best_cost_history") or []
    elapsed_seconds = stats.get("elapsed_seconds") or []
    if not best_cost_history:
        return 0.0, 0
    global_best = min(best_cost_history)
    for i, cost in enumerate(best_cost_history):
        if cost <= global_best + 1e-9:
            t = float(elapsed_seconds[i]) if i < len(elapsed_seconds) else 0.0
            return t, i
    return 0.0, 0


def _episode_time_to_best(restart_records: list) -> float:
    """Compute cumulative solver seconds until the global best was first found.

    Sums full solver time of restarts that completed before the best-finding
    restart, then adds the time-to-best within that restart.
    """
    if not restart_records:
        return 0.0
    global_best = min(r["best_cost"] for r in restart_records)
    cumulative_solver = 0.0
    for rec in restart_records:
        if rec["best_cost"] <= global_best + 1e-9:
            return cumulative_solver + float(rec.get("time_to_best_solver_seconds", 0.0))
        cumulative_solver += float(rec.get("solver_runtime_seconds", 0.0))
    return cumulative_solver


def _episode_vessel_stats(restart_records: list) -> tuple:
    """Return (initial_num_vessels_used, best_num_vessels_used) for an episode.

    initial is taken from the first restart (representative of the constructive
    heuristic's fleet size before search); best is the vessel count of whichever
    restart achieved the episode's global-best cost.
    """
    if not restart_records:
        return None, None
    initial_vessels = restart_records[0].get("initial_num_vessels_used")
    best_record = min(restart_records, key=lambda r: r["best_cost"])
    return initial_vessels, best_record.get("best_num_vessels_used")


def parse_size_list(size_arg: str) -> List[str]:
    sizes = [item.strip().lower() for item in str(size_arg).split(",") if item.strip()]
    deduped: List[str] = []
    seen = set()
    for size in sizes:
        if size not in seen:
            deduped.append(size)
            seen.add(size)
    return deduped


def infer_dataset_size_label(dataset_path: str) -> str:
    path = Path(dataset_path)
    try:
        return path.parent.parent.name or "unknown"
    except Exception:
        return "unknown"


def _build_exact_command() -> str:
    return " ".join([shlex.quote(sys.executable), *[shlex.quote(token) for token in sys.argv]])


def _write_command_file(out_dir: Path, command: str) -> None:
    (out_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")


def _collect_training_operator_usage(log_dir: Path) -> Dict[str, Any]:
    usage_dir = log_dir / "operator_usage"
    csv_paths = sorted(usage_dir.glob("operator_usage_train_*.csv"))
    if not csv_paths:
        csv_paths = sorted(log_dir.glob("operator_usage_train_*.csv"))

    operator_name_counts: Counter[str] = Counter()
    operator_type_counts: Counter[str] = Counter()
    source_files: List[str] = []

    for csv_path in csv_paths:
        source_files.append(str(csv_path))
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                operator_name = str(row.get("operator_name") or "")
                operator_type = str(row.get("operator_type") or "")
                if operator_name:
                    operator_name_counts[operator_name] += 1
                if operator_type:
                    operator_type_counts[operator_type] += 1

    return {
        "source_files": source_files,
        "operator_name_counts": dict(sorted(operator_name_counts.items())),
        "operator_type_counts": dict(sorted(operator_type_counts.items())),
        "unique_operator_names": sorted(operator_name_counts),
        "unique_operator_types": sorted(operator_type_counts),
        "total_operator_events": int(sum(operator_name_counts.values())),
    }


def _collect_action_space_trace(*, train_env: ALNSEnvironment) -> Dict[str, Any]:
    # NOTE: reads from action_impl (the actual restricted operator set the
    # action space uses), not train_env (the full unrestricted engine
    # catalog) -- these differ for every action module except the
    # deliberately-unrestricted adaptive_weighted_full_v1. Falling back to
    # train_env here previously made this trace misleadingly report e.g. 36
    # destroy operators for op_pair_v1, which only ever uses 3.
    action_impl = train_env.action_impl

    def _pick(attr: str) -> list:
        impl_value = getattr(action_impl, attr, None)
        if impl_value is not None:
            return list(impl_value)
        return list(getattr(train_env, attr, []))

    destroy_operators = _pick("destroy_operators")
    repair_operators = _pick("repair_operators")
    improvement_operators = _pick("improvement_operators")
    action_space: Dict[str, Any] = {
        "module": getattr(train_env, "action_module_name", None),
        "version": getattr(action_impl, "version", None),
        "space_n": int(action_impl.n()),
        "num_destroy": len(destroy_operators),
        "num_repair": len(repair_operators),
        "num_improvement": len(improvement_operators),
        "destroy_operators": destroy_operators,
        "repair_operators": repair_operators,
        "improvement_operators": improvement_operators,
    }
    if hasattr(action_impl, "num_pairs"):
        action_space["num_pairs"] = int(action_impl.num_pairs())
    if hasattr(action_impl, "first_improvement_action"):
        action_space["first_improvement_action"] = action_impl.first_improvement_action()
    return action_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traceable RL experiment runner for final model comparisons"
    )
    parser.add_argument("--size", default="small", help="Dataset size to run (default: small)")
    parser.add_argument(
        "--sizes",
        default=None,
        help="Comma-separated dataset sizes to include for joint train/eval (overrides --size)",
    )
    parser.add_argument("--datasets-per-size", type=int, default=5)
    parser.add_argument("--train-timesteps", type=int, default=30000)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument(
        "--eval-runtime-mode",
        choices=("iterations", "match_kisialiou"),
        default="match_kisialiou",
        help="Evaluation stopping rule: fixed iterations or wall-time matched to Kisialiou per dataset/seed",
    )
    parser.add_argument(
        "--eval-max-iterations",
        type=int,
        default=None,
        help="Iteration cap used during evaluation (defaults to --max-iterations, or a larger cap in match_kisialiou mode)",
    )
    parser.add_argument(
        "--eval-restart-max-iterations",
        type=int,
        default=300,
        help="Maximum iterations per single restart during evaluation before moving to next restart",
    )
    parser.add_argument(
        "--eval-runtime-reference-restarts",
        type=int,
        default=5,
        help="In match_kisialiou mode, number of Kisialiou restarts used to define runtime reference",
    )
    parser.add_argument(
        "--eval-fixed-restarts",
        type=int,
        default=None,
        help="If set, run exactly this many restarts per dataset/seed for all evaluation modes",
    )
    parser.add_argument(
        "--eval-runtime-budget-seconds",
        type=float,
        default=None,
        help=(
            "In match_kisialiou mode, cap the Kisialiou reference run's own wall time at this "
            "many seconds (in addition to --eval-runtime-reference-restarts), so a slow "
            "operator mix (e.g. large instances) can't silently blow up total eval time. "
            "Baseline/RL still match whatever wall time Kisialiou actually took, so this stays "
            "a genuine wall-time comparison -- it just bounds the reference measurement itself. "
            "Default (None) preserves the old unbounded-reference behavior."
        ),
    )
    parser.add_argument("--eval-seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--action-module",
        default="op_pair_v1",
        help="Action space registry key for train/eval environments",
    )
    parser.add_argument(
        "--baseline-action-module",
        default="adaptive_weighted_v1",
        help="Baseline action space key (default: adaptive_weighted_v1 = adaptive weighted ALNS)",
    )
    parser.add_argument(
        "--state-module",
        default="features_improvement_v1",
        help="State encoder registry key for train/eval environments",
    )
    parser.add_argument(
        "--reward-module",
        default="delta_cost_v4",
        help="Reward function registry key for train/eval environments",
    )
    parser.add_argument(
        "--baseline-state-module",
        default="features_v3",
        help="Baseline state encoder key (independent from RL policy state module)",
    )
    parser.add_argument(
        "--chain-max-consecutive-improvements",
        type=int,
        default=3,
        help="For improvement_chain_v1, force a backbone step after this many consecutive improvement-only actions",
    )
    parser.add_argument(
        "--anti-repeat-penalty",
        type=float,
        default=0.0,
        help="Reward penalty applied when the same improvement operator is repeated without finding a new best",
    )
    parser.add_argument(
        "--stochastic-check",
        action="store_true",
        help="Also evaluate RL policy with stochastic action sampling (deterministic=False)",
    )
    parser.add_argument("--run-tag", default="exp1_id1_op_pair_v1_features_improvement_v1_delta_cost_v4_train_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Allow overwriting an existing output/log folder if it already exists",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Also run evaluation episodes after training",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip PPO training and load an existing model for evaluation-only runs",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to existing PPO zip model (required with --skip-training)",
    )
    return parser.parse_args()


def parse_seed_list(raw: str) -> List[int]:
    seeds: List[int] = []
    for item in str(raw).split(","):
        val = item.strip()
        if not val:
            continue
        seeds.append(int(val))
    if not seeds:
        raise ValueError("At least one eval seed must be provided")
    return seeds


def run_policy_episode(
    *,
    model: PPO,
    dataset_path: str,
    seed: int,
    max_iterations: int,
    log_dir: Path,
    mode_label: str,
    size_label: str,
    action_module: str,
    state_module: str,
    reward_module: str,
    restart_max_iterations: int,
    chain_max_consecutive_improvements: Optional[int] = None,
    anti_repeat_penalty: float = 0.0,
    policy_deterministic: bool = True,
    target_runtime_seconds: Optional[float] = None,
    restart_limit: Optional[int] = None,
) -> tuple[EpisodeSummary, List[object], List[Dict[str, object]]]:
    restart_cap = max(1, min(int(restart_max_iterations), int(max_iterations)))
    effective_restart_limit = max(1, int(restart_limit)) if restart_limit is not None else int(max_iterations)
    solver_elapsed_ms_total = 0.0
    total_solver_runtime_seconds = 0.0
    total_wall_runtime_seconds = 0.0
    initial_cost: Optional[float] = None
    best_candidates: List[float] = []
    restart_snapshots: List[object] = []
    restart_records: List[Dict[str, object]] = []
    restart_idx = 0

    while restart_idx < effective_restart_limit:
        if target_runtime_seconds is not None and restart_idx > 0 and total_wall_runtime_seconds >= target_runtime_seconds:
            break

        restart_seed = _restart_seed(seed, restart_idx)
        restart_wall_start = time.perf_counter()

        env = ALNSEnvironment(
            problem_instance=dataset_path,
            max_iterations=restart_cap,
            seed=restart_seed,
            problem_instance_paths=[dataset_path],
            problem_sampling_strategy="round_robin",
            enable_operator_logging=True,
            operator_logging_mode="reinforcement_learning",
            operator_logging_mode_label=mode_label,
            operator_logging_size=size_label,
            operator_logging_dataset_name=f"{Path(dataset_path).name}__seed{seed}",
            operator_logging_restart_id=restart_idx,
            operator_logging_stable_filename=True,
            operator_logging_dir=str(log_dir),
            action_module=action_module,
            state_module=state_module,
            reward_module=reward_module,
            chain_max_consecutive_improvements=chain_max_consecutive_improvements,
            anti_repeat_penalty=anti_repeat_penalty,
            algorithm_mode="baseline",
            size_bucket="small" if size_label == "small" else "large",
        )

        obs, _ = env.reset(seed=restart_seed, options={"problem_path": dataset_path})
        restart_snapshots.append(env.get_initial_snapshot())

        done = False
        truncated = False
        steps = 0
        restart_elapsed_ms = 0.0
        while not done and not truncated and steps < restart_cap:
            action_raw, _ = model.predict(obs, deterministic=policy_deterministic)
            action = int(action_raw.item()) if hasattr(action_raw, "item") else int(action_raw)
            obs, _, done, truncated, info = env.step(action)
            step_info = info.get("step_info", {}) if isinstance(info, dict) else {}
            step_ms = float(step_info.get("operator_elapsed_ms", step_info.get("elapsed_ms", 0.0)) or 0.0)
            restart_elapsed_ms += step_ms
            solver_elapsed_ms_total += step_ms
            steps += 1

        stats = env.get_episode_statistics()
        env.close()

        restart_initial = float(stats.get("initial_cost") or 0.0)
        restart_best = float(stats.get("best_cost") or restart_initial)
        restart_initial_vessels = stats.get("initial_num_vessels_used")
        restart_best_vessels = stats.get("best_num_vessels_used")
        if initial_cost is None:
            initial_cost = restart_initial
        best_candidates.append(restart_best)

        elapsed_seq = stats.get("elapsed_seconds") or []
        restart_solver_runtime = float(elapsed_seq[-1]) if elapsed_seq else (restart_elapsed_ms / 1000.0)
        restart_wall_runtime = time.perf_counter() - restart_wall_start
        total_solver_runtime_seconds += restart_solver_runtime
        total_wall_runtime_seconds += restart_wall_runtime
        ttb_solver, ttb_iters = _time_to_best(stats)
        restart_records.append(
            {
                "restart_id": restart_idx,
                "seed": seed,
                "restart_seed": restart_seed,
                "solver_runtime_seconds": restart_solver_runtime,
                "wall_runtime_seconds": restart_wall_runtime,
                "time_to_best_solver_seconds": ttb_solver,
                "iterations_to_best": ttb_iters,
                "initial_cost": restart_initial,
                "best_cost": restart_best,
                "initial_num_vessels_used": restart_initial_vessels,
                "best_num_vessels_used": restart_best_vessels,
                "iterations": int(stats.get("total_iterations", steps)),
                "dataset_path": dataset_path,
                "mode_label": mode_label,
                "size_label": size_label,
                "algorithm_mode": "baseline",
            }
        )

        restart_idx += 1
        if target_runtime_seconds is None:
            break

    if initial_cost is None:
        initial_cost = 0.0
    global_best = min(best_candidates) if best_candidates else 0.0

    return (
        EpisodeSummary(
            initial_cost=float(initial_cost),
            best_cost=float(global_best),
            solver_runtime_seconds=float(total_solver_runtime_seconds),
            wall_runtime_seconds=float(total_wall_runtime_seconds),
        ),
        restart_snapshots,
        restart_records,
    )


def run_baseline_episode(
    *,
    dataset_path: str,
    seed: int,
    max_iterations: int,
    log_dir: Path,
    mode_label: str,
    size_label: str,
    initial_snapshots: Optional[List[object]],
    action_module: str,
    state_module: str,
    reward_module: str,
    restart_max_iterations: int,
    target_runtime_seconds: Optional[float] = None,
    restart_limit: Optional[int] = None,
) -> tuple[EpisodeSummary, List[Dict[str, object]]]:
    restart_cap = max(1, min(int(restart_max_iterations), int(max_iterations)))
    effective_restart_limit = max(1, int(restart_limit)) if restart_limit is not None else int(max_iterations)
    solver_elapsed_ms_total = 0.0
    total_solver_runtime_seconds = 0.0
    total_wall_runtime_seconds = 0.0
    initial_cost: Optional[float] = None
    best_candidates: List[float] = []
    restart_idx = 0
    restart_records: List[Dict[str, object]] = []

    while restart_idx < effective_restart_limit:
        if target_runtime_seconds is not None and restart_idx > 0 and total_wall_runtime_seconds >= target_runtime_seconds:
            break

        restart_seed = _restart_seed(seed, restart_idx)
        restart_wall_start = time.perf_counter()

        env = ALNSEnvironment(
            problem_instance=dataset_path,
            max_iterations=restart_cap,
            seed=restart_seed,
            problem_instance_paths=[dataset_path],
            problem_sampling_strategy="round_robin",
            enable_operator_logging=True,
            operator_logging_mode="baseline",
            operator_logging_mode_label=mode_label,
            operator_logging_size=size_label,
            operator_logging_dataset_name=f"{Path(dataset_path).name}__seed{seed}",
            operator_logging_restart_id=restart_idx,
            operator_logging_stable_filename=True,
            operator_logging_dir=str(log_dir),
            action_module=action_module,
            state_module=state_module,
            reward_module=reward_module,
            force_baseline_improvement=False,
            baseline_improvement_idx=None,
            chain_max_consecutive_improvements=None,
            anti_repeat_penalty=0.0,
            algorithm_mode="baseline",
            size_bucket="small" if size_label == "small" else "large",
        )

        reset_options: Dict[str, Any] = {"problem_path": dataset_path}
        if initial_snapshots and restart_idx < len(initial_snapshots):
            reset_options["initial_snapshot"] = initial_snapshots[restart_idx]

        obs, _ = env.reset(seed=restart_seed, options=reset_options)

        done = False
        truncated = False
        steps = 0
        rng = np.random.default_rng(restart_seed)
        restart_elapsed_ms = 0.0

        if not hasattr(env.action_impl, "num_pairs"):
            raise RuntimeError(
                f"Baseline action module '{action_module}' does not expose pair-only actions; "
                "use 'op_pair_v1' for baseline isolation"
            )

        while not done and not truncated and steps < restart_cap:
            num_pairs = int(env.action_impl.num_pairs())
            action = int(rng.integers(0, num_pairs)) if num_pairs > 0 else int(env.action_space.sample())
            obs, _, done, truncated, info = env.step(action)
            step_info = info.get("step_info", {}) if isinstance(info, dict) else {}
            step_ms = float(step_info.get("operator_elapsed_ms", step_info.get("elapsed_ms", 0.0)) or 0.0)
            restart_elapsed_ms += step_ms
            solver_elapsed_ms_total += step_ms
            steps += 1

        stats = env.get_episode_statistics()
        env.close()

        restart_initial = float(stats.get("initial_cost") or 0.0)
        restart_best = float(stats.get("best_cost") or restart_initial)
        restart_initial_vessels = stats.get("initial_num_vessels_used")
        restart_best_vessels = stats.get("best_num_vessels_used")
        if initial_cost is None:
            initial_cost = restart_initial
        best_candidates.append(restart_best)

        elapsed_seq = stats.get("elapsed_seconds") or []
        restart_solver_runtime = float(elapsed_seq[-1]) if elapsed_seq else (restart_elapsed_ms / 1000.0)
        restart_wall_runtime = time.perf_counter() - restart_wall_start
        total_solver_runtime_seconds += restart_solver_runtime
        total_wall_runtime_seconds += restart_wall_runtime
        ttb_solver, ttb_iters = _time_to_best(stats)
        restart_records.append(
            {
                "restart_id": restart_idx,
                "seed": seed,
                "restart_seed": restart_seed,
                "solver_runtime_seconds": restart_solver_runtime,
                "wall_runtime_seconds": restart_wall_runtime,
                "time_to_best_solver_seconds": ttb_solver,
                "iterations_to_best": ttb_iters,
                "initial_cost": restart_initial,
                "best_cost": restart_best,
                "initial_num_vessels_used": restart_initial_vessels,
                "best_num_vessels_used": restart_best_vessels,
                "iterations": int(stats.get("total_iterations", steps)),
                "dataset_path": dataset_path,
                "mode_label": mode_label,
                "size_label": size_label,
                "algorithm_mode": "baseline",
            }
        )

        restart_idx += 1
        if target_runtime_seconds is None:
            break

    if initial_cost is None:
        initial_cost = 0.0
    global_best = min(best_candidates) if best_candidates else 0.0

    return EpisodeSummary(
        initial_cost=float(initial_cost),
        best_cost=float(global_best),
        solver_runtime_seconds=float(total_solver_runtime_seconds),
        wall_runtime_seconds=float(total_wall_runtime_seconds),
    ), restart_records


def _safe_git_output(repo: Path, args: List[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def _collect_git_metadata(repo: Path) -> Dict[str, Any]:
    commit = _safe_git_output(repo, ["rev-parse", "HEAD"])
    branch = _safe_git_output(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    status = _safe_git_output(repo, ["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status) if status is not None else None,
        "status_porcelain": status.splitlines() if status else [],
    }


def _collect_runtime_metadata() -> Dict[str, Any]:
    package_versions: Dict[str, Optional[str]] = {
        "numpy": getattr(np, "__version__", None),
    }
    try:
        import gymnasium as gymn

        package_versions["gymnasium"] = getattr(gymn, "__version__", None)
    except Exception:
        package_versions["gymnasium"] = None

    try:
        import stable_baselines3 as sb3

        package_versions["stable_baselines3"] = getattr(sb3, "__version__", None)
    except Exception:
        package_versions["stable_baselines3"] = None

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
    }


def _build_run_manifest(
    *,
    args: argparse.Namespace,
    repo: Path,
    run_id: str,
    out_dir: Path,
    log_dir: Path,
    eval_seeds: List[int],
    eval_iteration_cap: int,
    train_paths: List[str],
    test_paths: List[str],
    exact_command: str,
    action_space_trace: Dict[str, Any],
    training_operator_usage: Dict[str, Any],
) -> Dict[str, Any]:
    train_env_kwargs = {
        "problem_sampling_strategy": "random",
        "action_module": args.action_module,
        "state_module": args.state_module,
        "reward_module": args.reward_module,
        "algorithm_mode": "baseline",
        "chain_max_consecutive_improvements": args.chain_max_consecutive_improvements,
        "anti_repeat_penalty": args.anti_repeat_penalty,
    }
    rl_eval_env_kwargs = {
        "action_module": args.action_module,
        "state_module": args.state_module,
        "reward_module": args.reward_module,
        "algorithm_mode": "baseline",
        "chain_max_consecutive_improvements": args.chain_max_consecutive_improvements,
        "anti_repeat_penalty": args.anti_repeat_penalty,
        "restart_max_iterations": args.eval_restart_max_iterations,
    }
    baseline_eval_env_kwargs = {
        "action_module": args.baseline_action_module,
        "state_module": args.baseline_state_module,
        "reward_module": args.reward_module,
        "algorithm_mode": "baseline",
        "force_baseline_improvement": False,
        "baseline_improvement_idx": None,
        "chain_max_consecutive_improvements": None,
        "anti_repeat_penalty": 0.0,
        "restart_max_iterations": args.eval_restart_max_iterations,
        "policy": "random_destroy_repair_pair_only_v1",
    }
    kisialiou_eval_env_kwargs = {
        "action_module": args.action_module,
        "state_module": args.state_module,
        "reward_module": args.reward_module,
        "algorithm_mode": "kisialiou",
        "chain_max_consecutive_improvements": args.chain_max_consecutive_improvements,
        "restart_max_iterations": args.eval_restart_max_iterations,
    }

    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "command": exact_command,
        "repo_root": str(repo),
        "output_dir": str(out_dir),
        "log_dir": str(log_dir),
        "cli_args": vars(args),
        "experiment_label": {
            "action_module": args.action_module,
            "state_module": args.state_module,
            "reward_module": args.reward_module,
            "tag": args.run_tag,
        },
        "resolved": {
            "skip_training": bool(args.skip_training),
            "model_path": args.model_path,
            "eval_seeds": eval_seeds,
            "eval_iteration_cap": eval_iteration_cap,
            "eval_restart_max_iterations": args.eval_restart_max_iterations,
            "train_paths": train_paths,
            "test_paths": test_paths,
            "train_env_kwargs": train_env_kwargs,
            "rl_eval_env_kwargs": rl_eval_env_kwargs,
            "baseline_eval_env_kwargs": baseline_eval_env_kwargs,
            "kisialiou_eval_env_kwargs": kisialiou_eval_env_kwargs,
            "training_action_space": action_space_trace,
            "training_operator_usage": training_operator_usage,
            "runtime_matching": {
                "mode": args.eval_runtime_mode,
                "target_source": (
                    "fixed_restart_count"
                    if args.eval_fixed_restarts is not None
                    else "kisialiou_runtime_seconds_per_dataset_seed"
                ),
                "reference_restarts": args.eval_runtime_reference_restarts,
                "reference_runtime_budget_seconds": args.eval_runtime_budget_seconds,
                "fixed_restarts": args.eval_fixed_restarts,
                "minimum_seconds": 1e-6,
            },
        },
        "git": _collect_git_metadata(repo),
        "runtime": _collect_runtime_metadata(),
    }


def run_kisialiou_episode(
    *,
    dataset_path: str,
    seed: int,
    max_iterations: int,
    log_dir: Path,
    mode_label: str,
    size_label: str,
    action_module: str,
    state_module: str,
    reward_module: str,
    restart_max_iterations: int,
    target_runtime_seconds: Optional[float] = None,
    chain_max_consecutive_improvements: Optional[int] = None,
    reference_restarts: Optional[int] = None,
) -> tuple[EpisodeSummary, List[Dict[str, object]]]:
    """
    Run Kisialiou deterministic ALNS algorithm with operator logging.
    
     IMPORTANT: We use step-by-step execution (env.reset() + env.step() loop) rather than
     env.run_with_restarts() to ensure operator usage is logged for each iteration.
     The run_with_restarts() method executes entirely in Rust without calling Python's
     step function, which bypasses the operator logging pipeline.
    
     By using Gym-compatible step-by-step execution with algorithm_mode="kisialiou",
     the Rust backend still implements the deterministic Kisialiou operator selection
     strategy, but each iteration is recorded in the operators logs (CSV files).
    
     This ensures consistency with other evaluation modes (baseline, RL) which also
     use step-by-step execution and produce detailed operator logs for analysis.
     """
    restart_cap = max(1, min(int(restart_max_iterations), int(max_iterations)))
    solver_elapsed_ms_total = 0.0
    total_solver_runtime_seconds = 0.0
    total_wall_runtime_seconds = 0.0
    initial_cost: Optional[float] = None
    best_candidates: List[float] = []
    restart_idx = 0
    restart_limit = max(1, int(reference_restarts)) if reference_restarts is not None else int(max_iterations)
    restart_records: List[Dict[str, object]] = []

    while restart_idx < restart_limit:
        if target_runtime_seconds is not None and restart_idx > 0 and total_wall_runtime_seconds >= target_runtime_seconds:
            break

        restart_seed = _restart_seed(seed, restart_idx)
        restart_wall_start = time.perf_counter()

        env = ALNSEnvironment(
            problem_instance=dataset_path,
            max_iterations=restart_cap,
            seed=restart_seed,
            problem_instance_paths=[dataset_path],
            problem_sampling_strategy="round_robin",
            enable_operator_logging=True,
            operator_logging_mode="kisialiou",
            operator_logging_mode_label=mode_label,
            operator_logging_size=size_label,
            operator_logging_dataset_name=f"{Path(dataset_path).name}__seed{seed}",
            operator_logging_restart_id=restart_idx,
            operator_logging_stable_filename=True,
            operator_logging_dir=str(log_dir),
            action_module=action_module,
            state_module=state_module,
            reward_module=reward_module,
            algorithm_mode="kisialiou",
            chain_max_consecutive_improvements=chain_max_consecutive_improvements,
            size_bucket="small" if size_label == "small" else "large",
        )

        obs, _ = env.reset(seed=restart_seed, options={"problem_path": dataset_path})

        done = False
        truncated = False
        steps = 0
        restart_elapsed_ms = 0.0

        # Run step-by-step to enable operator logging
        while not done and not truncated and steps < restart_cap:
            # For Kisialiou, let the environment handle action selection via the algorithm_mode
            # We use a dummy action (0) since Kisialiou is deterministic and handled by Rust
            action = 0
            obs, _, done, truncated, info = env.step(action)
            step_info = info.get("step_info", {}) if isinstance(info, dict) else {}
            step_ms = float(step_info.get("operator_elapsed_ms", step_info.get("elapsed_ms", 0.0)) or 0.0)
            restart_elapsed_ms += step_ms
            solver_elapsed_ms_total += step_ms
            steps += 1

        stats = env.get_episode_statistics()
        env.close()

        restart_initial = float(stats.get("initial_cost") or 0.0)
        restart_best = float(stats.get("best_cost") or restart_initial)
        restart_initial_vessels = stats.get("initial_num_vessels_used")
        restart_best_vessels = stats.get("best_num_vessels_used")
        if initial_cost is None:
            initial_cost = restart_initial
        best_candidates.append(restart_best)

        elapsed_seq = stats.get("elapsed_seconds") or []
        restart_solver_runtime = float(elapsed_seq[-1]) if elapsed_seq else (restart_elapsed_ms / 1000.0)
        restart_wall_runtime = time.perf_counter() - restart_wall_start
        total_solver_runtime_seconds += restart_solver_runtime
        total_wall_runtime_seconds += restart_wall_runtime
        ttb_solver, ttb_iters = _time_to_best(stats)
        restart_records.append(
            {
                "restart_id": restart_idx,
                "seed": seed,
                "restart_seed": restart_seed,
                "solver_runtime_seconds": restart_solver_runtime,
                "wall_runtime_seconds": restart_wall_runtime,
                "time_to_best_solver_seconds": ttb_solver,
                "iterations_to_best": ttb_iters,
                "initial_cost": restart_initial,
                "best_cost": restart_best,
                "initial_num_vessels_used": restart_initial_vessels,
                "best_num_vessels_used": restart_best_vessels,
                "iterations": int(stats.get("total_iterations", steps)),
                "dataset_path": dataset_path,
                "mode_label": mode_label,
                "size_label": size_label,
                "algorithm_mode": "kisialiou",
            }
        )

        restart_idx += 1
        if target_runtime_seconds is None and reference_restarts is None:
            break

    if initial_cost is None:
        initial_cost = 0.0
    global_best = min(best_candidates) if best_candidates else 0.0

    return EpisodeSummary(
        initial_cost=float(initial_cost),
        best_cost=float(global_best),
        solver_runtime_seconds=float(total_solver_runtime_seconds),
        wall_runtime_seconds=float(total_wall_runtime_seconds),
    ), restart_records


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    sizes = parse_size_list(args.sizes if args.sizes else args.size)
    if not sizes:
        raise RuntimeError("No dataset sizes were provided")

    eval_seeds = parse_seed_list(args.eval_seeds)
    eval_iteration_cap = int(args.eval_max_iterations) if args.eval_max_iterations is not None else int(args.max_iterations)
    if (
        args.eval_runtime_mode == "match_kisialiou"
        and args.eval_max_iterations is None
        and args.eval_fixed_restarts is None
    ):
        eval_iteration_cap = max(eval_iteration_cap, 200000)

    # Allow the caller to provide an explicit run id via environment (useful for
    # grouping or consistent timestamps when calling from shell wrappers).
    env_run_id = None
    try:
        import os

        env_run_id = os.environ.get("RL_RUN_ID")
    except Exception:
        env_run_id = None

    if env_run_id:
        run_id = env_run_id
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"rl_improvement_mvp_{ts}__{args.run_tag}"
    out_dir = repo / "output" / run_id
    log_dir = repo / "logs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manager = GeneratedDatasetManager()
    train_paths: List[str] = []
    test_paths: List[str] = []
    for size in sizes:
        splits = manager.prepare_size(size)
        train_paths.extend(splits.get("train", []))
        test_paths.extend(splits.get("test", [])[: args.datasets_per_size])

    if not train_paths or not test_paths:
        raise RuntimeError("Dataset manager returned empty train/test paths")

    if sizes != ["small"] and "small" in sizes:
        raise RuntimeError(
            "size_bucket (Kisialiou-calibrated destroy-operator variants) is fixed per training "
            "run and only correct for a single-size run; a joint run mixing 'small' with "
            "medium/large would need size_bucket resolved dynamically per-episode, which is not "
            "implemented. Train small separately, or extend size_bucket handling before mixing sizes."
        )
    size_bucket = "small" if sizes == ["small"] else "large"

    exact_command = _build_exact_command()
    reference_train_env = ALNSEnvironment(
        problem_instance=train_paths[0],
        max_iterations=int(args.max_iterations),
        seed=int(args.seed),
        problem_instance_paths=list(train_paths),
        problem_sampling_strategy="random",
        enable_operator_logging=False,
        action_module=args.action_module,
        state_module=args.state_module,
        reward_module=args.reward_module,
        algorithm_mode="baseline",
        chain_max_consecutive_improvements=args.chain_max_consecutive_improvements,
        anti_repeat_penalty=args.anti_repeat_penalty,
        size_bucket=size_bucket,
    )
    action_space_trace = _collect_action_space_trace(train_env=reference_train_env)
    reference_train_env.close()

    manifest = _build_run_manifest(
        args=args,
        repo=repo,
        run_id=run_id,
        out_dir=out_dir,
        log_dir=log_dir,
        eval_seeds=eval_seeds,
        eval_iteration_cap=eval_iteration_cap,
        train_paths=list(train_paths),
        test_paths=list(test_paths),
        exact_command=exact_command,
        action_space_trace=action_space_trace,
        training_operator_usage={},
    )
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_command_file(out_dir, exact_command)
    _write_command_file(log_dir, exact_command)

    model_base_path = out_dir / "model" / "ppo_rl_improvement_mvp"
    model_base_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_training:
        if not args.model_path:
            raise RuntimeError("--model-path is required when --skip-training is set")
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = (repo / model_path).resolve()
        if not model_path.exists():
            raise RuntimeError(f"Model path not found: {model_path}")
        model = PPO.load(str(model_path))
    else:
        model, vec_env = train_ppo_agent(
            total_timesteps=args.train_timesteps,
            learning_rate=3e-4,
            log_dir=str(log_dir / "train"),
            model_save_path=str(model_base_path),
            seed=args.seed,
            train_instance_paths=train_paths,
            max_iterations=args.max_iterations,
            sampling_strategy="random",
            action_module=args.action_module,
            state_module=args.state_module,
            reward_module=args.reward_module,
            enable_operator_logging=True,
            algorithm_mode="baseline",
            chain_max_consecutive_improvements=args.chain_max_consecutive_improvements,
            anti_repeat_penalty=args.anti_repeat_penalty,
            size_bucket=size_bucket,
        )
        vec_env.close()

    manifest["resolved"]["training_operator_usage"] = _collect_training_operator_usage(log_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.run_eval:
        return

    rows: List[Dict] = []
    restart_rows: List[Dict[str, object]] = []
    for dataset_path in test_paths:
        dataset_size_label = infer_dataset_size_label(dataset_path)
        for seed in eval_seeds:
            kisialiou_stats, kisialiou_restart_rows = run_kisialiou_episode(
                dataset_path=dataset_path,
                seed=seed,
                max_iterations=eval_iteration_cap,
                restart_max_iterations=args.eval_restart_max_iterations,
                log_dir=log_dir,
                mode_label="kisialiou",
                size_label=dataset_size_label,
                action_module=args.action_module,
                state_module=args.state_module,
                reward_module=args.reward_module,
                target_runtime_seconds=(
                    args.eval_runtime_budget_seconds
                    if args.eval_fixed_restarts is None and args.eval_runtime_mode == "match_kisialiou"
                    else None
                ),
                chain_max_consecutive_improvements=args.chain_max_consecutive_improvements,
                reference_restarts=(
                    max(1, int(args.eval_fixed_restarts))
                    if args.eval_fixed_restarts is not None
                    else (
                        max(1, int(args.eval_runtime_reference_restarts))
                        if args.eval_runtime_mode == "match_kisialiou"
                        else None
                    )
                ),
            )
            restart_rows.extend(
                {
                    **row,
                    "dataset_path": dataset_path,
                    "dataset": Path(dataset_path).name,
                    "size": dataset_size_label,
                    "seed": seed,
                    "mode": "kisialiou",
                }
                for row in kisialiou_restart_rows
            )

            runtime_target_seconds: Optional[float] = None
            if args.eval_runtime_mode == "match_kisialiou" and args.eval_fixed_restarts is None:
                runtime_target_seconds = max(float(kisialiou_stats.wall_runtime_seconds), 1e-6)

            rl_stats, snapshot, rl_restart_rows = run_policy_episode(
                model=model,
                dataset_path=dataset_path,
                seed=seed,
                max_iterations=eval_iteration_cap,
                restart_max_iterations=args.eval_restart_max_iterations,
                log_dir=log_dir,
                mode_label="rl_improvement_mvp",
                size_label=dataset_size_label,
                action_module=args.action_module,
                state_module=args.state_module,
                reward_module=args.reward_module,
                chain_max_consecutive_improvements=args.chain_max_consecutive_improvements,
                anti_repeat_penalty=args.anti_repeat_penalty,
                policy_deterministic=True,
                target_runtime_seconds=runtime_target_seconds,
                restart_limit=args.eval_fixed_restarts,
            )
            restart_rows.extend(
                {
                    **row,
                    "dataset_path": dataset_path,
                    "dataset": Path(dataset_path).name,
                    "size": dataset_size_label,
                    "seed": seed,
                    "mode": "rl_improvement_mvp",
                }
                for row in rl_restart_rows
            )

            baseline_stats, baseline_restart_rows = run_baseline_episode(
                dataset_path=dataset_path,
                seed=seed,
                max_iterations=eval_iteration_cap,
                restart_max_iterations=args.eval_restart_max_iterations,
                log_dir=log_dir,
                mode_label="baseline",
                size_label=dataset_size_label,
                initial_snapshots=snapshot,
                action_module=args.baseline_action_module,
                state_module=args.baseline_state_module,
                reward_module=args.reward_module,
                target_runtime_seconds=runtime_target_seconds,
                restart_limit=args.eval_fixed_restarts,
            )
            restart_rows.extend(
                {
                    **row,
                    "dataset_path": dataset_path,
                    "dataset": Path(dataset_path).name,
                    "size": dataset_size_label,
                    "seed": seed,
                    "mode": "baseline",
                }
                for row in baseline_restart_rows
            )

            baseline_gap = (
                ((baseline_stats.best_cost - rl_stats.best_cost) / baseline_stats.best_cost) * 100.0
                if baseline_stats.best_cost > 1e-12
                else 0.0
            )
            kisialiou_gap = (
                ((kisialiou_stats.best_cost - rl_stats.best_cost) / kisialiou_stats.best_cost) * 100.0
                if kisialiou_stats.best_cost > 1e-12
                else 0.0
            )
            baseline_vs_kisialiou_gap = (
                ((kisialiou_stats.best_cost - baseline_stats.best_cost) / kisialiou_stats.best_cost) * 100.0
                if kisialiou_stats.best_cost > 1e-12
                else 0.0
            )
            kisialiou_vs_baseline_gap = (
                ((baseline_stats.best_cost - kisialiou_stats.best_cost) / baseline_stats.best_cost) * 100.0
                if baseline_stats.best_cost > 1e-12
                else 0.0
            )

            restart_records_for_row = rl_restart_rows
            rows.append(
                {
                    "dataset_path": dataset_path,
                    "dataset": Path(dataset_path).name,
                    "size": dataset_size_label,
                    "seed": seed,
                    "mode": "rl_improvement_mvp",
                    "initial_cost": rl_stats.initial_cost,
                    "best_cost": rl_stats.best_cost,
                    "solver_runtime_seconds": rl_stats.solver_runtime_seconds,
                    "wall_runtime_seconds": rl_stats.wall_runtime_seconds,
                    "runtime_seconds": rl_stats.runtime_seconds,
                    "improvement_pct": rl_stats.improvement_pct,
                    "improvement_pct_per_wall_second": rl_stats.improvement_pct / max(rl_stats.wall_runtime_seconds, 1e-12),
                    "improvement_pct_per_solver_second": rl_stats.improvement_pct / max(rl_stats.solver_runtime_seconds, 1e-12),
                    "baseline_gap_pct": baseline_gap,
                    "kisialiou_gap_pct": kisialiou_gap,
                    "runtime_target_seconds": runtime_target_seconds if runtime_target_seconds is not None else float(eval_iteration_cap),
                    "time_to_best_solver_seconds": _episode_time_to_best(restart_records_for_row),
                    "initial_num_vessels_used": _episode_vessel_stats(restart_records_for_row)[0],
                    "best_num_vessels_used": _episode_vessel_stats(restart_records_for_row)[1],
                }
            )

            if args.stochastic_check:
                rl_stochastic_stats, _, rl_stochastic_restart_rows = run_policy_episode(
                    model=model,
                    dataset_path=dataset_path,
                    seed=seed,
                    max_iterations=eval_iteration_cap,
                    restart_max_iterations=args.eval_restart_max_iterations,
                    log_dir=log_dir,
                    mode_label="rl_improvement_mvp_stochastic",
                    size_label=dataset_size_label,
                    action_module=args.action_module,
                    state_module=args.state_module,
                    reward_module=args.reward_module,
                    chain_max_consecutive_improvements=args.chain_max_consecutive_improvements,
                    anti_repeat_penalty=args.anti_repeat_penalty,
                    policy_deterministic=False,
                    target_runtime_seconds=runtime_target_seconds,
                    restart_limit=args.eval_fixed_restarts,
                )
                restart_rows.extend(
                    {
                        **row,
                        "dataset_path": dataset_path,
                        "dataset": Path(dataset_path).name,
                        "size": dataset_size_label,
                        "seed": seed,
                        "mode": "rl_improvement_mvp_stochastic",
                    }
                    for row in rl_stochastic_restart_rows
                )
                baseline_gap_stochastic = (
                    ((baseline_stats.best_cost - rl_stochastic_stats.best_cost) / baseline_stats.best_cost) * 100.0
                    if baseline_stats.best_cost > 1e-12
                    else 0.0
                )
                kisialiou_gap_stochastic = (
                    ((kisialiou_stats.best_cost - rl_stochastic_stats.best_cost) / kisialiou_stats.best_cost) * 100.0
                    if kisialiou_stats.best_cost > 1e-12
                    else 0.0
                )
                restart_records_for_row = rl_stochastic_restart_rows
                rows.append(
                    {
                        "dataset_path": dataset_path,
                        "dataset": Path(dataset_path).name,
                        "size": dataset_size_label,
                        "seed": seed,
                        "mode": "rl_improvement_mvp_stochastic",
                        "initial_cost": rl_stochastic_stats.initial_cost,
                        "best_cost": rl_stochastic_stats.best_cost,
                        "solver_runtime_seconds": rl_stochastic_stats.solver_runtime_seconds,
                        "wall_runtime_seconds": rl_stochastic_stats.wall_runtime_seconds,
                        "runtime_seconds": rl_stochastic_stats.runtime_seconds,
                        "improvement_pct": rl_stochastic_stats.improvement_pct,
                        "improvement_pct_per_wall_second": rl_stochastic_stats.improvement_pct / max(rl_stochastic_stats.wall_runtime_seconds, 1e-12),
                        "improvement_pct_per_solver_second": rl_stochastic_stats.improvement_pct / max(rl_stochastic_stats.solver_runtime_seconds, 1e-12),
                        "baseline_gap_pct": baseline_gap_stochastic,
                        "kisialiou_gap_pct": kisialiou_gap_stochastic,
                        "runtime_target_seconds": runtime_target_seconds if runtime_target_seconds is not None else float(eval_iteration_cap),
                    "time_to_best_solver_seconds": _episode_time_to_best(restart_records_for_row),
                    "initial_num_vessels_used": _episode_vessel_stats(restart_records_for_row)[0],
                    "best_num_vessels_used": _episode_vessel_stats(restart_records_for_row)[1],
                    }
                )
            restart_records_for_row = baseline_restart_rows
            rows.append(
                {
                    "dataset_path": dataset_path,
                    "dataset": Path(dataset_path).name,
                    "size": dataset_size_label,
                    "seed": seed,
                    "mode": "baseline",
                    "initial_cost": baseline_stats.initial_cost,
                    "best_cost": baseline_stats.best_cost,
                    "solver_runtime_seconds": baseline_stats.solver_runtime_seconds,
                    "wall_runtime_seconds": baseline_stats.wall_runtime_seconds,
                    "runtime_seconds": baseline_stats.runtime_seconds,
                    "improvement_pct": baseline_stats.improvement_pct,
                    "improvement_pct_per_wall_second": baseline_stats.improvement_pct / max(baseline_stats.wall_runtime_seconds, 1e-12),
                    "improvement_pct_per_solver_second": baseline_stats.improvement_pct / max(baseline_stats.solver_runtime_seconds, 1e-12),
                    "baseline_gap_pct": 0.0,
                    "kisialiou_gap_pct": baseline_vs_kisialiou_gap,
                    "runtime_target_seconds": runtime_target_seconds if runtime_target_seconds is not None else float(eval_iteration_cap),
                    "time_to_best_solver_seconds": _episode_time_to_best(restart_records_for_row),
                    "initial_num_vessels_used": _episode_vessel_stats(restart_records_for_row)[0],
                    "best_num_vessels_used": _episode_vessel_stats(restart_records_for_row)[1],
                }
            )
            restart_records_for_row = kisialiou_restart_rows
            rows.append(
                {
                    "dataset_path": dataset_path,
                    "dataset": Path(dataset_path).name,
                    "size": dataset_size_label,
                    "seed": seed,
                    "mode": "kisialiou",
                    "initial_cost": kisialiou_stats.initial_cost,
                    "best_cost": kisialiou_stats.best_cost,
                    "solver_runtime_seconds": kisialiou_stats.solver_runtime_seconds,
                    "wall_runtime_seconds": kisialiou_stats.wall_runtime_seconds,
                    "runtime_seconds": kisialiou_stats.runtime_seconds,
                    "improvement_pct": kisialiou_stats.improvement_pct,
                    "improvement_pct_per_wall_second": kisialiou_stats.improvement_pct / max(kisialiou_stats.wall_runtime_seconds, 1e-12),
                    "improvement_pct_per_solver_second": kisialiou_stats.improvement_pct / max(kisialiou_stats.solver_runtime_seconds, 1e-12),
                    "baseline_gap_pct": kisialiou_vs_baseline_gap,
                    "kisialiou_gap_pct": 0.0,
                    "runtime_target_seconds": runtime_target_seconds if runtime_target_seconds is not None else float(eval_iteration_cap),
                    "time_to_best_solver_seconds": _episode_time_to_best(restart_records_for_row),
                    "initial_num_vessels_used": _episode_vessel_stats(restart_records_for_row)[0],
                    "best_num_vessels_used": _episode_vessel_stats(restart_records_for_row)[1],
                }
            )

    results_json = out_dir / "results.json"
    results_csv = out_dir / "results.csv"
    restart_csv = out_dir / "restart_results.csv"
    results_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if restart_rows:
        with restart_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(restart_rows[0].keys()))
            writer.writeheader()
            writer.writerows(restart_rows)

    by_mode: Dict[str, List[Dict]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)

    summary: Dict[str, Dict] = {}
    for mode, mode_rows in by_mode.items():
        total_improvement_pct = float(sum(float(r["improvement_pct"]) for r in mode_rows))
        total_wall_runtime = float(sum(float(r["wall_runtime_seconds"]) for r in mode_rows))
        total_solver_runtime = float(sum(float(r["solver_runtime_seconds"]) for r in mode_rows))
        improvement_per_wall_second = total_improvement_pct / max(total_wall_runtime, 1e-12)
        improvement_per_solver_second = total_improvement_pct / max(total_solver_runtime, 1e-12)
        baseline_gap_values = [float(r["baseline_gap_pct"]) for r in mode_rows if mode != "baseline"]
        kisialiou_gap_values = [float(r["kisialiou_gap_pct"]) for r in mode_rows if mode != "kisialiou"]
        summary[mode] = {
            "improvement_pct_per_wall_second": improvement_per_wall_second,
            "improvement_pct_per_solver_second": improvement_per_solver_second,
            "mean_baseline_gap_pct": float(np.mean(baseline_gap_values)) if baseline_gap_values else 0.0,
            "mean_kisialiou_gap_pct": float(np.mean(kisialiou_gap_values)) if kisialiou_gap_values else 0.0,
            "num_samples": len(mode_rows),
        }

    def _row_key(row: Dict[str, Any]) -> Optional[tuple[str, int]]:
        dataset_val = row.get("dataset")
        seed_val = row.get("seed")
        if dataset_val is None or seed_val is None:
            return None
        try:
            return str(dataset_val), int(seed_val)
        except (TypeError, ValueError):
            return None

    rl_rows = [row for row in rows if str(row.get("mode")) == "rl_improvement_mvp"]
    baseline_lookup: Dict[tuple[str, int], Dict] = {}
    kisialiou_lookup: Dict[tuple[str, int], Dict] = {}
    for row in rows:
        key = _row_key(row)
        if key is None:
            continue
        mode = str(row.get("mode"))
        if mode == "baseline":
            baseline_lookup[key] = row
        elif mode == "kisialiou":
            kisialiou_lookup[key] = row

    def _paired_record(comparator_lookup: Dict[tuple[str, int], Dict], key_name: str) -> Dict[str, float]:
        wins = losses = ties = 0
        gaps: List[float] = []
        for row in rl_rows:
            key = _row_key(row)
            if key is None:
                continue
            other = comparator_lookup.get(key)
            if other is None:
                continue
            rl_best = float(row.get("best_cost", 0.0))
            other_best = float(other.get("best_cost", 0.0))
            diff = rl_best - other_best
            if abs(diff) <= 1e-9:
                ties += 1
            elif diff < 0.0:
                wins += 1
            else:
                losses += 1
            if other_best > 1e-12:
                gaps.append(((other_best - rl_best) / other_best) * 100.0)
        total = wins + losses + ties
        return {
            f"wins_vs_{key_name}": wins,
            f"losses_vs_{key_name}": losses,
            f"ties_vs_{key_name}": ties,
            f"win_rate_vs_{key_name}": (wins / total) if total > 0 else 0.0,
            f"median_gap_vs_{key_name}_pct": statistics.median(gaps) if gaps else 0.0,
        }

    if "rl_improvement_mvp" in summary:
        summary["rl_improvement_mvp"].update(_paired_record(baseline_lookup, "baseline"))
        summary["rl_improvement_mvp"].update(_paired_record(kisialiou_lookup, "kisialiou"))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "RL Improvement-only MVP Report",
        "==============================",
        f"run_id: {run_id}",
        f"sizes: {', '.join(sizes)}",
        f"datasets_per_size: {args.datasets_per_size}",
        f"eval_seeds: {eval_seeds}",
        f"action_module: {args.action_module}",
        f"state_module: {args.state_module}",
        f"baseline_action_module: {args.baseline_action_module}",
        f"baseline_state_module: {args.baseline_state_module}",
        f"baseline_policy: {args.baseline_action_module}",
        f"chain_max_consecutive_improvements: {args.chain_max_consecutive_improvements}",
        f"anti_repeat_penalty: {args.anti_repeat_penalty}",
        f"stochastic_check: {args.stochastic_check}",
        f"eval_runtime_mode: {args.eval_runtime_mode}",
        f"eval_iteration_cap: {eval_iteration_cap}",
        f"eval_restart_max_iterations: {args.eval_restart_max_iterations}",
        f"eval_runtime_reference_restarts: {args.eval_runtime_reference_restarts}",
        f"eval_runtime_budget_seconds: {args.eval_runtime_budget_seconds}",
        f"eval_fixed_restarts: {args.eval_fixed_restarts}",
        f"run_manifest: {manifest_path}",
        "",
        "Metrics:",
        "- improvement_pct_per_wall_second",
        "- improvement_pct_per_solver_second",
        "- mean_baseline_gap_pct",
        "- mean_kisialiou_gap_pct",
        "",
    ]
    for mode, stats in sorted(summary.items()):
        report_lines.append(f"- {mode}:")
        report_lines.append(f"    improvement_pct_per_wall_second: {stats['improvement_pct_per_wall_second']:.6f}")
        report_lines.append(f"    improvement_pct_per_solver_second: {stats['improvement_pct_per_solver_second']:.6f}")
        report_lines.append(f"    mean_baseline_gap_pct: {stats['mean_baseline_gap_pct']:.6f}")
        report_lines.append(f"    mean_kisialiou_gap_pct: {stats['mean_kisialiou_gap_pct']:.6f}")
        report_lines.append(f"    num_samples: {stats['num_samples']}")
        if mode == "rl_improvement_mvp":
            report_lines.append(f"    wins_vs_baseline: {stats.get('wins_vs_baseline', 0)}")
            report_lines.append(f"    losses_vs_baseline: {stats.get('losses_vs_baseline', 0)}")
            report_lines.append(f"    ties_vs_baseline: {stats.get('ties_vs_baseline', 0)}")
            report_lines.append(f"    win_rate_vs_baseline: {stats.get('win_rate_vs_baseline', 0.0):.6f}")
            report_lines.append(f"    median_gap_vs_baseline_pct: {stats.get('median_gap_vs_baseline_pct', 0.0):.6f}")
            report_lines.append(f"    wins_vs_kisialiou: {stats.get('wins_vs_kisialiou', 0)}")
            report_lines.append(f"    losses_vs_kisialiou: {stats.get('losses_vs_kisialiou', 0)}")
            report_lines.append(f"    ties_vs_kisialiou: {stats.get('ties_vs_kisialiou', 0)}")
            report_lines.append(f"    win_rate_vs_kisialiou: {stats.get('win_rate_vs_kisialiou', 0.0):.6f}")
            report_lines.append(f"    median_gap_vs_kisialiou_pct: {stats.get('median_gap_vs_kisialiou_pct', 0.0):.6f}")

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Run complete: {run_id}")
    print(f"Results: {results_json}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
