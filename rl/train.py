"""CLI entrypoint for training PPO on the ALNS environment."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

from rl.cli.common import (
    get_config_value,
    load_config,
)
from rl.experiment import ExperimentManager, build_experiment_id
from rl.registries import DEFAULT_ACTION_KEY, DEFAULT_REWARD_KEY, DEFAULT_STATE_KEY
from rl.train_alns_rl import (
    compare_model_against_baseline,
    compare_with_baseline,
    evaluate_trained_model,
    prepare_dataset_splits,
    test_environment_manually,
    train_ppo_agent,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO policy for ALNS operator selection")
    parser.add_argument("--config", help="Path to YAML/JSON config file", default=None)
    parser.add_argument("--exp-name", help="Experiment name / run identifier", default=None)
    parser.add_argument("--dataset-size", help="Dataset split size (e.g., small)", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None, help="Number of PPO training timesteps")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max ALNS iterations per episode")
    parser.add_argument("--sampling-strategy", default=None, help="Instance sampling strategy (random/round_robin/seed)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log-dir", default=None, help="Base directory for run artefacts")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip pre-training environment smoke-test")
    parser.add_argument("--action-module", default=None, help="Registry key for action space implementation")
    parser.add_argument("--state-module", default=None, help="Registry key for state encoder implementation")
    parser.add_argument("--reward-module", default=None, help="Registry key for reward function implementation")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluations after training")
    return parser.parse_args()


def _resolve_parameters(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    timestamp = dt.datetime.now()
    timestamp_label = timestamp.strftime("%Y%m%d_%H%M%S")

    dataset_size = args.dataset_size or get_config_value(config, ("dataset", "size"), "small")
    total_timesteps = args.total_timesteps or get_config_value(config, ("train", "total_timesteps"), 50_000)
    max_iterations = args.max_iterations or get_config_value(config, ("train", "max_iterations"), 100)
    sampling_strategy = args.sampling_strategy or get_config_value(config, ("train", "sampling_strategy"), "random")
    seed = args.seed if args.seed is not None else get_config_value(config, ("train", "seed"), 42)
    learning_rate = float(get_config_value(config, ("train", "learning_rate"), 3e-4))
    algo = get_config_value(config, ("train", "algo"), "ppo")

    action_module = args.action_module or get_config_value(config, ("modules", "action"), DEFAULT_ACTION_KEY) or DEFAULT_ACTION_KEY
    state_module = args.state_module or get_config_value(config, ("modules", "state"), DEFAULT_STATE_KEY) or DEFAULT_STATE_KEY
    reward_module = args.reward_module or get_config_value(config, ("modules", "reward"), DEFAULT_REWARD_KEY) or DEFAULT_REWARD_KEY

    params: Dict[str, Any] = {
        "timestamp": timestamp,
        "timestamp_label": timestamp_label,
        "requested_exp_name": args.exp_name,
        "dataset_size": dataset_size,
        "total_timesteps": total_timesteps,
        "max_iterations": max_iterations,
        "sampling_strategy": sampling_strategy,
        "seed": seed,
        "log_base": Path(args.log_dir) if args.log_dir else Path(get_config_value(config, ("logging", "base_dir"), "runs")),
        "deterministic_eval": bool(get_config_value(config, ("evaluation", "deterministic"), True)),
        "eval_seeds": list(get_config_value(config, ("evaluation", "seeds"), [42])),
        "action_module": action_module,
        "state_module": state_module,
        "reward_module": reward_module,
        "algo": algo,
        "learning_rate": learning_rate,
    }

    return params


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    params = _resolve_parameters(args, config)

    default_exp_id = build_experiment_id(
        timestamp=params["timestamp_label"],
        dataset=params["dataset_size"],
        algo=params.get("algo", "ppo"),
        action_key=params["action_module"],
        state_key=params["state_module"],
        reward_key=params["reward_module"],
        seed=params["seed"],
    )
    exp_id = params.get("requested_exp_name") or default_exp_id
    params["exp_id"] = exp_id

    manager = ExperimentManager(
        exp_id=exp_id,
        base_dir=params["log_base"],
        created_at=params["timestamp"],
    )

    run_dir: Path = manager.paths.root
    params["run_dir"] = run_dir
    params["log_dir"] = manager.paths.tensorboard
    params["model_path"] = manager.paths.model_base
    params["evaluation_dir"] = manager.paths.evaluation
    params["baseline_dir"] = manager.paths.baseline
    params["comparison_dir"] = manager.paths.comparison
    params["artifacts_dir"] = manager.paths.artifacts

    # Persist resolved configuration/metadata
    def _serialise(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dt.datetime):
            return value.isoformat()
        if isinstance(value, (list, tuple)):
            return [_serialise(item) for item in value]
        return value

    final_config: Dict[str, Any] = {
        "config_path": args.config,
        "exp_id": exp_id,
        "timestamp": params["timestamp"].isoformat(),
        "parameters": {
            key: _serialise(value)
            for key, value in params.items()
            if key not in {"timestamp", "timestamp_label", "requested_exp_name"}
        },
    }
    manager.snapshot_config(final_config, Path(args.config) if args.config else None)

    # Prepare datasets
    dataset_splits = prepare_dataset_splits(params["dataset_size"])
    train_paths = dataset_splits.get("train", [])
    test_paths = dataset_splits.get("test", [])
    if not train_paths or not test_paths:
        raise RuntimeError("Dataset splits did not yield both train and test sets")

    if not args.skip_env_check:
        first_instance = train_paths[0]
        if not test_environment_manually(first_instance):
            raise RuntimeError("Environment smoke-test failed")

    # Train the PPO agent
    print("[INFO] Starting training run...")
    model, vec_env = train_ppo_agent(
        total_timesteps=params["total_timesteps"],
        learning_rate=params["learning_rate"],
        log_dir=str(params["log_dir"]),
        model_save_path=str(params["model_path"]),
        seed=params["seed"],
        train_instance_paths=train_paths,
        max_iterations=params["max_iterations"],
        sampling_strategy=params["sampling_strategy"],
        action_module=params["action_module"],
        state_module=params["state_module"],
        reward_module=params["reward_module"],
    )

    model_zip = manager.paths.model_zip

    # Collect module metadata from the underlying environment (if available)
    module_versions: Dict[str, Any] = {}
    if vec_env.envs:
        env = vec_env.envs[0]
        module_versions = getattr(env, "module_versions", {})
    vec_env.close()

    # Evaluate on held-out test set
    print("[INFO] Evaluating trained agent on test split...")
    eval_mean, eval_std, eval_details = evaluate_trained_model(
        model_path=str(model_zip),
        problem_paths=test_paths,
        n_eval_episodes=len(test_paths),
        deterministic=params["deterministic_eval"],
        output_dir=str(params["evaluation_dir"]),
        max_iterations=params["max_iterations"],
    )

    baseline_stats: Optional[Dict[str, Any]] = None
    if not args.skip_baseline:
        baseline_mean, baseline_improvement, baseline_details = compare_with_baseline(
            problem_paths=test_paths,
            max_iterations=params["max_iterations"],
            output_dir=str(params["baseline_dir"]),
        )
        baseline_stats = {
            "mean_reward": baseline_mean,
            "mean_improvement": baseline_improvement,
            "details": baseline_details,
            "output_dir": str(params["baseline_dir"]),
        }

        comparison_results = compare_model_against_baseline(
            model=model,
            problem_paths=test_paths,
            seeds=params["eval_seeds"],
            max_iterations=params["max_iterations"],
            output_dir=str(params["comparison_dir"]),
            deterministic=params["deterministic_eval"],
        )
    else:
        comparison_results = []

    # Persist manifest
    evaluation_summary: Dict[str, Any] = {
        "mean_reward": eval_mean,
        "std_reward": eval_std,
        "details": eval_details,
        "output_dir": str(params["evaluation_dir"]),
    }
    manifest = manager.compose_manifest(
        params=params,
        train_paths=train_paths,
        test_paths=test_paths,
        module_versions=module_versions,
        evaluation=evaluation_summary,
        baseline=baseline_stats,
        comparison=comparison_results,
    )
    manager.write_manifest(manifest)

    print("[DONE] Training pipeline completed successfully.")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
