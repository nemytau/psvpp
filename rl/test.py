"""CLI entrypoint for evaluating a trained model on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl.cli.common import (
    ensure_dir,
    get_config_value,
    load_config,
    save_json,
)
from rl.experiment import (
    ExperimentManager,
    find_manifest_for_model,
    load_manifest,
)
from rl.train_alns_rl import (
    compare_with_baseline,
    evaluate_trained_model,
    prepare_dataset_splits,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ALNS RL model on the test split")
    parser.add_argument("--model", required=True, help="Path to trained model .zip file")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config overriding defaults")
    parser.add_argument("--dataset-size", default=None, help="Dataset split size, e.g. small")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max ALNS iterations per episode")
    parser.add_argument("--output-dir", default=None, help="Directory to write evaluation artefacts")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (default true)")
    parser.add_argument("--stochastic", action="store_true", help="Force stochastic evaluation (overrides deterministic)")
    parser.add_argument("--include-baseline", action="store_true", help="Compute random-policy baseline metrics")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    model_path = Path(args.model)
    manifest_path = find_manifest_for_model(model_path)
    manifest: Optional[Dict[str, Any]] = None
    if manifest_path:
        manifest = load_manifest(manifest_path)

    dataset_size = args.dataset_size or get_config_value(config, ("dataset", "size"), "small")
    if manifest:
        dataset_size = manifest.get("dataset", {}).get("size", dataset_size)

    max_iterations = args.max_iterations or get_config_value(config, ("train", "max_iterations"), 100)
    if manifest:
        max_iterations = args.max_iterations or manifest.get("training", {}).get("max_iterations", max_iterations)

    manifest_eval = manifest.get("evaluation", {}) if manifest else {}
    manifest_det = manifest_eval.get("deterministic")
    base_deterministic = get_config_value(config, ("evaluation", "deterministic"), True)
    default_deterministic = base_deterministic if manifest_det is None else bool(manifest_det)
    if args.stochastic:
        deterministic = False
    elif args.deterministic:
        deterministic = True
    else:
        deterministic = default_deterministic

    manifest_seeds = manifest_eval.get("seeds") if manifest_eval else None
    seeds = list(manifest_seeds) if manifest_seeds else list(get_config_value(config, ("evaluation", "seeds"), [42]))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif manifest:
        output_dir = Path(manifest.get("run_dir", model_path.parent)) / "test"
    else:
        output_dir = Path(get_config_value(config, ("logging", "base_dir"), "runs")) / "test"

    ensure_dir(output_dir)

    test_paths: List[str] = []
    if manifest:
        test_split = manifest.get("dataset", {}).get("splits", {}).get("test")
        if test_split and test_split.get("paths"):
            test_paths = list(test_split["paths"])
            if not ExperimentManager.verify_split_integrity(test_split):
                print("[WARN] Manifest test split hash mismatch; recomputing using current datasets")
                test_paths = []
    if not test_paths:
        splits = prepare_dataset_splits(dataset_size)
        test_paths = splits.get("test", [])
    if not test_paths:
        raise RuntimeError("No test instances available for evaluation")

    mean_reward, std_reward, details = evaluate_trained_model(
        model_path=args.model,
        problem_paths=test_paths,
        n_eval_episodes=len(test_paths),
        deterministic=deterministic,
        output_dir=str(output_dir / "evaluation"),
        max_iterations=max_iterations,
    )

    summary: Dict[str, Any] = {
        "model": str(Path(args.model).resolve()),
        "dataset_size": dataset_size,
        "max_iterations": max_iterations,
        "deterministic": deterministic,
        "seeds": seeds,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "details": details,
    }
    if manifest_path:
        summary["manifest"] = str(Path(manifest_path).resolve())
        summary.setdefault("test_paths", test_paths)

    if args.include_baseline:
        baseline_mean, baseline_improvement, baseline_details = compare_with_baseline(
            problem_paths=test_paths,
            max_iterations=max_iterations,
            output_dir=str(output_dir / "baseline_random"),
        )
        summary["baseline"] = {
            "mean_reward": baseline_mean,
            "mean_improvement": baseline_improvement,
            "details": baseline_details,
        }

    save_json(summary, output_dir / "evaluation_summary.json")
    print("[DONE] Test evaluation artefacts written to", output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
