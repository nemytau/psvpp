"""CLI entrypoint for model-vs-baseline comparisons and diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from stable_baselines3 import PPO

from rl.cli.common import ensure_dir, get_config_value, load_config
from rl.experiment import (
    ExperimentManager,
    find_manifest_for_model,
    load_manifest,
)
from rl.registries import DEFAULT_ACTION_KEY, DEFAULT_REWARD_KEY, DEFAULT_STATE_KEY
from rl.train_alns_rl import compare_model_against_baseline, prepare_dataset_splits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a trained RL policy against baselines on PSVPP instances")
    parser.add_argument("--model", required=True, help="Path to trained model .zip file")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config")
    parser.add_argument("--dataset-size", default=None, help="Dataset split size, e.g. small")
    parser.add_argument("--output-dir", default=None, help="Directory to store comparison artefacts")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Seeds used for evaluation episodes")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max ALNS iterations per episode")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions (default true)")
    parser.add_argument("--stochastic", action="store_true", help="Force stochastic rollout")
    parser.add_argument("--action-module", default=None, help="Registry key for action space implementation")
    parser.add_argument("--state-module", default=None, help="Registry key for state encoder implementation")
    parser.add_argument("--reward-module", default=None, help="Registry key for reward function implementation")
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
    manifest_seeds = manifest_eval.get("seeds") if manifest_eval else None
    default_seeds: List[int] = list(get_config_value(config, ("evaluation", "seeds"), [42]))
    seeds = args.seeds if args.seeds else list(manifest_seeds) if manifest_seeds else default_seeds

    manifest_det = manifest_eval.get("deterministic") if manifest_eval else None
    base_det = get_config_value(config, ("evaluation", "deterministic"), True)
    default_det = base_det if manifest_det is None else bool(manifest_det)
    if args.stochastic:
        deterministic = False
    elif args.deterministic:
        deterministic = True
    else:
        deterministic = default_det

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif manifest:
        output_dir = Path(manifest.get("artifacts", {}).get("comparison_dir", manifest.get("run_dir", model_path.parent)))
    else:
        output_dir = Path(get_config_value(config, ("logging", "base_dir"), "runs")) / "evaluate"
    ensure_dir(output_dir)

    print(f"[INFO] Loading model from {args.model}")
    model = PPO.load(args.model)

    manifest_modules = (manifest or {}).get("training", {}).get("modules", {}) if manifest else {}
    action_module = args.action_module or manifest_modules.get("action", {}).get("key") or get_config_value(config, ("modules", "action"), DEFAULT_ACTION_KEY)
    state_module = args.state_module or manifest_modules.get("state", {}).get("key") or get_config_value(config, ("modules", "state"), DEFAULT_STATE_KEY)
    reward_module = args.reward_module or manifest_modules.get("reward", {}).get("key") or get_config_value(config, ("modules", "reward"), DEFAULT_REWARD_KEY)

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
        test_paths = splits.get("test", []) or splits.get("val", [])
    if not test_paths:
        raise RuntimeError("No dataset instances available for evaluation")

    compare_model_against_baseline(
        model=model,
        problem_paths=test_paths,
        seeds=seeds,
        max_iterations=max_iterations,
        output_dir=str(output_dir),
        deterministic=deterministic,
        action_module=action_module,
        state_module=state_module,
        reward_module=reward_module,
    )
    print("[DONE] Comparison outputs saved to", output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
