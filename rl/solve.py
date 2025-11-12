"""CLI entrypoint for solving a single PSVPP instance with a trained policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from stable_baselines3 import PPO

from rl.cli.common import ensure_dir, get_config_value, load_config
from rl.experiment import find_manifest_for_model, load_manifest
from rl.train_alns_rl import run_episode_with_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a single PSVPP instance using a trained RL policy")
    parser.add_argument("--model", required=True, help="Path to trained model .zip file")
    parser.add_argument("--instance", required=True, help="Path to processed dataset directory")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config containing defaults")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max ALNS iterations")
    parser.add_argument("--seed", type=int, default=None, help="Episode seed")
    parser.add_argument("--output-dir", default=None, help="Folder to write solution summary")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions (default deterministic)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    model_path = Path(args.model)
    manifest_path = find_manifest_for_model(model_path)
    manifest: Optional[Dict[str, Any]] = None
    if manifest_path:
        manifest = load_manifest(manifest_path)

    max_iterations = args.max_iterations or get_config_value(config, ("train", "max_iterations"), 100)
    if manifest:
        max_iterations = args.max_iterations or manifest.get("training", {}).get("max_iterations", max_iterations)

    default_seed = get_config_value(config, ("train", "seed"), 42)
    if manifest:
        default_seed = manifest.get("training", {}).get("seed", default_seed)
    seed = args.seed if args.seed is not None else default_seed

    manifest_eval = manifest.get("evaluation", {}) if manifest else {}
    base_det = manifest_eval.get("deterministic") if manifest_eval else None
    if base_det is None:
        base_det = get_config_value(config, ("evaluation", "deterministic"), True)
    deterministic = bool(base_det) and not args.stochastic

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif manifest:
        output_dir = Path(manifest.get("run_dir", model_path.parent)) / "solve"
    else:
        output_dir = Path(get_config_value(config, ("logging", "base_dir"), "runs")) / "solve"
    ensure_dir(output_dir)

    model = PPO.load(args.model)
    stats = run_episode_with_policy(
        policy_model=model,
        problem_path=args.instance,
        seed=seed,
        max_iterations=max_iterations,
        deterministic=deterministic,
    )

    summary: Dict[str, Any] = {
        "model": str(Path(args.model).resolve()),
        "instance": str(Path(args.instance).resolve()),
        "seed": seed,
        "max_iterations": max_iterations,
        "deterministic": deterministic,
        "best_cost": stats.get("best_cost"),
        "final_cost": stats.get("final_cost"),
        "iterations": stats.get("iterations"),
    }
    if manifest_path:
        summary["manifest"] = str(Path(manifest_path).resolve())

    summary_path = output_dir / f"solution_{Path(args.instance).name}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print("[DONE] Solution summary written to", summary_path)


if __name__ == "__main__":  # pragma: no cover
    main()
