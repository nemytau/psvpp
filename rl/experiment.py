"""Support for experiment IDs, manifests, and runtime metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

try:  # Python >=3.8
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python <3.8 fallback
    import importlib_metadata  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _slugify(component: str) -> str:
    text = component.strip().lower()
    if not text:
        return "default"
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "default"


def build_experiment_id(
    timestamp: str,
    dataset: str,
    algo: str,
    action_key: str,
    state_key: str,
    reward_key: str,
    seed: int,
) -> str:
    """Create a reproducible experiment identifier."""

    dataset_slug = _slugify(dataset)
    algo_slug = _slugify(algo)
    action_slug = _slugify(action_key)
    state_slug = _slugify(state_key)
    reward_slug = _slugify(reward_key)
    return (
        f"{timestamp}__{dataset_slug}__{algo_slug}__"
        f"A-{action_slug}__S-{state_slug}__R-{reward_slug}__seed{seed}"
    )


def _resolve_path(path_str: str, repo_root: Path = REPO_ROOT) -> Path:
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj
    return (repo_root / path_obj).resolve()


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(_hash_file(file_path).encode("utf-8"))
    return digest.hexdigest()


def hash_paths(paths: Sequence[str], repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    """Compute SHA256 hashes for each path in ``paths``."""

    hashes: Dict[str, str] = {}
    for raw_path in paths:
        resolved = _resolve_path(raw_path, repo_root=repo_root)
        if resolved.is_dir():
            hashes[raw_path] = _hash_directory(resolved)
        elif resolved.is_file():
            hashes[raw_path] = _hash_file(resolved)
        else:
            digest = hashlib.sha256()
            digest.update(str(resolved).encode("utf-8"))
            hashes[raw_path] = digest.hexdigest()
    return hashes


def combine_hashes(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(hashes):
        digest.update(key.encode("utf-8"))
        digest.update(hashes[key].encode("utf-8"))
    return digest.hexdigest()


def load_manifest(path: Path | str) -> Dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_repo_root() -> Path:
    return REPO_ROOT


def find_manifest_for_model(model_path: Path | str) -> Optional[Path]:
    model_file = Path(model_path)
    if model_file.is_dir():
        candidate = model_file / "manifest.json"
        return candidate if candidate.exists() else None
    run_dir = model_file.parent
    candidate = run_dir / "manifest.json"
    return candidate if candidate.exists() else None


@dataclass
class ExperimentPaths:
    root: Path

    @property
    def tensorboard(self) -> Path:
        return self.root / "tb"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def baseline(self) -> Path:
        return self.root / "baseline_random"

    @property
    def comparison(self) -> Path:
        return self.root / "model_vs_baseline"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def convergence(self) -> Path:
        return self.root / "convergence"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def config_snapshot(self) -> Path:
        return self.root / "config.yaml"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def model_base(self) -> Path:
        return self.root / "model"

    @property
    def model_zip(self) -> Path:
        return self.root / "model.zip"


class ExperimentManager:
    """Manage experiment directory layout and manifest creation."""

    def __init__(
        self,
        exp_id: str,
        base_dir: Path,
        created_at: Optional[datetime] = None,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self.exp_id = exp_id
        self.base_dir = base_dir
        self.repo_root = repo_root
        self.created_at = created_at or datetime.utcnow()
        self.paths = ExperimentPaths(self.base_dir / self.exp_id)
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        for directory in (
            self.paths.root,
            self.paths.tensorboard,
            self.paths.evaluation,
            self.paths.baseline,
            self.paths.comparison,
            self.paths.artifacts,
            self.paths.convergence,
            self.paths.config_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def snapshot_config(
        self,
        resolved_config: Dict[str, Any],
        source_config_path: Optional[Path] = None,
    ) -> None:
        with self.paths.config_snapshot.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved_config, handle, sort_keys=False)

        if source_config_path and source_config_path.exists():
            target_path = self.paths.config_dir / source_config_path.name
            if target_path.resolve() != source_config_path.resolve():
                content = source_config_path.read_text(encoding="utf-8")
                target_path.write_text(content, encoding="utf-8")

    def _git_metadata(self) -> Dict[str, Any]:
        def _run(*args: str) -> Optional[str]:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=self.repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()
            except Exception:
                return None

        commit = _run("rev-parse", "HEAD")
        branch = _run("rev-parse", "--abbrev-ref", "HEAD")
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            dirty = bool(status.stdout.strip())
        except Exception:
            dirty = None

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }

    def _runtime_metadata(self) -> Dict[str, Any]:
        packages = {}
        for name in (
            "stable-baselines3",
            "gymnasium",
            "torch",
            "numpy",
        ):
            try:
                packages[name] = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                continue

        rust_version: Optional[str] = None
        try:
            import rust_alns_py  # type: ignore

            rust_version = getattr(rust_alns_py, "__version__", None)
        except Exception:  # pragma: no cover - optional dependency
            rust_version = None

        return {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "packages": packages,
            "rust_alns_py": rust_version,
        }

    def _dataset_section(
        self,
        dataset_size: str,
        train_paths: Sequence[str],
        test_paths: Sequence[str],
    ) -> Dict[str, Any]:
        train_hashes = hash_paths(train_paths, repo_root=self.repo_root)
        test_hashes = hash_paths(test_paths, repo_root=self.repo_root)

        train_combined = combine_hashes(train_hashes)
        test_combined = combine_hashes(test_hashes)
        overall = combine_hashes({
            f"train::{key}": value for key, value in train_hashes.items()
        } | {
            f"test::{key}": value for key, value in test_hashes.items()
        })

        return {
            "size": dataset_size,
            "splits": {
                "train": {
                    "paths": list(train_paths),
                    "hashes": train_hashes,
                    "combined_hash": train_combined,
                    "count": len(train_paths),
                },
                "test": {
                    "paths": list(test_paths),
                    "hashes": test_hashes,
                    "combined_hash": test_combined,
                    "count": len(test_paths),
                },
            },
            "combined_hash": overall,
        }

    def compose_manifest(
        self,
        params: Dict[str, Any],
        train_paths: Sequence[str],
        test_paths: Sequence[str],
        module_versions: Mapping[str, Any],
        evaluation: Dict[str, Any],
        baseline: Optional[Dict[str, Any]],
        comparison: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        modules = {
            "action": {
                "key": params.get("action_module"),
                "version": module_versions.get("action"),
            },
            "state": {
                "key": params.get("state_module"),
                "version": module_versions.get("state"),
            },
            "reward": {
                "key": params.get("reward_module"),
                "version": module_versions.get("reward"),
            },
        }

        manifest = {
            "exp_id": self.exp_id,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "run_dir": str(self.paths.root.resolve()),
            "git": self._git_metadata(),
            "environment": self._runtime_metadata(),
            "dataset": self._dataset_section(
                dataset_size=params.get("dataset_size", "unknown"),
                train_paths=train_paths,
                test_paths=test_paths,
            ),
            "training": {
                "algo": params.get("algo"),
                "total_timesteps": params.get("total_timesteps"),
                "learning_rate": params.get("learning_rate"),
                "max_iterations": params.get("max_iterations"),
                "sampling_strategy": params.get("sampling_strategy"),
                "seed": params.get("seed"),
                "algorithm_mode": params.get("algorithm_mode"),
                "modules": modules,
            },
            "evaluation": {
                "deterministic": params.get("deterministic_eval"),
                "seeds": params.get("eval_seeds"),
                "mean_reward": evaluation.get("mean_reward"),
                "std_reward": evaluation.get("std_reward"),
                "details": evaluation.get("details"),
                "output_dir": evaluation.get("output_dir"),
            },
            "baseline": baseline,
            "comparison": list(comparison),
            "artifacts": {
                "model": str(self.paths.model_zip),
                "tensorboard": str(self.paths.tensorboard),
                "evaluation_dir": str(self.paths.evaluation),
                "baseline_dir": str(self.paths.baseline),
                "comparison_dir": str(self.paths.comparison),
                "config": str(self.paths.config_snapshot),
            },
        }
        return manifest

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        with self.paths.manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    @staticmethod
    def verify_split_integrity(
        split_manifest: Mapping[str, Any],
        repo_root: Path = REPO_ROOT,
    ) -> bool:
        paths = split_manifest.get("paths") or []
        if not paths:
            return True
        expected = split_manifest.get("combined_hash")
        if not expected:
            return True
        computed = combine_hashes(hash_paths(paths, repo_root=repo_root))
        return computed == expected