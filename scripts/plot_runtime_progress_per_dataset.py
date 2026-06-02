#!/usr/bin/env python3
"""Per-dataset runtime plotting aggregated over restarts (interactive Plotly HTML)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ("rl_improvement_mvp", "baseline", "kisialiou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one runtime progress plot per dataset with restart-level aggregation "
            "for each mode."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", type=str, default=None, help="Run id under logs/<run_id>")
    group.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run tag suffix; resolves latest output/rl_improvement_mvp_*__<run_tag>",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes (default: rl_improvement_mvp,baseline,kisialiou)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/<run_id>/runtime_progress_per_dataset)",
    )
    parser.add_argument(
        "--max-time-points",
        type=int,
        default=500,
        help="Maximum time points per aggregated curve",
    )
    parser.add_argument(
        "--max-best-dot-points",
        type=int,
        default=5000,
        help="Maximum displayed points in best-cost dot panel per mode (close points are aggregated)",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def resolve_run_id(run_id: str | None, run_tag: str | None) -> str:
    if run_id:
        return run_id
    if not run_tag:
        raise ValueError("Either --run-id or --run-tag must be provided")

    # Prefer active experiments folder when resolving run tags, then fallback
    # to the main output directory (and finally archive).
    output_root_active = REPO_ROOT / "output" / "active"
    output_root = REPO_ROOT / "output"
    candidates = []
    if output_root_active.exists():
        candidates = sorted(
            output_root_active.glob(f"rl_improvement_mvp_*__{run_tag}"),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
    if not candidates and output_root.exists():
        candidates = sorted(
            output_root.glob(f"rl_improvement_mvp_*__{run_tag}"),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
    if not candidates:
        raise FileNotFoundError(
            f"No output run found for run tag '{run_tag}' under {output_root_active} or {output_root}"
        )
    return candidates[-1].name


def parse_modes(raw_modes: str) -> List[str]:
    modes = [item.strip() for item in raw_modes.split(",") if item.strip()]
    if not modes:
        raise ValueError("No modes provided")
    return modes


def discover_mode_files(log_dir: Path, mode: str) -> List[Path]:
    direct = sorted(log_dir.glob(f"operator_usage_{mode}__*.csv"))
    if direct:
        return direct
    nested = sorted((log_dir / "operator_usage").glob(f"operator_usage_{mode}__*.csv"))
    if nested:
        return nested
    return []


def _coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _dataset_from_value(raw: str | float | int | None) -> str:
    if raw is None:
        return "unknown"
    text = str(raw).strip()
    if not text:
        return "unknown"
    if "/" in text or "\\" in text:
        return Path(text).name
    return text


def _dataset_series(df: pd.DataFrame) -> pd.Series:
    if "instance_name" in df.columns:
        series = df["instance_name"].astype(str).str.strip()
        if (series != "").any():
            return series.map(_dataset_from_value)
    if "instance_id" in df.columns:
        return df["instance_id"].map(_dataset_from_value)
    return pd.Series(["unknown"] * len(df), index=df.index)


def _last_non_nan(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def _restart_key(df: pd.DataFrame) -> pd.Series:
    for candidate in ("restart_id", "episode_id"):
        if candidate in df.columns:
            return df[candidate].fillna(-1).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)


def _seed_from_log_filename(file_path: Path) -> int:
    match = re.search(r"__seed(\d+)\.csv$", file_path.name)
    if not match:
        return -1
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return -1


def load_mode_trajectories_from_logs(mode: str, files: Sequence[Path]) -> Dict[str, List[pd.DataFrame]]:
    per_dataset: Dict[str, List[pd.DataFrame]] = {}
    needed_columns = {
        "iteration",
        "cost_best",
        "cost_current",
        "operator_elapsed_ms",
        "elapsed_ms",
        "restart_id",
        "episode_id",
        "instance_name",
        "instance_id",
    }

    for file_path in sorted(files):
        seed_id = _seed_from_log_filename(file_path)
        try:
            raw = pd.read_csv(
                file_path,
                low_memory=False,
                usecols=lambda column: column in needed_columns,
            )
        except Exception as exc:
            warn(f"Skipping unreadable file {file_path}: {exc}")
            continue
        if raw.empty:
            continue

        _coerce_numeric(
            raw,
            [
                "iteration",
                "cost_best",
                "cost_current",
                "operator_elapsed_ms",
                "elapsed_ms",
                "restart_id",
                "episode_id",
            ],
        )

        required = ["iteration", "cost_best", "cost_current"]
        if any(column not in raw.columns for column in required):
            warn(f"Skipping file {file_path.name}: missing one of {required}")
            continue

        elapsed_col = "operator_elapsed_ms" if "operator_elapsed_ms" in raw.columns else "elapsed_ms"
        if elapsed_col not in raw.columns:
            warn(f"Skipping file {file_path.name}: missing elapsed column")
            continue

        local = raw
        local["__row_order"] = np.arange(len(local), dtype=int)
        local["__restart"] = _restart_key(local)
        local["__dataset"] = _dataset_series(local)

        grouped = (
            local.sort_values(["__dataset", "__restart", "iteration", "__row_order"])
            .groupby(["__dataset", "__restart", "iteration"], as_index=False, sort=True)
            .agg(
                elapsed_ms=(elapsed_col, "max"),
                cost_best=("cost_best", "min"),
                cost_current=("cost_current", _last_non_nan),
            )
        )
        if grouped.empty:
            continue

        for (dataset, restart_id), part in grouped.groupby(["__dataset", "__restart"], sort=True):
            trajectory = part.sort_values("iteration").reset_index(drop=True)
            trajectory["elapsed_ms"] = trajectory["elapsed_ms"].fillna(0.0).clip(lower=0.0)
            trajectory["elapsed_s"] = trajectory["elapsed_ms"].cumsum() / 1000.0
            trajectory["best_so_far"] = trajectory["cost_best"].cummin()
            trajectory["mode"] = mode
            trajectory["dataset"] = str(dataset)
            trajectory["seed_id"] = seed_id
            try:
                restart_id_int = int(float(str(restart_id)))
            except (TypeError, ValueError):
                restart_id_int = -1
            trajectory["restart_id"] = restart_id_int
            per_dataset.setdefault(str(dataset), []).append(
                trajectory[["elapsed_s", "best_so_far", "cost_current", "mode", "dataset", "seed_id", "restart_id"]]
            )

    return per_dataset


def load_mode_trajectories_from_results(run_id: str, mode: str) -> Dict[str, List[pd.DataFrame]]:
    results_path = REPO_ROOT / "output" / run_id / "results.csv"
    if not results_path.exists():
        return {}

    try:
        results = pd.read_csv(results_path)
    except Exception as exc:
        warn(f"Failed reading fallback results.csv for mode '{mode}': {exc}")
        return {}

    if results.empty or "mode" not in results.columns:
        return {}

    rows = results[results["mode"].astype(str) == mode].copy()
    if rows.empty:
        return {}

    _coerce_numeric(rows, ["runtime_seconds", "best_cost", "seed"])
    if "dataset" not in rows.columns:
        rows["dataset"] = "unknown"
    rows = rows.dropna(subset=["runtime_seconds", "best_cost"]).reset_index(drop=True)
    if rows.empty:
        return {}

    per_dataset: Dict[str, List[pd.DataFrame]] = {}
    for _, row in rows.iterrows():
        dataset = _dataset_from_value(row.get("dataset"))
        seed_id = int(row.get("seed", -1)) if not pd.isna(row.get("seed", np.nan)) else -1
        restart_id = seed_id
        trajectory = pd.DataFrame(
            {
                "elapsed_s": [float(max(0.0, row["runtime_seconds"]))],
                "best_so_far": [float(row["best_cost"])],
                "cost_current": [float(row["best_cost"])],
                "mode": [mode],
                "dataset": [dataset],
                "seed_id": [seed_id],
                "restart_id": [restart_id],
            }
        )
        per_dataset.setdefault(dataset, []).append(trajectory)
    return per_dataset


def _step_sample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sampled = np.full(len(grid), np.nan, dtype=float)
    if len(times) == 0:
        return sampled

    order = np.argsort(times)
    times_sorted = times[order]
    values_sorted = values[order]

    uniq_times, uniq_indices = np.unique(times_sorted, return_index=True)
    uniq_values = values_sorted[uniq_indices]

    idx = np.searchsorted(uniq_times, grid, side="right") - 1
    valid = (idx >= 0) & (grid <= uniq_times[-1])
    sampled[valid] = uniq_values[idx[valid]]
    return sampled


def _step_sample_with_carry(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step-sample a curve on a grid and carry edge values outside support.

    This keeps completed trajectories active (holding their terminal best value)
    and aligns early-time behavior by holding the first known value before the
    first logged timestamp.
    """
    sampled = _step_sample(times, values, grid)
    if len(times) == 0:
        return sampled

    order = np.argsort(times)
    times_sorted = times[order]
    values_sorted = values[order]

    if len(times_sorted) == 0:
        return sampled

    first_time = float(times_sorted[0])
    last_time = float(times_sorted[-1])
    first_value = float(values_sorted[0])
    last_value = float(values_sorted[-1])

    sampled[grid < first_time] = first_value
    sampled[grid > last_time] = last_value
    return sampled


def aggregate_mode_dataset(trajectories: Sequence[pd.DataFrame], max_time_points: int) -> pd.DataFrame:
    if not trajectories:
        return pd.DataFrame()

    max_elapsed = max(float(traj["elapsed_s"].max()) for traj in trajectories if not traj.empty)
    if max_elapsed <= 0:
        time_grid = np.array([0.0])
    else:
        longest = max(len(traj) for traj in trajectories)
        n_points = min(max(2, int(max_time_points)), max(2, longest))
        time_grid = np.linspace(0.0, max_elapsed, num=n_points)

    best_rows: List[np.ndarray] = []
    current_rows: List[np.ndarray] = []

    for traj in trajectories:
        if traj.empty:
            continue
        times = traj["elapsed_s"].to_numpy(dtype=float)
        best_vals = traj["best_so_far"].to_numpy(dtype=float)
        curr_vals = traj["cost_current"].to_numpy(dtype=float)

        best_rows.append(_step_sample_with_carry(times, best_vals, time_grid))
        current_rows.append(_step_sample_with_carry(times, curr_vals, time_grid))

    if not best_rows or not current_rows:
        return pd.DataFrame()

    best_matrix = np.vstack(best_rows)
    current_matrix = np.vstack(current_rows)
    counts = np.sum(~np.isnan(current_matrix), axis=0)
    valid = counts > 0
    if not np.any(valid):
        return pd.DataFrame()

    with np.errstate(all="ignore"):
        best_over_restarts = np.nanmin(best_matrix, axis=0)
        current_mean = np.nanmean(current_matrix, axis=0)

    # Defensive monotonicity: best-over-restarts should never increase.
    best_over_restarts = np.minimum.accumulate(best_over_restarts)

    return pd.DataFrame(
        {
            "elapsed_s": time_grid[valid],
            "best_over_restarts": best_over_restarts[valid],
            "current_mean": current_mean[valid],
            "active_restarts": counts[valid].astype(int),
        }
    )


def collect_best_points(trajectories: Sequence[pd.DataFrame]) -> pd.DataFrame:
    elapsed_chunks: List[np.ndarray] = []
    best_chunks: List[np.ndarray] = []
    for traj in trajectories:
        if traj.empty:
            continue
        elapsed = traj["elapsed_s"].to_numpy(dtype=float)
        best = traj["best_so_far"].to_numpy(dtype=float)
        valid = np.isfinite(elapsed) & np.isfinite(best)
        if not np.any(valid):
            continue
        elapsed_chunks.append(elapsed[valid])
        best_chunks.append(best[valid])
    if not elapsed_chunks:
        return pd.DataFrame(columns=["elapsed_s", "best_so_far", "count"])
    elapsed_all = np.concatenate(elapsed_chunks)
    best_all = np.concatenate(best_chunks)
    return pd.DataFrame(
        {
            "elapsed_s": elapsed_all,
            "best_so_far": best_all,
            "count": np.ones(elapsed_all.shape[0], dtype=int),
        }
    )


def aggregate_concatenated_best_evolution(
    trajectories: Sequence[pd.DataFrame],
    max_points: int,
) -> pd.DataFrame:
    if not trajectories:
        return pd.DataFrame(columns=["elapsed_s", "best_over_restarts"])

    non_empty = [traj for traj in trajectories if not traj.empty]
    if not non_empty:
        return pd.DataFrame(columns=["elapsed_s", "best_over_restarts"])

    by_seed: Dict[int, List[pd.DataFrame]] = {}
    for traj in non_empty:
        if "seed_id" in traj.columns and not traj["seed_id"].empty:
            try:
                seed_id = int(float(traj["seed_id"].iloc[0]))
            except (TypeError, ValueError):
                seed_id = -1
        else:
            seed_id = -1
        by_seed.setdefault(seed_id, []).append(traj)

    seed_curves: List[tuple[np.ndarray, np.ndarray]] = []
    for _, seed_trajs in sorted(by_seed.items(), key=lambda item: item[0]):
        ordered = sorted(
            seed_trajs,
            key=lambda traj: int(traj["restart_id"].iloc[0]) if "restart_id" in traj.columns else 0,
        )

        elapsed_offset = 0.0
        global_best = np.inf
        time_chunks: List[np.ndarray] = []
        best_chunks: List[np.ndarray] = []

        for traj in ordered:
            elapsed = traj["elapsed_s"].to_numpy(dtype=float)
            best = traj["best_so_far"].to_numpy(dtype=float)
            valid = np.isfinite(elapsed) & np.isfinite(best)
            if not np.any(valid):
                continue

            elapsed_valid = elapsed[valid]
            best_valid = best[valid]
            best_global = np.minimum.accumulate(np.minimum(best_valid, global_best))
            global_best = min(global_best, float(np.min(best_valid)))
            elapsed_concat = elapsed_valid + elapsed_offset

            time_chunks.append(elapsed_concat)
            best_chunks.append(best_global)
            elapsed_offset = float(elapsed_concat[-1])

        if not time_chunks:
            continue

        time_all = np.concatenate(time_chunks)
        best_all = np.concatenate(best_chunks)
        seed_curves.append((time_all, best_all))

    if not seed_curves:
        return pd.DataFrame(columns=["elapsed_s", "best_over_restarts"])

    max_elapsed = max(float(times[-1]) for times, _ in seed_curves if len(times) > 0)
    if max_elapsed <= 0:
        time_grid = np.array([0.0])
    else:
        longest = max(len(times) for times, _ in seed_curves)
        n_points = min(max(2, int(max_points)), max(2, longest))
        time_grid = np.linspace(0.0, max_elapsed, num=n_points)

    rows: List[np.ndarray] = []
    for times, values in seed_curves:
        rows.append(_step_sample_with_carry(times, values, time_grid))

    matrix = np.vstack(rows)
    counts = np.sum(~np.isnan(matrix), axis=0)
    valid = counts > 0
    if not np.any(valid):
        return pd.DataFrame(columns=["elapsed_s", "best_over_restarts"])

    with np.errstate(all="ignore"):
        mean_curve = np.nanmean(matrix, axis=0)

    # Mean over non-increasing seed curves should be non-increasing as well.
    mean_curve = np.minimum.accumulate(mean_curve)

    return pd.DataFrame(
        {
            "elapsed_s": time_grid[valid],
            "best_over_restarts": mean_curve[valid],
            "active_seeds": counts[valid].astype(int),
        }
    )


def _log_time(values: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.maximum(arr, eps)


def aggregate_close_best_points(points: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if points.empty:
        return points
    if len(points) <= max_points:
        return points.sort_values("elapsed_s").reset_index(drop=True)

    x = points["elapsed_s"].to_numpy(dtype=float)
    y = points["best_so_far"].to_numpy(dtype=float)
    counts = points["count"].to_numpy(dtype=float)

    bins_axis = max(8, int(np.sqrt(max_points)))

    def _edges(values: np.ndarray, bins: int) -> np.ndarray:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return np.array([0.0, 1.0])
        if abs(hi - lo) <= 1e-12:
            hi = lo + 1.0
        return np.linspace(lo, hi, bins + 1)

    x_edges = _edges(x, bins_axis)
    y_edges = _edges(y, bins_axis)

    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, len(x_edges) - 2).astype(np.int64)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, len(y_edges) - 2).astype(np.int64)
    combined_bin = x_bin * (len(y_edges) - 1) + y_bin

    _, inv = np.unique(combined_bin, return_inverse=True)
    summed_counts = np.bincount(inv, weights=counts)
    summed_x = np.bincount(inv, weights=x)
    summed_y = np.bincount(inv, weights=y)

    mean_x = np.divide(summed_x, summed_counts, out=np.zeros_like(summed_x), where=summed_counts > 0)
    mean_y = np.divide(summed_y, summed_counts, out=np.zeros_like(summed_y), where=summed_counts > 0)

    grouped = pd.DataFrame(
        {
            "elapsed_s": mean_x,
            "best_so_far": mean_y,
            "count": summed_counts,
        }
    ).sort_values("elapsed_s").reset_index(drop=True)

    if len(grouped) > max_points:
        idx = np.linspace(0, len(grouped) - 1, num=max_points, dtype=int)
        grouped = grouped.iloc[np.unique(idx)].reset_index(drop=True)

    return grouped[["elapsed_s", "best_so_far", "count"]]


def _safe_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return clean or "unknown"


def make_dataset_figure(
    run_id: str,
    dataset: str,
    mode_to_agg: Dict[str, pd.DataFrame],
    mode_to_best_points: Dict[str, pd.DataFrame],
    mode_to_concat_best: Dict[str, pd.DataFrame],
    mode_to_restart_count: Dict[str, int],
    output_path: Path,
) -> None:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=(
            "Best-cost points across all restarts (dots; size = aggregated density)",
            "Current solution over restarts (mean)",
            "Best-of-best evolution (concatenated within each seed; then mean over seeds)",
        ),
    )

    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
    ]

    for idx, (mode, agg) in enumerate(mode_to_agg.items()):
        if agg.empty:
            continue
        color = palette[idx % len(palette)]
        best_points = mode_to_best_points.get(mode, pd.DataFrame())

        if not best_points.empty:
            max_count = max(int(best_points["count"].max()), 1)
            marker_sizes = 5.0 + 11.0 * np.sqrt(best_points["count"].to_numpy(dtype=float) / max_count)
            marker_text = [f"aggregated_points={int(c)}" for c in best_points["count"]]
            best_points_x = _log_time(best_points["elapsed_s"])
        else:
            marker_sizes = None
            marker_text = None
            best_points_x = np.array([])

        concat_best = mode_to_concat_best.get(mode, pd.DataFrame())

        fig.add_trace(
            go.Scatter(
                x=best_points_x,
                y=best_points.get("best_so_far", pd.Series(dtype=float)),
                mode="markers",
                marker={
                    "size": marker_sizes if marker_sizes is not None else 6,
                    "color": color,
                    "opacity": 0.85,
                    "line": {"width": 0.4, "color": color},
                },
                text=marker_text,
                hovertemplate="time=%{x:.3f}s<br>best=%{y:.3f}<br>%{text}<extra></extra>",
                name=f"{mode}",
                legendgroup=mode,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=concat_best.get("elapsed_s", pd.Series(dtype=float)),
                y=concat_best.get("best_over_restarts", pd.Series(dtype=float)),
                mode="lines",
                line={"width": 2.2, "color": color},
                showlegend=False,
                legendgroup=mode,
                hovertemplate="time=%{x:.3f}s<br>best-of-best=%{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=_log_time(agg["elapsed_s"]),
                y=agg["current_mean"],
                mode="lines",
                line={"width": 2, "color": color},
                name=f"{mode} mean",
                legendgroup=mode,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    restarts_label = ", ".join(
        f"{mode}: {count}" for mode, count in mode_to_restart_count.items() if count > 0
    )
    if not restarts_label:
        restarts_label = "none"

    fig.update_xaxes(type="log", title_text="Elapsed seconds (log scale)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Elapsed seconds (log scale)", row=2, col=1)
    fig.update_xaxes(title_text="Elapsed seconds", row=3, col=1)
    fig.update_yaxes(title_text="Best cost", row=1, col=1)
    fig.update_yaxes(title_text="Current cost", row=2, col=1)
    fig.update_yaxes(title_text="Best-of-best cost", row=3, col=1)
    fig.update_layout(
        title={
            "text": (
                f"Runtime Progress by Mode (Dataset: {dataset})"
                f"<br><sup>run_id: {run_id} | restarts per mode: {restarts_label}</sup>"
            )
        },
        template="plotly_white",
        height=1200,
        hovermode="x unified",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
        },
        margin={"t": 110, "r": 260},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True, config={"displaylogo": False})


def main() -> None:
    args = parse_args()
    run_id = resolve_run_id(args.run_id, args.run_tag)
    modes = parse_modes(args.modes)

    log_dir = REPO_ROOT / "logs" / run_id
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    output_dir = args.output_dir or (REPO_ROOT / "output" / run_id / "runtime_progress_per_dataset")

    by_mode_dataset: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
    all_datasets: set[str] = set()

    for mode in modes:
        files = discover_mode_files(log_dir, mode)
        data = load_mode_trajectories_from_logs(mode, files) if files else {}
        if not data:
            fallback = load_mode_trajectories_from_results(run_id, mode)
            if fallback:
                warn(f"Mode '{mode}' plotted from results.csv fallback (no operator logs found)")
                data = fallback
            else:
                warn(f"No usable data for mode '{mode}'")
                continue

        by_mode_dataset[mode] = data
        all_datasets.update(data.keys())

    if not by_mode_dataset:
        raise SystemExit("No data available for any mode")

    generated: List[Path] = []
    for dataset in sorted(all_datasets):
        mode_to_agg: Dict[str, pd.DataFrame] = {}
        mode_to_best_points: Dict[str, pd.DataFrame] = {}
        mode_to_concat_best: Dict[str, pd.DataFrame] = {}
        mode_to_restart_count: Dict[str, int] = {}

        for mode in modes:
            mode_data = by_mode_dataset.get(mode, {})
            trajectories = mode_data.get(dataset, [])
            if not trajectories:
                continue
            agg = aggregate_mode_dataset(trajectories, max_time_points=args.max_time_points)
            if agg.empty:
                continue
            best_points_raw = collect_best_points(trajectories)
            best_points = aggregate_close_best_points(best_points_raw, max_points=max(1, int(args.max_best_dot_points)))
            concat_best = aggregate_concatenated_best_evolution(
                trajectories,
                max_points=max(1, int(args.max_time_points)),
            )
            mode_to_agg[mode] = agg
            mode_to_best_points[mode] = best_points
            mode_to_concat_best[mode] = concat_best
            mode_to_restart_count[mode] = len(trajectories)

        if not mode_to_agg:
            continue

        output_path = output_dir / f"runtime_progress_{_safe_name(dataset)}.html"
        make_dataset_figure(
            run_id=run_id,
            dataset=dataset,
            mode_to_agg=mode_to_agg,
            mode_to_best_points=mode_to_best_points,
            mode_to_concat_best=mode_to_concat_best,
            mode_to_restart_count=mode_to_restart_count,
            output_path=output_path,
        )
        generated.append(output_path)

    if not generated:
        raise SystemExit("No per-dataset plots were generated")

    print(f"resolved_run_id: {run_id}")
    print(f"log_dir: {log_dir}")
    print(f"output_dir: {output_dir}")
    print(f"generated_plots: {len(generated)}")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
