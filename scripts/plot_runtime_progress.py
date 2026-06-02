#!/usr/bin/env python3
"""Interactive runtime-progress plotting for operator usage logs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ("rl_improvement_mvp", "baseline", "kisialiou")


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    files: int
    points_best: int
    points_current: int
    points_raw: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime progress from operator usage logs (interactive Plotly HTML)."
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
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path (default: output/<run_id>/runtime_progress_interactive.html)",
    )
    parser.add_argument("--max-points-best", type=int, default=6000)
    parser.add_argument("--max-points-current", type=int, default=4000)
    parser.add_argument("--max-points-raw", type=int, default=1200)
    parser.add_argument(
        "--show-raw-current",
        action="store_true",
        help="Overlay sparse raw current-cost points in subplot 2",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=101,
        help="Rolling window for current-cost median/quantile smoothing",
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


def evenly_spaced_indices(length: int, max_points: int) -> np.ndarray:
    if max_points <= 0:
        raise ValueError("max_points must be > 0")
    if length <= max_points:
        return np.arange(length, dtype=int)
    idx = np.linspace(0, length - 1, num=max_points, dtype=int)
    return np.unique(idx)


def decimate_frame(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if df.empty:
        return df
    idx = evenly_spaced_indices(len(df), max_points)
    return df.iloc[idx].reset_index(drop=True)


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


def _restart_key(df: pd.DataFrame) -> pd.Series:
    for candidate in ("restart_id", "episode_id"):
        if candidate in df.columns:
            return df[candidate].fillna(-1).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index, name="restart_id")


def _last_non_nan(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def dedupe_to_steps(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    local = df.copy()
    local["__row_order"] = np.arange(len(local), dtype=int)
    local["__restart"] = _restart_key(local)

    if "iteration" not in local.columns:
        raise ValueError("Missing required column: iteration")
    if "cost_best" not in local.columns:
        raise ValueError("Missing required column: cost_best")
    if "cost_current" not in local.columns:
        raise ValueError("Missing required column: cost_current")

    elapsed_col = "operator_elapsed_ms" if "operator_elapsed_ms" in local.columns else "elapsed_ms"
    if elapsed_col not in local.columns:
        raise ValueError("Missing required elapsed column: operator_elapsed_ms or elapsed_ms")

    grouped = (
        local.sort_values(["__restart", "iteration", "__row_order"])
        .groupby(["__restart", "iteration"], as_index=False, sort=True)
        .agg(
            elapsed_ms=(elapsed_col, "max"),
            cost_best=("cost_best", "min"),
            cost_current=("cost_current", _last_non_nan),
        )
    )
    grouped["source_file"] = source_file
    return grouped


def load_mode_points(mode: str, files: Sequence[Path]) -> pd.DataFrame:
    trajectories: List[pd.DataFrame] = []
    for file_path in sorted(files):
        try:
            raw = pd.read_csv(file_path)
        except Exception as exc:
            warn(f"Skipping unreadable file {file_path}: {exc}")
            continue
        if raw.empty:
            warn(f"Skipping empty file {file_path}")
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

        try:
            steps = dedupe_to_steps(raw, source_file=file_path.name)
        except ValueError as exc:
            warn(f"Skipping file {file_path.name}: {exc}")
            continue

        steps = steps.sort_values(["source_file", "__restart", "iteration"]).reset_index(drop=True)
        if steps.empty:
            continue
        trajectories.append(steps)

    if not trajectories:
        return pd.DataFrame()

    offsets: List[pd.DataFrame] = []
    elapsed_offset = 0.0
    for frame in trajectories:
        part = frame.copy()
        part["elapsed_ms"] = part["elapsed_ms"].fillna(0.0).clip(lower=0.0)
        part["elapsed_s_local"] = part["elapsed_ms"].cumsum() / 1000.0
        part["elapsed_s"] = part["elapsed_s_local"] + elapsed_offset
        elapsed_offset = float(part["elapsed_s"].iloc[-1]) if not part.empty else elapsed_offset
        offsets.append(part)

    merged = pd.concat(offsets, ignore_index=True)
    merged["mode"] = mode
    merged["best_so_far"] = merged["cost_best"].cummin()
    return merged


def load_mode_points_from_results(run_id: str, mode: str) -> pd.DataFrame:
    results_path = REPO_ROOT / "output" / run_id / "results.csv"
    if not results_path.exists():
        return pd.DataFrame()

    try:
        results = pd.read_csv(results_path)
    except Exception as exc:
        warn(f"Failed reading fallback results.csv for mode '{mode}': {exc}")
        return pd.DataFrame()

    if results.empty or "mode" not in results.columns:
        return pd.DataFrame()

    mode_rows = results[results["mode"].astype(str) == mode].copy()
    if mode_rows.empty:
        return pd.DataFrame()

    _coerce_numeric(mode_rows, ["runtime_seconds", "best_cost"])
    mode_rows = mode_rows.dropna(subset=["runtime_seconds", "best_cost"]).reset_index(drop=True)
    if mode_rows.empty:
        return pd.DataFrame()

    mode_rows["elapsed_s"] = mode_rows["runtime_seconds"].clip(lower=0.0).cumsum()
    mode_rows["elapsed_ms"] = mode_rows["runtime_seconds"].clip(lower=0.0) * 1000.0
    mode_rows["cost_best"] = mode_rows["best_cost"]
    mode_rows["cost_current"] = mode_rows["best_cost"]
    mode_rows["best_so_far"] = mode_rows["best_cost"].cummin()
    mode_rows["mode"] = mode
    if "dataset" in mode_rows.columns:
        mode_rows["instance_name"] = mode_rows["dataset"].astype(str)
    else:
        mode_rows["instance_name"] = "unknown"
    return mode_rows


def build_current_band(points: pd.DataFrame, window: int) -> pd.DataFrame:
    if points.empty:
        return points
    local = points.copy()
    win = max(1, int(window))
    rolling = local["cost_current"].rolling(window=win, min_periods=1)
    local["current_median"] = rolling.median()
    local["current_q10"] = rolling.quantile(0.10)
    local["current_q90"] = rolling.quantile(0.90)
    return local


def make_figure(
    mode_to_points: Dict[str, pd.DataFrame],
    run_id: str,
    dataset_label: str,
    max_points_best: int,
    max_points_current: int,
    max_points_raw: int,
    show_raw_current: bool,
    output_html: Path,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Best found so far vs elapsed time",
            "Current solution (rolling median with q10-q90 band)",
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

    for idx, mode in enumerate(mode_to_points.keys()):
        color = palette[idx % len(palette)]
        points = mode_to_points[mode]

        best_df = decimate_frame(points[["elapsed_s", "best_so_far"]], max_points_best)
        fig.add_trace(
            go.Scatter(
                x=best_df["elapsed_s"],
                y=best_df["best_so_far"],
                mode="lines",
                name=f"{mode} best",
                line={"color": color, "width": 2},
            ),
            row=1,
            col=1,
        )

        cur_df = decimate_frame(
            points[["elapsed_s", "current_median", "current_q10", "current_q90"]],
            max_points_current,
        )

        fig.add_trace(
            go.Scatter(
                x=cur_df["elapsed_s"],
                y=cur_df["current_q90"],
                mode="lines",
                line={"width": 0, "color": color},
                name=f"{mode} q90",
                showlegend=False,
                hoverinfo="skip",
                opacity=0.35,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cur_df["elapsed_s"],
                y=cur_df["current_q10"],
                mode="lines",
                line={"width": 0, "color": color},
                fill="tonexty",
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.08)",
                name=f"{mode} q10-q90",
                showlegend=False,
                opacity=0.35,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cur_df["elapsed_s"],
                y=cur_df["current_median"],
                mode="lines+markers",
                line={"color": color, "width": 1.5},
                marker={"size": 5, "opacity": 0.85, "color": color},
                name=f"{mode} current",
                opacity=0.55,
            ),
            row=2,
            col=1,
        )

        if show_raw_current:
            raw_df = decimate_frame(points[["elapsed_s", "cost_current"]], max_points_raw)
            fig.add_trace(
                go.Scatter(
                    x=raw_df["elapsed_s"],
                    y=raw_df["cost_current"],
                    mode="markers",
                    marker={"size": 7, "color": color, "opacity": 0.75},
                    name=f"{mode} raw",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.update_xaxes(title_text="Elapsed seconds", row=2, col=1)
    fig.update_yaxes(title_text="Best cost", row=1, col=1)
    fig.update_yaxes(title_text="Current cost", row=2, col=1)
    fig.update_layout(
        title={
            "text": f"Runtime Progress by Mode<br><sup>run_id: {run_id} | datasets: {dataset_label}</sup>"
        },
        template="plotly_white",
        height=900,
        hovermode="x unified",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "tracegroupgap": 8,
        },
        margin={"t": 110, "r": 240},
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True, config={"displaylogo": False})


def main() -> None:
    args = parse_args()
    run_id = resolve_run_id(args.run_id, args.run_tag)
    modes = parse_modes(args.modes)

    log_dir = REPO_ROOT / "logs" / run_id
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    output_html = args.output_html or (REPO_ROOT / "output" / run_id / "runtime_progress_interactive.html")

    mode_to_points: Dict[str, pd.DataFrame] = {}
    mode_summaries: List[ModeSummary] = []

    for mode in modes:
        files = discover_mode_files(log_dir, mode)
        points = load_mode_points(mode, files) if files else pd.DataFrame()
        used_fallback = False
        if points.empty:
            points = load_mode_points_from_results(run_id, mode)
            used_fallback = not points.empty
            if not used_fallback:
                warn(f"No files found for mode '{mode}' in {log_dir} and no usable fallback rows in results.csv")
                continue
            warn(f"Mode '{mode}' plotted from results.csv fallback (no operator logs found)")

        if points.empty:
            warn(f"No usable rows for mode '{mode}'")
            continue

        points = build_current_band(points, args.rolling_window)
        mode_to_points[mode] = points

        summary = ModeSummary(
            mode=mode,
            files=(len(files) if files else 1),
            points_best=min(len(points), args.max_points_best),
            points_current=min(len(points), args.max_points_current),
            points_raw=min(len(points), args.max_points_raw),
        )
        mode_summaries.append(summary)

    if not mode_to_points:
        raise SystemExit("No data available for any mode after filtering/validation")

    dataset_names: List[str] = []
    for points in mode_to_points.values():
        if "instance_name" in points.columns:
            names = [str(name) for name in points["instance_name"].dropna().astype(str).unique() if str(name).strip()]
            dataset_names.extend(names)
    unique_dataset_names = sorted(set(dataset_names))
    if not unique_dataset_names:
        dataset_label = "unknown"
    elif len(unique_dataset_names) <= 4:
        dataset_label = ", ".join(unique_dataset_names)
    else:
        dataset_label = ", ".join(unique_dataset_names[:4]) + f" ... (+{len(unique_dataset_names) - 4})"

    make_figure(
        mode_to_points=mode_to_points,
        run_id=run_id,
        dataset_label=dataset_label,
        max_points_best=args.max_points_best,
        max_points_current=args.max_points_current,
        max_points_raw=args.max_points_raw,
        show_raw_current=args.show_raw_current,
        output_html=output_html,
    )

    print(f"resolved_run_id: {run_id}")
    print(f"log_dir: {log_dir}")
    print(f"output_html: {output_html}")
    for summary in mode_summaries:
        print(
            f"mode={summary.mode} files={summary.files} "
            f"points(best/current/raw)={summary.points_best}/{summary.points_current}/{summary.points_raw}"
        )


if __name__ == "__main__":
    main()