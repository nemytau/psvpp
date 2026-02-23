#!/usr/bin/env python3
"""Explore and compare operator-usage logs produced by ALNS runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "operator_usage_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory and comparative analysis for operator usage CSV logs"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to operator usage CSV files",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each input (e.g., baseline kisialiou)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated summaries",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Top operators per mode to keep in the markdown preview",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling window for iteration trend plots",
    )
    return parser.parse_args()


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin({"1", "true", "yes"})


def _coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def load_inputs(paths: Sequence[str], labels: Sequence[str] | None) -> Tuple[pd.DataFrame, List[str]]:
    if labels and len(labels) != len(paths):
        raise ValueError("If provided, --labels count must match --inputs count")

    frames: List[pd.DataFrame] = []
    used_labels: List[str] = []

    for idx, raw_path in enumerate(paths):
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        label = labels[idx].strip() if labels and labels[idx].strip() else path.stem
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        df["mode_label"] = label
        frames.append(df)
        used_labels.append(label)

    if not frames:
        raise ValueError("No input data loaded")

    combined = pd.concat(frames, ignore_index=True)

    _coerce_numeric(
        combined,
        [
            "iteration",
            "reward",
            "cost_current",
            "cost_best",
            "elapsed_ms",
            "cost_delta",
            "best_cost_delta",
            "best_cost_delta_future",
            "lookahead_window",
            "operator_index",
            "destroy_idx",
            "repair_idx",
            "improvement_idx",
        ],
    )

    for boolean_col in ["accepted", "is_new_best"]:
        if boolean_col in combined.columns:
            combined[boolean_col] = _coerce_bool(combined[boolean_col])

    for text_col in ["operator_type", "operator_name", "instance_id"]:
        if text_col in combined.columns:
            combined[text_col] = combined[text_col].fillna("unknown").astype(str)

    return combined, used_labels


def summarize_steps(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_cols = [col for col in ["mode_label", "episode_id", "instance_id", "iteration"] if col in df.columns]
    step_df = df.drop_duplicates(subset=dedupe_cols, keep="first")

    grouped = step_df.groupby("mode_label", dropna=False)
    summary = grouped.agg(
        steps=("iteration", "count"),
        max_iteration=("iteration", "max"),
        mean_reward=("reward", "mean"),
        acceptance_rate=("accepted", "mean"),
        new_best_rate=("is_new_best", "mean"),
        mean_cost_delta=("cost_delta", "mean"),
        mean_best_cost_delta=("best_cost_delta", "mean"),
        mean_future_best_delta=("best_cost_delta_future", "mean"),
        mean_elapsed_ms=("elapsed_ms", "mean"),
        final_best_cost=("cost_best", "min"),
    )
    return summary.reset_index().sort_values("mode_label")


def extract_step_frame(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_cols = [col for col in ["mode_label", "episode_id", "instance_id", "iteration"] if col in df.columns]
    step_df = df.drop_duplicates(subset=dedupe_cols, keep="first").copy()
    step_df = step_df.sort_values(["mode_label", "iteration"]).reset_index(drop=True)
    return step_df


def extract_dataset_size(instance_path: str) -> str:
    """Extract dataset size category (small/medium/large) from instance path."""
    instance_lower = str(instance_path).lower()
    if "/small" in instance_lower or "small_" in instance_lower:
        return "small"
    elif "/medium" in instance_lower or "medium_" in instance_lower:
        return "medium"
    elif "/large" in instance_lower or "large_" in instance_lower:
        return "large"
    return "unknown"


def summarize_operator_usage(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["mode_label", "operator_type", "operator_name"], dropna=False)
    operator_summary = grouped.agg(
        usage_count=("operator_name", "size"),
        acceptance_rate=("accepted", "mean"),
        new_best_rate=("is_new_best", "mean"),
        mean_reward=("reward", "mean"),
        mean_cost_delta=("cost_delta", "mean"),
        mean_best_cost_delta=("best_cost_delta", "mean"),
        mean_future_best_delta=("best_cost_delta_future", "mean"),
        mean_elapsed_ms=("elapsed_ms", "mean"),
        total_elapsed_ms=("elapsed_ms", "sum"),
    ).reset_index()

    total_by_mode = operator_summary.groupby("mode_label")["usage_count"].transform("sum")
    operator_summary["usage_share_pct"] = (operator_summary["usage_count"] / total_by_mode) * 100.0

    return operator_summary.sort_values(
        ["mode_label", "usage_count", "operator_type", "operator_name"],
        ascending=[True, False, True, True],
    )


def summarize_operator_type(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["mode_label", "operator_type"], dropna=False)
    type_summary = grouped.agg(
        usage_count=("operator_type", "size"),
        acceptance_rate=("accepted", "mean"),
        new_best_rate=("is_new_best", "mean"),
        mean_reward=("reward", "mean"),
        mean_cost_delta=("cost_delta", "mean"),
        mean_best_cost_delta=("best_cost_delta", "mean"),
        mean_future_best_delta=("best_cost_delta_future", "mean"),
        total_elapsed_ms=("elapsed_ms", "sum"),
    ).reset_index()

    total_by_mode = type_summary.groupby("mode_label")["usage_count"].transform("sum")
    type_summary["usage_share_pct"] = (type_summary["usage_count"] / total_by_mode) * 100.0
    total_time_by_mode = type_summary.groupby("mode_label")["total_elapsed_ms"].transform("sum")
    type_summary["time_share_pct"] = (type_summary["total_elapsed_ms"] / total_time_by_mode.clip(lower=1e-9)) * 100.0
    return type_summary.sort_values(["mode_label", "usage_count"], ascending=[True, False])


def pairwise_step_diff(step_summary: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    if len(labels) < 2:
        return pd.DataFrame()

    left = step_summary[step_summary["mode_label"] == labels[0]].copy()
    right = step_summary[step_summary["mode_label"] == labels[1]].copy()
    if left.empty or right.empty:
        return pd.DataFrame()

    comparison_rows: List[Dict[str, float | str]] = []
    metrics = [
        "acceptance_rate",
        "new_best_rate",
        "mean_reward",
        "mean_cost_delta",
        "mean_best_cost_delta",
        "mean_future_best_delta",
        "mean_elapsed_ms",
        "final_best_cost",
    ]

    for metric in metrics:
        left_value = float(left.iloc[0][metric])
        right_value = float(right.iloc[0][metric])
        comparison_rows.append(
            {
                "metric": metric,
                f"value_{labels[0]}": left_value,
                f"value_{labels[1]}": right_value,
                f"diff_{labels[1]}_minus_{labels[0]}": right_value - left_value,
            }
        )

    return pd.DataFrame(comparison_rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def render_markdown(
    *,
    labels: Sequence[str],
    step_summary: pd.DataFrame,
    type_summary: pd.DataFrame,
    operator_summary: pd.DataFrame,
    pairwise_diff: pd.DataFrame,
    top_n: int,
) -> str:
    lines: List[str] = []
    lines.append("# Operator Usage Comparative Analysis")
    lines.append("")
    lines.append("## Inputs")
    for label in labels:
        lines.append(f"- {label}")
    lines.append("")

    lines.append("## Step-Level Summary")
    lines.append(dataframe_to_markdown(step_summary))
    lines.append("")

    lines.append("## Operator-Type Summary")
    lines.append(dataframe_to_markdown(type_summary))
    lines.append("")

    lines.append(f"## Top {top_n} Operators Per Mode")
    for label in labels:
        lines.append("")
        lines.append(f"### {label}")
        mode_ops = operator_summary[operator_summary["mode_label"] == label].head(top_n)
        if mode_ops.empty:
            lines.append("No operator rows found.")
        else:
            lines.append(dataframe_to_markdown(mode_ops))

    if not pairwise_diff.empty:
        lines.append("")
        lines.append("## Pairwise Difference")
        lines.append(dataframe_to_markdown(pairwise_diff))

    return "\n".join(lines)


def _safe_mode_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def generate_plots(
    *,
    output_dir: Path,
    step_df: pd.DataFrame,
    type_summary: pd.DataFrame,
    operator_summary: pd.DataFrame,
    rolling_window: int,
    top_n: int,
) -> List[str]:
    plot_paths: List[str] = []

    if not type_summary.empty:
        pivot = (
            type_summary.pivot_table(
                index="mode_label",
                columns="operator_type",
                values="time_share_pct",
                aggfunc="sum",
                fill_value=0.0,
            )
            .sort_index()
        )
        ax = pivot.plot(kind="bar", figsize=(9, 4))
        ax.set_title("Operator Type Time Share (%)")
        ax.set_ylabel("Time share (%)")
        ax.set_xlabel("Mode")
        ax.legend(title="Operator type", bbox_to_anchor=(1.01, 1.0), loc="upper left")
        plt.tight_layout()
        type_path = output_dir / "plot_operator_type_time_share.png"
        plt.savefig(type_path, dpi=150)
        plt.close()
        plot_paths.append(str(type_path))

    if not operator_summary.empty:
        top_rows = operator_summary.copy()
        top_rows["operator_label"] = top_rows["operator_type"].astype(str) + ":" + top_rows["operator_name"].astype(str)
        modes = sorted(top_rows["mode_label"].unique())
        fig, axes = plt.subplots(len(modes), 1, figsize=(12, max(4, 3.5 * len(modes))), squeeze=False)
        for idx, mode in enumerate(modes):
            part = (
                top_rows[top_rows["mode_label"] == mode]
                .sort_values("usage_count", ascending=False)
                .head(max(1, int(top_n)))
            )
            ax = axes[idx, 0]
            ax.bar(part["operator_label"], part["usage_count"], color="tab:blue", alpha=0.85)
            ax.set_title(f"{mode} - Top {max(1, int(top_n))} Operator Usage Counts")
            ax.set_ylabel("Usage count")
            ax.tick_params(axis="x", rotation=50)
            ax.grid(axis="y", alpha=0.25)
        axes[-1, 0].set_xlabel("Operator")
        plt.tight_layout()
        top_path = output_dir / "plot_top_operator_usage.png"
        plt.savefig(top_path, dpi=150)
        plt.close()
        plot_paths.append(str(top_path))

    if not operator_summary.empty:
        contribution_df = operator_summary.copy()
        contribution_df["operator_label"] = (
            contribution_df["operator_type"].astype(str) + ":" + contribution_df["operator_name"].astype(str)
        )
        contribution_df["cost_improvement_total"] = (
            (-contribution_df["mean_best_cost_delta"]).clip(lower=0.0) * contribution_df["usage_count"]
        )

        modes = sorted(contribution_df["mode_label"].unique())
        fig, axes = plt.subplots(len(modes), 1, figsize=(12, max(4, 3.5 * len(modes))), squeeze=False)
        for idx, mode in enumerate(modes):
            part = (
                contribution_df[contribution_df["mode_label"] == mode]
                .sort_values("cost_improvement_total", ascending=False)
                .head(max(1, int(top_n)))
            )
            ax = axes[idx, 0]
            ax.barh(part["operator_label"], part["cost_improvement_total"], color="tab:green", alpha=0.85)
            ax.set_title(f"{mode} - Top {max(1, int(top_n))} Operators by Total Best-Cost Improvement")
            ax.set_xlabel("Total best-cost improvement contribution")
            ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        contrib_path = output_dir / "plot_operator_cost_improvement.png"
        plt.savefig(contrib_path, dpi=150)
        plt.close()
        plot_paths.append(str(contrib_path))

    if not step_df.empty and "instance_id" in step_df.columns:
        step_df["dataset_size"] = step_df["instance_id"].apply(extract_dataset_size)
        
        sizes = sorted([s for s in step_df["dataset_size"].unique() if s != "unknown"])
        
        for size in sizes:
            size_df = step_df[step_df["dataset_size"] == size].copy()
            if size_df.empty:
                continue
            
            episodes = size_df.groupby(["mode_label", "episode_id"])
            initial_costs = episodes["cost_best"].first().reset_index()
            initial_costs.columns = ["mode_label", "episode_id", "initial_cost"]
            size_df = size_df.merge(initial_costs, on=["mode_label", "episode_id"], how="left")
            size_df["improvement_pct"] = (
                (size_df["initial_cost"] - size_df["cost_best"]) / size_df["initial_cost"].clip(lower=1e-9)
            ) * 100.0
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            for mode in sorted(size_df["mode_label"].unique()):
                mode_data = size_df[size_df["mode_label"] == mode]
                for episode in mode_data["episode_id"].unique():
                    episode_data = mode_data[mode_data["episode_id"] == episode].sort_values("iteration")
                    ax1.plot(
                        episode_data["iteration"],
                        episode_data["cost_best"],
                        alpha=0.6,
                        linewidth=1.5,
                        label=f"{mode}" if episode == mode_data["episode_id"].min() else None
                    )
            
            ax1.set_title(f"Best Cost vs Iteration ({size.capitalize()} Datasets)")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Best cost")
            ax1.legend(loc="best")
            ax1.grid(alpha=0.25)
            
            for mode in sorted(size_df["mode_label"].unique()):
                mode_data = size_df[size_df["mode_label"] == mode]
                for episode in mode_data["episode_id"].unique():
                    episode_data = mode_data[mode_data["episode_id"] == episode].sort_values("iteration")
                    ax2.plot(
                        episode_data["iteration"],
                        episode_data["improvement_pct"],
                        alpha=0.6,
                        linewidth=1.5,
                        label=f"{mode}" if episode == mode_data["episode_id"].min() else None
                    )
            
            ax2.set_title(f"Improvement from Initial (%) vs Iteration ({size.capitalize()} Datasets)")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Improvement (%)")
            ax2.legend(loc="best")
            ax2.grid(alpha=0.25)
            
            plt.tight_layout()
            best_iter_path = output_dir / f"plot_best_cost_vs_iteration_{size}.png"
            plt.savefig(best_iter_path, dpi=150)
            plt.close()
            plot_paths.append(str(best_iter_path))
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            for mode in sorted(size_df["mode_label"].unique()):
                mode_data = size_df[size_df["mode_label"] == mode]
                for episode in mode_data["episode_id"].unique():
                    episode_data = mode_data[mode_data["episode_id"] == episode].sort_values("iteration").copy()
                    episode_data["elapsed_seconds_cum"] = episode_data["elapsed_ms"].fillna(0.0).cumsum() / 1000.0
                    ax1.plot(
                        episode_data["elapsed_seconds_cum"],
                        episode_data["cost_best"],
                        alpha=0.6,
                        linewidth=1.5,
                        label=f"{mode}" if episode == mode_data["episode_id"].min() else None
                    )
            
            ax1.set_title(f"Best Cost vs Elapsed Time ({size.capitalize()} Datasets)")
            ax1.set_xlabel("Elapsed time (s)")
            ax1.set_ylabel("Best cost")
            ax1.legend(loc="best")
            ax1.grid(alpha=0.25)
            
            for mode in sorted(size_df["mode_label"].unique()):
                mode_data = size_df[size_df["mode_label"] == mode]
                for episode in mode_data["episode_id"].unique():
                    episode_data = mode_data[mode_data["episode_id"] == episode].sort_values("iteration").copy()
                    episode_data["elapsed_seconds_cum"] = episode_data["elapsed_ms"].fillna(0.0).cumsum() / 1000.0
                    ax2.plot(
                        episode_data["elapsed_seconds_cum"],
                        episode_data["improvement_pct"],
                        alpha=0.6,
                        linewidth=1.5,
                        label=f"{mode}" if episode == mode_data["episode_id"].min() else None
                    )
            
            ax2.set_title(f"Improvement from Initial (%) vs Elapsed Time ({size.capitalize()} Datasets)")
            ax2.set_xlabel("Elapsed time (s)")
            ax2.set_ylabel("Improvement (%)")
            ax2.legend(loc="best")
            ax2.grid(alpha=0.25)
            
            plt.tight_layout()
            best_time_path = output_dir / f"plot_best_cost_vs_time_{size}.png"
            plt.savefig(best_time_path, dpi=150)
            plt.close()
            plot_paths.append(str(best_time_path))

    if not step_df.empty:
        window = max(1, int(rolling_window))
        for mode in sorted(step_df["mode_label"].unique()):
            mode_df = step_df[step_df["mode_label"] == mode].copy()
            mode_df = mode_df.sort_values("iteration")
            mode_df["reward_roll"] = mode_df["reward"].rolling(window=window, min_periods=1).mean()
            mode_df["cost_delta_roll"] = mode_df["cost_delta"].rolling(window=window, min_periods=1).mean()

            fig, ax1 = plt.subplots(figsize=(10, 4))
            line1 = ax1.plot(mode_df["iteration"], mode_df["reward_roll"], label="Reward (rolling)", color="tab:blue")[0]
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Reward", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            line2 = ax2.plot(mode_df["iteration"], mode_df["cost_delta_roll"], label="Cost Δ (rolling)", color="tab:orange")[0]
            ax2.set_ylabel("Cost delta", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            plt.title(f"Iteration Trends - {mode} (window={window})")
            ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="best")
            fig.tight_layout()
            trend_path = output_dir / f"plot_trends_{_safe_mode_name(mode)}.png"
            plt.savefig(trend_path, dpi=150)
            plt.close()
            plot_paths.append(str(trend_path))

    return plot_paths


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined, labels = load_inputs(args.inputs, args.labels)

    step_df = extract_step_frame(combined)
    step_summary = summarize_steps(combined)
    type_summary = summarize_operator_type(combined)
    operator_summary = summarize_operator_usage(combined)
    pairwise_diff = pairwise_step_diff(step_summary, labels)

    plot_paths = generate_plots(
        output_dir=output_dir,
        step_df=step_df,
        type_summary=type_summary,
        operator_summary=operator_summary,
        rolling_window=args.rolling_window,
        top_n=args.top_n,
    )

    combined_path = output_dir / "combined_operator_usage.csv"
    step_summary_path = output_dir / "step_summary.csv"
    type_summary_path = output_dir / "operator_type_summary.csv"
    operator_summary_path = output_dir / "operator_summary.csv"
    pairwise_path = output_dir / "pairwise_step_diff.csv"
    report_path = output_dir / "report.md"
    meta_path = output_dir / "summary_meta.json"

    combined.to_csv(combined_path, index=False)
    step_summary.to_csv(step_summary_path, index=False)
    type_summary.to_csv(type_summary_path, index=False)
    operator_summary.to_csv(operator_summary_path, index=False)
    if not pairwise_diff.empty:
        pairwise_diff.to_csv(pairwise_path, index=False)

    report = render_markdown(
        labels=labels,
        step_summary=step_summary,
        type_summary=type_summary,
        operator_summary=operator_summary,
        pairwise_diff=pairwise_diff,
        top_n=max(1, int(args.top_n)),
    )
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "inputs": [str(Path(path).expanduser().resolve()) for path in args.inputs],
        "labels": labels,
        "rows_total": int(len(combined)),
        "steps_total": int(step_summary["steps"].sum()) if not step_summary.empty else 0,
        "outputs": {
            "combined": str(combined_path),
            "step_summary": str(step_summary_path),
            "operator_type_summary": str(type_summary_path),
            "operator_summary": str(operator_summary_path),
            "pairwise_step_diff": str(pairwise_path) if pairwise_path.exists() else None,
            "report": str(report_path),
            "plots": plot_paths,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Analysis complete. Output directory: {output_dir}")
    print(f"- step summary: {step_summary_path}")
    print(f"- operator summary: {operator_summary_path}")
    print(f"- report: {report_path}")
    if plot_paths:
        print("- plots:")
        for path in plot_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
