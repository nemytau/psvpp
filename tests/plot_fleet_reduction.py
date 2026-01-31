#!/usr/bin/env python3
"""Visualize fleet reduction snapshots exported by the Rust integration test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "rust_alns" / "output" / "tests"
DEFAULT_BEFORE = DEFAULT_OUTPUT_DIR / "fleet_reduction_before.json"
DEFAULT_AFTER = DEFAULT_OUTPUT_DIR / "fleet_reduction_after.json"


def load_schedule(path: Path) -> Tuple[List[dict], float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot {path} is missing. Run `cargo test fleet_reduction_merges_vessels_and_exports_snapshots` first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    voyages = list(payload.get("voyages", []))
    return voyages, float(payload.get("cost", 0.0))


def add_schedule_traces(
    fig: go.Figure, voyages: List[dict], column: int, palette: Dict[str, str]
) -> float:
    max_end = 0.0
    for entry in voyages:
        start = float(entry.get("Start", 0.0))
        end = float(entry.get("End", start))
        duration = max(end - start, 0.0)
        vessel = str(entry.get("Vessel", "?"))
        route = str(entry.get("Route", ""))
        load = str(entry.get("Load", ""))
        max_end = max(max_end, end)
        color = palette.setdefault(vessel, f"hsl({(len(palette) * 47) % 360},60%,55%)")
        hover = (
            f"Vessel: %{{y}}<br>Route: {route}<br>Load: {load}<br>"
            "Start: %{customdata[0]:.2f}h<br>End: %{customdata[1]:.2f}h"
            "<br>Duration: %{x:.2f}h<extra></extra>"
        )
        fig.add_trace(
            go.Bar(
                x=[duration],
                y=[vessel],
                base=start,
                orientation="h",
                marker=dict(color=color),
                hovertemplate=hover,
                customdata=[[start, end]],
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    return max_end


def build_comparison(before_path: Path, after_path: Path, title_suffix: str | None = None) -> go.Figure:
    before_voyages, before_cost = load_schedule(before_path)
    after_voyages, after_cost = load_schedule(after_path)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=(
            f"Before (cost {before_cost:.1f})",
            f"After (cost {after_cost:.1f})",
        ),
    )

    palette: Dict[str, str] = {}
    max_before = add_schedule_traces(fig, before_voyages, 1, palette)
    max_after = add_schedule_traces(fig, after_voyages, 2, palette)
    horizon = max(max_before, max_after, 24.0)

    fig.update_xaxes(title_text="Hours", range=[0, horizon], row=1, col=1)
    fig.update_xaxes(title_text="Hours", range=[0, horizon], row=1, col=2)
    fig.update_yaxes(title_text="Vessel")
    final_title = "Fleet Reduction Comparison"
    if title_suffix:
        final_title += f" - {title_suffix}"
    fig.update_layout(height=600, title=dict(text=final_title, x=0.5), barmode="stack", bargap=0.25)
    return fig
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE, help="Path to the pre-operator snapshot")
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER, help="Path to the post-operator snapshot")
    parser.add_argument("--output", type=Path, help="Optional file to save the figure as HTML")
    parser.add_argument("--title", type=str, help="Optional suffix appended to the chart title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure = build_comparison(args.before, args.after, args.title)
    if args.output:
        figure.write_html(args.output, auto_open=False)
        print(f"Visualization written to {args.output}")
    else:
        figure.show()


if __name__ == "__main__":
    main()
