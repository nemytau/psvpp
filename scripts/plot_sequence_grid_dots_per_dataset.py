"""
Dot plots for ALNS sequence grid results.
One HTML per dataset size. Each HTML = single figure with 2 subplots:
  Left:  x=mean runtime(s), y=mean improvement %
  Right: x=mean runtime(s), y=median gap vs Kisialiou %
All datasets of the same size on one plot.
Each dot = one sequence × one dataset (averaged across seeds).
Colour = sequence, marker shape = dataset.
Kisialiou reference shown as black X markers.
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

DEFAULT_RUN = (
    "output/alns_sequence_grid_20260316_195103__"
    "joint_small_medium_l1_l2_l3_s5_i300_timefair_perm"
)

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", default=DEFAULT_RUN)
args = parser.parse_args()

run_dir = Path(args.run_dir)
df = pd.read_csv(run_dir / "results.csv")

df["kisialiou_improvement_pct"] = (
    (df["initial_cost"] - df["reference_kisialiou_best_cost"])
    / df["initial_cost"] * 100
)

# ── Aggregate per (dataset, sequence) across seeds ───────────────────────────
seq_agg = (
    df.groupby(["size", "dataset", "sequence_key", "sequence_length"])
    .agg(
        mean_runtime=("runtime_seconds", "mean"),
        mean_improvement=("improvement_pct", "mean"),
        median_gap=("gap_vs_kisialiou_pct", "median"),
        n_seeds=("seed", "count"),
    )
    .reset_index()
)

# Kisialiou reference: mean runtime_budget_seconds per dataset (= Kisialiou time)
kis_agg = (
    df.groupby(["size", "dataset"])
    .agg(
        mean_runtime=("runtime_budget_seconds", "mean"),
        mean_improvement=("kisialiou_improvement_pct", "mean"),
        n_seeds=("seed", "count"),
    )
    .reset_index()
)

# ── Color rules by sequence membership ───────────────────────────────────────
VNR = "voyage_number_reduction"
FCR = "fleet_and_cost_reduction"

COLOR_VNR = "#1f77b4"      # blue
COLOR_OTHER = "#ff7f0e"    # orange (for everything else)


def sequence_color(sequence_key: str) -> str:
    has_vnr = VNR in sequence_key
    if has_vnr:
        return COLOR_VNR
    return COLOR_OTHER


KIS_COLOR = "#111111"

# Marker shape cycle for datasets (up to 10)
DS_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross",
              "star", "pentagon", "hexagon", "triangle-down", "bowtie"]

run_tag = run_dir.name.replace("alns_sequence_grid_", "")

# ── One HTML per size ─────────────────────────────────────────────────────────
for size in sorted(seq_agg["size"].unique()):
    size_seqs = seq_agg[seq_agg["size"] == size]
    size_kis = kis_agg[kis_agg["size"] == size]

    datasets_in_size = sorted(size_seqs["dataset"].unique())
    ds_symbol = {ds: DS_SYMBOLS[i % len(DS_SYMBOLS)] for i, ds in enumerate(datasets_in_size)}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{size} — improvement %",
            f"{size} — gap vs Kisialiou %",
        ],
        horizontal_spacing=0.10,
    )

    added_ds_legend = set()

    for _, seq_row in size_seqs.iterrows():
        seq = seq_row["sequence_key"]
        length = int(seq_row["sequence_length"])
        ds = seq_row["dataset"]
        color = sequence_color(seq)
        symbol = ds_symbol[ds]

        hover1 = (
            f"<b>{ds}</b><br>"
            f"seq: {seq}<br>"
            f"rt: {seq_row['mean_runtime']:.2f}s<br>"
            f"impr: {seq_row['mean_improvement']:.2f}%<br>"
            f"n={seq_row['n_seeds']}"
        )
        hover2 = (
            f"<b>{ds}</b><br>"
            f"seq: {seq}<br>"
            f"rt: {seq_row['mean_runtime']:.2f}s<br>"
            f"median gap: {seq_row['median_gap']:+.2f}%"
        )

        marker = dict(size=10, color=color, symbol=symbol,
                      opacity=0.82, line=dict(width=0.7, color="white"))

        fig.add_trace(go.Scatter(
            x=[seq_row["mean_runtime"]], y=[seq_row["mean_improvement"]],
            mode="markers", marker=marker,
            name=f"[{length}] {seq}",
            text=[hover1], hoverinfo="text",
            legendgroup=seq, showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[seq_row["mean_runtime"]], y=[seq_row["median_gap"]],
            mode="markers", marker=marker,
            name=f"[{length}] {seq}",
            text=[hover2], hoverinfo="text",
            legendgroup=seq, showlegend=False,
        ), row=1, col=2)

    # Color legend (dummy traces)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=COLOR_VNR),
        name="VNR in sequence",
        legendgroup="cat_vnr", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=COLOR_OTHER),
        name="Others (no VNR)",
        legendgroup="cat_other", showlegend=True,
    ), row=1, col=1)

    # Kisialiou reference dots — one per dataset, black X, shape=dataset
    for _, kr in size_kis.iterrows():
        ds = kr["dataset"]
        symbol = ds_symbol[ds]
        show_ds_legend = ds not in added_ds_legend
        kis_hover = (
            f"<b>{ds}</b><br>"
            f"<b>Kisialiou</b><br>"
            f"rt: {kr['mean_runtime']:.2f}s<br>"
            f"impr: {kr['mean_improvement']:.2f}%"
        )
        kis_marker = dict(size=14, color=KIS_COLOR, symbol=symbol,
                          line=dict(width=1.5, color="white"))

        fig.add_trace(go.Scatter(
            x=[kr["mean_runtime"]], y=[kr["mean_improvement"]],
            mode="markers", marker=kis_marker,
            name=f"Kisialiou · {ds}",
            text=[kis_hover], hoverinfo="text",
            legendgroup=f"kis_{ds}", showlegend=show_ds_legend,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[kr["mean_runtime"]], y=[0.0],
            mode="markers", marker=kis_marker,
            name=f"Kisialiou · {ds}",
            text=[kis_hover], hoverinfo="text",
            legendgroup=f"kis_{ds}", showlegend=False,
        ), row=1, col=2)

        added_ds_legend.add(ds)

    # Zero reference line on gap subplot
    fig.add_hline(y=0, line=dict(color="#888", width=1, dash="dot"), row=1, col=2)

    fig.update_xaxes(title_text="runtime (s)", showgrid=True, gridcolor="#eee", row=1, col=1)
    fig.update_xaxes(title_text="runtime (s)", showgrid=True, gridcolor="#eee", row=1, col=2)
    fig.update_yaxes(title_text="improvement %", showgrid=True, gridcolor="#eee", row=1, col=1)
    fig.update_yaxes(title_text="gap vs Kisialiou %", showgrid=True, gridcolor="#eee",
                     zeroline=True, zerolinecolor="#aaa", row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(
                f"ALNS Sequence Grid — {size} datasets — {run_tag}<br>"
                f"<sup>colour = sequence · shape = dataset · circle/diamond/square = len 1/2/3 (shape shared with dataset)</sup>"
            ),
            font=dict(size=13),
        ),
        height=700,
        width=1600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            x=1.01, y=1, xanchor="left",
            font=dict(size=8),
            tracegroupgap=2,
        ),
    )

    out_path = run_dir / f"plot_sequence_dots_{size}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved: {out_path}")
