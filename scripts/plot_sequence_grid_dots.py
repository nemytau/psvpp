"""
Dot plots for ALNS sequence grid results.
  Plot 1: x=runtime(s), y=mean improvement %  (avg across seeds per dataset)
  Plot 2: x=runtime(s), y=median gap vs Kisialiou %  (median across seeds per dataset)
Each dot = one dataset, one sequence.
Kisialiou reference shown as distinct black diamonds.
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

# ── Kisialiou improvement implied by each row ────────────────────────────────
df["kisialiou_improvement_pct"] = (
    (df["initial_cost"] - df["reference_kisialiou_best_cost"])
    / df["initial_cost"] * 100
)

# ── Aggregate sequence runs: average across seeds per (dataset, sequence) ────
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

# ── Kisialiou reference: average across seeds per dataset ────────────────────
#    runtime = runtime_budget_seconds (= Kisialiou actual runtime for that seed)
kis_agg = (
    df.groupby(["size", "dataset"])
    .agg(
        mean_runtime=("runtime_budget_seconds", "mean"),
        mean_improvement=("kisialiou_improvement_pct", "mean"),
        n_seeds=("seed", "count"),
    )
    .reset_index()
)
kis_agg["median_gap"] = 0.0
kis_agg["sequence_key"] = "kisialiou"

# ── Color scheme ──────────────────────────────────────────────────────────────
unique_seqs = sorted(seq_agg["sequence_key"].unique())
# Use a large qualitative palette; cycle if needed
base_palette = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
)
seq_colors = {seq: base_palette[i % len(base_palette)] for i, seq in enumerate(unique_seqs)}

KIS_COLOR = "#000000"
KIS_MARKER = dict(size=14, color=KIS_COLOR, symbol="diamond",
                  line=dict(width=1.5, color="white"))

SIZE_SYMBOL = {"small": "circle", "medium": "square"}
SIZE_LABEL = {"small": "small", "medium": "medium"}

# ── Build figure ──────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Improvement % (mean across seeds)",
        "Gap vs Kisialiou % (median across seeds)",
    ],
    horizontal_spacing=0.10,
)

# Helper: build hover text
def seq_hover(rows_iter):
    return [
        f"<b>{r.dataset}</b> ({r.size})<br>"
        f"seq: {r.sequence_key}<br>"
        f"rt: {r.mean_runtime:.2f}s<br>"
        f"impr: {r.mean_improvement:.2f}%<br>"
        f"median gap: {r.median_gap:+.2f}%<br>"
        f"n_seeds: {r.n_seeds}"
        for r in rows_iter
    ]

def kis_hover(rows_iter):
    return [
        f"<b>{r.dataset}</b> ({r.size})<br>"
        f"<b>Kisialiou</b><br>"
        f"rt: {r.mean_runtime:.2f}s<br>"
        f"impr: {r.mean_improvement:.2f}%<br>"
        f"gap: 0.0%<br>"
        f"n_seeds: {r.n_seeds}"
        for r in rows_iter
    ]

# One trace per sequence across both subplots (share legend via legendgroup)
for seq, grp in seq_agg.groupby("sequence_key"):
    color = seq_colors[seq]
    length = int(grp["sequence_length"].iloc[0])
    symbols = [SIZE_SYMBOL.get(s, "circle") for s in grp["size"]]
    hover1 = seq_hover(grp.itertuples())
    hover2 = [
        f"<b>{r.dataset}</b> ({r.size})<br>"
        f"seq: {r.sequence_key}<br>"
        f"rt: {r.mean_runtime:.2f}s<br>"
        f"median gap: {r.median_gap:+.2f}%<br>"
        f"mean impr: {r.mean_improvement:.2f}%<br>"
        f"n_seeds: {r.n_seeds}"
        for r in grp.itertuples()
    ]

    marker_base = dict(
        size=9,
        color=color,
        symbol=symbols,
        opacity=0.82,
        line=dict(width=0.6, color="white"),
    )

    fig.add_trace(go.Scatter(
        x=grp["mean_runtime"], y=grp["mean_improvement"],
        mode="markers",
        marker=marker_base,
        name=f"[{length}] {seq}",
        text=hover1, hoverinfo="text",
        legendgroup=seq, showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=grp["mean_runtime"], y=grp["median_gap"],
        mode="markers",
        marker=marker_base,
        name=f"[{length}] {seq}",
        text=hover2, hoverinfo="text",
        legendgroup=seq, showlegend=False,
    ), row=1, col=2)

# Kisialiou reference dots
kis_symbols = [SIZE_SYMBOL.get(s, "diamond") for s in kis_agg["size"]]
kis_marker_plot = dict(**KIS_MARKER)
kis_marker_plot["symbol"] = kis_symbols

fig.add_trace(go.Scatter(
    x=kis_agg["mean_runtime"], y=kis_agg["mean_improvement"],
    mode="markers",
    marker=dict(size=14, color=KIS_COLOR, symbol=kis_symbols,
                line=dict(width=1.5, color="white")),
    name="Kisialiou",
    text=kis_hover(kis_agg.itertuples()),
    hoverinfo="text",
    legendgroup="kisialiou", showlegend=True,
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=kis_agg["mean_runtime"], y=np.zeros(len(kis_agg)),
    mode="markers",
    marker=dict(size=14, color=KIS_COLOR, symbol=kis_symbols,
                line=dict(width=1.5, color="white")),
    name="Kisialiou",
    text=kis_hover(kis_agg.itertuples()),
    hoverinfo="text",
    legendgroup="kisialiou", showlegend=False,
), row=1, col=2)

# ── Zero line on plot 2 ───────────────────────────────────────────────────────
fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"), row=1, col=2)

# ── Layout ────────────────────────────────────────────────────────────────────
run_tag = run_dir.name.replace("alns_sequence_grid_", "")
fig.update_layout(
    title=dict(
        text=f"ALNS Sequence Grid — {run_tag}",
        font=dict(size=15),
    ),
    height=650,
    width=1700,
    legend=dict(
        title=dict(text="Sequence (circle=small, square=medium)", font=dict(size=10)),
        x=1.01, y=1, xanchor="left",
        font=dict(size=9),
        tracegroupgap=2,
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

for col in (1, 2):
    fig.update_xaxes(
        title_text="Mean runtime (s)",
        showgrid=True, gridcolor="#eeeeee",
        zeroline=True, zerolinecolor="#cccccc",
        row=1, col=col,
    )

fig.update_yaxes(
    title_text="Mean improvement %",
    showgrid=True, gridcolor="#eeeeee",
    row=1, col=1,
)
fig.update_yaxes(
    title_text="Median gap vs Kisialiou %",
    showgrid=True, gridcolor="#eeeeee",
    zeroline=True, zerolinecolor="#999999",
    row=1, col=2,
)

out_path = run_dir / "plot_sequence_dots.html"
fig.write_html(str(out_path), include_plotlyjs="cdn")
print(f"Saved: {out_path}")
