"""Slide figures for the NORM 2026 state / action / reward ablation.

Reads the same evaluation outputs as scripts/analyze_norm2026_ablation.py plus the PPO
TensorBoard logs, and writes PNG + SVG figures to output/<prefix>_analysis/figures/:

  ratio_<size>.png        mean ratio-to-best-known (95% bootstrap CI) per method, one panel
                          per design axis; reference highlighted, Kisialiou / baseline as context
  entropy_<size>.png      normalised policy entropy over training, one small multiple per
                          variant against the reference (does the policy collapse?)
  behaviour_<size>.png    how the trained policy uses its actions at evaluation time:
                          share of steps that switch operator, distinct actions per restart

    python scripts/plot_norm2026_ablation.py [--prefix norm26]
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_norm2026_ablation as analysis  # noqa: E402
import run_norm2026_ablation as queue  # noqa: E402

# Reference palette (dataviz skill, light mode): slots 1-2 validated, plus chrome/ink tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
CONTEXT = "#898781"          # muted gray for comparators / reference context lines
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

AXES = (("action", "Action space"), ("state", "State"), ("reward", "Reward"))
ACTION_SPACE_SIZE = {"op_triple_v2": 48, "op_triple_many_v1": 288, "improvement_chain_v2": 5}


def _style() -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _save(fig, out_dir: Path, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def _short(label: str) -> str:
    return label.split(": ", 1)[1] if ": " in label else label


# ------------------------------------------------------------------------- ratio plot

def ratio_figure(size: str, costs: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> None:
    complete = costs.dropna()
    if complete.empty:
        return
    best_known = raw.loc[complete.index].min(axis=1)
    ratio = complete.div(best_known, axis=0)
    ref = analysis.LABELS[analysis.REFERENCE]
    rng = np.random.default_rng(0)

    def stats(method: str) -> Tuple[float, float, float]:
        values = ratio[method].to_numpy()
        boots = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        return values.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    panels = []
    for axis, title in AXES:
        variants = [m for m in complete.columns if m.lower().startswith(axis + ":")]
        if variants:
            panels.append((title, variants))
    controls = [m for m in complete.columns if m.startswith("Control:")]
    if controls:
        panels.append(("Non-learned controls", controls))
    if not panels:
        return

    context = [m for m in (analysis.KISIALIOU, analysis.BASELINE) if m in complete]
    all_methods = [m for _, vs in panels for m in vs] + context + [ref]
    lo = min(stats(m)[1] for m in all_methods if m in complete)
    hi = max(stats(m)[2] for m in all_methods if m in complete)
    pad = (hi - lo) * 0.08 or 0.002

    def draw(ax, title: str, variants: List[str]) -> None:
        rows = context + [ref] + sorted(variants, key=lambda m: stats(m)[0], reverse=True)
        rows = [m for m in rows if m in complete]
        for y, method in enumerate(rows):
            mean, c_lo, c_hi = stats(method)
            color = ORANGE if method == ref else (CONTEXT if method in context else BLUE)
            ax.plot([c_lo, c_hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([mean], [y], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([("Reference (Exp 8)" if m == ref else _short(m)) for m in rows])
        ax.set_title(title, loc="left")
        ax.set_xlim(lo - pad, hi + pad)
        ax.grid(axis="y", visible=False)
        ax.axvline(1.0, color=AXIS, linewidth=1, zorder=1)
        ax.set_xlabel("cost ÷ best known")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=c, markeredgecolor=SURFACE)
               for c in (ORANGE, BLUE, CONTEXT)]
    legend_labels = ["Reference (Exp 8)", "Design variant / control", "Kisialiou ALNS, adaptive ALNS"]
    title = (f"{size.capitalize()} instances — mean cost ÷ best known, 95% CI, lower is better "
             f"({len(complete)} instances)")

    # Overview: every axis side by side.
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.9), sharex=True)
    for ax, (panel_title, variants) in zip(np.atleast_1d(axes), panels):
        draw(ax, panel_title, variants)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13, fontweight="semibold")
    fig.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(0.01, -0.08), ncol=3,
               frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    _save(fig, out_dir, f"ratio_{size}")

    # One figure per axis, sized for a slide.
    for panel_title, variants in panels:
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * (len(variants) + 3) + 1.4))
        draw(ax, panel_title, variants)
        ax.set_title(f"{panel_title} — {size} instances (lower is better)", loc="left")
        ax.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=3,
                  frameon=False, labelcolor=INK_2, fontsize=9)
        fig.tight_layout()
        _save(fig, out_dir, f"ratio_{size}_{panel_title.split()[0].lower().replace('-', '')}")


# ---------------------------------------------------------------------- entropy plot

def _entropy_curve(job: queue.Job) -> Optional[pd.Series]:
    pattern = str(REPO / "logs" / f"{queue.PREFIX}__{job.tag}__train" / "train" / "PPO_*" / "events*")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    if "train/entropy_loss" not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars("train/entropy_loss")
    n_actions = ACTION_SPACE_SIZE.get(job.action, 48)
    return pd.Series([-e.value / math.log(n_actions) for e in events], index=[e.step for e in events])


def entropy_figure(size: str, out_dir: Path) -> None:
    curves: Dict[str, List[pd.Series]] = {}
    for job in queue.build_jobs():
        if job.size != size or not job.needs_training:
            continue
        curve = _entropy_curve(job)
        if curve is not None:
            label = analysis.LABELS.get((job.axis, job.variant), f"{job.axis}: {job.variant}")
            curves.setdefault(label, []).append(curve)
    ref = analysis.LABELS[analysis.REFERENCE]
    variants = [lab for lab in curves if lab != ref]
    if not variants:
        return
    cols = 4
    rows = math.ceil(len(variants) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, label in zip(axes, variants):
        for curve in curves.get(ref, []):
            ax.plot(curve.index / 1000, curve.values, color=CONTEXT, linewidth=1.5)
        for curve in curves[label]:
            ax.plot(curve.index / 1000, curve.values, color=BLUE, linewidth=2)
        ax.set_title(_short(label), loc="left", fontsize=10.5)
        ax.set_ylim(0, 1.02)
    for ax in axes[len(variants):]:
        ax.set_visible(False)
    for ax in axes[::cols]:
        ax.set_ylabel("entropy ÷ max")
    for ax in axes[-cols:]:
        ax.set_xlabel("training steps (k)")
    fig.suptitle(f"{size.capitalize()} — policy entropy during training (blue: variant, one line per "
                 f"training seed; gray: reference). 1 = uniform over all actions, 0 = always the same action",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.tight_layout()
    _save(fig, out_dir, f"entropy_{size}")


# -------------------------------------------------------------------- behaviour plot

def behaviour_figure(size: str, beh: pd.DataFrame, out_dir: Path) -> None:
    if beh.empty:
        return
    grouped = beh.groupby(["axis", "variant"]).agg(
        switch_rate=("switch_rate", "mean"),
        distinct=("mean_distinct_actions_per_restart", "mean"),
    ).reset_index()
    grouped["label"] = [
        "Reference (Exp 8)" if (a, v) == analysis.REFERENCE else _short(analysis.LABELS.get((a, v), f"{a}: {v}"))
        for a, v in zip(grouped["axis"], grouped["variant"])
    ]
    grouped = grouped.sort_values("switch_rate")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(grouped) + 1.6), sharey=True)
    for ax, column, xlabel in (
        (axes[0], "switch_rate", "share of steps that switch to a different action"),
        (axes[1], "distinct", "distinct actions used per restart (mean)"),
    ):
        colors = [ORANGE if lab == "Reference (Exp 8)" else BLUE for lab in grouped["label"]]
        ax.barh(range(len(grouped)), grouped[column], height=0.55, color=colors)
        ax.set_yticks(range(len(grouped)))
        ax.set_yticklabels(grouped["label"])
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        for y, value in enumerate(grouped[column]):
            ax.text(value, y, f" {value:.2f}" if column == "switch_rate" else f" {value:.1f}",
                    va="center", color=INK_2, fontsize=9)
    fig.suptitle(f"{size.capitalize()} — does the trained policy adapt its operator choice within a run?",
                 x=0.01, ha="left", color=INK, fontsize=13, fontweight="semibold")
    fig.tight_layout()
    _save(fig, out_dir, f"behaviour_{size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="norm26")
    args = ap.parse_args()
    _style()
    out_dir = REPO / "output" / f"{args.prefix}_analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables, meta, restarts = analysis.load_costs(args.prefix)
    for size in ("small", "medium"):
        if size in tables:
            costs, _ = analysis.method_means(tables[size], meta)
            ratio_figure(size, costs, tables[size], out_dir)
            behaviour_figure(size, analysis.behaviour(restarts, meta, size), out_dir)
        entropy_figure(size, out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
