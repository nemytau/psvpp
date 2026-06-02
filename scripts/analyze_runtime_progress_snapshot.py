#!/usr/bin/env python3
"""Build analysis + summary for runtime-progress logs (small/medium snapshot)."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

RUN_ID = "rl_improvement_mvp_20260317_130103__joint_small_medium_only_matchk5_r300_evalonly_snapshot"
LOG_DIR = Path("logs") / RUN_ID
OUT_DIR = Path(
    "output/rl_improvement_mvp_20260317_130103__joint_small_medium_large_antirep01_matchk5_r300_evalonly/"
    "runtime_progress_per_dataset_small_medium"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAT = re.compile(
    r"operator_usage_(rl_improvement_mvp|baseline|kisialiou)__(small|medium)__(.+)__seed(\d+)\.csv$"
)


def _to_float(v: str | None, default: float = 0.0) -> float:
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def file_metrics(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {
            "best_cost": float("inf"),
            "runtime_seconds": 0.0,
            "restart_count": 0,
            "iterations": 0,
        }

    by_restart = defaultdict(list)
    for r in rows:
        restart = int(_to_float(r.get("restart_id"), 0.0))
        iteration = int(_to_float(r.get("iteration"), -1.0))
        by_restart[restart].append((iteration, r))

    restart_best = []
    restart_runtime = []
    total_iterations = 0

    for restart, entries in by_restart.items():
        entries.sort(key=lambda x: x[0])
        best = float("inf")
        runtime = 0.0
        seen_iter = set()
        for iteration, r in entries:
            best = min(best, _to_float(r.get("cost_best"), best))
            if iteration not in seen_iter:
                runtime += _to_float(r.get("operator_elapsed_ms"), 0.0)
                seen_iter.add(iteration)
        restart_best.append(best)
        restart_runtime.append(runtime / 1000.0)
        total_iterations += len(seen_iter)

    return {
        "best_cost": min(restart_best) if restart_best else float("inf"),
        "runtime_seconds": sum(restart_runtime),
        "restart_count": len(by_restart),
        "iterations": total_iterations,
    }


records = []
for p in sorted(LOG_DIR.glob("operator_usage_*__seed*.csv")):
    m = PAT.match(p.name)
    if not m:
        continue
    mode, size, dataset, seed = m.groups()
    metrics = file_metrics(p)
    records.append(
        {
            "mode": mode,
            "size": size,
            "dataset": dataset,
            "seed": int(seed),
            **metrics,
        }
    )

# Map per mode by dataset-seed
by_mode = defaultdict(dict)
for r in records:
    key = (r["size"], r["dataset"], r["seed"])
    by_mode[r["mode"]][key] = r

keys = sorted(set.intersection(*(set(by_mode[m].keys()) for m in ["rl_improvement_mvp", "baseline", "kisialiou"])))

comparisons = []
for key in keys:
    rr = by_mode["rl_improvement_mvp"][key]
    bb = by_mode["baseline"][key]
    kk = by_mode["kisialiou"][key]

    rl_best = rr["best_cost"]
    base_best = bb["best_cost"]
    kis_best = kk["best_cost"]

    comp = {
        "size": key[0],
        "dataset": key[1],
        "seed": key[2],
        "rl_best": rl_best,
        "baseline_best": base_best,
        "kisialiou_best": kis_best,
        "rl_gap_vs_baseline_pct": (base_best - rl_best) / base_best * 100.0,
        "rl_gap_vs_kisialiou_pct": (kis_best - rl_best) / kis_best * 100.0,
        "rl_runtime_s": rr["runtime_seconds"],
        "baseline_runtime_s": bb["runtime_seconds"],
        "kisialiou_runtime_s": kk["runtime_seconds"],
    }
    comparisons.append(comp)


def summarize(items: list[dict]) -> dict:
    if not items:
        return {}
    def col(name: str) -> list[float]:
        return [float(x[name]) for x in items]

    rl_vs_base = col("rl_gap_vs_baseline_pct")
    rl_vs_kis = col("rl_gap_vs_kisialiou_pct")
    rl_t = col("rl_runtime_s")
    base_t = col("baseline_runtime_s")
    kis_t = col("kisialiou_runtime_s")

    return {
        "n": len(items),
        "median_gap_vs_baseline_pct": median(rl_vs_base),
        "median_gap_vs_kisialiou_pct": median(rl_vs_kis),
        "wins_vs_baseline": sum(1 for v in rl_vs_base if v > 0),
        "wins_vs_kisialiou": sum(1 for v in rl_vs_kis if v > 0),
        "median_runtime_rl_s": median(rl_t),
        "median_runtime_baseline_s": median(base_t),
        "median_runtime_kisialiou_s": median(kis_t),
        "runtime_ratio_rl_over_baseline": median(rl_t) / max(median(base_t), 1e-9),
        "runtime_ratio_rl_over_kisialiou": median(rl_t) / max(median(kis_t), 1e-9),
    }


overall = summarize(comparisons)
by_size = {}
for size in ["small", "medium"]:
    by_size[size] = summarize([x for x in comparisons if x["size"] == size])

out = {
    "run_id": RUN_ID,
    "source_log_dir": str(LOG_DIR),
    "matched_dataset_seed_pairs": len(keys),
    "overall": overall,
    "by_size": by_size,
}

(OUT_DIR / "analysis_small_medium_summary.json").write_text(json.dumps(out, indent=2))

lines = []
lines.append("Runtime Progress Analysis (small + medium)")
lines.append("=" * 44)
lines.append(f"run_id: {RUN_ID}")
lines.append(f"matched dataset-seed pairs: {len(keys)}")
lines.append("")
lines.append("Overall")
lines.append(
    f"- RL vs baseline: median gap {overall['median_gap_vs_baseline_pct']:.3f}% | wins {overall['wins_vs_baseline']}/{overall['n']}"
)
lines.append(
    f"- RL vs Kisialiou: median gap {overall['median_gap_vs_kisialiou_pct']:.3f}% | wins {overall['wins_vs_kisialiou']}/{overall['n']}"
)
lines.append(
    f"- Median runtime (s): RL {overall['median_runtime_rl_s']:.3f}, baseline {overall['median_runtime_baseline_s']:.3f}, Kisialiou {overall['median_runtime_kisialiou_s']:.3f}"
)
lines.append(
    f"- Runtime ratio: RL/baseline {overall['runtime_ratio_rl_over_baseline']:.3f}x, RL/Kisialiou {overall['runtime_ratio_rl_over_kisialiou']:.3f}x"
)
lines.append("")
for size in ["small", "medium"]:
    s = by_size[size]
    lines.append(size.capitalize())
    lines.append(
        f"- RL vs baseline: median gap {s['median_gap_vs_baseline_pct']:.3f}% | wins {s['wins_vs_baseline']}/{s['n']}"
    )
    lines.append(
        f"- RL vs Kisialiou: median gap {s['median_gap_vs_kisialiou_pct']:.3f}% | wins {s['wins_vs_kisialiou']}/{s['n']}"
    )
    lines.append(
        f"- Median runtime (s): RL {s['median_runtime_rl_s']:.3f}, baseline {s['median_runtime_baseline_s']:.3f}, Kisialiou {s['median_runtime_kisialiou_s']:.3f}"
    )
    lines.append("")

(OUT_DIR / "analysis_small_medium_summary.txt").write_text("\n".join(lines))
print(f"Saved: {OUT_DIR / 'analysis_small_medium_summary.json'}")
print(f"Saved: {OUT_DIR / 'analysis_small_medium_summary.txt'}")
