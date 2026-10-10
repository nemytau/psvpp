"""Analysis of the NORM 2026 state / action / reward ablation (scripts/run_norm2026_ablation.py).

Reads every finished evaluation (output/<prefix>__<tag>/results.csv + restart_results.csv)
and writes to output/<prefix>_analysis/:

  ratio_to_best_<size>.csv / summary.md   per-method ratio to the best-known cost per instance
                                          (1.0000 = best of every method run on that instance)
  tests_<size>.csv                        paired Wilcoxon signed-rank tests + bootstrap CIs of
                                          each variant vs the reference and vs Kisialiou
  behaviour_<size>.csv                    how the policy uses its actions: distinct actions and
                                          switches per restart, action entropy, top actions

Trained variants with several training seeds are scored on the per-instance mean cost over
seeds; the spread between seeds is reported separately.

    python scripts/analyze_norm2026_ablation.py [--prefix norm26]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_norm2026_ablation as queue  # noqa: E402

AXIS_ORDER = ("action", "state", "reward", "control")
REFERENCE = ("ref", "op_triple_v2")
LABELS = {
    ("ref", "op_triple_v2"): "Reference (Exp 8 config: triple / v3 state / v9 reward)",
    ("action", "improvement_chain_v2"): "Action: improvement chain",
    ("action", "op_triple_many_v1"): "Action: many (288 parameterised)",
    ("state", "null"): "State: none (constant)",
    ("state", "instance"): "State: instance only",
    ("state", "search"): "State: search process only",
    ("state", "solution"): "State: solution only",
    ("state", "all"): "State: all three groups",
    ("reward", "alns_score_v1"): "Reward: ALNS score",
    ("reward", "alns_score_time_v1"): "Reward: ALNS score + time",
    ("reward", "new_best_v1"): "Reward: new best (sparse)",
    ("reward", "best_gain_v1"): "Reward: best-cost gain",
    ("control", "random_triple"): "Control: uniform random (triple)",
    ("control", "random_many"): "Control: uniform random (many)",
    ("control", "fixed_rnd_ff_fcr"): "Control: fixed random+first-fit+FCR",
    ("control", "fixed_rnd_ff_reloc"): "Control: fixed random+first-fit+relocation",
}
KISIALIOU = "Kisialiou ALNS"
BASELINE = "Adaptive ALNS (no improvement)"


# --------------------------------------------------------------------------- statistics

def wilcoxon_signed_rank(diff: np.ndarray) -> Tuple[int, float, float]:
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie correction.

    Zero differences are dropped (Wilcoxon's convention). Returns (n_nonzero, z, p).
    """
    d = diff[np.abs(diff) > 1e-12]
    n = len(d)
    if n == 0:
        return 0, 0.0, 1.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(counts ** 3 - counts)) / 48.0
    if var <= 0:
        return n, 0.0, 1.0
    z = (w_plus - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return n, z, p


def bootstrap_mean_ci(values: np.ndarray, reps: int = 10000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvalues: List[float]) -> List[float]:
    order = np.argsort(pvalues)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ------------------------------------------------------------------------------ loading

def load_costs(prefix: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict], Dict[str, pd.DataFrame]]:
    """Per size: wide table of best cost (index = instance, columns = method / seed)."""
    queue._configure(prefix, None, queue.TIMESTEPS)
    per_size: Dict[str, Dict[str, pd.Series]] = defaultdict(dict)
    meta: Dict[str, Dict] = {}
    restarts: Dict[str, pd.DataFrame] = {}
    for job in queue.build_jobs():
        results = job.eval_dir / "results.csv"
        if not (job.eval_dir / "summary.json").exists() or not results.exists():
            continue
        df = pd.read_csv(results)
        rl = df[df["mode"] == "rl_improvement_mvp"].set_index("dataset")["best_cost"]
        column = f"{job.size}|{job.axis}|{job.variant}|s{job.seed}"  # size in the key: names repeat across sizes
        per_size[job.size][column] = rl
        meta[column] = {"axis": job.axis, "variant": job.variant, "seed": job.seed, "size": job.size,
                        "action": job.action}
        for mode, name in (("kisialiou", KISIALIOU), ("baseline", BASELINE)):
            if name not in per_size[job.size]:
                per_size[job.size][name] = df[df["mode"] == mode].set_index("dataset")["best_cost"]
        restart_path = job.eval_dir / "restart_results.csv"
        if restart_path.exists():
            r = pd.read_csv(restart_path, low_memory=False)
            restarts[column] = r[r["mode"] == "rl_improvement_mvp"].copy()
    tables = {size: pd.DataFrame(cols).sort_index() for size, cols in per_size.items()}
    return tables, meta, restarts


def method_means(table: pd.DataFrame, meta: Dict[str, Dict]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Collapse training seeds: per-instance mean cost per (axis, variant)."""
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for column in table.columns:
        if column in meta:
            groups[(meta[column]["axis"], meta[column]["variant"])].append(column)
    out = pd.DataFrame(index=table.index)
    seeds: Dict[str, List[str]] = {}
    for name in (KISIALIOU, BASELINE):
        if name in table:
            out[name] = table[name]
    for key, columns in groups.items():
        label = LABELS.get(key, f"{key[0]}: {key[1]}")
        out[label] = table[columns].mean(axis=1)
        seeds[label] = columns
    return out, seeds


# ----------------------------------------------------------------------------- analysis

def ratio_table(costs: pd.DataFrame, raw: pd.DataFrame, seeds: Dict[str, List[str]]) -> pd.DataFrame:
    complete = costs.dropna()
    best_known = raw.loc[complete.index].min(axis=1)  # best over every method and every seed
    ratio = complete.div(best_known, axis=0)
    rows = []
    for method in complete.columns:
        r = ratio[method]
        row = {
            "method": method,
            "n": len(r),
            "mean_ratio": r.mean(),
            "median_ratio": r.median(),
            "max_ratio": r.max(),
            "wins_vs_kisialiou": int((complete[method] < complete[KISIALIOU] - 1e-6).sum()) if KISIALIOU in complete else None,
            "losses_vs_kisialiou": int((complete[method] > complete[KISIALIOU] + 1e-6).sum()) if KISIALIOU in complete else None,
            "training_seeds": len(seeds.get(method, [])),
        }
        cols = seeds.get(method, [])
        if len(cols) > 1:
            per_seed = raw.loc[complete.index, cols].div(best_known, axis=0).mean()
            row["seed_mean_ratio_min"] = per_seed.min()
            row["seed_mean_ratio_max"] = per_seed.max()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(costs: pd.DataFrame, reference_label: str) -> pd.DataFrame:
    complete = costs.dropna()
    rows = []
    for against in (reference_label, KISIALIOU):
        if against not in complete:
            continue
        tested = [m for m in complete.columns if m not in (against, KISIALIOU, BASELINE) or (against != KISIALIOU and m == BASELINE)]
        block = []
        for method in tested:
            if method == against:
                continue
            rel = (complete[method] / complete[against] - 1.0).to_numpy() * 100.0  # % cost difference, <0 = better
            n, z, p = wilcoxon_signed_rank(np.log(complete[method].to_numpy() / complete[against].to_numpy()))
            lo, hi = bootstrap_mean_ci(rel)
            block.append({
                "method": method, "against": against, "n_instances": len(rel), "n_nonzero": n,
                "mean_cost_diff_pct": rel.mean(), "ci95_low": lo, "ci95_high": hi,
                "median_cost_diff_pct": float(np.median(rel)),
                "better": int((rel < -1e-6).sum()), "worse": int((rel > 1e-6).sum()), "z": z, "p": p,
            })
        for row, p_holm in zip(block, holm([b["p"] for b in block])):
            row["p_holm"] = p_holm
        rows += block
    return pd.DataFrame(rows)


def behaviour(restarts: Dict[str, pd.DataFrame], meta: Dict[str, Dict], size: str) -> pd.DataFrame:
    rows = []
    decoders: Dict[str, List[str]] = {}
    for column, r in restarts.items():
        if meta[column]["size"] != size or "action_hist" not in r:
            continue
        hist: Counter = Counter()
        for raw in r["action_hist"].dropna():
            hist.update({int(k): v for k, v in json.loads(raw).items()})
        total = sum(hist.values())
        if not total:
            continue
        action_module = meta[column]["action"]
        if action_module not in decoders:
            decoders[action_module] = _decode_actions(action_module, size)
        names = decoders[action_module]
        probs = np.array([v / total for v in hist.values()])
        entropy = float(-(probs * np.log(probs)).sum())
        top = hist.most_common(3)
        steps = r["policy_steps"].sum()
        rows.append({
            "run": column, "axis": meta[column]["axis"], "variant": meta[column]["variant"], "seed": meta[column]["seed"],
            "restarts": len(r),
            "mean_distinct_actions_per_restart": r["distinct_actions"].mean(),
            "switch_rate": r["action_switches"].sum() / max(steps, 1),
            "actions_used_overall": len(hist),
            "action_space_size": len(names),
            "normalised_entropy": entropy / math.log(len(names)) if len(names) > 1 else 0.0,
            "top_action_share": top[0][1] / total,
            "top_actions": "; ".join(f"{names[a] if a < len(names) else a} ({c / total:.0%})" for a, c in top),
        })
    return pd.DataFrame(rows)


def _decode_actions(action_module: str, size: str) -> List[str]:
    from rl.dataset_manager import GeneratedDatasetManager
    from rl.rl_alns_environment import ALNSEnvironment

    path = GeneratedDatasetManager().prepare_size(size)["test"][0]
    env = ALNSEnvironment(problem_instance=path, max_iterations=2, seed=0, action_module=action_module,
                          state_module="features_null_v1", reward_module="new_best_v1",
                          size_bucket="small" if size == "small" else "large")
    short = lambda s: (s or "-").replace("_kisialiou_small_v1", "").replace("_kisialiou_large_v1", "").replace("_insertion_v1", "")
    names = []
    for a in range(env.action_impl.n()):
        d, r, i = env.action_impl.id_to_action(a)
        parts = [short(env.destroy_operators[d]) if d is not None else "adaptive D/R",
                 short(env.repair_operators[r]) if r is not None else None,
                 short(env.improvement_operators[i]) if i is not None else "no improvement"]
        names.append(" + ".join(p for p in parts if p))
    env.close()
    return names


def _md(df: pd.DataFrame, digits: int = 4) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="norm26")
    args = ap.parse_args()

    out_dir = REPO / "output" / f"{args.prefix}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables, meta, restarts = load_costs(args.prefix)
    md = ["# NORM 2026 ablation — results", ""]
    for size in ("small", "medium"):
        if size not in tables:
            continue
        raw = tables[size]
        costs, seeds = method_means(raw, meta)
        reference_label = LABELS[REFERENCE]
        ratios = ratio_table(costs, raw, seeds)
        tests = paired_tests(costs, reference_label)
        beh = behaviour(restarts, meta, size)
        ratios.to_csv(out_dir / f"ratio_to_best_{size}.csv", index=False)
        tests.to_csv(out_dir / f"tests_{size}.csv", index=False)
        beh.to_csv(out_dir / f"behaviour_{size}.csv", index=False)

        md += [f"## {size.capitalize()} ({len(costs.dropna())} instances)", ""]
        md += ["Ratio to the best-known cost per instance (1.0000 = best of all methods/seeds run on it).", ""]
        md += [_md(ratios, 4), ""]
        if not tests.empty:
            md += ["Paired tests (cost difference in %, negative = better than the comparison; "
                   "Wilcoxon signed-rank, Holm-adjusted within each comparison):", ""]
            md += [_md(tests, 4), ""]
        if not beh.empty:
            md += ["Policy behaviour during evaluation:", "", _md(beh, 3), ""]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
