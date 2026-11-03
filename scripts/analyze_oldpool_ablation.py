#!/usr/bin/env python3
"""Old-pool ablation of the Experiment 10: what did first-fit insertion do to the references?

Unit of analysis = one dataset: the best cost of the whole consecutive search on that instance
(restarts are part of the search, never a unit of comparison). Every table names its denominator.

Sources (all on Experiment 10's 150 test instances unless stated otherwise):
  old pool   runs/baseline_kisialiou_cache/rerun26_oldpool_v1   Kisialiou (5 restarts) + adaptive ALNS (no improvement operators)
                                                                 with the ORIGINAL repair pool, the original reference protocol
  first-fit  runs/baseline_kisialiou_cache/rerun26_v1           Experiment 10's references (adaptive_weighted_v2, first-fit in the pool)
  RL         output/rerun26__{main__delta_cost_v9,alt__alns_score_time_v1}__<size>__s42  (results.csv, restart_results.csv)
  Exp 8      the original Experiment 8 (its own 50 test instances per size, original pool) -- orientation only

Compute: each reference defines its own budget (Kisialiou's 5 restarts), so the old pool can use less or more CPU than the
first-fit pool and than the RL models (which were given the first-fit Kisialiou's time). The tables say so, and every
comparison is also given at EQUAL compute: runs are sequences of restarts, so a run can be cut to a smaller budget from its
restart rows (complete restarts only; the cut restart is lost, which slightly handicaps the longer run).

    python scripts/analyze_oldpool_ablation.py                 # whatever is finished so far
    python scripts/analyze_oldpool_ablation.py --sizes small   # one size
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
OLD_CACHE_DIR = REPO / "runs" / "baseline_kisialiou_cache" / "rerun26_oldpool_v1"
FF_CACHE_DIR = REPO / "runs" / "baseline_kisialiou_cache" / "rerun26_v1"
DESIGN = REPO / "py_alns" / "resource" / "test_set_v2_design.csv"
SIZES = ("small", "medium", "large")
SEED = 42
RL_JOBS = {"RL main": "rerun26__main__delta_cost_v9__{size}__s42", "RL ASt": "rerun26__alt__alns_score_time_v1__{size}__s42"}
EXP8 = {
    "small": "rl_improvement_mvp_20260906_063629__exp_reward_v9_timepenalty_op_triple_v2_small_ph300_50ds1seed",
    "medium": "rl_improvement_mvp_20260906_070345__exp_reward_v9_timepenalty_op_triple_v2_medium_ph300_50ds1seed",
    "large": "rl_improvement_mvp_20260907_125355__exp_reward_v9_timepenalty_op_triple_v2_large_ph300_50ds1seed_evalonly",
}
EPS = 1e-9
_memo: Dict[str, Tuple[float, object]] = {}


# ----------------------------------------------------------------------------- loading
def _load_json(path: Path) -> Optional[dict]:
    """JSON file, memoised by modification time; None if it is missing or being written right now."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _memo.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]  # type: ignore[return-value]
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    _memo[str(path)] = (mtime, data)
    return data


def load_cache(cache_dir: Path, size: str) -> Dict[str, dict]:
    """{dataset: {'kisialiou': entry, 'baseline': entry}} from the merged cache, else from the finished parts."""
    merged = _load_json(cache_dir / f"cache_{size}_seed{SEED}.json")
    entries: Dict[str, dict] = {}
    if merged is not None:
        entries = dict(merged)
    else:
        for part in sorted(glob.glob(str(cache_dir / f"{size}_part*.json"))):
            data = _load_json(Path(part))
            if data:
                entries.update(data)
    return {k.rsplit("__seed", 1)[0]: v for k, v in entries.items()}


def load_results(directory: str, mode_map: Dict[str, str]) -> Dict[str, Dict[str, dict]]:
    """results.csv of an evaluation: {dataset: {label: row}}."""
    path = REPO / "output" / directory / "results.csv"
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, dict]] = collections.defaultdict(dict)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = mode_map.get(row["mode"])
            if label:
                out[row["dataset"]][label] = row
    return out


def load_rl_restarts(directory: str) -> Dict[str, List[Tuple[float, float]]]:
    """{dataset: [(search CPU incl. decision charge, best cost of the restart), ...]} in restart order (RL rows only)."""
    path = REPO / "output" / directory / "restart_results.csv"
    if not path.exists():
        return {}
    out: Dict[str, List[Tuple[int, float, float]]] = collections.defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == "rl_improvement_mvp":
                out[row["dataset"]].append((int(row["restart_id"]), float(row["search_cpu_seconds"]), float(row["best_cost"])))
    return {d: [(cpu, cost) for _, cpu, cost in sorted(rows)] for d, rows in out.items()}


def load_design() -> Dict[str, dict]:
    with DESIGN.open(newline="") as handle:
        return {r["name"]: r for r in csv.DictReader(handle)}


# ----------------------------------------------------------------------------- per-dataset values
def _argmin_restart(entry: dict) -> dict:
    return min(entry["restart_rows"], key=lambda r: float(r["best_cost"]))


def best_cost(entry: dict) -> float:
    return float(entry["best_cost"])


def best_vessels(entry: dict) -> float:
    return float(_argmin_restart(entry)["best_num_vessels_used"])


def _prefix_best(cpu_and_cost: List[Tuple[float, float]], budget: float) -> float:
    """Best cost over the restarts finished within `budget` search-CPU seconds (at least the first restart)."""
    elapsed, best = 0.0, None
    for cpu, cost in cpu_and_cost:
        elapsed += cpu
        if best is not None and elapsed > budget * (1 + 1e-6):
            break
        best = cost if best is None else min(best, cost)
    return float(best)


def best_at(entry: dict, budget: float) -> float:
    """A reference run cut to `budget` CPU seconds: its result if it is no longer than that, else a prefix of its restarts."""
    if budget >= float(entry["search_cpu_seconds"]) * (1 - EPS):
        return best_cost(entry)
    return _prefix_best([(float(r["search_cpu_seconds"]), float(r["best_cost"])) for r in entry["restart_rows"]], budget)


def gmean(values: List[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def collect(cache_dir: Path, size: str, with_rl_restarts: bool = False) -> Dict[str, dict]:
    """Per dataset: costs / fleets / budgets of the old-pool and first-fit references and of the RL models."""
    old, ff = load_cache(cache_dir, size), load_cache(FF_CACHE_DIR, size)
    rl = {label: load_results(job.format(size=size), {"rl_improvement_mvp": label}) for label, job in RL_JOBS.items()}
    rl_restarts = load_rl_restarts(RL_JOBS["RL main"].format(size=size)) if with_rl_restarts else {}
    out: Dict[str, dict] = {}
    for dataset, entry in old.items():
        if dataset not in ff or "kisialiou" not in entry or "baseline" not in entry:
            continue
        k_old, a_old, k_ff, a_ff = entry["kisialiou"], entry["baseline"], ff[dataset]["kisialiou"], ff[dataset]["baseline"]
        row = {
            "K_old": best_cost(k_old), "A_old": best_cost(a_old), "K_ff": best_cost(k_ff), "A_ff": best_cost(a_ff),
            "VK_old": best_vessels(k_old), "VA_old": best_vessels(a_old), "VK_ff": best_vessels(k_ff), "VA_ff": best_vessels(a_ff),
            "T_old": float(k_old["search_cpu_seconds"]), "T_ff": float(k_ff["search_cpu_seconds"]),
            "restarts_K_old": len(k_old["restart_rows"]), "initial": float(k_old["initial_cost"]),
        }
        budget = min(row["T_old"], row["T_ff"])  # equal compute = the smaller of the two budgets
        row["B"] = budget
        row["K_old@B"], row["K_ff@B"] = best_at(k_old, budget), best_at(k_ff, budget)
        row["A_old@B"], row["A_ff@B"] = best_at(a_old, budget), best_at(a_ff, budget)
        for label, results in rl.items():
            r = results.get(dataset, {}).get(label)
            if r is not None:
                row[label] = float(r["best_cost"])
                row["V " + label] = float(r["best_num_vessels_used"])
        if dataset in rl_restarts and "RL main" in row:
            row["RL main@B"] = row["RL main"] if budget >= row["T_ff"] * (1 - EPS) else _prefix_best(rl_restarts[dataset], budget)
        out[dataset] = row
    return out


# ----------------------------------------------------------------------------- tables
def _ratio_to_best(rows: List[dict], methods: List[str]) -> Tuple[Dict[str, float], Dict[str, int]]:
    ratios = {m: [] for m in methods}
    wins = {m: 0 for m in methods}
    for r in rows:
        best = min(r[m] for m in methods)
        for m in methods:
            ratios[m].append(r[m] / best)
            wins[m] += r[m] <= best * (1 + EPS)
    return {m: mean(v) for m, v in ratios.items()}, wins


def _triplet(rows: List[dict], a: str, k: str, rl: str) -> List[dict]:
    return [{"A_old": r[a], "K_old": r[k], "RL main": r[rl]} for r in rows if rl in r]


def _paired_rows(res: Dict[str, Dict[str, dict]], only: Optional[set] = None) -> List[dict]:
    return [{"RL main": float(v["RL main"]["best_cost"]), "A_old": float(v["A_old"]["best_cost"]), "K_old": float(v["K_old"]["best_cost"])}
            for name, v in res.items() if (only is None or name in only) and all(k in v for k in ("RL main", "A_old", "K_old"))]


def exp8_rows(size: str) -> List[dict]:
    return _paired_rows(load_results(EXP8[size], {"rl_improvement_mvp": "RL main", "baseline": "A_old", "kisialiou": "K_old"}))


def exp10_rows(size: str, only: set) -> List[dict]:
    """The Exp 10 (first-fit references) on the same datasets as the ablation row, so the rows are like for like."""
    return _paired_rows(load_results(RL_JOBS["RL main"].format(size=size),
                                     {"rl_improvement_mvp": "RL main", "baseline": "A_old", "kisialiou": "K_old"}), only)


def status_lines(cache_dir: Path) -> List[str]:
    """Compact live table for the queue's status file (no RL restart files: those are large)."""
    lines = ["Each reference ran its own protocol (Kisialiou 5 restarts; adaptive ALNS for the same CPU time). old/ff = geometric mean "
             "over finished datasets of (old-pool cost) / (first-fit cost) for the same method; above 1.0000 = first-fit helped. "
             "CPU old/ff = old-pool budget relative to the first-fit budget. Ratio-to-best columns: denominator = best of adaptive / "
             "Kisialiou (both old pool) / RL main (given the first-fit budget) on the dataset, 1.0000 = best.", "",
             "| size | instances | Kisialiou old/ff | adaptive old/ff | CPU old/ff | adaptive (old) | Kisialiou (old) | RL main | "
             "RL cheaper than Kisialiou (old) |", "|---|---|---|---|---|---|---|---|---|"]
    for size in SIZES:
        data = collect(cache_dir, size)
        if not data:
            lines.append(f"| {size} | 0/50 | | | | | | | |")
            continue
        rows = list(data.values())
        head = (f"| {size} | {len(rows)}/50 | {gmean([r['K_old'] / r['K_ff'] for r in rows]):.4f} | "
                f"{gmean([r['A_old'] / r['A_ff'] for r in rows]):.4f} | {mean([r['T_old'] / r['T_ff'] for r in rows]):.3f} |")
        trip = _triplet(rows, "A_old", "K_old", "RL main")
        if trip:
            ratio, _ = _ratio_to_best(trip, ["A_old", "K_old", "RL main"])
            cheaper = sum(r["RL main"] < r["K_old"] * (1 - EPS) for r in trip)
            lines.append(f"{head} {ratio['A_old']:.4f} | {ratio['K_old']:.4f} | {ratio['RL main']:.4f} | {cheaper}/{len(trip)} |")
        else:
            lines.append(f"{head} | | | | |")
    return lines


def report(cache_dir: Path, sizes: Tuple[str, ...]) -> Tuple[str, List[dict]]:
    out: List[str] = []
    flat: List[dict] = []
    data = {s: collect(cache_dir, s, with_rl_restarts=True) for s in sizes}
    for s in sizes:
        for ds, r in data[s].items():
            flat.append({"size": s, "dataset": ds, **r})

    out += ["# Old-pool ablation (no first-fit in the references) -- per-dataset results", "",
            "One dataset = one consecutive search; the value compared is the best cost of the whole search. 'old' = original repair pool "
            "(3 destroy x 3 repair), 'ff' = pool with first-fit insertion (Experiment 10's references). Both reference protocols are the "
            "original one: Kisialiou runs 5 restarts of 300 iterations, the adaptive ALNS (no improvement operators) runs for Kisialiou's "
            "search CPU time. The RL models were given the first-fit Kisialiou's time.", ""]

    out += ["## 1. What first-fit did to the references", "",
            "old/ff = geometric mean over datasets of (final cost with the old pool) / (final cost with first-fit): above 1.0000 means "
            "first-fit lowered the cost. 'ff cheaper' = datasets where the first-fit run is strictly cheaper. Fleet = mean number of vessels "
            "in the final solution. CPU old/ff = mean ratio of the two runs' search CPU seconds (the old pool defines its own budget).", "",
            "| size | n | Kisialiou old/ff | ff cheaper | adaptive old/ff | ff cheaper | fleet K old / ff | fleet A old / ff | CPU old/ff |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in sizes:
        rows = list(data[s].values())
        if not rows:
            continue
        ks = [r["K_old"] / r["K_ff"] for r in rows]
        as_ = [r["A_old"] / r["A_ff"] for r in rows]
        out.append(f"| {s} | {len(rows)} | {gmean(ks):.4f} | {sum(x > 1 + EPS for x in ks)}/{len(rows)} | {gmean(as_):.4f} | "
                   f"{sum(x > 1 + EPS for x in as_)}/{len(rows)} | {mean([r['VK_old'] for r in rows]):.2f} / {mean([r['VK_ff'] for r in rows]):.2f} | "
                   f"{mean([r['VA_old'] for r in rows]):.2f} / {mean([r['VA_ff'] for r in rows]):.2f} | {mean([r['T_old'] / r['T_ff'] for r in rows]):.3f} |")
    out += ["", "### 1b. The same at equal compute", "",
            "Both runs cut to the smaller of the two budgets (from their restart rows; complete restarts only, so the longer run loses its "
            "cut restart -- Kisialiou restarts are long, so this handicaps the longer Kisialiou run noticeably).", "",
            "| size | n | Kisialiou old/ff | adaptive old/ff |", "|---|---|---|---|"]
    for s in sizes:
        rows = list(data[s].values())
        if rows:
            out.append(f"| {s} | {len(rows)} | {gmean([r['K_old@B'] / r['K_ff@B'] for r in rows]):.4f} | "
                       f"{gmean([r['A_old@B'] / r['A_ff@B'] for r in rows]):.4f} |")

    out += ["", "## 2. Does RL win again when the references lack first-fit? (ratio-to-best, 1.0000 = best)", "",
            "Denominator in every row: best of {adaptive, Kisialiou, RL main} on the dataset, within the row's own experiment. "
            "Exp 8 = original instances, pool and wall-clock budget; Exp 10 = first-fit references on the new instances; "
            "Ablation = old-pool references on the new instances (RL main is the same run as in Experiment 10, given the first-fit budget); "
            "Ablation at equal compute = every method cut to the smaller of the old-pool and first-fit budgets (RL from its restart rows).", "",
            "| size | experiment | n | adaptive | Kisialiou | RL main | best on dataset: A / K / RL |", "|---|---|---|---|---|---|---|"]

    def add_row(size: str, label: str, rows: List[dict]) -> None:
        if rows:
            ratio, wins = _ratio_to_best(rows, ["A_old", "K_old", "RL main"])
            out.append(f"| {size} | {label} | {len(rows)} | {ratio['A_old']:.4f} | {ratio['K_old']:.4f} | {ratio['RL main']:.4f} | "
                       f"{wins['A_old']} / {wins['K_old']} / {wins['RL main']} |")

    for s in sizes:
        add_row(s, "Exp 8 (old instances, old pool)", exp8_rows(s))
        add_row(s, "Exp 10 (new instances, first-fit pool)", exp10_rows(s, set(data[s])))
        add_row(s, "**Ablation (new instances, old pool)**", _triplet(list(data[s].values()), "A_old", "K_old", "RL main"))
        eq = [{"A_old": r["A_old@B"], "K_old": r["K_old@B"], "RL main": r["RL main@B"]} for r in data[s].values() if "RL main@B" in r]
        add_row(s, "Ablation at equal compute", eq)

    out += ["", "## 3. Paired comparison with Kisialiou, per dataset", "",
            "RL/Kis = geometric mean over datasets of RL's final cost / Kisialiou's final cost (below 1.0000 = RL cheaper); "
            "'RL cheaper' = datasets where RL is strictly cheaper. Fleet = RL's final vessel count compared with Kisialiou's: "
            "fewer / more datasets. Old pool: RL as evaluated vs old-pool Kisialiou; first-fit: Experiment 10's own comparison.", "",
            "| size | n | RL/Kis (old pool) | RL cheaper | fleet RL fewer / more | RL/Kis (ff pool) | RL cheaper | fleet RL fewer / more |",
            "|---|---|---|---|---|---|---|---|"]
    for s in sizes:
        rows = [r for r in data[s].values() if "RL main" in r]
        if not rows:
            continue
        o = [r["RL main"] / r["K_old"] for r in rows]
        f = [r["RL main"] / r["K_ff"] for r in rows]
        out.append(f"| {s} | {len(rows)} | {gmean(o):.4f} | {sum(x < 1 - EPS for x in o)}/{len(rows)} | "
                   f"{sum(r['V RL main'] < r['VK_old'] for r in rows)} / {sum(r['V RL main'] > r['VK_old'] for r in rows)} | "
                   f"{gmean(f):.4f} | {sum(x < 1 - EPS for x in f)}/{len(rows)} | "
                   f"{sum(r['V RL main'] < r['VK_ff'] for r in rows)} / {sum(r['V RL main'] > r['VK_ff'] for r in rows)} |")

    out += ["", "## 4. Every method, one denominator", "",
            "Denominator: best of {RL main, RL ASt, adaptive old, Kisialiou old, adaptive ff, Kisialiou ff} on the dataset "
            "(only datasets where all six exist; each at the budget of its own protocol). Mean ratio-to-best, 1.0000 = best; in brackets the "
            "number of datasets on which the method has the lowest cost.", "",
            "| size | n | RL main | RL ASt | adaptive old | Kisialiou old | adaptive ff | Kisialiou ff |", "|---|---|---|---|---|---|---|---|"]
    methods = ["RL main", "RL ASt", "A_old", "K_old", "A_ff", "K_ff"]
    for s in sizes:
        rows = [r for r in data[s].values() if all(m in r for m in methods)]
        if rows:
            ratio, wins = _ratio_to_best(rows, methods)
            out.append(f"| {s} | {len(rows)} | " + " | ".join(f"{ratio[m]:.4f} ({wins[m]})" for m in methods) + " |")

    design = load_design()
    levels = {key: sorted({d[key] for d in design.values()}) for key in ("tw_level", "df_level")}
    out += ["", "## 5. By instance type (RL main / Kisialiou final cost, geometric mean; below 1.0000 = RL cheaper)", "",
            "TW = share of installations with a time window; DF = demand/frequency level (DF1 many small calls ... DF5 few large calls). "
            "Cells: old pool / first-fit pool, n in brackets.", ""]
    for key, title in (("tw_level", "time-window density"), ("df_level", "demand/frequency level")):
        out += [f"By {title}:", "", "| size | " + " | ".join(levels[key]) + " |", "|---|" + "---|" * len(levels[key])]
        for s in sizes:
            cells = []
            for g in levels[key]:
                ds = [r for name, r in data[s].items() if "RL main" in r and design.get(name, {}).get(key) == g]
                cells.append(f"{gmean([r['RL main'] / r['K_old'] for r in ds]):.3f} / {gmean([r['RL main'] / r['K_ff'] for r in ds]):.3f} ({len(ds)})"
                             if ds else "")
            out.append(f"| {s} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n", flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=OLD_CACHE_DIR, help="old-pool reference cache folder")
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--out", type=Path, default=REPO / "output" / "rerun26oldpool__ANALYSIS.md")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else REPO / args.cache_dir
    text, flat = report(cache_dir, tuple(s for s in args.sizes.split(",") if s))
    print(text)
    if not args.no_write and flat:
        args.out.write_text(text)
        csv_path = args.out.with_suffix(".per_dataset.csv")
        fields = list(dict.fromkeys(k for row in flat for k in row))
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
        print(f"written: {args.out}\n         {csv_path}")


if __name__ == "__main__":
    main()
