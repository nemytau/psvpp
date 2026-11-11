#!/usr/bin/env python3
"""Experiment 10 (output prefix rerun26) -- performance analysis, per dataset and overall.

Unit of analysis = one dataset: the best solution of the whole consecutive search on that instance (restarts are part of the
search, never a unit of comparison). One denominator everywhere: ratio-to-best = a method's final cost / the cheapest final cost
of ANY method run on that dataset in Experiment 10 (small: 6 methods, medium and large: 5), 1.0000 = best.

Methods (all given Kisialiou's search CPU time on the dataset; RL charged its decision cost):
  Adaptive, Kisialiou   references with first-fit in the repair pool (Kisialiou: 5 restarts of 300 iterations)
  RL main               PPO policy, reward delta_cost_v9 (time-penalised cost delta)
  RL ASt                PPO policy, reward alns_score_time_v1
  Fixed FCR / reloc     no learning: random removal + first-fit + FCR (all sizes) / + relocation (small only) at every step

    python scripts/analyze_exp10.py            # writes docs/EXP10_RESULTS.md and docs/exp10_per_dataset.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "output"
DESIGN = REPO / "py_alns" / "resource" / "test_set_v2_design.csv"
SIZES = ("small", "medium", "large")
METHODS = ("Adaptive", "Kisialiou", "RL main", "RL ASt", "Fixed FCR", "Fixed reloc")
JOB = {
    "Adaptive": ("rerun26__main__delta_cost_v9__{s}__s42", "baseline"),
    "Kisialiou": ("rerun26__main__delta_cost_v9__{s}__s42", "kisialiou"),
    "RL main": ("rerun26__main__delta_cost_v9__{s}__s42", "rl_improvement_mvp"),
    "RL ASt": ("rerun26__alt__alns_score_time_v1__{s}__s42", "rl_improvement_mvp"),
    "Fixed FCR": ("rerun26__control__fixed_rnd_ff_fcr__{s}__s42", "rl_improvement_mvp"),
    "Fixed reloc": ("rerun26__control__fixed_rnd_ff_reloc__{s}__s42", "rl_improvement_mvp"),
}
EPS = 1e-9
D_NAMES, R_NAMES, I_NAMES = ("shaw", "random", "worst"), ("greedy", "regret2", "regret3", "firstfit"), ("VNR", "FCR", "reloc", "swap")


def action_name(a) -> str:
    a = int(a)
    return f"{D_NAMES[a // 16]}+{R_NAMES[(a // 4) % 4]}+{I_NAMES[a % 4]}"


# ----------------------------------------------------------------------------- loading
def load_method(size: str, method: str) -> Tuple[Dict[str, dict], Dict[str, list], Dict[str, collections.Counter], Dict[str, int]]:
    """results per dataset, restart rows per dataset, summed action histograms, single-action restart counts."""
    folder, mode = JOB[method]
    base = OUTPUT / folder.format(s=size)
    if not (base / "results.csv").exists():
        return {}, {}, {}, {}
    results = {}
    with (base / "results.csv").open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["mode"] == mode:
                results[r["dataset"]] = r
    restarts: Dict[str, list] = collections.defaultdict(list)
    hist: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    single: Dict[str, int] = collections.defaultdict(int)
    with (base / "restart_results.csv").open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["mode"] != mode:
                continue
            restarts[r["dataset"]].append((int(r["restart_id"]), float(r["search_cpu_seconds"]), float(r["best_cost"]),
                                           float(r["time_to_best_search_seconds"] or 0.0), int(float(r["best_num_vessels_used"]))))
            if r.get("action_hist"):
                for k, v in json.loads(r["action_hist"]).items():
                    hist[r["dataset"]][k] += v
                single[r["dataset"]] += r.get("distinct_actions") == "1"
    for rows in restarts.values():
        rows.sort()
    return results, restarts, hist, single


def load_design() -> Dict[str, dict]:
    with DESIGN.open(newline="") as handle:
        return {r["name"]: r for r in csv.DictReader(handle)}


def times(restarts: list, final_cost: float, target_fleet: Optional[int]) -> Tuple[float, float, Optional[float]]:
    """(time to best, time to within 1% of the final best, time to a solution with <= target_fleet vessels), search CPU seconds.

    Time to best is exact (CPU of the restarts before the best restart + time to best inside it). The other two are upper bounds at
    restart resolution: the moment the first restart that reached the level found its best solution."""
    cum, ttb, t1, tmin = 0.0, None, None, None
    global_best = min(r[2] for r in restarts)
    for _, cpu, cost, t_in, vessels in restarts:
        if ttb is None and cost <= global_best + 1e-9:  # the result files' definition: first restart within 1e-9 of the best
            ttb = cum + t_in
        if t1 is None and cost <= final_cost * 1.01:
            t1 = cum + t_in
        if target_fleet is not None and tmin is None and vessels <= target_fleet:
            tmin = cum + t_in
        cum += cpu
    return float(ttb), float(t1), tmin


def collect(size: str) -> Tuple[Dict[str, Dict[str, dict]], dict]:
    """{dataset: {method: record}} plus per-method action statistics."""
    loaded = {m: load_method(size, m) for m in METHODS}
    methods = [m for m in METHODS if loaded[m][0]]
    datasets = sorted(set.intersection(*[set(loaded[m][0]) for m in methods]))
    rows: Dict[str, Dict[str, dict]] = {}
    for d in datasets:
        recs = {}
        for m in methods:
            res = loaded[m][0][d]
            recs[m] = dict(cost=float(res["best_cost"]), init_cost=float(res["initial_cost"]), fleet=int(float(res["best_num_vessels_used"])),
                           init_fleet=int(float(res["initial_num_vessels_used"])), cpu=float(res["search_cpu_seconds"]),
                           budget=float(res["runtime_target_seconds"]), ttb_file=float(res["time_to_best_search_seconds"]),
                           restarts=len(loaded[m][1][d]))
        best = min(r["cost"] for r in recs.values())
        fstar = min(r["fleet"] for r in recs.values())
        for m, r in recs.items():
            r["ratio"] = r["cost"] / best
            r["fstar"] = fstar
            r["ttb"], r["t1"], r["tmin"] = times(loaded[m][1][d], r["cost"], fstar)
        rows[d] = recs
    policy = {}
    for m in ("RL main", "RL ASt"):
        if m in methods:
            tot = collections.Counter()
            for d in datasets:
                tot.update(loaded[m][2][d])
            n_restarts = sum(len(loaded[m][1][d]) for d in datasets)
            policy[m] = dict(top=[(action_name(k), v / sum(tot.values())) for k, v in tot.most_common(3)],
                             single=sum(loaded[m][3][d] for d in datasets) / max(n_restarts, 1))
    return rows, dict(methods=methods, policy=policy)


# ----------------------------------------------------------------------------- statistics
def gmean(xs: List[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(ratios: List[float]) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    lg = np.log(ratios)
    means = [rng.choice(lg, len(lg)).mean() for _ in range(4000)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return math.exp(lo), math.exp(hi)


def paired(rows: Dict[str, Dict[str, dict]], a: str, b: str) -> Optional[dict]:
    xs = [r[a]["cost"] / r[b]["cost"] for r in rows.values() if a in r and b in r]
    if not xs:
        return None
    w = sum(x < 1 - EPS for x in xs)
    l = sum(x > 1 + EPS for x in xs)
    lo, hi = boot_ci(xs)
    return dict(n=len(xs), g=gmean(xs), lo=lo, hi=hi, w=w, t=len(xs) - w - l, l=l, p=sign_p(w, l), med=statistics.median(xs))


def table(header: List[str], body: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body]


def f4(x: float) -> str:
    return f"{x:.4f}"


def pct(n: int, d: int) -> str:
    return f"{n}/{d}" if d else "-"


def prefix_best(restarts: list, budget: float) -> float:
    """Best restart cost among the restarts that finished within `budget` CPU seconds (at least the first one)."""
    cum, best = 0.0, None
    for _, cpu, cost, _t, _v in restarts:
        cum += cpu
        if best is not None and cum > budget * (1 + 1e-6):
            break
        best = cost if best is None else min(best, cost)
    return float(best)


PORTFOLIOS = (
    ("Kisialiou, full budget (= Exp 10)", {"Kisialiou": 1.0}),
    ("Kisialiou, half the budget", {"Kisialiou": 0.5}),
    ("Kisialiou 50% + Fixed FCR 50%", {"Kisialiou": 0.5, "Fixed FCR": 0.5}),
    ("Kisialiou 60% + Fixed FCR 40%", {"Kisialiou": 0.6, "Fixed FCR": 0.4}),
    ("Kisialiou 50% + RL ASt 50%", {"Kisialiou": 0.5, "RL ASt": 0.5}),
    ("Kisialiou 50% + RL main 50%", {"Kisialiou": 0.5, "RL main": 0.5}),
    ("Kisialiou 50% + Adaptive 50%", {"Kisialiou": 0.5, "Adaptive": 0.5}),
    ("Kisialiou 40% + Fixed FCR 30% + RL ASt 30%", {"Kisialiou": 0.4, "Fixed FCR": 0.3, "RL ASt": 0.3}),
    ("Fixed FCR 50% + RL ASt 50% (no Kisialiou)", {"Fixed FCR": 0.5, "RL ASt": 0.5}),
)


# ----------------------------------------------------------------------------- report
def build() -> Tuple[str, List[dict]]:
    design = load_design()
    data, meta = {}, {}
    for s in SIZES:
        data[s], meta[s] = collect(s)
    out: List[str] = []
    flat: List[dict] = []
    common = ("Adaptive", "Kisialiou", "RL main", "RL ASt", "Fixed FCR")  # present on every size

    out += ["# Experiment 10 — performance analysis, per dataset and overall", "",
            "Generated by `scripts/analyze_exp10.py` from `output/rerun26__*` (merged jobs). Experiment 10 = the Experiment 8 setup rerun with "
            "first-fit insertion added to both baselines' repair pool, search-CPU-time budgets, the NOK cost model and 50 diversified test "
            "instances per size (seed 42, one trained model per reward). It ran 300-iteration restarts against the engine's old horizon of 1000 "
            "(fixed 2026-11-03; see `docs/EXPERIMENTS.md`).", "",
            "**Conventions.** Unit = dataset (the best solution of the whole search). *Ratio-to-best* = a method's final cost / the cheapest final "
            "cost of any method run on that dataset in Experiment 10 (small: Adaptive, Kisialiou, RL main, RL ASt, Fixed FCR, Fixed reloc; "
            "medium and large: the first five), 1.0000 = best; averages are over the 50 datasets of a size. *Best-known fleet* f* = the fewest "
            "vessels in the final solution of any method on that dataset (not a proven bound). Every method got the same search CPU budget "
            "per dataset (Kisialiou's 5 restarts).", ""]

    # ------------------------------------------------------------------ summary
    summ = []
    for m in common:
        recs = [r[m] for s in SIZES for r in data[s].values()]
        strict = sum(1 for s in SIZES for r in data[s].values() if r[m]["cost"] < min(v["cost"] for k, v in r.items() if k != m) - 1e-6)
        hit = sum(1 for r in recs if r["fleet"] == r["fstar"])
        excess = sum(r["ratio"] - 1 for r in recs)
        miss_excess = sum(r["ratio"] - 1 for r in recs if r["fleet"] > r["fstar"])
        at = [r["ratio"] for r in recs if r["fleet"] == r["fstar"]]
        summ.append([m, f4(statistics.mean(r["ratio"] for r in recs)), str(strict), f"{hit}/150", f4(statistics.mean(at)), f"{miss_excess / excess * 100:.0f}%"])
    out += ["## Summary (all 150 datasets)", ""]
    out += table(["method", "mean ratio-to-best", "strictly cheapest on", "reached best-known fleet", "mean ratio when it did", "share of excess cost from fleet misses"], summ)
    out += ["", "Details by size, per dataset and by instance type follow.", ""]

    # ------------------------------------------------------------------ 1. cost
    out += ["## 1. Cost: ratio-to-best per dataset, summarised", ""]
    body = []
    for s in SIZES:
        rows = data[s]
        for m in meta[s]["methods"]:
            xs = [r[m]["ratio"] for r in rows.values()]
            body.append([s, m, str(len(xs)), f4(statistics.mean(xs)), f4(statistics.median(xs)), f4(max(xs)),
                         pct(sum(x <= 1 + EPS for x in xs), len(xs)), pct(sum(x <= 1.01 for x in xs), len(xs)),
                         pct(sum(x > 1.05 for x in xs), len(xs))])
    for m in common:
        xs = [r[m]["ratio"] for s in SIZES for r in data[s].values()]
        body.append(["all 150", m, str(len(xs)), f4(statistics.mean(xs)), f4(statistics.median(xs)), f4(max(xs)),
                     pct(sum(x <= 1 + EPS for x in xs), len(xs)), pct(sum(x <= 1.01 for x in xs), len(xs)), pct(sum(x > 1.05 for x in xs), len(xs))])
    out += table(["size", "method", "n", "mean", "median", "worst", "best on (ties count)", "within 1%", "more than 5% worse"], body)
    out += ["", "How often each method is *strictly* cheapest on a dataset (no ties): "
            + "; ".join(f"{s}: " + ", ".join(f"{m} {sum(1 for r in data[s].values() if r[m]['cost'] < min(v['cost'] for k, v in r.items() if k != m) - 1e-6)}"
                                              for m in meta[s]["methods"]) for s in SIZES) + ".", ""]

    # ------------------------------------------------------------------ 2. paired
    out += ["## 2. Paired comparisons on the same datasets", "",
            "Geometric mean of (method cost / reference cost) over the datasets, with a 95% bootstrap interval; below 1.0000 the method is cheaper. "
            "cheaper / tie / dearer count datasets; p = exact two-sided sign test on cheaper vs dearer (ties excluded).", "", "### 2a. Against Kisialiou", ""]
    body = []
    for s in SIZES:
        for m in meta[s]["methods"]:
            if m == "Kisialiou":
                continue
            c = paired(data[s], m, "Kisialiou")
            body.append([s, m, f4(c["g"]), f"{f4(c['lo'])} – {f4(c['hi'])}", f4(c["med"]), f"{c['w']} / {c['t']} / {c['l']}", f"{c['p']:.3f}"])
    out += table(["size", "method", "geo-mean ratio", "95% interval", "median ratio", "cheaper / tie / dearer", "p"], body)
    out += ["", "Many small losses and a few large wins show up as a median above 1 with a geometric mean near or below 1 (the wins are datasets "
            "where the method saves a vessel that Kisialiou does not).", ""]
    out += ["", "### 2b. Learned policy against the fixed action it resembles (no learning)", ""]
    body = []
    for s in SIZES:
        for a, b in (("RL main", "Fixed FCR"), ("RL ASt", "Fixed FCR"), ("RL main", "RL ASt"), ("RL main", "Fixed reloc"), ("RL ASt", "Fixed reloc")):
            if a in meta[s]["methods"] and b in meta[s]["methods"]:
                c = paired(data[s], a, b)
                body.append([s, f"{a} / {b}", f4(c["g"]), f"{f4(c['lo'])} – {f4(c['hi'])}", f"{c['w']} / {c['t']} / {c['l']}", f"{c['p']:.3f}", f4(c["med"])])
    out += table(["size", "pair", "geo-mean ratio", "95% interval", "cheaper / tie / dearer", "p", "median ratio"], body)
    out += ["", "A geometric mean above 1 together with a median below 1 means: cheaper on most datasets, but a few large losses (a vessel more).", ""]

    # ------------------------------------------------------------------ 3. fleet
    out += ["## 3. Fleet: was the minimum number of vessels reached?", "",
            "Vessels in the final (cheapest) solution. *At f\\** = share of datasets where the method's fleet equals the best-known fleet f\\* of that "
            "dataset; *+1* and *≥ +2* = one / two or more vessels more than f\\*. The last two columns split the method's mean ratio-to-best into the datasets "
            "where it reached f\\* and where it did not (charter is 81–87% of the cost, so a missed vessel is the main source of loss).", ""]
    body = []
    for s in SIZES:
        rows = data[s]
        for m in meta[s]["methods"]:
            recs = [r[m] for r in rows.values()]
            hit = [r for r in recs if r["fleet"] == r["fstar"]]
            plus1 = [r for r in recs if r["fleet"] == r["fstar"] + 1]
            more = [r for r in recs if r["fleet"] >= r["fstar"] + 2]
            miss = [r for r in recs if r["fleet"] > r["fstar"]]
            excess = sum(r["ratio"] - 1 for r in recs)
            share = sum(r["ratio"] - 1 for r in miss) / excess if excess > 1e-12 else float("nan")
            body.append([s, m, f"{statistics.mean(r['init_fleet'] for r in recs):.2f}", f"{statistics.mean(r['fleet'] for r in recs):.2f}",
                         f"{statistics.mean(r['init_fleet'] - r['fleet'] for r in recs):.2f}", pct(len(hit), len(recs)), str(len(plus1)), str(len(more)),
                         f4(statistics.mean(r["ratio"] for r in hit)) if hit else "-", f4(statistics.mean(r["ratio"] for r in miss)) if miss else "-",
                         f"{share * 100:.0f}%"])
    out += table(["size", "method", "initial fleet", "final fleet", "vessels removed", "at f*", "+1", "≥ +2", "mean ratio when at f*", "mean ratio when above f*",
                  "share of the method's excess cost from fleet misses"], body)
    cheapest_ok = {s: sum(1 for r in data[s].values() if any(v["ratio"] <= 1 + EPS and v["fleet"] == v["fstar"] for v in r.values())) for s in SIZES}
    out += ["", "On how many datasets does a cheapest solution also have the best-known fleet? " + ", ".join(f"{s} {cheapest_ok[s]}/{len(data[s])}" for s in SIZES) + ".", ""]

    out += ["", "### 3b. Which datasets are hard for the fleet?", "",
            "For each dataset, how many of the methods reached the best-known fleet f\\*. A dataset where only one or two methods reach f\\* is a hard cut.", ""]
    body = []
    hard = []
    for s in SIZES:
        n_m = len(meta[s]["methods"])
        counts = collections.Counter(sum(1 for v in r.values() if v["fleet"] == v["fstar"]) for r in data[s].values())
        body.append([s, str(n_m)] + [str(counts.get(k, 0)) for k in range(n_m, 0, -1)] + ["-"] * (6 - n_m))
        for d, r in data[s].items():
            at = [m for m, v in r.items() if v["fleet"] == v["fstar"]]
            if len(at) <= n_m // 2:
                hard.append((s, d, at, [m for m in r if m not in at]))
    out += table(["size", "methods", "all reach f*", "one fewer", "two fewer", "three fewer", "four fewer", "five fewer"], body)
    out += ["", "Datasets where at most half of the methods reach f* (hard cuts):", ""]
    out += table(["size", "dataset", "type", "reach f*", "above f*"],
                 [[s, d.replace(f"{s}_test_", ""), f"TW {design[d]['tw_level']}, DF {design[d]['df_level']}, n {design[d]['n_installations']}", ", ".join(at), ", ".join(ab)]
                  for s, d, at, ab in hard]) if hard else ["(none)"]
    out += [""]

    # ------------------------------------------------------------------ 4. time to best
    out += ["## 4. Time to best (search CPU seconds)", "",
            "Per dataset: CPU of the restarts before the best restart plus the time to the best inside it (exact; equals the result files' "
            "`time_to_best_search_seconds`). *Budget* = the CPU time every method was given. *Late* = best found in the last 20% of the budget "
            "(the search was still improving when time ran out). *Within 1%* and *to f\\** are upper bounds at restart resolution: the moment the first "
            "restart that reached the level (cost within 1% of the method's own final best / a fleet of f\\* vessels) found its best solution; for "
            "*to f\\** only datasets where the method reached f\\* count (n).", ""]
    body = []
    for s in SIZES:
        rows = data[s]
        for m in meta[s]["methods"]:
            recs = [r[m] for r in rows.values()]
            frac = [r["ttb"] / r["budget"] for r in recs]
            t1 = [r["t1"] / r["budget"] for r in recs]
            tm = [r["tmin"] / r["budget"] for r in recs if r["tmin"] is not None]
            body.append([s, m, f"{statistics.mean(r['budget'] for r in recs):.0f}", f"{statistics.median(r['ttb'] for r in recs):.1f}",
                         f"{statistics.median(frac):.2f}", f"{statistics.mean(frac):.2f}", pct(sum(f > 0.8 for f in frac), len(frac)),
                         f"{statistics.median(t1):.2f}", f"{statistics.median(tm):.2f} (n={len(tm)})" if tm else "-"])
    out += table(["size", "method", "mean budget (s)", "median time to best (s)", "median share of budget", "mean share", "late", "median to within 1% (share)",
                  "median to f* (share, n)"], body)
    out += [""]

    # ------------------------------------------------------------------ 5. effort and policies
    out += ["## 5. Search effort and what the learned policies play", "",
            "Restarts are search effort, not a performance unit. Policies: share of all steps by action (random = random-visit removal, firstfit = "
            "first-fit insertion), and the share of restarts in which the policy played a single action throughout.", ""]
    body = []
    for s in SIZES:
        for m in meta[s]["methods"]:
            if m in ("Kisialiou",):
                continue
            recs = [r[m] for r in data[s].values()]
            pol = meta[s]["policy"].get(m)
            body.append([s, m, f"{statistics.mean(r['restarts'] for r in recs):.0f}", f"{statistics.mean(r['cpu'] / r['budget'] for r in recs):.3f}",
                         ", ".join(f"{a} {v * 100:.0f}%" for a, v in pol["top"]) if pol else "fixed action", f"{pol['single'] * 100:.0f}%" if pol else "-"])
    out += table(["size", "method", "restarts per dataset (mean)", "search CPU used / budget", "top actions (share of steps)", "single-action restarts"], body)
    out += ["Kisialiou always runs 5 restarts.", ""]

    # ------------------------------------------------------------------ 6. by type
    def levels(key):
        return sorted({design[d][key] for s in SIZES for d in data[s]}, key=float)

    out += ["## 6. By instance type", "",
            "TW = share of installations with a tight time window; DF = demand/frequency level (DF1 many small calls … DF5 few large calls). "
            "Each cell is the mean ratio-to-best of the datasets in the cell (5 datasets per cell and size, 15 pooled; same denominator as above); "
            "in brackets the share of those datasets at the best-known fleet.", ""]
    for key, title in (("tw_level", "time-window density"), ("df_level", "demand/frequency level")):
        for scope in (*SIZES, "all 150"):
            ds = [(s, d) for s in SIZES for d in data[s] if scope in (s, "all 150")]
            ms = [m for m in common if all(m in data[s][d] for s, d in ds)]
            body = []
            for lv in levels(key):
                sel = [(s, d) for s, d in ds if design[d][key] == lv]
                body.append([lv, str(len(sel))] + [f"{f4(statistics.mean(data[s][d][m]['ratio'] for s, d in sel))} ({sum(data[s][d][m]['fleet'] == data[s][d][m]['fstar'] for s, d in sel)}/{len(sel)})" for m in ms])
            out += [f"**{title}, {scope}**", ""] + table([key.split('_')[0].upper(), "n"] + list(ms), body) + [""]

    # ------------------------------------------------------------------ 7. per dataset
    out += ["## 7. Per dataset", "",
            "One row per dataset. Cells: ratio-to-best and, in brackets, the final fleet; **bold** = cheapest (ties included); `*` = fleet above the "
            "best-known f\\*. T = budget in CPU seconds, n = installations, visits = visits per week.", ""]
    for s in SIZES:
        ms = meta[s]["methods"]
        out += [f"### {s}", ""]
        body = []
        for d in sorted(data[s], key=lambda d: (float(design[d]["tw_level"]), int(design[d]["df_level"]), d)):
            r = data[s][d]
            cells = []
            for m in ms:
                c = f"{f4(r[m]['ratio'])} ({r[m]['fleet']}{'*' if r[m]['fleet'] > r[m]['fstar'] else ''})"
                cells.append(f"**{c}**" if r[m]["ratio"] <= 1 + EPS else c)
            first = next(iter(r.values()))
            body.append([d.replace(f"{s}_test_", ""), design[d]["tw_level"], design[d]["df_level"], design[d]["n_installations"], design[d]["visits"],
                         f"{first['budget']:.0f}", f"{first['init_fleet']}→{first['fstar']}"] + cells)
        out += table(["dataset", "TW", "DF", "n", "visits", "T (s)", "fleet initial→f*"] + list(ms), body) + [""]
        out += [f"Time to best in CPU seconds ({s}); `—` is not applicable:", ""]
        body = []
        for d in sorted(data[s], key=lambda d: (float(design[d]["tw_level"]), int(design[d]["df_level"]), d)):
            r = data[s][d]
            body.append([d.replace(f"{s}_test_", "")] + [f"{r[m]['ttb']:.1f}" for m in ms])
        out += table(["dataset"] + list(ms), body) + [""]

        # largest differences between RL main and Kisialiou
        diffs = sorted(((r["RL main"]["cost"] / r["Kisialiou"]["cost"], d) for d, r in data[s].items()))
        out += [f"Datasets with the largest differences between RL main and Kisialiou ({s}), RL cost / Kisialiou cost (fleets RL vs Kisialiou):", ""]
        body = []
        for ratio, d in diffs[:4] + diffs[-6:]:
            r = data[s][d]
            body.append([d.replace(f"{s}_test_", ""), f4(ratio), f"{r['RL main']['fleet']} vs {r['Kisialiou']['fleet']}", f"TW {design[d]['tw_level']}, DF {design[d]['df_level']}, n {design[d]['n_installations']}"])
        out += table(["dataset", "RL / Kisialiou", "final fleets", "type"], body) + [""]

    # ------------------------------------------------------------------ 8. headroom
    arms = ("Kisialiou", "Fixed FCR", "RL ASt", "RL main", "Adaptive")
    restarts = {sz: {m: load_method(sz, m)[1] for m in arms} for sz in SIZES}
    out += ["## 8. Headroom: portfolios and configuration selection (offline reconstruction)", "",
            "Restarts are independent runs, so a run can be cut to a smaller budget (complete restarts only) and several configurations can share one "
            "budget. A *portfolio* gives each configuration a share of the dataset's budget (Kisialiou's search CPU time) and keeps the best solution "
            "found: the total compute is the same as in Experiment 10. Mean ratio-to-best with the same denominator as everywhere else (the cheapest "
            "full-budget result of any single method). **Reconstructed from recorded restarts on the same 150 datasets and with the splits tried here "
            "chosen after seeing the data; it has to be confirmed by a real run.**", ""]
    body = []
    for label, split in PORTFOLIOS:
        per = {}
        for sz in SIZES:
            xs = []
            for d, r in data[sz].items():
                budget = r["Kisialiou"]["budget"]
                best = min(v["cost"] for v in r.values())
                xs.append(min(prefix_best(restarts[sz][m][d], share * budget) for m, share in split.items()) / best)
            per[sz] = xs
        allx = [x for sz in SIZES for x in per[sz]]
        body.append([label] + [f4(statistics.mean(per[sz])) for sz in SIZES] + [f4(statistics.mean(allx))])
    out += table(["configuration (shares of the budget)", "small", "medium", "large", "all 150"], body)
    out += ["", "**Choosing the configuration from the instance type** (Kisialiou, Fixed FCR, RL ASt or Adaptive per time-window level / demand-frequency "
            "level / both, chosen on two sizes and applied to the third; leave-one-size-out):", ""]
    cands = ("Kisialiou", "Fixed FCR", "RL ASt", "Adaptive")
    allrows = [(sz, d, r) for sz in SIZES for d, r in data[sz].items()]

    def selection(keyf):
        picked = []
        for held in SIZES:
            by = collections.defaultdict(lambda: collections.defaultdict(list))
            for sz, d, r in allrows:
                if sz != held:
                    for m in cands:
                        by[keyf(d)][m].append(r[m]["ratio"])
            choice = {k: min(cands, key=lambda m: statistics.mean(v[m])) for k, v in by.items()}
            picked += [r[choice.get(keyf(d), "Kisialiou")]["ratio"] for sz, d, r in allrows if sz == held]
        return statistics.mean(picked)

    body = [["Kisialiou on every dataset", f4(statistics.mean(r["Kisialiou"]["ratio"] for _, _, r in allrows))],
            ["chosen by time-window level", f4(selection(lambda d: design[d]["tw_level"]))],
            ["chosen by demand/frequency level", f4(selection(lambda d: design[d]["df_level"]))],
            ["chosen by (time-window, demand/frequency) cell", f4(selection(lambda d: (design[d]["tw_level"], design[d]["df_level"])))],
            ["oracle: best of the four on every dataset (needs a perfect predictor or four runs)", f4(statistics.mean(min(r[m]["ratio"] for m in cands) for _, _, r in allrows))]]
    out += table(["rule", "mean ratio-to-best, all 150"], body)
    out += ["", "A selection that is learned from the instance type does not carry over between sizes here; the portfolio gain comes from running "
            "different configurations, not from predicting which one is better.", ""]

    for s in SIZES:
        for d, r in data[s].items():
            row = {"size": s, "dataset": d, "tw_level": design[d]["tw_level"], "df_level": design[d]["df_level"],
                   "installations": design[d]["n_installations"], "visits": design[d]["visits"]}
            first = next(iter(r.values()))
            row.update(budget_s=first["budget"], initial_cost=first["init_cost"], initial_fleet=first["init_fleet"], best_known_fleet=first["fstar"])
            for m in METHODS:
                if m in r:
                    x = r[m]
                    row.update({f"{m} cost": x["cost"], f"{m} ratio": x["ratio"], f"{m} fleet": x["fleet"], f"{m} time_to_best_s": x["ttb"],
                                f"{m} time_to_within_1pct_s": x["t1"], f"{m} time_to_fstar_s": x["tmin"], f"{m} restarts": x["restarts"]})
            flat.append(row)
    return "\n".join(out) + "\n", flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "EXP10_RESULTS.md")
    parser.add_argument("--csv", type=Path, default=REPO / "docs" / "exp10_per_dataset.csv")
    args = parser.parse_args()
    text, flat = build()
    args.out.write_text(text)
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    print(text)
    print(f"written: {args.out}\n         {args.csv}")


if __name__ == "__main__":
    main()
