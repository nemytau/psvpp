"""Experiment 8 rerun (2026-10): RL vs. adaptive baseline vs. Kisialiou, small / medium / large.

What changed relative to Experiment 8 (docs/EXPERIMENTS.md, "Experiment 8 rerun"):
  1. time budgets are search CPU time (Python overhead excluded) and RL is charged its native
     per-decision cost (rl/episode_runner.py, scripts/run_experiment_rl_improvement_mvp.py);
  2. the adaptive baseline and Kisialiou's ALNS get first-fit insertion in their repair pool
     (--baseline-action-module adaptive_weighted_v2) and their reference is regenerated;
  3. cost model in NOK: charter 7 x 250,000 per vessel and week, fuel price per tonne
     (rust_alns/src/structs/constants.rs), SA start temperature scaled with it;
  4. diversified test sets, 50 instances per size, every instance its own configuration
     (scripts/generate_test_set_v2.py); one training seed and one evaluation seed (42).

Work graph (every task is a separate process; tags in brackets):
    reference chunks   Kisialiou (5 reference restarts) + adaptive baseline on a cost-balanced
                       slice of the test set; merged per size into the cache the evaluations read
    train              PPO, op_triple_v2 / features_improvement_v3 / 200k steps / ent_coef 0.01 / ph300:
                       main = delta_cost_v9 (Exp 8 reward); alt = alns_score_time_v1 (Exp 9's best medium reward)
    eval shards        each trained model on cost-balanced slices of the 50 test instances, merged afterwards
    controls           fixed triple (random removal + first-fit + FCR / relocation) through the same loop

Time-fairness: the CPU time a fixed amount of work costs depends on machine load on the M1
(identical work: 2.37 s alone, ~3.9 s with 4 busy processes), so every reference/evaluation task
runs while exactly --workers CPU-bound processes are busy (filler processes cover gaps; they exit
by themselves if this runner dies). The machine should otherwise be idle.

Layout (everything under the repo's output/ and logs/, which live on the external SSD):
    data/rerun26/{generated,processed/alns}/...        datasets of this experiment (+ DATA_MANIFEST.json)
    runs/baseline_kisialiou_cache/rerun26_v1/          reference parts + merged caches
    output/rerun26__<tag>[__train|__shNN]/             per task: run_manifest.json, run_command.txt, results, model
    logs/rerun26__<tag>[...]/                          per task: operator-usage logs, training logs
    logs/rerun26_queue/                                events.log, queue.log, one log per task and phase
    output/rerun26__QUEUE_MANIFEST.json                the whole plan (commands, shards, predictions, outcomes)
    RERUN26_STATUS.md                                  refreshed every minute

Resumable at every level: finished reference chunks, trainings, evaluation shards and merges are
never redone; an interrupted evaluation shard / reference chunk resumes after its last finished
instance. A failed task is retried once.

    nohup caffeinate -is python scripts/run_exp8_rerun.py --workers 4 --alt-sizes small,medium,large \\
        > logs/rerun26_queue/queue.log 2>&1 &
    python scripts/run_exp8_rerun.py --list          # plan with predicted durations, no execution
    python scripts/run_exp8_rerun.py --status
    python scripts/run_exp8_rerun.py --smoke         # tiny end-to-end check (own prefix, 2 small instances)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import shutil
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
PY = "/Users/timothy/opt/miniconda3/envs/py312/bin/python"
SIZES = ("large", "medium", "small")  # scheduling order: the longest chain first
SEED = 42
TIMESTEPS = 200_000
BASELINE_MODULE = "adaptive_weighted_v2"
REF = dict(action="op_triple_v2", state="features_improvement_v3")
# op_triple_v2 index = d*16 + r*4 + i; d (shaw, random, worst), r (greedy, regret2, regret3,
# first-fit), i (VNR, FCR, relocation, swap): the triples Exp 8/9 policies collapsed to.
FIXED_TRIPLES = {"fcr": 29, "reloc": 30}
CONTROLS = {"small": ("fcr", "reloc"), "medium": ("fcr",), "large": ("fcr",)}
REFERENCE_CHUNKS = {"small": 2, "medium": 5, "large": 12}
EVAL_SHARDS = {"small": 1, "medium": 3, "large": 6}
# Per-iteration operator-usage rows of the reference and learned-policy runs: all restarts for every
# size (decision 2026-10-22: the SSD has the room; expected ~300 GB in total, ~200 GB of it large).
# None = log everything; N = only the first N restarts per episode.
OPERATOR_LOG_RESTARTS = {"small": None, "medium": None, "large": None}
# Fixed-configuration controls play one triple forever, so their per-iteration rows carry no
# information beyond the first restarts; the restart summaries are complete.
CONTROL_LOG_RESTARTS = {"small": None, "medium": 20, "large": 20}
N_TEST = 50

# ----------------------------------------------------------------------------- paths / config
PREFIX = "rerun26"
DATA_ROOT = REPO / "data" / "rerun26"
GENERATED = DATA_ROOT / "generated"
PROCESSED = DATA_ROOT / "processed" / "alns"
QUEUE_DIR = REPO / "logs" / "rerun26_queue"
STATUS_PATH = REPO / "RERUN26_STATUS.md"
EVENTS_PATH = QUEUE_DIR / "events.log"
CACHE_DIR = REPO / "runs" / "baseline_kisialiou_cache" / "rerun26_v1"
QUEUE_MANIFEST = REPO / "output" / "rerun26__QUEUE_MANIFEST.json"
SMOKE = False


def _configure_smoke() -> None:
    """Tiny end-to-end variant: own prefix/dirs, 2 small instances, 3k training steps, sharded."""
    global PREFIX, QUEUE_DIR, STATUS_PATH, EVENTS_PATH, CACHE_DIR, QUEUE_MANIFEST, SMOKE
    global SIZES, TIMESTEPS, REFERENCE_CHUNKS, EVAL_SHARDS, N_TEST, OPERATOR_LOG_RESTARTS, CONTROL_LOG_RESTARTS
    SMOKE = True
    PREFIX = "rerun26smoke"
    QUEUE_DIR = REPO / "logs" / "rerun26smoke_queue"
    STATUS_PATH = REPO / "RERUN26SMOKE_STATUS.md"
    EVENTS_PATH = QUEUE_DIR / "events.log"
    CACHE_DIR = REPO / "runs" / "baseline_kisialiou_cache" / "rerun26smoke"
    QUEUE_MANIFEST = REPO / "output" / "rerun26smoke__QUEUE_MANIFEST.json"
    SIZES = ("small",)
    TIMESTEPS = 3_000
    REFERENCE_CHUNKS = {"small": 2}
    EVAL_SHARDS = {"small": 2}
    OPERATOR_LOG_RESTARTS = {"small": 3}
    CONTROL_LOG_RESTARTS = {"small": 3}
    N_TEST = 2


def task_env(run_id: str) -> Dict[str, str]:
    return dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
        RL_RUN_ID=run_id,
        PSVPP_GENERATED_ROOT=str(GENERATED),
        PSVPP_PROCESSED_ROOT=str(PROCESSED),
    )


# ----------------------------------------------------------------------------- cost model
# Predicted durations (minutes) for ordering and ETA only; calibrated on Exp 8 caches (old test
# sets, alone) and Exp 9's queue (small/medium evaluations under the 4-process load).
LOAD = 1.37  # CPU time of fixed work with 4 busy processes vs alone
OLD_REF = {"small": (17.2, 11.1, 2.9), "medium": (40.2, 127.7, 3.3), "large": (82.3, 2020.5, 3.85)}
PY_RL, PY_CTRL = 1.03, 0.55  # ms of Python per decision under load (with / without policy inference)
E_REF = {"medium": (0.23, 40.2), "large": (0.59, 82.3)}  # FCR triple ms/step at reference visits
E_EXP = 1.3
REF_RATIO = {"small": 3.2, "medium": 2.3, "large": 2.1}
TRAIN_MIN = {("small", "main"): 21, ("small", "alt"): 33, ("medium", "main"): 58, ("medium", "alt"): 71,
             ("large", "main"): 480, ("large", "alt"): 576}


# Measured / predicted Kisialiou search CPU on 10 of the new instances under the 4-process load
# (2026-10-22, scripts/timing/calibrate_reference_cost.py): small 0.7-1.3 (mean 1.0), medium 1.4-1.6,
# large 0.95-1.54 (all-tight-window instances are the slow ones).
K_CORRECTION = {"small": 1.0, "medium": 1.3, "large": 1.3}


def predicted_kisialiou_seconds(size: str, visits: int) -> float:
    v_ref, k_ref, p = OLD_REF[size]
    return k_ref * (visits / v_ref) ** p * LOAD * K_CORRECTION[size]


def eval_ratio(size: str, visits: int, kind: str) -> float:
    if size == "small":
        # measured on Exp 9's small evaluations under the 4-process load (wall / search CPU)
        return {"main": 1.7, "alt": 1.6, "ctrl": 6.7, "ctrl_reloc": 1.4}[kind]
    e_ref, v_ref = E_REF[size]
    e = e_ref * (visits / v_ref) ** E_EXP
    if kind == "alt":
        e *= 2.2
    py = PY_CTRL if kind.startswith("ctrl") else PY_RL
    return 1 + py / e


# ----------------------------------------------------------------------------- data
@dataclass
class SizeData:
    size: str
    names: List[str]          # test instance directory names (sorted, = index order)
    visits: List[int]

    def predicted_k(self) -> List[float]:
        return [predicted_kisialiou_seconds(self.size, v) for v in self.visits]


def _generate_data() -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO), PSVPP_GENERATED_ROOT=str(GENERATED), PSVPP_PROCESSED_ROOT=str(PROCESSED))
    GENERATED.mkdir(parents=True, exist_ok=True)
    for cmd in (
        [PY, "scripts/generate_datasets.py", "--samples", "5", "--sizes", "small", "medium", "large",
         "--only-split", "train", "--output-root", str(GENERATED)],
        [PY, "scripts/generate_test_set_v2.py", "--output-root", str(GENERATED)],
    ):
        _event("DATA " + " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL)
    code = ("from rl.dataset_manager import GeneratedDatasetManager as M\n"
            "m = M()\n"
            "for s in ('small', 'medium', 'large'):\n"
            "    p = m.prepare_size(s)\n"
            "    print(s, len(p['train']), len(p['test']))\n")
    out = subprocess.run([PY, "-c", code], cwd=REPO, env=env, check=True, capture_output=True, text=True).stdout
    _event("DATA converted " + out.replace("\n", "; "))


def ensure_data() -> Dict[str, SizeData]:
    needed = [PROCESSED / s / "test" for s in ("small", "medium", "large")]
    if not all(p.is_dir() and len(list(p.iterdir())) >= 50 for p in needed) or not (GENERATED / "test_set_v2_design.csv").exists():
        _generate_data()
    data: Dict[str, SizeData] = {}
    for size in SIZES:
        names = sorted(d.name for d in (PROCESSED / size / "test").iterdir() if d.is_dir())[:N_TEST]
        visits = []
        for name in names:
            with open(PROCESSED / size / "test" / name / "installations.csv", newline="") as handle:
                visits.append(sum(int(r["visit_frequency"]) for r in csv.DictReader(handle)))
        data[size] = SizeData(size, names, visits)
    return data


def write_data_manifest(data: Dict[str, SizeData]) -> None:
    def digest(paths: List[Path]) -> str:
        h = hashlib.sha256()
        for p in paths:
            h.update(p.name.encode())
            h.update(p.read_bytes())
        return h.hexdigest()

    info = {"created_at": datetime.now().isoformat(), "generated_root": str(GENERATED),
            "processed_root": str(PROCESSED), "git": _git(), "sizes": {}}
    for size in ("small", "medium", "large"):
        entry = {}
        for split in ("train", "test"):
            dirs = sorted(d for d in (PROCESSED / size / split).iterdir() if d.is_dir())
            files = [f for d in dirs for f in sorted(d.glob("*.csv"))]
            entry[split] = {"instances": len(dirs), "sha256_of_all_csv": digest(files)}
        info["sizes"][size] = entry
    design = GENERATED / "test_set_v2_design.csv"
    info["test_set_design_csv"] = {"path": str(design), "sha256": hashlib.sha256(design.read_bytes()).hexdigest()}
    (DATA_ROOT / "DATA_MANIFEST.json").write_text(json.dumps(info, indent=2))


def balanced_groups(costs: List[float], n: int) -> List[List[int]]:
    """Split instance indices into n groups of near-equal predicted cost (longest first)."""
    n = max(1, min(n, len(costs)))
    bins = [[0.0, []] for _ in range(n)]
    for idx in sorted(range(len(costs)), key=lambda i: -costs[i]):
        b = min(bins, key=lambda x: x[0])
        b[0] += costs[idx]
        b[1].append(idx)
    return [sorted(b[1]) for b in bins]


# ----------------------------------------------------------------------------- disk guard
MIN_FREE_GB = 120.0  # do not start new tasks below this (running ones continue); the queue resumes by itself


def free_gb() -> float:
    """Free space of the volume holding output/ and logs/ (the external SSD); 0 if it is not reachable."""
    try:
        return shutil.disk_usage(REPO / "output").free / 1e9
    except OSError:
        return 0.0


# ----------------------------------------------------------------------------- tasks
@dataclass
class Task:
    kind: str                      # ref | train | eval | merge
    tag: str
    size: str
    minutes: float                 # predicted duration (ordering / ETA only)
    deps: List[str] = field(default_factory=list)
    optional: bool = False
    command: List[str] = field(default_factory=list)
    run_id: str = ""
    out_dir: Optional[Path] = None
    done_file: Optional[Path] = None
    meta: Dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.tag}"

    def done(self) -> bool:
        return bool(self.done_file and self.done_file.exists())


def _common(size: str, indices: Optional[List[int]] = None) -> List[str]:
    cmd = [PY, "scripts/run_experiment_rl_improvement_mvp.py", "--sizes", size,
           "--eval-seeds", str(SEED), "--max-iterations", "300", "--seed", str(SEED),
           "--baseline-action-module", BASELINE_MODULE, "--force-overwrite"]
    if indices is None:
        cmd += ["--datasets-per-size", str(N_TEST)]
    else:
        cmd += ["--dataset-indices", ",".join(str(i) for i in indices)]
    return cmd


def build_tasks(data: Dict[str, SizeData], alt_sizes: List[str], controls: bool) -> List[Task]:
    tasks: List[Task] = []
    for size in SIZES:
        sd = data[size]
        k = sd.predicted_k()
        cap = OPERATOR_LOG_RESTARTS[size]
        cap_args = ["--operator-log-restarts", str(cap)] if cap is not None else []
        merged_cache = CACHE_DIR / f"cache_{size}_seed{SEED}.json"

        # reference chunks (Kisialiou + adaptive baseline)
        ref_tags = []
        for ci, group in enumerate(balanced_groups([x * REF_RATIO[size] / 60 for x in k], REFERENCE_CHUNKS[size])):
            tag = f"reference__{size}__c{ci + 1:02d}"
            part = CACHE_DIR / f"{size}_part{ci + 1:02d}.json"
            cmd = _common(size, group) + ["--baseline-kisialiou-only", "--baseline-kisialiou-cache", str(part),
                                          "--run-tag", tag] + cap_args
            tasks.append(Task("ref", tag, size, sum(k[i] * REF_RATIO[size] for i in group) / 60, [], False, cmd,
                              f"{PREFIX}__{tag}", REPO / "output" / f"{PREFIX}__{tag}", part,
                              {"instances": [sd.names[i] for i in group], "part": str(part), "merged_cache": str(merged_cache)}))
            ref_tags.append(f"ref:{tag}")

        # trained models: main (Exp 8 reward) and alt (Exp 9 best reward)
        variants = [("main", "delta_cost_v9", False)]
        if size in alt_sizes:
            variants.append(("alt", "alns_score_time_v1", True))
        for axis, reward, optional in variants:
            jtag = f"{axis}__{reward}__{size}__s{SEED}"
            train_dir = REPO / "output" / f"{PREFIX}__{jtag}__train"
            cmd = _common(size) + ["--action-module", REF["action"], "--state-module", REF["state"],
                                   "--reward-module", reward, "--train-timesteps", str(TIMESTEPS),
                                   "--ent-coef", "0.01", "--run-tag", f"{jtag}__train"]
            tasks.append(Task("train", f"{jtag}__train", size, TRAIN_MIN[(size, axis)] if not SMOKE else 3, [], optional,
                              cmd, f"{PREFIX}__{jtag}__train", train_dir,
                              train_dir / "model" / "ppo_rl_improvement_mvp.zip", {"reward": reward}))
            model = train_dir / "model" / "ppo_rl_improvement_mvp.zip"
            ratio_kind = axis
            shard_tags = []
            for si, group in enumerate(balanced_groups([k[i] * eval_ratio(size, sd.visits[i], ratio_kind) / 60
                                                         for i in range(len(k))], EVAL_SHARDS[size])):
                stag = f"{jtag}__sh{si + 1:02d}"
                sdir = REPO / "output" / f"{PREFIX}__{stag}"
                cmd = _common(size, group) + ["--action-module", REF["action"], "--state-module", REF["state"],
                                              "--reward-module", reward, "--run-eval", "--resume", "--skip-cache-copy",
                                              "--baseline-kisialiou-cache", str(merged_cache), "--skip-training",
                                              "--model-path", str(model), "--run-tag", stag] + cap_args
                mins = sum(k[i] * eval_ratio(size, sd.visits[i], ratio_kind) for i in group) / 60
                tasks.append(Task("eval", stag, size, mins, ref_tags + [f"train:{jtag}__train"], optional, cmd,
                                  f"{PREFIX}__{stag}", sdir, sdir / "summary.json",
                                  {"instances": [sd.names[i] for i in group], "job": jtag}))
                shard_tags.append(f"eval:{stag}")
            tasks.append(Task("merge", jtag, size, 1.0, shard_tags, optional, [], f"{PREFIX}__{jtag}",
                              REPO / "output" / f"{PREFIX}__{jtag}", REPO / "output" / f"{PREFIX}__{jtag}" / "summary.json",
                              {"shards": shard_tags}))

        # fixed-configuration controls: RL's collapsed triples without RL
        for ctl in (CONTROLS[size] if controls else ()):
            jtag = f"control__fixed_rnd_ff_{ctl}__{size}__s{SEED}"
            shard_tags = []
            ckind = "ctrl_reloc" if ctl == "reloc" else "ctrl"
            for si, group in enumerate(balanced_groups([k[i] * eval_ratio(size, sd.visits[i], ckind) / 60
                                                         for i in range(len(k))], EVAL_SHARDS[size])):
                stag = f"{jtag}__sh{si + 1:02d}"
                sdir = REPO / "output" / f"{PREFIX}__{stag}"
                cmd = _common(size, group) + ["--action-module", REF["action"], "--state-module", REF["state"],
                                              "--reward-module", "delta_cost_v9", "--run-eval", "--resume", "--skip-cache-copy",
                                              "--baseline-kisialiou-cache", str(merged_cache), "--policy", "fixed",
                                              "--fixed-action", str(FIXED_TRIPLES[ctl]), "--run-tag", stag,
                                              ] + (["--operator-log-restarts", str(CONTROL_LOG_RESTARTS[size])]
                                                   if CONTROL_LOG_RESTARTS[size] is not None else [])
                mins = sum(k[i] * eval_ratio(size, sd.visits[i], ckind) for i in group) / 60
                tasks.append(Task("eval", stag, size, mins, ref_tags, False, cmd, f"{PREFIX}__{stag}", sdir,
                                  sdir / "summary.json", {"instances": [sd.names[i] for i in group], "job": jtag}))
                shard_tags.append(f"eval:{stag}")
            tasks.append(Task("merge", jtag, size, 1.0, shard_tags, False, [], f"{PREFIX}__{jtag}",
                              REPO / "output" / f"{PREFIX}__{jtag}", REPO / "output" / f"{PREFIX}__{jtag}" / "summary.json",
                              {"shards": shard_tags}))
    return tasks


# ----------------------------------------------------------------------------- execution state
_lock = threading.Lock()
_state: Dict[str, Dict] = {}   # task name -> {phase, started, finished, attempt}
_started = time.time()


def _event(text: str) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        with EVENTS_PATH.open("a") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {text}\n")


def _git() -> Dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": run("status", "--porcelain").splitlines()}


def _set(name: str, **kw) -> None:
    with _lock:
        _state.setdefault(name, {}).update(kw)


def _running_pids(task: Task) -> List[int]:
    """Processes of this task started by an earlier queue (a restarted runner must not duplicate them)."""
    out = subprocess.run(["pgrep", "-f", "--", f"--run-tag {task.tag}( |$)"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if int(x) != os.getpid()]


def run_subprocess_task(task: Task) -> bool:
    log_path = QUEUE_DIR / f"{task.tag}.log"
    pids = _running_pids(task)
    if pids:
        _set(task.name, phase="run", started=time.time(), attempt=1)
        _event(f"ADOPT {task.kind:5s} {task.tag}: already running (pids {pids}); waiting for it to finish")
        while _running_pids(task):
            time.sleep(30)
        if _task_complete(task):
            _event(f"END   {task.kind:5s} {task.tag} (adopted process)")
            return True
        _event(f"FAIL  {task.kind:5s} {task.tag}: adopted process ended without a result; starting it again")
    for attempt in (1, 2):
        _set(task.name, phase="run", started=time.time(), attempt=attempt)
        _event(f"START {task.kind:5s} {task.tag} attempt={attempt}")
        with log_path.open("a") as log:
            log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} attempt {attempt}: {' '.join(task.command)}\n")
            log.flush()
            rc = subprocess.call(task.command, cwd=REPO, env=task_env(task.run_id), stdout=log, stderr=subprocess.STDOUT)
        ok = rc == 0 and _task_complete(task)
        _event(f"{'END  ' if ok else 'FAIL '} {task.kind:5s} {task.tag} rc={rc} "
               f"{(time.time() - _state[task.name]['started']) / 60:.1f}min")
        if ok:
            return True
    return False


def _task_complete(task: Task) -> bool:
    if task.kind == "train":
        manifest = task.out_dir / "run_manifest.json"
        try:
            return task.done() and "finished_at" in json.loads(manifest.read_text())
        except (OSError, ValueError):
            return False
    if task.kind == "ref":
        try:
            return len(json.loads(task.done_file.read_text())) >= len(task.meta["instances"])
        except (OSError, ValueError):
            return False
    return task.done()


def merge_reference(size: str, tasks: List[Task]) -> None:
    chunks = [t for t in tasks if t.kind == "ref" and t.size == size]
    target = Path(chunks[0].meta["merged_cache"])
    with _lock:
        if target.exists() or not all(_task_complete(c) for c in chunks):
            return
        merged: Dict[str, object] = {}
        for c in chunks:
            merged.update(json.loads(c.done_file.read_text()))
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(merged))
        tmp.replace(target)
    _event(f"REFERENCE {size} merged: {len(merged)} instances -> {target}")


def _load_experiment_script():
    spec = importlib.util.spec_from_file_location("exp_script", REPO / "scripts" / "run_experiment_rl_improvement_mvp.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO))
    spec.loader.exec_module(module)
    return module


def merge_shards(task: Task) -> bool:
    """Combine the evaluation shards of one job into a standard output folder (results, summary, manifest)."""
    out = task.out_dir
    out.mkdir(parents=True, exist_ok=True)
    shard_dirs = [REPO / "output" / f"{PREFIX}__{s.split(':', 1)[1]}" for s in task.meta["shards"]]
    rows: List[Dict] = []
    for d in shard_dirs:
        rows.extend(json.loads((d / "results.json").read_text()))
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["mode"])))
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    with (out / "results.csv").open("w", newline="") as handle:
        fields: List[str] = []
        for r in rows:
            for key in r:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # restart rows: union of columns, streamed
    fields = []
    for d in shard_dirs:
        with (d / "restart_results.csv").open(newline="") as handle:
            for key in next(csv.reader(handle)):
                if key not in fields:
                    fields.append(key)
    with (out / "restart_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for d in shard_dirs:
            with (d / "restart_results.csv").open(newline="") as src:
                for row in csv.DictReader(src):
                    writer.writerow(row)
    module = _load_experiment_script()
    summary = module.compute_summary(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    manifests = []
    for d in shard_dirs:
        m = json.loads((d / "run_manifest.json").read_text())
        manifests.append({"run_id": m.get("run_id"), "output_dir": str(d), "log_dir": m.get("log_dir"),
                          "command": m.get("command"), "created_at": m.get("created_at"),
                          "finished_at": m.get("finished_at"), "instances": len(m["resolved"]["test_paths"])})
    first = json.loads((shard_dirs[0] / "run_manifest.json").read_text())
    merged_manifest = {
        "kind": "merged_evaluation", "job": task.tag, "created_at": datetime.now().isoformat(),
        "rows": len(rows), "instances": len({r["dataset"] for r in rows}),
        "shards": manifests,
        "cli_args_of_first_shard": first.get("cli_args"),
        "resolved_of_first_shard": {k: v for k, v in first.get("resolved", {}).items() if k != "test_paths"},
        "git": first.get("git"), "runtime": first.get("runtime"),
    }
    (out / "run_manifest.json").write_text(json.dumps(merged_manifest, indent=2))
    _event(f"MERGE {task.tag}: {len(rows)} rows from {len(shard_dirs)} shards")
    return True


def run_task(task: Task, all_tasks: List[Task]) -> None:
    if _task_complete(task):
        if task.kind == "ref":
            merge_reference(task.size, all_tasks)
        _set(task.name, phase="done", finished=time.time())
        return
    if task.kind == "merge":
        _set(task.name, phase="run", started=time.time())
        try:
            ok = merge_shards(task)
        except Exception as exc:  # noqa: BLE001
            _event(f"FAIL  merge {task.tag}: {exc!r}")
            ok = False
        _set(task.name, phase="done" if ok else "failed", finished=time.time())
        return
    ok = run_subprocess_task(task)
    if ok and task.kind == "ref":
        merge_reference(task.size, all_tasks)  # before "done": evaluations wait on the merged cache
    _set(task.name, phase="done" if ok else "failed", finished=time.time())


# ----------------------------------------------------------------------------- load fillers
FILLER_CODE = "import os\np = os.getppid()\nwhile os.getppid() == p:\n    sum(range(200000))"
_fillers: List[subprocess.Popen] = []


def _busy_job_processes() -> int:
    out = subprocess.run(["ps", "-Ao", "stat=,command="], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines()
               if "run_experiment_rl_improvement_mvp.py" in line and not line.lstrip().startswith("T"))


def _budgeted_running() -> bool:
    with _lock:
        return any(st.get("phase") == "run" and (n.startswith("ref:") or n.startswith("eval:")) for n, st in _state.items())


def filler_loop(stop: threading.Event, workers: int) -> None:
    while not stop.is_set():
        want = max(0, workers - _busy_job_processes()) if _budgeted_running() else 0
        _fillers[:] = [f for f in _fillers if f.poll() is None]
        while len(_fillers) < want:
            _fillers.append(subprocess.Popen([PY, "-c", FILLER_CODE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        while len(_fillers) > want:
            _fillers.pop().kill()
        stop.wait(3)
    for f in _fillers:
        f.kill()


def probe_loop(stop: threading.Event, interval_s: int = 1800) -> None:
    """Log the CPU time of a fixed piece of search work every 30 min (machine-speed drift record)."""
    path = QUEUE_DIR / "speed_probe.csv"
    if not path.exists():
        header = subprocess.run([PY, "scripts/timing/speed_probe.py", "--header"], cwd=REPO, capture_output=True, text=True).stdout
        path.write_text(header)
    while not stop.is_set():
        busy = _busy_job_processes() + len([f for f in _fillers if f.poll() is None])
        out = subprocess.run([PY, "scripts/timing/speed_probe.py", "--busy", str(busy)], cwd=REPO,
                             env=task_env("probe"), capture_output=True, text=True).stdout.strip()
        if out:
            with path.open("a") as fh:
                fh.write(out.splitlines()[-1] + "\n")
        stop.wait(interval_s)


# ----------------------------------------------------------------------------- status
def _tail(path: Path, nbytes: int = 30000) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(fh.tell() - nbytes, 0))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _progress(task: Task) -> str:
    if task.kind == "train":
        hits = re.findall(r"total_timesteps\s*\|\s*(\d+)", _tail(QUEUE_DIR / f"{task.tag}.log"))
        return f"{min(int(hits[-1]) / TIMESTEPS, 1.0) * 100:.0f}% of {TIMESTEPS // 1000}k steps" if hits else "starting"
    if task.kind == "eval":
        try:
            p = json.loads((task.out_dir / "progress.json").read_text())
            return f"{p.get('done')}/{p.get('total')} instances"
        except (OSError, ValueError):
            return "starting"
    if task.kind == "ref":
        try:
            return f"{len(json.loads(task.done_file.read_text()))}/{len(task.meta['instances'])} instances"
        except (OSError, ValueError):
            return "starting"
    return ""


def _partial_results_rows(tasks: List[Task]) -> List[str]:
    """One row per evaluation job from the shards' partial_rows.jsonl (written after every instance)."""
    jobs: Dict[str, List[Task]] = {}
    for t in tasks:
        if t.kind == "eval":
            jobs.setdefault(t.meta["job"], []).append(t)
    out: List[str] = []
    modes = ("rl_improvement_mvp", "baseline", "kisialiou")
    for job, shards in sorted(jobs.items()):
        by: Dict[str, Dict[str, float]] = {}
        for t in shards:
            try:
                text = (t.out_dir / "partial_rows.jsonl").read_text()
            except OSError:
                continue
            for line in text.splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a line being written right now
                by.setdefault(str(r["dataset"]), {})[str(r["mode"])] = float(r["best_cost"])
        full = [d for d in by.values() if all(m in d for m in modes)]
        total = sum(len(t.meta["instances"]) for t in shards)
        if not full:
            out.append(f"| {job} | 0/{total} | | | | | |")
            continue
        ratio = {m: sum(d[m] / min(d[x] for x in modes) for d in full) / len(full) for m in modes}

        def wtl(other: str) -> str:
            w = sum(1 for d in full if d[modes[0]] < d[other] - 1e-9)
            lo = sum(1 for d in full if d[modes[0]] > d[other] + 1e-9)
            return f"{w}/{len(full) - w - lo}/{lo}"

        out.append(f"| {job} | {len(full)}/{total} | {ratio[modes[0]]:.4f} | {ratio[modes[1]]:.4f} | {ratio[modes[2]]:.4f} | "
                   f"{wtl('kisialiou')} | {wtl('baseline')} |")
    return out


def write_status(tasks: List[Task], workers: int) -> None:
    with _lock:
        states = {n: dict(s) for n, s in _state.items()}
    counts = {"done": 0, "run": 0, "failed": 0}
    rows, remaining = [], 0.0
    for t in tasks:
        st = states.get(t.name, {})
        phase = st.get("phase", "queued")
        if phase in counts:
            counts[phase] += 1
        if phase == "run" and t.kind != "merge":
            elapsed = (time.time() - st["started"]) / 60
            rows.append(f"| {t.name} | {t.size} | running (attempt {st.get('attempt', 1)}) | {elapsed:.0f} min | {_progress(t)} |")
            remaining += max(t.minutes - elapsed, 0.0)
        elif phase not in ("done", "failed"):
            remaining += t.minutes
    eta_h = remaining / max(workers, 1) / 60
    lines = [f"# Experiment 8 rerun — status ({PREFIX})", "",
             f"Updated {datetime.now():%Y-%m-%d %H:%M:%S} · queue running {(time.time() - _started) / 3600:.1f} h · "
             f"**{counts['done']}/{len(tasks)} tasks done**, {counts['run']} running, "
             f"{len(tasks) - sum(counts.values())} queued, {counts['failed']} failed", "",
             f"SSD free: {free_gb():.0f} GB (new tasks pause below {MIN_FREE_GB:.0f} GB)", "",
             f"Rough ETA for everything: ~{eta_h:.1f} h (≈ {(datetime.now() + timedelta(hours=eta_h)):%a %d %b %H:%M}) from "
             f"predicted task durations / {workers} workers; refines as tasks finish.", "",
             "## Running", "", "| task | size | state | elapsed | progress |", "|---|---|---|---|---|"]
    lines += rows or ["| — | | | | |"]
    lines += ["", "## Results so far (finished instances of every evaluation job, updated as instances finish)", "",
              "Mean ratio to the best of the three methods on each finished instance (1.0000 = best); policy = the job's "
              "trained model or fixed control. W/T/L = per-instance best-cost wins/ties/losses of the policy.", "",
              "| job | instances | policy | baseline | Kisialiou | W/T/L vs Kisialiou | W/T/L vs baseline |", "|---|---|---|---|---|---|---|"]
    lines += _partial_results_rows(tasks)
    failed = [t.name for t in tasks if states.get(t.name, {}).get("phase") == "failed"]
    lines += ["", "## Failures", "", ", ".join(failed) if failed else "none", "",
              f"Event log: `{EVENTS_PATH.relative_to(REPO)}` · per-task logs: `{QUEUE_DIR.relative_to(REPO)}/`"]
    STATUS_PATH.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- scheduler
def schedule(tasks: List[Task], workers: int) -> None:
    by_name = {t.name: t for t in tasks}
    children: Dict[str, List[str]] = {t.name: [] for t in tasks}
    for t in tasks:
        for d in t.deps:
            children[d].append(t.name)
    level: Dict[str, float] = {}

    def lv(n: str) -> float:
        if n not in level:
            level[n] = by_name[n].minutes + max((lv(c) for c in children[n]), default=0.0)
        return level[n]

    for t in tasks:
        lv(t.name)
    stop = threading.Event()
    threading.Thread(target=filler_loop, args=(stop, workers), daemon=True).start()
    threading.Thread(target=probe_loop, args=(stop, 120 if SMOKE else 1800), daemon=True).start()
    threads: Dict[str, threading.Thread] = {}
    last_status = 0.0
    low_disk_reported = False
    try:
        while True:
            with _lock:
                phases = {n: s.get("phase") for n, s in _state.items()}
            running = [n for n, th in threads.items() if th.is_alive()]
            pending = [t for t in tasks if t.name not in threads]
            if not pending and not running:
                break
            ready = []
            for t in pending:
                dep_phases = [phases.get(d) for d in t.deps]
                if any(p == "failed" for p in dep_phases):
                    _set(t.name, phase="failed")
                    _event(f"BLOCKED {t.name}: a dependency failed")
                    threads[t.name] = threading.Thread(target=lambda: None)
                    threads[t.name].start()
                elif all(p == "done" for p in dep_phases):
                    ready.append(t)
            # required work first, then optional work, each by highest level first
            ready.sort(key=lambda t: (t.optional, -level[t.name]))
            free = workers - len([n for n in running if by_name[n].kind != "merge"])
            disk_ok = free_gb() >= MIN_FREE_GB
            if not disk_ok and not low_disk_reported:
                _event(f"PAUSED: only {free_gb():.0f} GB free on the SSD (< {MIN_FREE_GB:.0f}); no new tasks until space is freed")
                low_disk_reported = True
            elif disk_ok and low_disk_reported:
                _event(f"RESUMED: {free_gb():.0f} GB free on the SSD")
                low_disk_reported = False
            for t in ready:
                if t.kind != "merge" and (free <= 0 or not disk_ok):
                    continue
                th = threading.Thread(target=run_task, args=(t, tasks), daemon=True)
                threads[t.name] = th
                _set(t.name, phase="run", started=time.time())
                th.start()
                if t.kind != "merge":
                    free -= 1
            if time.time() - last_status > 60:
                write_status(tasks, workers)
                last_status = time.time()
            time.sleep(5)
    finally:
        write_status(tasks, workers)
        stop.set()


def write_queue_manifest(tasks: List[Task], args: argparse.Namespace, data: Dict[str, SizeData]) -> None:
    manifest = {
        "kind": "experiment_8_rerun_queue", "prefix": PREFIX, "created_at": datetime.now().isoformat(),
        "arguments": vars(args), "git": _git(),
        "machine": {"platform": platform.platform(), "python": PY, "cpu_count": os.cpu_count(), "workers": args.workers},
        "protocol": {
            "time_budget": "search CPU time of Kisialiou's 5 reference restarts (x 300 iterations); RL charged native per-decision cost",
            "baseline_action_module": BASELINE_MODULE,
            "constant_load": f"{args.workers} busy processes whenever a reference/evaluation task runs",
            "seed": SEED, "train_timesteps": TIMESTEPS, "ent_coef": 0.01, "progress_horizon": 300,
            "operator_log_restarts": OPERATOR_LOG_RESTARTS, "n_test_per_size": N_TEST,
            "controls": {s: list(CONTROLS[s]) for s in CONTROLS} if not args.no_controls else {},
            "alt_sizes": args.alt_sizes,
        },
        "data": {"generated_root": str(GENERATED), "processed_root": str(PROCESSED),
                 "manifest": str(DATA_ROOT / "DATA_MANIFEST.json")},
        "tasks": [{"name": t.name, "kind": t.kind, "size": t.size, "optional": t.optional, "deps": t.deps,
                   "predicted_minutes": round(t.minutes, 1), "run_id": t.run_id,
                   "output_dir": str(t.out_dir) if t.out_dir else None,
                   "log_dir": str(REPO / "logs" / t.run_id), "command": " ".join(t.command), "meta": t.meta}
                  for t in tasks],
    }
    QUEUE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_MANIFEST.write_text(json.dumps(manifest, indent=2))


def simulate(tasks: List[Task], workers: int, scale: float = 1.0) -> tuple:
    """List-schedule the predicted durations (the runner's own priority rule): (finish days, process hours)."""
    by_name = {t.name: t for t in tasks}
    children = {t.name: [] for t in tasks}
    for t in tasks:
        for d in t.deps:
            children[d].append(t.name)
    level: Dict[str, float] = {}

    def lv(n: str) -> float:
        if n not in level:
            level[n] = by_name[n].minutes + max((lv(c) for c in children[n]), default=0.0)
        return level[n]

    done, running, now, free = {}, [], 0.0, workers
    pending = {t.name for t in tasks}
    busy = sum(t.minutes for t in tasks if t.kind != "merge") * scale
    while pending or running:
        ready = sorted((n for n in pending if all(d in done for d in by_name[n].deps)),
                       key=lambda n: (by_name[n].optional, -lv(n)))
        for n in ready:
            if by_name[n].kind != "merge" and free <= 0:
                continue
            pending.discard(n)
            heapq.heappush(running, (now + by_name[n].minutes * scale, n))
            if by_name[n].kind != "merge":
                free -= 1
        if not running:
            break
        now, n = heapq.heappop(running)
        done[n] = now
        if by_name[n].kind != "merge":
            free += 1
    return now / 60 / 24, busy / 60


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--alt-sizes", default="", help="sizes that also train/evaluate the alns_score_time_v1 model (comma list)")
    parser.add_argument("--no-controls", action="store_true", help="skip the fixed-configuration controls")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end check with its own prefix")
    parser.add_argument("--list", action="store_true", help="print the plan and predicted durations, execute nothing")
    parser.add_argument("--status", action="store_true", help="print the status file")
    parser.add_argument("--prepare-data-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _configure_smoke()
        args.alt_sizes = args.alt_sizes or "small"
    if args.status:
        print(STATUS_PATH.read_text() if STATUS_PATH.exists() else "no status file yet")
        return
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = ensure_data()
    write_data_manifest(data)
    if args.prepare_data_only:
        print("data ready:", {s: len(d.names) for s, d in data.items()})
        return
    alt_sizes = [s for s in args.alt_sizes.split(",") if s]
    tasks = build_tasks(data, alt_sizes, controls=not args.no_controls)
    write_queue_manifest(tasks, args, data)
    if args.list:
        for t in tasks:
            print(f"{t.name:70s} {t.size:6s} {t.minutes / 60:7.2f} h  {'optional' if t.optional else ''}")
        for scale in (0.8, 1.0, 1.4):
            days, hours = simulate(tasks, args.workers, scale)
            print(f"scale x{scale}: process-hours {hours:.0f}, scheduled finish {days:.2f} days")
        return
    _event(f"QUEUE START workers={args.workers} tasks={len(tasks)} prefix={PREFIX} alt={alt_sizes} controls={not args.no_controls}")
    schedule(tasks, args.workers)
    states = {t.name: _state.get(t.name, {}).get("phase") for t in tasks}
    manifest = json.loads(QUEUE_MANIFEST.read_text())
    manifest["finished_at"] = datetime.now().isoformat()
    manifest["outcome"] = states
    QUEUE_MANIFEST.write_text(json.dumps(manifest, indent=2))
    _event(f"QUEUE END done={sum(1 for v in states.values() if v == 'done')}/{len(tasks)}")


if __name__ == "__main__":
    main()
