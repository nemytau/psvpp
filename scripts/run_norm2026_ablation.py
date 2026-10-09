"""NORM 2026 state / action / reward design ablation (small + medium).

One-factor-at-a-time around the Experiment 8 reference configuration -- op_triple_v2 /
features_improvement_v3 / delta_cost_v9 / ent_coef 0.01 / ph300 / 200k PPO steps -- plus
non-learned control policies. Each job trains a PPO model (train-only run) and then
evaluates it (eval-only run) on the first test instances of its size (small 30, medium 20),
eval seed 42. Training seeds: small 42/43 per variant, medium 42 (reference: medium
42/43/44, small 42/43).

Time budgets are search CPU time (engine CPU time actually spent searching, measured in ns
inside the Rust engine; Python overhead excluded): RL and the adaptive baseline get exactly
the search CPU time Kisialiou needs for its 5 reference restarts on each instance. The
Kisialiou/baseline reference is therefore regenerated under this rule first, in parallel
chunks (runs/baseline_kisialiou_cache/search_cpu_v1/), before any evaluation starts.

On the M1 the CPU time a fixed amount of search costs depends on machine load (measured:
2.37 s alone, ~3.1 s with 3 jobs, ~3.9 s with 4 jobs, for identical work), so every
time-budgeted task must run under the same load: whenever a reference or evaluation task
is running, filler processes keep exactly --workers CPU-bound processes busy (they exit by
themselves if this runner dies). The machine should otherwise be idle.

Layout per job tag "<axis>__<variant>__<size>__s<seed>":
    output/norm26__<tag>__train/model/ppo_rl_improvement_mvp.zip   (trained model)
    output/norm26__<tag>/{results.csv,summary.json,progress.json}   (evaluation)
    logs/norm2026_queue/<tag>__{train,eval}.log                      (stdout of each phase)

Resumable at every level: finished reference chunks, trainings and evaluations are never
redone, an interrupted evaluation or reference chunk resumes after its last finished
instance, and a training process that was paused (SIGSTOP) is resumed instead of restarted.
A failed task is retried once. Progress is written every minute to NORM2026_STATUS.md.

    nohup caffeinate -is python scripts/run_norm2026_ablation.py --workers 4 > logs/norm2026_queue/queue.log 2>&1 &
    python scripts/run_norm2026_ablation.py --status
    python scripts/run_norm2026_ablation.py --list
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
PY = "/Users/timothy/opt/miniconda3/envs/py312/bin/python"
PREFIX = "norm26"
QUEUE_DIR = REPO / "logs" / "norm2026_queue"
STATUS_PATH = REPO / "NORM2026_STATUS.md"
EVENTS_PATH = QUEUE_DIR / "events.log"

DATASETS_BY_SIZE = {"small": 30, "medium": 20}
REFERENCE_PARTS = {"medium": 5, "small": 2}
TIMESTEPS = 200_000


def _configure(prefix: str, datasets: Optional[int], timesteps: int) -> None:
    """Override output prefix / dataset count / training steps (used for smoke tests)."""
    global PREFIX, QUEUE_DIR, STATUS_PATH, EVENTS_PATH, DATASETS_BY_SIZE, TIMESTEPS
    PREFIX, TIMESTEPS = prefix, timesteps
    if datasets is not None:
        DATASETS_BY_SIZE = {size: datasets for size in DATASETS_BY_SIZE}
    if prefix != "norm26":
        QUEUE_DIR = REPO / "logs" / f"{prefix}_queue"
        STATUS_PATH = QUEUE_DIR / "STATUS.md"
        EVENTS_PATH = QUEUE_DIR / "events.log"
REF = dict(action="op_triple_v2", state="features_improvement_v3", reward="delta_cost_v9")

# Experiment 8 models (reference config, training seed 42): re-evaluated, not retrained, so
# they are scored under the same machine load as every other job.
EXP8_MODELS = {
    "small": "output/rl_improvement_mvp_20260906_063629__exp_reward_v9_timepenalty_op_triple_v2_small_ph300_50ds1seed/model/ppo_rl_improvement_mvp.zip",
    "medium": "output/rl_improvement_mvp_20260906_070345__exp_reward_v9_timepenalty_op_triple_v2_medium_ph300_50ds1seed/model/ppo_rl_improvement_mvp.zip",
}

# op_triple_v2 index = d*16 + r*4 + i; d (shaw, random, worst), r (greedy, regret2, regret3,
# first-fit), i (VNR, FCR, relocation, swap). These are the triples Exp 8's policies collapsed to.
FIXED_TRIPLES = {"rnd_ff_fcr": 29, "rnd_ff_reloc": 30}

# Rough per-task durations (minutes) for the ETA before a task reports progress, under the
# 4-job load. Training: past runs (46-280 min medium, 7-18 min small) x load. Evaluation:
# Kisialiou's search CPU under load (~15 s small / ~175 s medium per instance) x the RL
# wall/search overhead with operator logging (measured 1.7x small, 4.5x medium).
# Reference: Kisialiou (~1.25x its search time) + baseline (~2.6x the same budget).
EST_TRAIN_MIN = {"small": 30.0, "medium": 150.0}
EST_EVAL_MIN_PER_INSTANCE = {"small": 0.75, "medium": 10.0}
EST_REFERENCE_MIN_PER_INSTANCE = {"small": 1.0, "medium": 11.0}


def _cache_dir() -> Path:
    suffix = "" if PREFIX == "norm26" else f"_{PREFIX}"
    return REPO / "runs" / "baseline_kisialiou_cache" / f"search_cpu_v1{suffix}"


def merged_cache(size: str) -> Path:
    return _cache_dir() / f"cache_{size}_seed42.json"


@dataclass
class RefChunk:
    """Kisialiou (5 reference restarts) + adaptive baseline for a slice of the test set."""

    size: str
    start: int
    count: int

    @property
    def tag(self) -> str:
        return f"reference__{self.size}__{self.start + 1:02d}-{self.start + self.count:02d}"

    @property
    def part_path(self) -> Path:
        return _cache_dir() / f"{self.size}_part{self.start + 1:02d}.json"

    def entries(self) -> int:
        try:
            return len(json.loads(self.part_path.read_text()))
        except (OSError, ValueError):
            return 0

    def done(self) -> bool:
        return self.entries() >= self.count

    def command(self) -> List[str]:
        return [
            PY, "scripts/run_experiment_rl_improvement_mvp.py",
            "--sizes", self.size,
            "--dataset-start", str(self.start),
            "--datasets-per-size", str(self.count),
            "--eval-seeds", "42",
            "--max-iterations", "300",
            "--seed", "42",
            "--baseline-kisialiou-only",
            "--baseline-kisialiou-cache", str(self.part_path),
            "--run-tag", self.tag,
            "--force-overwrite",
        ]


def reference_chunks() -> List[RefChunk]:
    chunks: List[RefChunk] = []
    for size in ("medium", "small"):
        total = DATASETS_BY_SIZE[size]
        step = math.ceil(total / REFERENCE_PARTS[size])
        for start in range(0, total, step):
            chunks.append(RefChunk(size, start, min(step, total - start)))
    return chunks


@dataclass
class Job:
    axis: str
    variant: str
    size: str
    seed: int = 42
    action: str = REF["action"]
    state: str = REF["state"]
    reward: str = REF["reward"]
    policy: str = "ppo"            # ppo | random | fixed
    fixed_action: Optional[int] = None
    model_path: Optional[str] = None  # pre-trained model: skip the training phase

    @property
    def tag(self) -> str:
        return f"{self.axis}__{self.variant}__{self.size}__s{self.seed}"

    @property
    def needs_training(self) -> bool:
        return self.policy == "ppo" and self.model_path is None

    @property
    def train_dir(self) -> Path:
        return REPO / "output" / f"{PREFIX}__{self.tag}__train"

    @property
    def eval_dir(self) -> Path:
        return REPO / "output" / f"{PREFIX}__{self.tag}"

    @property
    def trained_model(self) -> Path:
        return Path(self.model_path) if self.model_path else self.train_dir / "model" / "ppo_rl_improvement_mvp.zip"

    def _common(self) -> List[str]:
        return [
            PY, "scripts/run_experiment_rl_improvement_mvp.py",
            "--sizes", self.size,
            "--datasets-per-size", str(DATASETS_BY_SIZE[self.size]),
            "--eval-seeds", "42",
            "--max-iterations", "300",
            "--seed", str(self.seed),
            "--action-module", self.action,
            "--state-module", self.state,
            "--reward-module", self.reward,
            "--force-overwrite",
        ]

    def train_command(self) -> List[str]:
        return self._common() + [
            "--train-timesteps", str(TIMESTEPS),
            "--ent-coef", "0.01",
            "--run-tag", f"{self.tag}__train",
        ]

    def eval_command(self) -> List[str]:
        cmd = self._common() + [
            "--run-eval", "--resume", "--skip-cache-copy",
            "--baseline-kisialiou-cache", str(merged_cache(self.size)),
            "--run-tag", self.tag,
        ]
        if self.policy == "ppo":
            cmd += ["--skip-training", "--model-path", str(self.trained_model)]
        else:
            cmd += ["--policy", self.policy]
            if self.fixed_action is not None:
                cmd += ["--fixed-action", str(self.fixed_action)]
        return cmd


def build_jobs() -> List[Job]:
    jobs: List[Job] = []
    for size in ("medium", "small"):
        seeds = (42,) if size == "medium" else (42, 43)
        # Trained variants, longest-expected first within each size (non-time-aware
        # rewards and the 288-action space risk slow, expensive-operator policies).
        for seed in seeds:
            jobs += [
                Job("action", "op_triple_many_v1", size, seed, action="op_triple_many_v1"),
                Job("reward", "alns_score_v1", size, seed, reward="alns_score_v1"),
                Job("reward", "new_best_v1", size, seed, reward="new_best_v1"),
                Job("reward", "best_gain_v1", size, seed, reward="best_gain_v1"),
                Job("action", "improvement_chain_v2", size, seed, action="improvement_chain_v2"),
                Job("reward", "alns_score_time_v1", size, seed, reward="alns_score_time_v1"),
                Job("state", "null", size, seed, state="features_null_v1"),
                Job("state", "instance", size, seed, state="features_instance_v1"),
                Job("state", "search", size, seed, state="features_search_v1"),
                Job("state", "solution", size, seed, state="features_solution_v1"),
                Job("state", "all", size, seed, state="features_all_v1"),
            ]
        # Reference retrained under the search-CPU-time reward (Exp 8's models used wall time).
        ref_seeds = (42, 43, 44) if size == "medium" else (42, 43)
        jobs += [Job("ref", "op_triple_v2", size, seed) for seed in ref_seeds]
        jobs.append(Job("control", "random_triple", size, policy="random"))
        jobs.append(Job("control", "random_many", size, action="op_triple_many_v1", policy="random"))
        for name, idx in FIXED_TRIPLES.items():
            jobs.append(Job("control", f"fixed_{name}", size, policy="fixed", fixed_action=idx))
    return jobs


# ----------------------------------------------------------------------------- state

_lock = threading.Lock()
_state: Dict[str, Dict] = {}   # tag -> {phase, started, phase_started, finished, failures}
_queue_started = time.time()


def _event(text: str) -> None:
    with _lock:
        with EVENTS_PATH.open("a") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {text}\n")


def _set(tag: str, **kw) -> None:
    with _lock:
        _state.setdefault(tag, {}).update(kw)


def _eval_done(job: Job) -> bool:
    return (job.eval_dir / "summary.json").exists()


def _run_command(tag: str, phase: str, cmd: List[str], run_id: str, ok: Callable[[], bool]) -> bool:
    """Run one task command (stdout to logs/.../<tag>__<phase>.log); retried once on failure."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(REPO), RL_RUN_ID=run_id)
    log_path = QUEUE_DIR / f"{tag}__{phase}.log"
    for attempt in (1, 2):
        _set(tag, phase=phase, phase_started=time.time(), attempt=attempt)
        _event(f"START {phase:5s} {tag} attempt={attempt}")
        with log_path.open("a") as log:
            log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} attempt {attempt}: {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        success = rc == 0 and ok()
        _event(f"{'END  ' if success else 'FAIL '} {phase:5s} {tag} rc={rc} "
               f"{(time.time() - _state[tag]['phase_started']) / 60:.1f}min")
        if success:
            return True
    return False


def _merge_reference(size: str) -> None:
    chunks = [c for c in reference_chunks() if c.size == size]
    target = merged_cache(size)
    with _lock:
        if target.exists() or not all(c.done() for c in chunks):
            return
        merged: Dict[str, object] = {}
        for chunk in chunks:
            merged.update(json.loads(chunk.part_path.read_text()))
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(merged))
        tmp.replace(target)
    _event(f"REFERENCE {size} merged: {len(merged)} instances")


def run_reference(chunk: RefChunk) -> None:
    if not chunk.done():
        chunk.part_path.parent.mkdir(parents=True, exist_ok=True)
        if not _run_command(chunk.tag, "ref", chunk.command(), f"{PREFIX}__{chunk.tag}", chunk.done):
            _set(chunk.tag, phase="failed (ref)", finished=time.time())
            return
    _set(chunk.tag, phase="done", finished=time.time())
    _merge_reference(chunk.size)


def _training_pids(job: Job) -> List[int]:
    out = subprocess.run(["pgrep", "-f", "--", f"--run-tag {job.tag}__train"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def run_training(job: Job) -> None:
    if not job.needs_training or job.trained_model.exists():
        return
    pids = _training_pids(job)
    if pids:
        # A training started by an earlier queue and paused (SIGSTOP): resume it, don't restart.
        _set(job.tag, phase="train", phase_started=time.time(), attempt=1, resumed=True)
        _event(f"RESUME train {job.tag} pids={pids}")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        while _training_pids(job):
            time.sleep(20)
        if job.trained_model.exists():
            _event(f"END   train {job.tag} (resumed process) {(time.time() - _state[job.tag]['phase_started']) / 60:.1f}min")
            _set(job.tag, phase="trained")
            return
        _event(f"FAIL  train {job.tag} (resumed process ended without a model); training afresh")
    if not _run_command(job.tag, "train", job.train_command(), f"{PREFIX}__{job.tag}__train",
                        job.trained_model.exists):
        _set(job.tag, phase="failed (train)", finished=time.time())
        return
    _set(job.tag, phase="trained")


def run_evaluation(job: Job) -> None:
    if _eval_done(job):
        _set(job.tag, phase="done", finished=time.time())
        return
    # Needs its size's search-CPU reference and (for trained variants) its model.
    while True:
        phase = str(_state.get(job.tag, {}).get("phase", ""))
        if phase.startswith("failed"):
            return
        # The model file appears as soon as SB3 starts writing it, so also require that this
        # queue's training task (if it ran one) has finished.
        ready_model = not job.needs_training or (job.trained_model.exists() and phase != "train")
        if merged_cache(job.size).exists() and ready_model:
            break
        _set(job.tag, eval_waiting=True)
        time.sleep(30)
    _set(job.tag, eval_waiting=False)
    if not _run_command(job.tag, "eval", job.eval_command(), f"{PREFIX}__{job.tag}", lambda: _eval_done(job)):
        _set(job.tag, phase="failed (eval)", finished=time.time())
        return
    _set(job.tag, phase="done", finished=time.time())


# A filler is a CPU-bound Python loop that exits as soon as its parent (this runner) is gone,
# so it can never be left spinning after a crash or kill.
FILLER_CODE = "import os\np = os.getppid()\nwhile os.getppid() == p:\n    sum(range(200000))"
_fillers: List[subprocess.Popen] = []


def _busy_job_processes() -> int:
    """Running (not stopped) experiment processes, whoever started them."""
    out = subprocess.run(["ps", "-Ao", "stat=,command="], capture_output=True, text=True).stdout
    return sum(
        1 for line in out.splitlines()
        if "run_experiment_rl_improvement_mvp.py" in line and not line.lstrip().startswith("T")
    )


def _budgeted_task_running() -> bool:
    with _lock:
        return any(st.get("phase") in ("ref", "eval") for st in _state.values())


def filler_loop(stop: threading.Event, workers: int) -> None:
    """Keep exactly `workers` busy processes while any time-budgeted task runs."""
    while not stop.is_set():
        want = max(0, workers - _busy_job_processes()) if _budgeted_task_running() else 0
        _fillers[:] = [f for f in _fillers if f.poll() is None]
        while len(_fillers) < want:
            _fillers.append(subprocess.Popen([PY, "-c", FILLER_CODE], stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL))
        while len(_fillers) > want:
            _fillers.pop().kill()
        stop.wait(3)
    for filler in _fillers:
        filler.kill()


def build_tasks(jobs: List[Job]) -> List[Tuple[str, object]]:
    """Order: paused trainings (resume at once), reference chunks, remaining trainings, evaluations."""
    paused = [j for j in jobs if j.needs_training and not j.trained_model.exists() and _training_pids(j)]
    tasks: List[Tuple[str, object]] = [("train", j) for j in paused]
    tasks += [("ref", c) for c in reference_chunks() if c.size in {j.size for j in jobs}]
    tasks += [("train", j) for j in jobs if j.needs_training and j not in paused]
    tasks += [("eval", j) for j in jobs]
    return tasks


def run_task(task: Tuple[str, object]) -> None:
    kind, item = task
    {"ref": run_reference, "train": run_training, "eval": run_evaluation}[kind](item)


# ---------------------------------------------------------------------------- status

def _tail(path: Path, nbytes: int = 30000) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(fh.tell() - nbytes, 0))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _train_progress(job: Job) -> Optional[float]:
    hits = re.findall(r"total_timesteps\s*\|\s*(\d+)", _tail(QUEUE_DIR / f"{job.tag}__train.log"))
    return min(int(hits[-1]) / TIMESTEPS, 1.0) if hits else None


def _eval_progress(job: Job) -> Optional[Dict]:
    try:
        return json.loads((job.eval_dir / "progress.json").read_text())
    except (OSError, ValueError):
        return None


def _summary(job: Job) -> Optional[Dict]:
    try:
        return json.loads((job.eval_dir / "summary.json").read_text())["rl_improvement_mvp"]
    except (OSError, ValueError, KeyError):
        return None


def _remaining_minutes(job: Job) -> float:
    st = _state.get(job.tag, {})
    phase = st.get("phase")
    if phase == "done" or (phase or "").startswith("failed") or _eval_done(job):
        return 0.0
    train_left = 0.0
    if job.needs_training and not job.trained_model.exists() and phase != "trained":
        frac = _train_progress(job) if phase == "train" else None
        if frac and frac > 0.02:
            elapsed = (time.time() - st["phase_started"]) / 60
            train_left = elapsed * (1 - frac) / frac
        else:
            train_left = EST_TRAIN_MIN[job.size]
    eval_left = EST_EVAL_MIN_PER_INSTANCE[job.size] * DATASETS_BY_SIZE[job.size]
    prog = _eval_progress(job) if phase == "eval" else None
    if prog and prog.get("done"):
        eval_left *= 1 - prog["done"] / max(prog["total"], 1)
    return train_left + eval_left


def _system_lines() -> List[str]:
    lines = []
    try:
        load = os.getloadavg()
        lines.append(f"load average {load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f} (4 performance cores)")
    except OSError:
        pass
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", vm).group(1))
        pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", vm)}
        lines.append(f"available RAM ~{sum(pages.values()) * page / 1e9:.1f} GB of 16 GB")
    except Exception:
        pass
    lines.append(f"constant-load filler processes running: {sum(1 for f in _fillers if f.poll() is None)}")
    for label, path in (("external SSD (output/logs)", REPO / "output"), ("internal disk", Path("/"))):
        try:
            lines.append(f"{label}: {shutil.disk_usage(path).free / 1e9:.0f} GB free")
        except OSError:
            lines.append(f"{label}: NOT AVAILABLE")
    return lines


def _reference_lines(chunks: List[RefChunk], states: Dict[str, Dict], now: float) -> Tuple[List[str], float]:
    lines, remaining = [], 0.0
    for chunk in chunks:
        st = states.get(chunk.tag, {})
        entries = chunk.entries()
        if chunk.done():
            state = "done"
        elif str(st.get("phase", "")).startswith("failed"):
            state = "**FAILED**"
        elif st.get("phase") == "ref":
            state = f"running {(now - st['phase_started']) / 60:.0f} min"
        else:
            state = "queued"
        remaining += (chunk.count - min(entries, chunk.count)) * EST_REFERENCE_MIN_PER_INSTANCE[chunk.size]
        lines.append(f"| {chunk.size} {chunk.start + 1}-{chunk.start + chunk.count} | {state} | {entries}/{chunk.count} |")
    return lines, remaining


def write_status(jobs: List[Job], workers: int) -> None:
    with _lock:
        states = {tag: dict(st) for tag, st in _state.items()}
    now = time.time()
    counts = {"done": 0, "running": 0, "queued": 0, "failed": 0}
    running_rows, failed_rows, result_rows = [], [], []
    chunks = [c for c in reference_chunks() if c.size in {j.size for j in jobs}]
    reference_rows, remaining = _reference_lines(chunks, states, now)
    for chunk in chunks:
        if str(states.get(chunk.tag, {}).get("phase", "")).startswith("failed"):
            tail = " / ".join(line.strip() for line in _tail(QUEUE_DIR / f"{chunk.tag}__ref.log", 3000).splitlines()[-3:])
            failed_rows.append(f"- **{chunk.tag}** (reference): `{tail[-300:]}`")
    for job in jobs:
        st = states.get(job.tag, {})
        phase = st.get("phase")
        if phase == "done" or _eval_done(job):
            counts["done"] += 1
            s = _summary(job)
            if s:
                result_rows.append(
                    f"| {job.size} | {job.axis} | {job.variant} | {job.seed} "
                    f"| {s.get('mean_kisialiou_gap_pct', 0):+.2f}% | {s.get('wins_vs_kisialiou')}/"
                    f"{s.get('ties_vs_kisialiou')}/{s.get('losses_vs_kisialiou')} "
                    f"| {s.get('mean_baseline_gap_pct', 0):+.2f}% | {s.get('wins_vs_baseline')}/{s.get('num_samples')} |"
                )
        elif phase and phase.startswith("failed"):
            counts["failed"] += 1
            log = QUEUE_DIR / f"{job.tag}__{'train' if 'train' in phase else 'eval'}.log"
            tail = " / ".join(line.strip() for line in _tail(log, 3000).splitlines()[-3:])
            failed_rows.append(f"- **{job.tag}** ({phase}): `{tail[-300:]}`")
        elif phase in ("train", "eval"):
            counts["running"] += 1
            elapsed = (now - st["phase_started"]) / 60
            if phase == "train":
                frac = _train_progress(job)
                detail = f"{frac * 100:.0f}% of {TIMESTEPS // 1000}k steps" if frac is not None else "starting"
            else:
                prog = _eval_progress(job)
                if prog:
                    gap = prog.get("rl_mean_kisialiou_gap_pct")
                    detail = (f"{prog['done']}/{prog['total']} instances, so far vs Kisialiou "
                              f"{gap:+.2f}% (W/L {prog['rl_wins_vs_kisialiou']}/{prog['rl_losses_vs_kisialiou']})"
                              if gap is not None else f"{prog['done']}/{prog['total']} instances")
                else:
                    detail = "starting"
            resumed = " (resumed)" if phase == "train" and st.get("resumed") else ""
            running_rows.append(f"| {job.tag} | {phase}{resumed} (attempt {st.get('attempt', 1)}) | {elapsed:.0f} min | {detail} |")
        else:
            counts["queued"] += 1
            if st.get("eval_waiting"):
                running_rows.append(f"| {job.tag} | waiting | | evaluation waits for the {job.size} reference |")
        remaining += _remaining_minutes(job)

    eta_min = remaining / max(workers, 1)
    lines = [
        "# NORM 2026 ablation — status",
        "",
        f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · queue running for "
        f"{(now - _queue_started) / 3600:.1f} h · **{counts['done']}/{len(jobs)} jobs done**, "
        f"{counts['running']} running, {counts['queued']} queued, {counts['failed']} failed",
        "",
        f"Rough ETA for everything: ~{eta_min / 60:.1f} h "
        f"(≈ {(datetime.now() + timedelta(minutes=eta_min)).strftime('%a %H:%M')}), "
        f"from typical phase durations; refines as jobs report progress.",
        "",
        "## Kisialiou / baseline reference (defines every search-CPU budget)",
        "",
        "| instances | state | done |",
        "|---|---|---|",
        *reference_rows,
        "",
        "## Running",
        "",
        "| job | phase | elapsed | progress |",
        "|---|---|---|---|",
        *(running_rows or ["| — | | | |"]),
        "",
        "## Finished evaluations",
        "",
        "Gap = mean % by which RL's best cost beats the reference (positive = RL better). "
        f"W/T/L = per-instance wins/ties/losses vs Kisialiou (small {DATASETS_BY_SIZE['small']} / "
        f"medium {DATASETS_BY_SIZE['medium']} instances per job).",
        "",
        "| size | axis | variant | seed | gap vs Kisialiou | W/T/L vs Kisialiou | gap vs baseline | wins vs baseline |",
        "|---|---|---|---|---|---|---|---|",
        *(result_rows or ["| — | | | | | | | |"]),
        "",
        "## Failures",
        "",
        *(failed_rows or ["none"]),
        "",
        "## Machine",
        "",
        *(f"- {line}" for line in _system_lines()),
        "",
        f"Event log: `{EVENTS_PATH.relative_to(REPO)}` · per-job logs: `{QUEUE_DIR.relative_to(REPO)}/`",
    ]
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(STATUS_PATH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="Comma-separated substrings; keep jobs matching any")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true", help="Print the latest status file")
    ap.add_argument("--prefix", default="norm26", help="Output prefix (non-default = smoke test)")
    ap.add_argument("--datasets", type=int, default=None, help="Override instances per size (smoke tests)")
    ap.add_argument("--timesteps", type=int, default=200_000)
    args = ap.parse_args()
    _configure(args.prefix, args.datasets, args.timesteps)

    if args.status:
        print(STATUS_PATH.read_text() if STATUS_PATH.exists() else "No status yet.")
        return
    jobs = build_jobs()
    if args.only:
        patterns = [x for x in args.only.split(",") if x]
        jobs = [j for j in jobs if any(x in j.tag for x in patterns)]
    tasks = build_tasks(jobs)
    if args.list:
        for kind, item in tasks:
            if kind == "ref":
                state = "done" if item.done() else f"todo ({item.entries()}/{item.count})"
            elif kind == "train":
                state = "done" if item.trained_model.exists() else ("PAUSED -> resume" if _training_pids(item) else "todo")
            else:
                state = "done" if _eval_done(item) else "todo"
            print(f"{kind:5s}  {state:16s}  {item.tag}")
        print(f"{len(tasks)} tasks: {sum(k == 'ref' for k, _ in tasks)} reference chunks, "
              f"{sum(k == 'train' for k, _ in tasks)} trainings, {sum(k == 'eval' for k, _ in tasks)} evaluations")
        return

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    _event(f"QUEUE START workers={args.workers} jobs={len(jobs)} tasks={len(tasks)}")
    stop = threading.Event()

    def status_loop() -> None:
        while not stop.is_set():
            try:
                write_status(jobs, args.workers)
            except Exception as exc:  # status must never take the queue down
                _event(f"STATUS ERROR {exc!r}")
            stop.wait(60)

    threading.Thread(target=status_loop, daemon=True).start()
    filler_thread = threading.Thread(target=filler_loop, args=(stop, args.workers), daemon=True)
    filler_thread.start()

    def _shutdown(signum, frame):  # stop fillers before exiting on SIGTERM / Ctrl-C
        stop.set()
        for filler in _fillers:
            filler.kill()
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_task, tasks))
    stop.set()
    filler_thread.join(timeout=10)
    write_status(jobs, args.workers)
    failed = [t for t, st in _state.items() if str(st.get("phase", "")).startswith("failed")]
    _event(f"QUEUE DONE failed={len(failed)}")


if __name__ == "__main__":
    main()
