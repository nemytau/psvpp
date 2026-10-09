"""NORM 2026 state / action / reward design ablation (small + medium).

One-factor-at-a-time around the Experiment 8 reference configuration -- op_triple_v2 /
features_improvement_v3 / delta_cost_v9 / ent_coef 0.01 / ph300 / 200k PPO steps -- plus
non-learned control policies. Each job trains a PPO model (train-only run) and then
evaluates it (eval-only run) on the first 50 test instances of its size, eval seed 42,
wall-time matched to the cached Kisialiou results (per-size cache slices).

Layout per job tag "<axis>__<variant>__<size>__s<seed>":
    output/norm26__<tag>__train/model/ppo_rl_improvement_mvp.zip   (trained model)
    output/norm26__<tag>/{results.csv,summary.json,progress.json}   (evaluation)
    logs/norm2026_queue/<tag>__{train,eval}.log                      (stdout of each phase)

Resumable at every level: a finished training is never redone, a finished evaluation is
skipped, and an interrupted evaluation resumes after its last finished dataset (--resume).
A failed phase is retried once. Progress is written every minute to NORM2026_STATUS.md.

    nohup caffeinate -is python scripts/run_norm2026_ablation.py --workers 4 > logs/norm2026_queue/queue.log 2>&1 &
    python scripts/run_norm2026_ablation.py --status
    python scripts/run_norm2026_ablation.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
PY = "/Users/timothy/opt/miniconda3/envs/py312/bin/python"
PREFIX = "norm26"
QUEUE_DIR = REPO / "logs" / "norm2026_queue"
STATUS_PATH = REPO / "NORM2026_STATUS.md"
EVENTS_PATH = QUEUE_DIR / "events.log"

DATASETS = 50
TIMESTEPS = 200_000


def _configure(prefix: str, datasets: int, timesteps: int) -> None:
    """Override output prefix / dataset count / training steps (used for smoke tests)."""
    global PREFIX, QUEUE_DIR, STATUS_PATH, EVENTS_PATH, DATASETS, TIMESTEPS
    PREFIX, DATASETS, TIMESTEPS = prefix, datasets, timesteps
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

# Rough per-phase durations (minutes) for ETA before a phase has reported progress:
# training from past runs (46-110 min medium, 7-18 min small); evaluation = sum of cached
# Kisialiou wall times over the 50 test instances (small 9 min, medium 106 min) + overhead.
EST_TRAIN_MIN = {"small": 18.0, "medium": 80.0}
EST_EVAL_MIN = {"small": 11.0, "medium": 112.0}


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
            "--datasets-per-size", str(DATASETS),
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
            "--baseline-kisialiou-cache", f"runs/baseline_kisialiou_cache/cache_{self.size}_seed42.json",
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
        seeds = (42,) if size == "medium" else (42, 43, 44)
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
        jobs += [Job("ref", "op_triple_v2", size, seed) for seed in (43, 44)]
        jobs.append(Job("ref", "op_triple_v2", size, 42, model_path=str(REPO / EXP8_MODELS[size])))
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


def _run_phase(job: Job, phase: str, cmd: List[str], run_id: str) -> int:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(REPO), RL_RUN_ID=run_id)
    log_path = QUEUE_DIR / f"{job.tag}__{phase}.log"
    rc = 1
    for attempt in (1, 2):
        _set(job.tag, phase=phase, phase_started=time.time(), attempt=attempt)
        _event(f"START {phase:5s} {job.tag} attempt={attempt}")
        with log_path.open("a") as log:
            log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} attempt {attempt}: {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        ok = rc == 0 and (job.trained_model.exists() if phase == "train" else _eval_done(job))
        _event(f"{'END  ' if ok else 'FAIL '} {phase:5s} {job.tag} rc={rc} "
               f"{(time.time() - _state[job.tag]['phase_started']) / 60:.1f}min")
        if ok:
            return 0
    return rc or 1


def run_job(job: Job) -> None:
    if _eval_done(job):
        _set(job.tag, phase="done", finished=time.time())
        return
    _set(job.tag, started=time.time())
    if job.needs_training and not job.trained_model.exists():
        if _run_phase(job, "train", job.train_command(), f"{PREFIX}__{job.tag}__train") != 0:
            _set(job.tag, phase="failed (train)", finished=time.time())
            return
    if _run_phase(job, "eval", job.eval_command(), f"{PREFIX}__{job.tag}") != 0:
        _set(job.tag, phase="failed (eval)", finished=time.time())
        return
    _set(job.tag, phase="done", finished=time.time())


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
    if job.needs_training and not job.trained_model.exists():
        frac = _train_progress(job) if phase == "train" else None
        if frac and frac > 0.02:
            elapsed = (time.time() - st["phase_started"]) / 60
            train_left = elapsed * (1 - frac) / frac
        else:
            train_left = EST_TRAIN_MIN[job.size]
    eval_left = EST_EVAL_MIN[job.size]
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
    for label, path in (("external SSD (output/logs)", REPO / "output"), ("internal disk", Path("/"))):
        try:
            lines.append(f"{label}: {shutil.disk_usage(path).free / 1e9:.0f} GB free")
        except OSError:
            lines.append(f"{label}: NOT AVAILABLE")
    return lines


def write_status(jobs: List[Job], workers: int) -> None:
    with _lock:
        states = {tag: dict(st) for tag, st in _state.items()}
    now = time.time()
    counts = {"done": 0, "running": 0, "queued": 0, "failed": 0}
    running_rows, failed_rows, result_rows = [], [], []
    remaining = 0.0
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
            running_rows.append(f"| {job.tag} | {phase} (attempt {st.get('attempt', 1)}) | {elapsed:.0f} min | {detail} |")
        else:
            counts["queued"] += 1
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
        "## Running",
        "",
        "| job | phase | elapsed | progress |",
        "|---|---|---|---|",
        *(running_rows or ["| — | | | |"]),
        "",
        "## Finished evaluations",
        "",
        "Gap = mean % by which RL's best cost beats the reference (positive = RL better). "
        f"W/T/L = per-instance wins/ties/losses vs Kisialiou ({DATASETS} instances per job).",
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
    ap.add_argument("--datasets", type=int, default=50)
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
    if args.list:
        for j in jobs:
            print(f"{'done' if _eval_done(j) else 'todo'}  {'train+eval' if j.needs_training else 'eval only':10s}  {j.tag}")
        print(f"{len(jobs)} jobs, {sum(j.needs_training for j in jobs)} trainings")
        return

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    _event(f"QUEUE START workers={args.workers} jobs={len(jobs)}")
    stop = threading.Event()

    def status_loop() -> None:
        while not stop.is_set():
            try:
                write_status(jobs, args.workers)
            except Exception as exc:  # status must never take the queue down
                _event(f"STATUS ERROR {exc!r}")
            stop.wait(60)

    threading.Thread(target=status_loop, daemon=True).start()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_job, jobs))
    stop.set()
    write_status(jobs, args.workers)
    failed = [t for t, st in _state.items() if str(st.get("phase", "")).startswith("failed")]
    _event(f"QUEUE DONE failed={len(failed)}")


if __name__ == "__main__":
    main()
