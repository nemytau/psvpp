#!/usr/bin/env bash
# Old-pool reference ablation of the Experiment 10 -- TODO, prepared 2026-11-03, NOT RUN YET.
# Plan, reasoning and how to read the result: docs/PAPER1_ACTION_SPACE_PLAN.md, section 17.
#
# What it runs: Kisialiou ALNS (5 restarts x 300 iterations) and the plain adaptive ALNS (same CPU time, no improvement
# operators) with the ORIGINAL repair pool (adaptive_weighted_v1: greedy rank 1, k-regret-2, k-regret-3; no first-fit
# insertion) on the 150 test instances of the Experiment 10 (data/rerun26: small, medium, large x 50), seed 42, with
# the same engine, cost model and 4-process load as Experiment 10's references (the engine runs with horizon 1000 via
# PSVPP_ENGINE_HORIZON, and the fuel price is 13,500 NOK/t via PSVPP_FUEL_PRICE, as in Experiment 10; the main line's defaults since 2026-11-03 are the
# restart length and the original fuel-to-charter ratio). Nothing else: no training, no RL runs.
# The Rust extension must be built from the horizon fix (`cd rust_alns && maturin develop --release`); the smoke test checks that
# it reproduces two Kisialiou results of the pre-fix build exactly.
#
# The run measures CPU time, so the PC must be otherwise IDLE and awake (on AC power) for the whole run: about 12 hours
# (range 9-14 h) with 4 busy processes. Close browsers, IDEs and other experiments first. Start it from a normal Terminal;
# it keeps running if the Terminal is closed, but not across a reboot (`start` again resumes, finished instances are kept).
#
#   scripts/run_oldpool_ablation.sh check     preflight only (SSD, disk, power, other jobs, idle machine, engine build, data hashes)
#   scripts/run_oldpool_ablation.sh smoke     2-instance end-to-end test of the whole pipeline (~3 min, own prefix)
#   scripts/run_oldpool_ablation.sh start     check + smoke + detached full run   <- the normal way
#   scripts/run_oldpool_ablation.sh status    status file, last events, running processes
#   scripts/run_oldpool_ablation.sh analyze   per-dataset comparison with Experiment 10 and the RL models
#   scripts/run_oldpool_ablation.sh stop      stop this run only
#
# Environment: FORCE=1 skips the soft checks (power, idle machine, engine build); SKIP_SMOKE=1 skips the smoke test in `start`.

set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
PY=/Users/timothy/opt/miniconda3/envs/py312/bin/python
SSD=/Volumes/TimSSD
RUNNER=scripts/run_exp8_rerun.py
QUEUE_DIR=logs/rerun26oldpool_queue
STATUS=RERUN26OLDPOOL_STATUS.md
WORKERS=4
MIN_FREE_GB=120
MIN_IDLE_PCT=90

cd "$REPO" || exit 1

say()  { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
soft() { # a soft check failed: stop unless FORCE=1
  if [ "${FORCE:-0}" = "1" ]; then warn "$* (ignored because FORCE=1)"; else fail "$* (set FORCE=1 to run anyway)"; fi
}
full_run_pids() { pgrep -f "run_exp8_rerun.py --oldpool-ablation --workers" || true; }

check_storage() {
  [ -d "$SSD/psvpp" ] || fail "the external SSD is not mounted ($SSD/psvpp is missing); data/, logs/, output/ and runs/ are symlinks into it"
  for d in data/rerun26/processed/alns output logs runs; do
    [ -d "$d/" ] || fail "$d is not reachable (dangling symlink?)"
  done
  local free_gb
  free_gb=$(df -k "$SSD" | tail -1 | awk '{printf "%d", $4/1024/1024}')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || fail "only ${free_gb} GB free on the SSD (need >= ${MIN_FREE_GB}; the run itself needs ~6 GB)"
  say "ssd: mounted, ${free_gb} GB free"
}

check_data() {
  # the runner hashes every instance csv against data/rerun26/DATA_MANIFEST.json (written when Experiment 10 started), ~10 s
  "$PY" "$RUNNER" --oldpool-ablation --prepare-data-only || fail "the test instances differ from Experiment 10's (see message above)"
}

check_machine() {
  if pmset -g batt | head -1 | grep -q "AC Power"; then say "power: AC"; else soft "not on AC power (the run takes ~12 h)"; fi

  local other
  other=$(pgrep -fl "run_experiment_rl_improvement_mvp.py|run_exp8_rerun.py|train_alns_rl.py|maturin|cargo " | grep -v oldpool || true)
  if [ -n "$other" ]; then
    say "$other"
    fail "other project jobs are running (listed above); the CPU-time protocol needs an idle machine"
  fi
  say "no other project job is running"

  local idle
  idle=$(top -l 2 -s 1 -n 0 | grep "CPU usage" | tail -1 | sed -E 's/.* ([0-9.]+)% idle.*/\1/')
  if [ -z "$idle" ]; then
    warn "could not read the CPU load from top"
  elif awk -v v="$idle" -v m="$MIN_IDLE_PCT" 'BEGIN { exit !(v + 0 >= m + 0) }'; then
    say "cpu: ${idle}% idle"
  else
    say "busiest processes right now:"
    top -l 2 -s 1 -n 6 -o cpu -stats pid,command,cpu | awk '/^PID/ { blk = "" } { blk = blk $0 "\n" } END { printf "%s", blk }'
    soft "the machine is only ${idle}% idle (need >= ${MIN_IDLE_PCT}%): CPU-time budgets are only comparable on an idle machine"
  fi
}

check_engine() {
  # initialize_alns must take the restart length and the fuel price (changes of 2026-11-03); the runs then use PSVPP_ENGINE_HORIZON=1000 and
  # PSVPP_FUEL_PRICE=13500 like Experiment 10 (the main line's own defaults are the restart length and the original fuel-to-charter ratio).
  # The smoke test verifies that this reproduces the pre-fix build exactly.
  local d
  d=$(ls -d data/rerun26/processed/alns/small/test/*/ | head -1)
  "$PY" -c 'import sys, rust_alns_py as r; r.RustALNSInterface().initialize_alns(sys.argv[1], 42, max_iterations=300, fuel_price=13500.0)' "${d%/}" > /dev/null 2>&1 \
    || fail "rust_alns_py is older than the horizon / fuel-price parameters (initialize_alns takes no max_iterations or fuel_price); rebuild it: cd rust_alns && maturin develop --release"
  say "engine: takes the horizon and the fuel price; this run sets PSVPP_ENGINE_HORIZON=1000 and PSVPP_FUEL_PRICE=13500 like Experiment 10"
}

check() {
  say "== preflight $(date '+%F %T')"
  check_storage
  check_machine
  check_engine
  check_data
  say "git: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD); uncommitted in scripts/rl/rust_alns/py_alns: $(git status --porcelain scripts rl rust_alns py_alns | awk '{print $2}' | tr '\n' ' ')"
  say "== preflight ok"
}

smoke() {
  say "== smoke test: 2 small instances through the whole pipeline (own prefix rerun26oldpoolsmoke, ~3 min)"
  check_storage
  # an earlier smoke run must not be reused: the queue treats finished tasks as done and would test nothing
  rm -rf RERUN26OLDPOOLSMOKE_STATUS.md logs/rerun26oldpoolsmoke_queue output/rerun26oldpoolsmoke__* logs/rerun26oldpoolsmoke__* runs/baseline_kisialiou_cache/rerun26_oldpool_smoke
  mkdir -p logs/rerun26oldpoolsmoke_queue
  "$PY" "$RUNNER" --oldpool-ablation --smoke --workers "$WORKERS" > logs/rerun26oldpoolsmoke_queue/queue.log 2>&1 \
    || { tail -n 25 logs/rerun26oldpoolsmoke_queue/queue.log; fail "the smoke run crashed"; }
  grep -q "QUEUE END done=2/2" logs/rerun26oldpoolsmoke_queue/events.log \
    || { cat logs/rerun26oldpoolsmoke_queue/events.log; fail "the smoke queue did not finish its 2 tasks"; }
  local out=output/rerun26oldpoolsmoke__reference__small f
  for f in results.csv restart_results.csv results.json summary.json run_manifest.json; do
    [ -s "$out/$f" ] || fail "smoke: $out/$f is missing"
  done
  for f in c01 c02; do
    [ -s "${out}__$f/run_manifest.json" ] || fail "smoke: the manifest of chunk $f is missing"
  done
  [ -s output/rerun26oldpoolsmoke__QUEUE_MANIFEST.json ] || fail "smoke: the queue manifest is missing"
  grep -q "adaptive_weighted_v1" "${out}__c01/run_command.txt" || fail "smoke: the chunk command does not use adaptive_weighted_v1"
  grep -q "adaptive_weighted_v1" "$out/results.csv" || fail "smoke: results.csv does not record the old pool"
  # Kisialiou's search is deterministic given the seed: these two results come from the first smoke run of 2026-11-03 on the pre-fix
  # build (horizon 1000). A different value means the engine horizon or the engine itself differs from Experiment 10's.
  "$PY" - "$out/results.csv" <<'PYEOF' || fail "smoke: Kisialiou does not reproduce the pre-fix build's results (engine horizon or engine changed)"
import csv, sys
EXPECT = {"small_test_tw000_df1_01": 4022340.6655561617, "small_test_tw000_df1_02": 4125536.54962781}
got = {r["dataset"]: float(r["best_cost"]) for r in csv.DictReader(open(sys.argv[1])) if r["mode"] == "kisialiou"}
bad = {d: (got.get(d), v) for d, v in EXPECT.items() if got.get(d) is None or abs(got[d] - v) > 1e-6 * v}
if bad:
    print("got / expected:", bad)
    sys.exit(1)
PYEOF
  say "smoke: ok -- dataset, mode, initial cost, best cost, search CPU s, restarts, vessels:"
  cut -d, -f1,4,6,7,9,13,15 "$out/results.csv"
}

start() {
  if [ -n "$(full_run_pids)" ]; then say "the full run is already running -- see: $SELF status"; exit 0; fi
  check
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then smoke; fi
  mkdir -p "$QUEUE_DIR"
  say "== launching the full run (detached)"
  nohup caffeinate -is "$SELF" _worker > "$QUEUE_DIR/queue.log" 2>&1 &
  disown 2> /dev/null || true
  sleep 20
  if [ -z "$(full_run_pids)" ]; then tail -n 25 "$QUEUE_DIR/queue.log"; fail "the queue did not start"; fi
  cat <<EOF

Started $(date '+%F %T'): old-pool references (Kisialiou 5 restarts + adaptive ALNS for the same CPU time) on 150 instances,
19 chunks, $WORKERS workers, order small -> medium -> large. Expected finish in ~12 h (9-14 h): about $(date -v+12H '+%a %d %b %H:%M').
Keep the PC awake, on AC power and otherwise idle until then.

  status file (refreshed every minute) : $REPO/$STATUS
  events / per-task logs               : $REPO/$QUEUE_DIR/
  instance results (after each one)    : $REPO/runs/baseline_kisialiou_cache/rerun26_oldpool_v1/<size>_partNN.json
  per-size result files, manifests     : $REPO/output/rerun26oldpool__reference__<size>/  and  ...__QUEUE_MANIFEST.json
  analysis (written at the end)        : $REPO/output/rerun26oldpool__ANALYSIS.md

  $SELF status      look at progress
  $SELF analyze     comparison so far (also runs by itself at the end)
  $SELF stop        stop this run; \`start\` resumes
EOF
}

_worker() { # runs detached under caffeinate: the queue with a retry on crashes, then the analysis
  cd "$REPO" || exit 1
  local i=1 last
  while [ "$i" -le 5 ]; do
    "$PY" "$RUNNER" --oldpool-ablation --workers "$WORKERS" && break
    echo "$(date '+%F %T') the queue exited with an error (attempt $i of 5); retrying in 120 s"
    i=$((i + 1))
    sleep 120
  done
  last=$(grep "QUEUE END" "$QUEUE_DIR/events.log" 2> /dev/null | tail -n 1)
  echo "$(date '+%F %T') $last"
  if "$PY" scripts/analyze_oldpool_ablation.py > "$QUEUE_DIR/analysis.log" 2>&1; then
    echo "$(date '+%F %T') analysis written: output/rerun26oldpool__ANALYSIS.md"
  else
    echo "$(date '+%F %T') the analysis failed, see $QUEUE_DIR/analysis.log"
  fi
  case "$last" in
    *"done=19/19"*) echo "COMPLETE" ;;
    *) echo "INCOMPLETE: some chunks failed or did not run; \`$SELF start\` resumes them (finished chunks are kept)" ;;
  esac
}

status() {
  if [ -f "$STATUS" ]; then cat "$STATUS"; else say "no status file yet ($STATUS)"; fi
  say ""
  say "-- last events ($QUEUE_DIR/events.log):"
  tail -n 12 "$QUEUE_DIR/events.log" 2> /dev/null || say "(none)"
  say ""
  say "-- processes:"
  pgrep -fl "run_exp8_rerun.py --oldpool-ablation|rerun26_oldpool_v1" | cut -c1-160 || say "none running"
}

stop() {
  pkill -f "run_oldpool_ablation.sh _worker" 2> /dev/null || true # first: so that no retry loop restarts the queue
  pkill -f "run_exp8_rerun.py --oldpool-ablation" 2> /dev/null || true
  pkill -f "rerun26_oldpool_v1" 2> /dev/null || true # the chunk processes (their command line names the cache folder)
  sleep 2
  say "stopped. Finished instances stay in runs/baseline_kisialiou_cache/rerun26_oldpool_v1; \`$SELF start\` resumes."
}

case "${1:-}" in
  check) check ;;
  smoke) check_data; smoke ;;
  start) start ;;
  status) status ;;
  analyze) shift; "$PY" scripts/analyze_oldpool_ablation.py "$@" ;;
  stop) stop ;;
  _worker) _worker ;;
  *) sed -n '2,22p' "$SELF" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
