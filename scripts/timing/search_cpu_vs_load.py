"""Search CPU time of a fixed, deterministic amount of work, to see how it depends on machine load.

Runs Kisialiou's ALNS for two 300-iteration restarts on the first small test instance with a
fixed seed (identical work every time, same best cost) and prints the engine's search CPU time
and the wall time. Run it alone, as N simultaneous copies, or next to background CPU hogs, and
compare: on the M1 Air (2026-10-09) the same work took 2.37 s alone, 3.0-3.3 s with 3 copies,
3.8-4.0 s with 4 copies, and 3.3-3.5 s next to 7 `yes > /dev/null` hogs; "--qos interactive"
(QOS_CLASS_USER_INTERACTIVE) did not help. This is why every time-budgeted task of Experiment 9
runs under the same constant load (docs/EXPERIMENTS.md).

    PYTHONPATH=. python scripts/timing/search_cpu_vs_load.py                         # alone
    for i in 1 2 3 4; do PYTHONPATH=. python scripts/timing/search_cpu_vs_load.py & done; wait
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qos", choices=("default", "interactive"), default="default")
    args = ap.parse_args()
    if args.qos == "interactive":
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        assert libc.pthread_set_qos_class_self_np(ctypes.c_uint(0x21), ctypes.c_int(0)) == 0

    from rl.dataset_manager import GeneratedDatasetManager
    from rl.rl_alns_environment import ALNSEnvironment

    path = GeneratedDatasetManager().prepare_size("small")["test"][0]
    results = []
    for _ in range(2):
        env = ALNSEnvironment(problem_instance=path, max_iterations=300, seed=42, action_module="adaptive_weighted_v1",
                              state_module="features_v3", reward_module="delta_cost_v9", algorithm_mode="kisialiou",
                              size_bucket="small")
        env.reset(seed=42)
        start = time.perf_counter()
        for _ in range(300):
            env.step(0)
        results.append((env.search_cpu_seconds, time.perf_counter() - start, env.get_episode_statistics()["best_cost"]))
        env.close()
    print(f"qos={args.qos}: " + "  ".join(f"search {s:.2f}s wall {w:.2f}s best {b:.2f}" for s, w, b in results))


if __name__ == "__main__":
    main()
