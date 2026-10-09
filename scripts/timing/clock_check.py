"""Check which CPU clock is trustworthy in this Python (e.g. x86_64 under Rosetta 2).

Busy-loops in pure Python and prints thread CPU time, process CPU time and wall time; on an
idle machine all three should agree. Under Rosetta 2 on Apple Silicon (2026-10-09) the thread
clock (CLOCK_THREAD_CPUTIME_ID) read ~41.7x too small -- it returns 24 MHz Mach ticks instead of
nanoseconds -- while the process clock was correct. Hence the engine measures search CPU time
with CLOCK_PROCESS_CPUTIME_ID (rust_alns/src/alns/engine.rs, process_cpu_time_ns).

    python scripts/timing/clock_check.py
"""

import platform
import time

c0, p0, w0 = time.thread_time_ns(), time.process_time_ns(), time.perf_counter_ns()
x = 0
for i in range(3_000_000):
    x += i
thread_ms = (time.thread_time_ns() - c0) / 1e6
process_ms = (time.process_time_ns() - p0) / 1e6
wall_ms = (time.perf_counter_ns() - w0) / 1e6
print(f"python {platform.machine()}: thread CPU {thread_ms:.1f} ms | process CPU {process_ms:.1f} ms | wall {wall_ms:.1f} ms")
print(f"thread clock / wall = {thread_ms / wall_ms:.3f}  (1.0 expected; ~0.024 means ticks, not ns)")
