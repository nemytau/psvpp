#!/usr/bin/env python3
"""Regression test for the engine's iteration horizon (fixed 2026-11-03).

The horizon rho is the length of one restart. The Rust engine derives its cooling rate (mu = 1 - theta / rho) and the
start of its aggressive phase (aggressive_search_factor * rho: non-improving iterations return to the restart's best)
from it. `initialize_alns` used to hard-code 1000, so 300-iteration restarts ended at 12% of the start temperature
and never reached the aggressive phase (Experiments 8-10). Now the caller passes `max_iterations`, and
`ALNSEnvironment` passes its restart length.

Needs the Rust extension built from the fix (`cd rust_alns && maturin develop --release`) and a processed test
instance under data/processed/alns (or $PSVPP_PROCESSED_ROOT). Runs in a few seconds:

    python tests/test_engine_horizon.py        (or: python -m unittest tests.test_engine_horizon)
"""

import glob
import os
import sys
import unittest
from pathlib import Path

import rust_alns_py  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

THETA, T0 = 7.0, 1000.0


def _instance() -> str:
    for root in filter(None, [os.environ.get("PSVPP_PROCESSED_ROOT"), str(ROOT / "data" / "processed" / "alns")]):
        hits = sorted(glob.glob(os.path.join(root, "small", "test", "*", "")))
        if hits:
            return hits[0].rstrip("/")
    raise unittest.SkipTest("no processed small test instance found")


def _interface(**kwargs):
    iface = rust_alns_py.RustALNSInterface()
    try:
        iface.initialize_alns(_instance(), 42, **kwargs)
    except TypeError as exc:
        raise AssertionError(f"rust_alns_py is older than the horizon fix, rebuild it: {exc}") from exc
    return iface


def _temperature_after(iterations: int, **kwargs) -> float:
    iface = _interface(temperature=T0, theta=THETA, **kwargs)
    result = None
    for i in range(iterations):
        result = iface.execute_iteration(i, mode="random")
    return float(result["temperature"])


def _tail_gaps(**kwargs):
    """current cost - best cost over the last iterations of a 40-iteration restart that accepts every valid candidate."""
    iface = _interface(temperature=1e12, aggressive_search_factor=0.5, **kwargs)
    gaps = []
    for i in range(40):
        result = iface.execute_iteration(i, mode="random")
        if i >= 25:
            gaps.append(float(result["current_cost"]) - float(result["best_cost"]))
    return gaps


class EngineHorizonTest(unittest.TestCase):
    def test_cooling_follows_the_horizon(self):
        horizon = 40
        expected = T0 * (1.0 - THETA / horizon) ** horizon
        self.assertAlmostEqual(_temperature_after(horizon, max_iterations=horizon) / expected, 1.0, places=9)

    def test_default_horizon_is_the_historical_1000(self):
        expected = T0 * (1.0 - THETA / 1000) ** 40
        self.assertAlmostEqual(_temperature_after(40) / expected, 1.0, places=9)

    def test_aggressive_phase_starts_at_the_horizon_fraction(self):
        # horizon 40, factor 0.5: from iteration 20 every iteration ends with current == best
        self.assertTrue(all(abs(g) < 1e-9 for g in _tail_gaps(max_iterations=40)))
        # horizon 1000 (the old behaviour): the phase starts at 500, so a random walk stays away from its best
        self.assertTrue(any(abs(g) > 1e-9 for g in _tail_gaps()))

    def test_zero_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            _interface(max_iterations=0)

    def test_environment_hands_its_restart_length_to_the_engine(self):
        from rl.rl_alns_environment import ENGINE_HORIZON_ENV, ALNSEnvironment

        env = ALNSEnvironment(problem_instance=_instance(), max_iterations=40, seed=42)
        saved = os.environ.pop(ENGINE_HORIZON_ENV, None)
        try:
            self.assertEqual(env._engine_horizon(), 40)
            os.environ[ENGINE_HORIZON_ENV] = "1000"
            self.assertEqual(env._engine_horizon(), 1000)
        finally:
            os.environ.pop(ENGINE_HORIZON_ENV, None)
            if saved is not None:
                os.environ[ENGINE_HORIZON_ENV] = saved
        env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
