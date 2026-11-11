#!/usr/bin/env python3
"""Tests for the cost model handed to the engine (added 2026-11-03).

The cost of a solution is  vessels used x weekly charter (dataset)  +  fuel price x fuel burnt. The fuel price is a parameter of the engine
(`initialize_alns(fuel_price=...)`, default = Experiment 10's 13,500 NOK/t so that code that passes none keeps its numbers); the environment of the
main line passes `rl.cost_model.fuel_price()`: 13,500 NOK/t (Experiment 10's market price).

Needs the Rust extension built from this commit (`cd rust_alns && maturin develop --release`) and a processed test instance:

    python tests/test_cost_model.py        (or: python -m unittest tests.test_cost_model)
"""

import glob
import math
import os
import sys
import unittest
from pathlib import Path

import rust_alns_py  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl import cost_model  # noqa: E402


def _instance() -> str:
    for root in filter(None, [os.environ.get("PSVPP_PROCESSED_ROOT"), str(ROOT / "data" / "processed" / "alns")]):
        hits = sorted(glob.glob(os.path.join(root, "small", "test", "*", "")))
        if hits:
            return hits[0].rstrip("/")
    raise unittest.SkipTest("no processed small test instance found")


def _init(**kwargs) -> dict:
    try:
        return rust_alns_py.RustALNSInterface().initialize_alns(_instance(), 42, **kwargs)
    except TypeError as exc:
        raise AssertionError(f"rust_alns_py is older than the fuel-price parameter, rebuild it: {exc}") from exc


class CostModelModuleTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop(cost_model.FUEL_PRICE_ENV, None)

    def tearDown(self):
        os.environ.pop(cost_model.FUEL_PRICE_ENV, None)
        if self._saved is not None:
            os.environ[cost_model.FUEL_PRICE_ENV] = self._saved

    def test_default_price_and_the_original_ratio_price(self):
        self.assertEqual(cost_model.DEFAULT_FUEL_PRICE, 13500.0)
        self.assertEqual(cost_model.fuel_price(), 13500.0)
        # the price that would keep the original fuel-to-charter ratio: 650 * 1.75M / 150k
        self.assertAlmostEqual(cost_model.ORIGINAL_RATIO_FUEL_PRICE, 7583.333333333333, places=9)

    def test_start_temperature_is_500_at_the_original_price_and_proportional(self):
        self.assertEqual(cost_model.initial_temperature(650.0), 500.0)
        self.assertAlmostEqual(cost_model.initial_temperature(13500.0), 10384.615384615385, places=9)
        self.assertAlmostEqual(cost_model.initial_temperature(cost_model.DEFAULT_FUEL_PRICE), 10384.615384615385, places=9)

    def test_environment_variable_overrides_and_is_validated(self):
        os.environ[cost_model.FUEL_PRICE_ENV] = "13500"
        self.assertEqual(cost_model.fuel_price(), 13500.0)
        for bad in ("0", "-5", "nan", "inf", "abc"):
            os.environ[cost_model.FUEL_PRICE_ENV] = bad
            with self.assertRaises(ValueError):
                cost_model.fuel_price()


class EngineFuelPriceTest(unittest.TestCase):
    def test_default_is_experiment_10s_price(self):
        self.assertEqual(rust_alns_py.fuel_price_nok_per_tonne(), 13500.0)
        a, b = _init(), _init(fuel_price=13500.0)
        self.assertEqual(a["variable_cost"], b["variable_cost"])
        self.assertEqual(a["fixed_cost"], b["fixed_cost"])

    def test_variable_cost_scales_with_the_price_and_fixed_cost_does_not(self):
        a, b = _init(fuel_price=13500.0), _init(fuel_price=6750.0)
        self.assertEqual(a["fixed_cost"], b["fixed_cost"])
        self.assertAlmostEqual(b["variable_cost"] / a["variable_cost"], 0.5, places=6)

    def test_start_temperature_follows_the_price(self):
        for price in (650.0, 7583.333333333333, 13500.0):
            self.assertAlmostEqual(_init(fuel_price=price)["temperature"] / (500.0 * price / 650.0), 1.0, places=9)
        self.assertAlmostEqual(rust_alns_py.default_initial_temperature(), 10384.615384615385, places=6)
        self.assertAlmostEqual(rust_alns_py.default_initial_temperature(650.0), 500.0, places=9)

    def test_invalid_prices_are_rejected(self):
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                _init(fuel_price=bad)


class EnvironmentCostModelTest(unittest.TestCase):
    def test_environment_passes_the_main_line_price_and_a_consistent_temperature(self):
        from rl.rl_alns_environment import ALNSEnvironment

        saved = os.environ.pop(cost_model.FUEL_PRICE_ENV, None)
        try:
            env = ALNSEnvironment(problem_instance=_instance(), max_iterations=40, seed=42, enable_operator_logging=False)
            self.assertEqual(env.fuel_price, cost_model.DEFAULT_FUEL_PRICE)
            self.assertAlmostEqual(env.temperature, 10384.615384615385, places=9)
            env.reset(seed=42)
            self.assertAlmostEqual(env.initial_cost / _init(fuel_price=cost_model.DEFAULT_FUEL_PRICE)["total_cost"], 1.0, places=9)
            env.close()

            os.environ[cost_model.FUEL_PRICE_ENV] = "13500"
            env = ALNSEnvironment(problem_instance=_instance(), max_iterations=40, seed=42, enable_operator_logging=False)
            self.assertEqual(env.fuel_price, 13500.0)
            self.assertAlmostEqual(env.temperature, 10384.615384615385, places=9)
            env.reset(seed=42)
            self.assertAlmostEqual(env.initial_cost / _init()["total_cost"], 1.0, places=9)
            env.close()
        finally:
            os.environ.pop(cost_model.FUEL_PRICE_ENV, None)
            if saved is not None:
                os.environ[cost_model.FUEL_PRICE_ENV] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
