"""Cost model of the experiments (NOK): the part of it that the environment hands to the Rust engine.

The cost of a solution is  (vessels used) x (weekly charter, from the dataset's vessels.csv)
                        + fuel price x (fuel burnt in the week).
Which moves pay off (cutting a vessel against sailing less) depends only on the *ratio* of the two prices, and the rewards are relative to the
initial cost, so a cost model is characterised by that ratio. The simulated-annealing start temperature is in cost units and follows the fuel price.

  original model (Experiments 0-9)    charter 150,000   fuel    650 NOK/t   T0    500    fuel / charter = 0.00433   fuel ~ 8-12% of the cost of best solutions
  Experiment 10 and main line         charter 1,750,000 fuel 13,500 NOK/t   T0 10,385    fuel / charter = 0.00771   fuel ~ 16-18% (Exp 10), ~18% with the new consumption below

The main-line price is Experiment 10's again, 13,500 NOK/t (marine gas oil, MGO 1,425 USD/t at 9.5 NOK/USD, Ship & Bunker, Rotterdam, 21 Oct 2026), which is close to the
market. For a few days it was changed to 7,583 (the original fuel-to-charter ratio, 2026-11-03) and for a day to 22,500 (fuel a quarter of the cost, 2026-11-11); both were dropped
because the market price is realistic. What changes for new datasets is the consumption (py_alns/resource/generation_config.yaml): sailing 0.575 and waiting 0.22 t/h (mean of drifting and DP, port idle left out) instead
of 0.43 / 0.26, the simple means of a burn-rate table for offshore supply vessels, which lifts the fuel share of Experiment 10's solutions from 16-18% to about 18-20%
(sailing is 62-70% of the fuel burnt in the initial solutions; estimate, see docs/EXPERIMENTS.md).

The engine itself still defaults to Experiment 10's price when no fuel price is passed (rust_alns/src/structs/constants.rs), so code that passes none, such as the
frozen branch exp10-legacy-horizon, keeps reproducing Experiments 8-10. `PSVPP_FUEL_PRICE` overrides the price for every environment of the process
(7583.33 gives the original fuel-to-charter ratio).
"""

from __future__ import annotations

import math
import os
from typing import Dict

ORIGINAL_CHARTER = 150_000.0
ORIGINAL_FUEL_PRICE = 650.0
ORIGINAL_START_TEMPERATURE = 500.0
CHARTER = 1_750_000.0  # weekly charter of the current datasets (py_alns/resource/generation_config.yaml)
FUEL_PRICE_ENV = "PSVPP_FUEL_PRICE"
# Experiment 10's market price (see the module docstring); 7,583.33 would keep the original fuel-to-charter ratio
DEFAULT_FUEL_PRICE = 13_500.0
ORIGINAL_RATIO_FUEL_PRICE = ORIGINAL_FUEL_PRICE * CHARTER / ORIGINAL_CHARTER


def fuel_price() -> float:
    """Fuel price in NOK per tonne: PSVPP_FUEL_PRICE if set, else DEFAULT_FUEL_PRICE (13,500 NOK/t)."""
    override = os.environ.get(FUEL_PRICE_ENV)
    if not override:
        return DEFAULT_FUEL_PRICE
    value = float(override)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{FUEL_PRICE_ENV} must be a positive number of NOK per tonne (got {override!r})")
    return value


def initial_temperature(price: float) -> float:
    """SA start temperature in cost units for a fuel price (500 at 650 NOK/t, proportional to the price)."""
    return ORIGINAL_START_TEMPERATURE * price / ORIGINAL_FUEL_PRICE


def describe(price: float | None = None) -> Dict[str, float]:
    """The effective cost-model parameters, for run manifests."""
    price = fuel_price() if price is None else float(price)
    return {
        "fuel_price_nok_per_tonne": price,
        "sa_initial_temperature": initial_temperature(price),
        "fuel_to_charter_ratio": price / CHARTER,
        "ratio_relative_to_original_model": (price / CHARTER) / (ORIGINAL_FUEL_PRICE / ORIGINAL_CHARTER),
    }
