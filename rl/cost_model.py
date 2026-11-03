"""Cost model of the experiments (NOK): the part of it that the environment hands to the Rust engine.

The cost of a solution is  (vessels used) x (weekly charter, from the dataset's vessels.csv)
                        + fuel price x (fuel burnt in the week).
Which moves pay off (cutting a vessel against sailing less) depends only on the *ratio* of the two prices, and the rewards are relative to the
initial cost, so a cost model is characterised by that ratio. The simulated-annealing start temperature is in cost units and follows the fuel price.

  original model (Experiments 0-9)    charter 150,000   fuel    650 NOK/t   T0 500      fuel / charter = 0.00433
  Experiment 10                       charter 1,750,000 fuel 13,500 NOK/t   T0 10,385   fuel / charter = 0.00771 (1.78 x the original)
  main line since 2026-11-03          charter 1,750,000 fuel  7,583 NOK/t   T0  5,833   fuel / charter = 0.00433 (the original ratio)

The main line therefore *is* the original cost model multiplied by 11.667 (charter, fuel and temperature alike). In Experiment 10 a vessel cut was worth
relatively half as much as in Experiments 8-9 compared with saving fuel, which may have changed what the learned policies prefer (FCR -> VNR).

The engine itself still defaults to Experiment 10's price when no fuel price is passed (rust_alns/src/structs/constants.rs), so code that passes none, such as the
frozen branch exp10-legacy-horizon, keeps reproducing Experiments 8-10. `PSVPP_FUEL_PRICE` overrides the price for every environment of the process
(13500 reproduces Experiment 10's pricing on the main line).
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
# the original fuel-to-charter ratio at the current charter: 650 * 1,750,000 / 150,000
DEFAULT_FUEL_PRICE = ORIGINAL_FUEL_PRICE * CHARTER / ORIGINAL_CHARTER


def fuel_price() -> float:
    """Fuel price in NOK per tonne: PSVPP_FUEL_PRICE if set, else the original fuel-to-charter ratio at the current charter."""
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
