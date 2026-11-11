// constants.rs

// Define constants for global values

pub const DAYS_IN_PERIOD: u32 = 7; // Number of days in the period
pub const HOURS_IN_DAY: u32 = 24; // Number of hours in a day
pub const HOURS_IN_PERIOD: u32 = DAYS_IN_PERIOD * HOURS_IN_DAY; // Total hours in the period
pub const REL_DEPARTURE_TIME: u32 = 16; // Default departure time (in hours) relative to 00:00
pub const MAX_INST_PER_VOYAGE: u32 = 5; // Maximum number of installations per voyage
pub const MAX_ATTEMPTS_TO_INIT: u32 = 10; // Maximum number of attempts to initialize a solution

// ---- Cost model -------------------------------------------------------------------------
// All costs are in NOK. The vessel charter cost is per vessel and week and lives in the
// dataset (`cost` column of vessels.csv, set in py_alns/resource/generation_config.yaml);
// the fuel price is global and lives here.

/// Fuel price, NOK per tonne, that the engine uses when the caller gives none: the price of Experiment 10 (marine gas
/// oil, MGO 1,425 USD/t, Ship & Bunker, Rotterdam, 21 Oct 2026, at 9.5 NOK/USD, rounded). It is a historical default,
/// kept so that code that passes no fuel price (Experiments 8-10, the frozen branch exp10-legacy-horizon) keeps its
/// numbers with any build. The environment of the main line passes its own price (rl/cost_model.py: 13,500 again) to `initialize_alns(fuel_price=...)`.
pub const FUEL_PRICE_NOK_PER_TONNE: f64 = 13_500.0;

/// The simulated-annealing start temperature is expressed in cost units, so it has to follow the
/// cost scale. The original cost model (fuel price 650 per tonne, charter 150,000 per week) was
/// calibrated with a start temperature of 500; scaling it with the fuel price keeps the
/// acceptance probability of fuel-sized cost increases (the typical worsening move) unchanged.
pub const CALIBRATION_FUEL_PRICE: f64 = 650.0;
pub const CALIBRATION_INITIAL_TEMPERATURE: f64 = 500.0;

/// SA start temperature in cost units for a given fuel price (about 0.77 tonnes of fuel).
pub fn initial_temperature_for(fuel_price: f64) -> f64 {
    CALIBRATION_INITIAL_TEMPERATURE * fuel_price / CALIBRATION_FUEL_PRICE
}

/// Default SA start temperature in cost units, at the default fuel price.
pub fn default_initial_temperature() -> f64 {
    initial_temperature_for(FUEL_PRICE_NOK_PER_TONNE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_temperature_follows_the_fuel_price() {
        assert!((initial_temperature_for(CALIBRATION_FUEL_PRICE) - CALIBRATION_INITIAL_TEMPERATURE).abs() < 1e-9);
        assert!((default_initial_temperature() - 10_384.615384615385).abs() < 1e-6);
        // the original fuel-to-charter ratio at a charter of 1,750,000 (150,000 originally): 650 * 1.75M / 150k
        let matched = CALIBRATION_FUEL_PRICE * 1_750_000.0 / 150_000.0;
        assert!((matched - 7_583.333333333333).abs() < 1e-9);
        assert!((initial_temperature_for(matched) - 5_833.333333333333).abs() < 1e-9);
    }
}
