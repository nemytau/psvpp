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

/// Fuel price, NOK per tonne (marine gas oil). Basis: MGO 1,425 USD/t (Ship & Bunker,
/// Rotterdam, 21 Oct 2026; 2-week range 1,333-1,437) at 9.5 NOK/USD, rounded.
pub const FUEL_PRICE_NOK_PER_TONNE: f64 = 13_500.0;

/// The simulated-annealing start temperature is expressed in cost units, so it has to follow the
/// cost scale. The original cost model (fuel price 650 per tonne, charter 150,000 per week) was
/// calibrated with a start temperature of 500; scaling it with the fuel price keeps the
/// acceptance probability of fuel-sized cost increases (the typical worsening move) unchanged.
pub const CALIBRATION_FUEL_PRICE: f64 = 650.0;
pub const CALIBRATION_INITIAL_TEMPERATURE: f64 = 500.0;

/// Default SA start temperature in cost units (about 0.77 tonnes of fuel).
pub fn default_initial_temperature() -> f64 {
    CALIBRATION_INITIAL_TEMPERATURE * FUEL_PRICE_NOK_PER_TONNE / CALIBRATION_FUEL_PRICE
}
