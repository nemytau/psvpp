// constants.rs

// Define constants for global values

pub const DAYS_IN_PERIOD: u32 = 7;        // Number of days in the period
pub const HOURS_IN_DAY: u32 = 24;         // Number of hours in a day
pub const HOURS_IN_PERIOD: u32 = DAYS_IN_PERIOD * HOURS_IN_DAY;  // Total hours in the period
pub const REL_DEPARTURE_TIME: u32 = 16;       // Default departure time (in hours) relative to 00:00
pub const MAX_INST_PER_VOYAGE: u32 = 5;        // Maximum number of installations per voyage
pub const MAX_ATTEMPTS_TO_INIT: u32 = 10;      // Maximum number of attempts to initialize a solution