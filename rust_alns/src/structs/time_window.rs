use crate::structs::constants::{HOURS_IN_DAY, HOURS_IN_PERIOD};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct TimeWindow {
    pub earliest: Option<f64>, // Start of the time window in hours (None if open all day)
    pub latest: Option<f64>,   // End of the time window in hours (None if open all day)
    pub is_open: bool,         // True if the time window is open all day
}

impl Default for TimeWindow {
    fn default() -> Self {
        Self {
            earliest: None,
            latest: None,
            is_open: true,
        }
    }
}

#[derive(Debug)]
pub enum TimeWindowError {
    InvalidRange((u32, f64, f64)),
}

impl TimeWindow {
    /// Creates a new TimeWindow object.
    // is_open should be passed, the following logic is valid if there is single time window per day
    // and open all day is set as None or TW with diff >= 24h
    pub fn new(earliest: Option<f64>, latest: Option<f64>) -> Result<Self, TimeWindowError> {
        if let (Some(earliest), Some(latest)) = (earliest, latest) {
            if earliest >= HOURS_IN_DAY as f64 || latest > HOURS_IN_DAY as f64 {
                return Err(TimeWindowError::InvalidRange((1, earliest, latest)));
            }
            if earliest > latest {
                return Err(TimeWindowError::InvalidRange((2, earliest, latest)));
            }
        }
        if earliest.is_none() || latest.is_none() {
            return Ok(Self::default());
        }
        let is_open = (latest.unwrap() - earliest.unwrap()).abs() >= HOURS_IN_DAY as f64;
        Ok(Self {
            earliest,
            latest,
            is_open,
        })
    }

    pub fn contains(&self, timestamp: f64) -> bool {
        if self.is_open {
            return true;
        }
        match (self.earliest, self.latest) {
            (Some(earliest), Some(latest)) => timestamp >= earliest && timestamp < latest,
            (Some(earliest), None) => timestamp >= earliest && timestamp < 24.0,
            (None, Some(latest)) => timestamp < latest,
            (None, None) => true,
        }
    }
}
