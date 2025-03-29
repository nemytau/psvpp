use serde::Deserialize;
use crate::structs::constants::{HOURS_IN_DAY, HOURS_IN_PERIOD};


#[derive(Debug, Clone, Deserialize)]
pub struct TimeWindow {
    pub earliest: Option<f64>, // Start of the time window in hours (None if open all day)
    pub latest: Option<f64>,   // End of the time window in hours (None if open all day)
}

impl Default for TimeWindow {
    fn default() -> Self {
        Self { earliest: None, latest: None }
    }
}

#[derive(Debug)]
pub enum TimeWindowError {
    InvalidRange((u32, f64, f64)),
}

impl TimeWindow {
    pub fn new(earliest: Option<f64>, latest: Option<f64>) -> Result<Self, TimeWindowError> {
        if let (Some(earliest), Some(latest)) = (earliest, latest) {
            if earliest >= HOURS_IN_PERIOD as f64 || latest > HOURS_IN_PERIOD as f64 {
                return Err(TimeWindowError::InvalidRange((1, earliest, latest)));
            }
            if earliest > latest && !((earliest % HOURS_IN_DAY as f64 == 0.0) && (latest % HOURS_IN_DAY as f64 == 0.0) && ((latest - earliest).abs() < f64::EPSILON + HOURS_IN_DAY as f64)) {
                return Err(TimeWindowError::InvalidRange((2, earliest, latest)));
            }
        }
        Ok(Self { earliest, latest })
    }

    pub fn contains(&self, timestamp: f64) -> bool {
        match (self.earliest, self.latest) {
            (Some(earliest), Some(latest)) => timestamp >= earliest && timestamp < latest,
            (Some(earliest), None) => timestamp >= earliest && timestamp < 24.0,
            (None, Some(latest)) => timestamp < latest,
            (None, None) => true,
        }
    }
}