use serde::Deserialize;
use std::error::Error;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::thread_rng;

#[derive(Debug, Clone, Deserialize)]
pub struct TimeWindow {
    pub earliest: Option<u32>, // Start of the time window in hours (None if open all day)
    pub latest: Option<u32>,   // End of the time window in hours (None if open all day)
}

impl Default for TimeWindow {
    fn default() -> Self {
        Self { earliest: None, latest: None }
    }
}

#[derive(Debug)]
pub enum TimeWindowError {
    InvalidRange((u32, u32, u32)),
}

impl TimeWindow {
    pub fn new(earliest: Option<u32>, latest: Option<u32>) -> Result<Self, TimeWindowError> {
        if let (Some(earliest), Some(latest)) = (earliest, latest) {
            if earliest >= HOURS_IN_PERIOD || latest > HOURS_IN_PERIOD {
                return Err(TimeWindowError::InvalidRange((1, earliest, latest)));
            }
            if earliest > latest && !((earliest % HOURS_IN_DAY == 0) && (latest % HOURS_IN_DAY == 0) && (latest == earliest + HOURS_IN_DAY)) {
                return Err(TimeWindowError::InvalidRange((2, earliest, latest)));
            }
            // Check if the time window is within a single day or spans exactly one full day (like 0-24, 24-48, etc.)
            if earliest / HOURS_IN_DAY != latest / HOURS_IN_DAY && 
               !((earliest % HOURS_IN_DAY == 0) && (latest % HOURS_IN_DAY == 0) && (latest == earliest + HOURS_IN_DAY)) {
                return Err(TimeWindowError::InvalidRange((3, earliest, latest)));
            }
        }
        Ok(Self { earliest, latest })
    }

    pub fn contains(&self, timestamp: u32) -> bool {
        match (self.earliest, self.latest) {
            (Some(earliest), Some(latest)) => timestamp >= earliest && timestamp < latest,
            (Some(earliest), None) => timestamp >= earliest && timestamp < 24,
            (None, Some(latest)) => timestamp < latest,
            (None, None) => true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
}

impl Default for Location {
    fn default() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
        }
    }
}

impl Location {
    pub fn new(latitude: f64, longitude: f64) -> Self {
        Self { latitude, longitude }
    }
}

pub trait HasLocation {
    fn get_location(&self) -> &Location;
}
pub trait HasTimeWindows {
    fn get_service_time_windows(&self) -> &Vec<TimeWindow>;
}
#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    pub idx: u32,
    pub name: String,
    pub location: Location,
}

impl Node {
    pub fn new(idx: u32, name: String, location: Location) -> Self {
        Self { idx, name, location }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Installation {
    pub node: Node,
    pub deck_demand: u32,
    pub visit_frequency: u32,
    pub installation_type: String,
    pub departure_spread: u32,
    pub service_time: f64,
    pub time_window: TimeWindow,
    pub service_tw: Vec<TimeWindow>,  // Changed from time_windows to service_TW
}

impl Installation {
    pub fn builder() -> InstallationBuilder {
        InstallationBuilder::default()
    }

    pub fn generate_visit_scenario(&self) -> Vec<u32> {
        // TODO: Implement visit scenario generation
        // Placeholder implementation: random visit days
        let mut rng = thread_rng();
        (0..DAYS_IN_PERIOD)
            .map(|_| if rng.gen_range(0..100) < self.visit_frequency { 1 } else { 0 })
            .collect()
    }
}
#[derive(Default)]
pub struct InstallationBuilder {
    idx: u32,
    name: String,
    location: Location,
    deck_demand: u32,
    visit_frequency: u32,
    installation_type: String,
    departure_spread: u32,
    service_time: f64,
    time_window: TimeWindow,
    service_tw: Vec<TimeWindow>,  // Changed from time_windows to service_TW
}


impl InstallationBuilder {
    pub fn idx(mut self, idx: u32) -> Self {
        self.idx = idx;
        self
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn location(mut self, location: Location) -> Self {
        self.location = location;
        self
    }

    pub fn deck_demand(mut self, deck_demand: u32) -> Self {
        self.deck_demand = deck_demand;
        self
    }

    pub fn visit_frequency(mut self, visit_frequency: u32) -> Self {
        self.visit_frequency = visit_frequency;
        self
    }

    pub fn installation_type(mut self, installation_type: String) -> Self {
        self.installation_type = installation_type;
        self
    }

    pub fn departure_spread(mut self, departure_spread: u32) -> Self {
        self.departure_spread = departure_spread;
        self
    }

    pub fn service_time(mut self, service_time: f64) -> Self {
        self.service_time = service_time;
        self
    }

    pub fn time_window(mut self, time_window: TimeWindow) -> Self {
        self.time_window = time_window;
        self
    }

    // Creates a time window for each day in the period by adding 24-hour shifts
    pub fn service_time_windows(mut self, time_window: TimeWindow) -> Self {
        // If time_window is (0, 24) or default (None, None), use default time windows that always return true
        if (time_window.earliest == Some(0) && time_window.latest == Some(24)) || 
           (time_window.earliest.is_none() && time_window.latest.is_none()) {
            self.service_tw = vec![TimeWindow::default(); DAYS_IN_PERIOD as usize];
        } else {
            self.service_tw = (0..DAYS_IN_PERIOD)
                .map(|day| TimeWindow::new(
                    time_window.earliest.map(|earliest| (earliest + day * HOURS_IN_DAY)),
                    time_window.latest.map(|latest| (latest + day * HOURS_IN_DAY).saturating_sub(self.service_time as u32)),
                ).unwrap())
                .collect();
        }
        self
    }

    pub fn build(self) -> Result<Installation, &'static str> {
        // Initialize service_TW from time_window if not explicitly set
        let service_tw = if self.service_tw.is_empty() {
            if (self.time_window.earliest == Some(0) && self.time_window.latest == Some(24)) || 
                   (self.time_window.earliest.is_none() && self.time_window.latest.is_none()) {
                    vec![TimeWindow::default(); DAYS_IN_PERIOD as usize]
            } else {
                (0..DAYS_IN_PERIOD)
                    .map(|day| TimeWindow::new(
                        self.time_window.earliest.map(|earliest| (earliest + day * HOURS_IN_DAY)),
                        self.time_window.latest.map(|latest| (latest + day * HOURS_IN_DAY).saturating_sub(self.service_time as u32)),
                    ).unwrap())
                    .collect()
            }
        } else {
            self.service_tw
        };
        
        Ok(Installation {
            node: Node::new(self.idx, self.name, self.location),
            deck_demand: self.deck_demand,
            visit_frequency: self.visit_frequency,
            installation_type: self.installation_type,
            departure_spread: self.departure_spread,
            service_time: self.service_time,
            time_window: self.time_window,
            service_tw,
        })
    }
}

impl HasLocation for Installation {
    fn get_location(&self) -> &Location {
        &self.node.location
    }
}

impl HasTimeWindows for Installation {
    fn get_service_time_windows(&self) -> &Vec<TimeWindow> {
        &self.service_tw
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Base {
    pub node: Node,
    pub service_time: f64,
    pub time_window: TimeWindow,
    pub service_tw: Vec<TimeWindow>,  // Changed from service_TW to service_tw
}

impl Base {
    pub fn builder() -> BaseBuilder {
        BaseBuilder::default()
    }
}

#[derive(Default)]
pub struct BaseBuilder {
    idx: Option<u32>,
    name: Option<String>,
    location: Option<Location>,
    service_time: Option<f64>,
    time_window: Option<TimeWindow>,
    service_tw: Vec<TimeWindow>,  // Changed from service_TW to service_tw
}

impl BaseBuilder {
    pub fn idx(mut self, idx: u32) -> Self {
        self.idx = Some(idx);
        self
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn location(mut self, location: Location) -> Self {
        self.location = Some(location);
        self
    }

    pub fn service_time(mut self, service_time: f64) -> Self {
        self.service_time = Some(service_time);
        self
    }

    pub fn time_window(mut self, time_window: TimeWindow) -> Self {
        self.time_window = Some(time_window);
        self
    }

    // Creates a time window for each day in the period by adding 24-hour shifts
    pub fn service_time_windows(mut self, time_window: TimeWindow) -> Self {
        // If time_window is (0, 24) or default (None, None), use default time windows that always return true
        if (time_window.earliest == Some(0) && time_window.latest == Some(24)) || 
           (time_window.earliest.is_none() && time_window.latest.is_none()) {
            self.service_tw = vec![TimeWindow::default(); DAYS_IN_PERIOD as usize];
        } else {
            self.service_tw = (0..DAYS_IN_PERIOD)
                .map(|day| TimeWindow::new(
                    time_window.earliest.map(|earliest| (earliest + day * HOURS_IN_DAY) % HOURS_IN_PERIOD),
                    time_window.latest.map(|latest| (latest + day * HOURS_IN_DAY).saturating_sub(self.service_time.unwrap_or(0.0) as u32) % HOURS_IN_PERIOD),
                ).unwrap())
                .collect();
        }
        self
    }

    pub fn build(self) -> Result<Base, &'static str> {
        let time_window = self.time_window.ok_or("time_window is required")?;
        
        // If service_tw wasn't explicitly set, generate it from the time_window
        let service_tw = if self.service_tw.is_empty() {
            if (time_window.earliest == Some(0) && time_window.latest == Some(24)) || 
               (time_window.earliest.is_none() && time_window.latest.is_none()) {
                vec![TimeWindow::default(); DAYS_IN_PERIOD as usize]
            } else {
                (0..DAYS_IN_PERIOD)
                    .map(|day| TimeWindow::new(
                        time_window.earliest.map(|earliest| (earliest + day * HOURS_IN_DAY) % HOURS_IN_PERIOD),
                        time_window.latest.map(|latest| (latest + day * HOURS_IN_DAY).saturating_sub(self.service_time.unwrap_or(0.0) as u32) % HOURS_IN_PERIOD),
                    ).unwrap())
                    .collect()
            }
        } else {
            self.service_tw
        };
        
        Ok(Base {
            node: Node::new(
                self.idx.ok_or("idx is required")?,
                self.name.ok_or("name is required")?,
                self.location.ok_or("location is required")?,
            ),
            service_time: self.service_time.ok_or("service_time is required")?,
            time_window,
            service_tw,
        })
    }
}

impl HasLocation for Base {
    fn get_location(&self) -> &Location {
        &self.node.location
    }
}

impl HasTimeWindows for Base {
    fn get_service_time_windows(&self) -> &Vec<TimeWindow> {
        &self.service_tw
    }
}