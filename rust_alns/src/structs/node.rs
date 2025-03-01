use serde::Deserialize;
use std::error::Error;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Debug, Clone, Deserialize)]
pub struct TimeWindow {
    pub start: Option<u32>, // Start of the time window in hours (None if open all day)
    pub end: Option<u32>,   // End of the time window in hours (None if open all day)
}

impl Default for TimeWindow {
    fn default() -> Self {
        Self { start: None, end: None }
    }
}

#[derive(Debug)]
pub enum TimeWindowError {
    InvalidStartTime(u32),
    InvalidEndTime(u32),
}

impl TimeWindow {
    pub fn new(start: Option<u32>, end: Option<u32>) -> Result<Self, TimeWindowError> {
        if let Some(s) = start {
            if s >= 24 {
                return Err(TimeWindowError::InvalidStartTime(s));
            }
        }

        if let Some(e) = end {
            if e >= 24 {
                return Err(TimeWindowError::InvalidEndTime(e));
            }
        }

        Ok(Self { start, end })
    }

    pub fn contains(&self, timestamp: u32) -> bool {
        match (self.start, self.end) {
            (Some(start), Some(end)) => timestamp >= start && timestamp < end,
            (Some(start), None) => timestamp >= start && timestamp < 24,
            (None, Some(end)) => timestamp < end,
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
pub trait HasTimeWindow {
    fn get_time_window(&self) -> &TimeWindow;
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
}

impl Installation {
    pub fn builder() -> InstallationBuilder {
        InstallationBuilder::default()
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

    pub fn build(self) -> Result<Base, &'static str> {
        Ok(Base {
            node: Node::new(
                self.idx.ok_or("idx is required")?,
                self.name.ok_or("name is required")?,
                self.location.ok_or("location is required")?,
            ),
            service_time: self.service_time.ok_or("service_time is required")?,
            time_window: self.time_window.ok_or("time_window is required")?,
        })
    }
}
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

    pub fn build(self) -> Installation {
        Installation {
            node: Node::new(self.idx, self.name, self.location),
            deck_demand: self.deck_demand,
            visit_frequency: self.visit_frequency,
            installation_type: self.installation_type,
            departure_spread: self.departure_spread,
            service_time: self.service_time,
            time_window: self.time_window,
        }
    }
}

impl HasLocation for Installation {
    fn get_location(&self) -> &Location {
        &self.node.location
    }
}

impl HasTimeWindow for Installation {
    fn get_time_window(&self) -> &TimeWindow {
        &self.time_window
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Base {
    pub node: Node,
    pub service_time: f64,
    pub time_window: TimeWindow,
}

impl Base {
    pub fn new(
        idx: u32,
        name: String,
        location: Location,
        service_time: f64,
        time_window: TimeWindow,
    ) -> Self {
        Self {
            node: Node::new(idx, name, location),
            service_time,
            time_window,
        }
    }
}

impl HasLocation for Base {
    fn get_location(&self) -> &Location {
        &self.node.location
    }
}

impl HasTimeWindow for Base {
    fn get_time_window(&self) -> &TimeWindow {
        &self.time_window
    }
}
