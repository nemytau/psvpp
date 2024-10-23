use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Clone, Deserialize)]
pub struct TimeWindow {
    pub start: Option<u32>,  // Start of the time window in hours (None if open all day)
    pub end: Option<u32>,    // End of the time window in hours (None if open all day)
}

#[derive(Debug)]
pub enum TimeWindowError {
    InvalidStartTime(u32),
    InvalidEndTime(u32),
}

impl TimeWindow {
    // Constructor for TimeWindow with validation
    pub fn new(start: Option<u32>, end: Option<u32>) -> Result<Self, TimeWindowError> {
        // Validate the start time if present
        if let Some(s) = start {
            if s > 24 {
                return Err(TimeWindowError::InvalidStartTime(s));
            }
        }

        // Validate the end time if present
        if let Some(e) = end {
            if e > 24 {
                return Err(TimeWindowError::InvalidEndTime(e));
            }
        }

        // If validation passed, return the TimeWindow
        Ok(Self { start, end })
    }

    // Method to check if a timestamp (in hours) is within the time window
    pub fn contains(&self, timestamp: u32) -> bool {
        match (self.start, self.end) {
            (Some(start), Some(end)) => timestamp >= start && timestamp <= end,
            (Some(start), None) => timestamp >= start,  // No upper bound
            (None, Some(end)) => timestamp <= end,      // No lower bound
            (None, None) => true,  // No time window restrictions
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
}

impl Location {
    pub fn new(latitude: f64, longitude: f64) -> Self {
        Self { latitude, longitude }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    pub id: u32,
    pub name: String,
    pub location: Location,
    pub service_time: f64,       // Service time in minutes
    pub time_window: TimeWindow, // Time window for visiting the node
}

impl Node {
    pub fn new(
        id: u32,
        name: String,
        location: Location,
        service_time: f64,
        time_window: TimeWindow,
    ) -> Self {
        Self {
            id,
            name,
            location,
            service_time,
            time_window,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Installation {
    pub node: Node,               // Base node properties
    pub demand: f64,              // Demand of the installation (e.g., volume of goods)
    pub visit_frequency: u32,     // Number of visits required in a given period
    pub installation_type: String, // Type of installation (e.g., "Oil Rig", "Warehouse")
    pub departure_spread: u32,    // Spread of departure times for the installation
}

impl Installation {
    pub fn new(
        id: u32,
        name: String,
        location: Location,
        service_time: f64,
        time_window: TimeWindow,
        demand: f64,
        visit_frequency: u32,
        installation_type: String,
        departure_spread: u32,
    ) -> Self {
        Self {
            node: Node::new(id, name, location, service_time, time_window),
            demand,
            visit_frequency,
            installation_type,
            departure_spread,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Base {
    pub node: Node,  // Base just holds a Node for now
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
            node: Node::new(idx, name, location, service_time, time_window),
        }
    }
}