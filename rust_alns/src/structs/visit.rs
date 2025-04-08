use crate::structs::node::Installation;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use std::sync::atomic::{AtomicU32, Ordering};

/// `Visit` represents a single visit to an installation within a route.
/// Question: Should departure_day and vessel_id be Visit fields?
#[derive(Debug, Clone)]
pub struct Visit {
    pub id: usize,
    pub installation_id: usize,
    pub departure_day: Option<usize>,
    pub assigned_voyage_id: Option<usize>,
    pub is_assigned: bool,
}

impl Visit {
    /// Creates a new visit to a given installation.
    pub fn new(id: usize, installation_id: usize) -> Self {
        Visit {
            id,
            installation_id,
            departure_day: None,
            assigned_voyage_id: None,
            is_assigned: false,
        }
        
    }
}
