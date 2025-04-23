use crate::structs::node::Installation;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use std::sync::atomic::{AtomicU32, Ordering};

/// `Visit` represents a single visit to an installation within a route.
/// Question: Should departure_day and vessel_id be Visit fields?
#[derive(Debug, Clone)]
pub struct Visit {
    id: usize,
    installation_id: usize,
    demand: u32,
    service_time: f64,
    pub departure_day: Option<usize>,
    pub assigned_voyage_id: Option<usize>,
    pub is_assigned: bool,
}

impl Visit {
    /// Creates a new visit to a given installation.
    pub fn new(id: usize, installation_id: usize, demand: u32, service_time: f64) -> Self {
        Visit {
            id,
            installation_id,
            demand,
            service_time,
            departure_day: None,
            assigned_voyage_id: None,
            is_assigned: false,
        }
    }
    
    pub fn assign_to_voyage(&mut self, voyage_id: usize) {
        self.assigned_voyage_id = Some(voyage_id);
        self.is_assigned = true;
    }
    
    pub fn unassign(&mut self) {
        self.assigned_voyage_id = None;
        self.is_assigned = false;
    }

    /// Returns the visit ID
    pub fn id(&self) -> usize {
        self.id
    }
    
    /// Returns the installation ID this visit is for
    pub fn installation_id(&self) -> usize {
        self.installation_id
    }
    
    /// Returns the demand for this visit
    pub fn demand(&self) -> u32 {
        self.demand
    }
    
    /// Returns the service time required for this visit
    pub fn service_time(&self) -> f64 {
        self.service_time
    }
}
