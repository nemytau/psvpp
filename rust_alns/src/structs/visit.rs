use crate::structs::node::Installation;
use std::rc::Rc;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};

/// `Visit` represents a single visit to an installation within a route.
#[derive(Debug, Clone)]
pub struct Visit {
    installation: Rc<Installation>, // Shared, immutable reference to the installation
    pub departure_day: Option<u32>, // The scheduled departure day for the visit
    pub vessel_id: Option<u32>,     // The vessel assigned to this visit
    pub is_assigned: bool,           // Indicates if the visit is fully scheduled
}

impl Visit {
    /// Creates a new visit to a given installation.
    pub fn new(installation: Rc<Installation>) -> Self {
        Visit {
            installation,
            departure_day: None,
            vessel_id: None,
            is_assigned: false,
        }
    }

    /// Returns the ID of the installation (delegated from Installation).
    pub fn installation_id(&self) -> u32 {
        self.installation.node.idx
    }

    /// Sets the departure day for the visit and updates `is_assigned` status.
    pub fn set_departure_day(&mut self, departure_day: u32) {
        if departure_day >= DAYS_IN_PERIOD {
            panic!("Departure day exceeds the allowed period range of 0 to {}", DAYS_IN_PERIOD - 1);
        }
        self.departure_day = Some(departure_day);
        self.update_assign_status();
    }

    /// Sets the vessel for the visit and updates `is_assigned` status.
    pub fn assign_vessel(&mut self, vessel_id: u32) {
        self.vessel_id = Some(vessel_id);
        self.update_assign_status();
    }

    /// Updates the `is_assigned` status based on whether both `departure_day` and `vessel_id` are set.
    fn update_assign_status(&mut self) {
        self.is_assigned = self.departure_day.is_some() && self.vessel_id.is_some();
    }

    /// Returns the deck demand for the installation.
    pub fn deck_demand(&self) -> u32 {
        self.installation.deck_demand
    }
}