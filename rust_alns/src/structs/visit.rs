use crate::structs::node::Installation;
use std::rc::Rc;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};

/// `Visit` represents a single visit to an installation within a route.
/// Question: Should departure_day and vessel_id be Visit fields?
#[derive(Debug, Clone)]
pub struct Visit {
    installation: Rc<Installation>, // Shared, immutable reference to the installation
}

impl Visit {
    /// Creates a new visit to a given installation.
    pub fn new(installation: Rc<Installation>) -> Self {
        Visit {
            installation,
        }
    }

    /// Returns the ID of the installation (delegated from Installation).
    pub fn installation_id(&self) -> u32 {
        self.installation.node.idx
    }

    /// Returns the deck demand for the installation.
    pub fn deck_demand(&self) -> u32 {
        self.installation.deck_demand
    }
}