use crate::structs::node::Installation;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use std::sync::atomic::{AtomicU32, Ordering};

/// `Visit` represents a single visit to an installation within a route.
/// Question: Should departure_day and vessel_id be Visit fields?
#[derive(Debug, Clone)]
pub struct Visit {
    idx: u32, // Unique identifier for the visit (not the installation)
    installation: Installation, // Installation to visit
}

impl Visit {
    /// Creates a new visit to a given installation.
    pub fn new(installation: Installation) -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(1);
        let idx = COUNTER.fetch_add(1, Ordering::SeqCst);
        Visit {
            idx,
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