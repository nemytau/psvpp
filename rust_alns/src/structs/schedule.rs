use std::rc::Rc;
use crate::structs::node::{Installation, Base, Node, HasLocation};
use crate::structs::voyage::Voyage;
use crate::structs::vessel::Vessel;
use crate::structs::visit::Visit;
use crate::structs::distance_manager::DistanceManager;
use crate::structs::transaction::Transaction;
use std::collections::{BTreeSet, HashMap};
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use rand::seq::SliceRandom;


#[derive(Debug, Clone)]
pub struct Schedule {
    /// voyage_id → departure time
    pub voyage_start_times: HashMap<usize, f64>,

    /// visit_id → actual service start time (if known)
    pub visit_service_times: HashMap<usize, f64>,

    /// (vessel_id, day) → voyage_ids
    pub vessel_day_voyages: HashMap<(usize, usize), Vec<usize>>,

    /// installation_id → set of visit_ids
    pub departures_by_installation: HashMap<usize, BTreeSet<usize>>,
}

impl Schedule{
    pub fn empty() -> Self {
        Schedule {
            voyage_start_times: HashMap::new(),
            visit_service_times: HashMap::new(),
            vessel_day_voyages: HashMap::new(),
            departures_by_installation: HashMap::new(),
        }
    }
}