use crate::structs::distance_manager::DistanceManager;
use crate::structs::node::{Base, HasLocation, Installation};
use crate::structs::vessel::Vessel;
use crate::structs::visit::Visit;

use super::node::HasTimeWindows;

#[derive(Clone)]
pub struct ProblemData {
    pub vessels: Vec<Vessel>,
    pub installations: Vec<Installation>,
    pub base: Base,
    pub distance_manager: DistanceManager,
}

impl ProblemData {
    /// Creates a new problem data object
    pub fn new(vessels: Vec<Vessel>, installations: Vec<Installation>, base: Base) -> Self {
        let installations_with_locations: Vec<&dyn HasLocation> = installations
            .iter()
            .map(|i| i as &dyn HasLocation)
            .collect();
        let distance_manager =
            DistanceManager::from_base_and_installations(&base, &installations_with_locations);
        Self {
            vessels,
            installations,
            base,
            distance_manager,
        }
    }

    pub fn fcs(&self) -> f64 {
        0.43
    }
    pub fn fcw(&self) -> f64 {
        0.26
    }
    pub fn lng_cost(&self) -> f64 {
        650.0
    }
    pub fn generate_visits(&self) -> Vec<Visit> {
        let mut visits = Vec::new();

        for installation in &self.installations {
            for _visit_num in 0..installation.visit_frequency {
                let id = visits.len();
                let installation_id = installation.id;
                // Create a new visit with the installation ID and visit number
                visits.push(Visit::new(
                    id,
                    installation_id,
                    installation.deck_demand,
                    installation.service_time,
                ));
            }
        }

        visits
    }

    pub fn get_time_window(&self, installation_id: usize) -> (f64, f64) {
        let installation = &self.installations[installation_id - 1];
        (
            installation.service_time_window.earliest.unwrap_or(0.0),
            installation.service_time_window.latest.unwrap_or(24.0),
        )
    }

    pub fn get_installation_by_id(&self, id: usize) -> Option<&Installation> {
        self.installations.get(id.wrapping_sub(1))
    }
}
