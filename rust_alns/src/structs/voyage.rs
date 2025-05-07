use rand::seq::index::sample;
use itertools::Itertools;

use crate::structs::node::{Base, Installation};
use crate::structs::distance_manager::DistanceManager;
use crate::structs::visit::Visit;
use crate::structs::vessel::Vessel;
use crate::utils::tsp_solver::{TSPSolver, TSPResult};
use core::panic;
use std::rc::Rc;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use std::clone::Clone;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::structs::problem_data::ProblemData;
use crate::structs::context::Context;

use super::visit;

static VOYAGE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct Voyage {
    pub id: usize, // always required
    pub vessel_id: Option<usize>,        
    pub voyage_speed: Option<f64>,
    pub departure_day: Option<usize>,    // 0..DAYS_IN_PERIOD-1 
    pub visit_ids: Vec<usize>,           // Represents the ROUTE of the voyage
    pub sailing_time: Option<f64>,       // Total sailing time
    pub waiting_time: Option<f64>,       // Total waiting time at installations
    pub arrival_time: Option<f64>,       // Time when vessel arrives back at the base [0, +∞)
    pub end_time_at_base: Option<f64>,   // (arrival_time + wait_time + service_time) [0, +∞)
    pub is_feasible: bool,
    pub load: Option<u32>,       
    pub need_update: bool, // Indicates if the voyage needs an update
}

impl Voyage {       
    // Creates a new voyage object
    pub fn new() -> Self {
        let id = VOYAGE_COUNTER.fetch_add(1, Ordering::Relaxed);

        Self {
            id,
            vessel_id: None,
            voyage_speed: None,
            departure_day: None,
            visit_ids: Vec::new(),
            sailing_time: None,
            waiting_time: None,
            arrival_time: None,
            end_time_at_base: None,
            is_feasible: false,
            load: Some(0),

            need_update: false, // Initialize to false
        }
    }
    // Creates a new voyage object with a specific id
    pub fn new_with_id(id: usize) -> Self {
        Self {
            id,
            vessel_id: None,
            voyage_speed: None,
            departure_day: None,
            visit_ids: Vec::new(),
            sailing_time: None,
            waiting_time: None,
            arrival_time: None,
            end_time_at_base: None,
            is_feasible: false,
            load: Some(0),
            need_update: false, // Initialize to false
        }
    }

    pub fn new_with_visit_ids(visit_ids: Vec<usize>, departure_day: Option<usize>) -> Self {
        let id = VOYAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            vessel_id: None,
            voyage_speed: None,
            departure_day,
            visit_ids,
            sailing_time: None,
            waiting_time: None,
            arrival_time: None,
            end_time_at_base: None,
            is_feasible: false,
            load: None, // Load is not set yet
            need_update: true, // Initialize to false
        }
    }
    
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn speed(&self) -> Option<f64> {
        self.voyage_speed
    }

    pub fn load(&self) -> Option<u32> {
        self.load
    }

    pub fn start_time(&self) -> Option<f64> {
        Some(self.departure_day.unwrap() as f64 * HOURS_IN_DAY as f64 + REL_DEPARTURE_TIME as f64)
    }

    pub fn end_time(&self) -> Option<f64> {
        self.end_time_at_base
    }

    pub fn assign_vessel(&mut self, vessel: &Vessel, context: &Context) {
        self.vessel_id = Some(vessel.id);
        self.voyage_speed = Some(vessel.speed);
        // self.optimize_route(context);
        // Where to get Visits ???
        panic!("Voyage::assign_vessel: Not implemented yet");
    }

    pub fn update_from_tsp_result(&mut self, result: TSPResult) {
        self.visit_ids = result.visit_ids_seq;
        self.sailing_time = Some(result.sailing_time);
        self.waiting_time = Some(result.waiting_time);
        self.arrival_time = Some(result.arrival_time);
        self.end_time_at_base = Some(result.end_time);
        self.need_update = true; // Set to true after update
    }

    pub fn update_load(&mut self, visits: &[Visit]) {
        let total_load = self.visit_ids
            .iter()
            .filter_map(|id| visits.iter().find(|v| v.id() == *id))
            .map(|v| v.demand())
            .sum();
        self.load = Some(total_load);
    }

    pub fn select_visits_to_remove(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<usize> {
        let len = self.visit_ids.len();
        let to_remove = n.min(len);
        let indices = sample(rng, len, to_remove).into_vec();

        indices.into_iter()
            .map(|i| self.visit_ids[i])
            .collect()
    }

    pub fn remove_visits(&mut self, visit_ids_to_remove: &[usize]) -> usize {
        let before = self.visit_ids.len();
        self.visit_ids.retain(|id| !visit_ids_to_remove.contains(id));
        let removed = before - self.visit_ids.len();
        if removed > 0 {
            self.need_update = true;
        }
        removed
    }

    // Calculates the lowest insertion cost of a visit into the voyage
    pub fn best_insertion_cost(&self, context: &Context, visit: usize) -> f64 {
        return 0.0;
    }
    pub fn insertion_cost(&self, context: &Context, visit: usize, position: usize) -> f64 {
        return 0.0;
    }

    pub fn get_visit_sequence(&self) -> Vec<usize> {
        self.visit_ids.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_with_id() {
        let id = 42;
        let voyage = Voyage::new_with_id(id);
        
        assert_eq!(voyage.id, id);
        assert_eq!(voyage.vessel_id, None);
        assert_eq!(voyage.departure_day, None);
        assert!(voyage.visit_ids.is_empty());
        assert_eq!(voyage.sailing_time, None);
        assert_eq!(voyage.waiting_time, None);
        assert_eq!(voyage.arrival_time, None);
        assert_eq!(voyage.end_time_at_base, None);
        assert_eq!(voyage.is_feasible, false);
    }
}
