use rand::seq::index::sample;
use itertools::Itertools;

use crate::structs::node::{Base, Installation};
use crate::structs::distance_manager::DistanceManager;
use crate::structs::visit::Visit;
use crate::structs::vessel::Vessel;
use crate::utils::tsp_solver::{self, TSPResult, TSPSolver};
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
    pub load: Option<u32>,
    pub route_dirty: bool,     // Indicates if routing needs recomputation
    pub state_dirty: bool,     // Indicates if derived state (load, feasibility) needs update
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
            load: Some(0),
            route_dirty: false,
            state_dirty: false,
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
            load: Some(0),
            route_dirty: false,
            state_dirty: false,
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
            load: None, // Load is not set yet
            route_dirty: true,
            state_dirty: true,
        }
    }
    
    // Creates an empty voyage object
    pub fn empty() -> Self {
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
            load: None,
            route_dirty: false,
            state_dirty: false,
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

    pub fn set_load(&mut self, load: u32) {
        self.load = Some(load);
    }

    pub fn start_time(&self) -> Option<f64> {
        Some(self.departure_day.unwrap() as f64 * HOURS_IN_DAY as f64 + REL_DEPARTURE_TIME as f64)
    }

    pub fn end_time(&self) -> Option<f64> {
        self.end_time_at_base
    }

    pub fn assign_vessel(&mut self, vessel: &Vessel, _context: &Context) {
        self.vessel_id = Some(vessel.id);
        self.voyage_speed = Some(vessel.speed);
        // self.optimize_route(context);
        // Where to get Visits ???
        panic!("Voyage::assign_vessel: Not implemented yet");
    }

    pub fn apply_tsp_result(&mut self, result: TSPResult) {
        self.visit_ids = result.visit_ids_seq;
        self.sailing_time = Some(result.sailing_time);
        self.waiting_time = Some(result.waiting_time);
        self.arrival_time = Some(result.arrival_time);
        self.end_time_at_base = Some(result.end_time);
        self.route_dirty = false;
        // We conservatively assume state is dirty to ensure load and feasibility are re-evaluated.
        self.state_dirty = true;
    }

    pub fn finalize_state(&mut self, total_load: u32) {
        self.set_load(total_load);
        self.state_dirty = false;
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
            self.route_dirty = true;
            self.state_dirty = true;
        }
        removed
    }

    pub fn added_cost_from_result(&self, new_result: &TSPResult, context: &Context) -> f64 {
        let old_cost = self.objective_cost(context);
        let new_cost = new_result.arrival_time;
        new_cost - old_cost
    }
    pub fn objective_cost(&self, _context: &Context) -> f64 {
        self.arrival_time.unwrap_or(0.0) - self.start_time().unwrap_or(0.0)
    }
    pub fn objective_cost_from_result(&self, result: &TSPResult, _context: &Context) -> f64 {
        result.arrival_time - self.start_time().unwrap_or(0.0)
    }
    pub fn get_visit_sequence(&self) -> Vec<usize> {
        self.visit_ids.clone()
    }

    pub fn add_visit(&mut self, visit_id: usize) {
        self.visit_ids.push(visit_id);
        self.route_dirty = true;
        self.state_dirty = true;
    }

    /// Update timing and metadata (sailing_time, waiting_time, arrival_time, end_time) using the current visit_ids order.
    /// This does not solve TSP or change the route. Sets route_dirty = false after update.
    pub fn update_details(&mut self, tsp_solver: &crate::utils::tsp_solver::TSPSolver) {
        let speed = self.speed().unwrap_or_else(|| panic!("Vessel speed must be set up for the voyage"));
        let start_time = self.start_time().unwrap_or_else(|| panic!("Start time must be set up for the voyage"));
        // Use the visit_ids order directly, do not clone self or solve TSP
        let route = tsp_solver.visit_sequence_to_route_public(&self.visit_ids);
        let (sailing_time, waiting_time, arrival_time, end_time) = tsp_solver.calculate_voyage_details_public(&route, speed, start_time);
        self.sailing_time = Some(sailing_time);
        self.waiting_time = Some(waiting_time);
        self.arrival_time = Some(arrival_time);
        self.end_time_at_base = Some(end_time);
        self.route_dirty = false;
    }

    pub fn is_empty(&self) -> bool {
        self.visit_ids.is_empty()
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
    }
}
