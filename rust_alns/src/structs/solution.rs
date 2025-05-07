use crate::structs::{
    vessel::Vessel,
    visit::Visit,
    voyage::Voyage,
    schedule::Schedule,
    node::{Base, Installation, HasLocation},
    distance_manager::DistanceManager,
    problem_data::ProblemData,
    context::Context,
};

#[derive(Clone)]
pub struct Solution {
    pub voyages: Vec<Voyage>, // All voyages, assigned to vesels
    pub visits: Vec<Visit>, // All visits, including unserved ones
    pub schedule: Schedule, // Informational class on how voyages are assigned to vessels
    pub total_cost: f64,
    pub is_feasible: bool,
}

impl Solution {
    /// Creates a new solution object
    pub fn new(visits: Vec<Visit>) -> Self {
        Self {
            voyages: Vec::new(),
            visits,
            schedule: Schedule::empty(),
            total_cost: 0.0,
            is_feasible: false,
        }
    }
    // Assumed that voyage is feasible and insertion is valid
    pub fn add_voyage(&mut self, voyage: Voyage) {
        self.schedule.assign_voyage(&voyage, &self.visits);
        for visit_id in &voyage.visit_ids {
            let visit = &mut self.visits[*visit_id];
            visit.assign_to_voyage(voyage.id());
        }
        self.voyages.push(voyage);
    }
    pub fn construct_initial_solution(
        &mut self,
        context: &Context,
    ) {
        // Initialize the solution with a greedy or random approach
        // This is a placeholder for the actual implementation
        // You can call the construct_initial_solution function here
        // e.g., construct_initial_solution(&self.visits, &context.problem.vessels);
    }

    pub fn unassign_visits(&mut self, visit_ids: &[usize]) {
        for visit_index in visit_ids {
            if let Some(visit) = self.visits.get_mut(*visit_index) {
                visit.unassign();
            }
        }
        // Removes visits from voyages without recalculating voyages and the schedule
        for voyage in &mut self.voyages {
            // NOTE: It passes ALL visit_ids to remove, even if they are not in the voyage
            // This is a potential performance issue, but it is safe
            voyage.remove_visits(visit_ids);
        }
        // Update schedule to match updated voyages
        self.schedule.set_need_update(true); // Use setter method
    }

    pub fn optimize_voyage_route(&mut self, voyage: &mut Voyage, context: &Context) {
        let voyage_visits = self.get_visits_for_voyage(voyage);
        let result = context.tsp_solver.solve_for_voyage(voyage, &voyage_visits);
        voyage.update_from_tsp_result(result);
    }

    pub fn get_visits_for_voyage(&self, voyage: &Voyage) -> Vec<&Visit> {
        voyage
            .visit_ids
            .iter()
            .filter_map(|id| self.visits.iter().find(|v| v.id() == *id))
            .collect()
    }

    pub fn is_feasible(&self) -> bool {
        self.is_feasible_light()
    }

    /// Lightweight feasibility check: runtime-suitable
    pub fn is_feasible_light(&self) -> bool {
        // 1. All visits are assigned

        for visit in &self.visits {
            if !visit.is_assigned {
                println!("Visit {} is not assigned", visit.id());
                // Output Installation
                println!("Visit installation: {:?}", visit.installation_id());
            }
        }

        println!("End of visit check");
        if self.visits.iter().any(|v| !v.is_assigned) {
            return false;
        }
        true
    }

    /// Heavy feasibility and consistency check: for testing and debugging
    pub fn is_feasible_deep(&self) -> bool {
        if !self.is_feasible_light() {
            return false;
        }
        // No overlapping voyages for the same vessel (based on the schedule)
        for (vessel_day, voyages) in &self.schedule.vessel_day_voyages {
            let times: Vec<_> = voyages.iter()
                .filter_map(|id| {
                    self.schedule.voyage_start_times.get(id).zip(
                        self.schedule.voyage_end_times.get(id)
                    )
                })
                .collect();
            for (i, (start_i, end_i)) in times.iter().enumerate() {
                for (j, (start_j, end_j)) in times.iter().enumerate() {
                    if i != j &&
                        crate::utils::utils::cyclic_intervals_overlap(**start_i, **end_i, **start_j, **end_j, crate::structs::constants::HOURS_IN_PERIOD as f64)
                    {
                        return false;
                    }
                }
            }
        }
        // All visits must be linked back correctly
        for visit in &self.visits {
            if let Some(voyage_id) = visit.assigned_voyage_id {
                if !self.voyages.iter().any(|v| v.id == voyage_id && v.visit_ids.contains(&visit.id())) {
                    return false;
                }
            }
        }
        // Check voyage and schedule consistency
        for voyage in &self.voyages {
            let voyage_id = voyage.id;
            if !self.schedule.voyage_start_times.contains_key(&voyage_id) {
                return false;
            }
            if !self.schedule.voyage_end_times.contains_key(&voyage_id) {
                return false;
            }
            let vessel_day_key = (voyage.vessel_id.unwrap(), voyage.departure_day.unwrap());
            if let Some(ids) = self.schedule.vessel_day_voyages.get(&vessel_day_key) {
                if !ids.contains(&voyage_id) {
                    return false;
                }
            } else {
                return false;
            }
            for visit_id in &voyage.visit_ids {
                let inst_id = self.visits[*visit_id].installation_id();
                if let Some(set) = self.schedule.departures_by_installation.get(&inst_id) {
                    if !set.contains(visit_id) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }

    // Returns unassigned visits
    pub fn get_unassigned_visits(&self) -> Vec<&Visit> {
        self.visits.iter().filter(|v| !v.is_assigned).collect()
    }
}