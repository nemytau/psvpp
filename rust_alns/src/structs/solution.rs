use crate::structs::{
    visit::Visit,
    voyage::Voyage,
    schedule::Schedule,
    context::Context,
};
use std::cell::RefCell;
use log::{warn};

#[derive(Clone)]
pub struct Solution {
    pub voyages: Vec<RefCell<Voyage>>, // All voyages, assigned to vesels
    _visits: Vec<Visit>, // All visits, including unserved ones (private)
    pub schedule: Schedule, // Informational class on how voyages are assigned to vessels
    pub total_cost: f64,
    pub is_feasible: bool,
}

impl Solution {
    /// Creates a new solution object
    pub fn new(visits: Vec<Visit>) -> Self {
        Self {
            voyages: Vec::new(),
            _visits: visits,
            schedule: Schedule::empty(),
            total_cost: 0.0,
            is_feasible: false,
        }
    }
    // Assumed that voyage is feasible and insertion is valid
    pub fn add_voyage(&mut self, voyage: Voyage) {
        self.schedule.assign_voyage(&voyage, &self._visits);
        for visit_id in &voyage.visit_ids {
            let visit = self._visits.get_mut(*visit_id).expect("Invalid visit_id");
            visit.assign_to_voyage(voyage.id());
        }
        self.voyages.push(RefCell::new(voyage));
        self.schedule.set_need_update(true);
    }
    pub fn construct_initial_solution(
        &mut self,
        _context: &Context,
    ) {
        // Initialize the solution with a greedy or random approach
        // This is a placeholder for the actual implementation
        // You can call the construct_initial_solution function here
        // e.g., construct_initial_solution(&self.visits, &context.problem.vessels);
    }

    pub fn unassign_visits(&mut self, visit_ids: &[usize]) {
        for visit_index in visit_ids {
            if let Some(visit) = self._visits.get_mut(*visit_index) {
                visit.unassign();
            }
        }
        for voyage_cell in &self.voyages {
            let mut voyage = voyage_cell.borrow_mut();
            let removed = voyage.remove_visits(visit_ids);
            if removed > 0 {
                voyage.route_dirty = true;
                voyage.state_dirty = true;
            }
        }
        self.schedule.set_need_update(true);
    }

    pub fn optimize_voyage_route(&mut self, voyage: &mut Voyage, context: &Context) {
        let result = context.tsp_solver.solve_for_voyage(voyage);
        voyage.apply_tsp_result(result);
    }

    pub fn get_visits_for_voyage(&self, voyage: &Voyage) -> Vec<&Visit> {
        voyage
            .visit_ids
            .iter()
            .filter_map(|id| self._visits.iter().find(|v| v.id() == *id))
            .collect()
    }

    /// Checks if the solution is complete, meaning all visits are assigned to voyages.
    /// This does not check feasibility, only completeness.
    pub fn is_complete_solution(&self) -> bool {
        
        // TODO: Spread of departures check

        if self._visits.iter().any(|v| !v.is_assigned) {
            return false;
        }
        true
    }

    /// Checks if the solution meets all the constraints. Might be incomplete.
    pub fn is_fully_feasible(&mut self, context: &Context) -> bool {
        self.ensure_consistency_updated(context);
        use itertools::Itertools;
        use crate::structs::constants::HOURS_IN_PERIOD;
        use crate::utils::utils::cyclic_intervals_overlap;
        let mut vessel_voyages: std::collections::HashMap<usize, Vec<_>> = std::collections::HashMap::new();
        for voyage_cell in &self.voyages {
            let voyage = voyage_cell.borrow();
            if let Some(vessel_id) = voyage.vessel_id {
                vessel_voyages.entry(vessel_id).or_default().push(voyage.clone());
            }
        }
        for (vessel_id, voyages) in &vessel_voyages {
            for (a, b) in voyages.iter().tuple_combinations() {
                if let (Some(a_start), Some(a_end), Some(b_start), Some(b_end)) = (a.start_time(), a.end_time(), b.start_time(), b.end_time()) {
                    if cyclic_intervals_overlap(a_start, a_end, b_start, b_end, HOURS_IN_PERIOD as f64) {
                        println!("Infeasible: Vessel {} has overlapping voyages {} and {}", vessel_id, a.id, b.id);
                        return false;
                    }
                }
            }
        }

        // All visits must be linked back correctly
        for visit in &self._visits {
            if let Some(voyage_id) = visit.assigned_voyage_id {
                if !self.voyages.iter().any(|v| {
                    let v = v.borrow();
                    v.id == voyage_id && v.visit_ids.contains(&visit.id())
                }) {
                    println!("Infeasible: Visit {} assigned to voyage {} but not found in voyage's visit_ids", visit.id(), voyage_id);
                    return false;
                }
            }
        }
        // Check voyage and schedule consistency
        for voyage_cell in &self.voyages {
            let voyage = voyage_cell.borrow();
            let voyage_id = voyage.id;
            if !self.schedule.voyage_start_times.contains_key(&voyage_id) {
                println!("Infeasible: Voyage {} missing from schedule.voyage_start_times", voyage_id);
                return false;
            }
            if !self.schedule.voyage_end_times.contains_key(&voyage_id) {
                println!("Infeasible: Voyage {} missing from schedule.voyage_end_times", voyage_id);
                return false;
            }
            let vessel_day_key = (voyage.vessel_id.unwrap(), voyage.departure_day.unwrap());
            if let Some(&scheduled_voyage_id) = self.schedule.vessel_day_voyages.get(&vessel_day_key) {
                if scheduled_voyage_id != voyage_id {
                    println!("Infeasible: Schedule vessel_day_voyages for vessel {:?} day {:?} points to voyage {} but expected {}", voyage.vessel_id, voyage.departure_day, scheduled_voyage_id, voyage_id);
                    return false;
                }
            } else {
                println!("Infeasible: No entry in schedule.vessel_day_voyages for vessel {:?} day {:?}", voyage.vessel_id, voyage.departure_day);
                return false;
            }
            for visit_id in &voyage.visit_ids {
                let inst_id = self._visits[*visit_id].installation_id();
                if let Some(set) = self.schedule.departures_by_installation.get(&inst_id) {
                    if !set.contains(visit_id) {
                        println!("Infeasible: Visit {} (installation {}) not found in schedule.departures_by_installation", visit_id, inst_id);
                        return false;
                    }
                } else {
                    println!("Infeasible: No entry in schedule.departures_by_installation for installation {}", inst_id);
                    return false;
                }
            }
        }
        true
    }

    pub fn ensure_consistency_updated(&mut self, context: &Context) {
        // Remove all empty voyages before any updates
        let before = self.voyages.len();
        self.voyages.retain(|voyage_cell| {
            let voyage = voyage_cell.borrow();
            !voyage.is_empty()
        });
        let after = self.voyages.len();
        let removed = before - after;
        if removed > 0 {
            log::info!("Removed {} empty voyages from solution ({} -> {})", removed, before, after);
        } else {
            log::debug!("No empty voyages removed. Total voyages: {}", after);
        }
        let mut route_updates = 0;
        let mut state_updates = 0;
        for voyage_cell in &self.voyages {
            let mut voyage = voyage_cell.borrow_mut();
            // Only update route/timing if route_dirty is set (do NOT re-optimize route if not needed)
            if voyage.route_dirty {
                // Update timing and metadata using the current visit_ids order, do not solve TSP
                voyage.update_details(&context.tsp_solver);
                route_updates += 1;
            }
            // Always update load and other state if state_dirty is set
            if voyage.state_dirty {
                self.update_voyage_load(&mut voyage);
                state_updates += 1;
            }
        }
        if route_updates > 0 || state_updates > 0 {
            log::info!("Updated {} voyage routes and {} voyage states", route_updates, state_updates);
        } else {
            log::debug!("No voyage route/state updates needed");
        }
        if self.schedule.needs_update() {
            log::info!("Schedule marked as needing update, rebuilding schedule");
            self.ensure_schedule_is_updated();
        } else {
            log::debug!("Schedule is up-to-date, no rebuild needed");
        }
    }

    // WARNING: This function deletes all empty voyages and idle vessels from the solution.
    pub fn ensure_schedule_is_updated(&mut self) {
        if !self.is_schedule_up_to_date() {
            // Rebuild the schedule from current voyages and visits
            self.schedule = Schedule::empty();
            for voyage_cell in &self.voyages {
                let voyage = voyage_cell.borrow();
                // Skip empty voyages (idle voyages)
                if voyage.is_empty() {
                    continue;
                }
                self.schedule.assign_voyage(&voyage, &self._visits);
            }
            self.schedule.set_need_update(false);
        }
    }

    pub fn is_schedule_up_to_date(&self) -> bool {
        !self.schedule.needs_update()
    }

    // Returns unassigned visits
    pub fn get_unassigned_visits(&self) -> Vec<&Visit> {
        self._visits.iter().filter(|v| !v.is_assigned).collect()
    }
    
    /// Returns the top-k cheapest feasible insertion costs for a visit.
    ///
    /// # Schedule Consistency
    /// The caller is responsible for ensuring the schedule is up-to-date before calling this function.
    /// This method assumes `schedule.needs_update() == false`.
    pub fn top_k_visit_insertion_costs(
        &self,
        context: &Context,
        visit: usize,
        k: usize,
    ) -> Vec<(usize, f64)> {
        debug_assert!(
            !self.schedule.needs_update(),
            "Schedule must be up-to-date before calling top_k_visit_insertion_costs."
        );
        let mut costs = Vec::new();
        for voyage_cell in self.voyages.iter() {
            let voyage = voyage_cell.borrow();
            if !self.visit_insertion_is_possible(context, visit, voyage.id) {
                continue;
            }
            let tsp_result = context.tsp_solver.evaluate_greedy_insertion(&*voyage, visit);
            let new_start = voyage.start_time().unwrap_or(0.0);
            let new_end = tsp_result.end_time;
            let vessel_id = match voyage.vessel_id {
                Some(id) => id,
                None => continue,
            };

            if self.overlaps_with_real_voyages(vessel_id, voyage.id, new_start, new_end) {
                continue;
            }

            let cost = voyage.added_cost_from_result(&tsp_result, context);
            costs.push((voyage.id(), cost));
        }
        costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        costs.truncate(k);
        costs
    }

    // Same installation not visited on the same day, spread of departures not violated.
    pub fn visit_insertion_is_possible(
        &self,
        context: &Context,
        visit_id: usize,
        voyage_id: usize,
    ) -> bool {
        let visit = match self.visit(visit_id) {
            Some(v) => v,
            None => return false,
        };
        let voyage = match self.voyages.iter().find_map(|v| {
            let v = v.borrow();
            if v.id == voyage_id { Some(v) } else { None }
        }) {
            Some(v) => v,
            None => return false,
        };
        let installation_id = visit.installation_id();
        let departure_day = match visit.departure_day {
            Some(day) => day,
            None => return false, // Can't check spread if no departure day
        };
        if voyage.visit_ids.iter().any(|&vid| self._visits[vid].installation_id() == installation_id) {
            return false;
        }
        // Get the installation to check spread
        let installation = match context.problem.get_installation_by_id(installation_id) {
            Some(inst) => inst,
            None => return false,
        };
        let spread = installation.departure_spread as i32;
        let period = crate::structs::constants::DAYS_IN_PERIOD as i32;
        // Check all other visits to this installation
        if let Some(departures) = self.schedule.departures_by_installation.get(&installation_id) {
            for &other_visit_id in departures {
                if other_visit_id == visit_id { continue; }
                if let Some(other_visit) = self.visit(other_visit_id) {
                    if let Some(other_day) = other_visit.departure_day {
                        // Cyclic difference
                        let diff = (departure_day as i32 - other_day as i32).abs().min(
                            period - (departure_day as i32 - other_day as i32).abs()
                        );
                        if diff < spread {
                            return false;
                        }
                    }
                }
            }
        }
        // Prevent inserting the same installation twice in the voyage
        // Check vessel capacity
        let vessel_id = match voyage.vessel_id {
            Some(id) => id,
            None => return false,
        };
        let vessel = match context.problem.vessels.get(vessel_id) {
            Some(v) => v,
            None => return false,
        };
        // TODO: Voyage's load is calculated from scratch, not from the current state. It prevents problems, but it is not the place to do.
        // I might change it to first check if it is updated, and if not, calculate it.
        let current_load: u32 = voyage.visit_ids.iter().map(|vid| self._visits[*vid].demand()).sum();
        let visit_demand = visit.demand();
        if (current_load + visit_demand) as f64 > vessel.deck_capacity {
            return false;
        }
        true
    }

    pub fn greedy_insert_visit(
        &mut self,
        visit_id: usize,
        voyage_id: usize,
        context: &Context,
    ) -> Result<(), String> {
        let voyage_cell = self
            .voyages
            .iter()
            .find(|v| v.borrow().id == voyage_id)
            .ok_or_else(|| format!("Voyage {} not found", voyage_id))?;

        {
            let mut voyage = voyage_cell.borrow_mut();
            let result = context.tsp_solver.evaluate_greedy_insertion(&*voyage, visit_id);
            voyage.apply_tsp_result(result);
            // Update the voyage load after insertion
            self.update_voyage_load(&mut voyage);
        }
        let visit = self.visit_mut(visit_id)
            .ok_or_else(|| format!("Visit {} not found", visit_id))?;
        visit.assign_to_voyage(voyage_id);
        self.schedule.set_need_update(true);
        Ok(())
    }

    pub fn optimal_insert_visit(
        &mut self,
        _visit_id: usize,
        _voyage_id: usize,
        _context: &Context,
    ) -> Result<(), String> {
        // let voyage_cell = self
        //     .voyages
        //     .iter()
        //     .find(|v| v.borrow().id == voyage_id)
        //     .ok_or_else(|| format!("Voyage {} not found", voyage_id))?;
        // let mut voyage = voyage_cell.borrow_mut();
        // voyage.add_visit(visit_id);
        // self.optimize_voyage_route(&mut *voyage, context);
        // self.update_voyage_load(&mut *voyage);
        // let visit = self.visit_mut(visit_id)
        //     .ok_or_else(|| format!("Visit {} not found", visit_id))?;
        // visit.assign_to_voyage(voyage_id);
        Ok(())
    }

    pub fn update_voyage_load(&self, voyage: &mut Voyage) {
        let mut total_load = 0;
        for visit_id in &voyage.visit_ids { 
            if let Some(visit) = self._visits.get(*visit_id) {
                total_load += visit.demand();
            }
        }
        voyage.finalize_state(total_load);
    }
    

    /// Returns the number of visits in the solution.
    pub fn visit_count(&self) -> usize {
        self._visits.len()
    }

    /// Returns a reference to the visit at the given id (index).
    pub fn visit(&self, id: usize) -> Option<&Visit> {
        self._visits.get(id)
    }

    /// Returns a mutable reference to the visit at the given id (index).
    pub fn visit_mut(&mut self, id: usize) -> Option<&mut Visit> {
        self._visits.get_mut(id)
    }

    /// Returns a slice of all visits (read-only).
    pub fn all_visits(&self) -> &[Visit] {
        &self._visits
    }

    /// Returns a mutable slice of all visits.
    pub fn all_visits_mut(&mut self) -> &mut [Visit] {
        &mut self._visits
    }

    // Ensure schedule is up-to-date before output/visualization
    pub fn get_schedule(&mut self, context: &Context) -> &Schedule {
        self.ensure_consistency_updated(context);
        &self.schedule
    }

    /// Adds one idle vessel (completely unused) and fills it with empty voyages for all days.
    /// Also fills all other vessels' free days with empty voyages.
    pub fn add_idle_vessel_and_add_empty_voyages(&mut self, context: &Context) {
        use crate::structs::constants::DAYS_IN_PERIOD;
        let mut vessel_has_voyage = vec![false; context.problem.vessels.len()];
        for voyage_cell in &self.voyages {
            let voyage = voyage_cell.borrow();
            if let Some(vessel_id) = voyage.vessel_id {
                if vessel_id < vessel_has_voyage.len() {
                    vessel_has_voyage[vessel_id] = true;
                }
            }
        }
        // Find one completely unused vessel
        let maybe_idle_vessel = vessel_has_voyage.iter().position(|&used| !used);
        if let Some(idle_vessel_id) = maybe_idle_vessel {
            for day in 0..DAYS_IN_PERIOD {
                let mut voyage = Voyage::empty();
                voyage.vessel_id = Some(idle_vessel_id);
                voyage.departure_day = Some(day as usize);
                if let Some(vessel) = context.problem.vessels.get(idle_vessel_id) {
                    voyage.voyage_speed = Some(vessel.speed);
                }
                voyage.sailing_time = Some(0.0);
                voyage.waiting_time = Some(0.0);
                let start_time = voyage.start_time().unwrap_or(0.0);
                voyage.arrival_time = Some(start_time);
                voyage.end_time_at_base = Some(start_time);
                voyage.load = Some(0);
                self.schedule.assign_voyage(&voyage, &self._visits);
                self.voyages.push(std::cell::RefCell::new(voyage));
            }
            vessel_has_voyage[idle_vessel_id] = true;
        }
        // For all other vessels, fill their free days with empty voyages
        for (vessel_id, &used) in vessel_has_voyage.iter().enumerate() {
            if !used { continue; } // skip the newly added idle vessel (already filled)
            let mut assigned_days = std::collections::HashSet::new();
            for day in 0..DAYS_IN_PERIOD {
                let day_usize = day as usize;
                if self.schedule.vessel_day_voyages.get(&(vessel_id, day_usize)).is_some() {
                    assigned_days.insert(day_usize);
                }
            }
            for day in 0..DAYS_IN_PERIOD {
                let day_usize = day as usize;
                if assigned_days.contains(&day_usize) {
                    continue;
                }
                let mut voyage = Voyage::empty();
                voyage.vessel_id = Some(vessel_id);
                voyage.departure_day = Some(day_usize);
                if let Some(vessel) = context.problem.vessels.get(vessel_id) {
                    voyage.voyage_speed = Some(vessel.speed);
                }
                voyage.sailing_time = Some(0.0);
                voyage.waiting_time = Some(0.0);
                let start_time = voyage.start_time().unwrap_or(0.0);
                voyage.arrival_time = Some(start_time);
                voyage.end_time_at_base = Some(start_time);
                voyage.load = Some(0);
                self.schedule.assign_voyage(&voyage, &self._visits);
                self.voyages.push(std::cell::RefCell::new(voyage));
            }
        }
        // It is redundant since we just filled the schedule, but it indicates that the schedule has empty voyages now.
        self.schedule.set_need_update(true);
    }

    /// Returns true if the voyage is an empty (idle) voyage (no visits).
    pub fn is_empty_voyage(voyage: &Voyage) -> bool {
        voyage.visit_ids.is_empty()
    }

    /// Returns true if the voyage with the given id is empty (no visits).
    pub fn is_empty_voyage_by_id(&self, voyage_id: usize) -> bool {
        // Try to find the voyage in the current solution
        if let Some(voyage) = self.voyages.iter().find_map(|v| {
            let v = v.borrow();
            if v.id == voyage_id { Some(v) } else { None }
        }) {
            return Self::is_empty_voyage(&voyage);
        }
        false
    }

    /// Checks for overlaps with real (non-empty) voyages only.
    pub fn overlaps_with_real_voyages(&self, vessel_id: usize, voyage_id: usize, start_time: f64, end_time: f64) -> bool {
        self.schedule.overlaps_with_other_voyages(
            vessel_id,
            voyage_id,
            start_time,
            end_time,
            Some(|id| self.is_empty_voyage_by_id(id)),
        )
    }

    /// Adds empty (idle) voyages for all vessels on their free days, and if there is a vessel with no assigned days, adds it as a new idle vessel.
    ///
    /// !!! TODO: Consider adding empty voyages only to days when vessel is free. Now it is added to all days with no departures.
    /// !!! TODO: Select vessels in a random order instead of iterating in order.
    pub fn add_idle_voyages_for_all_vessels(&mut self, context: &Context) {
        use crate::structs::constants::DAYS_IN_PERIOD;
        let mut vessel_has_voyage = vec![false; context.problem.vessels.len()];
        // Mark vessels that have at least one voyage assigned
        for voyage_cell in &self.voyages {
            let voyage = voyage_cell.borrow();
            if let Some(vessel_id) = voyage.vessel_id {
                if vessel_id < vessel_has_voyage.len() {
                    vessel_has_voyage[vessel_id] = true;
                }
            }
        }
        // Try to find one completely unused vessel
        let maybe_idle_vessel = vessel_has_voyage.iter().position(|&used| !used);
        if let Some(idle_vessel_id) = maybe_idle_vessel {
            // Add empty voyages for all days for this idle vessel
            for day in 0..DAYS_IN_PERIOD {
                let mut voyage = Voyage::empty();
                voyage.vessel_id = Some(idle_vessel_id);
                voyage.departure_day = Some(day as usize);
                if let Some(vessel) = context.problem.vessels.get(idle_vessel_id) {
                    voyage.voyage_speed = Some(vessel.speed);
                }
                voyage.sailing_time = Some(0.0);
                voyage.waiting_time = Some(0.0);
                let start_time = voyage.start_time().unwrap_or(0.0);
                voyage.arrival_time = Some(start_time);
                voyage.end_time_at_base = Some(start_time);
                voyage.load = Some(0);
                self.schedule.assign_voyage(&voyage, &self._visits);
                self.voyages.push(std::cell::RefCell::new(voyage));
            }
            vessel_has_voyage[idle_vessel_id] = true;
        }
        // For all other vessels, fill their free days with empty voyages
        for (vessel_id, &used) in vessel_has_voyage.iter().enumerate() {
            if !used { continue; } // skip the newly added idle vessel (already filled)
            let mut assigned_days = std::collections::HashSet::new();
            for day in 0..DAYS_IN_PERIOD {
                let day_usize = day as usize;
                if self.schedule.vessel_day_voyages.get(&(vessel_id, day_usize)).is_some() {
                    assigned_days.insert(day_usize);
                }
            }
            for day in 0..DAYS_IN_PERIOD {
                let day_usize = day as usize;
                if assigned_days.contains(&day_usize) {
                    continue;
                }
                let mut voyage = Voyage::empty();
                voyage.vessel_id = Some(vessel_id);
                voyage.departure_day = Some(day_usize);
                if let Some(vessel) = context.problem.vessels.get(vessel_id) {
                    voyage.voyage_speed = Some(vessel.speed);
                }
                voyage.sailing_time = Some(0.0);
                voyage.waiting_time = Some(0.0);
                let start_time = voyage.start_time().unwrap_or(0.0);
                voyage.arrival_time = Some(start_time);
                voyage.end_time_at_base = Some(start_time);
                voyage.load = Some(0);
                self.schedule.assign_voyage(&voyage, &self._visits);
                self.voyages.push(std::cell::RefCell::new(voyage));
            }
        }
        self.schedule.set_need_update(true);
    }
}