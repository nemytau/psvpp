use crate::structs::{
    solution::Solution,
    visit::Visit,
    vessel::Vessel,
    voyage::Voyage,
    schedule::Schedule,
    context::Context,  
};
use crate::utils::assignment::assign_smallest_available_vessel;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;

const MAX_VISITS_PER_VOYAGE: usize = 5;
const MAX_ATTEMPTS: usize = 1;
const MAX_VESSEL_CAPACITY: f64 = 100.0; // Define appropriate capacity value

pub fn construct_initial_solution(
    context: &Context,
    rng: &mut impl Rng,
) -> Solution {
    // Base meaning basic???
    let mut base_visits = context.problem.generate_visits();
    'outer: for attempt in 0..MAX_ATTEMPTS {   
        let mut solution = Solution::new(base_visits.clone());
        let mut day_to_visits: HashMap<usize, Vec<usize>> = HashMap::new();
        
        // === 1. Randomly select one feasible departure scenario per installation ===
        for inst in &context.problem.installations {
            // Collect all visit indices and mutable references for this installation (not just those matching index == inst.id)
            let mut inst_visits: Vec<(usize, &mut Visit)> = solution
                .visits
                .iter_mut()
                .enumerate()
                .filter(|(_, visit)| visit.installation_id() == inst.id)
                .collect();

            if !inst_visits.is_empty() {
                let visit_days = inst.generate_departure_scenario(rng);

                // Make sure we don't assign more days than visits
                if visit_days.len() > inst_visits.len() {
                    panic!("More days than visits for installation {}", inst.id);
                }

                // Randomly shuffle visits to vary distribution
                inst_visits.shuffle(rng);

                for (day, (i, visit)) in visit_days.into_iter().zip(inst_visits.into_iter()) {
                    visit.departure_day = Some(day);
                    day_to_visits.entry(day).or_default().push(i);
                }
            }
        }

        // Probably it is needed to recalculated after the assignment, not critical
        let MIN_VESSEL_CAPACITY = context.problem.vessels.iter().map(|v| v.deck_capacity).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        // === 2. For each day, split visits into voyages ===
        for (day, visit_indices) in day_to_visits {
            let mut unassigned = visit_indices.clone();
            
            while !unassigned.is_empty() {
                let mut voyage_visits = Vec::new();
                let mut current_demand = 0.0;

                unassigned.shuffle(rng);
                for &visit_idx in &unassigned {
                    let visit = &solution.visits[visit_idx];
                    let demand = visit.demand() as f64;

                    if current_demand + demand <= MIN_VESSEL_CAPACITY {
                        voyage_visits.push(visit_idx);
                        current_demand += demand;
                    }

                    // Optional: break early if we've reached "enough" load (e.g., 80%)
                }
                
                if voyage_visits.is_empty() {
                    // Couldn't pack any visit — fallback logic, or break
                    break;
                }

                // Remove used visits from unassigned
                unassigned.retain(|i| !voyage_visits.contains(i));

                let mut voyage = Voyage::new_with_visit_ids(voyage_visits, Some(day));
                // In this approach update_load and optimize_route filter all visits to find visits belonging to voyage, so it's done twice unnecessarily
                // however, it is not costly. 
                // EDIT: Even trice.
                voyage.update_load(&solution.visits);
                if assign_smallest_available_vessel(&mut voyage, context, &mut solution) {
                    solution.optimize_voyage_route(&mut voyage, context);
                    solution.add_voyage(voyage);
                } else {
                    break 'outer; // Retry whole solution construction
                }
            }

            // If unassigned visits remain, not enough vessels, next attempt
            // Kind of redundant, because if some visits are unassigned, the schedule is not feasible
            // Or we were just unlucky?
            if !unassigned.is_empty() {
                continue 'outer;
            }
        }

        // === 5. Finalize solution ===
        // Check if solution is feasible
        if solution.is_feasible() {
            // Optionally, the solution can be optimized further here
            return solution;
        }
    }

    panic!("Failed to generate feasible initial solution after {MAX_ATTEMPTS} attempts");
}

pub fn construct_greedy_initial_solution(visits: &[Visit], vessels: &[Vessel]) -> () {
    // Greedy initialization logic
    // e.g., assign visits to vessels based on some heuristic (e.g., nearest neighbor)
}