use crate::structs::{
    solution::Solution,
    visit::Visit,
    vessel::Vessel,
    voyage::Voyage,
    schedule::Schedule,
    context::Context,  
};
use crate::utils::assignment::assign_smallest_available_vessel;
use log::{debug, warn, error};
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

const MAX_VISITS_PER_VOYAGE: usize = 5;
const MAX_ATTEMPTS: usize = 10;
const MAX_VESSEL_CAPACITY: f64 = 100.0; // Define appropriate capacity value

pub fn construct_initial_solution(
    context: &Context,
    rng: &mut impl Rng,
) -> Solution {
    debug!(target: "operator::initial", "[InitialSolution] Construction started");
    let base_visits = context.problem.generate_visits();
    'outer: for attempt in 0..MAX_ATTEMPTS {
        let mut solution = Solution::new(base_visits.clone());
        let mut day_to_visits: HashMap<usize, Vec<usize>> = HashMap::new();
        // === 1. Randomly select one feasible departure scenario per installation ===
        for inst in &context.problem.installations {
            let mut inst_visits: Vec<(usize, &mut Visit)> = solution
                .all_visits_mut()
                .iter_mut()
                .enumerate()
                .filter(|(_, visit)| visit.installation_id() == inst.id)
                .collect();
            if !inst_visits.is_empty() {
                let visit_days = inst.generate_departure_scenario(rng);
                if visit_days.len() > inst_visits.len() {
                    panic!("More days than visits for installation {}", inst.id);
                }
                inst_visits.shuffle(rng);
                for (day, (i, visit)) in visit_days.into_iter().zip(inst_visits.into_iter()) {
                    visit.departure_day = Some(day);
                    day_to_visits.entry(day).or_default().push(i);
                }
            }
        }
        let min_vessel_capacity = context.problem.vessels.iter().map(|v| v.deck_capacity).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        // === 2. For each day, split visits into voyages ===
        let mut days: Vec<_> = day_to_visits.keys().cloned().collect();
        days.sort_unstable(); // Ensure deterministic order
        for day in days {
            let visit_indices = &day_to_visits[&day];
            let mut unassigned = visit_indices.clone();
            while !unassigned.is_empty() {
                let mut voyage_visits = Vec::new();
                let mut current_demand = 0.0;
                unassigned.shuffle(rng);
                for &visit_idx in &unassigned {
                    let visit = solution.visit(visit_idx).expect("Invalid visit index");
                    let demand = visit.demand() as f64;
                    if current_demand + demand <= min_vessel_capacity {
                        voyage_visits.push(visit_idx);
                        current_demand += demand;
                    }
                }
                if voyage_visits.is_empty() {
                    break;
                }
                // Remove used visits from unassigned in a deterministic way
                voyage_visits.sort_unstable();
                unassigned.retain(|i| !voyage_visits.contains(i));
                let mut voyage = Voyage::new_with_visit_ids(voyage_visits, Some(day));
                solution.update_voyage_load(&mut voyage);
                if assign_smallest_available_vessel(&mut voyage, context, &mut solution) {
                    solution.optimize_voyage_route(&mut voyage, context);
                    let cost = voyage.objective_cost(context);
                    debug!(target: "operator::initial", "[InitialSolution] Day {}, Voyage {} assigned vessel {:?}, cost={}", day, voyage.id, voyage.vessel_id, cost);
                    solution.add_voyage(voyage);
                } else {
                    warn!(target: "operator::initial", "[InitialSolution] Failed to assign vessel for voyage on day {}", day);
                    break 'outer;
                }
            }
            if !unassigned.is_empty() {
                warn!(target: "operator::initial", "[InitialSolution] Unassigned visits remain for day {}", day);
                continue 'outer;
            }
        }
        if solution.is_fully_feasible(context) {
            let num_voyages = solution.voyages.len();
            let mut vessels_used = std::collections::HashSet::new();
            for voyage in &solution.voyages {
                let voyage = voyage.borrow();
                if let Some(vessel_id) = voyage.vessel_id {
                    vessels_used.insert(vessel_id);
                }
            }
            let num_vessels_used = vessels_used.len();
            let is_complete = solution.is_complete_solution();
            let is_feasible = solution.is_fully_feasible(context);
            debug!(target: "operator::initial", "[InitialSolution] Created {} voyages, used {} vessels", num_voyages, num_vessels_used);
            debug!(target: "operator::initial", "[InitialSolution] Feasibility: complete={}, fully_feasible={}", is_complete, is_feasible);
            return solution;
        } else {
            warn!(target: "operator::initial", "[InitialSolution] Infeasible solution on attempt {}", attempt);
        }
    }
    error!(target: "operator::initial", "[InitialSolution] Failed to generate feasible initial solution after {} attempts", MAX_ATTEMPTS);
    panic!("Failed to generate feasible initial solution after {MAX_ATTEMPTS} attempts");
}

pub fn construct_greedy_initial_solution(visits: &[Visit], vessels: &[Vessel]) -> () {
    // Greedy initialization logic
    // e.g., assign visits to vessels based on some heuristic (e.g., nearest neighbor)
}