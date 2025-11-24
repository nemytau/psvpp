use crate::structs::context::Context;
use crate::structs::problem_data::ProblemData;
use crate::structs::solution::Solution;
use crate::structs::vessel::Vessel;
use crate::structs::voyage::Voyage;

/// Finds and assigns the smallest available vessel for the given voyage.
/// Conditions to meet:
/// 1. Enough capacity
/// 2. Vessel is available: no overlaps with current assignment
/// This function performs the assignment directly if a suitable vessel is found.
/// Returns true if assignment succeeded.
pub fn assign_smallest_available_vessel(
    voyage: &mut Voyage,
    context: &Context,
    solution: &mut Solution,
) -> bool {
    let load = voyage.load().unwrap();
    let mut vessels = context.problem.vessels.iter().collect::<Vec<_>>();
    vessels.sort_by(|a, b| a.deck_capacity.partial_cmp(&b.deck_capacity).unwrap());

    let start_time = voyage.start_time().unwrap();
    let tsp_solver = &context.tsp_solver;
    let schedule = &solution.schedule;

    for vessel in vessels {
        if vessel.deck_capacity < load as f64 {
            continue; // not enough capacity
        }
        let end_time =
            tsp_solver.solve_and_get_end_time(&voyage.visit_ids, vessel.speed, start_time);

        if schedule.is_vessel_available_for_period(vessel.id(), start_time, end_time) {
            // Perform assignment here: modifies the voyage directly
            voyage.vessel_id = Some(vessel.id);
            voyage.voyage_speed = Some(vessel.speed);

            // Vessel has been successfully assigned
            return true;
        }
    }

    // No vessel could be assigned
    false
}
