use crate::operators::traits::RepairOperator;
use crate::structs::{context::Context, solution::Solution};
use log::{debug, info, warn};
use rand::seq::SliceRandom;
use rand::RngCore;

/// Cheap, non-cost-ranked repair operator: for each unassigned visit, try
/// voyages in a randomly shuffled order and insert into the first one that's
/// feasible -- no per-candidate cost computation, unlike DeepGreedyInsertion/
/// KRegretInsertion (which rank every feasible voyage by insertion cost).
///
/// Existing (non-empty) voyages are shuffled and tried first; only if none of
/// them can accept the visit does it fall back to the empty voyage slots that
/// `add_idle_vessel_and_add_empty_voyages` (called every iteration, before
/// repair runs) guarantees are already present -- one idle vessel's worth of
/// empty voyages across the full period, plus any free days on already-used
/// vessels. This keeps the operator's default behavior conservative about
/// growing the fleet (existing capacity is always tried first) while still
/// allowing it when nothing existing fits.
///
/// The per-call random order is deliberate: a fixed-order first-fit would
/// behave like a near-deterministic operator once repeatedly retried from an
/// unchanged base solution (see this project's finding that near-deterministic
/// destroy operators get stuck in exactly this way, while stochastic ones
/// keep proposing genuinely different candidates on each retry).
pub struct FirstFitInsertion;

impl RepairOperator for FirstFitInsertion {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore) {
        info!(target: "operator::repair", "[FirstFitInsertion] Invoked");
        if !solution.is_schedule_up_to_date() {
            solution.ensure_schedule_is_updated();
        }

        let unassigned_visit_ids: Vec<usize> = solution
            .get_unassigned_visits()
            .iter()
            .map(|v| v.id())
            .collect();
        debug!(
            target: "operator::repair",
            "[FirstFitInsertion] Starting with {} uninserted visits",
            unassigned_visit_ids.len()
        );

        let mut non_empty_voyage_ids: Vec<usize> = Vec::new();
        let mut empty_voyage_ids: Vec<usize> = Vec::new();
        for voyage_cell in &solution.voyages {
            let voyage = voyage_cell.borrow();
            if voyage.is_empty() {
                empty_voyage_ids.push(voyage.id);
            } else {
                non_empty_voyage_ids.push(voyage.id);
            }
        }
        non_empty_voyage_ids.shuffle(rng);
        empty_voyage_ids.shuffle(rng);

        let mut still_unassigned: Vec<usize> = Vec::new();

        for visit_id in unassigned_visit_ids {
            let mut inserted = false;
            for &voyage_id in non_empty_voyage_ids.iter().chain(empty_voyage_ids.iter()) {
                if !solution.visit_insertion_is_possible(context, visit_id, voyage_id) {
                    continue;
                }
                // visit_insertion_is_possible alone is not sufficient: it doesn't account for
                // the voyage's *new* timing after this insertion clashing with another voyage
                // already scheduled on the same vessel. top_k_visit_insertion_costs (used by
                // DeepGreedyInsertion/KRegretInsertion) guards against this with
                // overlaps_with_real_voyages before accepting a candidate -- replicate that
                // check here too, or this operator can silently produce an infeasible solution
                // that the engine has no mechanism to reject or revert once accepted.
                let overlap_check = {
                    let voyage_cell = solution
                        .voyages
                        .iter()
                        .find(|v| v.borrow().id == voyage_id);
                    let Some(voyage_cell) = voyage_cell else {
                        continue;
                    };
                    let voyage = voyage_cell.borrow();
                    let Some(vessel_id) = voyage.vessel_id else {
                        continue;
                    };
                    let tsp_result = context.tsp_solver.evaluate_greedy_insertion(&*voyage, visit_id);
                    let new_start = voyage.start_time().unwrap_or(0.0);
                    let new_end = tsp_result.end_time;
                    solution.overlaps_with_real_voyages(vessel_id, voyage_id, new_start, new_end)
                };
                if overlap_check {
                    continue;
                }
                if solution
                    .greedy_insert_visit(visit_id, voyage_id, context)
                    .is_ok()
                {
                    debug!(
                        target: "operator::repair",
                        "[FirstFitInsertion] Inserted visit {} into voyage {}",
                        visit_id, voyage_id
                    );
                    // TODO: Instead of fully rebuilding the schedule, we could just update the affected voyage
                    solution.ensure_schedule_is_updated();
                    inserted = true;
                    break;
                }
            }
            if !inserted {
                warn!(
                    target: "operator::repair",
                    "[FirstFitInsertion] No feasible voyage found for visit {}",
                    visit_id
                );
                still_unassigned.push(visit_id);
            }
        }

        if still_unassigned.is_empty() {
            info!(target: "operator::repair", "[FirstFitInsertion] All visits inserted successfully");
        } else {
            warn!(
                target: "operator::repair",
                "[FirstFitInsertion] Some visits could not be inserted: {:?}",
                still_unassigned
            );
        }

        if solution.schedule.needs_update() {
            solution.ensure_schedule_is_updated();
        }
        info!(target: "operator::repair", "[FirstFitInsertion] Completed");
    }
}
