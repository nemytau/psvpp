use core::panic;
use log::{info, debug, warn, error};
use rand::RngCore;
use crate::structs::{context::Context, solution::Solution};
use crate::operators::traits::RepairOperator;

pub struct DeepGreedyInsertion;

impl RepairOperator for DeepGreedyInsertion {
    fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
        info!(target: "operator::repair", "[DeepGreedyInsertion] Invoked");
        if !solution.is_schedule_up_to_date() {
            info!(target: "operator::repair", "Solution schedule is not up-to-date before operator application, updating now.");
            solution.ensure_schedule_is_updated();
        }
        let mut uninserted_visits: Vec<usize> = solution.get_unassigned_visits().iter().map(|v| v.id()).collect();
        let mut iteration = 0;
        debug!(target: "operator::repair", "[DeepGreedyInsertion] Starting with {} uninserted visits", uninserted_visits.len());
        while !uninserted_visits.is_empty() {
            let mut best_insertion: Option<(usize, usize, f64)> = None;
            for &visit_id in &uninserted_visits {
                // top_k_visit_insertion_costs already ensures the visit is not in the voyage
                let costs = solution.top_k_visit_insertion_costs(context, visit_id, 3); // log top 3 for more info
                if let Some((voyage_id, cost)) = costs.iter().cloned().next() {
                    if best_insertion.is_none() || cost < best_insertion.as_ref().unwrap().2 {
                        best_insertion = Some((visit_id, voyage_id, cost));
                    }
                }
            }

            if let Some((visit_id, voyage_id, cost)) = best_insertion {
                debug!(target: "operator::repair", "[DeepGreedyInsertion] Iteration {}: Chosen insertion: visit_id={}, voyage_id={}, cost={}", iteration, visit_id, voyage_id, cost);
                if let Some(voyage_cell) = solution.voyages.iter().find(|v| v.borrow().id == voyage_id) {
                    let voyage = voyage_cell.borrow();
                    debug!(target: "operator::repair", "  Target voyage: id={}, vessel_id={:?}, start_time={:?}, end_time={:?}, visit_ids={:?}",
                        voyage.id, voyage.vessel_id, voyage.start_time(), voyage.end_time(), voyage.visit_ids);
                }
                let possible = solution.visit_insertion_is_possible(context, visit_id, voyage_id);
                debug!(target: "operator::repair", "  visit_insertion_is_possible for visit {} into voyage {}: {}", visit_id, voyage_id, possible);
                if solution.greedy_insert_visit(visit_id, voyage_id, context).is_ok() {
                    // TODO: Instead of fully rebuilding the schedule, we could just update the affected voyage
                    solution.ensure_schedule_is_updated(); // Ensure schedule is up-to-date before next iteration
                    uninserted_visits.retain(|&v_id| v_id != visit_id);
                } else {
                    error!(target: "operator::repair", "Failed to insert visit {} into voyage {}", visit_id, voyage_id);
                    break;
                }
            } else {
                warn!(target: "operator::repair", "No feasible insertions left at iteration {}", iteration);
                break;
            }
            iteration += 1;
        }
        if uninserted_visits.is_empty() {
            info!(target: "operator::repair", "[DeepGreedyInsertion] All visits inserted successfully");
        } else {
            warn!(target: "operator::repair", "[DeepGreedyInsertion] Some visits could not be inserted: {:?}", uninserted_visits);
        }
        // After all modifications, update the schedule for consistency
        if solution.schedule.needs_update() {
            solution.ensure_schedule_is_updated();
        }
        info!(target: "operator::repair", "[DeepGreedyInsertion] Completed");
    }
}