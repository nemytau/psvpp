use crate::operators::traits::RepairOperator;
use crate::structs::{context::Context, solution::Solution};
use log::{debug, error, info, warn};
use rand::RngCore;

pub struct KRegretInsertion {
    pub k: usize,
}

impl RepairOperator for KRegretInsertion {
    fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
        info!(target: "operator::repair", "[KRegretInsertion] Invoked with k={}", self.k);
        if !solution.is_schedule_up_to_date() {
            solution.ensure_schedule_is_updated();
        }
        let mut uninserted_visits: Vec<usize> = solution
            .get_unassigned_visits()
            .iter()
            .map(|v| v.id())
            .collect();
        let mut iteration = 0;
        debug!(target: "operator::repair", "[KRegretInsertion] Starting with {} uninserted visits", uninserted_visits.len());
        while !uninserted_visits.is_empty() {
            let mut best_visit: Option<usize> = None;
            let mut best_voyage: Option<usize> = None;
            let mut max_regret = f64::NEG_INFINITY;

            for &visit_id in &uninserted_visits {
                let costs = solution.top_k_visit_insertion_costs(context, visit_id, self.k);
                if costs.len() < self.k {
                    continue;
                }
                let regret = costs[self.k - 1].1 - costs[0].1;
                if regret > max_regret {
                    max_regret = regret;
                    best_visit = Some(visit_id);
                    best_voyage = Some(costs[0].0);
                }
            }
            if let (Some(visit_id), Some(voyage_id)) = (best_visit, best_voyage) {
                debug!(target: "operator::repair", "[KRegretInsertion] Iteration {}: Chosen insertion: visit_id={}, voyage_id={}, regret={:.2}", iteration, visit_id, voyage_id, max_regret);
                if let Some(voyage_cell) =
                    solution.voyages.iter().find(|v| v.borrow().id == voyage_id)
                {
                    let voyage = voyage_cell.borrow();
                    debug!(target: "operator::repair", "  Target voyage: id={}, vessel_id={:?}, start_time={:?}, end_time={:?}, visit_ids={:?}",
                        voyage.id, voyage.vessel_id, voyage.start_time(), voyage.end_time(), voyage.visit_ids);
                }
                let possible = solution.visit_insertion_is_possible(context, visit_id, voyage_id);
                debug!(target: "operator::repair", "  visit_insertion_is_possible for visit {} into voyage {}: {}", visit_id, voyage_id, possible);
                if solution
                    .greedy_insert_visit(visit_id, voyage_id, context)
                    .is_ok()
                {
                    solution.ensure_schedule_is_updated();
                    uninserted_visits.retain(|&v| v != visit_id);
                } else {
                    log::error!(target: "operator::repair", "Failed to insert visit {} into voyage {}", visit_id, voyage_id);
                    break;
                }
            } else {
                log::warn!(target: "operator::repair", "No feasible insertions left at iteration {}", iteration);
                break;
            }
            iteration += 1;
        }
        if uninserted_visits.is_empty() {
            info!(target: "operator::repair", "[KRegretInsertion] All visits inserted successfully");
        } else {
            warn!(target: "operator::repair", "[KRegretInsertion] Some visits could not be inserted: {:?}", uninserted_visits);
        }
        if solution.schedule.needs_update() {
            solution.ensure_schedule_is_updated();
        }
        debug!(target: "operator::repair", "[KRegretInsertion] Completed");
    }

    fn requires_consistency(&self) -> bool {
        true
    }
}
