use rand::RngCore;
use crate::structs::{context::Context, solution::Solution};
use crate::operators::traits::RepairOperator;

pub struct KRegretInsertion {
    pub k: usize,
}

impl RepairOperator for KRegretInsertion {
    fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
        if !solution.is_schedule_up_to_date() {
            solution.ensure_schedule_is_updated();
        }
        let mut uninserted_visits: Vec<usize> = solution.get_unassigned_visits().iter().map(|v| v.id()).collect();
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
                if solution.greedy_insert_visit(visit_id, voyage_id, context).is_ok() {
                    solution.ensure_schedule_is_updated();
                    uninserted_visits.retain(|&v| v != visit_id);
                } else {
                    break;
                }
            } else {
                // No valid insertions with at least k options, stop
                log::warn!("[KRegretInsertion] No feasible solution found with KRegretInsertion: no valid insertions with at least k={} options", self.k);
                break;
            }
        }
    }

    fn requires_consistency(&self) -> bool {
        true
    }
}