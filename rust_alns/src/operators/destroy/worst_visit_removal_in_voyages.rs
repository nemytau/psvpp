use log::info;
use log::{debug};
use rand::RngCore;
use rand::Rng;
use crate::structs::{solution::Solution, context::Context};
use crate::operators::traits::DestroyOperator;

pub struct WorstVisitRemovalInVoyages {
    pub xi_min: f64,
    pub xi_max: f64,
    pub p: f64, // determinism parameter
}

impl DestroyOperator for WorstVisitRemovalInVoyages {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore) {
        info!(target: "operator::destroy", "[WorstVisitRemovalInVoyages] Invoked");
        // Determine number of visits to remove
        let n_visits = solution.voyages.iter().map(|v| v.borrow().visit_ids.len()).sum::<usize>();
        if n_visits == 0 {
            return;
        }
        let frac = self.xi_min + rng.gen_range(0.0..1.0) * (self.xi_max - self.xi_min);
        let to_remove = ((frac * n_visits as f64).round() as usize).min(n_visits);
        let mut removed_visit_ids = Vec::new();
        for _ in 0..to_remove {
            // 1. Collect all current visits and their removal costs
            let mut all_visits = Vec::new();
            for voyage in &solution.voyages {
                let voyage = voyage.borrow();
                for &visit_id in &voyage.visit_ids {
                    let cost_with = solution.cost_with_context(context);
                    let cost_without = solution.cost_without_visit_full(visit_id, context);
                    let removal_cost = cost_with - cost_without;
                    all_visits.push((visit_id, removal_cost));
                }
            }
            if all_visits.is_empty() { break; }
            // Sort by removal cost, descending
            all_visits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            // Select index using p-deterministic selection
            let r: f64 = rng.gen_range(0.0..1.0);
            let idx = ((r.powf(self.p)) * (all_visits.len() as f64)).floor() as usize;
            let idx = idx.min(all_visits.len() - 1);
            let (visit_id, removal_cost) = all_visits.remove(idx);
            removed_visit_ids.push(visit_id);
            // Remove the visit immediately
            solution.unassign_visits(&[visit_id]);
            solution.schedule.set_need_update(true);
        }
        removed_visit_ids.sort_unstable();
        info!(target: "operator::destroy", "[WorstVisitRemovalInVoyages] Completed");
    }
}
