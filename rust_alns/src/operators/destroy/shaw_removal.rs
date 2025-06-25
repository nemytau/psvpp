use log::info;
use log::{debug, warn};
use rand::RngCore;
use rand::Rng;
use crate::structs::{solution::Solution, context::Context};
use crate::operators::traits::DestroyOperator;

pub struct ShawRemoval {
    pub xi_min: f64,
    pub xi_max: f64,
    pub p: f64, // determinism parameter
    pub alpha: f64, // weight for travel time
    pub beta: f64,  // weight for arrival time difference
    pub phi: f64,   // weight for load difference
}

impl DestroyOperator for ShawRemoval {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore) {
        info!(target: "operator::destroy", "[ShawRemoval] Invoked");
        // Collect all visit ids
        let mut all_visit_ids = Vec::new();
        for voyage in &solution.voyages {
            let voyage = voyage.borrow();
            for &visit_id in &voyage.visit_ids {
                all_visit_ids.push(visit_id);
            }
        }
        let n_visits = all_visit_ids.len();
        if n_visits == 0 {
            warn!(target: "operator::destroy", "No visits to remove");
            return;
        }
        let frac = self.xi_min + rng.gen_range(0.0..1.0) * (self.xi_max - self.xi_min);
        let to_remove = ((frac * n_visits as f64).round() as usize).min(n_visits);
        debug!(target: "operator::destroy", "Removing {} of {} visits (frac={:.2})", to_remove, n_visits, frac);
        debug!(target: "operator::destroy", "[ShawRemoval] Candidates: {:?}", all_visit_ids);
        // Randomly select a seed visit
        let mut removed_visit_ids = Vec::new();
        let mut candidates: Vec<usize> = all_visit_ids.clone();
        let seed_idx = rng.gen_range(0..candidates.len());
        let seed = candidates.remove(seed_idx);
        debug!(target: "operator::destroy", "[ShawRemoval] Seed visit: {}", seed);
        removed_visit_ids.push(seed);
        // Iteratively select most related visits
        let mut iteration = 0;
        while removed_visit_ids.len() < to_remove && !candidates.is_empty() {
            // For each candidate, compute min relatedness to any already removed
            let mut relatedness: Vec<(usize, f64)> = candidates.iter().map(|&j| {
                let r = removed_visit_ids.iter().map(|&i| {
                    shaw_relatedness(i, j, solution, context, self.alpha, self.beta, self.phi)
                }).fold(f64::INFINITY, f64::min);
                (j, r)
            }).collect();
            // Sort by relatedness (lower is more related)
            relatedness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            // p-deterministic selection
            let r: f64 = rng.gen_range(0.0..1.0);
            let idx = ((r.powf(self.p)) * (relatedness.len() as f64)).floor() as usize;
            let idx = idx.min(relatedness.len() - 1);
            let (selected, rel_val) = relatedness.remove(idx);
            debug!(target: "operator::destroy", "[ShawRemoval] Iteration {}: Removing visit {} (relatedness={:.2})", iteration, selected, rel_val);
            // Remove from candidates and add to removed
            if let Some(pos) = candidates.iter().position(|&x| x == selected) {
                candidates.remove(pos);
            }
            removed_visit_ids.push(selected);
            iteration += 1;
        }
        debug!(target: "operator::destroy", "[ShawRemoval] Removed visits: {:?}", removed_visit_ids);
        removed_visit_ids.sort_unstable();
        solution.unassign_visits(&removed_visit_ids);
        solution.schedule.set_need_update(true);
        info!(target: "operator::destroy", "[ShawRemoval] Completed");
    }
}

// Helper: compute relatedness between two visits
fn shaw_relatedness(i: usize, j: usize, solution: &Solution, context: &Context, alpha: f64, beta: f64, phi: f64) -> f64 {
    // Get visits
    let visit_i = solution.visit(i).expect("Invalid visit id i");
    let visit_j = solution.visit(j).expect("Invalid visit id j");
    // Get installation ids
    let inst_i = visit_i.installation_id();
    let inst_j = visit_j.installation_id();
    // Travel time between installations (use distance as proxy, assuming unit speed)
    let t_ij = context.problem.distance_manager.distance(inst_i, inst_j);
    // Arrival times (if available, else 0.0)
    // Use departure_day as a proxy for arrival time
    let t_i = visit_i.departure_day.unwrap_or(0) as f64;
    let t_j = visit_j.departure_day.unwrap_or(0) as f64;
    // Delivery load (deck demand)
    let q_i = visit_i.demand() as f64;
    let q_j = visit_j.demand() as f64;

    alpha * t_ij + beta * ((t_i - t_j) as f64).abs() + phi * ((q_i - q_j) as f64).abs()
}
