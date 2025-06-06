use log::{info, debug, warn, error};
use rand::seq::SliceRandom;
use rand::RngCore;
use rand::Rng;
use crate::structs::{solution::Solution, context::Context};
use crate::operators::traits::DestroyOperator;

pub struct RandomVisitRemovalInVoyages {
    pub xi_min: f64,
    pub xi_max: f64,
}

impl DestroyOperator for RandomVisitRemovalInVoyages {
    fn apply(&self, solution: &mut Solution, _context: &Context, rng: &mut dyn RngCore) {
        info!(target: "operator::destroy", "[RandomVisitRemovalInVoyages] Invoked");
        let voyage_count = solution.voyages.len();
        if voyage_count == 0 {
            warn!(target: "operator::destroy", "No voyages to affect");
            return;
        }
        // Calculate the number of voyages to affect based on xi_max
        let num_to_affect = ((self.xi_max * voyage_count as f64).ceil() as usize).min(voyage_count);

        let mut indices: Vec<usize> = (0..voyage_count).collect();
        indices.shuffle(rng);
        // Select a subset of voyages to affect
        for &i in indices.iter().take(num_to_affect) {
            let voyage = solution.voyages[i].borrow_mut();
            let n_visits = voyage.visit_ids.len();
            let frac = self.xi_min + rng.gen::<f64>() * (self.xi_max - self.xi_min);
            let to_remove = ((frac * n_visits as f64).round() as usize).min(n_visits);
            debug!(target: "operator::destroy", "Voyage {}: removing {} of {} visits (frac={:.2})", voyage.id, to_remove, n_visits, frac);
            // Ensure deterministic removal: sort visit IDs before unassigning
            let mut removed_visit_ids = voyage.select_visits_to_remove(to_remove, rng);
            removed_visit_ids.sort_unstable();
            drop(voyage); // Explicitly drop borrow before mutating solution
            solution.unassign_visits(&removed_visit_ids);
        }
        // Mark schedule as needing update after any removals
        solution.schedule.set_need_update(true);
        info!(target: "operator::destroy", "[RandomVisitRemovalInVoyages] Completed");
    }
}