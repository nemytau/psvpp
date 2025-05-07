

use rand::RngCore;
use crate::structs::{context::Context, solution::Solution};
use crate::operators::traits::RepairOperator;

pub struct DeepGreedyInsertion;

impl RepairOperator for DeepGreedyInsertion {
    fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
        // let mut uninserted_visits = solution.get_unassigned_visits();
        // while !uninserted_visits.is_empty() {
        //     let mut best_visit = None;
        //     let mut best_cost = f64::INFINITY;
        //     let mut best_voyage = None;
        //     let mut best_position = 0;

        //     for &visit in &uninserted_visits {
        //         for (voyage_idx, voyage) in solution.voyages.iter_mut().enumerate() {
        //             let (pos, cost) = voyage.best_insertion(context, visit);  
        //             if cost < best_cost {
        //                 best_cost = cost;
        //                 best_visit = Some(visit);
        //                 best_voyage = Some(voyage_idx);
        //                 best_position = pos;
        //             }
        //         }
        //     }

        //     if let (Some(visit), Some(v_idx)) = (best_visit, best_voyage) {
        //         solution.insert_visit(v_idx, best_position, visit);
        //     } else {
        //         panic!("No insertion found for unassigned visits");
        //     }
        // }
    }

    fn requires_consistency(&self) -> bool {
        true
    }
}