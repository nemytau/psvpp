

use rand::RngCore;
use crate::structs::{context::Context, solution::Solution};
use crate::operators::traits::RepairOperator;

pub struct KRegretInsertion {
    pub k: usize,
}

impl RepairOperator for KRegretInsertion {
    fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
        // let mut uninserted_visits = solution.get_uninserted_visits();

        // while !uninserted_visits.is_empty() {
        //     let mut best_visit = None;
        //     let mut best_voyage = None;
        //     let mut best_position = 0;
        //     let mut max_regret = f64::NEG_INFINITY;

        //     for &visit in &uninserted_visits {
        //         let mut costs = vec![];

        //         for (voyage_idx, voyage) in solution.voyages_mut().iter_mut().enumerate() {
        //             for pos in 0..=voyage.len() {
        //                 let cost = voyage.insertion_cost(context, visit, pos);
        //                 costs.push((cost, voyage_idx, pos));
        //             }
        //         }

        //         costs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        //         if costs.len() < self.k {
        //             continue;
        //         }

        //         let regret = costs[self.k - 1].0 - costs[0].0;
        //         if regret > max_regret {
        //             max_regret = regret;
        //             best_visit = Some(visit);
        //             best_voyage = Some(costs[0].1);
        //             best_position = costs[0].2;
        //         }
        //     }

        //     if let (Some(visit), Some(v_idx)) = (best_visit, best_voyage) {
        //         solution.insert_visit(v_idx, best_position, visit);
        //         uninserted_visits.retain(|&v| v != visit);
        //     } else {
        //         break;
        //     }
        // }
    }

    fn requires_consistency(&self) -> bool {
        true
    }
}