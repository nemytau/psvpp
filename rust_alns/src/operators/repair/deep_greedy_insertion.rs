use core::panic;
use rand::RngCore;
use crate::structs::{context::Context, solution::Solution};
use crate::operators::traits::RepairOperator;

pub struct DeepGreedyInsertion;

impl RepairOperator for DeepGreedyInsertion {
    fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
        // Always ensure schedule is up-to-date before any cost-based insertion logic.
        solution.ensure_schedule_is_updated();
        let mut uninserted_visits: Vec<usize> = solution.get_unassigned_visits().iter().map(|v| v.id()).collect();
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
                // --- LOGGING START ---
                println!("[DeepGreedyInsertion] Chosen insertion: visit_id={}, voyage_id={}, cost={}", visit_id, voyage_id, cost);
                if let Some(voyage_cell) = solution.voyages.iter().find(|v| v.borrow().id == voyage_id) {
                    let voyage = voyage_cell.borrow();
                    println!("  Target voyage: id={}, vessel_id={:?}, start_time={:?}, end_time={:?}, visit_ids={:?}",
                        voyage.id, voyage.vessel_id, voyage.start_time(), voyage.end_time(), voyage.visit_ids);
                    if let Some(vessel_id) = voyage.vessel_id {
                        let other_voyages: Vec<_> = solution.voyages.iter()
                            .filter_map(|v| {
                                let v = v.borrow();
                                if v.vessel_id == Some(vessel_id) && v.id != voyage_id {
                                    Some((v.id, v.start_time(), v.end_time()))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        println!("  Other voyages for vessel {}:", vessel_id);
                        for (oid, st, et) in &other_voyages {
                            println!("    voyage_id={}, start_time={:?}, end_time={:?}", oid, st, et);
                        }
                    }
                }
                let possible = solution.visit_insertion_is_possible(context, visit_id, voyage_id);
                println!("  visit_insertion_is_possible for visit {} into voyage {}: {}", visit_id, voyage_id, possible);
                // --- LOGGING END ---
                if solution.greedy_insert_visit(visit_id, voyage_id, context).is_ok() {
                    solution.ensure_schedule_is_updated(); // Ensure schedule is up-to-date before next iteration
                    uninserted_visits.retain(|&v_id| v_id != visit_id);
                } else {
                    panic!("Failed to insert visit {} into voyage {}", visit_id, voyage_id);
                }
            } else {
                break; // No feasible insertions left
            }
        }
        // After all modifications, update the schedule for consistency
        if solution.schedule.needs_update() {
            solution.ensure_schedule_is_updated();
        }
    }

    fn requires_consistency(&self) -> bool {
        true
    }
}