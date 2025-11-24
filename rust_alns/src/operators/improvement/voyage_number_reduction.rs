// Implements the voyage number reduction improvement operator
// See: user prompt for algorithm description

use crate::operators::traits::ImprovementOperator;
use crate::structs::{context::Context, solution::Solution};
use rand::RngCore;

pub struct VoyageNumberReduction;

impl ImprovementOperator for VoyageNumberReduction {
    fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
        // Ensure schedule is up-to-date
        solution.ensure_schedule_is_updated();
        loop {
            // 1. For each voyage, try to relocate all its visits to other voyages using deep greedy insertion
            let mut best_voyage: Option<(usize, f64, Vec<(usize, usize)>)> = None; // (voyage_id, cost_increase, relocations)
            let mut any_found = false;
            for voyage_cell in &solution.voyages {
                let voyage = voyage_cell.borrow();
                if voyage.visit_ids.is_empty() {
                    continue;
                }
                let voyage_id = voyage.id;
                let visit_ids = voyage.visit_ids.clone();
                drop(voyage);
                // Try to relocate all visits
                let mut temp_solution = solution.clone();
                let mut relocations = Vec::new();
                let mut feasible = true;
                temp_solution.unassign_visits(&visit_ids);
                temp_solution.ensure_schedule_is_updated(); // <-- Fix: update schedule after unassign
                for &visit_id in &visit_ids {
                    // Ensure schedule is up-to-date before each insertion cost calculation
                    temp_solution.ensure_schedule_is_updated();
                    // Find best feasible insertion using deep greedy insertion logic
                    let mut best_insertion: Option<(usize, f64)> = None;
                    for voyage_cell2 in &temp_solution.voyages {
                        let v2 = voyage_cell2.borrow();
                        let v2_id = v2.id;
                        if v2_id == voyage_id {
                            continue;
                        }
                        if temp_solution.visit_insertion_is_possible(context, visit_id, v2_id) {
                            let costs =
                                temp_solution.top_k_visit_insertion_costs(context, visit_id, 1);
                            if let Some((_, cost)) = costs.first() {
                                if best_insertion.is_none()
                                    || *cost < best_insertion.as_ref().unwrap().1
                                {
                                    best_insertion = Some((v2_id, *cost));
                                }
                            }
                        }
                    }
                    if let Some((target_voyage, _)) = best_insertion {
                        // Insert visit into target voyage
                        if temp_solution
                            .greedy_insert_visit(visit_id, target_voyage, context)
                            .is_ok()
                        {
                            relocations.push((visit_id, target_voyage));
                        } else {
                            feasible = false;
                            break;
                        }
                    } else {
                        feasible = false;
                        break;
                    }
                }
                if feasible {
                    any_found = true;
                    temp_solution.ensure_consistency_updated(context);
                    let cost_increase = temp_solution.cost_with_context(context)
                        - solution.cost_with_context(context);
                    if best_voyage.is_none() || cost_increase < best_voyage.as_ref().unwrap().1 {
                        best_voyage = Some((voyage_id, cost_increase, relocations));
                    }
                }
            }
            // 2. If no voyage can be removed, break
            if !any_found || best_voyage.is_none() {
                break;
            }
            // 3. Apply the best relocation to the real solution
            let (_voyage_id, _cost_increase, relocations) = best_voyage.unwrap();
            // Remove all visits from the voyage
            let visit_ids: Vec<usize> = relocations.iter().map(|(visit_id, _)| *visit_id).collect();
            solution.unassign_visits(&visit_ids);
            // Insert each visit into its new voyage
            for (visit_id, target_voyage) in relocations {
                let _ = solution.greedy_insert_visit(visit_id, target_voyage, context);
            }
            // Remove empty voyages and update consistency
            solution.ensure_consistency_updated(context);
        }
    }
}
