use crate::operators::traits::ImprovementOperator;
use crate::structs::{context::Context, solution::Solution};
use log::{debug, error, info, warn};
use rand::RngCore;

/// Relocates every assigned visit greedily while strict improvements exist.
pub struct DeepRelocation;

impl DeepRelocation {
	fn find_best_relocation(
		solution: &Solution,
		context: &Context,
	) -> Option<(usize, usize, usize, f64)> {
		let mut working_solution = solution.clone();
		working_solution.ensure_consistency_updated(context);
		let current_cost = working_solution.cost_with_context(context);
		let mut best_move: Option<(usize, usize, usize, f64)> = None;

		for visit in working_solution.all_visits() {
			if !visit.is_assigned {
				continue;
			}
			let visit_id = visit.id();
			let origin_voyage = match visit.assigned_voyage_id {
				Some(id) => id,
				None => continue,
			};

			for voyage_cell in &working_solution.voyages {
				let voyage = voyage_cell.borrow();
				let target_voyage_id = voyage.id;
				if target_voyage_id == origin_voyage {
					continue;
				}
				if voyage.is_empty() {
					// Allow relocation into empty voyages; keep iteration going.
				}
				drop(voyage);

				let mut candidate = working_solution.clone();
				candidate.unassign_visits(&[visit_id]);
				candidate.ensure_schedule_is_updated();
				if !candidate.visit_insertion_is_possible(context, visit_id, target_voyage_id) {
					continue;
				}
				if candidate
					.greedy_insert_visit(visit_id, target_voyage_id, context)
					.is_err()
				{
					continue;
				}
				candidate.ensure_consistency_updated(context);
				if !candidate.is_fully_feasible(context) {
					continue;
				}
				let candidate_cost = candidate.cost_with_context(context);
				let delta = candidate_cost - current_cost;
				if delta < -f64::EPSILON
					&& best_move
						.as_ref()
						.map(|(_, _, _, best_delta)| delta < *best_delta)
						.unwrap_or(true)
				{
					best_move = Some((visit_id, origin_voyage, target_voyage_id, delta));
				}
			}
		}

		best_move
	}
}

impl ImprovementOperator for DeepRelocation {
	fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
		info!(target: "operator::improvement", "[DeepRelocation] Invoked");
		solution.ensure_consistency_updated(context);
		let original_solution = solution.clone();
		let mut original_feasibility_probe = original_solution.clone();
		let original_was_feasible = original_feasibility_probe.is_fully_feasible(context);

		loop {
			// Save state before attempting move to enable clean rollback
			let iteration_start_solution = solution.clone();
			
			let baseline_cost = solution.cost_with_context(context);
			debug!(
				target: "operator::improvement",
				"[DeepRelocation] Baseline cost {:.2}",
				baseline_cost
			);

			let best_move = Self::find_best_relocation(solution, context);
			let Some((visit_id, origin_voyage, target_voyage, delta)) = best_move else {
				info!(
					target: "operator::improvement",
					"[DeepRelocation] No improving relocation found"
				);
				break;
			};

			info!(
				target: "operator::improvement",
				"[DeepRelocation] Relocate visit {} from voyage {} to {} (Δ={:.2})",
				visit_id,
				origin_voyage,
				target_voyage,
				delta
			);

			solution.unassign_visits(&[visit_id]);
			solution.ensure_schedule_is_updated();
			if !solution.visit_insertion_is_possible(context, visit_id, target_voyage) {
				debug!(
					target: "operator::improvement",
					"[DeepRelocation] Insertion of visit {} into voyage {} deemed infeasible; restoring iteration state",
					visit_id,
					target_voyage
				);
				*solution = iteration_start_solution;
				break;
			}
			if let Err(err) = solution.greedy_insert_visit(visit_id, target_voyage, context) {
				debug!(
					target: "operator::improvement",
					"[DeepRelocation] Failed to apply relocation: {}; restoring iteration state",
					err
				);
				*solution = iteration_start_solution;
				break;
			}

			solution.ensure_consistency_updated(context);
			let updated_cost = solution.cost_with_context(context);
			debug!(
				target: "operator::improvement",
				"[DeepRelocation] Cost reduced to {:.2}",
				updated_cost
			);
		}

		solution.ensure_consistency_updated(context);
		if !solution.is_fully_feasible(context) {
			if original_was_feasible {
				warn!(
					target: "operator::improvement",
					"[DeepRelocation] Operator produced infeasible candidate; reverting to previous solution"
				);
			} else {
				debug!(
					target: "operator::improvement",
					"[DeepRelocation] Candidate remained infeasible (input already infeasible); reverting"
				);
			}
			*solution = original_solution;
			solution.ensure_consistency_updated(context);

			let mut reverted_solution = solution.clone();
			if original_was_feasible && !reverted_solution.is_fully_feasible(context) {
				error!(
					target: "operator::improvement",
					"[DeepRelocation] Revert failed: restored solution is still infeasible"
				);
			}
		}
	}
}

