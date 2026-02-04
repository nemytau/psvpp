use crate::operators::traits::ImprovementOperator;
use crate::structs::{context::Context, solution::Solution};
use log::{debug, error, info, warn};
use rand::RngCore;

/// Swaps pairs of assigned visits greedily while strict improvements exist.
pub struct DeepSwap;

impl DeepSwap {
	fn find_best_swap(
		solution: &Solution,
		context: &Context,
	) -> Option<(usize, usize, usize, usize, f64)> {
		let mut working_solution = solution.clone();
		working_solution.ensure_consistency_updated(context);
		let current_cost = working_solution.cost_with_context(context);
		let mut best_move: Option<(usize, usize, usize, usize, f64)> = None;

		let visits = working_solution.all_visits();
		for (idx_a, visit_a) in visits.iter().enumerate() {
			if !visit_a.is_assigned {
				continue;
			}
			let Some(origin_voyage_a) = visit_a.assigned_voyage_id else {
				continue;
			};
			for visit_b in visits.iter().skip(idx_a + 1) {
				if !visit_b.is_assigned {
					continue;
				}
				if visit_a.installation_id() == visit_b.installation_id() {
					continue;
				}
				let Some(origin_voyage_b) = visit_b.assigned_voyage_id else {
					continue;
				};

				let visit_a_id = visit_a.id();
				let visit_b_id = visit_b.id();

				let mut candidate = working_solution.clone();
				candidate.unassign_visits(&[visit_a_id, visit_b_id]);
				candidate.ensure_schedule_is_updated();

				if !candidate.visit_insertion_is_possible(context, visit_a_id, origin_voyage_b) {
					continue;
				}
				if !candidate.visit_insertion_is_possible(context, visit_b_id, origin_voyage_a) {
					continue;
				}
				if candidate
					.greedy_insert_visit(visit_a_id, origin_voyage_b, context)
					.is_err()
				{
					continue;
				}
				if candidate
					.greedy_insert_visit(visit_b_id, origin_voyage_a, context)
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
						.map(|(_, _, _, _, best_delta)| delta < *best_delta)
						.unwrap_or(true)
				{
					best_move = Some((visit_a_id, origin_voyage_a, visit_b_id, origin_voyage_b, delta));
				}
			}
		}

		best_move
	}
}

impl ImprovementOperator for DeepSwap {
	fn apply(&self, solution: &mut Solution, context: &Context, _rng: &mut dyn RngCore) {
		info!(target: "operator::improvement", "[DeepSwap] Invoked");
		solution.ensure_consistency_updated(context);
		let original_solution = solution.clone();

		loop {
			let baseline_cost = solution.cost_with_context(context);
			debug!(
				target: "operator::improvement",
				"[DeepSwap] Baseline cost {:.2}",
				baseline_cost
			);

			let best_move = Self::find_best_swap(solution, context);
			let Some((visit_a, origin_voyage_a, visit_b, origin_voyage_b, delta)) = best_move else {
				info!(
					target: "operator::improvement",
					"[DeepSwap] No improving swap found"
				);
				break;
			};

			info!(
				target: "operator::improvement",
				"[DeepSwap] Swap visit {} (voyage {}) with visit {} (voyage {}) (Δ={:.2})",
				visit_a,
				origin_voyage_a,
				visit_b,
				origin_voyage_b,
				delta
			);

			solution.unassign_visits(&[visit_a, visit_b]);
			solution.ensure_schedule_is_updated();
			if !solution.visit_insertion_is_possible(context, visit_a, origin_voyage_b)
				|| !solution.visit_insertion_is_possible(context, visit_b, origin_voyage_a)
			{
				warn!(
					target: "operator::improvement",
					"[DeepSwap] Swap feasibility check failed for visits {} and {}",
					visit_a,
					visit_b
				);
				if let Err(revert_err) = solution.greedy_insert_visit(visit_a, origin_voyage_a, context) {
					warn!(
						target: "operator::improvement",
						"[DeepSwap] Rollback failed for visit {}: {}",
						visit_a,
						revert_err
					);
				}
				if let Err(revert_err) = solution.greedy_insert_visit(visit_b, origin_voyage_b, context) {
					warn!(
						target: "operator::improvement",
						"[DeepSwap] Rollback failed for visit {}: {}",
						visit_b,
						revert_err
					);
				}
				solution.ensure_consistency_updated(context);
				break;
			}
			if let Err(err) = solution.greedy_insert_visit(visit_a, origin_voyage_b, context) {
				warn!(
					target: "operator::improvement",
					"[DeepSwap] Failed to insert visit {} into voyage {}: {}",
					visit_a,
					origin_voyage_b,
					err
				);
				let _ = solution.greedy_insert_visit(visit_a, origin_voyage_a, context);
				let _ = solution.greedy_insert_visit(visit_b, origin_voyage_b, context);
				solution.ensure_consistency_updated(context);
				break;
			}
			if let Err(err) = solution.greedy_insert_visit(visit_b, origin_voyage_a, context) {
				warn!(
					target: "operator::improvement",
					"[DeepSwap] Failed to insert visit {} into voyage {}: {}",
					visit_b,
					origin_voyage_a,
					err
				);
				if let Err(revert_err) = solution.greedy_insert_visit(visit_b, origin_voyage_b, context) {
					warn!(
						target: "operator::improvement",
						"[DeepSwap] Rollback failed for visit {}: {}",
						visit_b,
						revert_err
					);
				}
				let _ = solution.greedy_insert_visit(visit_a, origin_voyage_a, context);
				solution.ensure_consistency_updated(context);
				break;
			}

			solution.ensure_consistency_updated(context);
			let updated_cost = solution.cost_with_context(context);
			debug!(
				target: "operator::improvement",
				"[DeepSwap] Cost reduced to {:.2}",
				updated_cost
			);
		}

		solution.ensure_consistency_updated(context);
		if !solution.is_fully_feasible(context) {
			error!(
				target: "operator::improvement",
				"[DeepSwap] Operator produced infeasible solution; reverting"
			);
			*solution = original_solution;
			solution.ensure_consistency_updated(context);
		}
	}
}

