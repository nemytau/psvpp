use rand::rngs::StdRng;
use rand::SeedableRng;
use rust_alns_py::alns::engine::{ALNSEngine, ALNSAlgorithmMode};
use rust_alns_py::operators::improvement::deep_swap::DeepSwap;
use rust_alns_py::operators::traits::ImprovementOperator;
use rust_alns_py::structs::solution::Solution;

fn introduce_worse_swap(solution: &Solution, context: &rust_alns_py::structs::context::Context) -> Solution {
    let mut baseline = solution.clone();
    baseline.ensure_consistency_updated(context);
    let baseline_cost = baseline.cost_with_context(context);

    let visits = baseline.all_visits();

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

            let mut modified = baseline.clone();
            modified.unassign_visits(&[visit_a_id, visit_b_id]);
            modified.ensure_schedule_is_updated();
            if !modified.visit_insertion_is_possible(context, visit_a_id, origin_voyage_b) {
                continue;
            }
            if !modified.visit_insertion_is_possible(context, visit_b_id, origin_voyage_a) {
                continue;
            }
            if modified
                .greedy_insert_visit(visit_a_id, origin_voyage_b, context)
                .is_err()
            {
                continue;
            }
            if modified
                .greedy_insert_visit(visit_b_id, origin_voyage_a, context)
                .is_err()
            {
                continue;
            }
            modified.ensure_consistency_updated(context);
            if !modified.is_fully_feasible(context) {
                continue;
            }
            let candidate_cost = modified.cost_with_context(context);
            if candidate_cost > baseline_cost + 1e-6 {
                return modified;
            }
        }
    }

    panic!("failed to build degraded solution for swap test");
}

#[test]
fn deep_swap_improves_cost() {
    let mut engine = ALNSEngine::new_from_instance(
        "SMALL_1",
        7,
        100.0,
        0.9,
        10,
        1,
        ALNSAlgorithmMode::Baseline,
    )
    .expect("Failed to initialize engine");
    let context = engine.context.clone();

    let worse_solution = introduce_worse_swap(&engine.current_solution, &context);
    let worse_cost = worse_solution.cost_with_context(&context);

    let baseline_cost = engine.current_solution.cost_with_context(&context);
    assert!(
        worse_cost > baseline_cost + 1e-6,
        "expected manipulated solution to have higher cost"
    );

    let mut improved = worse_solution.clone();
    let mut rng = StdRng::seed_from_u64(42);
    DeepSwap.apply(&mut improved, &context, &mut rng);
    improved.ensure_consistency_updated(&context);
    let improved_cost = improved.cost_with_context(&context);

    assert!(
        improved_cost + 1e-6 < worse_cost,
        "deep swap should improve the manipulated solution"
    );
    assert!(
        improved.is_fully_feasible(&context),
        "improved solution must remain feasible"
    );
}
