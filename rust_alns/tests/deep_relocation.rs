use rand::rngs::StdRng;
use rand::SeedableRng;
use rust_alns_py::alns::engine::{ALNSEngine, ALNSAlgorithmMode};
use rust_alns_py::operators::improvement::deep_relocation::DeepRelocation;
use rust_alns_py::operators::traits::ImprovementOperator;
use rust_alns_py::structs::solution::Solution;

fn introduce_worse_relocation(solution: &Solution, context: &rust_alns_py::structs::context::Context) -> Solution {
    let mut baseline = solution.clone();
    baseline.ensure_consistency_updated(context);
    let baseline_cost = baseline.cost_with_context(context);

    let mut selected: Option<Solution> = None;

    for origin_cell in &baseline.voyages {
        let origin = origin_cell.borrow();
        if origin.visit_ids.len() <= 1 {
            continue;
        }
        for &visit_id in &origin.visit_ids {
            for target_cell in &baseline.voyages {
                let target = target_cell.borrow();
                if target.id == origin.id || target.is_empty() {
                    continue;
                }
                let target_id = target.id;
                drop(target);
                let mut modified = baseline.clone();
                modified.unassign_visits(&[visit_id]);
                if modified
                    .greedy_insert_visit(visit_id, target_id, context)
                    .is_err()
                {
                    continue;
                }
                modified.ensure_consistency_updated(context);
                let cost = modified.cost_with_context(context);
                if cost > baseline_cost + 1e-6 {
                    selected = Some(modified);
                    break;
                }
            }
            if selected.is_some() {
                break;
            }
        }
        if selected.is_some() {
            break;
        }
    }

    selected.expect("failed to build degraded solution for relocation test")
}

#[test]
fn deep_relocation_improves_cost() {
    let mut engine = ALNSEngine::new_from_instance(
        "SMALL_1",
        7,
        100.0,
        0.9,
        10,
        1,
        ALNSAlgorithmMode::Baseline,
    )
        .expect("failed to construct deterministic engine");
    let context = engine.context.clone();

    let worse_solution = introduce_worse_relocation(&engine.current_solution, &context);
    let worse_cost = worse_solution.cost_with_context(&context);

    let baseline_cost = engine.current_solution.cost_with_context(&context);
    assert!(
        worse_cost > baseline_cost + 1e-6,
        "expected manipulated solution to have higher cost"
    );

    let mut improved = worse_solution.clone();
    let mut rng = StdRng::seed_from_u64(42);
    DeepRelocation.apply(&mut improved, &context, &mut rng);
    improved.ensure_consistency_updated(&context);
    let improved_cost = improved.cost_with_context(&context);

    assert!(
        improved_cost + 1e-6 < worse_cost,
        "deep relocation should improve the manipulated solution"
    );
    assert!(
        improved.is_fully_feasible(&context),
        "improved solution must remain feasible"
    );
}
