use std::collections::HashSet;
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rust_alns_py::alns::engine::ALNSEngine;
use rust_alns_py::operators::improvement::fleet_and_cost_reduction::FleetAndCostReduction;
use rust_alns_py::operators::traits::ImprovementOperator;
use rust_alns_py::structs::solution::Solution;
use rust_alns_py::utils::serialization::dump_schedule_to_json;

fn used_vessel_ids(solution: &Solution) -> HashSet<usize> {
    let mut ids = HashSet::new();
    for voyage_cell in &solution.voyages {
        let voyage = voyage_cell.borrow();
        if voyage.visit_ids.is_empty() {
            continue;
        }
        if let Some(vessel_id) = voyage.vessel_id {
            ids.insert(vessel_id);
        }
    }
    ids
}

#[test]
fn fleet_reduction_merges_vessels_and_exports_snapshots() {
    let mut engine = ALNSEngine::new_from_instance("SMALL_1", 7, 100.0, 0.9, 10, 1)
        .expect("failed to construct deterministic engine");
    let context = engine.context.clone();

    let mut manipulated_solution = engine.current_solution.clone();
    manipulated_solution.ensure_consistency_updated(&context);
    manipulated_solution.update_total_cost(&context);

    let used_before_setup = used_vessel_ids(&manipulated_solution);
    assert!(
        !used_before_setup.is_empty(),
        "expected at least one active vessel"
    );

    let donor_vessel_id = *used_before_setup
        .iter()
        .min()
        .expect("missing donor vessel");
    let free_vessel_id = (0..context.problem.vessels.len())
        .find(|id| !used_before_setup.contains(id))
        .expect("dataset must contain an idle vessel for the test scenario");

    let mut donor_voyage_index: Option<usize> = None;
    for (idx, voyage_cell) in manipulated_solution.voyages.iter().enumerate() {
        let voyage = voyage_cell.borrow();
        if !voyage.visit_ids.is_empty() && voyage.vessel_id == Some(donor_vessel_id) {
            donor_voyage_index = Some(idx);
            break;
        }
    }
    let donor_voyage_index =
        donor_voyage_index.expect("donor vessel should own at least one voyage");

    {
        let mut voyage = manipulated_solution.voyages[donor_voyage_index].borrow_mut();
        voyage.vessel_id = Some(free_vessel_id);
    }
    manipulated_solution.schedule.set_need_update(true);
    manipulated_solution.ensure_consistency_updated(&context);
    manipulated_solution.update_total_cost(&context);

    assert!(
        manipulated_solution.get_unassigned_visits().is_empty(),
        "setup must keep all visits assigned"
    );
    let before_cost = manipulated_solution.cost_with_context(&context);
    let before_used_vessels = used_vessel_ids(&manipulated_solution);
    assert!(
        before_used_vessels.contains(&free_vessel_id),
        "manipulation should activate the spare vessel"
    );

    let solution_before = manipulated_solution.clone();

    let mut improved_solution = manipulated_solution.clone();
    let mut rng = StdRng::seed_from_u64(1_234_567);
    FleetAndCostReduction.apply(&mut improved_solution, &context, &mut rng);
    improved_solution.ensure_consistency_updated(&context);
    improved_solution.update_total_cost(&context);

    assert!(
        improved_solution.get_unassigned_visits().is_empty(),
        "repair step must leave the solution complete"
    );

    let after_cost = improved_solution.cost_with_context(&context);
    assert!(
        after_cost + 1e-6 < before_cost,
        "cost should decrease after fleet reduction, before={before_cost}, after={after_cost}"
    );

    let before_vessels = used_vessel_ids(&solution_before);
    let after_vessels = used_vessel_ids(&improved_solution);
    assert!(
        after_vessels.len() < before_vessels.len(),
        "expected vessel usage to drop, before={before_vessels:?}, after={after_vessels:?}"
    );

    let mut feasibility_probe = improved_solution.clone();
    assert!(
        feasibility_probe.is_fully_feasible(&context),
        "improved solution must remain feasible"
    );

    let output_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("output/tests");
    std::fs::create_dir_all(&output_root).expect("failed to create snapshot directory");
    let before_path = output_root.join("fleet_reduction_before.json");
    let after_path = output_root.join("fleet_reduction_after.json");

    dump_schedule_to_json(
        &solution_before,
        &context.problem.vessels,
        before_path
            .to_str()
            .expect("snapshot path must be valid UTF-8"),
        &context,
    );
    dump_schedule_to_json(
        &improved_solution,
        &context.problem.vessels,
        after_path
            .to_str()
            .expect("snapshot path must be valid UTF-8"),
        &context,
    );
}
