mod alns;
mod operators;
mod structs;
mod utils;

use log::info;
use std::fs;
use std::time::Instant;

use crate::alns::acceptance;
use crate::alns::context::ALNSContext;
use crate::alns::legacy::ALNSEngineLegacy;
use crate::operators::repair::k_regret_insertion::KRegretInsertion;
use crate::operators::traits::{DestroyOperator, RepairOperator};
use crate::structs::context::Context;
use crate::structs::data_loader;
use crate::structs::problem_data::ProblemData;
use crate::utils::serialization::{dump_explicit_schedule_to_json, dump_schedule_to_json};
use crate::utils::tsp_solver::TSPSolver;
use operators::initial_solution::construct_initial_solution;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Run the ALNS engine with detailed logging and export iteration stats to CSV
fn run_alns_with_logging(seed: u64) -> Result<(), Box<dyn std::error::Error>> {
    // Prepare data and context (reuse from test_main)
    let installations_path = "../sample/installations/SMALL_1/i_test1.csv";
    let vessels_path = "../sample/vessels/SMALL_1/v_test1.csv";
    let base_path = "../sample/base/SMALL_1/b_test1.csv";

    let data = structs::data_loader::read_data(installations_path, vessels_path, base_path)?;
    let problem_data = structs::problem_data::ProblemData::new(
        data.vessels.clone(),
        data.installations.clone(),
        data.base.clone(),
    );
    let tsp_solver = utils::tsp_solver::TSPSolver::new_from_problem_data(&problem_data);
    let context = structs::context::Context {
        problem: problem_data,
        tsp_solver,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let initial_solution =
        operators::initial_solution::construct_initial_solution(&context, &mut rng);

    // Set up ALNS operator registry (add your operators here)
    let mut operator_registry = crate::operators::registry::OperatorRegistry::new();
    // Destroy operators
    operator_registry.add_destroy_operator(Box::new(
        crate::operators::destroy::shaw_removal::ShawRemoval {
            xi_min: 0.2,
            xi_max: 0.4,
            p: 5.0,
            alpha: 1.0,
            beta: 5.0,
            phi: 2.0,
        },
    ));
    operator_registry.add_destroy_operator(Box::new(
        crate::operators::destroy::random_visit_removal_in_voyages::RandomVisitRemovalInVoyages {
            xi_min: 0.2,
            xi_max: 0.4,
        },
    ));
    operator_registry.add_destroy_operator(Box::new(
        crate::operators::destroy::worst_visit_removal_in_voyages::WorstVisitRemovalInVoyages {
            xi_min: 0.2,
            xi_max: 0.4,
            p: 5.0,
        },
    ));
    // Repair operators
    operator_registry.add_repair_operator(Box::new(
        crate::operators::repair::deep_greedy_insertion::DeepGreedyInsertion {},
    ));
    operator_registry.add_repair_operator(Box::new(
        crate::operators::repair::k_regret_insertion::KRegretInsertion { k: 2 },
    ));
    operator_registry.add_repair_operator(Box::new(
        crate::operators::repair::k_regret_insertion::KRegretInsertion { k: 3 },
    ));
    // Add more operators as needed

    // ALNS parameters
    let temperature = 1000.0;
    let theta = 0.9;
    let max_iterations = 100;
    let reaction_factor = 0.2;
    let reward_values = vec![33.0, 9.0, 3.0]; // Example: [σ1, σ2, σ3]
    let n_destroy = operator_registry.destroy_operators.len();
    let n_repair = operator_registry.repair_operators.len();

    // Initialize ALNSContext
    let mut alns_context = ALNSContext {
        iteration: 0,
        temperature,
        best_cost: initial_solution.total_cost,
        rng: StdRng::seed_from_u64(seed),
        destroy_operator_weights: vec![1.0; n_destroy],
        repair_operator_weights: vec![1.0; n_repair],
        destroy_operator_scores: vec![0.0; n_destroy],
        repair_operator_scores: vec![0.0; n_repair],
        destroy_operator_counts: vec![0; n_destroy],
        repair_operator_counts: vec![0; n_repair],
        cost_history: Vec::new(),
        reaction_factor,
        reward_values,
    };

    let mut logger = crate::alns::logger::AlnsLogger::new();
    println!("Starting ALNS run with logging...");

    let mut engine = ALNSEngineLegacy::new(
        &context,
        &mut alns_context,
        operator_registry,
        initial_solution,
        temperature,
        theta,
        max_iterations,
        seed,
    );
    engine.run_with_logger(&mut logger);

    std::fs::create_dir_all("../logs").ok();
    logger.export_csv("../logs/alns_run.csv")?;
    println!("ALNS run complete. Logs exported to ../logs/alns_run.csv");
    Ok(())
}

fn test_main(seed: u64) -> Result<(), Box<dyn std::error::Error>> {
    let installations_path = "../sample/installations/SMALL_1/i_test1.csv";
    let vessels_path = "../sample/vessels/SMALL_1/v_test1.csv";
    let base_path = "../sample/base/SMALL_1/b_test1.csv";

    let data = data_loader::read_data(installations_path, vessels_path, base_path)?;
    let problem_data = ProblemData::new(
        data.vessels.clone(),
        data.installations.clone(),
        data.base.clone(),
    );
    let tsp_solver = TSPSolver::new_from_problem_data(&problem_data);
    let context = Context {
        problem: problem_data,
        tsp_solver,
    };
    // Use deterministic RNG with a fixed seed
    let mut rng = StdRng::seed_from_u64(seed);
    let mut solution = construct_initial_solution(&context, &mut rng);

    dump_solution(
        &solution,
        &context.problem.vessels,
        "../output/solution_vis.json",
        &context,
    )?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule.json")?;

    // Check initial solution feasibility
    let is_complete_initial = solution.is_complete_solution();
    let is_feasible_initial = solution.is_fully_feasible(&context);
    if !(is_complete_initial && is_feasible_initial) {
        println!(
            "Initial solution: complete={}, feasible={}",
            is_complete_initial, is_feasible_initial
        );
    }

    // Apply destroy operator
    let destroy_operator = crate::operators::destroy::shaw_removal::ShawRemoval {
        xi_min: 0.2,
        xi_max: 0.4,
        p: 5.0,
        alpha: 1.0,
        beta: 5.0,
        phi: 2.0,
    };
    destroy_operator.apply(&mut solution, &context, &mut rng);

    // Ensure consistency after destroy, before any further operations
    solution.ensure_consistency_updated(&context);

    // Feasibility check after destroy
    let is_complete_destroy = solution.is_complete_solution();
    let is_feasible_destroy = solution.is_fully_feasible(&context);
    if !(is_complete_destroy == false && is_feasible_destroy == true) {
        println!(
            "After destroy: complete={}, feasible={}",
            is_complete_destroy, is_feasible_destroy
        );
    }

    dump_solution(
        &solution,
        &context.problem.vessels,
        "../output/solution_vis_after_destroy.json",
        &context,
    )?;
    dump_explicit_solution(
        &solution,
        &context,
        "../output/explicit_schedule_after_destroy.json",
    )?;

    // Add idle voyages for only one vessel (after destroy)
    solution.add_idle_vessel_and_add_empty_voyages(&context);

    // Apply DeepGreedyInsertion repair operator
    use crate::operators::repair::k_regret_insertion::KRegretInsertion;
    let repair_operator = KRegretInsertion { k: 3 };
    repair_operator.apply(&mut solution, &context, &mut rng);

    solution.ensure_consistency_updated(&context);
    // Feasibility check after repair
    let is_complete_repair = solution.is_complete_solution();
    let is_feasible_repair = solution.is_fully_feasible(&context);
    if !(is_complete_repair && is_feasible_repair) {
        println!(
            "After repair: complete={}, feasible={}",
            is_complete_repair, is_feasible_repair
        );
    }

    dump_solution(
        &solution,
        &context.problem.vessels,
        "../output/solution_vis_after_repair.json",
        &context,
    )?;
    dump_explicit_solution(
        &solution,
        &context,
        "../output/explicit_schedule_after_repair.json",
    )?;

    Ok(())
}

pub fn dump_solution(
    solution: &structs::solution::Solution,
    vessels: &Vec<structs::vessel::Vessel>,
    path: &str,
    context: &Context,
) -> Result<(), Box<dyn std::error::Error>> {
    dump_schedule_to_json(solution, vessels, path, context);
    Ok(())
}

pub fn dump_explicit_solution(
    solution: &structs::solution::Solution,
    context: &Context,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    dump_explicit_schedule_to_json(solution, context, path);
    Ok(())
}

fn test_feasibility_over_seeds() -> Result<(), Box<dyn std::error::Error>> {
    let installations_path = "../sample/installations/SMALL_1/i_test1.csv";
    let vessels_path = "../sample/vessels/SMALL_1/v_test1.csv";
    let base_path = "../sample/base/SMALL_1/b_test1.csv";

    let data = structs::data_loader::read_data(installations_path, vessels_path, base_path)?;
    let problem_data = structs::problem_data::ProblemData::new(
        data.vessels.clone(),
        data.installations.clone(),
        data.base.clone(),
    );
    let tsp_solver = utils::tsp_solver::TSPSolver::new_from_problem_data(&problem_data);
    let context = structs::context::Context {
        problem: problem_data,
        tsp_solver,
    };

    let mut incomplete_initial = 0;
    let mut incomplete_initial_complete = 0;
    let mut incomplete_initial_feasible = 0;
    let mut incomplete_destroy = 0;
    let mut incomplete_destroy_complete = 0;
    let mut incomplete_destroy_feasible = 0;
    let mut incomplete_repair = 0;
    let mut incomplete_repair_complete = 0;
    let mut incomplete_repair_feasible = 0;
    let mut repair_incomplete_seeds = Vec::new();
    let mut repair_better_count = 0;
    let mut repair_better_and_feasible_count = 0;
    let total_seeds = 100;

    // Use ShawRemoval as destroy operator (copied from test_main)
    let destroy_operator = crate::operators::destroy::shaw_removal::ShawRemoval {
        xi_min: 0.2,
        xi_max: 0.4,
        p: 5.0,
        alpha: 1.0,
        beta: 5.0,
        phi: 2.0,
    };
    let repair_operator = KRegretInsertion { k: 3 };

    for seed in 0..100u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut solution =
            operators::initial_solution::construct_initial_solution(&context, &mut rng);
        let initial_cost = solution.total_cost;
        // Initial solution feasibility
        let is_complete_initial = solution.is_complete_solution();
        let is_feasible_initial = solution.is_fully_feasible(&context);
        if !is_complete_initial || !is_feasible_initial {
            incomplete_initial += 1;
            if !is_complete_initial {
                incomplete_initial_complete += 1;
            }
            if !is_feasible_initial {
                incomplete_initial_feasible += 1;
            }
        }
        // After destroy
        destroy_operator.apply(&mut solution, &context, &mut rng);
        // let destroy_cost = solution.total_cost; // Remove unused variable
        let is_complete_destroy = solution.is_complete_solution();
        let is_feasible_destroy = solution.is_fully_feasible(&context);
        if !is_complete_destroy || !is_feasible_destroy {
            incomplete_destroy += 1;
            if !is_complete_destroy {
                incomplete_destroy_complete += 1;
            }
            if !is_feasible_destroy {
                incomplete_destroy_feasible += 1;
            }
        }
        // After repair
        solution.ensure_consistency_updated(&context);
        // Add idle voyages for only one vessel (after destroy and after feasibility checks)
        solution.add_idle_vessel_and_add_empty_voyages(&context);
        repair_operator.apply(&mut solution, &context, &mut rng);
        let repair_cost = solution.total_cost;
        solution.ensure_consistency_updated(&context);
        let is_complete_repair = solution.is_complete_solution();
        let is_feasible_repair = solution.is_fully_feasible(&context);
        if !is_complete_repair || !is_feasible_repair {
            incomplete_repair += 1;
            if !is_complete_repair {
                incomplete_repair_complete += 1;
            }
            if !is_feasible_repair {
                incomplete_repair_feasible += 1;
            }
            repair_incomplete_seeds.push(seed);
        }
        if repair_cost < initial_cost {
            repair_better_count += 1;
            if is_complete_repair && is_feasible_repair {
                repair_better_and_feasible_count += 1;
            }
        }
    }
    println!("Feasibility stats over {} seeds:", total_seeds);
    println!(
        "Initial solution: {} not fully feasible ({} incomplete, {} not feasible)",
        incomplete_initial, incomplete_initial_complete, incomplete_initial_feasible
    );
    println!(
        "After destroy:    {} not fully feasible ({} incomplete, {} not feasible)",
        incomplete_destroy, incomplete_destroy_complete, incomplete_destroy_feasible
    );
    println!(
        "After repair:     {} not fully feasible ({} incomplete, {} not feasible)",
        incomplete_repair, incomplete_repair_complete, incomplete_repair_feasible
    );
    println!(
        "Seeds where repair operator led to infeasibility: {:?}",
        repair_incomplete_seeds
    );
    println!(
        "\nRepair operator produced BETTER cost than initial in {} out of {} seeds",
        repair_better_count, total_seeds
    );
    println!(
        "Repair operator produced BETTER cost AND feasible solution in {} out of {} seeds",
        repair_better_and_feasible_count, total_seeds
    );
    Ok(())
}

fn test_voyage_number_reduction_over_seeds() -> Result<(), Box<dyn std::error::Error>> {
    use crate::operators::improvement::voyage_number_reduction::VoyageNumberReduction;
    use crate::operators::traits::ImprovementOperator;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let installations_path = "../sample/installations/SMALL_1/i_test1.csv";
    let vessels_path = "../sample/vessels/SMALL_1/v_test1.csv";
    let base_path = "../sample/base/SMALL_1/b_test1.csv";

    let data = structs::data_loader::read_data(installations_path, vessels_path, base_path)?;
    let problem_data = structs::problem_data::ProblemData::new(
        data.vessels.clone(),
        data.installations.clone(),
        data.base.clone(),
    );
    let tsp_solver = utils::tsp_solver::TSPSolver::new_from_problem_data(&problem_data);
    let context = structs::context::Context {
        problem: problem_data,
        tsp_solver,
    };
    let mut reduced_seeds = Vec::new();
    let total_seeds = 100;
    for seed in 0..total_seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut solution =
            operators::initial_solution::construct_initial_solution(&context, &mut rng);
        let before = solution
            .voyages
            .iter()
            .filter(|v| !v.borrow().visit_ids.is_empty())
            .count();
        let op = VoyageNumberReduction;
        op.apply(&mut solution, &context, &mut rng);
        let after = solution
            .voyages
            .iter()
            .filter(|v| !v.borrow().visit_ids.is_empty())
            .count();
        if after < before {
            reduced_seeds.push(seed);
        }
    }
    let ratio = reduced_seeds.len() as f64 / total_seeds as f64;
    println!(
        "Ratio of seeds where number of voyages reduced: {:.2}",
        ratio
    );
    println!("Seeds where reduction occurred: {reduced_seeds:?}");
    Ok(())
}

fn test_voyage_number_reduction_dump(seed: u64) -> Result<(), Box<dyn std::error::Error>> {
    use crate::operators::improvement::voyage_number_reduction::VoyageNumberReduction;
    use crate::operators::traits::ImprovementOperator;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let installations_path = "../sample/installations/SMALL_1/i_test1.csv";
    let vessels_path = "../sample/vessels/SMALL_1/v_test1.csv";
    let base_path = "../sample/base/SMALL_1/b_test1.csv";

    let data = structs::data_loader::read_data(installations_path, vessels_path, base_path)?;
    let problem_data = structs::problem_data::ProblemData::new(
        data.vessels.clone(),
        data.installations.clone(),
        data.base.clone(),
    );
    let tsp_solver = utils::tsp_solver::TSPSolver::new_from_problem_data(&problem_data);
    let context = structs::context::Context {
        problem: problem_data,
        tsp_solver,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut solution = operators::initial_solution::construct_initial_solution(&context, &mut rng);
    dump_solution(
        &solution,
        &context.problem.vessels,
        &format!("../output/voyage_reduction_init_seed{}.json", seed),
        &context,
    )?;
    let op = VoyageNumberReduction;
    op.apply(&mut solution, &context, &mut rng);
    dump_solution(
        &solution,
        &context.problem.vessels,
        &format!("../output/voyage_reduction_improved_seed{}.json", seed),
        &context,
    )?;
    // Check if solution is feasible after reduction
    let is_complete = solution.is_complete_solution();
    let is_feasible = solution.is_fully_feasible(&context);
    if !is_complete || !is_feasible {
        println!(
            "After voyage reduction: complete={}, feasible={}",
            is_complete, is_feasible
        );
    } else {
        println!("After voyage reduction: solution is complete and feasible");
    }
    // Output departure spread for each installation
    Ok(())
}

/// Test the new unified ALNS engine
fn test_alns_engine_iterations() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing unified ALNS engine...");

    // Create engine using the new unified interface
    let mut engine = crate::alns::engine::ALNSEngine::new_from_instance(
        "SMALL_1", 42,    // seed
        500.0, // temperature
        0.9,   // theta
        10,    // weight_update_interval
        20,    // max_iterations
    )?;

    println!("Initial cost: {:.4}", engine.initial_cost);

    // Run 20 iterations manually to see progress
    for iteration in 0..20 {
        let destroy_idx = iteration % 3;
        let repair_idx = (iteration + 1) % 3;

        match engine.run_iteration_explicit(destroy_idx, repair_idx, None, iteration) {
            Ok(metrics) => {
                let weight_marker = if (iteration + 1) % engine.weight_update_interval == 0 {
                    " [WEIGHTS]"
                } else {
                    ""
                };
                println!(
                    "Iter {:2}: cost={:7.4}, best={:7.4}, temp={:6.1}, accepted={:5}{}",
                    iteration + 1,
                    metrics.total_cost,
                    metrics.best_cost,
                    metrics.temperature,
                    metrics.accepted,
                    weight_marker
                );
            }
            Err(e) => {
                println!("Error in iteration {}: {}", iteration, e);
                break;
            }
        }
    }

    println!("Final best cost: {:.4}", engine.best_solution.total_cost);

    // Export solution
    engine.export_solution("output/unified_engine_solution.json");
    println!("Solution exported to output/unified_engine_solution.json");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!(target: "alns::main", "ALNS process started");
    // test_feasibility_over_seeds()?;
    // Uncomment to run ALNS with detailed logging:
    // run_alns_with_logging(42)?;
    // test_voyage_number_reduction_over_seeds()?;
    // test_voyage_number_reduction_dump(43)?;
    // test_main(47)?; // Use a fixed seed for reproducibility
    test_alns_engine_iterations()?; // Test the new unified engine
    info!(target: "alns::main", "ALNS process finished");
    Ok(())
}
