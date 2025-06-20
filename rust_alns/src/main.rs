mod operators;
mod structs;
mod utils;

use log::info;

use operators::initial_solution::construct_initial_solution;
use crate::operators::repair::k_regret_insertion::KRegretInsertion;
use crate::operators::destroy::worst_visit_removal_in_voyages::WorstVisitRemovalInVoyages;
use crate::structs::context::Context;
use crate::structs::data_loader;
use crate::structs::problem_data::ProblemData;
use crate::utils::tsp_solver::TSPSolver;
use rand::SeedableRng;
use rand::rngs::StdRng;
use crate::operators::traits::{DestroyOperator, RepairOperator};
use crate::utils::serialization::{dump_schedule_to_json, dump_explicit_schedule_to_json};

fn test_main(seed: u64) -> Result<(), Box<dyn std::error::Error>> {
    let installations_path = "../sample/installations/SMALL_1/i_test1.csv";
    let vessels_path = "../sample/vessels/SMALL_1/v_test1.csv";
    let base_path = "../sample/base/SMALL_1/b_test1.csv";

    let data = data_loader::read_data(installations_path, vessels_path, base_path)?;
    let problem_data = ProblemData::new(data.vessels.clone(), data.installations.clone(), data.base.clone());
    let tsp_solver = TSPSolver::new_from_problem_data(&problem_data);
    let context = Context {
        problem: problem_data,
        tsp_solver,
    };
    // Use deterministic RNG with a fixed seed
    let mut rng = StdRng::seed_from_u64(seed);
    let mut solution = construct_initial_solution(&context, &mut rng);

    dump_solution(&solution, &context.problem.vessels, "../output/solution_vis.json", &context)?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule.json")?;

    // Check initial solution feasibility
    let is_complete_initial = solution.is_complete_solution();
    let is_feasible_initial = solution.is_fully_feasible(&context);
    if !(is_complete_initial && is_feasible_initial) {
        println!("Initial solution: complete={}, feasible={}", is_complete_initial, is_feasible_initial);
    }

    // Apply destroy operator
    let destroy_operator = WorstVisitRemovalInVoyages { xi_min: 0.2, xi_max: 0.8, p: 5.0 };
    destroy_operator.apply(&mut solution, &context, &mut rng);

    // Ensure consistency after destroy, before any further operations
    solution.ensure_consistency_updated(&context);

    // Feasibility check after destroy
    let is_complete_destroy = solution.is_complete_solution();
    let is_feasible_destroy = solution.is_fully_feasible(&context);
    if !(is_complete_destroy == false && is_feasible_destroy == true) {
        println!("After destroy: complete={}, feasible={}", is_complete_destroy, is_feasible_destroy);
    }

    dump_solution(&solution, &context.problem.vessels, "../output/solution_vis_after_destroy.json", &context)?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule_after_destroy.json")?;

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
        println!("After repair: complete={}, feasible={}", is_complete_repair, is_feasible_repair);
    }

    dump_solution(&solution, &context.problem.vessels, "../output/solution_vis_after_repair.json", &context)?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule_after_repair.json")?;

    Ok(())
}

pub fn dump_solution(solution: &structs::solution::Solution, vessels: &Vec<structs::vessel::Vessel>, path: &str, context: &Context) -> Result<(), Box<dyn std::error::Error>> {
    dump_schedule_to_json(solution, vessels, path, context);
    Ok(())
}

pub fn dump_explicit_solution(solution: &structs::solution::Solution, context: &Context, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    dump_explicit_schedule_to_json(solution, context, path);
    Ok(())
}

fn test_feasibility_over_seeds() -> Result<(), Box<dyn std::error::Error>> {
    let installations_path = "../sample/installations/SMALL_1/i_test1.csv";
    let vessels_path = "../sample/vessels/SMALL_1/v_test1.csv";
    let base_path = "../sample/base/SMALL_1/b_test1.csv";

    let data = structs::data_loader::read_data(installations_path, vessels_path, base_path)?;
    let problem_data = structs::problem_data::ProblemData::new(data.vessels.clone(), data.installations.clone(), data.base.clone());
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

    let destroy_operator = WorstVisitRemovalInVoyages { xi_min: 0.2, xi_max: 0.8, p: 5.0 };
    let repair_operator = KRegretInsertion { k: 3 };

    for seed in 0..100u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut solution = operators::initial_solution::construct_initial_solution(&context, &mut rng);
        let initial_cost = solution.total_cost;
        // Initial solution feasibility
        let is_complete_initial = solution.is_complete_solution();
        let is_feasible_initial = solution.is_fully_feasible(&context);
        if !is_complete_initial || !is_feasible_initial {
            incomplete_initial += 1;
            if !is_complete_initial { incomplete_initial_complete += 1; }
            if !is_feasible_initial { incomplete_initial_feasible += 1; }
        }
        // After destroy
        destroy_operator.apply(&mut solution, &context, &mut rng);
        // let destroy_cost = solution.total_cost; // Remove unused variable
        let is_complete_destroy = solution.is_complete_solution();
        let is_feasible_destroy = solution.is_fully_feasible(&context);
        if !is_complete_destroy || !is_feasible_destroy {
            incomplete_destroy += 1;
            if !is_complete_destroy { incomplete_destroy_complete += 1; }
            if !is_feasible_destroy { incomplete_destroy_feasible += 1; }
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
            if !is_complete_repair { incomplete_repair_complete += 1; }
            if !is_feasible_repair { incomplete_repair_feasible += 1; }
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
    println!("Initial solution: {} not fully feasible ({} incomplete, {} not feasible)", incomplete_initial, incomplete_initial_complete, incomplete_initial_feasible);
    println!("After destroy:    {} not fully feasible ({} incomplete, {} not feasible)", incomplete_destroy, incomplete_destroy_complete, incomplete_destroy_feasible);
    println!("After repair:     {} not fully feasible ({} incomplete, {} not feasible)", incomplete_repair, incomplete_repair_complete, incomplete_repair_feasible);
    println!("Seeds where repair operator led to infeasibility: {:?}", repair_incomplete_seeds);
    println!("\nRepair operator produced BETTER cost than initial in {} out of {} seeds", repair_better_count, total_seeds);
    println!("Repair operator produced BETTER cost AND feasible solution in {} out of {} seeds", repair_better_and_feasible_count, total_seeds);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!(target: "alns::main", "ALNS process started");
    test_feasibility_over_seeds()?;
    test_main(89)?; // Use a fixed seed for reproducibility
    info!(target: "alns::main", "ALNS process finished");
    Ok(())
}