mod operators;
mod structs;
mod utils;

use log::{info, debug, warn, error};
use rand::SeedableRng;
use rand::rngs::StdRng;


use operators::initial_solution::construct_initial_solution;
use crate::operators::destroy::random_visit_removal_in_voyages::RandomVisitRemovalInVoyages;
use crate::operators::traits::DestroyOperator;
use structs::context::Context;
use structs::data_loader;
use structs::distance_manager::DistanceManager;
use structs::node::HasLocation;
use structs::problem_data::ProblemData;
use utils::serialization::{dump_explicit_schedule_to_json, dump_schedule_to_json};
use utils::tsp_solver::TSPSolver;
use crate::structs::solution::Solution;

fn debug_main() ->  Result<(), Box<dyn std::error::Error>> {
    // Define the file paths
    let installations_path = "../sample/installations/SMALL_1/i_test1.csv";
    let vessels_path = "../sample/vessels/SMALL_1/v_test1.csv";
    let base_path = "../sample/base/SMALL_1/b_test1.csv";

    // Load the data
    let data = data_loader::read_data(installations_path, vessels_path, base_path)?;

    // Print the data to check the output
    println!("Installations:");
    for installation in &data.installations {
        println!("{:?}", installation);
    }

    println!("\nVessels:");
    for vessel in &data.vessels {
        println!("{:?}", vessel);
    }

    println!("\nBase:");
    println!("{:?}", data.base);

    let mut dm = DistanceManager::new(data.installations.len() + 1);
    let installations: Vec<&dyn HasLocation> = data.installations.iter().map(|i| i as &dyn HasLocation).collect();
    dm.calculate_distances(&data.base, &installations);
    
    // Test TSP solver on route nodes : 1, 2, 3
    // println!("Testing TSP solver:");
    
    // Create time windows and service times for each node
    // For simplicity, let's use: 
    // - Depot (0): No time window constraint, 0 service time
    // - Node 1: Time window (8, 16), 2 hrs service time
    // - Node 2: Time window (10, 18), 3 hrs service time  
    // - Node 3: Time window (8, 14), 1 hr service time
    let _time_windows = vec![
        (0.0, 24.0),   // Depot - always open
        (8.0, 16.0),   // Node 1
        (10.0, 18.0),  // Node 2
        (8.0, 14.0),   // Node 3
    ];
    
    let _service_times = vec![
        0.0,  // Depot
        2.0,  // Node 1
        3.0,  // Node 2
        1.0,  // Node 3
    ];
    Ok(())
}   

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

    dump_solution(&solution, &context.problem.vessels, "../output/solution_vis.json")?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule.json")?;

    // Apply destroy operator
    let destroy_operator = RandomVisitRemovalInVoyages { xi_min: 0.2, xi_max: 0.8 };
    destroy_operator.apply(&mut solution, &context, &mut rng);

    // Ensure consistency after destroy, before any further operations
    println!("Ensuring consistency after destroy...");
    solution.ensure_consistency_updated(&context);
    println!("Consistency ensured.");
    
    // Feasibility check after destroy
    println!("Feasibility after destroy (light): {}", solution.is_complete_solution());
    println!("Feasibility after destroy (deep): {}", solution.is_fully_feasible(&context));
    
    dump_solution(&solution, &context.problem.vessels, "../output/solution_vis_after_destroy.json")?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule_after_destroy.json")?;

    // Add idle voyages for only one vessel (after destroy)
    solution.add_idle_vessel_and_add_empty_voyages(&context);

    // Apply DeepGreedyInsertion repair operator
    use crate::operators::repair::deep_greedy_insertion::DeepGreedyInsertion;
    use crate::operators::traits::RepairOperator;
    let repair_operator = DeepGreedyInsertion;
    repair_operator.apply(&mut solution, &context, &mut rng);

    solution.ensure_consistency_updated(&context);
    // Feasibility check after repair
    println!("Feasibility after repair (light): {}", solution.is_complete_solution());
    println!("Feasibility after repair (deep): {}", solution.is_fully_feasible(&context));

    println!("Applied DeepGreedyInsertion repair operator.");
    // Optionally, print unassigned visits after repair
    dump_solution(&solution, &context.problem.vessels, "../output/solution_vis_after_repair.json")?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule_after_repair.json")?;
    Ok(())
}

pub fn dump_solution(solution: &structs::solution::Solution, vessels: &Vec<structs::vessel::Vessel>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    dump_schedule_to_json(solution, vessels, path);
    Ok(())
}

pub fn dump_explicit_solution(solution: &structs::solution::Solution, context: &Context, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    dump_explicit_schedule_to_json(solution, context, path);
    Ok(())
}

fn test_feasibility_over_seeds() -> Result<(), Box<dyn std::error::Error>> {
    use crate::operators::repair::deep_greedy_insertion::DeepGreedyInsertion;
    use crate::operators::traits::{DestroyOperator, RepairOperator};
    use crate::operators::destroy::random_visit_removal_in_voyages::RandomVisitRemovalInVoyages;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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

    let destroy_operator = RandomVisitRemovalInVoyages { xi_min: 0.2, xi_max: 0.8 };
    let repair_operator = DeepGreedyInsertion;

    for seed in 0..100u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut solution = operators::initial_solution::construct_initial_solution(&context, &mut rng);
        let initial_cost = solution.total_cost;
        info!(target: "alns::iteration", "Seed {}: Initial solution cost = {}", seed, initial_cost);
        // Initial solution feasibility
        let is_complete_initial = solution.is_complete_solution();
        let is_feasible_initial = solution.is_fully_feasible(&context);
        if !is_complete_initial || !is_feasible_initial {
            warn!(target: "alns::iteration", "Seed {}: Initial solution incomplete or infeasible (complete={}, feasible={})", seed, is_complete_initial, is_feasible_initial);
            incomplete_initial += 1;
            if !is_complete_initial { incomplete_initial_complete += 1; }
            if !is_feasible_initial { incomplete_initial_feasible += 1; }
        }
        // After destroy
        destroy_operator.apply(&mut solution, &context, &mut rng);
        let destroy_cost = solution.total_cost;
        info!(target: "alns::iteration", "Seed {}: After destroy cost = {}", seed, destroy_cost);
        let is_complete_destroy = solution.is_complete_solution();
        let is_feasible_destroy = solution.is_fully_feasible(&context);
        if !is_complete_destroy || !is_feasible_destroy {
            warn!(target: "alns::iteration", "Seed {}: Destroy incomplete or infeasible (complete={}, feasible={})", seed, is_complete_destroy, is_feasible_destroy);
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
        info!(target: "alns::iteration", "Seed {}: After repair cost = {}", seed, repair_cost);
        solution.ensure_consistency_updated(&context);
        let is_complete_repair = solution.is_complete_solution();
        let is_feasible_repair = solution.is_fully_feasible(&context);
        if !is_complete_repair || !is_feasible_repair {
            error!(target: "alns::iteration", "Seed {}: Repair incomplete or infeasible (complete={}, feasible={})", seed, is_complete_repair, is_feasible_repair);
            incomplete_repair += 1;
            if !is_complete_repair { incomplete_repair_complete += 1; }
            if !is_feasible_repair { incomplete_repair_feasible += 1; }
            repair_incomplete_seeds.push(seed);
        }
        debug!(target: "alns::iteration", "Seed {}: Cost change: initial={} -> destroy={} -> repair={}", seed, initial_cost, destroy_cost, repair_cost);
    }
    println!("Feasibility stats over 100 seeds:");
    println!("Initial solution: {} not fully feasible ({} incomplete, {} not feasible)", incomplete_initial, incomplete_initial_complete, incomplete_initial_feasible);
    println!("After destroy:    {} not fully feasible ({} incomplete, {} not feasible)", incomplete_destroy, incomplete_destroy_complete, incomplete_destroy_feasible);
    println!("After repair:     {} not fully feasible ({} incomplete, {} not feasible)", incomplete_repair, incomplete_repair_complete, incomplete_repair_feasible);
    println!("Seeds where repair operator led to infeasibility: {:?}", repair_incomplete_seeds);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!(target: "alns::main", "ALNS process started");
    // test_feasibility_over_seeds()?;
    test_main(11)?;
    info!(target: "alns::main", "ALNS process finished");
    Ok(())
}