mod operators;
mod structs;
mod utils;

use rand::SeedableRng;
use rand::rngs::StdRng;


use operators::initial_solution::construct_initial_solution;
use crate::operators::destroy::random_visit_removal_in_voyages::RandomVisitRemovalInVoyages;
use crate::operators::traits::DestroyOperator;
use structs::context::Context;
use structs::data_loader;
use structs::distance_manager::DistanceManager;
use structs::node::{HasLocation, HasTimeWindows};
use structs::problem_data::ProblemData;
use utils::serialization::{dump_explicit_schedule_to_json, dump_schedule_to_json};
use utils::tsp_solver::TSPSolver;

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
    let time_windows = vec![
        (0.0, 24.0),   // Depot - always open
        (8.0, 16.0),   // Node 1
        (10.0, 18.0),  // Node 2
        (8.0, 14.0),   // Node 3
    ];
    
    let service_times = vec![
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


    println!("Constructed solution:");
    for voyage in &solution.voyages {
        let voyage = voyage.borrow();
        println!("Voyage ID: {}, Vessel ID: {:?}, Visits: {:?}", voyage.id, voyage.vessel_id, voyage.visit_ids);
    }
    println!("Total voyages: {}", solution.voyages.len());
    println!("Total visits: {}", solution.all_visits().len());
    println!("Feasibility (light): {}", solution.is_feasible_light());
    println!("Feasibility (deep): {}", solution.is_feasible_deep(&context));

    // Print the schedule
    println!("Schedule: {:?}", solution.get_schedule(&context));

    dump_solution(&solution, &context.problem.vessels, "../output/solution_vis.json")?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule.json")?;

    // Apply destroy operator
    let destroy_operator = RandomVisitRemovalInVoyages { xi_min: 0.2, xi_max: 0.8 };
    destroy_operator.apply(&mut solution, &context, &mut rng);

    // Ensure consistency after destroy, before any further operations
    solution.ensure_consistency_updated(&context);

    // Feasibility check after destroy
    println!("Feasibility after destroy (light): {}", solution.is_feasible_light());
    println!("Feasibility after destroy (deep): {}", solution.is_feasible_deep(&context));

    // Output unassigned visits
    println!("Unassigned visits after destroy:");
    for (i, visit) in solution.all_visits().iter().enumerate() {
        if !visit.is_assigned {
            println!("Visit index: {}, Visit ID: {}", i, visit.id());
        }
    }
    dump_solution(&solution, &context.problem.vessels, "../output/solution_vis_after_destroy.json")?;
    dump_explicit_solution(&solution, &context, "../output/explicit_schedule_after_destroy.json")?;

    // Apply DeepGreedyInsertion repair operator
    use crate::operators::repair::deep_greedy_insertion::DeepGreedyInsertion;
    use crate::operators::traits::RepairOperator;
    let repair_operator = DeepGreedyInsertion;
    repair_operator.apply(&mut solution, &context, &mut rng);

    // Feasibility check after repair
    println!("Feasibility after repair (light): {}", solution.is_feasible_light());
    println!("Feasibility after repair (deep): {}", solution.is_feasible_deep(&context));

    println!("Applied DeepGreedyInsertion repair operator.");
    // Optionally, print unassigned visits after repair
    println!("Unassigned visits after repair:");
    for (i, visit) in solution.all_visits().iter().enumerate() {
        if !visit.is_assigned {
            println!("Visit index: {}, Visit ID: {}", i, visit.id());
        }
    }
    solution.ensure_consistency_updated(&context);
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

    let mut infeasible_initial = 0;
    let mut infeasible_initial_light = 0;
    let mut infeasible_initial_deep = 0;
    let mut infeasible_destroy = 0;
    let mut infeasible_destroy_light = 0;
    let mut infeasible_destroy_deep = 0;
    let mut infeasible_repair = 0;
    let mut infeasible_repair_light = 0;
    let mut infeasible_repair_deep = 0;
    let mut repair_infeasible_seeds = Vec::new();

    let destroy_operator = RandomVisitRemovalInVoyages { xi_min: 0.2, xi_max: 0.8 };
    let repair_operator = DeepGreedyInsertion;

    for seed in 0..100u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut solution = operators::initial_solution::construct_initial_solution(&context, &mut rng);
        // Initial solution feasibility
        let initial_light = solution.is_feasible_light();
        let initial_deep = solution.is_feasible_deep(&context);
        if !initial_light || !initial_deep {
            infeasible_initial += 1;
            if !initial_light { infeasible_initial_light += 1; }
            if !initial_deep { infeasible_initial_deep += 1; }
        }
        // After destroy
        destroy_operator.apply(&mut solution, &context, &mut rng);
        let destroy_light = solution.is_feasible_light();
        let destroy_deep = solution.is_feasible_deep(&context);
        if !destroy_light || !destroy_deep {
            infeasible_destroy += 1;
            if !destroy_light { infeasible_destroy_light += 1; }
            if !destroy_deep { infeasible_destroy_deep += 1; }
        }
        // After repair
        solution.ensure_consistency_updated(&context);
        repair_operator.apply(&mut solution, &context, &mut rng);
        let repair_light = solution.is_feasible_light();
        let repair_deep = solution.is_feasible_deep(&context);
        if !repair_light || !repair_deep {
            infeasible_repair += 1;
            if !repair_light { infeasible_repair_light += 1; }
            if !repair_deep { infeasible_repair_deep += 1; }
            repair_infeasible_seeds.push(seed);
        }
    }
    println!("Infeasibility stats over 100 seeds:");
    println!("Initial solution: {} infeasible ({} light, {} deep)", infeasible_initial, infeasible_initial_light, infeasible_initial_deep);
    println!("After destroy:    {} infeasible ({} light, {} deep)", infeasible_destroy, infeasible_destroy_light, infeasible_destroy_deep);
    println!("After repair:     {} infeasible ({} light, {} deep)", infeasible_repair, infeasible_repair_light, infeasible_repair_deep);
    println!("Seeds where repair operator led to infeasibility: {:?}", repair_infeasible_seeds);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    test_feasibility_over_seeds()?;
    // test_main(0)?;
    Ok(())
}