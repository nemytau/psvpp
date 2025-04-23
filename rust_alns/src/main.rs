mod structs;
mod utils;
mod operators;
use structs::problem_data::ProblemData;
use serde::de;
use structs::data_loader;
use structs::distance_manager::DistanceManager;
use structs::node::{HasLocation, HasTimeWindows};
use utils::tsp_solver::TSPSolver;
use operators::initial_solution::construct_initial_solution;


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
    println!("\nTesting TSP solver:");
    
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

use structs::context::Context;
use rand::thread_rng;

fn test_main() -> Result<(), Box<dyn std::error::Error>> {
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

    let mut rng = thread_rng();
    let solution = construct_initial_solution(&context, &mut rng);

    println!("Constructed solution:");
    println!("Total voyages: {}", solution.voyages.len());
    println!("Total visits: {}", solution.visits.len());
    println!("Feasibility (light): {}", solution.is_feasible_light());
    println!("Feasibility (deep): {}", solution.is_feasible_deep());

    for voyage in &solution.voyages {
        println!("Voyage ID: {}", voyage.id);
        println!("Visit IDs: {:?}", voyage.visit_ids);
        println!("Sailing time: {:?}", voyage.sailing_time);
        println!("Waiting time: {:?}", voyage.waiting_time);
        println!("Arrival time: {:?}", voyage.arrival_time);
        println!("End time at base: {:?}", voyage.end_time_at_base);
        println!("Is feasible: {}", voyage.is_feasible);
    }
    for visit in &solution.visits {
        println!("Visit ID: {}", visit.id());
        println!("Installation ID: {}", visit.installation_id());
        println!("Departure day: {:?}", visit.departure_day);
        println!("Service time: {:?}", visit.service_time());
        println!("Time window: {:?}", context.problem.get_time_window(visit.installation_id()));
    }
    // Print the schedule
    println!("Schedule: {:?}", solution.schedule);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    test_main()?;
    Ok(())
}