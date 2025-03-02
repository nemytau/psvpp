use permutohedron::Heap;

pub struct TSPSolver<'a> {
    distances: &'a Vec<Vec<f64>>,
    time_windows: Vec<(f64, f64)>,
    service_times: Vec<f64>,
    num_nodes: usize,
}

// Add constant for hours in a day
const HOURS_PER_DAY: f64 = 24.0;

impl<'a> TSPSolver<'a> {
    pub fn new(distances: &'a Vec<Vec<f64>>, time_windows: Vec<(f64, f64)>, service_times: Vec<f64>) -> Self {
        let num_nodes = distances.len();
        TSPSolver { distances, time_windows, service_times, num_nodes }
    }

    pub fn solve_tsp_full_enumeration(
        &self, 
        node_ids: Vec<usize>, 
        vessel_speed: Option<f64>, 
        start_time: Option<f64>
    ) -> (Vec<usize>, f64) {
        // Set default values if not provided
        let vessel_speed = vessel_speed.unwrap_or(12.0);
        let start_time = start_time.unwrap_or(16.0);
        
        if node_ids.is_empty() {
            return (vec![], start_time);
        }
        
        let mut nodes: Vec<usize> = node_ids.into_iter().filter(|&id| id != 0).collect();
        let heap = Heap::new(&mut nodes);
        let mut min_end_time = f64::INFINITY;
        let mut best_route = Vec::new();
        
        for perm in heap {
            let mut route = vec![0];
            route.extend(&perm);
            route.push(0);
            
            let end_time = self.calculate_voyage_end_time(&route, vessel_speed, start_time);
            if end_time < min_end_time {
                min_end_time = end_time;
                best_route = route;
            }
        }
        
        (best_route, min_end_time - start_time) // Return total voyage duration as second value
    }
    
    fn calculate_voyage_end_time(&self, route: &Vec<usize>, vessel_speed: f64, start_time: f64) -> f64 {
        let mut current_time = start_time;
        
        for i in 0..route.len() - 1 {
            let current_node = route[i];
            let next_node = route[i + 1];
            let travel_time = self.distances[current_node][next_node] / vessel_speed;
            
            // Update time considering travel
            current_time += travel_time;
            
            // Only apply time window constraints for non-depot nodes
            if next_node != 0 {
                // Check time window constraints
                let (earliest, latest) = self.time_windows[next_node];
                
                // Convert to time of day
                let arrival_day = (current_time / HOURS_PER_DAY).floor();
                let arrival_time_of_day = current_time % HOURS_PER_DAY;
                
                // Determine when to start service based on time window
                if arrival_time_of_day < earliest {
                    // Wait until the time window opens on the same day
                    current_time = arrival_day * HOURS_PER_DAY + earliest;
                } else if arrival_time_of_day > latest {
                    // Wait until the time window opens on the next day
                    current_time = (arrival_day + 1.0) * HOURS_PER_DAY + earliest;
                }
                // Otherwise, we can start service right away
                
                // Add service time
                current_time += self.service_times[next_node];
            }
        }
        
        current_time
    }

    pub fn solve_tsp_branch_and_cut(&self) -> Vec<usize> {
        // Implement branch and cut TSP solver with time windows here
        vec![]
    }

    pub fn solve_tsp_fast_reinsertion(&self) -> Vec<usize> {
        // Implement fast reinsertion TSP solver with time windows here
        vec![]
    }

    pub fn analyze_route(&self, route: &Vec<usize>, vessel_speed: Option<f64>, start_time: Option<f64>) -> String {
        let vessel_speed = vessel_speed.unwrap_or(12.0);
        let start_time = start_time.unwrap_or(16.0);
        
        let mut current_time = start_time;
        let mut result = String::new();
        
        result.push_str(&format!("Route analysis (speed: {:.1}kn, start time: {:.1}h):\n", vessel_speed, start_time));
        result.push_str(&format!("Departure from depot (node {}) at time: {:.2}h\n", route[0], current_time));
        
        for i in 0..route.len() - 1 {
            let current_node = route[i];
            let next_node = route[i + 1];
            let distance = self.distances[current_node][next_node];
            let travel_time = distance / vessel_speed;
            
            // Calculate arrival time
            let arrival_time = current_time + travel_time;
            
            result.push_str(&format!(
                "Travel from node {} to node {}: distance {:.2}nm, time {:.2}h\n", 
                current_node, next_node, distance, travel_time
            ));
            
            result.push_str(&format!("Arrive at node {} at time: {:.2}h", next_node, arrival_time));
            
            // Update current time considering travel
            current_time = arrival_time;
            
            // Only apply time window constraints for non-depot nodes
            if next_node != 0 {
                // Check time window constraints
                let (earliest, latest) = self.time_windows[next_node];
                
                // Convert to time of day
                let arrival_day = (current_time / HOURS_PER_DAY).floor();
                let arrival_time_of_day = current_time % HOURS_PER_DAY;
                
                result.push_str(&format!(
                    " (day {:.0}, time of day: {:.2}h, time window: [{:.2}h, {:.2}h])\n", 
                    arrival_day, arrival_time_of_day, earliest, latest
                ));
                
                // Determine when to start service based on time window
                let mut wait_time = 0.0;
                
                if arrival_time_of_day < earliest {
                    // Wait until the time window opens on the same day
                    wait_time = earliest - arrival_time_of_day;
                    current_time = arrival_day * HOURS_PER_DAY + earliest;
                    result.push_str(&format!("Wait {:.2}h until time window opens\n", wait_time));
                } else if arrival_time_of_day > latest {
                    // Wait until the time window opens on the next day
                    wait_time = (HOURS_PER_DAY - arrival_time_of_day) + earliest;
                    current_time = (arrival_day + 1.0) * HOURS_PER_DAY + earliest;
                    result.push_str(&format!("Wait {:.2}h until next day's time window opens\n", wait_time));
                } else {
                    // No waiting needed
                    result.push_str("No waiting needed\n");
                }
                
                // Add service time
                let service_time = self.service_times[next_node];
                result.push_str(&format!("Service time at node {}: {:.2}h\n", next_node, service_time));
                current_time += service_time;
                result.push_str(&format!("Departure from node {} at time: {:.2}h\n", next_node, current_time));
            } else {
                // For depot
                result.push_str("\n");
            }
        }
        
        let total_duration = current_time - start_time;
        result.push_str(&format!("Total voyage duration: {:.2}h\n", total_duration));
        
        result
    }
}
