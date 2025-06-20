use core::time;
use std::collections::HashMap;
use std::time::Duration;
use permutohedron::Heap;
use crate::structs::{constants::{DAYS_IN_PERIOD, HOURS_IN_DAY, HOURS_IN_PERIOD}, problem_data::ProblemData, time_window::TimeWindow, visit::Visit, voyage::Voyage};

#[derive(Clone)]
pub struct TSPResult {
    pub visit_ids_seq: Vec<usize>,
    pub sailing_time: f64,
    pub waiting_time: f64,
    pub arrival_time: f64,
    pub end_time: f64, // Absolute end time, including wait at depot and service at depot
}

/*
Given  n  installations, where each installation  i :
	•	Must be visited exactly once.
	•	Has service time s_i
	•	Has a distance matrix  D[i][j] .
	•	Has a time window  [e_i, l_i].
	•	The ship starts and ends at base - node 0, ship has travelling speed - v.
	•	The objective is to minimize total time while satisfying constraints.
    •	Base is node 0, route starts and ends at base.
*/

// NOTE: Problem formulation says that time windows are same for all days, but in the code we have a vector of time windows for each day.
// This is a simplification, but it allows for more flexibility in the future: different time windows for different days.
pub struct TSPSolver {
    distances: Vec<Vec<f64>>,
    daily_time_windows: Vec<TimeWindow>,
    service_times: Vec<f64>,
    num_nodes: usize,
    pub visit_to_installation: HashMap<usize, usize>,
}

impl TSPSolver {
    pub fn new(
        distances: Vec<Vec<f64>>, 
        daily_time_windows: Vec<TimeWindow>,
        service_times: Vec<f64>,
        visit_to_installation: HashMap<usize, usize>,
    ) -> Self {
        let num_nodes = distances.len();
        TSPSolver { 
            distances,
            daily_time_windows,
            service_times,
            num_nodes,
            visit_to_installation,
        }
    }
    pub fn new_from_problem_data(
        problem_data: &ProblemData,
    ) -> Self {
        let distances = problem_data.distance_manager.distances().clone();

        let mut daily_time_windows = Vec::with_capacity(problem_data.installations.len() + 1);
        daily_time_windows.push(problem_data.base.service_time_window.clone());
        daily_time_windows.extend(
            problem_data.installations.iter()
                .map(|installation| installation.service_time_window.clone())
        );

        let mut service_times = Vec::with_capacity(problem_data.installations.len() + 1);
        service_times.push(problem_data.base.service_time);
        service_times.extend(
            problem_data.installations.iter()
                .map(|installation| installation.get_service_time())
        );

        let visits = problem_data.generate_visits();
        // IMPORTANT: installation_id == index in the distance matrix
        let visit_to_installation: HashMap<usize, usize> = visits.iter()
            .map(|v| (v.id(), v.installation_id()))
            .collect();

        TSPSolver::new(distances, daily_time_windows, service_times, visit_to_installation)
    }

    // TODO: Consider following comment:
    // During solve_for_voyage, accept a Vec<&Installation> or Vec<&dyn HasLocationAndTW>.
    // Let Visit remain a planning layer, but decouple it from optimization core.
    // It is not really solving for the voyage, it only takes the speed from the voyage, but
    // it is solving for the visits provided.
    pub fn solve_for_voyage(&self, voyage: &Voyage) -> TSPResult {
        let speed = voyage.speed().unwrap_or_else(|| panic!("Vessel speed must be set up for the voyage"));
        let start_time = voyage.start_time().unwrap_or_else(|| panic!("Start time must be set up for the voyage"));
        self.solve_internal(voyage.get_visit_sequence(), speed, start_time)
    }

    /// Evaluates all possible greedy insertions of an extra visit into a voyage, returns the best result.
    pub fn evaluate_greedy_insertion(&self, voyage: &Voyage, extra_visit: usize) -> TSPResult {
        let current_visit_sequence = voyage.get_visit_sequence();
        let speed = voyage.speed().unwrap_or_else(|| panic!("Vessel speed must be set up for the voyage"));
        let start_time = voyage.start_time().unwrap_or_else(|| panic!("Start time must be set up for the voyage"));

        let mut best_route = Vec::new();
        let mut best_cost = f64::INFINITY;
        let mut best_sailing_time = 0.0;
        let mut best_waiting_time = 0.0;
        let mut best_arrival_time = 0.0;
        let mut best_end_time = 0.0;

        for i in 0..=current_visit_sequence.len() {
            let mut updated_visit_sequence = current_visit_sequence.to_vec();
            updated_visit_sequence.insert(i, extra_visit);

            let result = self.visit_sequence_to_result(&updated_visit_sequence, speed, start_time);
            if result.end_time < best_cost {
                best_route = result.visit_ids_seq.clone();
                best_cost = result.end_time;
                best_sailing_time = result.sailing_time;
                best_waiting_time = result.waiting_time;
                best_arrival_time = result.arrival_time;
                best_end_time = result.end_time;
            }
        }

        TSPResult {
            visit_ids_seq: best_route,
            sailing_time: best_sailing_time,
            waiting_time: best_waiting_time,
            arrival_time: best_arrival_time,
            end_time: best_end_time,
        }
    }

    pub fn solve_and_get_end_time(
        &self,
        visit_ids: &[usize],
        speed: f64,
        start_time: f64
    ) -> f64 {
        self.solve_internal(visit_ids.to_vec(), speed, start_time).end_time
    }

    // I f*cked up this method. I have to lookup for each installation its corresponding visit id which is O(n^2).
    // It is not critical since we have at maximum 5-10 visits on the route, but it is not optimal.
    // Possible fixes:
    // 1. Create a map of installation id to visit id in the constructor of the solver. It will decrease the time complexity to O(n).
    // 2. Create hashmap here, it will be O(n) but we have to create it every time we call this method.
    // 3. Use visit ids internally, it will increase distance matrix size, but it will be O(1) lookup.
    // 4. Shift visit focus on installations and use installations ids as input. It requires more changes in the architecture.
    // 5. Store visit_ids in voyage and route as sequence of installation ids. Probably right solution.
    // Fixed (?), still n2 though.
    fn solve_internal(&self, visit_ids: Vec<usize>, speed: f64, start_time: f64) -> TSPResult {
        // Convert visit IDs to installation IDs
        let inst_ids: Vec<usize> = self.visit_ids_to_installation_ids_sequence(&visit_ids);
        // Solve TSP using installation IDs
        let (best_route, _) = self.solve_tsp_branch_and_bound(inst_ids.clone(), Some(speed), Some(start_time));
        let (sailing_time, waiting_time, arrival_time, end_time) = 
            self.calculate_voyage_details(&best_route, speed, start_time);

        // Strip depot from route
        let route_without_depot: Vec<usize> = if best_route.len() >= 2 {
            best_route[1..best_route.len()-1].to_vec()
        } else {
            Vec::new()
        };
        // Convert installation IDs back to visit IDs
        let route_without_depot = self.installation_ids_to_visit_ids_sequence(&route_without_depot, &visit_ids);
        TSPResult {
            visit_ids_seq: route_without_depot,
            sailing_time,
            waiting_time,
            arrival_time,
            end_time,
        }
    }
    
    /// Public wrapper for visit_ids_to_installation_ids_sequence
    pub fn visit_ids_to_installation_ids_sequence_public(&self, visit_ids: &[usize]) -> Vec<usize> {
        self.visit_ids_to_installation_ids_sequence(visit_ids)
    }
    /// Public wrapper for calculate_voyage_details
    pub fn calculate_voyage_details_public(&self, route: &Vec<usize>, vessel_speed: f64, start_time: f64) -> (f64, f64, f64, f64) {
        self.calculate_voyage_details(route, vessel_speed, start_time)
    }
    /// Public wrapper for visit_sequence_to_route
    pub fn visit_sequence_to_route_public(&self, visit_sequence: &Vec<usize>) -> Vec<usize> {
        self.visit_sequence_to_route(visit_sequence)
    }

    fn visit_ids_to_installation_ids_sequence(
        &self,
        visit_ids: &[usize],
    ) -> Vec<usize> {
        visit_ids.iter()
            .filter_map(|&visit_id| self.visit_to_installation.get(&visit_id))
            .cloned()
            .collect()
    }
    fn installation_ids_to_visit_ids_sequence(
        &self,
        installation_ids: &[usize],
        visit_ids: &[usize],
    ) -> Vec<usize> {
        let mut remaining_visits: Vec<usize> = visit_ids.to_vec();
        let mut result = Vec::with_capacity(installation_ids.len());
    
        for &inst_id in installation_ids {
            if let Some(pos) = remaining_visits.iter().position(|&vid| {
                self.visit_to_installation.get(&vid) == Some(&inst_id)
            }) {
                result.push(remaining_visits.remove(pos));
            }
        }
    
        result
    }
    pub fn get_time_window(
        &self,
        node_id: usize,
    ) -> Option<(f64, f64)> {
        if node_id < self.daily_time_windows.len() {
            let tw = &self.daily_time_windows[node_id];
            Some((tw.earliest.unwrap_or(0.0), tw.latest.unwrap_or(HOURS_IN_DAY as f64)))
        } else {
            None
        }
    }
    fn visit_sequence_to_result(
        &self,
        visit_sequence: &Vec<usize>,
        vessel_speed: f64,
        start_time: f64
    ) -> TSPResult {
        let route = self.visit_sequence_to_route(visit_sequence);
        let (sailing_time, waiting_time, arrival_time, end_time) = 
            self.calculate_voyage_details(&route, vessel_speed, start_time);        
        TSPResult {
            visit_ids_seq: visit_sequence.clone(),
            sailing_time,
            waiting_time,
            arrival_time,
            end_time,
        }
    }

    fn visit_sequence_to_route(
        &self,
        visit_sequence: &Vec<usize>,
    ) -> Vec<usize> {
        let inst_sequence: Vec<usize> = self.visit_ids_to_installation_ids_sequence(visit_sequence);
        self.inst_sequence_to_route(&inst_sequence)
    }
    fn inst_sequence_to_route(
        &self,
        inst_sequence: &Vec<usize>,
    ) -> Vec<usize> {
        let mut route = vec![0]; // Start at depot
        route.extend(inst_sequence);
        route.push(0); // End at depot
        route
    }

    // Calculate the voyage details including sailing time, waiting time, arrival time, and end time
    // Expected to have the route starting and ending at the depot (node 0)
    fn calculate_voyage_details(
        &self,
        route: &Vec<usize>,
        vessel_speed: f64,
        start_time: f64
    ) -> (f64, f64, f64, f64) { // (sailing_time, waiting_time, arrival_time, end_time)
        let mut current_time = start_time;
        let mut total_sailing_time = 0.0;
        let mut total_waiting_time = 0.0;
        
        // ERROR: PANICKED HERE DURING CONSISTENCY CHECK
        if route.is_empty() || route[0] != 0 || route.last() != Some(&0) {
            panic!("Route must start and end at the depot (node 0)");
        }
        for i in 0..route.len() - 1 {
            let current_node = route[i];
            let next_node = route[i + 1];
            let travel_time = self.distances[current_node][next_node] / vessel_speed;
            
            // Track sailing time
            total_sailing_time += travel_time;
            
            // Update time considering travel
            current_time += travel_time;
            
            // Only apply time window constraints for non-depot nodes
            if next_node != 0 {
                // Calculate wait time
                let wait_time = self.compute_wait_time(next_node, current_time).unwrap_or(0.0);
                total_waiting_time += wait_time;
                
                // Update time with wait and service
                current_time += wait_time;
                current_time += self.service_times[next_node];
            }
        }
        
        let arrival_time = current_time;
        // For end_time, check if there's a wait time at the depot
        let wait_at_depot = self.compute_wait_time(0, arrival_time).unwrap_or(0.0);
        let end_time = arrival_time + wait_at_depot + self.service_times[0];
        
        (total_sailing_time, total_waiting_time, arrival_time, end_time)
    }

    pub fn solve_tsp_full_enumeration(
        &self, 
        inst_indices: Vec<usize>, 
        vessel_speed: Option<f64>, 
        start_time: Option<f64>
    ) -> (Vec<usize>, f64) {
        // Set default values if not provided
        let vessel_speed = vessel_speed.unwrap_or(12.0);
        let start_time = start_time.unwrap_or(16.0);
        
        if inst_indices.is_empty() {
            return (vec![], start_time);
        }
        
        let mut nodes: Vec<usize> = inst_indices.into_iter().filter(|&id| id != 0).collect();
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
    
    fn calculate_voyage_end_time(
        &self, 
        route: &Vec<usize>, 
        vessel_speed: f64, 
        start_time: f64
    ) -> f64 {
// Arrival time to the depot
        self.calculate_voyage_details(route, vessel_speed, start_time).2
    }

    // TODO: Implement cost calculation by adding fuel consumption and other factors
    // inst_indices - the list of nodes to visit EXCLUDING depot
    pub fn solve_tsp_branch_and_bound(
        &self, 
        inst_indices: Vec<usize>, 
        vessel_speed: Option<f64>, 
        start_time: Option<f64>
    ) -> (Vec<usize>, f64) {
        // Defensive check for duplicate installation IDs
    use std::collections::HashSet;
    if inst_indices.iter().collect::<HashSet<_>>().len() != inst_indices.len() {
        eprintln!("[ERROR] Duplicate installation IDs detected in TSP input: {:?}", inst_indices);
        return (vec![], f64::INFINITY);
    }
        // Implement branch and bound TSP solver with time windows here
        let speed = vessel_speed.unwrap_or(12.0);
        let init_time = start_time.unwrap_or(16.0);
        
        // Placeholder for best solution tracking
        let mut best_route = Vec::new();
        let mut best_cost = f64::INFINITY;

        // Start from base node 0
        let mut visited = vec![false; self.num_nodes];
        visited[0] = true;

        let mut current_route = vec![0];
        self.branch_and_bound(
            0,
            init_time,
            0.0,
            &inst_indices,
            &mut visited,
            &mut current_route,
            &mut best_route,
            &mut best_cost,
            speed
        );

        (best_route, best_cost)
    }

    fn branch_and_bound(
        &self,
        current_node: usize,
        current_time: f64,
        current_cost: f64,
        node_ids: &Vec<usize>,
        visited: &mut Vec<bool>,
        current_route: &mut Vec<usize>,
        best_route: &mut Vec<usize>,
        best_cost: &mut f64,
        speed: f64
    ) {
        if current_route.len() == node_ids.len() + 1 {
            // Add return to depot
            let return_cost = self.distances[current_node][0]/speed;
            let total_cost = current_cost + return_cost;
            if total_cost < *best_cost {
                let mut complete_route = current_route.clone();
                complete_route.push(0);
                *best_cost = total_cost;
                *best_route = complete_route;
            }
            return;
        }
        if current_cost >= *best_cost {
            return; // Prune this branch
        }
        for &next_node in node_ids {
            if visited[next_node] || next_node == 0 { continue; }
            let arrival_time = current_time + self.distances[current_node][next_node] / speed;
            let wait_time = self.compute_wait_time(next_node, arrival_time).unwrap_or(0.0);
            let begin_service = arrival_time + wait_time;
            let finish_service = begin_service + self.service_times[next_node];
            let next_node_visit_cost = finish_service - current_time;
            visited[next_node] = true;
            current_route.push(next_node);

            self.branch_and_bound(
                next_node,
                finish_service,
                current_cost + next_node_visit_cost,
                node_ids,
                visited,
                current_route,
                best_route,
                best_cost,
                speed
            );
            current_route.pop();
            visited[next_node] = false;
        }
    }

    pub fn compute_wait_time(
        &self,
        node: usize,
        arrival_time: f64,
    ) -> Option<f64> {
        let local_hour = arrival_time % (HOURS_IN_DAY as f64);
        let tw = &self.daily_time_windows[node];
        
        // Check if arrival time is within the time window
        if tw.contains(local_hour) {
            return Some(0.0); // No wait needed
        } 
        else {
            // Need to wait until the time window opens
            let earliest = tw.earliest.unwrap_or(0.0);
        if local_hour < earliest {
            // Wait until the time window opens today
                return Some(earliest - local_hour);
            } else {
// Wait until the time window opens tomorrow
                return Some(HOURS_IN_DAY as f64 + earliest - local_hour);
            }
                }
    }

    #[allow(dead_code)]
    pub fn solve_tsp_fast_reinsertion(&self) -> Vec<usize> {
        // Implement fast reinsertion TSP solver with time windows here
        vec![]
    }

    pub fn analyze_route(&self, route: &Vec<usize>, vessel_speed: Option<f64>, start_time: Option<f64>) -> String {
        let vessel_speed = vessel_speed.unwrap_or(12.0);
        let start_time = start_time.unwrap_or(16.0);
        let mut total_wait_time_at_installations = 0.0;
        let mut wait_time_at_depot = 0.0;
        let mut total_service_time = 0.0;
        let mut total_sailing_time = 0.0;
        let mut total_distance = 0.0;
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
            current_time = arrival_time;
            
            total_sailing_time += travel_time;
            total_distance += distance;

            // Only apply time window constraints for non-depot nodes
            if next_node != 0 {
                // Convert to time of day
                let arrival_day = (arrival_time / (HOURS_IN_DAY as f64)).floor() as u32;
                let arrival_time_of_day = arrival_time % (HOURS_IN_DAY as f64);
                let tw = &self.daily_time_windows[next_node];
                result.push_str(&format!(
                    " (day {:.0}, time of day: {:.2}h, time window: [{:.2}, {:.2}])\n", 
                    arrival_day, arrival_time_of_day, tw.earliest.unwrap_or(0.0), tw.latest.unwrap_or(HOURS_IN_DAY as f64)
                ));
                
                // Determine when to start service based on time window
                let wait_time = self.compute_wait_time(next_node, arrival_time).unwrap_or(0.0);
                if wait_time > 0.0 {
                    result.push_str(&format!("Waiting for {:.2}h until time window opens\n", wait_time));
                }
                current_time = arrival_time + wait_time;
                // Add service time
                let service_time = self.service_times[next_node];
                result.push_str(&format!("Service time at node {}: {:.2}h\n", next_node, service_time));
                current_time += service_time;
                result.push_str(&format!("Departure from node {} at time: {:.2}h\n", next_node, current_time));

                total_wait_time_at_installations += wait_time;
                total_service_time += service_time;

            } else {
                // For depot
                let wait_time = self.compute_wait_time(next_node, arrival_time).unwrap_or(0.0);
                wait_time_at_depot += wait_time;
                result.push_str("\n");
            }
        }
        
        let total_duration = current_time - start_time;
        result.push_str(&format!("Total voyage duration: {:.2}h\n", total_duration));
        result.push_str(&format!("Total sailing time: {:.2}h\n", total_sailing_time));
        result.push_str(&format!("Total distance: {:.2}nm\n", total_distance));
        result.push_str(&format!("Total service time: {:.2}h\n", total_service_time));
        result.push_str(&format!("Total wait time at installations: {:.2}h\n", total_wait_time_at_installations));
        result.push_str(&format!("Total wait time at depot: {:.2}h\n", wait_time_at_depot));
        result.push_str(&format!("Total wait time: {:.2}h\n", total_wait_time_at_installations + wait_time_at_depot));
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    // Remove unused imports
    // use crate::structs::node::Node;
    // use crate::structs::node::{Base, Installation};
    use crate::structs::time_window::TimeWindow;
    use crate::structs::constants::{HOURS_IN_DAY, DAYS_IN_PERIOD};

    fn new_solver_with_single_node_and_tw(earliest: Option<f64>, latest: Option<f64>) -> TSPSolver {
        let distances = vec![
            vec![0.0, 24.0],
            vec![24.0, 0.0],
        ];
        let service_times = vec![8.0, 1.0];
        let daily_time_windows = vec![
            TimeWindow::new(Some(8.0), Some(8.0)).unwrap(),
            TimeWindow::new(earliest, latest).unwrap(),
        ];
        let visit_to_installation = [(0, 0), (1, 1)].iter().cloned().collect();
        TSPSolver::new(distances, daily_time_windows, service_times, visit_to_installation)
    }
    
    #[test]
    fn test_tsp_solver(){    // Test the TSP solver implementation
        let distances = vec![
            vec![0.0, 50.0, 60.0, 70.0],
            vec![50.0, 0.0, 40.0, 30.0],
            vec![60.0, 40.0, 0.0, 20.0],
            vec![70.0, 30.0, 20.0, 0.0],
        ];
        let raw_tw_per_node = vec![
            (Some(8.0), Some(8.0)),
            (Some(9.0), Some(13.0)),
            (Some(7.0), Some(14.0)),
            (Some(11.0), Some(15.0)),
        ];

        let daily_time_windows = raw_tw_per_node.iter()
            .map(|&(earliest, latest)| TimeWindow::new(earliest, latest).unwrap())
            .collect::<Vec<_>>();
        let service_times = vec![1.0, 2.0, 3.0, 4.0];
        let inst_indices = vec![1, 2, 3];
        let visit_indices = vec![1, 2, 3];
        let mut visit_to_installation = HashMap::new();
        for i in 0..visit_indices.len() {
            visit_to_installation.insert(visit_indices[i], inst_indices[i]);
        }
        let tsp_solver = TSPSolver::new(distances.clone(), daily_time_windows.clone(), service_times.clone(), visit_to_installation.clone());
        
        let (best_route, total_duration) = tsp_solver.solve_tsp_full_enumeration(inst_indices.clone(), None, None);
        let (bb_route, bb_duration) = tsp_solver.solve_tsp_branch_and_bound(inst_indices.clone(), None, None);

        assert_eq!(best_route.len(), inst_indices.len() + 2); // Including depot
        assert_eq!(total_duration > 0.0, true);
        assert_eq!(bb_route.len(), inst_indices.len() + 2); // Including depot
        assert_eq!(bb_duration > 0.0, true);
        assert_eq!(best_route, bb_route); // Both methods should yield the same route
        assert_eq!(total_duration, bb_duration); // Both methods should yield the same duration
        // Analyze the route
        let analysis = tsp_solver.analyze_route(&best_route, None, None);
        println!("Route analysis:\n{}", analysis);
    }

    /// Case 1: Arrival within time window => wait = 0
    #[test]
    fn test_wait_time_computation_tw_case_1() {
        let solver = new_solver_with_single_node_and_tw(Some(9.0), Some(13.0));
        let arrival = 10.0; // within today's time window
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 1: wait = {:?}", wait);
        assert_eq!(wait, Some(0.0));
    }
    /// Case 2: Arrival before time window => wait = earliest - arrival
    #[test]
    fn test_wait_time_computation_tw_case_2() {
        let solver = new_solver_with_single_node_and_tw(Some(9.0), Some(13.0));
        let arrival = 8.0;
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 2: wait = {:?}", wait);
        assert_eq!(wait, Some(1.0));
    }
    /// Case 3: Arrival today, but today’s time window is invalid → use tomorrow
    #[test]
    fn test_wait_time_computation_tw_case_3() {
        let mut solver = new_solver_with_single_node_and_tw(Some(9.0), Some(13.0));
        
        let arrival = 22.0; // near end of day 0
        let wait = solver.compute_wait_time(1, arrival);
        let expected = (1.0 * HOURS_IN_DAY as f64 + 9.0) - arrival;
        println!("Case 3: wait = {:?}, expected = {}", wait, expected);
        assert!((wait.unwrap() - expected).abs() < 1e-6);
    }
    /// Case 4: Arrival on same weekday, next week → wait = 0
    #[test]
    fn test_wait_time_computation_tw_case_4() {
        let solver = new_solver_with_single_node_and_tw(Some(9.0), Some(13.0));
        let arrival = 10.0 + (HOURS_IN_DAY * DAYS_IN_PERIOD) as f64; // next Monday at 09:00
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 4: wait = {:?}", wait);
        assert_eq!(wait, Some(0.0));
    }
    /// Case 5: Arrival next week, wait till next day
    #[test]
    fn test_wait_time_computation_tw_case_5() {
        let mut solver = new_solver_with_single_node_and_tw(Some(9.0), Some(13.0));
        let arrival = 14.0 + DAYS_IN_PERIOD as f64 * HOURS_IN_DAY as f64; // next Tuesday 09:00
        let expected = 9.0 + (1.0 + DAYS_IN_PERIOD as f64) * HOURS_IN_DAY as f64 - arrival; // next Wednesday 09:00 - arrival
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 5: wait = {:?}, expected = {}", wait, expected);
        assert!((wait.unwrap() - expected).abs() < 1e-6);
    }
    /// Case 6: Arrival on last day (Sunday), after time window → wait until next week's window
    #[test]
    fn test_wait_time_computation_tw_case_6() {
        let solver = new_solver_with_single_node_and_tw(Some(9.0), Some(13.0));

        let sunday = DAYS_IN_PERIOD - 1; // Sunday = day 6 (0-based)
        let arrival = (sunday as f64) * HOURS_IN_DAY as f64 + 20.0; // Sunday at 20:00

        let next_week_window_start = 9.0 + (DAYS_IN_PERIOD as f64) * HOURS_IN_DAY as f64;
        let expected = next_week_window_start - arrival;

        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 6: wait = {:?}, expected = {}", wait, expected);
        assert!((wait.unwrap() - expected).abs() < 1e-6);
    }
    /// Case 7: Arrival at 11:30 p.m. on Tuesday, service 1 hour, installation TW setup as [0, 24]
    #[test]
    fn test_wait_time_computation_tw_case_7() {
        let solver = new_solver_with_single_node_and_tw(Some(0.0), Some(24.0));
        let arrival = HOURS_IN_DAY as f64 * 1.0 + 23.5; // Tuesday at 11:30 p.m.
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 7: wait = {:?}", wait);
        assert_eq!(wait, Some(0.0));
    }
    /// Case 8: Arrival at 11:30 p.m. on Tuesday, service 1 hour, installation TW setup as [None, None]
    #[test]
    fn test_wait_time_computation_tw_case_8() {
        let solver = new_solver_with_single_node_and_tw(None, None);
        let arrival = HOURS_IN_DAY as f64 * 1.0 + 23.5; // Tuesday at 11:30 p.m.
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 8: wait = {:?}", wait);
        assert_eq!(wait, Some(0.0));
    }
}