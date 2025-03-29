use core::time;
use permutohedron::Heap;
use crate::structs::{constants::{DAYS_IN_PERIOD, HOURS_IN_DAY, HOURS_IN_PERIOD}, time_window::TimeWindow};

/*
Given  n  installations, where each installation  i :
	•	Must be visited exactly once.
	•	Has service time s_i
	•	Has a distance matrix  D[i][j] .
	•	Has a time window  [e_i, l_i] , where:
	•	The visit to city  i  must happen between  e_i  and  l_i of any day k. Could be represented as multiple time window matrix e_i_k, l_i_k
	•	The ship starts and ends at base - node 0, ship has travelling speed - v.
	•	The objective is to minimize total time while satisfying constraints.
*/
pub struct TSPSolver {
    // All fields now take ownership of their data
    distances: Vec<Vec<f64>>,
    service_time_windows: Vec<Vec<TimeWindow>>,
    service_times: Vec<f64>,
    num_nodes: usize,
}

impl TSPSolver {
    pub fn new(
        distances: Vec<Vec<f64>>, 
        service_time_windows: Vec<Vec<TimeWindow>>,
        service_times: Vec<f64>
    ) -> Self {
        let num_nodes = distances.len();
        TSPSolver { distances, service_time_windows, service_times, num_nodes }
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
    
    fn calculate_voyage_end_time(
        &self, 
        route: &Vec<usize>, 
        vessel_speed: f64, 
        start_time: f64
    ) -> f64 {
        let mut current_time = start_time;
        
        for i in 0..route.len() - 1 {
            let current_node = route[i];
            let next_node = route[i + 1];
            let travel_time = self.distances[current_node][next_node] / vessel_speed;
            
            // Update time considering travel
            current_time += travel_time;
            
            // Only apply time window constraints for non-depot nodes
            if next_node != 0 {
                // Convert to time of day
                let wait_time = self.compute_wait_time(next_node, current_time).unwrap_or(0.0);
                current_time += wait_time;
                // Add service time
                current_time += self.service_times[next_node];
            }
        }
        
        current_time
    }

    // TODO: Implement cost calculation by adding fuel consumption and other factors
    #[allow(dead_code)]
    pub fn solve_tsp_branch_and_bound(
        &self, 
        node_ids: Vec<usize>, 
        vessel_speed: Option<f64>, 
        start_time: Option<f64>
    ) -> (Vec<usize>, f64) {
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
            &node_ids,
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

    fn compute_wait_time(
        &self,
        node: usize,
        arrival_time: f64,
    ) -> Option<f64> {
        let relative_arrival_day = ((arrival_time / HOURS_IN_DAY as f64).floor() as usize) % DAYS_IN_PERIOD as usize;
        let relative_arrival_time = arrival_time % (HOURS_IN_PERIOD as f64);
        // Try today’s window first
        let today_tw = &self.service_time_windows[node][relative_arrival_day];
        if today_tw.contains(relative_arrival_time) {
            return Some(0.0); // No wait needed
        }
        else {
            let earliest = today_tw.earliest.unwrap_or(0.0);
            let wait_time = earliest - relative_arrival_time;
            if wait_time > 0.0 && wait_time <= HOURS_IN_DAY as f64 {
                return Some(wait_time); // Wait until the time window opens
            }
            if wait_time < 0.0 {

            }
        }
        // Try next day's window if value was not found yet(assumes next day is valid)
        let next_day = (relative_arrival_day + 1) % DAYS_IN_PERIOD as usize;
        let next_tw = &self.service_time_windows[node][next_day];
        let earliest = next_tw.earliest.unwrap_or(0.0);
        let wait_time = earliest - relative_arrival_time;
        if wait_time < 0.0 {
            // Case when arrival time on the first week and time window on the next week
            let wait_time = (HOURS_IN_PERIOD as f64) + wait_time;
            if wait_time > HOURS_IN_DAY as f64 {
                println!("Wait time exceeds 24 hours, wait time calculation in tsp_solver.rs is incorrect");
                None // Cannot wait — violates next day's window
            } else if wait_time < 0.0 {
                println!("Wait time is negative, wait time calculation in tsp_solver.rs is incorrect");
                None // Cannot wait — violates next day's window
            } else {
                Some(wait_time)
            }
        } else {
            Some(wait_time)
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
                let tw = &self.service_time_windows[next_node][arrival_day as usize];
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
    use crate::structs::node::Node;
    use crate::structs::node::{Base, Installation};
    use crate::structs::time_window::TimeWindow;
    use crate::structs::constants::{HOURS_IN_DAY, DAYS_IN_PERIOD};
    
    /// Helper to create a full week of default time windows for one node
    fn make_weekly_tw(earliest: f64, latest: f64) -> Vec<TimeWindow> {
        let mut tws = Vec::with_capacity(DAYS_IN_PERIOD as usize);
        for day in 0..DAYS_IN_PERIOD {
            tws.push(TimeWindow::new(
                Some(earliest + day as f64 * HOURS_IN_DAY as f64),
                Some(latest + day as f64 * HOURS_IN_DAY as f64)
            ).unwrap());
        }
        tws
    }

    fn new_solver_with_single_node_and_tw(earliest: f64, latest: f64) -> TSPSolver {
        let distances = vec![
            vec![0.0, 24.0],
            vec![24.0, 0.0],
        ];
        let service_times = vec![8.0, 1.0];
        let service_time_windows = vec![make_weekly_tw(8.0, 8.0), make_weekly_tw(earliest, latest)];
        TSPSolver::new(distances, service_time_windows, service_times)
    }

    #[test]
    fn test_tsp_solver() {
        // Test the TSP solver with a simple example
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

        let service_time_windows: Vec<Vec<TimeWindow>> = 
            raw_tw_per_node.iter()
                .map(|&(earliest, latest)| {
                    make_weekly_tw(earliest.unwrap_or(0.0), latest.unwrap_or(24.0))
                })
                .collect();
        let service_times = vec![1.0, 2.0, 3.0, 4.0];
        let tsp_solver = TSPSolver::new(distances.clone(), service_time_windows.clone(), service_times.clone());
        
        let node_ids = vec![1, 2, 3];
        let (best_route, total_duration) = tsp_solver.solve_tsp_full_enumeration(node_ids.clone(), None, None);
        let (bb_route, bb_duration) = tsp_solver.solve_tsp_branch_and_bound(node_ids.clone(), None, None);

        assert_eq!(best_route.len(), node_ids.len() + 2); // Including depot
        assert_eq!(total_duration > 0.0, true);
        assert_eq!(bb_route.len(), node_ids.len() + 2); // Including depot
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
        let solver = new_solver_with_single_node_and_tw(9.0, 13.0);
        let arrival = 10.0; // within today's time window
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 1: wait = {:?}", wait);
        assert_eq!(wait, Some(0.0));
    }
    /// Case 2: Arrival before time window => wait = earliest - arrival
    #[test]
    fn test_wait_time_computation_tw_case_2() {
        let solver = new_solver_with_single_node_and_tw(9.0, 13.0);
        let arrival = 8.0;
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 2: wait = {:?}", wait);
        assert_eq!(wait, Some(1.0));
    }
    /// Case 3: Arrival today, but today’s time window is invalid → use tomorrow
    #[test]
    fn test_wait_time_computation_tw_case_3() {
        let mut solver = new_solver_with_single_node_and_tw(9.0, 13.0);
        
        let arrival = 22.0; // near end of day 0
        let wait = solver.compute_wait_time(1, arrival);
        let expected = (1.0 * HOURS_IN_DAY as f64 + 9.0) - arrival;
        println!("Case 3: wait = {:?}, expected = {}", wait, expected);
        assert!((wait.unwrap() - expected).abs() < 1e-6);
    }
    /// Case 4: Arrival on same weekday, next week → wait = 0
    #[test]
    fn test_wait_time_computation_tw_case_4() {
        let solver = new_solver_with_single_node_and_tw(9.0, 13.0);
        let arrival = 10.0 + (HOURS_IN_DAY * DAYS_IN_PERIOD) as f64; // next Monday at 09:00
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 4: wait = {:?}", wait);
        assert_eq!(wait, Some(0.0));
    }
    /// Case 5: Arrival next week, wait till next day
    #[test]
    fn test_wait_time_computation_tw_case_5() {
        let mut solver = new_solver_with_single_node_and_tw(9.0, 13.0);
        let arrival = 14.0 + DAYS_IN_PERIOD as f64 * HOURS_IN_DAY as f64; // next Tuesday 09:00
        let expected = 9.0 + (1.0 + DAYS_IN_PERIOD as f64) * HOURS_IN_DAY as f64 - arrival; // next Wednesday 09:00 - arrival
        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 5: wait = {:?}, expected = {}", wait, expected);
        assert!((wait.unwrap() - expected).abs() < 1e-6);
    }
    /// Case 6: Arrival on last day (Sunday), after time window → wait until next week's window
    #[test]
    fn test_wait_time_computation_tw_case_6() {
        let solver = new_solver_with_single_node_and_tw(9.0, 13.0);

        let sunday = DAYS_IN_PERIOD - 1; // Sunday = day 6 (0-based)
        let arrival = (sunday as f64) * HOURS_IN_DAY as f64 + 20.0; // Sunday at 20:00

        let next_week_window_start = 9.0 + (DAYS_IN_PERIOD as f64) * HOURS_IN_DAY as f64;
        let expected = next_week_window_start - arrival;

        let wait = solver.compute_wait_time(1, arrival);
        println!("Case 6: wait = {:?}, expected = {}", wait, expected);
        assert!((wait.unwrap() - expected).abs() < 1e-6);
    }
}