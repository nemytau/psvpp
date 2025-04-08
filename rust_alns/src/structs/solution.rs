use crate::structs::{
    vessel::Vessel,
    visit::Visit,
    voyage::Voyage,
    schedule::Schedule,
    node::{Base, Installation, HasLocation},
    distance_manager::DistanceManager,
};

pub struct ProblemData {
    pub vessels: Vec<Vessel>,
    pub installations: Vec<Installation>,
    pub base: Base,
    pub distance_manager: DistanceManager,
}

impl ProblemData {
    /// Creates a new problem data object
    pub fn new(vessels: Vec<Vessel>, installations: Vec<Installation>, base: Base) -> Self {
        let installations_with_locations:  Vec<&dyn HasLocation> = installations.iter().map(|i| i as &dyn HasLocation).collect();
        let distance_manager = DistanceManager::from_base_and_installations(&base, &installations_with_locations);
        Self {
            vessels,
            installations,
            base,
            distance_manager,
        }
    }

    pub fn generate_visits(&self) -> Vec<Visit> {
        let mut visits = Vec::new();

        for installation in &self.installations {
            for _visit_num in 0..installation.visit_frequency {
                let id = visits.len();
                let installation_id = installation.id;
                // Create a new visit with the installation ID and visit number
                visits.push(Visit::new(id, installation_id));
            }
        }

        visits
    }
}

#[derive(Clone)]
pub struct Solution {
    pub voyages: Vec<Voyage>,
    pub visits: Vec<Visit>, // All visits, including unserved ones
    pub schedule: Schedule,
    pub total_cost: f64,
    pub is_feasible: bool,
}

impl Solution {
    /// Creates a new solution object
    pub fn new(visits: Vec<Visit>) -> Self {
        Self {
            voyages: Vec::new(),
            visits,
            schedule: Schedule::empty(),
            total_cost: 0.0,
            is_feasible: false,
        }
    }

    pub fn construct_initial_solution(
        &mut self,
        problem_data: &ProblemData,
    ) {
        // Initialize the solution with a greedy or random approach
        // This is a placeholder for the actual implementation
        // You can call the construct_initial_solution function here
        // e.g., construct_initial_solution(&self.visits, &problem_data.vessels);
    }
}