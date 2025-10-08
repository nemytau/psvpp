use crate::structs::solution::Solution;
use crate::structs::context::Context;
use crate::operators::registry::OperatorRegistry;
use crate::alns::acceptance;
use crate::alns::context::ALNSContext;
use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};
use std::collections::HashSet;

/// ALNS operator selection modes
#[derive(Clone)]
pub enum ALNSRunMode {
    Random,
    Weighted,
    Explicit(usize, usize),
}

/// Comprehensive metrics returned by each ALNS iteration
#[derive(Clone)]
pub struct ALNSMetrics {
    pub total_cost: f64,
    pub best_cost: f64,
    pub accepted: bool,
    pub is_new_best: bool,
    pub is_better_than_current: bool,
    pub destroy_idx: usize,
    pub repair_idx: usize,
    pub temperature: f64,
    pub stagnation_count: usize,
    pub iteration: usize,
    pub initial_cost: f64,
    pub is_complete: bool,
    pub is_feasible: bool,
    pub num_voyages: usize,
    pub num_empty_voyages: usize,
    pub num_vessels_used: usize,
    pub avg_voyage_utilization: f64,
    pub destroy_success_rates: Vec<f64>,
    pub repair_success_rates: Vec<f64>,
    pub recent_rewards: Vec<f64>,
    pub destroy_weights: Vec<f64>,
    pub repair_weights: Vec<f64>,
    pub elapsed_ms: u128,
}

/// Unified ALNS Engine - canonical implementation for all interfaces
pub struct ALNSEngine {
    pub context: Context,
    pub alns_context: ALNSContext,
    pub operator_registry: OperatorRegistry,
    pub current_solution: Solution,
    pub best_solution: Solution,
    pub initial_solution: Solution,
    pub rng: StdRng,
    pub iteration: usize,
    pub max_iterations: usize,
    pub temperature: f64,
    pub theta: f64,
    pub weight_update_interval: usize,
    pub stagnation_count: usize,
    pub initial_cost: f64,
}

impl ALNSEngine {
    /// Create new engine from problem instance with configurable parameters
    pub fn new_from_instance(
        problem_instance: &str,
        seed: u64,
        temperature: f64,
        theta: f64,
        weight_update_interval: usize,
        max_iterations: usize,
    ) -> Result<Self, String> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        // The sample directory is in the parent project directory, not in rust_alns
        let project_dir = std::path::Path::new(manifest_dir).parent().unwrap();
        let (installations_path, vessels_path, base_path) = match problem_instance {
            "SMALL_1" => (
                format!("{}/sample/installations/SMALL_1/i_test1.csv", project_dir.display()),
                format!("{}/sample/vessels/SMALL_1/v_test1.csv", project_dir.display()),
                format!("{}/sample/base/SMALL_1/b_test1.csv", project_dir.display())
            ),
            _ => return Err(format!("Unknown problem instance: {}", problem_instance)),
        };

        let data = crate::structs::data_loader::read_data(&installations_path, &vessels_path, &base_path)
            .map_err(|e| format!("Failed to load data: {}", e))?;

        let problem_data = crate::structs::problem_data::ProblemData::new(
            data.vessels.clone(),
            data.installations.clone(),
            data.base.clone(),
        );
        let tsp_solver = crate::utils::tsp_solver::TSPSolver::new_from_problem_data(&problem_data);
        let context = Context { problem: problem_data, tsp_solver };

        let mut rng = StdRng::seed_from_u64(seed);
        let mut initial_solution = crate::operators::initial_solution::construct_initial_solution(&context, &mut rng);
        initial_solution.update_total_cost(&context);

        let mut operator_registry = OperatorRegistry::new();
        Self::setup_operators(&mut operator_registry);

        let n_destroy = operator_registry.destroy_operators.len();
        let n_repair = operator_registry.repair_operators.len();
        let alns_context = ALNSContext {
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
            reaction_factor: 0.2,
            reward_values: vec![33.0, 9.0, 3.0],
        };

        let initial_cost = initial_solution.total_cost;

        Ok(Self {
            context,
            alns_context,
            operator_registry,
            current_solution: initial_solution.clone(),
            best_solution: initial_solution.clone(),
            initial_solution,
            rng: StdRng::seed_from_u64(seed),
            iteration: 0,
            max_iterations,
            temperature,
            theta,
            weight_update_interval,
            stagnation_count: 0,
            initial_cost,
        })
    }

    /// Setup standard ALNS operators
    pub fn setup_operators(registry: &mut OperatorRegistry) {
        registry.add_destroy_operator(Box::new(
            crate::operators::destroy::shaw_removal::ShawRemoval {
                xi_min: 0.2, xi_max: 0.4, p: 5.0, alpha: 1.0, beta: 5.0, phi: 2.0,
            }
        ));
        registry.add_destroy_operator(Box::new(
            crate::operators::destroy::random_visit_removal_in_voyages::RandomVisitRemovalInVoyages {
                xi_min: 0.2, xi_max: 0.4,
            }
        ));
        registry.add_destroy_operator(Box::new(
            crate::operators::destroy::worst_visit_removal_in_voyages::WorstVisitRemovalInVoyages {
                xi_min: 0.2, xi_max: 0.4, p: 5.0,
            }
        ));
        registry.add_repair_operator(Box::new(
            crate::operators::repair::deep_greedy_insertion::DeepGreedyInsertion {}
        ));
        registry.add_repair_operator(Box::new(
            crate::operators::repair::k_regret_insertion::KRegretInsertion { k: 2 }
        ));
        registry.add_repair_operator(Box::new(
            crate::operators::repair::k_regret_insertion::KRegretInsertion { k: 3 }
        ));
    }

    /// Get operator information for external interfaces
    pub fn get_operator_info() -> (Vec<String>, Vec<String>) {
        let destroy_operators = vec![
            "shaw_removal".to_string(),
            "random_visit_removal".to_string(), 
            "worst_visit_removal".to_string()
        ];
        
        let repair_operators = vec![
            "deep_greedy_insertion".to_string(),
            "k_regret_2".to_string(),
            "k_regret_3".to_string()
        ];
        
        (destroy_operators, repair_operators)
    }

    /// Run a single ALNS iteration with specified mode
    pub fn run_iteration(
        &mut self,
        mode: ALNSRunMode,
        iteration: usize,
    ) -> Result<ALNSMetrics, String> {
        // Select operator indices based on mode
        let (destroy_operator_idx, repair_operator_idx) = match mode {
            ALNSRunMode::Random => {
                let destroy_idx = self.rng.gen_range(0..self.operator_registry.destroy_operators.len());
                let repair_idx = self.rng.gen_range(0..self.operator_registry.repair_operators.len());
                (destroy_idx, repair_idx)
            },
            ALNSRunMode::Weighted => {
                let destroy_idx = self.alns_context.pick_destroy_operator_idx();
                let repair_idx = self.alns_context.pick_repair_operator_idx();
                (destroy_idx, repair_idx)
            },
            ALNSRunMode::Explicit(destroy_idx, repair_idx) => (destroy_idx, repair_idx),
        };

        self.run_iteration_internal(destroy_operator_idx, repair_operator_idx, iteration)
    }

    /// Backward compatibility method for explicit operator selection
    pub fn run_iteration_explicit(
        &mut self,
        destroy_operator_idx: usize,
        repair_operator_idx: usize,
        iteration: usize,
    ) -> Result<ALNSMetrics, String> {
        self.run_iteration(ALNSRunMode::Explicit(destroy_operator_idx, repair_operator_idx), iteration)
    }

    /// Internal implementation of ALNS iteration logic
    fn run_iteration_internal(
        &mut self,
        destroy_operator_idx: usize,
        repair_operator_idx: usize,
        iteration: usize,
    ) -> Result<ALNSMetrics, String> {
        let start_time = std::time::Instant::now();
        
        if destroy_operator_idx >= self.operator_registry.destroy_operators.len() {
            return Err(format!("Invalid destroy operator index: {}", destroy_operator_idx));
        }
        if repair_operator_idx >= self.operator_registry.repair_operators.len() {
            return Err(format!("Invalid repair operator index: {}", repair_operator_idx));
        }

        let destroy_op = self.operator_registry.get_destroy_operator(destroy_operator_idx);
        let repair_op = self.operator_registry.get_repair_operator(repair_operator_idx);

        let mut candidate_solution = self.current_solution.clone();

        // Apply destroy and repair operators
        destroy_op.apply(&mut candidate_solution, &self.context, &mut self.rng);
        candidate_solution.ensure_consistency_updated(&self.context);
        candidate_solution.add_idle_vessel_and_add_empty_voyages(&self.context);
        repair_op.apply(&mut candidate_solution, &self.context, &mut self.rng);
        candidate_solution.ensure_consistency_updated(&self.context);
        candidate_solution.update_total_cost(&self.context);

        let candidate_cost = candidate_solution.total_cost;
        let current_cost = self.current_solution.total_cost;
        let best_cost = self.best_solution.total_cost;

        // Acceptance decision
        let accept = acceptance::accept(current_cost, candidate_cost, self.temperature, &mut self.rng);
        let mut accepted = false;
        let mut is_new_best = false;
        let mut is_better_than_current = false;

        if candidate_cost < best_cost {
            self.best_solution = candidate_solution.clone();
            self.current_solution = candidate_solution;
            accepted = true;
            is_new_best = true;
            is_better_than_current = true;
            self.stagnation_count = 0;
        } else if candidate_cost < current_cost {
            self.current_solution = candidate_solution;
            accepted = true;
            is_better_than_current = true;
            self.stagnation_count = 0;
        } else if accept {
            self.current_solution = candidate_solution;
            accepted = true;
            self.stagnation_count += 1;
        } else {
            self.stagnation_count += 1;
        }

        // Update temperature
        self.temperature = acceptance::cool_down(self.temperature, self.theta, iteration + 1);
        self.iteration = iteration;
        self.alns_context.iteration = iteration;

        // Reward operators based on performance
        if is_new_best {
            self.alns_context.reward_operator("destroy", destroy_operator_idx, 0);
            self.alns_context.reward_operator("repair", repair_operator_idx, 0);
        } else if is_better_than_current {
            self.alns_context.reward_operator("destroy", destroy_operator_idx, 1);
            self.alns_context.reward_operator("repair", repair_operator_idx, 1);
        } else if accepted {
            self.alns_context.reward_operator("destroy", destroy_operator_idx, 2);
            self.alns_context.reward_operator("repair", repair_operator_idx, 2);
        }

        // Update operator weights at configurable intervals
        if (iteration + 1) % self.weight_update_interval == 0 {
            self.alns_context.update_operator_weights("destroy");
            self.alns_context.update_operator_weights("repair");
            self.alns_context.reset_segment_scores();
        }

        // Calculate elapsed time
        let elapsed_ms = start_time.elapsed().as_millis();

        // Log iteration progress
        log::info!(
            "Iter {:3}: cost={:.4}, best={:.4}, temp={:6.1}, destroy={}, repair={}, accepted={}, elapsed={}ms",
            iteration + 1,
            candidate_cost,
            self.best_solution.total_cost,
            self.temperature,
            destroy_operator_idx,
            repair_operator_idx,
            accepted,
            elapsed_ms
        );

        log::debug!("Iteration {} completed in {}ms", iteration + 1, elapsed_ms);

        // Calculate solution metrics
        let current_solution = &self.current_solution;
        let best_solution = &self.best_solution;

        let mut temp_solution = current_solution.clone();
        let is_feasible = temp_solution.is_fully_feasible(&self.context);
        let is_complete = current_solution.is_complete_solution();

        let non_empty_voyages = current_solution.voyages.iter().filter(|v| !v.borrow().visit_ids.is_empty()).count();
        let empty_voyages = current_solution.voyages.len() - non_empty_voyages;

        let mut vessels_used = HashSet::new();
        for voyage in &current_solution.voyages { 
            if !voyage.borrow().visit_ids.is_empty() { 
                vessels_used.insert(voyage.borrow().vessel_id); 
            } 
        }

        let total_visits: usize = current_solution.voyages.iter().map(|v| v.borrow().visit_ids.len()).sum();
        let avg_utilization = if non_empty_voyages > 0 { 
            total_visits as f64 / non_empty_voyages as f64 
        } else { 
            0.0 
        };

        let destroy_success_rates: Vec<f64> = self.alns_context.destroy_operator_scores.iter()
            .zip(self.alns_context.destroy_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        let repair_success_rates: Vec<f64> = self.alns_context.repair_operator_scores.iter()
            .zip(self.alns_context.repair_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        let recent_rewards: Vec<f64> = self.alns_context.cost_history.iter().rev().take(5).cloned().collect();

        Ok(ALNSMetrics {
            total_cost: current_solution.total_cost,
            best_cost: best_solution.total_cost,
            accepted,
            is_new_best,
            is_better_than_current,
            destroy_idx: destroy_operator_idx,
            repair_idx: repair_operator_idx,
            temperature: self.temperature,
            stagnation_count: self.stagnation_count,
            iteration: self.iteration,
            initial_cost: self.initial_cost,
            is_complete,
            is_feasible,
            num_voyages: non_empty_voyages,
            num_empty_voyages: empty_voyages,
            num_vessels_used: vessels_used.len(),
            avg_voyage_utilization: avg_utilization,
            destroy_success_rates,
            repair_success_rates,
            recent_rewards,
            destroy_weights: self.alns_context.destroy_operator_weights.clone(),
            repair_weights: self.alns_context.repair_operator_weights.clone(),
            elapsed_ms,
        })
    }

    /// Run the full ALNS algorithm for max_iterations
    pub fn run(&mut self) -> ALNSMetrics {
        let mut last_metrics = ALNSMetrics {
            total_cost: self.current_solution.total_cost,
            best_cost: self.best_solution.total_cost,
            accepted: false,
            is_new_best: false,
            is_better_than_current: false,
            destroy_idx: 0,
            repair_idx: 0,
            temperature: self.temperature,
            stagnation_count: self.stagnation_count,
            iteration: 0,
            initial_cost: self.initial_cost,
            is_complete: self.current_solution.is_complete_solution(),
            is_feasible: self.current_solution.clone().is_fully_feasible(&self.context),
            num_voyages: 0,
            num_empty_voyages: 0,
            num_vessels_used: 0,
            avg_voyage_utilization: 0.0,
            destroy_success_rates: vec![],
            repair_success_rates: vec![],
            recent_rewards: vec![],
            destroy_weights: vec![],
            repair_weights: vec![],
            elapsed_ms: 0,
        };

        for iter in 0..self.max_iterations {
            let destroy_idx = self.alns_context.pick_destroy_operator_idx();
            let repair_idx = self.alns_context.pick_repair_operator_idx();
            
            match self.run_iteration(ALNSRunMode::Explicit(destroy_idx, repair_idx), iter) {
                Ok(metrics) => last_metrics = metrics,
                Err(e) => {
                    log::error!("Error in iteration {}: {}", iter, e);
                    break;
                }
            }
        }

        last_metrics
    }

    /// Export current solution to JSON file
    pub fn export_solution(&self, filepath: &str) {
        crate::utils::serialization::dump_schedule_to_json(
            &self.current_solution,
            &self.context.problem.vessels,
            filepath,
            &self.context,
        );
    }
}

/// Enable file logging for ALNS operations
pub fn enable_file_logging(log_path: &str) -> Result<(), String> {
    use std::fs::OpenOptions;
    use std::io::Write;
    
    let target = Box::new(OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(log_path)
        .map_err(|e| format!("Failed to open log file {}: {}", log_path, e))?);
    
    env_logger::Builder::from_default_env()
        .target(env_logger::Target::Pipe(target))
        .try_init()
        .map_err(|e| format!("Logger already initialized or configuration error: {}", e))?;
    
    Ok(())
}

/// Enable console logging for ALNS operations
pub fn enable_console_logging() -> Result<(), String> {
    env_logger::Builder::from_default_env()
        .target(env_logger::Target::Stdout)
        .try_init()
        .map_err(|e| format!("Logger already initialized or configuration error: {}", e))
}
