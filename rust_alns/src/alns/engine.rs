use crate::alns::acceptance;
use crate::alns::context::ALNSContext;
use crate::operators::improvement::deep_relocation::DeepRelocation;
use crate::operators::improvement::deep_swap::DeepSwap;
use crate::operators::improvement::fleet_and_cost_reduction::FleetAndCostReduction;
use crate::operators::improvement::voyage_number_reduction::VoyageNumberReduction;
use crate::operators::registry::OperatorRegistry;
use crate::structs::context::Context;
use crate::structs::solution::Solution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::f64::{INFINITY, NEG_INFINITY};
use std::path::{Path, PathBuf};

/// High-level algorithm variants for the ALNS engine
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ALNSAlgorithmMode {
    Baseline,
    Kisialiou,
    ReinforcementLearning,
}

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
    pub destroy_idx: Option<usize>,
    pub repair_idx: Option<usize>,
    pub destroy_operator_name: Option<String>,
    pub repair_operator_name: Option<String>,
    pub destroy_operator_type: Option<String>,
    pub repair_operator_type: Option<String>,
    pub destroy_removed_requests: Option<usize>,
    pub repair_inserted_requests: Option<usize>,
    pub destroy_operator_type_id: Option<i32>,
    pub repair_operator_type_id: Option<i32>,
    pub improvement_idx: Option<usize>,
    pub improvement_operator_name: Option<String>,
    pub improvement_operator_type: Option<String>,
    pub improvement_operator_type_id: Option<i32>,
    pub improvement_sequence: Vec<usize>,
    pub improvement_costs: Vec<f64>,  // Cost after each improvement operator
    pub improvement_step_metrics: Vec<ALNSImprovementStepMetric>,
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
    pub avg_vessel_load_utilization: f64,
    pub min_vessel_load_utilization: f64,
    pub max_vessel_load_utilization: f64,
    pub avg_vessel_time_utilization: f64,
    pub min_vessel_time_utilization: f64,
    pub max_vessel_time_utilization: f64,
    pub destroy_success_rates: Vec<f64>,
    pub repair_success_rates: Vec<f64>,
    pub recent_rewards: Vec<f64>,
    pub destroy_weights: Vec<f64>,
    pub repair_weights: Vec<f64>,
    pub elapsed_ms: u128,
}

#[derive(Clone)]
pub struct ALNSImprovementStepMetric {
    pub operator_idx: usize,
    pub operator_name: String,
    pub sequence_position: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub cost_delta: f64,
}

#[derive(Clone)]
pub struct ALNSRestartSummary {
    pub restart_index: usize,
    pub seed: u64,
    pub initial_cost: f64,
    pub best_cost: f64,
    pub final_cost: f64,
    pub iterations_completed: usize,
    pub elapsed_ms: u128,
    pub best_improvement_pct: f64,
}

#[derive(Clone)]
pub struct ALNSRunWithRestartsResult {
    pub global_metrics: ALNSMetrics,
    pub restart_summaries: Vec<ALNSRestartSummary>,
}

#[derive(Clone)]
pub struct ALNSEngineSnapshot {
    pub current_solution: Solution,
    pub best_solution: Solution,
    pub initial_solution: Solution,
    pub alns_context: ALNSContext,
    pub rng: StdRng,
    pub iteration: usize,
    pub max_iterations: usize,
    pub temperature: f64,
    pub theta: f64,
    pub weight_update_interval: usize,
    pub aggressive_search_factor: f64,
    pub stagnation_count: usize,
    pub initial_cost: f64,
    pub base_seed: u64,
    pub initial_temperature: f64,
    pub algorithm_mode: ALNSAlgorithmMode,
}

/// Unified ALNS Engine - canonical implementation for all interfaces
pub struct ALNSEngine {
    pub context: Context,
    pub alns_context: ALNSContext,
    pub operator_registry: OperatorRegistry,
    pub destroy_operator_labels: Vec<String>,
    pub repair_operator_labels: Vec<String>,
    pub improvement_operator_labels: Vec<String>,
    pub current_solution: Solution,
    pub best_solution: Solution,
    pub initial_solution: Solution,
    pub rng: StdRng,
    pub iteration: usize,
    pub max_iterations: usize,
    pub temperature: f64,
    pub theta: f64,
    pub weight_update_interval: usize,
    pub aggressive_search_factor: f64,
    pub stagnation_count: usize,
    pub initial_cost: f64,
    pub base_seed: u64,
    pub initial_temperature: f64,
    pub algorithm_mode: ALNSAlgorithmMode,
}

impl ALNSEngine {
    /// Create new engine from problem instance with configurable parameters
    pub fn new_from_instance(
        problem_instance: &str,
        seed: u64,
        temperature: f64,
        theta: f64,
        weight_update_interval: usize,
        aggressive_search_factor: f64,
        max_iterations: usize,
        algorithm_mode: ALNSAlgorithmMode,
    ) -> Result<Self, String> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        // The sample directory is in the parent project directory, not in rust_alns
        let project_dir = std::path::Path::new(manifest_dir).parent().unwrap();
        let (installations_path, vessels_path, base_path) = {
            let candidate_path = Path::new(problem_instance);
            if candidate_path.exists() {
                let directory: PathBuf = if candidate_path.is_dir() {
                    candidate_path.to_path_buf()
                } else {
                    candidate_path
                        .parent()
                        .ok_or_else(|| {
                            format!(
                                "Provided dataset path has no parent directory: {}",
                                problem_instance
                            )
                        })?
                        .to_path_buf()
                };

                let installations = directory.join("installations.csv");
                let vessels = directory.join("vessels.csv");
                let base = directory.join("base.csv");

                if !installations.exists() {
                    return Err(format!(
                        "installations.csv not found for dataset: {}",
                        directory.display()
                    ));
                }
                if !vessels.exists() {
                    return Err(format!(
                        "vessels.csv not found for dataset: {}",
                        directory.display()
                    ));
                }
                if !base.exists() {
                    return Err(format!(
                        "base.csv not found for dataset: {}",
                        directory.display()
                    ));
                }

                (
                    installations.to_string_lossy().into_owned(),
                    vessels.to_string_lossy().into_owned(),
                    base.to_string_lossy().into_owned(),
                )
            } else {
                match problem_instance {
                    "SMALL_1" => (
                        format!(
                            "{}/sample/installations/SMALL_1/i_test1.csv",
                            project_dir.display()
                        ),
                        format!(
                            "{}/sample/vessels/SMALL_1/v_test1.csv",
                            project_dir.display()
                        ),
                        format!("{}/sample/base/SMALL_1/b_test1.csv", project_dir.display()),
                    ),
                    _ => {
                        let candidate_relative = project_dir.join(problem_instance);
                        if candidate_relative.exists() {
                            let directory = if candidate_relative.is_dir() {
                                candidate_relative
                            } else {
                                candidate_relative
                                    .parent()
                                    .ok_or_else(|| {
                                        format!(
                                            "Provided dataset path has no parent directory: {}",
                                            problem_instance
                                        )
                                    })?
                                    .to_path_buf()
                            };
                            let installations = directory.join("installations.csv");
                            let vessels = directory.join("vessels.csv");
                            let base = directory.join("base.csv");
                            if !installations.exists() || !vessels.exists() || !base.exists() {
                                return Err(format!(
                                    "Dataset folder is missing required CSV files: {}",
                                    directory.display()
                                ));
                            }
                            (
                                installations.to_string_lossy().into_owned(),
                                vessels.to_string_lossy().into_owned(),
                                base.to_string_lossy().into_owned(),
                            )
                        } else {
                            return Err(format!(
                                "Unknown problem instance or dataset path: {}",
                                problem_instance
                            ));
                        }
                    }
                }
            }
        };

        let data =
            crate::structs::data_loader::read_data(&installations_path, &vessels_path, &base_path)
                .map_err(|e| format!("Failed to load data: {}", e))?;

        let problem_data = crate::structs::problem_data::ProblemData::new(
            data.vessels.clone(),
            data.installations.clone(),
            data.base.clone(),
        );
        let tsp_solver = crate::utils::tsp_solver::TSPSolver::new_from_problem_data(&problem_data);
        let context = Context {
            problem: problem_data,
            tsp_solver,
        };

        let mut rng = StdRng::seed_from_u64(seed);
        let mut initial_solution =
            crate::operators::initial_solution::construct_initial_solution(&context, &mut rng);
        initial_solution.update_total_cost(&context);

        let mut operator_registry = OperatorRegistry::new();
        Self::setup_operators(&mut operator_registry);

        let (destroy_operator_labels, repair_operator_labels, improvement_operator_labels) =
            Self::get_operator_info();

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
        let aggressive_search_factor = aggressive_search_factor.clamp(0.0, 1.0);

        Ok(Self {
            context,
            alns_context,
            operator_registry,
            destroy_operator_labels,
            repair_operator_labels,
            improvement_operator_labels,
            current_solution: initial_solution.clone(),
            best_solution: initial_solution.clone(),
            initial_solution,
            rng: StdRng::seed_from_u64(seed),
            iteration: 0,
            max_iterations,
            temperature,
            theta,
            weight_update_interval,
            aggressive_search_factor,
            stagnation_count: 0,
            initial_cost,
            base_seed: seed,
            initial_temperature: temperature,
            algorithm_mode,
        })
    }

    /// Capture a clone of the current engine state for later reuse.
    pub fn create_snapshot(&self) -> ALNSEngineSnapshot {
        ALNSEngineSnapshot {
            current_solution: self.current_solution.clone(),
            best_solution: self.best_solution.clone(),
            initial_solution: self.initial_solution.clone(),
            alns_context: self.alns_context.clone(),
            rng: self.rng.clone(),
            iteration: self.iteration,
            max_iterations: self.max_iterations,
            temperature: self.temperature,
            theta: self.theta,
            weight_update_interval: self.weight_update_interval,
            aggressive_search_factor: self.aggressive_search_factor,
            stagnation_count: self.stagnation_count,
            initial_cost: self.initial_cost,
            base_seed: self.base_seed,
            initial_temperature: self.initial_temperature,
            algorithm_mode: self.algorithm_mode,
        }
    }

    /// Restore the engine state from a previously captured snapshot.
    pub fn apply_snapshot(&mut self, snapshot: &ALNSEngineSnapshot) {
        self.current_solution = snapshot.current_solution.clone();
        self.best_solution = snapshot.best_solution.clone();
        self.initial_solution = snapshot.initial_solution.clone();
        self.alns_context = snapshot.alns_context.clone();
        self.rng = snapshot.rng.clone();
        self.iteration = snapshot.iteration;
        self.max_iterations = snapshot.max_iterations;
        self.temperature = snapshot.temperature;
        self.theta = snapshot.theta;
        self.weight_update_interval = snapshot.weight_update_interval;
        self.aggressive_search_factor = snapshot.aggressive_search_factor;
        self.stagnation_count = snapshot.stagnation_count;
        self.initial_cost = snapshot.initial_cost;
        self.base_seed = snapshot.base_seed;
        self.initial_temperature = snapshot.initial_temperature;
        self.algorithm_mode = snapshot.algorithm_mode;
    }

    fn create_default_metrics(&self) -> ALNSMetrics {
        ALNSMetrics {
            total_cost: self.current_solution.total_cost,
            best_cost: self.best_solution.total_cost,
            accepted: false,
            is_new_best: false,
            is_better_than_current: false,
            destroy_idx: None,
            repair_idx: None,
            destroy_operator_name: None,
            repair_operator_name: None,
            destroy_operator_type: None,
            repair_operator_type: None,
            destroy_operator_type_id: None,
            repair_operator_type_id: None,
            improvement_idx: None,
            improvement_operator_name: None,
            improvement_operator_type: None,
            improvement_operator_type_id: None,
            improvement_sequence: Vec::new(),
            improvement_costs: Vec::new(),
            improvement_step_metrics: Vec::new(),
            destroy_removed_requests: None,
            repair_inserted_requests: None,
            temperature: self.temperature,
            stagnation_count: self.stagnation_count,
            iteration: 0,
            initial_cost: self.initial_cost,
            is_complete: self.current_solution.is_complete_solution(),
            is_feasible: self
                .current_solution
                .clone()
                .is_fully_feasible(&self.context),
            num_voyages: 0,
            num_empty_voyages: 0,
            num_vessels_used: 0,
            avg_voyage_utilization: 0.0,
            avg_vessel_load_utilization: 0.0,
            min_vessel_load_utilization: 0.0,
            max_vessel_load_utilization: 0.0,
            avg_vessel_time_utilization: 0.0,
            min_vessel_time_utilization: 0.0,
            max_vessel_time_utilization: 0.0,
            destroy_success_rates: vec![],
            repair_success_rates: vec![],
            recent_rewards: vec![],
            destroy_weights: vec![],
            repair_weights: vec![],
            elapsed_ms: 0,
        }
    }

    fn reset_for_restart(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);

        let mut initial_solution =
            crate::operators::initial_solution::construct_initial_solution(&self.context, &mut self.rng);
        initial_solution.update_total_cost(&self.context);

        let n_destroy = self.operator_registry.destroy_operators.len();
        let n_repair = self.operator_registry.repair_operators.len();
        self.alns_context = ALNSContext {
            iteration: 0,
            temperature: self.initial_temperature,
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

        self.initial_cost = initial_solution.total_cost;
        self.initial_solution = initial_solution.clone();
        self.current_solution = initial_solution.clone();
        self.best_solution = initial_solution;
        self.iteration = 0;
        self.temperature = self.initial_temperature;
        self.stagnation_count = 0;
    }

    fn is_in_aggressive_phase(&self, iteration: usize) -> bool {
        let threshold = (self.aggressive_search_factor * self.max_iterations as f64).floor() as usize;
        iteration >= threshold
    }

    /// Setup standard ALNS operators
    pub fn setup_operators(registry: &mut OperatorRegistry) {
        registry.add_destroy_operator(Box::new(
            crate::operators::destroy::shaw_removal::ShawRemoval {
                xi_min: 0.2,
                xi_max: 0.4,
                p: 5.0,
                alpha: 1.0,
                beta: 5.0,
                phi: 2.0,
            },
        ));
        registry.add_destroy_operator(Box::new(
            crate::operators::destroy::random_visit_removal_in_voyages::RandomVisitRemovalInVoyages {
                xi_min: 0.2, xi_max: 0.4,
            }
        ));
        registry.add_destroy_operator(Box::new(
            crate::operators::destroy::worst_visit_removal_in_voyages::WorstVisitRemovalInVoyages {
                xi_min: 0.2,
                xi_max: 0.4,
                p: 5.0,
            },
        ));
        registry.add_repair_operator(Box::new(
            crate::operators::repair::deep_greedy_insertion::DeepGreedyInsertion {},
        ));
        registry.add_repair_operator(Box::new(
            crate::operators::repair::k_regret_insertion::KRegretInsertion { k: 2 },
        ));
        registry.add_repair_operator(Box::new(
            crate::operators::repair::k_regret_insertion::KRegretInsertion { k: 3 },
        ));

        registry.add_improvement_operator(Box::new(VoyageNumberReduction));
        registry.add_improvement_operator(Box::new(FleetAndCostReduction));
        registry.add_improvement_operator(Box::new(DeepRelocation));
        registry.add_improvement_operator(Box::new(DeepSwap));
    }

    /// Get operator information for external interfaces
    pub fn get_operator_info() -> (Vec<String>, Vec<String>, Vec<String>) {
        let destroy_operators = vec![
            "shaw_removal".to_string(),
            "random_visit_removal".to_string(),
            "worst_visit_removal".to_string(),
        ];

        let repair_operators = vec![
            "deep_greedy_insertion".to_string(),
            "k_regret_2".to_string(),
            "k_regret_3".to_string(),
        ];

        let improvement_operators = vec![
            "voyage_number_reduction".to_string(),
            "fleet_and_cost_reduction".to_string(),
            "deep_relocation".to_string(),
            "deep_swap".to_string(),
        ];

        (destroy_operators, repair_operators, improvement_operators)
    }

    /// Run a single ALNS iteration with specified mode
    pub fn run_iteration(
        &mut self,
        mode: ALNSRunMode,
        iteration: usize,
        improvement_operator_idx: Option<usize>,
    ) -> Result<ALNSMetrics, String> {
        let (destroy_operator_idx, repair_operator_idx) = self.resolve_operator_pair(mode);

        let improvement_sequence: Vec<usize> = match self.algorithm_mode {
            ALNSAlgorithmMode::Baseline | ALNSAlgorithmMode::ReinforcementLearning => {
                improvement_operator_idx.iter().copied().collect()
            }
            ALNSAlgorithmMode::Kisialiou => {
                let mut sequence = self.compute_kisialiou_improvement_sequence();
                if let Some(extra_idx) = improvement_operator_idx {
                    sequence.push(extra_idx);
                }
                sequence
            }
        };

        self.run_iteration_internal(
            destroy_operator_idx,
            repair_operator_idx,
            &improvement_sequence,
            iteration,
        )
    }

    /// Backward compatibility method for explicit operator selection
    pub fn run_iteration_explicit(
        &mut self,
        destroy_operator_idx: usize,
        repair_operator_idx: usize,
        improvement_operator_idx: Option<usize>,
        iteration: usize,
    ) -> Result<ALNSMetrics, String> {
        self.run_iteration(
            ALNSRunMode::Explicit(destroy_operator_idx, repair_operator_idx),
            iteration,
            improvement_operator_idx,
        )
    }

    /// Internal implementation of ALNS iteration logic
    fn run_iteration_internal(
        &mut self,
        destroy_operator_idx: usize,
        repair_operator_idx: usize,
        improvement_sequence: &[usize],
        iteration: usize,
    ) -> Result<ALNSMetrics, String> {
        let start_time = std::time::Instant::now();

        if destroy_operator_idx >= self.operator_registry.destroy_operators.len() {
            return Err(format!(
                "Invalid destroy operator index: {}",
                destroy_operator_idx
            ));
        }
        if repair_operator_idx >= self.operator_registry.repair_operators.len() {
            return Err(format!(
                "Invalid repair operator index: {}",
                repair_operator_idx
            ));
        }
        for &idx in improvement_sequence {
            if idx >= self.operator_registry.improvement_operators.len() {
                return Err(format!("Invalid improvement operator index: {}", idx));
            }
        }

        let destroy_op = self
            .operator_registry
            .get_destroy_operator(destroy_operator_idx);
        let repair_op = self
            .operator_registry
            .get_repair_operator(repair_operator_idx);

        let improvement_operator_name = improvement_sequence.last().map(|idx| {
            let index = *idx;
            self.improvement_operator_labels
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("improvement_{}", index))
        });
        let improvement_operator_type = improvement_operator_name.as_ref().map(|_| "improvement".to_string());
        let improvement_operator_type_id = improvement_operator_name
            .as_ref()
            .map(|_| 2);

        let destroy_operator_name = self
            .destroy_operator_labels
            .get(destroy_operator_idx)
            .cloned()
            .unwrap_or_else(|| format!("destroy_{}", destroy_operator_idx));
        let repair_operator_name = self
            .repair_operator_labels
            .get(repair_operator_idx)
            .cloned()
            .unwrap_or_else(|| format!("repair_{}", repair_operator_idx));
        let destroy_operator_type_id = 0;
        let repair_operator_type_id = 1;

        let mut candidate_solution = self.current_solution.clone();

        // Apply destroy and repair operators
        // TODO: expose destroy_removed_requests + fraction_removed to Python result
        destroy_op.apply(&mut candidate_solution, &self.context, &mut self.rng);
        candidate_solution.ensure_consistency_updated(&self.context);
        candidate_solution.add_idle_vessel_and_add_empty_voyages(&self.context);
        repair_op.apply(&mut candidate_solution, &self.context, &mut self.rng);
        candidate_solution.ensure_consistency_updated(&self.context);

        // Capture cost after destroy+repair and track each improvement operator separately
        candidate_solution.update_total_cost(&self.context);
        let mut previous_improvement_cost = candidate_solution.total_cost;

        let mut improvement_costs = Vec::with_capacity(improvement_sequence.len());
        let mut improvement_step_metrics = Vec::with_capacity(improvement_sequence.len());
        for &idx in improvement_sequence {
            let improvement_op = self.operator_registry.get_improvement_operator(idx);
            let cost_before = previous_improvement_cost;
            improvement_op.apply(&mut candidate_solution, &self.context, &mut self.rng);
            candidate_solution.ensure_consistency_updated(&self.context);
            candidate_solution.update_total_cost(&self.context);
            let cost_after = candidate_solution.total_cost;
            improvement_costs.push(cost_after);

            let operator_name = self
                .improvement_operator_labels
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("improvement_{}", idx));
            let sequence_position = improvement_step_metrics.len();
            improvement_step_metrics.push(ALNSImprovementStepMetric {
                operator_idx: idx,
                operator_name,
                sequence_position,
                cost_before,
                cost_after,
                cost_delta: cost_after - cost_before,
            });
            previous_improvement_cost = cost_after;
        }

        // Final cost update (redundant if improvements were applied, but safe)
        candidate_solution.update_total_cost(&self.context);

        let candidate_cost = candidate_solution.total_cost;
        let current_cost = self.current_solution.total_cost;
        let best_cost = self.best_solution.total_cost;

        // Acceptance decision
        let accept = acceptance::accept(
            current_cost,
            candidate_cost,
            self.temperature,
            &mut self.rng,
        );
        let in_aggressive_phase = self.is_in_aggressive_phase(iteration);
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
        } else if in_aggressive_phase {
            self.current_solution = self.best_solution.clone();
            self.stagnation_count += 1;
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
            self.alns_context
                .reward_operator("destroy", destroy_operator_idx, 0);
            self.alns_context
                .reward_operator("repair", repair_operator_idx, 0);
        } else if is_better_than_current {
            self.alns_context
                .reward_operator("destroy", destroy_operator_idx, 1);
            self.alns_context
                .reward_operator("repair", repair_operator_idx, 1);
        } else if accepted {
            self.alns_context
                .reward_operator("destroy", destroy_operator_idx, 2);
            self.alns_context
                .reward_operator("repair", repair_operator_idx, 2);
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
        let improvement_marker = if improvement_sequence.is_empty() {
            "-".to_string()
        } else {
            improvement_sequence
                .iter()
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join("/")
        };

        log::info!(
            "Iter {:3}: cost={:.4}, best={:.4}, temp={:6.1}, destroy={}, repair={}, impr={}, accepted={}, elapsed={}ms",
            iteration + 1,
            candidate_cost,
            self.best_solution.total_cost,
            self.temperature,
            destroy_operator_idx,
            repair_operator_idx,
            improvement_marker,
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

        let structure_metrics = compute_solution_structure_metrics(current_solution, &self.context);

        let destroy_success_rates: Vec<f64> = self
            .alns_context
            .destroy_operator_scores
            .iter()
            .zip(self.alns_context.destroy_operator_counts.iter())
            .map(|(score, count)| {
                if *count > 0 {
                    score / (*count as f64)
                } else {
                    0.5
                }
            })
            .collect();
        let repair_success_rates: Vec<f64> = self
            .alns_context
            .repair_operator_scores
            .iter()
            .zip(self.alns_context.repair_operator_counts.iter())
            .map(|(score, count)| {
                if *count > 0 {
                    score / (*count as f64)
                } else {
                    0.5
                }
            })
            .collect();
        let recent_rewards: Vec<f64> = self
            .alns_context
            .cost_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        Ok(ALNSMetrics {
            total_cost: current_solution.total_cost,
            best_cost: best_solution.total_cost,
            accepted,
            is_new_best,
            is_better_than_current,
            destroy_idx: Some(destroy_operator_idx),
            repair_idx: Some(repair_operator_idx),
            destroy_operator_name: Some(destroy_operator_name),
            repair_operator_name: Some(repair_operator_name),
            destroy_operator_type: Some("destroy".to_string()),
            repair_operator_type: Some("repair".to_string()),
            destroy_operator_type_id: Some(destroy_operator_type_id),
            repair_operator_type_id: Some(repair_operator_type_id),
            improvement_idx: improvement_sequence.last().copied(),
            improvement_operator_name,
            improvement_operator_type,
            improvement_operator_type_id,
            improvement_sequence: improvement_sequence.to_vec(),
            improvement_costs: improvement_costs.clone(),
            improvement_step_metrics,
            destroy_removed_requests: None,
            repair_inserted_requests: None,
            temperature: self.temperature,
            stagnation_count: self.stagnation_count,
            iteration: self.iteration,
            initial_cost: self.initial_cost,
            is_complete,
            is_feasible,
            num_voyages: structure_metrics.num_voyages,
            num_empty_voyages: structure_metrics.num_empty_voyages,
            num_vessels_used: structure_metrics.num_vessels_used,
            avg_voyage_utilization: structure_metrics.avg_voyage_utilization,
            avg_vessel_load_utilization: structure_metrics.avg_vessel_load_utilization,
            min_vessel_load_utilization: structure_metrics.min_vessel_load_utilization,
            max_vessel_load_utilization: structure_metrics.max_vessel_load_utilization,
            avg_vessel_time_utilization: structure_metrics.avg_vessel_time_utilization,
            min_vessel_time_utilization: structure_metrics.min_vessel_time_utilization,
            max_vessel_time_utilization: structure_metrics.max_vessel_time_utilization,
            destroy_success_rates,
            repair_success_rates,
            recent_rewards,
            destroy_weights: self.alns_context.destroy_operator_weights.clone(),
            repair_weights: self.alns_context.repair_operator_weights.clone(),
            elapsed_ms,
        })
    }

    fn resolve_operator_pair(&mut self, mode: ALNSRunMode) -> (usize, usize) {
        match mode {
            ALNSRunMode::Random => {
                let destroy_idx = self
                    .rng
                    .gen_range(0..self.operator_registry.destroy_operators.len());
                let repair_idx = self
                    .rng
                    .gen_range(0..self.operator_registry.repair_operators.len());
                (destroy_idx, repair_idx)
            }
            ALNSRunMode::Weighted => {
                let destroy_idx = self.alns_context.pick_destroy_operator_idx();
                let repair_idx = self.alns_context.pick_repair_operator_idx();
                (destroy_idx, repair_idx)
            }
            ALNSRunMode::Explicit(destroy_idx, repair_idx) => (destroy_idx, repair_idx),
        }
    }

    fn compute_kisialiou_improvement_sequence(&self) -> Vec<usize> {
        const ORDERED_NAMES: [&str; 6] = [
            "voyage_number_reduction",
            "fleet_and_cost_reduction",
            "deep_relocation",
            "fleet_and_cost_reduction",
            "deep_swap",
            "fleet_and_cost_reduction"
        ];

        ORDERED_NAMES
            .iter()
            .filter_map(|name| self.find_improvement_operator_index(name))
            .collect()
    }

    fn find_improvement_operator_index(&self, label: &str) -> Option<usize> {
        self.improvement_operator_labels
            .iter()
            .position(|candidate| candidate == label)
    }

    pub fn run_improvement_only(
        &mut self,
        iteration: usize,
        improvement_operator_idx: usize,
    ) -> Result<ALNSMetrics, String> {
        let start_time = std::time::Instant::now();

        if improvement_operator_idx >= self.operator_registry.improvement_operators.len() {
            return Err(format!(
                "Invalid improvement operator index: {}",
                improvement_operator_idx
            ));
        }

        let improvement_operator_name = self
            .improvement_operator_labels
            .get(improvement_operator_idx)
            .cloned()
            .unwrap_or_else(|| format!("improvement_{}", improvement_operator_idx));

        let improvement_operator_type = Some("improvement".to_string());
        let improvement_operator_type_id = Some(2);

        let mut candidate_solution = self.current_solution.clone();
        let improvement_op = self
            .operator_registry
            .get_improvement_operator(improvement_operator_idx);
        improvement_op.apply(&mut candidate_solution, &self.context, &mut self.rng);
        candidate_solution.ensure_consistency_updated(&self.context);
        candidate_solution.update_total_cost(&self.context);

        let candidate_cost = candidate_solution.total_cost;
        let current_cost = self.current_solution.total_cost;
        let best_cost = self.best_solution.total_cost;

        let accept = acceptance::accept(
            current_cost,
            candidate_cost,
            self.temperature,
            &mut self.rng,
        );
        let in_aggressive_phase = self.is_in_aggressive_phase(iteration);

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
        } else if in_aggressive_phase {
            self.current_solution = self.best_solution.clone();
            self.stagnation_count += 1;
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

        self.temperature = acceptance::cool_down(self.temperature, self.theta, iteration + 1);
        self.iteration = iteration;
        self.alns_context.iteration = iteration;

        if (iteration + 1) % self.weight_update_interval == 0 {
            self.alns_context.update_operator_weights("destroy");
            self.alns_context.update_operator_weights("repair");
            self.alns_context.reset_segment_scores();
        }

        let elapsed_ms = start_time.elapsed().as_millis();

        log::info!(
            "Iter {:3} (improvement): cost={:.4}, best={:.4}, temp={:6.1}, improvement={}, accepted={}, elapsed={}ms",
            iteration + 1,
            candidate_cost,
            self.best_solution.total_cost,
            self.temperature,
            improvement_operator_idx,
            accepted,
            elapsed_ms
        );

        let current_solution = &self.current_solution;
        let best_solution = &self.best_solution;

        let mut temp_solution = current_solution.clone();
        let is_feasible = temp_solution.is_fully_feasible(&self.context);
        let is_complete = current_solution.is_complete_solution();

        let structure_metrics = compute_solution_structure_metrics(current_solution, &self.context);

        let destroy_success_rates: Vec<f64> = self
            .alns_context
            .destroy_operator_scores
            .iter()
            .zip(self.alns_context.destroy_operator_counts.iter())
            .map(|(score, count)| {
                if *count > 0 {
                    score / (*count as f64)
                } else {
                    0.5
                }
            })
            .collect();
        let repair_success_rates: Vec<f64> = self
            .alns_context
            .repair_operator_scores
            .iter()
            .zip(self.alns_context.repair_operator_counts.iter())
            .map(|(score, count)| {
                if *count > 0 {
                    score / (*count as f64)
                } else {
                    0.5
                }
            })
            .collect();
        let recent_rewards: Vec<f64> = self
            .alns_context
            .cost_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        Ok(ALNSMetrics {
            total_cost: current_solution.total_cost,
            best_cost: best_solution.total_cost,
            accepted,
            is_new_best,
            is_better_than_current,
            destroy_idx: None,
            repair_idx: None,
            destroy_operator_name: None,
            repair_operator_name: None,
            destroy_operator_type: None,
            repair_operator_type: None,
            destroy_operator_type_id: None,
            repair_operator_type_id: None,
            improvement_idx: Some(improvement_operator_idx),
            improvement_operator_name: Some(improvement_operator_name.clone()),
            improvement_operator_type,
            improvement_operator_type_id,
            improvement_sequence: vec![improvement_operator_idx],
            improvement_costs: vec![candidate_cost],
            improvement_step_metrics: vec![ALNSImprovementStepMetric {
                operator_idx: improvement_operator_idx,
                operator_name: improvement_operator_name.clone(),
                sequence_position: 0,
                cost_before: current_cost,
                cost_after: candidate_cost,
                cost_delta: candidate_cost - current_cost,
            }],
            destroy_removed_requests: None,
            repair_inserted_requests: None,
            temperature: self.temperature,
            stagnation_count: self.stagnation_count,
            iteration: self.iteration,
            initial_cost: self.initial_cost,
            is_complete,
            is_feasible,
            num_voyages: structure_metrics.num_voyages,
            num_empty_voyages: structure_metrics.num_empty_voyages,
            num_vessels_used: structure_metrics.num_vessels_used,
            avg_voyage_utilization: structure_metrics.avg_voyage_utilization,
            avg_vessel_load_utilization: structure_metrics.avg_vessel_load_utilization,
            min_vessel_load_utilization: structure_metrics.min_vessel_load_utilization,
            max_vessel_load_utilization: structure_metrics.max_vessel_load_utilization,
            avg_vessel_time_utilization: structure_metrics.avg_vessel_time_utilization,
            min_vessel_time_utilization: structure_metrics.min_vessel_time_utilization,
            max_vessel_time_utilization: structure_metrics.max_vessel_time_utilization,
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
        let mut last_metrics = self.create_default_metrics();

        for iter in 0..self.max_iterations {
            let destroy_idx = self.alns_context.pick_destroy_operator_idx();
            let repair_idx = self.alns_context.pick_repair_operator_idx();

            match self.run_iteration(ALNSRunMode::Explicit(destroy_idx, repair_idx), iter, None) {
                Ok(metrics) => last_metrics = metrics,
                Err(e) => {
                    log::error!("Error in iteration {}: {}", iter, e);
                    break;
                }
            }
        }

        last_metrics
    }

    /// Run the full ALNS algorithm for multiple restarts.
    ///
    /// Each restart runs a full `max_iterations` pass from a fresh initial solution,
    /// using deterministic seed schedule `base_seed + restart_index`.
    /// Only the best-found solution across all restarts is retained as global best.
    pub fn run_with_restarts(&mut self, restarts: usize) -> ALNSRunWithRestartsResult {
        let restart_count = restarts.max(1);
        let mut restart_summaries = Vec::with_capacity(restart_count);

        let mut global_best_solution: Option<Solution> = None;
        let mut global_best_cost = INFINITY;
        let mut global_metrics = self.create_default_metrics();

        for restart_idx in 0..restart_count {
            let restart_seed = self.base_seed.wrapping_add(restart_idx as u64);
            self.reset_for_restart(restart_seed);

            let restart_start = std::time::Instant::now();
            let restart_initial_cost = self.initial_cost;
            let mut last_metrics = self.create_default_metrics();
            let mut iterations_completed = 0usize;

            for iter in 0..self.max_iterations {
                let destroy_idx = self.alns_context.pick_destroy_operator_idx();
                let repair_idx = self.alns_context.pick_repair_operator_idx();

                match self.run_iteration(ALNSRunMode::Explicit(destroy_idx, repair_idx), iter, None) {
                    Ok(metrics) => {
                        last_metrics = metrics;
                        iterations_completed = iter + 1;
                    }
                    Err(e) => {
                        log::error!(
                            "Error in restart {} iteration {}: {}",
                            restart_idx,
                            iter,
                            e
                        );
                        break;
                    }
                }
            }

            let restart_best_cost = self.best_solution.total_cost;
            let restart_final_cost = self.current_solution.total_cost;
            let restart_elapsed_ms = restart_start.elapsed().as_millis();
            let best_improvement_pct = if restart_initial_cost.abs() > f64::EPSILON {
                (restart_initial_cost - restart_best_cost) / restart_initial_cost * 100.0
            } else {
                0.0
            };

            restart_summaries.push(ALNSRestartSummary {
                restart_index: restart_idx,
                seed: restart_seed,
                initial_cost: restart_initial_cost,
                best_cost: restart_best_cost,
                final_cost: restart_final_cost,
                iterations_completed,
                elapsed_ms: restart_elapsed_ms,
                best_improvement_pct,
            });

            if restart_best_cost < global_best_cost {
                global_best_cost = restart_best_cost;
                global_best_solution = Some(self.best_solution.clone());
                global_metrics = last_metrics;
            }
        }

        if let Some(best_solution) = global_best_solution {
            self.best_solution = best_solution.clone();
            self.current_solution = best_solution;
            self.alns_context.best_cost = self.best_solution.total_cost;
            global_metrics.best_cost = self.best_solution.total_cost;
            global_metrics.total_cost = self.best_solution.total_cost;
        }

        ALNSRunWithRestartsResult {
            global_metrics,
            restart_summaries,
        }
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

#[derive(Clone)]
pub(crate) struct SolutionStructureMetrics {
    pub num_voyages: usize,
    pub num_empty_voyages: usize,
    pub num_vessels_used: usize,
    pub avg_voyage_utilization: f64,
    pub avg_vessel_load_utilization: f64,
    pub min_vessel_load_utilization: f64,
    pub max_vessel_load_utilization: f64,
    pub avg_vessel_time_utilization: f64,
    pub min_vessel_time_utilization: f64,
    pub max_vessel_time_utilization: f64,
}

pub(crate) fn compute_solution_structure_metrics(
    solution: &Solution,
    context: &Context,
) -> SolutionStructureMetrics {
    use crate::structs::constants::HOURS_IN_PERIOD;

    let mut non_empty_voyages = 0usize;
    let mut total_visits = 0usize;
    let mut vessels_used: HashSet<usize> = HashSet::new();
    let mut vessel_load_totals: HashMap<usize, (f64, usize)> = HashMap::new();
    let mut vessel_time_totals: HashMap<usize, f64> = HashMap::new();

    for voyage_cell in &solution.voyages {
        let voyage = voyage_cell.borrow();
        if voyage.visit_ids.is_empty() {
            continue;
        }
        non_empty_voyages += 1;
        total_visits += voyage.visit_ids.len();

        if let Some(vessel_id) = voyage.vessel_id {
            vessels_used.insert(vessel_id);

            let load = voyage.load.unwrap_or(0) as f64;
            let entry = vessel_load_totals.entry(vessel_id).or_insert((0.0, 0));
            entry.0 += load;
            entry.1 += 1;

            if let (Some(start_time), Some(end_time)) = (voyage.start_time(), voyage.end_time()) {
                let mut duration = end_time - start_time;
                if duration < 0.0 {
                    duration += HOURS_IN_PERIOD as f64;
                }
                vessel_time_totals
                    .entry(vessel_id)
                    .and_modify(|total| *total += duration)
                    .or_insert(duration);
            }
        }
    }

    let empty_voyages = solution.voyages.len().saturating_sub(non_empty_voyages);
    let avg_voyage_utilization = if non_empty_voyages > 0 {
        total_visits as f64 / non_empty_voyages as f64
    } else {
        0.0
    };

    let period_hours = HOURS_IN_PERIOD as f64;
    let mut vessel_load_metrics = Vec::new();
    let mut vessel_time_metrics = Vec::new();

    for (vessel_id, (total_load, voyage_count)) in &vessel_load_totals {
        if let Some(vessel) = context.problem.vessels.get(*vessel_id) {
            let total_capacity = vessel.deck_capacity * (*voyage_count as f64);
            if total_capacity > 0.0 {
                vessel_load_metrics.push(total_load / total_capacity);
            }
        }
        if let Some(total_time) = vessel_time_totals.get(vessel_id) {
            if period_hours > 0.0 {
                vessel_time_metrics.push(*total_time / period_hours);
            }
        }
    }

    let (avg_load, min_load, max_load) = summarize_utilization(&vessel_load_metrics);
    let (avg_time, min_time, max_time) = summarize_utilization(&vessel_time_metrics);

    SolutionStructureMetrics {
        num_voyages: non_empty_voyages,
        num_empty_voyages: empty_voyages,
        num_vessels_used: vessels_used.len(),
        avg_voyage_utilization,
        avg_vessel_load_utilization: avg_load,
        min_vessel_load_utilization: min_load,
        max_vessel_load_utilization: max_load,
        avg_vessel_time_utilization: avg_time,
        min_vessel_time_utilization: min_time,
        max_vessel_time_utilization: max_time,
    }
}

fn summarize_utilization(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let sum: f64 = values.iter().sum();
    let avg = sum / values.len() as f64;
    let min = values.iter().copied().fold(INFINITY, f64::min);
    let max = values.iter().copied().fold(NEG_INFINITY, f64::max);

    (avg, min, max)
}

/// Enable file logging for ALNS operations
pub fn enable_file_logging(log_path: &str) -> Result<(), String> {
    use log::LevelFilter;
    use std::fs::OpenOptions;
    use std::io::Write;

    let log_pipe = Box::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(log_path)
            .map_err(|e| format!("Failed to open log file {}: {}", log_path, e))?,
    );

    let mut builder = env_logger::Builder::from_default_env();
    builder.filter_level(LevelFilter::Info);
    builder.target(env_logger::Target::Pipe(log_pipe));
    builder
        .try_init()
        .map_err(|e| format!("Logger already initialized or configuration error: {}", e))?;

    Ok(())
}

/// Enable console logging for ALNS operations
pub fn enable_console_logging() -> Result<(), String> {
    use log::LevelFilter;

    let mut builder = env_logger::Builder::from_default_env();
    builder.filter_level(LevelFilter::Info);
    builder.target(env_logger::Target::Stdout);
    builder
        .try_init()
        .map_err(|e| format!("Logger already initialized or configuration error: {}", e))
}
