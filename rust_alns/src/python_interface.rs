use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};
use crate::structs::{solution::Solution, context::Context};
use crate::alns::{engine::ALNSEngine, context::ALNSContext};
use crate::operators::registry::OperatorRegistry;
use crate::operators::traits::{DestroyOperator, RepairOperator};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[pyclass(unsendable)]
pub struct RustALNSInterface {
    context: Option<Context>,
    alns_context: Option<ALNSContext>,
    operator_registry: Option<OperatorRegistry>,
    current_solution: Option<Solution>,
    best_solution: Option<Solution>,
    initial_solution: Option<Solution>,
    rng: Option<StdRng>,
    iteration_count: usize,
    max_iterations: usize,
    temperature: f64,
    theta: f64,
    stagnation_count: usize,
    initial_cost: f64,
}

#[pymethods]
impl RustALNSInterface {
    #[new]
    fn new() -> Self {
        Self {
            context: None,
            alns_context: None,
            operator_registry: None,
            current_solution: None,
            best_solution: None,
            initial_solution: None,
            rng: None,
            iteration_count: 0,
            max_iterations: 1000,
            temperature: 1000.0,
            theta: 0.9,
            stagnation_count: 0,
            initial_cost: 0.0,
        }
    }
    
    /// Initialize ALNS run with problem instance and seed
    fn initialize_alns(&mut self, py: Python, problem_instance: &str, seed: u64) -> PyResult<PyObject> {
        // Load problem data based on instance name
        let (installations_path, vessels_path, base_path) = match problem_instance {
            "SMALL_1" => (
                "sample/installations/SMALL_1/i_test1.csv",
                "sample/vessels/SMALL_1/v_test1.csv",
                "sample/base/SMALL_1/b_test1.csv"
            ),
            _ => return Err(PyValueError::new_err("Unknown problem instance")),
        };
        
        // Load data and create context
        let data = crate::structs::data_loader::read_data(installations_path, vessels_path, base_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load data: {}", e)))?;
        
        let problem_data = crate::structs::problem_data::ProblemData::new(
            data.vessels.clone(), 
            data.installations.clone(), 
            data.base.clone()
        );
        
        let tsp_solver = crate::utils::tsp_solver::TSPSolver::new_from_problem_data(&problem_data);
        let context = Context { problem: problem_data, tsp_solver };
        
        // Create initial solution
        let mut rng = StdRng::seed_from_u64(seed);
        let mut initial_solution = crate::operators::initial_solution::construct_initial_solution(&context, &mut rng);
        initial_solution.update_total_cost(&context);
        
        // Set up operator registry
        let mut operator_registry = OperatorRegistry::new();
        self.setup_operators(&mut operator_registry);
        
        // Initialize ALNS context
        let n_destroy = operator_registry.destroy_operators.len();
        let n_repair = operator_registry.repair_operators.len();
        let alns_context = ALNSContext {
            iteration: 0,
            temperature: self.temperature,
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
        
        // Store state
        self.initial_cost = initial_solution.total_cost;
        self.context = Some(context);
        self.alns_context = Some(alns_context);
        self.operator_registry = Some(operator_registry);
        self.current_solution = Some(initial_solution.clone());
        self.best_solution = Some(initial_solution.clone());
        self.initial_solution = Some(initial_solution);
        self.rng = Some(StdRng::seed_from_u64(seed));
        self.iteration_count = 0;
        self.stagnation_count = 0;
        
        // Return initial solution metrics as Python dict
        self.extract_solution_metrics(py)
    }
    
    /// Execute one ALNS iteration with specified operators
    fn execute_iteration(&mut self, py: Python, destroy_operator_idx: usize, repair_operator_idx: usize, iteration: usize) -> PyResult<PyObject> {
        let context = self.context.as_ref().ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;
        let operator_registry = self.operator_registry.as_ref().ok_or_else(|| PyRuntimeError::new_err("Operator registry not available"))?;
        let current_solution = self.current_solution.as_mut().ok_or_else(|| PyRuntimeError::new_err("Current solution not available"))?;
        let best_solution = self.best_solution.as_mut().ok_or_else(|| PyRuntimeError::new_err("Best solution not available"))?;
        let rng = self.rng.as_mut().ok_or_else(|| PyRuntimeError::new_err("RNG not available"))?;
        
        // Validate operator indices
        if destroy_operator_idx >= operator_registry.destroy_operators.len() {
            return Err(PyValueError::new_err("Invalid destroy operator index"));
        }
        if repair_operator_idx >= operator_registry.repair_operators.len() {
            return Err(PyValueError::new_err("Invalid repair operator index"));
        }
        
        // Get operators
        let destroy_op = operator_registry.get_destroy_operator(destroy_operator_idx);
        let repair_op = operator_registry.get_repair_operator(repair_operator_idx);
        
        // Create candidate solution
        let mut candidate_solution = current_solution.clone();
        
        // Apply operators
        destroy_op.apply(&mut candidate_solution, context, rng);
        candidate_solution.ensure_consistency_updated(context);
        candidate_solution.add_idle_vessel_and_add_empty_voyages(context);
        repair_op.apply(&mut candidate_solution, context, rng);
        candidate_solution.ensure_consistency_updated(context);
        candidate_solution.update_total_cost(context);
        
        // Acceptance decision
        let candidate_cost = candidate_solution.total_cost;
        let current_cost = current_solution.total_cost;
        let best_cost = best_solution.total_cost;
        
        let accept = crate::alns::acceptance::accept(current_cost, candidate_cost, self.temperature, rng);
        let mut accepted = false;
        let mut is_new_best = false;
        let mut is_better_than_current = false;
        
        // Update solutions based on acceptance
        if candidate_cost < best_cost {
            *best_solution = candidate_solution.clone();
            *current_solution = candidate_solution;
            accepted = true;
            is_new_best = true;
            is_better_than_current = true;
            self.stagnation_count = 0;
        } else if candidate_cost < current_cost {
            *current_solution = candidate_solution;
            accepted = true;
            is_better_than_current = true;
            self.stagnation_count = 0;
        } else if accept {
            *current_solution = candidate_solution;
            accepted = true;
            self.stagnation_count += 1;
        } else {
            self.stagnation_count += 1;
        }
        
        // Cool down temperature
        self.temperature = crate::alns::acceptance::cool_down(self.temperature, self.theta, iteration + 1);
        self.iteration_count = iteration;
        
        // Extract current values we need before getting mutable access to alns_context
        let current_cost = current_solution.total_cost;
        let best_cost = best_solution.total_cost;
        let current_temperature = self.temperature;
        let current_stagnation = self.stagnation_count;
        let current_iteration = self.iteration_count;
        let current_initial_cost = self.initial_cost;
        
        // Now update ALNS context
        let alns_context = self.alns_context.as_mut().ok_or_else(|| PyRuntimeError::new_err("ALNS context not available"))?;
        alns_context.iteration = iteration;
        
        // Reward operators based on performance
        if is_new_best {
            alns_context.reward_operator("destroy", destroy_operator_idx, 0); // Best reward
            alns_context.reward_operator("repair", repair_operator_idx, 0);
        } else if is_better_than_current {
            alns_context.reward_operator("destroy", destroy_operator_idx, 1); // Better reward
            alns_context.reward_operator("repair", repair_operator_idx, 1);
        } else if accepted {
            alns_context.reward_operator("destroy", destroy_operator_idx, 2); // Accepted reward
            alns_context.reward_operator("repair", repair_operator_idx, 2);
        }
        
        // Update operator weights periodically
        if (iteration + 1) % 10 == 0 {
            alns_context.update_operator_weights("destroy");
            alns_context.update_operator_weights("repair");
            alns_context.reset_segment_scores();
        }
        
        let dict = PyDict::new(py);
        
        // Create solution metrics dict manually to avoid borrowing issues
        let solution_dict = PyDict::new(py);
        solution_dict.set_item("total_cost", current_cost)?;
        
        // Check feasibility with temp clone 
        let mut temp_solution = current_solution.clone();
        let is_feasible = temp_solution.is_fully_feasible(&context);
        let is_complete = current_solution.is_complete_solution();
        solution_dict.set_item("is_complete", is_complete)?;
        solution_dict.set_item("is_feasible", is_feasible)?;
        
        // Solution structure metrics
        let non_empty_voyages = current_solution.voyages.iter()
            .filter(|v| !v.borrow().visit_ids.is_empty())
            .count();
        let empty_voyages = current_solution.voyages.len() - non_empty_voyages;
        solution_dict.set_item("num_voyages", non_empty_voyages)?;
        solution_dict.set_item("num_empty_voyages", empty_voyages)?;
        
        // Calculate vessels used
        let mut vessels_used = HashSet::new();
        for voyage in &current_solution.voyages {
            if !voyage.borrow().visit_ids.is_empty() {
                vessels_used.insert(voyage.borrow().vessel_id);
            }
        }
        solution_dict.set_item("num_vessels_used", vessels_used.len())?;
        
        // Average voyage utilization
        let total_visits: usize = current_solution.voyages.iter()
            .map(|v| v.borrow().visit_ids.len())
            .sum();
        let avg_utilization = if non_empty_voyages > 0 {
            total_visits as f64 / non_empty_voyages as f64
        } else {
            0.0
        };
        solution_dict.set_item("avg_voyage_utilization", avg_utilization)?;
        
        // Search progression
        solution_dict.set_item("iteration", current_iteration)?;
        solution_dict.set_item("temperature", current_temperature)?;
        solution_dict.set_item("stagnation_count", current_stagnation)?;
        solution_dict.set_item("best_cost", best_cost)?;
        solution_dict.set_item("initial_cost", current_initial_cost)?;
        
        // Operator performance
        let destroy_success_rates: Vec<f64> = alns_context.destroy_operator_scores.iter()
            .zip(alns_context.destroy_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        
        let repair_success_rates: Vec<f64> = alns_context.repair_operator_scores.iter()
            .zip(alns_context.repair_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        
        solution_dict.set_item("destroy_success_rates", PyList::new(py, destroy_success_rates))?;
        solution_dict.set_item("repair_success_rates", PyList::new(py, repair_success_rates))?;
        
        // Recent rewards (last 5 from cost history)
        let recent_rewards: Vec<f64> = alns_context.cost_history.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();
        solution_dict.set_item("recent_rewards", PyList::new(py, recent_rewards))?;
        
        dict.set_item("solution_metrics", solution_dict)?;
        
        // Iteration-specific information
        dict.set_item("accepted", accepted)?;
        dict.set_item("is_new_best", is_new_best)?;
        dict.set_item("is_better_than_current", is_better_than_current)?;
        dict.set_item("candidate_cost", candidate_cost)?;
        dict.set_item("current_cost", current_solution.total_cost)?;
        dict.set_item("best_cost", best_solution.total_cost)?;
        dict.set_item("temperature", self.temperature)?;
        dict.set_item("stagnation_count", self.stagnation_count)?;
        dict.set_item("destroy_operator_idx", destroy_operator_idx)?;
        dict.set_item("repair_operator_idx", repair_operator_idx)?;
        
        // Operator performance
        dict.set_item("destroy_weights", PyList::new(py, alns_context.destroy_operator_weights.clone()))?;
        dict.set_item("repair_weights", PyList::new(py, alns_context.repair_operator_weights.clone()))?;
        dict.set_item("destroy_scores", PyList::new(py, alns_context.destroy_operator_scores.clone()))?;
        dict.set_item("repair_scores", PyList::new(py, alns_context.repair_operator_scores.clone()))?;
        
        Ok(dict.into())
    }
    
    /// Extract comprehensive solution metrics for RL state
    fn extract_solution_metrics(&self, py: Python) -> PyResult<PyObject> {
        let context = self.context.as_ref().ok_or_else(|| PyRuntimeError::new_err("Context not available"))?;
        let current_solution = self.current_solution.as_ref().ok_or_else(|| PyRuntimeError::new_err("Current solution not available"))?;
        let best_solution = self.best_solution.as_ref().ok_or_else(|| PyRuntimeError::new_err("Best solution not available"))?;
        let alns_context = self.alns_context.as_ref().ok_or_else(|| PyRuntimeError::new_err("ALNS context not available"))?;
        
        let dict = PyDict::new(py);
        
        // Basic solution properties
        dict.set_item("total_cost", current_solution.total_cost)?;
        dict.set_item("is_complete", current_solution.is_complete_solution())?;
        
        // We need to clone to check feasibility since it requires &mut self
        let mut temp_solution = current_solution.clone();
        let is_feasible = temp_solution.is_fully_feasible(context);
        dict.set_item("is_feasible", is_feasible)?;
        
        // Solution structure
        let non_empty_voyages = current_solution.voyages.iter()
            .filter(|v| !v.borrow().visit_ids.is_empty())
            .count();
        let empty_voyages = current_solution.voyages.len() - non_empty_voyages;
        
        dict.set_item("num_voyages", non_empty_voyages)?;
        dict.set_item("num_empty_voyages", empty_voyages)?;
        
        // Calculate vessels used
        let mut vessels_used = std::collections::HashSet::new();
        for voyage in &current_solution.voyages {
            if !voyage.borrow().visit_ids.is_empty() {
                vessels_used.insert(voyage.borrow().vessel_id);
            }
        }
        dict.set_item("num_vessels_used", vessels_used.len())?;
        
        // Average voyage utilization
        let total_visits: usize = current_solution.voyages.iter()
            .map(|v| v.borrow().visit_ids.len())
            .sum();
        let avg_utilization = if non_empty_voyages > 0 {
            total_visits as f64 / non_empty_voyages as f64
        } else {
            0.0
        };
        dict.set_item("avg_voyage_utilization", avg_utilization)?;
        
        // Search progression
        dict.set_item("iteration", self.iteration_count)?;
        dict.set_item("temperature", self.temperature)?;
        dict.set_item("stagnation_count", self.stagnation_count)?;
        dict.set_item("best_cost", best_solution.total_cost)?;
        dict.set_item("initial_cost", self.initial_cost)?;
        
        // Operator performance
        let destroy_success_rates: Vec<f64> = alns_context.destroy_operator_scores.iter()
            .zip(alns_context.destroy_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        
        let repair_success_rates: Vec<f64> = alns_context.repair_operator_scores.iter()
            .zip(alns_context.repair_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        
        dict.set_item("destroy_success_rates", PyList::new(py, destroy_success_rates))?;
        dict.set_item("repair_success_rates", PyList::new(py, repair_success_rates))?;
        
        // Recent rewards (last 5 from cost history)
        let recent_rewards: Vec<f64> = alns_context.cost_history.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();
        dict.set_item("recent_rewards", PyList::new(py, recent_rewards))?;
        
        Ok(dict.into())
    }
    
    /// Get operator names and descriptions
    fn get_operator_info(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        // These match your actual operators
        let destroy_operators = vec![
            "shaw_removal",
            "random_visit_removal", 
            "worst_visit_removal"
        ];
        
        let repair_operators = vec![
            "deep_greedy_insertion",
            "k_regret_2",
            "k_regret_3"
        ];
        
        dict.set_item("destroy_operators", PyList::new(py, destroy_operators))?;
        dict.set_item("repair_operators", PyList::new(py, repair_operators))?;
        
        Ok(dict.into())
    }
    
    /// Export current solution to JSON file
    fn export_solution(&self, filepath: &str) -> PyResult<()> {
        let context = self.context.as_ref().ok_or_else(|| PyRuntimeError::new_err("Context not available"))?;
        let current_solution = self.current_solution.as_ref().ok_or_else(|| PyRuntimeError::new_err("Current solution not available"))?;
        
        crate::utils::serialization::dump_schedule_to_json(
            current_solution, 
            &context.problem.vessels, 
            filepath, 
            context
        );
        
        Ok(())
    }
}

impl RustALNSInterface {
    fn setup_operators(&self, registry: &mut OperatorRegistry) {
        // Add destroy operators
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
        
        // Add repair operators
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
}

/// Python module definition
#[pymodule]
fn rust_alns_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustALNSInterface>()?;
    Ok(())
}