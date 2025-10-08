use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};
use crate::structs::{solution::Solution, context::Context};
use crate::alns::engine::{ALNSEngine, ALNSMetrics};
use crate::operators::registry::OperatorRegistry;
use crate::operators::traits::{DestroyOperator, RepairOperator};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};

// Global flag to track if logging has been initialized for Python interface
static PYTHON_LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

// ================= ENGINE CORE (pure Rust) =================
// All ALNS logic now lives in alns/engine.rs
// This module provides PyO3 bindings

#[pyclass(unsendable)]
pub struct RustALNSInterface {
    engine: Option<ALNSEngine>,
}

#[pymethods]
impl RustALNSInterface {
    #[new]
    fn new() -> Self { Self { engine: None } }
    
    fn initialize_alns(&mut self, py: Python, problem_instance: &str, seed: u64, temperature: Option<f64>, theta: Option<f64>, weight_update_interval: Option<usize>) -> PyResult<PyObject> {
        // Initialize logging for Python interface (only if not already initialized)
        if !PYTHON_LOGGING_INITIALIZED.load(Ordering::Relaxed) {
            let _ = env_logger::Builder::from_default_env()
                .target(env_logger::Target::Stdout)
                .try_init();
            PYTHON_LOGGING_INITIALIZED.store(true, Ordering::Relaxed);
        }
            
        let temperature = temperature.unwrap_or(500.0);
        let theta = theta.unwrap_or(0.9);
        let weight_update_interval = weight_update_interval.unwrap_or(10);
        let max_iterations = 1000; // Default max iterations
        
        let engine = ALNSEngine::new_from_instance(problem_instance, seed, temperature, theta, weight_update_interval, max_iterations)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        self.engine = Some(engine);
        self.extract_solution_metrics(py)
    }
    
    #[pyo3(signature = (iteration, destroy_operator_idx=None, repair_operator_idx=None, mode=None))]
    fn execute_iteration(&mut self, py: Python, iteration: usize, destroy_operator_idx: Option<usize>, repair_operator_idx: Option<usize>, mode: Option<&str>) -> PyResult<PyObject> {
        let engine = self.engine.as_mut().ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;
        
        // Parse mode and create ALNSRunMode
        let run_mode = match mode {
            Some("random") => crate::alns::engine::ALNSRunMode::Random,
            Some("weighted") => crate::alns::engine::ALNSRunMode::Weighted,
            Some("explicit") | None => {
                // For explicit mode or backward compatibility, require operator indices
                let destroy_idx = destroy_operator_idx.ok_or_else(|| 
                    PyRuntimeError::new_err("destroy_operator_idx required for explicit mode"))?;
                let repair_idx = repair_operator_idx.ok_or_else(|| 
                    PyRuntimeError::new_err("repair_operator_idx required for explicit mode"))?;
                crate::alns::engine::ALNSRunMode::Explicit(destroy_idx, repair_idx)
            },
            Some(invalid_mode) => return Err(PyRuntimeError::new_err(format!("Invalid mode: {}. Use 'random', 'weighted', or 'explicit'", invalid_mode))),
        };
        
        let metrics = engine.run_iteration(run_mode, iteration)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        
        let dict = PyDict::new(py);
        let solution_dict = PyDict::new(py);
        solution_dict.set_item("total_cost", metrics.total_cost)?;
        solution_dict.set_item("is_complete", metrics.is_complete)?;
        solution_dict.set_item("is_feasible", metrics.is_feasible)?;
        solution_dict.set_item("num_voyages", metrics.num_voyages)?;
        solution_dict.set_item("num_empty_voyages", metrics.num_empty_voyages)?;
        solution_dict.set_item("num_vessels_used", metrics.num_vessels_used)?;
        solution_dict.set_item("avg_voyage_utilization", metrics.avg_voyage_utilization)?;
        solution_dict.set_item("iteration", metrics.iteration)?;
        solution_dict.set_item("temperature", metrics.temperature)?;
        solution_dict.set_item("stagnation_count", metrics.stagnation_count)?;
        solution_dict.set_item("best_cost", metrics.best_cost)?;
        solution_dict.set_item("initial_cost", metrics.initial_cost)?;
        solution_dict.set_item("destroy_success_rates", PyList::new(py, metrics.destroy_success_rates))?;
        solution_dict.set_item("repair_success_rates", PyList::new(py, metrics.repair_success_rates))?;
        solution_dict.set_item("recent_rewards", PyList::new(py, metrics.recent_rewards))?;
        dict.set_item("solution_metrics", solution_dict)?;
        dict.set_item("accepted", metrics.accepted)?;
        dict.set_item("is_new_best", metrics.is_new_best)?;
        dict.set_item("is_better_than_current", metrics.is_better_than_current)?;
        dict.set_item("candidate_cost", metrics.total_cost)?;
        dict.set_item("current_cost", metrics.total_cost)?;
        dict.set_item("best_cost", metrics.best_cost)?;
        dict.set_item("temperature", metrics.temperature)?;
        dict.set_item("stagnation_count", metrics.stagnation_count)?;
        dict.set_item("destroy_operator_idx", metrics.destroy_idx)?;
        dict.set_item("repair_operator_idx", metrics.repair_idx)?;
        dict.set_item("destroy_weights", PyList::new(py, metrics.destroy_weights))?;
        dict.set_item("repair_weights", PyList::new(py, metrics.repair_weights))?;
        dict.set_item("elapsed_ms", metrics.elapsed_ms)?;
        Ok(dict.into())
    }
    
    fn extract_solution_metrics(&self, py: Python) -> PyResult<PyObject> {
        let engine = self.engine.as_ref().ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;
        
        // Get metrics from the current solution
        let current_solution = &engine.current_solution;
        let best_solution = &engine.best_solution;
        let mut temp_solution = current_solution.clone();
        let is_feasible = temp_solution.is_fully_feasible(&engine.context);
        let is_complete = current_solution.is_complete_solution();
        
        let non_empty_voyages = current_solution.voyages.iter().filter(|v| !v.borrow().visit_ids.is_empty()).count();
        let empty_voyages = current_solution.voyages.len() - non_empty_voyages;
        
        let mut vessels_used = std::collections::HashSet::new();
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
        
        let destroy_success_rates: Vec<f64> = engine.alns_context.destroy_operator_scores.iter()
            .zip(engine.alns_context.destroy_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        let repair_success_rates: Vec<f64> = engine.alns_context.repair_operator_scores.iter()
            .zip(engine.alns_context.repair_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        let recent_rewards: Vec<f64> = engine.alns_context.cost_history.iter().rev().take(5).cloned().collect();
        
        let dict = PyDict::new(py);
        dict.set_item("total_cost", current_solution.total_cost)?;
        dict.set_item("is_complete", is_complete)?;
        dict.set_item("is_feasible", is_feasible)?;
        dict.set_item("num_voyages", non_empty_voyages)?;
        dict.set_item("num_empty_voyages", empty_voyages)?;
        dict.set_item("num_vessels_used", vessels_used.len())?;
        dict.set_item("avg_voyage_utilization", avg_utilization)?;
        dict.set_item("iteration", engine.iteration)?;
        dict.set_item("temperature", engine.temperature)?;
        dict.set_item("stagnation_count", engine.stagnation_count)?;
        dict.set_item("best_cost", best_solution.total_cost)?;
        dict.set_item("initial_cost", engine.initial_cost)?;
        dict.set_item("destroy_success_rates", PyList::new(py, destroy_success_rates))?;
        dict.set_item("repair_success_rates", PyList::new(py, repair_success_rates))?;
        dict.set_item("recent_rewards", PyList::new(py, recent_rewards))?;
        Ok(dict.into())
    }
    
    /// Get operator names and descriptions
    fn get_operator_info(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let (destroy_operators, repair_operators) = ALNSEngine::get_operator_info();
        dict.set_item("destroy_operators", PyList::new(py, destroy_operators))?;
        dict.set_item("repair_operators", PyList::new(py, repair_operators))?;
        Ok(dict.into())
    }
    
    fn export_solution(&self, filepath: &str) -> PyResult<()> {
        let engine = self.engine.as_ref().ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;
        engine.export_solution(filepath);
        Ok(())
    }

    /// Enable file logging for ALNS operations
    #[staticmethod]
    fn enable_file_logging(log_path: &str) -> PyResult<()> {
        crate::alns::engine::enable_file_logging(log_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to enable file logging: {}", e)))
    }

    /// Enable console logging for ALNS operations  
    #[staticmethod]
    fn enable_console_logging() -> PyResult<()> {
        crate::alns::engine::enable_console_logging()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to enable console logging: {}", e)))
    }

    /// Check if Python logging has been initialized
    #[staticmethod]
    fn is_logging_initialized() -> bool {
        PYTHON_LOGGING_INITIALIZED.load(Ordering::Relaxed)
    }

    /// Reset the Python logging initialization flag (for testing purposes)
    #[staticmethod]
    fn reset_logging_flag() {
        PYTHON_LOGGING_INITIALIZED.store(false, Ordering::Relaxed);
    }
}

/// Python module definition
#[pymodule]
fn rust_alns_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustALNSInterface>()?;
    Ok(())
}