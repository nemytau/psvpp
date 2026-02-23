use crate::alns::engine::{
    compute_solution_structure_metrics, ALNSEngine, ALNSEngineSnapshot,
    ALNSRunWithRestartsResult, ALNSMetrics,
};
use crate::utils::serialization::dump_schedule_to_json;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

// Global flag to track if logging has been initialized for Python interface
static PYTHON_LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

fn metrics_to_pydict(py: Python<'_>, metrics: &ALNSMetrics) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    let solution_dict = PyDict::new(py);
    solution_dict.set_item("total_cost", metrics.total_cost)?;
    solution_dict.set_item("is_complete", metrics.is_complete)?;
    solution_dict.set_item("is_feasible", metrics.is_feasible)?;
    solution_dict.set_item("num_voyages", metrics.num_voyages)?;
    solution_dict.set_item("num_empty_voyages", metrics.num_empty_voyages)?;
    solution_dict.set_item("num_vessels_used", metrics.num_vessels_used)?;
    solution_dict.set_item("avg_voyage_utilization", metrics.avg_voyage_utilization)?;
    solution_dict.set_item(
        "avg_vessel_load_utilization",
        metrics.avg_vessel_load_utilization,
    )?;
    solution_dict.set_item(
        "min_vessel_load_utilization",
        metrics.min_vessel_load_utilization,
    )?;
    solution_dict.set_item(
        "max_vessel_load_utilization",
        metrics.max_vessel_load_utilization,
    )?;
    solution_dict.set_item(
        "avg_vessel_time_utilization",
        metrics.avg_vessel_time_utilization,
    )?;
    solution_dict.set_item(
        "min_vessel_time_utilization",
        metrics.min_vessel_time_utilization,
    )?;
    solution_dict.set_item(
        "max_vessel_time_utilization",
        metrics.max_vessel_time_utilization,
    )?;
    solution_dict.set_item("iteration", metrics.iteration)?;
    solution_dict.set_item("temperature", metrics.temperature)?;
    solution_dict.set_item("stagnation_count", metrics.stagnation_count)?;
    solution_dict.set_item("best_cost", metrics.best_cost)?;
    solution_dict.set_item("initial_cost", metrics.initial_cost)?;
    solution_dict.set_item(
        "destroy_success_rates",
        PyList::new(py, metrics.destroy_success_rates.clone()),
    )?;
    solution_dict.set_item(
        "repair_success_rates",
        PyList::new(py, metrics.repair_success_rates.clone()),
    )?;
    solution_dict.set_item(
        "recent_rewards",
        PyList::new(py, metrics.recent_rewards.clone()),
    )?;
    dict.set_item("solution_metrics", solution_dict)?;
    dict.set_item("accepted", metrics.accepted)?;
    dict.set_item("is_new_best", metrics.is_new_best)?;
    dict.set_item("is_better_than_current", metrics.is_better_than_current)?;
    dict.set_item("candidate_cost", metrics.total_cost)?;
    dict.set_item("current_cost", metrics.total_cost)?;
    dict.set_item("best_cost", metrics.best_cost)?;
    dict.set_item("temperature", metrics.temperature)?;
    dict.set_item("stagnation_count", metrics.stagnation_count)?;

    match metrics.destroy_idx {
        Some(value) => dict.set_item("destroy_operator_idx", value)?,
        None => dict.set_item("destroy_operator_idx", py.None())?,
    }
    match metrics.repair_idx {
        Some(value) => dict.set_item("repair_operator_idx", value)?,
        None => dict.set_item("repair_operator_idx", py.None())?,
    }
    match metrics.destroy_operator_name.as_ref() {
        Some(name) => dict.set_item("destroy_operator_name", name)?,
        None => dict.set_item("destroy_operator_name", py.None())?,
    }
    match metrics.repair_operator_name.as_ref() {
        Some(name) => dict.set_item("repair_operator_name", name)?,
        None => dict.set_item("repair_operator_name", py.None())?,
    }
    match metrics.destroy_operator_type.as_ref() {
        Some(value) => dict.set_item("destroy_operator_type", value)?,
        None => dict.set_item("destroy_operator_type", py.None())?,
    }
    match metrics.repair_operator_type.as_ref() {
        Some(value) => dict.set_item("repair_operator_type", value)?,
        None => dict.set_item("repair_operator_type", py.None())?,
    }
    match metrics.destroy_operator_type_id {
        Some(value) => dict.set_item("destroy_operator_type_id", value)?,
        None => dict.set_item("destroy_operator_type_id", py.None())?,
    }
    match metrics.repair_operator_type_id {
        Some(value) => dict.set_item("repair_operator_type_id", value)?,
        None => dict.set_item("repair_operator_type_id", py.None())?,
    }

    match metrics.improvement_idx {
        Some(value) => dict.set_item("improvement_operator_idx", value)?,
        None => dict.set_item("improvement_operator_idx", py.None())?,
    }
    match metrics.improvement_operator_name.as_ref() {
        Some(name) => dict.set_item("improvement_operator_name", name)?,
        None => dict.set_item("improvement_operator_name", py.None())?,
    }
    match metrics.improvement_operator_type.as_ref() {
        Some(value) => dict.set_item("improvement_operator_type", value)?,
        None => dict.set_item("improvement_operator_type", py.None())?,
    }
    match metrics.improvement_operator_type_id {
        Some(value) => dict.set_item("improvement_operator_type_id", value)?,
        None => dict.set_item("improvement_operator_type_id", py.None())?,
    }

    dict.set_item(
        "improvement_sequence",
        PyList::new(py, metrics.improvement_sequence.clone()),
    )?;
    dict.set_item(
        "improvement_costs",
        PyList::new(py, metrics.improvement_costs.clone()),
    )?;

    match metrics.cost_before_destroy {
        Some(value) => dict.set_item("cost_before_destroy", value)?,
        None => dict.set_item("cost_before_destroy", py.None())?,
    }
    match metrics.cost_after_destroy {
        Some(value) => dict.set_item("cost_after_destroy", value)?,
        None => dict.set_item("cost_after_destroy", py.None())?,
    }
    match metrics.cost_after_repair {
        Some(value) => dict.set_item("cost_after_repair", value)?,
        None => dict.set_item("cost_after_repair", py.None())?,
    }

    let improvement_step_metrics = PyList::empty(py);
    for step in &metrics.improvement_step_metrics {
        let step_dict = PyDict::new(py);
        step_dict.set_item("operator_idx", step.operator_idx)?;
        step_dict.set_item("operator_name", &step.operator_name)?;
        step_dict.set_item("sequence_position", step.sequence_position)?;
        step_dict.set_item("cost_before", step.cost_before)?;
        step_dict.set_item("cost_after", step.cost_after)?;
        step_dict.set_item("cost_delta", step.cost_delta)?;
        improvement_step_metrics.append(step_dict)?;
    }
    dict.set_item("improvement_step_metrics", improvement_step_metrics)?;

    dict.set_item("destroy_removed_requests", metrics.destroy_removed_requests)?;
    dict.set_item("repair_inserted_requests", metrics.repair_inserted_requests)?;
    dict.set_item(
        "destroy_weights",
        PyList::new(py, metrics.destroy_weights.clone()),
    )?;
    dict.set_item(
        "repair_weights",
        PyList::new(py, metrics.repair_weights.clone()),
    )?;
    dict.set_item("elapsed_ms", metrics.elapsed_ms)?;
    Ok(dict.into())
}

fn run_with_restarts_result_to_pydict(
    py: Python<'_>,
    result: &ALNSRunWithRestartsResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    let summaries = PyList::empty(py);

    for summary in &result.restart_summaries {
        let entry = PyDict::new(py);
        entry.set_item("restart_index", summary.restart_index)?;
        entry.set_item("seed", summary.seed)?;
        entry.set_item("initial_cost", summary.initial_cost)?;
        entry.set_item("best_cost", summary.best_cost)?;
        entry.set_item("final_cost", summary.final_cost)?;
        entry.set_item("iterations_completed", summary.iterations_completed)?;
        entry.set_item("elapsed_ms", summary.elapsed_ms)?;
        entry.set_item("best_improvement_pct", summary.best_improvement_pct)?;
        summaries.append(entry)?;
    }

    dict.set_item(
        "global_metrics",
        metrics_to_pydict(py, &result.global_metrics)?,
    )?;
    dict.set_item("restart_summaries", summaries)?;

    Ok(dict.into())
}

#[pyclass(unsendable)]
pub struct RustALNSSnapshot {
    snapshot: Option<ALNSEngineSnapshot>,
}

#[pymethods]
impl RustALNSSnapshot {
    #[new]
    fn new() -> Self {
        Self { snapshot: None }
    }

    fn is_initialized(&self) -> bool {
        self.snapshot.is_some()
    }

    fn initial_cost(&self) -> PyResult<f64> {
        self.snapshot
            .as_ref()
            .map(|s| s.initial_cost)
            .ok_or_else(|| PyValueError::new_err("Snapshot is empty"))
    }

    fn duplicate(&self) -> Self {
        Self {
            snapshot: self.snapshot.clone(),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        let repr = match &self.snapshot {
            Some(snapshot) => format!(
                "<RustALNSSnapshot initial_cost={:.3} iteration={}>",
                snapshot.initial_cost, snapshot.iteration
            ),
            None => "<RustALNSSnapshot empty>".to_string(),
        };
        Ok(repr)
    }
}

impl RustALNSSnapshot {
    fn from_snapshot(snapshot: ALNSEngineSnapshot) -> Self {
        Self {
            snapshot: Some(snapshot),
        }
    }

    fn clone_internal(&self) -> PyResult<ALNSEngineSnapshot> {
        self.snapshot
            .clone()
            .ok_or_else(|| PyValueError::new_err("Snapshot is empty"))
    }
}

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
    fn new() -> Self {
        Self { engine: None }
    }

    #[pyo3(signature = (
        problem_instance,
        seed,
        temperature=None,
        theta=None,
        weight_update_interval=None,
        aggressive_search_factor=None,
        algorithm_mode=None
    ))]
    fn initialize_alns(
        &mut self,
        py: Python,
        problem_instance: &str,
        seed: u64,
        temperature: Option<f64>,
        theta: Option<f64>,
        weight_update_interval: Option<usize>,
        aggressive_search_factor: Option<f64>,
        algorithm_mode: Option<&str>,
    ) -> PyResult<PyObject> {
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
        let aggressive_search_factor = aggressive_search_factor.unwrap_or(0.85);
        let max_iterations = 1000; // Default max iterations

        let algorithm_mode = algorithm_mode
            .map(|mode| mode.to_ascii_lowercase())
            .as_deref()
            .map(|mode| match mode {
                "baseline" => Ok(crate::alns::engine::ALNSAlgorithmMode::Baseline),
                "kisialiou" => Ok(crate::alns::engine::ALNSAlgorithmMode::Kisialiou),
                "reinforcement_learning" | "rl" => {
                    Ok(crate::alns::engine::ALNSAlgorithmMode::ReinforcementLearning)
                }
                other => Err(PyValueError::new_err(format!(
                    "Invalid algorithm_mode '{}'. Use 'baseline', 'kisialiou', or 'reinforcement_learning'",
                    other
                ))),
            })
            .transpose()?;

        let engine = ALNSEngine::new_from_instance(
            problem_instance,
            seed,
            temperature,
            theta,
            weight_update_interval,
            aggressive_search_factor,
            max_iterations,
            algorithm_mode.unwrap_or(crate::alns::engine::ALNSAlgorithmMode::Baseline),
        )
        .map_err(|e| PyRuntimeError::new_err(e))?;
        self.engine = Some(engine);
        self.extract_solution_metrics(py)
    }

    fn create_snapshot(&self) -> PyResult<RustALNSSnapshot> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;
        Ok(RustALNSSnapshot::from_snapshot(engine.create_snapshot()))
    }

    fn apply_snapshot(&mut self, py: Python, snapshot: &RustALNSSnapshot) -> PyResult<PyObject> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;
        let snapshot_data = snapshot.clone_internal()?;
        engine.apply_snapshot(&snapshot_data);
        self.extract_solution_metrics(py)
    }

    fn dump_current_solution(&self, path: &str) -> PyResult<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;

        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to create snapshot directory {}: {}",
                        parent.display(),
                        e
                    ))
                })?;
            }
        }

        dump_schedule_to_json(
            &engine.current_solution,
            &engine.context.problem.vessels,
            path,
            &engine.context,
        );

        Ok(())
    }

    #[pyo3(signature = (
        iteration,
        destroy_operator_idx=None,
        repair_operator_idx=None,
        improvement_operator_idx=None,
        mode=None
    ))]
    fn execute_iteration(
        &mut self,
        py: Python,
        iteration: usize,
        destroy_operator_idx: Option<usize>,
        repair_operator_idx: Option<usize>,
        improvement_operator_idx: Option<usize>,
        mode: Option<&str>,
    ) -> PyResult<PyObject> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;

        // Parse mode and create ALNSRunMode
        let run_mode = match mode {
            Some("random") => crate::alns::engine::ALNSRunMode::Random,
            Some("weighted") => crate::alns::engine::ALNSRunMode::Weighted,
            Some("explicit") | None => {
                // For explicit mode or backward compatibility, require operator indices
                let destroy_idx = destroy_operator_idx.ok_or_else(|| {
                    PyRuntimeError::new_err("destroy_operator_idx required for explicit mode")
                })?;
                let repair_idx = repair_operator_idx.ok_or_else(|| {
                    PyRuntimeError::new_err("repair_operator_idx required for explicit mode")
                })?;
                crate::alns::engine::ALNSRunMode::Explicit(destroy_idx, repair_idx)
            }
            Some(invalid_mode) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Invalid mode: {}. Use 'random', 'weighted', or 'explicit'",
                    invalid_mode
                )))
            }
        };

        let metrics = engine
            .run_iteration(run_mode, iteration, improvement_operator_idx)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        metrics_to_pydict(py, &metrics)
    }

    #[pyo3(signature = (iteration, improvement_operator_idx))]
    fn execute_improvement_only(
        &mut self,
        py: Python,
        iteration: usize,
        improvement_operator_idx: usize,
    ) -> PyResult<PyObject> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;

        let metrics = engine
            .run_improvement_only(iteration, improvement_operator_idx)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        metrics_to_pydict(py, &metrics)
    }

    #[pyo3(signature = (iteration, improvement_operator_idx, before_path=None, after_path=None))]
    fn execute_improvement_with_snapshots(
        &mut self,
        py: Python,
        iteration: usize,
        improvement_operator_idx: usize,
        before_path: Option<&str>,
        after_path: Option<&str>,
    ) -> PyResult<PyObject> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;

        let solution_before = engine.current_solution.clone();

        let metrics = engine
            .run_improvement_only(iteration, improvement_operator_idx)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        if let Some(path) = before_path {
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to create snapshot directory {}: {}",
                            parent.display(),
                            e
                        ))
                    })?;
                }
            }

            dump_schedule_to_json(
                &solution_before,
                &engine.context.problem.vessels,
                path,
                &engine.context,
            );
        }

        if let Some(path) = after_path {
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to create snapshot directory {}: {}",
                            parent.display(),
                            e
                        ))
                    })?;
                }
            }

            dump_schedule_to_json(
                &engine.current_solution,
                &engine.context.problem.vessels,
                path,
                &engine.context,
            );
        }

        metrics_to_pydict(py, &metrics)
    }

    #[pyo3(signature = (restarts=1))]
    fn run_with_restarts(&mut self, py: Python, restarts: usize) -> PyResult<PyObject> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;

        let result = engine.run_with_restarts(restarts);
        run_with_restarts_result_to_pydict(py, &result)
    }

    fn extract_solution_metrics(&self, py: Python) -> PyResult<PyObject> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;

        // Get metrics from the current solution
        let current_solution = &engine.current_solution;
        let best_solution = &engine.best_solution;
        let mut temp_solution = current_solution.clone();
        let is_feasible = temp_solution.is_fully_feasible(&engine.context);
        let is_complete = current_solution.is_complete_solution();
        let structure_metrics =
            compute_solution_structure_metrics(current_solution, &engine.context);

        let destroy_success_rates: Vec<f64> = engine
            .alns_context
            .destroy_operator_scores
            .iter()
            .zip(engine.alns_context.destroy_operator_counts.iter())
            .map(|(score, count)| {
                if *count > 0 {
                    score / (*count as f64)
                } else {
                    0.5
                }
            })
            .collect();
        let repair_success_rates: Vec<f64> = engine
            .alns_context
            .repair_operator_scores
            .iter()
            .zip(engine.alns_context.repair_operator_counts.iter())
            .map(|(score, count)| {
                if *count > 0 {
                    score / (*count as f64)
                } else {
                    0.5
                }
            })
            .collect();
        let recent_rewards: Vec<f64> = engine
            .alns_context
            .cost_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        let dict = PyDict::new(py);
        dict.set_item("total_cost", current_solution.total_cost)?;
        dict.set_item("is_complete", is_complete)?;
        dict.set_item("is_feasible", is_feasible)?;
        dict.set_item("num_voyages", structure_metrics.num_voyages)?;
        dict.set_item("num_empty_voyages", structure_metrics.num_empty_voyages)?;
        dict.set_item("num_vessels_used", structure_metrics.num_vessels_used)?;
        dict.set_item(
            "avg_voyage_utilization",
            structure_metrics.avg_voyage_utilization,
        )?;
        dict.set_item(
            "avg_vessel_load_utilization",
            structure_metrics.avg_vessel_load_utilization,
        )?;
        dict.set_item(
            "min_vessel_load_utilization",
            structure_metrics.min_vessel_load_utilization,
        )?;
        dict.set_item(
            "max_vessel_load_utilization",
            structure_metrics.max_vessel_load_utilization,
        )?;
        dict.set_item(
            "avg_vessel_time_utilization",
            structure_metrics.avg_vessel_time_utilization,
        )?;
        dict.set_item(
            "min_vessel_time_utilization",
            structure_metrics.min_vessel_time_utilization,
        )?;
        dict.set_item(
            "max_vessel_time_utilization",
            structure_metrics.max_vessel_time_utilization,
        )?;
        dict.set_item("iteration", engine.iteration)?;
        dict.set_item("temperature", engine.temperature)?;
        dict.set_item("stagnation_count", engine.stagnation_count)?;
        dict.set_item("best_cost", best_solution.total_cost)?;
        dict.set_item("initial_cost", engine.initial_cost)?;
        dict.set_item(
            "destroy_success_rates",
            PyList::new(py, destroy_success_rates),
        )?;
        dict.set_item(
            "repair_success_rates",
            PyList::new(py, repair_success_rates),
        )?;
        dict.set_item("recent_rewards", PyList::new(py, recent_rewards))?;
        Ok(dict.into())
    }

    /// Get operator names and descriptions
    fn get_operator_info(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let (destroy_operators, repair_operators, improvement_operators) =
            ALNSEngine::get_operator_info();
        dict.set_item("destroy_operators", PyList::new(py, destroy_operators))?;
        dict.set_item("repair_operators", PyList::new(py, repair_operators))?;
        dict.set_item(
            "improvement_operators",
            PyList::new(py, improvement_operators),
        )?;
        Ok(dict.into())
    }

    fn export_solution(&self, filepath: &str) -> PyResult<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;
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
        crate::alns::engine::enable_console_logging().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to enable console logging: {}", e))
        })
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
    m.add_class::<RustALNSSnapshot>()?;
    m.add_class::<RustALNSInterface>()?;
    Ok(())
}
