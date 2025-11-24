"""
PyO3 Interface Design for Rust ALNS Integration

This file outlines the Python-Rust interface design using PyO3 bindings.
The interface allows the Python RL environment to control the Rust ALNS engine
for single-step execution and state extraction.

Key Design Principles:
1. Minimize data serialization overhead between Python/Rust
2. Expose fine-grained control over ALNS iterations
3. Provide comprehensive solution state information
4. Handle errors gracefully with proper exception mapping
"""

# Required PyO3 additions to Cargo.toml:
# [dependencies]
# pyo3 = { version = "0.20", features = ["extension-module"] }
# 
# [lib]
# name = "rust_alns_py"
# crate-type = ["cdylib"]

# ============================================================================
# RUST SIDE: PyO3 Interface Implementation (to be added to Rust project)
# ============================================================================

"""
// File: rust_alns/src/python_interface.rs
// This is the Rust code that needs to be implemented

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use std::collections::HashMap;
use crate::structs::{solution::Solution, context::Context};
use crate::alns::{engine::ALNSEngine, context::ALNSContext};
use crate::operators::registry::OperatorRegistry;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[pyclass]
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
        }
    }
    
    /// Initialize ALNS run with problem instance and seed
    fn initialize_alns(&mut self, problem_instance: &str, seed: u64) -> PyResult<PyDict> {
        // Load problem data based on instance name
        let (installations_path, vessels_path, base_path) = match problem_instance {
            "SMALL_1" => (
                "../sample/installations/SMALL_1/i_test1.csv",
                "../sample/vessels/SMALL_1/v_test1.csv",
                "../sample/base/SMALL_1/b_test1.csv"
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
        let initial_solution = crate::operators::initial_solution::construct_initial_solution(&context, &mut rng);
        
        // Set up operator registry
        let mut operator_registry = crate::operators::registry::OperatorRegistry::new();
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
        self.context = Some(context);
        self.alns_context = Some(alns_context);
        self.operator_registry = Some(operator_registry);
        self.current_solution = Some(initial_solution.clone());
        self.best_solution = Some(initial_solution.clone());
        self.initial_solution = Some(initial_solution.clone());
        self.rng = Some(StdRng::seed_from_u64(seed));
        self.iteration_count = 0;
        self.stagnation_count = 0;
        
        // Return initial solution metrics as Python dict
        Python::with_gil(|py| -> PyResult<PyDict> {
            self.extract_solution_metrics(py)
        })
    }
    
    /// Execute one ALNS iteration with specified operators
    fn execute_iteration(
        &mut self,
        destroy_operator_idx: usize,
        repair_operator_idx: usize,
        improvement_operator_idx: Option<usize>,
        iteration: usize,
    ) -> PyResult<PyDict> {
        let context = self.context.as_ref().ok_or_else(|| PyRuntimeError::new_err("ALNS not initialized"))?;
        let alns_context = self.alns_context.as_mut().ok_or_else(|| PyRuntimeError::new_err("ALNS context not available"))?;
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
        repair_op.apply(&mut candidate_solution, context, rng);
        candidate_solution.update_total_cost(context);
        
        // Acceptance decision
        let candidate_cost = candidate_solution.total_cost;
        let current_cost = current_solution.total_cost;
        let best_cost = best_solution.total_cost;
        
        let accept = crate::alns::acceptance::accept(current_cost, candidate_cost, self.temperature, rng);
        let mut accepted = false;
        let mut is_new_best = false;
        
        // Update solutions based on acceptance
        if candidate_cost < best_cost {
            *best_solution = candidate_solution.clone();
            *current_solution = candidate_solution;
            accepted = true;
            is_new_best = true;
            self.stagnation_count = 0;
        } else if candidate_cost < current_cost {
            *current_solution = candidate_solution;
            accepted = true;
            self.stagnation_count = 0;
        } else if accept {
            *current_solution = candidate_solution;
            accepted = true;
            self.stagnation_count += 1;
        } else {
            self.stagnation_count += 1;
        }
        
        // Update ALNS context
        alns_context.iteration = iteration;
        
        // Reward operators based on performance
        if is_new_best {
            alns_context.reward_operator("destroy", destroy_operator_idx, 0); // Best reward
            alns_context.reward_operator("repair", repair_operator_idx, 0);
        } else if accepted && candidate_cost < current_cost {
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
        
        // Cool down temperature
        self.temperature = crate::alns::acceptance::cool_down(self.temperature, self.theta, iteration + 1);
        self.iteration_count = iteration;
        
        // Return iteration results
        Python::with_gil(|py| -> PyResult<PyDict> {
            let dict = PyDict::new(py);
            
            // Solution metrics
            let solution_metrics = self.extract_solution_metrics(py)?;
            dict.set_item("solution_metrics", solution_metrics)?;
            
            // Iteration-specific information
            dict.set_item("accepted", accepted)?;
            dict.set_item("is_new_best", is_new_best)?;
            dict.set_item("candidate_cost", candidate_cost)?;
            dict.set_item("current_cost", current_solution.total_cost)?;
            dict.set_item("best_cost", best_solution.total_cost)?;
            dict.set_item("temperature", self.temperature)?;
            dict.set_item("stagnation_count", self.stagnation_count)?;
            dict.set_item("destroy_operator_idx", destroy_operator_idx)?;
            dict.set_item("repair_operator_idx", repair_operator_idx)?;
            
            // Operator performance
            dict.set_item("destroy_weights", alns_context.destroy_operator_weights.clone())?;
            dict.set_item("repair_weights", alns_context.repair_operator_weights.clone())?;
            dict.set_item("destroy_scores", alns_context.destroy_operator_scores.clone())?;
            dict.set_item("repair_scores", alns_context.repair_operator_scores.clone())?;
            
            Ok(dict)
        })
    }
    
    /// Extract comprehensive solution metrics for RL state
    fn extract_solution_metrics(&self, py: Python) -> PyResult<PyDict> {
        let context = self.context.as_ref().ok_or_else(|| PyRuntimeError::new_err("Context not available"))?;
        let current_solution = self.current_solution.as_ref().ok_or_else(|| PyRuntimeError::new_err("Current solution not available"))?;
        let best_solution = self.best_solution.as_ref().ok_or_else(|| PyRuntimeError::new_err("Best solution not available"))?;
        let alns_context = self.alns_context.as_ref().ok_or_else(|| PyRuntimeError::new_err("ALNS context not available"))?;
        
        let dict = PyDict::new(py);
        
        // Basic solution properties
        dict.set_item("total_cost", current_solution.total_cost)?;
        dict.set_item("is_complete", current_solution.is_complete_solution())?;
        dict.set_item("is_feasible", current_solution.is_fully_feasible(context))?;  // Note: this mutates
        
        // Solution structure and vessel utilization (shared helper)
        let structure = compute_solution_structure_metrics(current_solution, context);
        dict.set_item("num_voyages", structure.num_voyages)?;
        dict.set_item("num_empty_voyages", structure.num_empty_voyages)?;
        dict.set_item("num_vessels_used", structure.num_vessels_used)?;
        dict.set_item("avg_voyage_utilization", structure.avg_voyage_utilization)?;
        dict.set_item("avg_vessel_load_utilization", structure.avg_vessel_load_utilization)?;
        dict.set_item("min_vessel_load_utilization", structure.min_vessel_load_utilization)?;
        dict.set_item("max_vessel_load_utilization", structure.max_vessel_load_utilization)?;
        dict.set_item("avg_vessel_time_utilization", structure.avg_vessel_time_utilization)?;
        dict.set_item("min_vessel_time_utilization", structure.min_vessel_time_utilization)?;
        dict.set_item("max_vessel_time_utilization", structure.max_vessel_time_utilization)?;
        
        // Search progression
        dict.set_item("iteration", self.iteration_count)?;
        dict.set_item("temperature", self.temperature)?;
        dict.set_item("stagnation_count", self.stagnation_count)?;
        dict.set_item("best_cost", best_solution.total_cost)?;
        
        // Operator performance
        let destroy_success_rates: Vec<f64> = alns_context.destroy_operator_scores.iter()
            .zip(alns_context.destroy_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        
        let repair_success_rates: Vec<f64> = alns_context.repair_operator_scores.iter()
            .zip(alns_context.repair_operator_counts.iter())
            .map(|(score, count)| if *count > 0 { score / (*count as f64) } else { 0.5 })
            .collect();
        
        dict.set_item("destroy_success_rates", destroy_success_rates)?;
        dict.set_item("repair_success_rates", repair_success_rates)?;
        
        // Recent rewards (last 5 from cost history)
        let recent_rewards: Vec<f64> = alns_context.cost_history.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();
        dict.set_item("recent_rewards", recent_rewards)?;
        
        Ok(dict)
    }
    
    /// Get operator names and descriptions
    fn get_operator_info(&self, py: Python) -> PyResult<PyDict> {
        let dict = PyDict::new(py);
        
        // These should match your actual operators
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
        
        dict.set_item("destroy_operators", destroy_operators)?;
        dict.set_item("repair_operators", repair_operators)?;
        
        Ok(dict)
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
"""

# ============================================================================
# PYTHON SIDE: Interface Usage and Integration
# ============================================================================

import sys
from pathlib import Path

class RustALNSBridge:
    """
    Python wrapper for the Rust ALNS interface providing convenient access
    and error handling for the RL environment.
    """
    
    def __init__(self):
        try:
            # Import the compiled PyO3 module
            import rust_alns_py
            self.rust_interface = rust_alns_py.RustALNSInterface()
            self.operators_info = None
            self.initialized = False
        except ImportError as e:
            raise ImportError(
                f"Failed to import rust_alns_py module. "
                f"Make sure it's compiled and in Python path: {e}"
            )
    
    def initialize(self, problem_instance: str = "SMALL_1", seed: int = 42) -> dict:
        """Initialize ALNS with problem instance and return initial metrics"""
        try:
            result = self.rust_interface.initialize_alns(problem_instance, seed)
            self.operators_info = self.rust_interface.get_operator_info()
            self.initialized = True
            return dict(result)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ALNS: {e}")
    
    def execute_iteration(
        self,
        iteration: int,
        destroy_idx: int,
        repair_idx: int,
        improvement_idx: int | None,
    ) -> dict:
        """Execute one ALNS iteration and return results"""
        if not self.initialized:
            raise RuntimeError("ALNS not initialized. Call initialize() first.")
        
        try:
            result = self.rust_interface.execute_iteration(
                iteration,
                destroy_operator_idx=destroy_idx,
                repair_operator_idx=repair_idx,
                improvement_operator_idx=improvement_idx,
            )
            return dict(result)
        except Exception as e:
            raise RuntimeError(f"Failed to execute iteration {iteration}: {e}")
    
    def get_operator_names(self) -> tuple:
        """Get lists of available operators"""
        if not self.initialized:
            raise RuntimeError("ALNS not initialized.")
        
        return (
            self.operators_info["destroy_operators"],
            self.operators_info["repair_operators"],
            self.operators_info.get("improvement_operators", []),
        )
    
    def export_solution(self, filepath: str) -> None:
        """Export current solution to JSON file"""
        if not self.initialized:
            raise RuntimeError("ALNS not initialized.")
        
        try:
            self.rust_interface.export_solution(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to export solution: {e}")
    
    def get_current_metrics(self) -> dict:
        """Get current solution metrics"""
        if not self.initialized:
            raise RuntimeError("ALNS not initialized.")
        
        try:
            result = self.rust_interface.extract_solution_metrics()
            return dict(result)
        except Exception as e:
            raise RuntimeError(f"Failed to extract metrics: {e}")

# ============================================================================
# BUILD INSTRUCTIONS AND INTEGRATION GUIDE
# ============================================================================

def build_instructions():
    """
    Instructions for setting up the PyO3 integration:
    
    1. Update rust_alns/Cargo.toml:
       [dependencies]
       pyo3 = { version = "0.20", features = ["extension-module"] }
       
       [lib]
       name = "rust_alns_py"
       crate-type = ["cdylib"]
    
    2. Add python_interface.rs to rust_alns/src/
    
    3. Update rust_alns/src/lib.rs:
       pub mod python_interface;
       pub use python_interface::*;
    
    4. Build the Python extension:
       cd rust_alns
       maturin develop --release  # or pip install maturin && maturin develop
    
    5. Test the interface:
       python -c "import rust_alns_py; print('Success!')"
    """
    print(build_instructions.__doc__)

def test_integration():
    """Test the Rust-Python interface"""
    try:
        bridge = RustALNSBridge()
        
        # Initialize
        initial_metrics = bridge.initialize("SMALL_1", seed=42)
        print("Initial metrics:", initial_metrics)
        
        # Get operator info
        destroy_ops, repair_ops, improvement_ops = bridge.get_operator_names()
        print("Destroy operators:", destroy_ops)
        print("Repair operators:", repair_ops)
        print("Improvement operators:", improvement_ops)
        
        # Execute a few iterations
        for i in range(5):
            result = bridge.execute_iteration(
                i,
                i % len(destroy_ops),
                i % len(repair_ops),
                None,
            )
            
            print(f"Iteration {i}: Cost={result['current_cost']:.2f}, "
                  f"Accepted={result['accepted']}, Best={result['best_cost']:.2f}")
        
        # Export solution
        bridge.export_solution("test_solution.json")
        print("Solution exported successfully")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        print("Make sure to build the PyO3 extension first using maturin")

if __name__ == "__main__":
    print("PyO3 Interface Design for Rust ALNS")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_integration()
    elif len(sys.argv) > 1 and sys.argv[1] == "build":
        build_instructions()
    else:
        print("Usage:")
        print("  python pyo3_interface.py build  # Show build instructions")
        print("  python pyo3_interface.py test   # Test the interface")
        print()
        print("Key Interface Functions:")
        print("- initialize_alns(problem_instance, seed)")
        print("- execute_iteration(iteration, destroy_operator_idx?, repair_operator_idx?, improvement_operator_idx?, mode?)")
        print("- extract_solution_metrics()")
        print("- get_operator_info()")
        print("- export_solution(filepath)")