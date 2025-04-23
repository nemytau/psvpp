use crate::{structs::problem_data::ProblemData, utils::tsp_solver::TSPSolver};

pub struct Context {
    pub problem: ProblemData,
    pub tsp_solver: TSPSolver,
    // maybe: distance_manager, cost_evaluator, logger, ...
}

impl Context {
    pub fn new(problem: ProblemData, tsp_solver: TSPSolver) -> Self {
        Self {
            problem,
            tsp_solver,
        }
    }
}