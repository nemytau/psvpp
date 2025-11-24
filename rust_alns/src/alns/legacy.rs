use crate::alns::acceptance;
use crate::alns::context::ALNSContext;
use crate::operators::registry::OperatorRegistry;
use crate::structs::context::Context;
use crate::structs::solution::Solution;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Legacy ALNS Engine for backward compatibility with existing Rust code
pub struct ALNSEngineLegacy<'a> {
    pub context: &'a Context,
    pub alns_context: &'a mut ALNSContext,
    pub operator_registry: OperatorRegistry,
    pub current_solution: Solution,
    pub best_solution: Solution,
    pub temperature: f64,
    pub theta: f64, // cooling parameter
    pub iteration: usize,
    pub max_iterations: usize,
    pub rng: StdRng,
}

impl<'a> ALNSEngineLegacy<'a> {
    pub fn new(
        context: &'a Context,
        alns_context: &'a mut ALNSContext,
        operator_registry: OperatorRegistry,
        mut initial_solution: Solution,
        temperature: f64,
        theta: f64,
        max_iterations: usize,
        rng_seed: u64,
    ) -> Self {
        initial_solution.update_total_cost(context);
        let best_solution = initial_solution.clone();
        Self {
            context,
            alns_context,
            operator_registry,
            current_solution: initial_solution,
            best_solution,
            temperature,
            theta,
            iteration: 0,
            max_iterations,
            rng: StdRng::seed_from_u64(rng_seed),
        }
    }

    pub fn run(&mut self) {
        for iter in 0..self.max_iterations {
            self.iteration = iter;
            // Select destroy and repair operator indices using context weights
            let destroy_idx = self.alns_context.pick_destroy_operator_idx();
            let repair_idx = self.alns_context.pick_repair_operator_idx();
            let destroy_op = self.operator_registry.get_destroy_operator(destroy_idx);
            let repair_op = self.operator_registry.get_repair_operator(repair_idx);

            // Clone current solution for modification
            let mut candidate_solution = self.current_solution.clone();
            // Apply destroy
            destroy_op.apply(&mut candidate_solution, self.context, &mut self.rng);
            // Apply repair
            repair_op.apply(&mut candidate_solution, self.context, &mut self.rng);

            // Evaluate
            let candidate_cost = candidate_solution.total_cost;
            let current_cost = self.current_solution.total_cost;
            let best_cost = self.best_solution.total_cost;

            // Accept or reject
            let accept = acceptance::accept(
                current_cost,
                candidate_cost,
                self.temperature,
                &mut self.rng,
            );
            if candidate_cost < best_cost {
                self.best_solution = candidate_solution.clone();
                self.current_solution = candidate_solution;
            } else if candidate_cost < current_cost {
                self.current_solution = candidate_solution;
            } else if accept {
                self.current_solution = candidate_solution;
            }
            // Cooling
            self.temperature = acceptance::cool_down(self.temperature, self.theta, iter + 1);
            // TODO: Update operator scores (stub)
        }
    }

    /// Run the ALNS engine, logging each iteration using the provided logger
    pub fn run_with_logger(&mut self, logger: &mut crate::alns::logger::AlnsLogger) {
        use std::time::Instant;
        for iter in 0..self.max_iterations {
            self.iteration = iter;
            let iter_start = Instant::now();
            let destroy_weights_before = self.alns_context.destroy_operator_weights.clone();
            let repair_weights_before = self.alns_context.repair_operator_weights.clone();
            let destroy_idx = self.alns_context.pick_destroy_operator_idx();
            let repair_idx = self.alns_context.pick_repair_operator_idx();
            let destroy_weight = destroy_weights_before[destroy_idx];
            let repair_weight = repair_weights_before[repair_idx];
            let destroy_op = self.operator_registry.get_destroy_operator(destroy_idx);
            let repair_op = self.operator_registry.get_repair_operator(repair_idx);
            let mut candidate_solution = self.current_solution.clone();
            let mut accepted = false;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                destroy_op.apply(&mut candidate_solution, self.context, &mut self.rng);
                repair_op.apply(&mut candidate_solution, self.context, &mut self.rng);
                candidate_solution.update_total_cost(self.context);
            }));
            if let Err(e) = result {
                println!("[ERROR] Panic at iteration {}: {:?}", iter, e);
            }
            let candidate_cost = candidate_solution.total_cost;
            let current_cost = self.current_solution.total_cost;
            let best_cost = self.best_solution.total_cost;
            let accept = crate::alns::acceptance::accept(
                current_cost,
                candidate_cost,
                self.temperature,
                &mut self.rng,
            );
            if candidate_cost < best_cost {
                self.best_solution = candidate_solution.clone();
                self.best_solution.total_cost = candidate_cost;
                self.current_solution = candidate_solution;
                self.current_solution.update_total_cost(self.context);
                accepted = true;
            } else if candidate_cost < current_cost {
                self.current_solution = candidate_solution;
                self.current_solution.update_total_cost(self.context);
                accepted = true;
            } else if accept {
                self.current_solution = candidate_solution;
                self.current_solution.update_total_cost(self.context);
                accepted = true;
            }
            // Reward assignment
            let mut destroy_reward_idx = None;
            let mut repair_reward_idx = None;
            if candidate_cost < best_cost {
                destroy_reward_idx = Some(0);
                repair_reward_idx = Some(0);
            } else if candidate_cost < current_cost {
                destroy_reward_idx = Some(1);
                repair_reward_idx = Some(1);
            } else if accept {
                destroy_reward_idx = Some(2);
                repair_reward_idx = Some(2);
            }
            if let Some(idx) = destroy_reward_idx {
                self.alns_context
                    .reward_operator("destroy", destroy_idx, idx);
            }
            if let Some(idx) = repair_reward_idx {
                self.alns_context.reward_operator("repair", repair_idx, idx);
            }
            // Weight update every 10 iterations
            if (iter + 1) % 10 == 0 {
                self.alns_context.update_operator_weights("destroy");
                self.alns_context.update_operator_weights("repair");
                self.alns_context.reset_segment_scores();
                log::debug!("[ALNS] Operator weights updated at iteration {}", iter + 1);
            }
            let duration = iter_start.elapsed();
            logger.log_iteration(
                iter,
                self.current_solution.total_cost,
                self.best_solution.total_cost,
                destroy_idx,
                repair_idx,
                destroy_weight,
                repair_weight,
                accepted,
                self.temperature,
                duration,
            );
            // Optionally print operator weights
            println!("[{}] Destroy weights: {:?}", iter, destroy_weights_before);
            println!("[{}] Repair weights:  {:?}", iter, repair_weights_before);
        }
    }
}
