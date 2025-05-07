use rand::Rng;
use crate::operators::traits::{DestroyOperator, RepairOperator, ImprovementOperator};

pub struct OperatorRegistry {
    pub destroy_operators: Vec<Box<dyn DestroyOperator>>,
    pub repair_operators: Vec<Box<dyn RepairOperator>>,
    pub improvement_operators: Vec<Box<dyn ImprovementOperator>>,
    pub destroy_weights: Vec<f64>,
    pub repair_weights: Vec<f64>,
    pub improvement_weights: Vec<f64>,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self {
            destroy_operators: Vec::new(),
            repair_operators: Vec::new(),
            improvement_operators: Vec::new(),
            destroy_weights: Vec::new(),
            repair_weights: Vec::new(),
            improvement_weights: Vec::new(),
        }
    }

    pub fn add_destroy_operator(&mut self, operator: Box<dyn DestroyOperator>, weight: f64) {
        self.destroy_operators.push(operator);
        self.destroy_weights.push(weight);
    }

    pub fn add_repair_operator(&mut self, operator: Box<dyn RepairOperator>, weight: f64) {
        self.repair_operators.push(operator);
        self.repair_weights.push(weight);
    }

    pub fn add_improvement_operator(&mut self, operator: Box<dyn ImprovementOperator>, weight: f64) {
        self.improvement_operators.push(operator);
        self.improvement_weights.push(weight);
    }

    pub fn pick_destroy_operator(&self, rng: &mut impl Rng) -> &dyn DestroyOperator {
        let idx = self.weighted_random_choice(&self.destroy_weights, rng);
        &*self.destroy_operators[idx]
    }

    pub fn pick_repair_operator(&self, rng: &mut impl Rng) -> &dyn RepairOperator {
        let idx = self.weighted_random_choice(&self.repair_weights, rng);
        &*self.repair_operators[idx]
    }

    pub fn pick_improvement_operator(&self, rng: &mut impl Rng) -> &dyn ImprovementOperator {
        let idx = self.weighted_random_choice(&self.improvement_weights, rng);
        &*self.improvement_operators[idx]
    }

    fn weighted_random_choice(&self, weights: &[f64], rng: &mut impl Rng) -> usize {
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            // Handle the case where all weights are zero or negative
            return rng.gen_range(0..weights.len());
        }
        
        let r = rng.gen_range(0.0..total_weight);
        let mut cumulative = 0.0;
        
        for (i, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if r < cumulative {
                return i;
            }
        }
        
        // Fallback in case of floating-point precision issues
        weights.len() - 1
    }
}