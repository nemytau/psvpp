use crate::operators::traits::{DestroyOperator, RepairOperator, ImprovementOperator};

pub struct OperatorRegistry {
    pub destroy_operators: Vec<Box<dyn DestroyOperator>>,
    pub repair_operators: Vec<Box<dyn RepairOperator>>,
    pub improvement_operators: Vec<Box<dyn ImprovementOperator>>,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self {
            destroy_operators: Vec::new(),
            repair_operators: Vec::new(),
            improvement_operators: Vec::new(),
        }
    }

    pub fn add_destroy_operator(&mut self, operator: Box<dyn DestroyOperator>) {
        self.destroy_operators.push(operator);
    }

    pub fn add_repair_operator(&mut self, operator: Box<dyn RepairOperator>) {
        self.repair_operators.push(operator);
    }

    pub fn add_improvement_operator(&mut self, operator: Box<dyn ImprovementOperator>) {
        self.improvement_operators.push(operator);
    }

    pub fn get_destroy_operator(&self, idx: usize) -> &dyn DestroyOperator {
        &*self.destroy_operators[idx]
    }

    pub fn get_repair_operator(&self, idx: usize) -> &dyn RepairOperator {
        &*self.repair_operators[idx]
    }

    pub fn get_improvement_operator(&self, idx: usize) -> &dyn ImprovementOperator {
        &*self.improvement_operators[idx]
    }
}