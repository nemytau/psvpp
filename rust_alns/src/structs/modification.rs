#[derive(Debug, Clone)]
pub struct GreedyInsertion {
    pub visit_id: usize,
    pub voyage_id: usize,
    pub cost: f64,
}

impl GreedyInsertion {
    pub fn new(visit_id: usize, voyage_id: usize, cost: f64) -> Self {
        Self {
            visit_id,
            voyage_id,
            cost,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Modification {
    GreedyInsertion(GreedyInsertion),
    // Future types:
    // OptInsertion(OptInsertion),
    // Relocation(Relocation),
    // Swap(Swap),
    // DepartureShift(DepartureShift),
}

pub trait SolutionModification {
    fn apply(&self, solution: &mut crate::structs::solution::Solution, context: &crate::structs::context::Context) -> Result<(), String>;
    fn cost(&self) -> f64;
}

impl SolutionModification for GreedyInsertion {
    fn apply(&self, solution: &mut crate::structs::solution::Solution, context: &crate::structs::context::Context) -> Result<(), String> {
        solution.greedy_insert_visit(self.visit_id, self.voyage_id, context)
            .map_err(|e| format!("Failed to apply GreedyInsertion: {}", e))
    }

    fn cost(&self) -> f64 {
        self.cost
    }
}