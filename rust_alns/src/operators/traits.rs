use crate::structs::context::Context;
use crate::structs::solution::Solution;
use rand::RngCore;

pub trait DestroyOperator {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore);
    fn requires_consistency(&self) -> bool {
        false
    }
    fn requires_schedule_update(&self) -> bool {
        false
    }
}

pub trait RepairOperator {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore);
    fn requires_consistency(&self) -> bool {
        false
    }
    fn requires_schedule_update(&self) -> bool {
        true
    }
}

pub trait ImprovementOperator {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore);
}
