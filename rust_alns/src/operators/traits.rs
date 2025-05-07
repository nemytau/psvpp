use rand::RngCore;
use crate::structs::solution::Solution;
use crate::structs::context::Context;

pub trait DestroyOperator {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore);
}

pub trait RepairOperator {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore);
}

pub trait ImprovementOperator {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore);
}