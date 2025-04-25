use rand::Rng;
use crate::structs::solution::Solution;
use crate::structs::context::Context;

pub trait DestroyOperator {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut impl Rng);
}

pub trait RepairOperator {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut impl Rng);
}