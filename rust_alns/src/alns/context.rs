use rand::rngs::StdRng;

pub struct ALNSContext {
    pub iteration: usize,
    pub temperature: f64,
    pub best_cost: f64,
    pub rng: StdRng,
    // pub operator_scores: Vec<OperatorScore>, // (add later)
    // pub history: Vec<f64>,                  // (add later)
}