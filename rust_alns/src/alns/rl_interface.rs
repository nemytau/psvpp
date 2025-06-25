use rand::Rng;

pub trait RLInterface {
    fn select_operator(&self, features: &[f64]) -> usize;
    fn update(&mut self, features: &[f64], reward: f64);
}

pub struct RandomRL {
    pub num_operators: usize,
}

impl RLInterface for RandomRL {
    fn select_operator(&self, _features: &[f64]) -> usize {
        let mut rng = rand::thread_rng();
        rng.gen_range(0..self.num_operators)
    }

    fn update(&mut self, _features: &[f64], _reward: f64) {
        // No-op for random policy
    }
}
