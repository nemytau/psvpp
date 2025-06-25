use rand::rngs::StdRng;
use rand::Rng; // Ensure Rng trait is imported

pub struct ALNSContext {
    pub iteration: usize,
    pub temperature: f64,
    pub best_cost: f64,
    pub rng: StdRng,
    pub destroy_operator_weights: Vec<f64>,
    pub repair_operator_weights: Vec<f64>,
    pub destroy_operator_scores: Vec<f64>, // πo for destroy
    pub repair_operator_scores: Vec<f64>,  // πo for repair
    pub destroy_operator_counts: Vec<usize>,
    pub repair_operator_counts: Vec<usize>,
    pub cost_history: Vec<f64>,
    pub reaction_factor: f64, // γ
    pub reward_values: Vec<f64>, // [σ1, σ2, σ3]
}

impl ALNSContext {
    // Call at the start of each segment
    pub fn reset_segment_scores(&mut self) {
        self.destroy_operator_scores.fill(0.0);
        self.repair_operator_scores.fill(0.0);
        self.destroy_operator_counts.fill(0);
        self.repair_operator_counts.fill(0);
    }

    // Call when an operator is selected and receives a reward (reward_idx: 0,1,2 for σ1,σ2,σ3)
    pub fn reward_operator(&mut self, operator_type: &str, index: usize, reward_idx: usize) {
        let reward = self.reward_values[reward_idx];
        match operator_type {
            "destroy" => {
                if let Some(score) = self.destroy_operator_scores.get_mut(index) {
                    *score += reward;
                }
                if let Some(count) = self.destroy_operator_counts.get_mut(index) {
                    *count += 1;
                }
            },
            "repair" => {
                if let Some(score) = self.repair_operator_scores.get_mut(index) {
                    *score += reward;
                }
                if let Some(count) = self.repair_operator_counts.get_mut(index) {
                    *count += 1;
                }
            },
            _ => {}
        }
    }

    // Call at the end of a segment to update weights
    pub fn update_operator_weights(&mut self, operator_type: &str) {
        match operator_type {
            "destroy" => {
                for i in 0..self.destroy_operator_weights.len() {
                    let n = self.destroy_operator_counts[i].max(1) as f64;
                    let pi = self.destroy_operator_scores[i];
                    let w = self.destroy_operator_weights[i];
                    self.destroy_operator_weights[i] = (1.0 - self.reaction_factor) * w + self.reaction_factor * (pi / n);
                }
            },
            "repair" => {
                for i in 0..self.repair_operator_weights.len() {
                    let n = self.repair_operator_counts[i].max(1) as f64;
                    let pi = self.repair_operator_scores[i];
                    let w = self.repair_operator_weights[i];
                    self.repair_operator_weights[i] = (1.0 - self.reaction_factor) * w + self.reaction_factor * (pi / n);
                }
            },
            _ => {}
        }
    }

    // Select destroy or repair operator index using weighted roulette
    pub fn select_operator(&mut self, operator_type: &str) -> usize {
        let weights = match operator_type {
            "destroy" => &self.destroy_operator_weights,
            "repair" => &self.repair_operator_weights,
            _ => return 0,
        };
        let total_weight: f64 = weights.iter().sum();
        if total_weight == 0.0 {
            return self.rng.gen_range(0..weights.len());
        }
        let mut pick = self.rng.gen_range(0.0..total_weight);
        for (i, w) in weights.iter().enumerate() {
            if pick < *w {
                return i;
            }
            pick -= *w;
        }
        weights.len() - 1 // fallback
    }

    // Utility to pick operator by index (for use with registry)
    pub fn pick_destroy_operator_idx(&mut self) -> usize {
        self.select_operator("destroy")
    }
    pub fn pick_repair_operator_idx(&mut self) -> usize {
        self.select_operator("repair")
    }

    // Add cost to history
    pub fn add_cost_history(&mut self, cost: f64) {
        self.cost_history.push(cost);
    }
}