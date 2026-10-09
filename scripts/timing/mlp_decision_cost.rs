//! Native cost of one RL decision's forward pass (PPO actor + argmax), as an optimised
//! implementation would run it. Used for the per-decision charge in
//! scripts/run_experiment_rl_improvement_mvp.py (NATIVE_FORWARD_US) and for the overhead
//! analysis in docs/EXPERIMENTS.md ("Timing protocol and RL decision overhead").
//!
//! Network: <inputs> -> 64 -> 64 -> <actions>, tanh hidden activations (SB3 MlpPolicy actor).
//! Plain scalar loops, no SIMD: a conservative native figure.
//!
//!   rustc -C opt-level=3 -o /tmp/mlp_decision_cost scripts/timing/mlp_decision_cost.rs
//!   /tmp/mlp_decision_cost 45 48      # features_improvement_v3 + op_triple_v2
//!
//! Build natively (arm64 on Apple Silicon); do not cross-compile for x86_64.
use std::hint::black_box;
use std::time::Instant;

fn layer(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32], act: bool) {
    let n_in = x.len();
    for (j, o) in out.iter_mut().enumerate() {
        let row = &w[j * n_in..(j + 1) * n_in];
        let mut s = b[j];
        for i in 0..n_in {
            s += row[i] * x[i];
        }
        *o = if act { s.tanh() } else { s };
    }
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("usage: <inputs> <actions>")).collect();
    let (d_in, h, d_out) = (args[0], 64usize, args[1]);
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / 2147483648.0) - 0.5
    };
    let w1: Vec<f32> = (0..h * d_in).map(|_| rnd()).collect();
    let b1: Vec<f32> = (0..h).map(|_| rnd()).collect();
    let w2: Vec<f32> = (0..h * h).map(|_| rnd()).collect();
    let b2: Vec<f32> = (0..h).map(|_| rnd()).collect();
    let w3: Vec<f32> = (0..d_out * h).map(|_| rnd()).collect();
    let b3: Vec<f32> = (0..d_out).map(|_| rnd()).collect();
    let mut x: Vec<f32> = (0..d_in).map(|_| rnd()).collect();
    let (mut h1, mut h2, mut logits) = (vec![0f32; h], vec![0f32; h], vec![0f32; d_out]);
    let n = 1_000_000;
    let start = Instant::now();
    let mut acc = 0usize;
    for k in 0..n {
        x[k % d_in] += 1e-6; // a new state for every decision
        layer(&w1, &b1, black_box(&x), &mut h1, true);
        layer(&w2, &b2, &h1, &mut h2, true);
        layer(&w3, &b3, &h2, &mut logits, false);
        let mut best = 0;
        for j in 1..d_out {
            if logits[j] > logits[best] {
                best = j;
            }
        }
        acc += best;
    }
    let us = start.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    println!("{:>3} inputs -> 64 -> 64 -> {:>3} actions: {:.2} us per decision (checksum {})", d_in, d_out, us, black_box(acc));
}
