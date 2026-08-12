use rand::Rng;

/// Simulated Annealing acceptance criterion.
/// Returns true if the new solution should be accepted.
///
/// # Arguments
/// * `current_cost` - Cost of the current solution (C(s))
/// * `new_cost` - Cost of the new solution (C(s'))
/// * `temperature` - Current temperature (τ)
/// * `rng` - Random number generator implementing rand::Rng
pub fn accept<R: Rng + ?Sized>(
    current_cost: f64,
    new_cost: f64,
    temperature: f64,
    rng: &mut R,
) -> bool {
    if new_cost < current_cost {
        true
    } else {
        let prob = (-(new_cost - current_cost) / temperature).exp();
        let r = rng.gen_range(0.0..1.0);
        r < prob
    }
}

/// Cooling schedule for Simulated Annealing, per Kisialiou et al. (2018),
/// Section 3.8: mu = 1 - theta/rho is a FIXED per-restart cooling rate
/// (rho = total iterations for the restart, not the current iteration
/// index), applied geometrically every iteration: temp_t = temp_0 * mu^t.
///
/// # Arguments
/// * `temperature` - Current temperature (τ)
/// * `theta` - User-defined parameter controlling cooling rate (θ)
/// * `max_iterations` - Total iteration budget for this restart (ρ), fixed
///   for the whole restart -- NOT the current iteration counter.
pub fn cool_down(temperature: f64, theta: f64, max_iterations: usize) -> f64 {
    if max_iterations == 0 {
        temperature
    } else {
        let mu = 1.0 - theta / (max_iterations as f64);
        temperature * mu
    }
}
