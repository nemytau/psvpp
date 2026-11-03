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
        // A horizon shorter than theta (diagnostic runs of a few iterations) would give a negative rate.
        let mu = (1.0 - theta / (max_iterations as f64)).max(0.0);
        temperature * mu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_full_horizon_cools_to_about_exp_minus_theta() {
        // Kisialiou et al. (2018), section 3.8: mu = 1 - theta / rho per iteration, so after rho iterations
        // T = T0 * (1 - theta / rho)^rho ~ T0 * exp(-theta). The engine must be given the real restart length rho.
        let (theta, rho) = (7.0_f64, 300_usize);
        let mut t = 1000.0;
        for _ in 0..rho {
            t = cool_down(t, theta, rho);
        }
        let expected = 1000.0 * (1.0 - theta / rho as f64).powi(rho as i32);
        assert!((t - expected).abs() < 1e-9 * expected);
        assert!(t / 1000.0 > 0.0008 && t / 1000.0 < 0.0011, "got {}", t / 1000.0);
    }

    #[test]
    fn a_300_iteration_restart_against_a_1000_horizon_ends_hot() {
        // The pre-2026-11-03 behaviour (Experiments 8-10): horizon 1000 for 300-iteration restarts.
        let mut t = 1000.0;
        for _ in 0..300 {
            t = cool_down(t, 7.0, 1000);
        }
        assert!(t / 1000.0 > 0.12 && t / 1000.0 < 0.13, "got {}", t / 1000.0);
    }

    #[test]
    fn cooling_rate_never_goes_negative_for_a_very_short_horizon() {
        assert_eq!(cool_down(100.0, 7.0, 2), 0.0);
        assert!(cool_down(100.0, 7.0, 7) >= 0.0);
    }
}
