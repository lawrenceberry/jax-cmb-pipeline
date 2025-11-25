"""
Prior definitions for cosmological parameters.
"""
import jax
import jax.numpy as jnp


def create_log_prior(config):
    """
    Create a log-prior function with uniform priors on cosmological parameters.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing prior bounds under config['priors']

    Returns
    -------
    log_prior : callable
        JIT-compiled function that computes log-prior for given parameters
    """
    # Extract prior bounds from config
    priors = config['priors']

    omega_b_min, omega_b_max = priors['omega_b']
    omega_cdm_min, omega_cdm_max = priors['omega_cdm']
    h_min, h_max = priors['h']
    tau_min, tau_max = priors['tau']
    n_s_min, n_s_max = priors['n_s']
    ln10_10_A_s_min, ln10_10_A_s_max = priors['ln10_10_A_s']

    @jax.jit
    def log_prior(params):
        """
        Log prior for cosmological parameters with uniform distributions.

        Parameters (in order):
        - omega_b: Physical baryon density (ωb = Ωb*h^2)
        - omega_cdm: Physical cold dark matter density (ωc = Ωc*h^2)
        - h: Reduced Hubble constant (H0 = 100h km/s/Mpc)
        - tau: Optical depth to reionization
        - n_s: Scalar spectral index
        - ln10^10A_s: Log of primordial curvature perturbation amplitude

        Returns
        -------
        log_prob : float
            log(prior probability) = 0 if within bounds, -inf otherwise
        """
        omega_b, omega_cdm, h, tau, n_s, ln10_10_A_s = params

        # Check if all parameters are within bounds
        # For uniform prior: log(P) = 0 if in bounds, -inf if outside
        in_bounds = (
            (omega_b >= omega_b_min) & (omega_b <= omega_b_max) &
            (omega_cdm >= omega_cdm_min) & (omega_cdm <= omega_cdm_max) &
            (h >= h_min) & (h <= h_max) &
            (tau >= tau_min) & (tau <= tau_max) &
            (n_s >= n_s_min) & (n_s <= n_s_max) &
            (ln10_10_A_s >= ln10_10_A_s_min) & (ln10_10_A_s <= ln10_10_A_s_max)
        )

        # Return 0.0 if in bounds, -inf otherwise (using jax.numpy)
        return jnp.where(in_bounds, 0.0, -jnp.inf)

    print("=" * 60)
    print("Prior Configuration")
    print("=" * 60)
    print(f"  omega_b:     [{omega_b_min}, {omega_b_max}]")
    print(f"  omega_cdm:   [{omega_cdm_min}, {omega_cdm_max}]")
    print(f"  h:           [{h_min}, {h_max}]")
    print(f"  tau:         [{tau_min}, {tau_max}]")
    print(f"  n_s:         [{n_s_min}, {n_s_max}]")
    print(f"  ln10^10A_s:  [{ln10_10_A_s_min}, {ln10_10_A_s_max}]")
    print("=" * 60)
    print()

    return log_prior
