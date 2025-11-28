"""
Likelihood computation for CMB data.
"""
import jax.numpy as jnp
import numpy as np

# CMB temperature for unit conversion
TCMB = 2.7255  # Kelvin


def create_log_likelihood(observed_tt_ell, observed_tt_cl, observed_tt_cov, emulator):
    """
    Create a log-likelihood function using a factory pattern.

    This function creates and returns a JIT-compiled log-likelihood function
    that computes the Gaussian likelihood for CMB TT power spectrum data.

    Parameters
    ----------
    observed_tt_ell : np.ndarray
        Multipole moments from observed data
    observed_tt_cl : np.ndarray
        Observed C_ℓ values
    observed_tt_cov : np.ndarray
        Covariance matrix
    emulator : CosmoPowerJAX
        Initialized emulator for theory predictions

    Returns
    -------
    log_likelihood : callable
        JIT-compiled function that computes log-likelihood for given parameters
    """
    # Precompute indices for efficient JAX-based indexing
    # Convert to numpy arrays for index computation
    emulator_modes_np = np.array(emulator.modes)
    observed_tt_ell_int = observed_tt_ell.astype(int)

    # For each multipole in observed_tt_ell, find its index in emulator.modes
    indices = np.array([np.where(emulator_modes_np == ell)[0][0]
                        for ell in observed_tt_ell_int])

    # Convert to JAX array (immutable, can be captured in JIT)
    indices_jax = jnp.array(indices)

    def log_likelihood(params):
        """
        Compute log-likelihood for given cosmological parameters.

        Parameters
        ----------
        params : Dict
            Cosmological parameters {omega_b, omega_cdm, h, tau_reio, n_s, ln10^{10}A_s}

        Returns
        -------
        log_like : float
            Log-likelihood value
        """
        flat_params = jnp.array([params['omega_b'], params['omega_cdm'], params['h'], params['tau_reio'], params['n_s'], params['ln10^{10}A_s']])
        emulator_predictions = emulator.predict(flat_params)

        # Convert emulator output to match Planck data units
        # Emulator outputs dimensionless C_ℓ/(TCMB²), Planck data is in (TCMB*μK)²
        emulator_predictions_corrected = emulator_predictions * (TCMB**2) * 1e12
        predicted_tt_cl = emulator_predictions_corrected[indices_jax]  # extract only the multipoles we observe

        delta = observed_tt_cl - predicted_tt_cl
        inv_cov = jnp.linalg.inv(observed_tt_cov)
        chi2 = delta.T @ inv_cov @ delta
        log_det_cov = jnp.linalg.slogdet(observed_tt_cov)[1]
        n = len(observed_tt_cl)
        log_like = -0.5 * (chi2 + log_det_cov + n * jnp.log(2 * jnp.pi))

        return log_like
    
    print("=" * 60)
    print("Likelihood Configuration")
    print("=" * 60)
    print(f"  Data points: {len(observed_tt_cl)}")
    print(f"  Multipole range: ℓ = {int(observed_tt_ell[0])} to {int(observed_tt_ell[-1])}")
    print(f"  Emulator range: ℓ = {int(emulator_modes_np[0])} to {int(emulator_modes_np[-1])}")
    print(f"  Unit conversion: TCMB = {TCMB} K, scaling = {(TCMB**2) * 1e12:.3e}")
    print("=" * 60)
    print()

    return log_likelihood
