"""
Likelihood computation for CMB data.
"""
import jax
import jax.numpy as jnp


def create_log_likelihood(tt_cl, tt_cov, emulator, indices_jax):
    """
    Create a log-likelihood function using a factory pattern.

    This function creates and returns a JIT-compiled log-likelihood function
    that computes the Gaussian likelihood for CMB TT power spectrum data.

    Parameters
    ----------
    tt_cl : np.ndarray
        Observed C_ℓ values
    tt_cov : np.ndarray
        Covariance matrix
    emulator : CosmoPowerJAX
        Initialized emulator for theory predictions
    indices_jax : jax.numpy.ndarray
        Precomputed indices for selecting observed multipoles

    Returns
    -------
    log_likelihood : callable
        JIT-compiled function that computes log-likelihood for given parameters
    """
    def params_to_theory_specs(params):
        """
        Extract theory predictions at observed multipoles using precomputed indices.

        Parameters
        ----------
        params : jax.numpy.ndarray
            Cosmological parameters [omega_b, omega_cdm, h, tau, n_s, ln10^10A_s]

        Returns
        -------
        predicted_tt_cl : jax.numpy.ndarray
            Theory predictions at observed multipoles
        """
        emulator_predictions = emulator.predict(params)
        # Use JAX array indexing with precomputed indices
        predicted_tt_cl = emulator_predictions[indices_jax]
        return predicted_tt_cl

    def _log_likelihood(predicted_cl, observed_cl, cov_matrix):
        """
        Compute the Gaussian log-likelihood.

        Parameters
        ----------
        predicted_cl : jax.numpy.ndarray
            Predicted C_ℓ values
        observed_cl : np.ndarray
            Observed C_ℓ values
        cov_matrix : np.ndarray
            Covariance matrix

        Returns
        -------
        log_like : float
            Log-likelihood value
        """
        delta = observed_cl - predicted_cl
        inv_cov = jnp.linalg.inv(cov_matrix)
        chi2 = delta.T @ inv_cov @ delta
        log_det_cov = jnp.linalg.slogdet(cov_matrix)[1]
        n = len(observed_cl)
        log_like = -0.5 * (chi2 + log_det_cov + n * jnp.log(2 * jnp.pi))
        return log_like

    @jax.jit
    def log_likelihood(params):
        """
        Compute log-likelihood for given cosmological parameters.

        Parameters
        ----------
        params : jax.numpy.ndarray
            Cosmological parameters [omega_b, omega_cdm, h, tau, n_s, ln10^10A_s]

        Returns
        -------
        log_like : float
            Log-likelihood value
        """
        predicted_tt_cl = params_to_theory_specs(params)
        return _log_likelihood(predicted_tt_cl, tt_cl, tt_cov)

    print("=" * 60)
    print("Likelihood Configuration")
    print("=" * 60)
    print(f"  Data points: {len(tt_cl)}")
    print(f"  Covariance matrix shape: {tt_cov.shape}")
    print("  Likelihood type: Gaussian")
    print("  JIT-compiled: Yes")
    print("=" * 60)
    print()

    return log_likelihood
