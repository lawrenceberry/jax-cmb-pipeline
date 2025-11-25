"""
JAX CMB Pipeline - Main Entry Point

Cosmological parameter inference pipeline for Planck TT data using
NUTS MCMC sampling with JAX-accelerated likelihoods.
"""
import yaml
from pathlib import Path

# Import pipeline modules
from jax_cmb_pipeline.data_loading import load_planck_data
from jax_cmb_pipeline.emulator import setup_emulator
from jax_cmb_pipeline.priors import create_log_prior
from jax_cmb_pipeline.likelihood import create_log_likelihood
from jax_cmb_pipeline.inference import run_inference
from jax_cmb_pipeline.plotting import analyze_results


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main pipeline execution.

    Pipeline steps:
    1. Load configuration
    2. Load Planck TT data
    3. Setup emulator and precompute indices
    4. Create likelihood function
    5. Create prior function
    6. Create posterior function
    7. Run NUTS inference
    8. Analyze and plot results
    """
    print("\n" + "=" * 60)
    print("JAX CMB Pipeline - Planck TT Inference")
    print("=" * 60)
    print()

    # ============================================================
    # 1. Load Configuration
    # ============================================================
    print("Loading configuration...")
    config = load_config('config.yaml')
    print("  Configuration loaded successfully")
    print()

    # ============================================================
    # 2. Load Planck Data
    # ============================================================
    tt_ell, tt_cl, tt_err, tt_cov = load_planck_data(config)

    # ============================================================
    # 3. Setup Emulator
    # ============================================================
    emulator, indices_jax = setup_emulator(config, tt_ell)

    # ============================================================
    # 4. Create Likelihood Function
    # ============================================================
    log_likelihood = create_log_likelihood(tt_cl, tt_cov, emulator, indices_jax)

    # ============================================================
    # 5. Create Prior Function
    # ============================================================
    log_prior = create_log_prior(config)

    # ============================================================
    # 6. Create Posterior Function
    # ============================================================
    print("=" * 60)
    print("Creating Posterior Function")
    print("=" * 60)
    print("  Posterior = Prior + Likelihood")
    print("=" * 60)
    print()

    def log_posterior(params):
        """Combine prior and likelihood."""
        return log_prior(params) + log_likelihood(params)

    # ============================================================
    # 7. Run NUTS Inference
    # ============================================================
    states = run_inference(log_posterior, config)

    # ============================================================
    # 8. Analyze and Plot Results
    # ============================================================
    analyze_results(states, config)

    # ============================================================
    # Done
    # ============================================================
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nResults saved to: {config['output']['directory']}/")
    print()


if __name__ == "__main__":
    main()
