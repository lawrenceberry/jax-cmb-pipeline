"""
JAX CMB Pipeline - Main Entry Point

Cosmological parameter inference pipeline for Planck TT data using
nested sampling with JAX-accelerated likelihoods.
"""
import yaml

# Import pipeline modules
from jax_cmb_pipeline.data_loading import load_planck_data
from jax_cmb_pipeline.emulator import setup_emulator
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
    3. Setup emulator
    4. Create likelihood function
    5. Run nested sampling inference (with uniform priors)
    6. Analyze and plot results
    """
    print("\n" + "=" * 60)
    print("JAX CMB Pipeline - Planck TT Nested Sampling")
    print("=" * 60)
    print()

    # ============================================================
    # 1. Load Configuration
    # ============================================================
    print("Loading configuration...")
    config = load_config('config.yaml')
    print("  Configuration loaded successfully")
    print()

    # Load Planck Data
    observed_tt_ell, observed_tt_cl, observed_tt_cov = load_planck_data(config)

    # Setup Emulator
    emulator = setup_emulator(config)

    # Create Likelihood Function
    log_likelihood = create_log_likelihood(observed_tt_ell, observed_tt_cl, observed_tt_cov, emulator)

    # Run Nested Sampling Inference
    # Note: Nested sampling uses uniform_prior internally to generate
    # both initial points and the prior function
    results = run_inference(config, log_likelihood)

    # Analyze and Plot Results
    analyze_results(results, config)
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nResults saved to: {config['output']['directory']}/")
    print()


if __name__ == "__main__":
    main()
