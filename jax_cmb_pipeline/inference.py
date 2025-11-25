"""
MCMC inference using NUTS sampler.
"""
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from datetime import datetime


def run_inference(log_posterior, config):
    """
    Run NUTS MCMC sampling to infer cosmological parameters.

    Parameters
    ----------
    log_posterior : callable
        JIT-compiled log-posterior function
    config : dict
        Configuration dictionary containing sampling settings and initial position

    Returns
    -------
    states : blackjax.mcmc.nuts.NUTSInfo
        NUTS sampling states containing parameter traces and diagnostics
    """
    print("=" * 60)
    print("MCMC Inference Configuration")
    print("=" * 60)

    # Extract sampling settings
    num_samples = config['sampling']['num_samples']
    step_size = config['sampling']['step_size']
    inv_mass_matrix = np.array(config['sampling']['inv_mass_matrix'])
    random_seed = config['sampling']['random_seed']

    # Extract initial position
    init_pos = config['initial_position']
    initial_position = np.array([
        init_pos['omega_b'],
        init_pos['omega_cdm'],
        init_pos['h'],
        init_pos['tau'],
        init_pos['n_s'],
        init_pos['ln10_10_A_s']
    ])

    print(f"  Sampler: NUTS (No-U-Turn Sampler)")
    print(f"  Number of samples: {num_samples}")
    print(f"  Step size: {step_size}")
    print(f"  Random seed: {random_seed}")
    print(f"  Initial position:")
    print(f"    omega_b     = {initial_position[0]}")
    print(f"    omega_cdm   = {initial_position[1]}")
    print(f"    h           = {initial_position[2]}")
    print(f"    tau         = {initial_position[3]}")
    print(f"    n_s         = {initial_position[4]}")
    print(f"    ln10^10A_s  = {initial_position[5]}")
    print("=" * 60)
    print()

    # Initialize NUTS sampler
    print("Initializing NUTS sampler...")
    nuts = blackjax.nuts(log_posterior, step_size, inv_mass_matrix)
    initial_state = nuts.init(initial_position)

    # Create JIT-compiled kernel
    nuts_kernel = jax.jit(nuts.step)

    # Define inference loop
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        """Run MCMC sampling loop."""
        @jax.jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states

    # Run sampling
    print(f"Running MCMC sampling ({num_samples} samples)...")
    print("This may take several minutes...")
    start_time = datetime.now()

    rng_key = jax.random.PRNGKey(random_seed)
    states = inference_loop(rng_key, nuts_kernel, initial_state, num_samples)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\nSampling completed in {duration:.2f} seconds")
    print(f"Average time per sample: {duration/num_samples:.3f} seconds")
    print("=" * 60)
    print()

    return states
