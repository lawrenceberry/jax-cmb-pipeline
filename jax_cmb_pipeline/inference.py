"""
Nested sampling inference using blackjax.nss.
"""
import jax
import blackjax
from blackjax.ns.utils import uniform_prior
from tqdm import tqdm

def run_inference(config, log_likelihood):
    """
    Run nested sampling to infer cosmological parameters.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing sampling settings and prior bounds
    log_likelihood : callable
        JIT-compiled log-likelihood function

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'samples': Parameter samples (combined dead + live points)
        - 'logL': Log-likelihood values for each sample
        - 'logwt': Log-weights for each sample
        - 'logZ': Log-evidence estimate
        - 'logZ_err': Uncertainty in log-evidence
        - 'live': Final live points state
    """
    print("=" * 60)
    print("Nested Sampling Configuration")
    print("=" * 60)

    # Extract sampling settings
    num_live_points = config['sampling']['num_live_points']
    num_delete = config['sampling'].get('num_delete', 50)
    num_inner_steps = config['sampling'].get('num_inner_steps', 20)
    convergence_threshold = config['sampling'].get('convergence_threshold', -3.0)
    random_seed = config['sampling']['random_seed']

    # Extract prior bounds for uniform_prior
    priors_config = config['priors']
    prior_bounds = {
        'omega_b': tuple(priors_config['omega_b']),
        'omega_cdm': tuple(priors_config['omega_cdm']),
        'h': tuple(priors_config['h']),
        'tau_reio': tuple(priors_config['tau_reio']),
        'n_s': tuple(priors_config['n_s']),
        'ln10^{10}A_s': tuple(priors_config['ln10^{10}A_s']),
    }

    print(f"  Live points: {num_live_points}")
    print(f"  Delete per iteration: {num_delete}")
    print(f"  Inner MCMC steps: {num_inner_steps}")
    print(f"  Convergence: {convergence_threshold}")
    print("=" * 60)
    print()

    # Generate initial live points and prior function
    rng_key = jax.random.PRNGKey(random_seed)
    rng_key, prior_key = jax.random.split(rng_key)
    particles, uniform_log_prior = uniform_prior(prior_key, num_live_points, prior_bounds)

    print(f"\nInitializing with {num_live_points} live points...")
    print(f"Parameters: {', '.join(prior_bounds.keys())}")

    nested_sampler = blackjax.nss(
        logprior_fn=uniform_log_prior,
        loglikelihood_fn=log_likelihood,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )

    init_fn = jax.jit(nested_sampler.init)
    step_fn = jax.jit(nested_sampler.step)

    live = init_fn(particles)
    dead = []
    with tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not live.logZ_live - live.logZ < convergence_threshold:
            rng_key, subkey = jax.random.split(rng_key, 2)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(num_delete)

    dead = blackjax.ns.utils.finalise(live, dead)

    print(f"\n{'=' * 60}")
    print(f"Nested Sampling Complete - {len(dead)} iterations")
    print("=" * 60)
    print()

    return dead