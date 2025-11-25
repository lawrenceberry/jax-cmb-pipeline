"""
Emulator module for CosmoPowerJAX.
"""
import numpy as np
import jax.numpy as jnp
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ


def setup_emulator(config, tt_ell):
    """
    Initialize CosmoPowerJAX emulator and precompute indices for data selection.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing emulator settings
    tt_ell : np.ndarray
        Multipole moments from observed data

    Returns
    -------
    emulator : CosmoPowerJAX
        Initialized emulator instance
    indices_jax : jax.numpy.ndarray
        Precomputed indices for selecting observed multipoles from emulator output
    """
    print("=" * 60)
    print("Setting Up Emulator")
    print("=" * 60)

    # Initialize emulator
    probe = config['emulator']['probe']
    print(f"Initializing CosmoPowerJAX with probe: {probe}")
    emulator = CPJ(probe=probe)

    print(f"  Emulator modes shape: {emulator.modes.shape}")
    print(f"  Emulator mode range: ℓ = {int(emulator.modes[0])} to {int(emulator.modes[-1])}")

    # ============================================================
    # Precompute indices for efficient JAX-based indexing
    # ============================================================
    print("\nPrecomputing indices for data selection...")

    # Convert to numpy arrays for index computation
    emulator_modes_np = np.array(emulator.modes)
    tt_ell_int = tt_ell.astype(int)

    # For each multipole in tt_ell, find its index in emulator.modes
    indices = np.array([np.where(emulator_modes_np == ell)[0][0]
                        for ell in tt_ell_int])

    # Convert to JAX array (immutable, can be captured in JIT)
    indices_jax = jnp.array(indices)

    print(f"  Precomputed {len(indices_jax)} indices for data selection")
    print(f"  First few indices: {indices_jax[:5]}")
    print(f"  First few multipoles: {tt_ell_int[:5]}")
    print(f"  Corresponding emulator modes: {emulator_modes_np[indices[:5]]}")

    print("=" * 60)
    print()

    return emulator, indices_jax
