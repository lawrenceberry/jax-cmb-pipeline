"""
Emulator module for CosmoPowerJAX.
"""
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ


def setup_emulator(config):
    """
    Initialize CosmoPowerJAX emulator.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing emulator settings

    Returns
    -------
    emulator : CosmoPowerJAX
        Initialized emulator instance
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

    print("=" * 60)
    print()

    return emulator
