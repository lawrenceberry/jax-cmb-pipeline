"""
Data loading module for Planck TT spectrum and covariance.
"""
import numpy as np
import struct
from pathlib import Path


def load_planck_data(config):
    """
    Load Planck TT power spectrum data and covariance matrix.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing data paths under config['data']

    Returns
    -------
    tt_ell : np.ndarray
        Multipole moments (shape: 215,)
    tt_cl : np.ndarray
        C_ℓ values (shape: 215,)
    tt_err : np.ndarray
        Uncertainties (shape: 215,)
    tt_cov : np.ndarray
        Covariance matrix (shape: 215x215)
    """
    # Extract paths from config
    base_path = Path(config['data']['base_path'])
    cl_file = config['data']['cl_file']
    cov_file = config['data']['cov_file']

    print("=" * 60)
    print("Loading Planck TT Data")
    print("=" * 60)

    # ============================================================
    # 1. Read TT Power Spectrum (first 215 lines)
    # ============================================================
    cl_path = base_path / cl_file
    print(f"Loading power spectrum from: {cl_path}")

    data = np.loadtxt(cl_path)
    tt_data = data[0:215]  # Extract only TT spectrum

    # Split into columns
    tt_ell = tt_data[:, 0]      # Multipole moments
    tt_cl = tt_data[:, 1]       # C_ℓ values
    tt_err = tt_data[:, 2]      # Uncertainties

    print(f"  TT spectrum: {len(tt_ell)} data points")
    print(f"  Multipole range: ℓ = {int(tt_ell[0])} to {int(tt_ell[-1])}")

    # ============================================================
    # 2. Read TT Covariance Matrix (top-left 215×215 block)
    # ============================================================
    cov_path = base_path / cov_file
    print(f"Loading covariance matrix from: {cov_path}")

    with open(cov_path, 'rb') as f:
        # Skip 4-byte Fortran record marker
        marker1 = struct.unpack('<i', f.read(4))[0]

        # Read the full 613×613 covariance matrix
        cov_full = np.fromfile(f, dtype='<f8', count=613*613)
        cov_full = cov_full.reshape(613, 613)

        # Extract only the TT block (top-left 215×215)
        tt_cov = cov_full[0:215, 0:215]

    print(f"  Covariance matrix shape: {tt_cov.shape}")
    print(f"  Diagonal range: {tt_cov.diagonal().min():.3e} to {tt_cov.diagonal().max():.3e}")

    # ============================================================
    # Verification
    # ============================================================
    print("\nFirst few data points:")
    print(f"{'ℓ':>6} {'C_ℓ':>12} {'σ(C_ℓ)':>12} {'sqrt(Cov_ii)':>12}")
    print("-" * 48)
    for i in range(5):
        print(f"{int(tt_ell[i]):>6} {tt_cl[i]:>12.6e} {tt_err[i]:>12.6e} {np.sqrt(tt_cov[i,i]):>12.6e}")

    print("=" * 60)
    print()

    return tt_ell, tt_cl, tt_err, tt_cov
