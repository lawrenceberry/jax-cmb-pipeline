"""
Plotting and analysis module for nested sampling results using anesthetic.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from anesthetic import NestedSamples, make_2d_axes


def analyze_results(results, config):
    """
    Analyze nested sampling results and generate plots and statistics using anesthetic.

    Parameters
    ----------
    results : BlackJAX nested sampling final state
    config : dict
        Configuration dictionary containing output settings

    Returns
    -------
    None
        Saves plots to output directory and prints statistics
    """
    print("=" * 60)
    print("Analyzing Results with Anesthetic")
    print("=" * 60)

    param_labels = {
        'omega_b': r'$\omega_b$',
        'omega_cdm': r'$\omega_{\mathrm{cdm}}$',
        'h': r'$h$',
        'tau_reio': r'$\tau$',
        'n_s': r'$n_s$',
        'ln10^{10}A_s': r'$\ln(10^{10}A_s)$',
    }

    ns_samples = NestedSamples(
        data=results.particles,
        logL=results.loglikelihood,
        logL_birth=results.loglikelihood_birth,
        labels=param_labels,
    )  # type: ignore

    # Create output directory
    output_dir = Path(config['output']['directory'])
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {len(ns_samples)}")
    print(f"Log-evidence: {ns_samples.logZ():.4f}")
    print()

    # Compute statistics using anesthetic
    print("Posterior Statistics:")
    print("=" * 80)
    print(f"{'Parameter':<20} {'Mean':<12} {'Std':<12}")
    print("-" * 80)

    statistics = {}
    for name in param_labels:
        mean = ns_samples[name].mean()
        std = ns_samples[name].std()

        statistics[name] = {
            'mean': float(mean),
            'std': float(std),
        }

        print(f"{name:<20} {mean:<12.6f} {std:<12.6f}")

    print("=" * 80)
    print()

    # Generate corner plot using anesthetic
    if config['output'].get('generate_plots', True):
        print("Generating corner plot with anesthetic...")

        # Create triangle plot (corner plot)
        kinds = {
            'lower': 'kde_2d',      # 2D KDE contours
            'diagonal': 'hist_1d',  # 1D histograms
            'upper': 'scatter_2d'   # 2D scatter plots
        }

        params = list(param_labels.keys())
        _, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w')
        ns_samples.plot_2d(axes, kinds=kinds, figsize=(12, 12))
        corner_file = output_dir / 'corner_plot.png'
        plt.savefig(corner_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {corner_file}")
        plt.close()

    # Save samples to CSV using anesthetic
    if config['output'].get('save_samples', True):
        print("Saving samples to CSV...")
        samples_file = output_dir / 'nested_samples.csv'
        ns_samples.to_csv(samples_file)
        print(f"  Saved: {samples_file}")

        # Also save a summary
        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Nested Sampling Results Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Log-evidence: {ns_samples.logZ():.4f}\n")
            f.write(f"Total samples: {len(ns_samples)}\n\n")
            f.write("Parameter Statistics:\n")
            f.write("-" * 60 + "\n")
            for name in param_labels.keys():
                mean = ns_samples[name].mean()
                std = ns_samples[name].std()
                f.write(f"{name:20s}: {mean:12.6f} ± {std:12.6f}\n")
        print(f"  Saved: {summary_file}")

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print()
