"""
Plotting and analysis module for MCMC results.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from getdist import MCSamples
    from getdist import plots as gdplots
    GETDIST_AVAILABLE = True
except ImportError:
    GETDIST_AVAILABLE = False


def analyze_results(states, config):
    """
    Analyze MCMC results and generate plots and statistics.

    Parameters
    ----------
    states : blackjax.mcmc.nuts.NUTSInfo
        NUTS sampling states containing parameter traces
    config : dict
        Configuration dictionary containing output settings

    Returns
    -------
    None
        Saves plots to output directory and prints statistics
    """
    print("=" * 60)
    print("Analyzing Results")
    print("=" * 60)

    # Extract parameter chains
    samples = np.array(states.position)
    param_names = ['omega_b', 'omega_cdm', 'h', 'tau', 'n_s', 'ln10^10A_s']
    param_labels = [r'$\omega_b$', r'$\omega_{cdm}$', r'$h$', r'$\tau$', r'$n_s$', r'$\ln(10^{10}A_s)$']

    # Create output directory
    output_dir = Path(config['output']['directory'])
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Chain shape: {samples.shape}")
    print()

    # ============================================================
    # Compute statistics
    # ============================================================
    print("Posterior Statistics:")
    print("=" * 80)
    print(f"{'Parameter':<15} {'Mean':<12} {'Std':<12} {'16%':<12} {'50%':<12} {'84%':<12}")
    print("-" * 80)

    statistics = {}
    for i, (name, label) in enumerate(zip(param_names, param_labels)):
        chain = samples[:, i]
        mean = np.mean(chain)
        std = np.std(chain)
        q16, q50, q84 = np.percentile(chain, [16, 50, 84])

        statistics[name] = {
            'mean': mean,
            'std': std,
            'q16': q16,
            'median': q50,
            'q84': q84
        }

        print(f"{name:<15} {mean:<12.6f} {std:<12.6f} {q16:<12.6f} {q50:<12.6f} {q84:<12.6f}")

    print("=" * 80)
    print()

    # ============================================================
    # Generate trace plots
    # ============================================================
    if config['output'].get('generate_plots', True):
        print("Generating trace plots...")

        fig, axes = plt.subplots(6, 1, figsize=(12, 14))
        fig.suptitle('MCMC Trace Plots', fontsize=16, y=0.995)

        for i, (name, label) in enumerate(zip(param_names, param_labels)):
            ax = axes[i]
            chain = samples[:, i]

            # Plot trace
            ax.plot(chain, linewidth=0.5, alpha=0.7)
            ax.axhline(statistics[name]['mean'], color='r', linestyle='--',
                      linewidth=1, label=f"Mean: {statistics[name]['mean']:.4f}")
            ax.axhline(statistics[name]['q16'], color='gray', linestyle=':',
                      linewidth=0.5, alpha=0.7)
            ax.axhline(statistics[name]['q84'], color='gray', linestyle=':',
                      linewidth=0.5, alpha=0.7)

            ax.set_ylabel(label, fontsize=12)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            if i == 5:
                ax.set_xlabel('Sample', fontsize=12)

        plt.tight_layout()
        trace_file = output_dir / 'trace_plots.png'
        plt.savefig(trace_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {trace_file}")
        plt.close()

        # ============================================================
        # Generate marginal distributions
        # ============================================================
        print("Generating marginal distributions...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Marginal Posterior Distributions', fontsize=16)

        axes = axes.flatten()
        for i, (name, label) in enumerate(zip(param_names, param_labels)):
            ax = axes[i]
            chain = samples[:, i]

            # Plot histogram
            ax.hist(chain, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')

            # Add statistics lines
            ax.axvline(statistics[name]['mean'], color='r', linestyle='--',
                      linewidth=2, label=f"Mean: {statistics[name]['mean']:.4f}")
            ax.axvline(statistics[name]['q16'], color='gray', linestyle=':',
                      linewidth=1.5, label=f"16%: {statistics[name]['q16']:.4f}")
            ax.axvline(statistics[name]['q84'], color='gray', linestyle=':',
                      linewidth=1.5, label=f"84%: {statistics[name]['q84']:.4f}")

            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        marginal_file = output_dir / 'marginal_distributions.png'
        plt.savefig(marginal_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {marginal_file}")
        plt.close()

        # ============================================================
        # Generate corner plot using GetDist
        # ============================================================
        if GETDIST_AVAILABLE:
            print("Generating corner plot...")

            # Create GetDist MCSamples object
            # GetDist expects parameter names without special characters
            gd_names = ['omega_b', 'omega_cdm', 'h', 'tau', 'n_s', 'ln10^{10}A_s']
            gd_labels = [r'\omega_b', r'\omega_{\rm cdm}', r'h', r'\tau', r'n_s', r'\ln(10^{10}A_s)']

            mc_samples = MCSamples(
                samples=samples,
                names=gd_names,
                labels=gd_labels,
                settings={'smooth_scale_2D': 0.3, 'smooth_scale_1D': 0.3}
            )

            # Create triangle plot (corner plot)
            g = gdplots.get_subplot_plotter(width_inch=10)
            g.triangle_plot(
                [mc_samples],
                filled=True,
                contour_colors=['steelblue'],
                title_limit=1  # Show 68% confidence intervals in titles
            )

            corner_file = output_dir / 'corner_plot.png'
            plt.savefig(corner_file, dpi=150, bbox_inches='tight')
            print(f"  Saved: {corner_file}")
            plt.close()
        else:
            print("  Skipping corner plot (getdist not available)")

    # ============================================================
    # Save samples
    # ============================================================
    if config['output'].get('save_samples', True):
        print("Saving samples...")
        samples_file = output_dir / 'samples.npz'
        np.savez(samples_file,
                 samples=samples,
                 param_names=param_names,
                 statistics=statistics)
        print(f"  Saved: {samples_file}")

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print()
