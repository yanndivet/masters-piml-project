# import os
# os.environ['JAX_PLATFORMS'] = 'cpu'

# from jax import random
# import jax.numpy as jnp
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# import distributions as dist
# import data_generation as data_gen
# import constants as cs
# import sampled_distributions as sd

# def plot_push_forward_analysis(M=10000):
#     # Set up the random keys
#     key = random.key(42)
#     keys = random.split(key, 4)
    
#     # Sample from hyperprior
#     sampled_mu = sd.sample_normal_distribution(keys[0], cs.MU_PHI, cs.SIGMA_PHI, m=M)
#     sampled_tau = sd.sample_inverse_gamma_distribution(keys[1], cs.A_PHI, cs.B_PHI, m=M)
    
#     # Generate population parameters
#     population_samples = sd.sample_lognormal(keys[2], sampled_mu, sampled_tau, M)
    
#     # Create figure with subplots
#     fig = plt.figure(figsize=(15, 12))
#     gs = fig.add_gridspec(3, 2, hspace=0.3)
    
#     # Plot hyperprior distributions
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
    
#     # μ hyperprior
#     sns.histplot(np.array(sampled_mu[0]), bins=40, stat='density', alpha=0.6, ax=ax1, label='μ_ω samples')
#     sns.histplot(np.array(sampled_mu[1]), bins=40, stat='density', alpha=0.6, ax=ax2, label='μ_γ samples')
    
#     # Add true values
#     for ax, true_val, title in zip([ax1, ax2], cs.MU_TARGET, ['μ_ω', 'μ_γ']):
#         ax.axvline(true_val, color='r', linestyle='--', label='True Value')
#         ax.set_title(f'Hyperprior Distribution - {title}')
#         ax.set_xlabel('Value')
#         ax.set_ylabel('Density')
#         ax.legend()
    
#     # Plot population parameter distributions
#     ax3 = fig.add_subplot(gs[1, 0])
#     ax4 = fig.add_subplot(gs[1, 1])
    
#     # Population parameters with better binning
#     true_vals_real = np.exp(cs.MU_TARGET)
    
#     for ax, samples, true_val, title, xlim in zip(
#         [ax3, ax4], 
#         population_samples, 
#         true_vals_real, 
#         ['ω', 'γ'], 
#         [2000, 5000]
#     ):
#         # Create bins focused on the bulk of the distribution
#         q95 = np.percentile(samples, 95)
#         bins = np.linspace(0, min(q95 * 1.5, xlim), 40)
        
#         # Plot histogram with custom bins
#         counts, edges = np.histogram(samples, bins=bins, density=True)
#         ax.bar(edges[:-1], counts, width=np.diff(edges), alpha=0.6, 
#                label=f'{title} samples', align='edge')
#         ax.axvline(true_val, color='r', linestyle='--', label='True Value')
#         ax.set_title(f'Population Distribution - {title}')
#         ax.set_xlabel('Value')
#         ax.set_ylabel('Density')
#         ax.set_xlim(0, xlim)
#         ax.legend()
    
#     # Plot joint distribution
#     ax5 = fig.add_subplot(gs[2, :])
    
#     # Create proper levels for the contour plot
#     x = np.array(population_samples[0])
#     y = np.array(population_samples[1])
    
#     # Filter out extreme values for better visualization
#     mask = (x <= 2000) & (y <= 5000)
#     x = x[mask]
#     y = y[mask]
    
#     # Calculate KDE manually
#     kde = sns.kdeplot(
#         x=x, y=y,
#         ax=ax5,
#         levels=np.linspace(0, 1, 11)[1:],  # 10 increasing levels from 0.1 to 1
#         fill=True,
#         cmap='viridis'
#     )
    
#     ax5.scatter(true_vals_real[0], true_vals_real[1], 
#                 color='red', marker='*', s=200, label='True Value')
#     ax5.set_title('Joint Distribution of Population Parameters')
#     ax5.set_xlabel('ω')
#     ax5.set_ylabel('γ')
#     ax5.set_xlim(0, 2000)
#     ax5.set_ylim(0, 5000)
#     ax5.legend()
    
#     # Add overall title
#     fig.suptitle(f'Push Forward Analysis (M={M} samples)', fontsize=16, y=0.95)
    
#     plt.savefig(f'simulation_results/push_forward_M={M}.png', 
#                 dpi=300, bbox_inches='tight')
#     plt.close()

# if __name__ == "__main__":
#     plot_push_forward_analysis(M=10000)


import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import distributions as dist
import data_generation as data_gen
import constants as cs
import sampled_distributions as sd

def plot_push_forward_analysis(M=10000):
    # Set up the random keys
    key = random.key(42)
    keys = random.split(key, 4)
    
    # Sample from hyperprior
    sampled_mu = sd.sample_normal_distribution(keys[0], cs.MU_PHI, cs.SIGMA_PHI, m=M)
    sampled_tau = sd.sample_inverse_gamma_distribution(keys[1], cs.A_PHI, cs.B_PHI, m=M)
    
    # Generate population parameters
    population_samples = sd.sample_lognormal(keys[2], sampled_mu, sampled_tau, M)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3)
    
    # Plot hyperprior distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # μ hyperprior
    sns.histplot(np.array(sampled_mu[0]), bins=40, stat='density', alpha=0.6, ax=ax1, label='μ_ω samples')
    sns.histplot(np.array(sampled_mu[1]), bins=40, stat='density', alpha=0.6, ax=ax2, label='μ_γ samples')
    
    # Add true values
    for ax, true_val, title in zip([ax1, ax2], cs.MU_TARGET, ['μ_ω', 'μ_γ']):
        ax.axvline(true_val, color='r', linestyle='--', label='True Value')
        ax.set_title(f'Hyperprior Distribution - {title}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    # Plot population parameter distributions
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Population parameters with specific cutoffs
    true_vals_real = np.exp(cs.MU_TARGET)
    xlims = [100, 400]  # Specific cutoffs for ω and γ
    
    for ax, samples, true_val, title, xlim in zip(
        [ax3, ax4], 
        population_samples, 
        true_vals_real, 
        ['ω', 'γ'], 
        xlims
    ):
        # Create bins up to the cutoff
        bins = np.linspace(0, xlim, 40)
        
        # Plot histogram with custom bins
        counts, edges = np.histogram(samples, bins=bins, density=True)
        ax.bar(edges[:-1], counts, width=np.diff(edges), alpha=0.6, 
               label=f'{title} samples', align='edge')
        ax.axvline(true_val, color='r', linestyle='--', label='True Value')
        ax.set_title(f'Population Distribution - {title}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_xlim(0, xlim)
        ax.legend()
    
    # Plot joint distribution
    ax5 = fig.add_subplot(gs[2, :])
    
    # Filter samples for joint plot with specific cutoffs
    x = np.array(population_samples[0])
    y = np.array(population_samples[1])
    mask = (x <= 500) & (y <= 2500)
    x = x[mask]
    y = y[mask]
    
    # Calculate KDE
    kde = sns.kdeplot(
        x=x, y=y,
        ax=ax5,
        levels=np.linspace(0, 1, 11)[1:],
        fill=True,
        cmap='viridis'
    )
    
    ax5.scatter(true_vals_real[0], true_vals_real[1], 
                color='red', marker='*', s=200, label='True Value')
    ax5.set_title('Joint Distribution of Population Parameters')
    ax5.set_xlabel('ω')
    ax5.set_ylabel('γ')
    ax5.set_xlim(0, 200)
    ax5.set_ylim(0, 900)
    ax5.legend()
    
    # Add overall title
    fig.suptitle(f'Push Forward Analysis (M={M} samples)', fontsize=16, y=0.95)
    
    plt.savefig(f'simulation_results/push_forward_M={M}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_push_forward_analysis(M=10000)