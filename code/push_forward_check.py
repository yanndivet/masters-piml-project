from jax import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants as cs
import sampled_distributions as sd

def plot_push_forward_analysis(M=20000):
    # Set up the random keys
    key = random.key(42)
    keys = random.split(key, 6)
    
    # Sample from hyperpriors
    sampled_mu = sd.sample_normal_distribution(keys[0], cs.MU_PHI, cs.SIGMA_PHI, m=M)
    sampled_tau = sd.sample_inverse_gamma_distribution(keys[1], cs.A_PHI, cs.B_PHI, m=M)
    
    # Generate population parameters
    population_samples = sd.sample_lognormal_distribution(keys[2], sampled_mu, sampled_tau, M)
    
    # Print some diagnostic information
    print(f"True μ values: {cs.MU_TARGET}")
    print(f"True exp(μ) values: {np.exp(cs.MU_TARGET)}")
    print(f"Mean of sampled population parameters:")
    print(f"ω: {np.median(population_samples[0]):.2f}")
    print(f"γ: {np.median(population_samples[1]):.2f}")
    
    # Create figure with subplots (3x2 grid)
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.3)
    
    # Plot μ hyperprior distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # μ hyperprior histograms
    for ax, samples, true_val, title in zip([ax1, ax2], sampled_mu, cs.MU_TARGET, ['ω', 'γ']):
        sns.histplot(data=np.array(samples), bins=40, stat='density', alpha=0.6, ax=ax, label=f'μ_{title} samples')
        ax.axvline(true_val, color='r', linestyle='--', 
                   label=f'True Value: {true_val:.2f}')
        ax.set_title(f'Hyperprior Distribution - μ_{title}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    # Plot τ hyperprior distributions
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # τ hyperprior histograms
    for ax, samples, true_val, title in zip([ax3, ax4], sampled_tau, cs.TAU_TARGET, ['ω', 'γ']):
        sns.histplot(data=np.array(samples), bins=40, stat='density', alpha=0.6, ax=ax, label=f'τ_{title} samples')
        ax.axvline(true_val, color='r', linestyle='--', 
                   label=f'True Value: {true_val:.2f}')
        ax.set_title(f'Hyperprior Distribution - τ_{title}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    # Plot population parameter distributions
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Population parameters with specific cutoffs
    true_vals_real = np.exp(cs.MU_TARGET)
    xlims = [20, 40]  # Increased limits to show more of the distribution
    
    # For each parameter (ω and γ)
    for ax, samples, mu_true, tau_true, title, xlim in zip(
        [ax5, ax6], 
        population_samples,
        cs.MU_TARGET,  # True μ values
        cs.TAU_TARGET, # True τ values
        ['ω', 'γ'], 
        xlims
    ):
        # Plot histogram of samples
        samples_filtered = np.array(samples)[np.array(samples) <= xlim]
        sns.histplot(data=samples_filtered, bins=40, stat='density', alpha=0.6, ax=ax,
                    label=f'{title} samples')
        
        # Generate points for the true log-normal distribution
        x = np.linspace(0.01, xlim, 1000)
        # Calculate the true log-normal PDF using the true μ and τ
        pdf = np.exp(-(np.log(x) - mu_true)**2 / (2/tau_true)) / (x * np.sqrt(2*np.pi/tau_true))
        
        # Plot the true distribution
        ax.plot(x, pdf, 'r--', label=f'True Log-Normal\n(μ={mu_true:.2f}, τ={tau_true:.2f})')
        
        ax.set_title(f'Population Distribution - {title}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_xlim(0, xlim)
        ax.legend()
    
    # Add overall title
    fig.suptitle(f'Push Forward Analysis (M={M} samples)', fontsize=16, y=0.95)
    
    plt.savefig(f'simulation_results/push_forward_M={M}.pdf', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_push_forward_analysis(M=10000)