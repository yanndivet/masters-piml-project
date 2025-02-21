import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'

from jax import random
import jax.numpy as jnp

from numpyro.infer import MCMC, NUTS

from time import time
import polars as pl

import io
from contextlib import redirect_stdout

import constants as cs
import distributions as dist
import true_observations as to

def plot_mcmc_chains(posterior_samples, warmup_samples, number_systems, folder_name):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define parameter names using LaTeX notation
    HYPERPARAMETER_NAMES = [r'$\mu_\omega$', r'$\mu_\gamma$', r'$\tau_\omega$', r'$\tau_\gamma$']
    
    # Reshape samples to (n_samples, n_params)
    posterior_reshaped = posterior_samples.squeeze()
    warmup_reshaped = warmup_samples.squeeze()
    
    # Create subplots for first few parameters
    n_params_to_plot = min(4, posterior_reshaped.shape[1])
    fig, axes = plt.subplots(n_params_to_plot, 1, figsize=(12, 3*n_params_to_plot))
    
    # Use LaTeX for the title
    fig.suptitle(f'MCMC Chain Traces (including warmup) for $N = {number_systems}$')
    
    # Plot each parameter
    for i in range(n_params_to_plot):
        # Get samples for current parameter
        warmup_param = warmup_reshaped[:, i]
        posterior_param = posterior_reshaped[:, i]
        
        # Create x-axes for both phases
        x_warmup = np.arange(len(warmup_param))
        x_posterior = np.arange(len(warmup_param), len(warmup_param) + len(posterior_param))
        
        # Plot warmup phase
        axes[i].plot(x_warmup, warmup_param, 'gray', alpha=0.5, linewidth=0.5, label='Warmup')
        
        # Plot sampling phase
        axes[i].plot(x_posterior, posterior_param, 'b-', linewidth=0.5, label='Sampling')
        
        # Add true value line
        axes[i].axhline(y=cs.TARGET_HYPERPARAMETERS[i], color='r', linestyle='--', 
                       alpha=0.8, label='True Value')
        
        # Add vertical line separating warmup and sampling phases
        axes[i].axvline(x=len(warmup_param), color='k', linestyle=':', alpha=0.5,
                       label='End of Warmup')
        
        # Customize plot with LaTeX parameter names
        axes[i].set_title(HYPERPARAMETER_NAMES[i])
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add legend (only for the first subplot to avoid redundancy)
        if i == 0:
            axes[i].legend(loc='upper right', fontsize='small')
    
    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.tight_layout()
    plt.savefig(f'simulation_results/{folder_name}/mcmc_chains_N={number_systems}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def parse_mcmc_summary(mcmc: MCMC, number_systems):
    # Capture print_summary output
    f = io.StringIO()
    with redirect_stdout(f):
        mcmc.print_summary()
    summary_text = f.getvalue()

    df_mcmc_results_per_system = pl.DataFrame(schema=[
        ('number of systems', pl.Int32),
        ('hyperparameter', pl.String),
        ('5th percentile', pl.Float64),
        ('mean', pl.Float64), 
        ('95th percentile', pl.Float64)
    ])
    
    for i, line in enumerate(summary_text.split('\n')[2:6]):  # Skip header, take first 4 params
        if not line.strip():  # Skip empty lines
            continue
        values = line.split()
        stats_data = pl.DataFrame({
            'number of systems': number_systems,
            'hyperparameter': cs.HYPERPARAMETER_NAMES[i],
            '5th percentile': float(values[4]),
            'mean': float(values[1]),
            '95th percentile': float(values[5])}, 
            schema={'number of systems': pl.Int32,
                    'hyperparameter': pl.String,
                    '5th percentile': pl.Float64,
                    'mean': pl.Float64, 
                    '95th percentile': pl.Float64}
        )
        df_mcmc_results_per_system.extend(stats_data)

    return df_mcmc_results_per_system

def get_acceptance_stats(mcmc: MCMC):
    """Get acceptance statistics from MCMC sampler"""
    if not isinstance(mcmc.sampler, NUTS):
        raise ValueError("This function only works with NUTS sampler")
    
    # Get the current state
    state = mcmc._states[-1]  # Get last state
    diagnostics = mcmc.sampler.get_diagnostics_str(state)
    print("Raw diagnostics:", diagnostics)
    
    return diagnostics

def run_mcmc(number_systems=cs.N_SYSTEMS, folder_name="mcmc_results2"):
    key = random.key(number_systems)
    key_mcmc_warmup, key_mcmc_posterior = random.split(key, 2)
    
    population_initial_parameters = jnp.tile(cs.MU_INITIAL, number_systems)
    initial_parameters = jnp.concatenate([cs.INITIAL_HYPERPARAMETERS, population_initial_parameters])
    initial_parameters_tiled = jnp.tile(initial_parameters, (cs.N_CHAINS, 1))
    noisy_initial_parameters_tiled = initial_parameters_tiled # add noise?

    target_distribution = lambda params: dist.log_posterior_distribution(
        params, 
        to.read_true_observations(number_systems), 
        number_systems
    )

    kernel = NUTS(potential_fn=target_distribution,
             step_size=1e-4,  
             adapt_step_size=True,
             target_accept_prob=0.65,  
             max_tree_depth=8,
             adapt_mass_matrix=True,
             dense_mass=False)
    mcmc = MCMC(kernel, 
                num_warmup=500,
                num_samples=1000,
                num_chains=cs.N_CHAINS,
                chain_method='parallel')
    
    warmup_keys = random.split(key_mcmc_warmup, cs.N_CHAINS)
    posterior_keys = random.split(key_mcmc_posterior, cs.N_CHAINS)

    start_time = time()
    mcmc.warmup(warmup_keys, collect_warmup=True, init_params=noisy_initial_parameters_tiled)
    warmup_samples = mcmc.get_samples()
    mcmc.run(posterior_keys, init_params=noisy_initial_parameters_tiled)
    end_time = time()
    
    posterior_samples = mcmc.get_samples()

    plot_mcmc_chains(posterior_samples, warmup_samples, number_systems, folder_name)

    divergences = mcmc.get_extra_fields()["diverging"]
    acceptance_prob_est = 1.0 - jnp.mean(divergences)
    print("Estimated Acceptance Probability:", acceptance_prob_est)



    df_mcmc_time_per_N = pl.DataFrame({
        'number of systems': number_systems, 
        'run time': end_time - start_time}, 
        schema={'number of systems': pl.UInt16, 
                'run time': pl.Float64})
    
    df_mcmc_results_per_N = parse_mcmc_summary(mcmc, number_systems)
    return df_mcmc_results_per_N, df_mcmc_time_per_N

def run_full_mcmc(folder_name="mcmc_results_diagonal_mass_matrix"):
    for N in cs.N_VALUES:
        df_mcmc_results_per_N, df_mcmc_time_per_N = run_mcmc(N, folder_name)
        df_mcmc_results_per_N.write_parquet(f"simulation_results/{folder_name}/mcmc_results_N={N}.parquet")
        df_mcmc_time_per_N.write_parquet(f"simulation_results/{folder_name}/mcmc_times_N={N}.parquet")

run_full_mcmc()