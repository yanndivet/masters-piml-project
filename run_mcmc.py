from jax import random, jit
import jax.numpy as jnp

from numpyro import set_platform
set_platform('cpu')
from numpyro.infer import MCMC, NUTS

from time import time
import polars as pl

import io
from contextlib import redirect_stdout

import constants as cs
import distributions as dist
import true_observations as to

# def plot_mcmc_chains(posterior_samples, warmup_samples, number_systems, experiment_number):
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Reshape samples to (n_samples, n_params)
#     posterior_reshaped = posterior_samples.squeeze()
#     warmup_reshaped = warmup_samples.squeeze()
    
#     n_params_to_plot = posterior_reshaped.shape[1]
#     fig, axes = plt.subplots(n_params_to_plot, 1, figsize=(12, 3*n_params_to_plot))
    
#     # Use LaTeX for the title
#     fig.suptitle(f'MCMC Chain Traces (including warmup) for $N = {number_systems}$')
    
#     # Plot each parameter
#     for i in range(n_params_to_plot):
#         # Get samples for current parameter
#         warmup_param = warmup_reshaped[:, i]
#         posterior_param = posterior_reshaped[:, i] 
        
#         # Create x-axes for both phases
#         x_warmup = np.arange(len(warmup_param))
#         x_posterior = np.arange(len(warmup_param), len(warmup_param) + len(posterior_param))
        
#         # Plot warmup phase
#         axes[i].plot(x_warmup, warmup_param, 'gray', alpha=0.5, linewidth=0.5, label='Warmup')
        
#         # Plot sampling phase
#         axes[i].plot(x_posterior, posterior_param, 'b-', linewidth=0.5, label='Sampling')
        
#         # Add true value line
#         axes[i].axhline(y=cs.TARGET_HYPERPARAMETERS[i], color='r', linestyle='--', 
#                        alpha=0.8, label='True Value')
        
#         # Add vertical line separating warmup and sampling phases
#         axes[i].axvline(x=len(warmup_param), color='k', linestyle=':', alpha=0.5,
#                        label='End of Warmup')
        
#         # Customize plot with LaTeX parameter names
#         axes[i].set_title(cs.HYPERPARAMETER_NAMES[i])
#         axes[i].set_xlabel('Iteration')
#         axes[i].set_ylabel('Value')
#         axes[i].grid(True, linestyle='--', alpha=0.7)
        
#         # Add legend (only for the first subplot to avoid redundancy)
#         if i == 0:
#             axes[i].legend(loc='upper right', fontsize='small')
    
#     # Enable LaTeX rendering
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    
#     plt.tight_layout()
#     plt.savefig(f'{cs.PROBLEM}/MCMC/Chains/experiment={experiment_number}_N={number_systems}.pdf', 
#                 dpi=300, bbox_inches='tight')
#     plt.close()

def plot_mcmc_chains(posterior_samples, warmup_samples, number_systems, experiment_number, 
                     param_indices=None, param_type="", filename_suffix=""):
    """
    Plot MCMC chains for selected parameters
    
    Args:
        posterior_samples: Posterior samples from MCMC
        warmup_samples: Warmup samples from MCMC
        number_systems: Number of systems in the model
        experiment_number: Experiment number
        param_indices: Indices of parameters to plot (if None, plots first 4)
        param_type: Type of parameters being plotted (e.g., "mu" or "tau")
        filename_suffix: Additional text to add to the filename
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Reshape samples to (n_samples, n_params)
    posterior_reshaped = posterior_samples.squeeze()
    warmup_reshaped = warmup_samples.squeeze()
    
    # Get parameter indices to plot
    if param_indices is None:
        n_params_to_plot = min(4, posterior_reshaped.shape[1])
        param_indices = list(range(n_params_to_plot))
    else:
        n_params_to_plot = len(param_indices)
    
    # Create subplots for selected parameters
    fig, axes = plt.subplots(n_params_to_plot, 1, figsize=(12, 3*n_params_to_plot))
    
    # Handle case when n_params_to_plot is 1
    if n_params_to_plot == 1:
        axes = [axes]
    
    # Use LaTeX for the title
    title = f'MCMC Chain Traces (including warmup) for $N = {number_systems}$'
    if param_type:
        title += f' - {param_type} parameters'
    fig.suptitle(title)
    
    # Plot each parameter
    for i, param_idx in enumerate(param_indices):
        # Get samples for current parameter
        warmup_param = warmup_reshaped[:, param_idx]
        posterior_param = posterior_reshaped[:, param_idx] 
        
        # Create x-axes for both phases
        x_warmup = np.arange(len(warmup_param))
        x_posterior = np.arange(len(warmup_param), len(warmup_param) + len(posterior_param))
        
        # Plot warmup phase
        axes[i].plot(x_warmup, warmup_param, 'gray', alpha=0.5, linewidth=0.5, label='Warmup')
        
        # Plot sampling phase
        axes[i].plot(x_posterior, posterior_param, 'b-', linewidth=0.5, label='Sampling')
        
        # Add true value line if available
        if param_idx < len(cs.TARGET_HYPERPARAMETERS):
            axes[i].axhline(y=cs.TARGET_HYPERPARAMETERS[param_idx], color='r', linestyle='--', 
                          alpha=0.8, label='True Value')
        
        # Add vertical line separating warmup and sampling phases
        axes[i].axvline(x=len(warmup_param), color='k', linestyle=':', alpha=0.5,
                       label='End of Warmup')
        
        # Customize plot with LaTeX parameter names
        param_name = cs.HYPERPARAMETER_NAMES[param_idx] if param_idx < len(cs.HYPERPARAMETER_NAMES) else f"Parameter {param_idx}"
        axes[i].set_title(param_name)
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
    file_path = f'{cs.PROBLEM}/MCMC/Chains/experiment={experiment_number}_N={number_systems}{filename_suffix}.pdf'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path

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

def run_mcmc(number_systems, experiment_number):
    unique_seed = experiment_number * 1000 + number_systems
    key = random.key(unique_seed)
    key_mcmc_warmup, key_mcmc_posterior = random.split(key, 2)
    
    # Create initial parameters correctly shaped for multiple chains
    population_initial_parameters = jnp.tile(cs.MU_INITIAL, number_systems)
    initial_parameters = jnp.concatenate([cs.INITIAL_HYPERPARAMETERS, population_initial_parameters])
    
    # Make sure initial parameters are properly shaped for multiple chains
    initial_parameters_tiled = jnp.tile(initial_parameters, (cs.N_CHAINS, 1))

    true_observations = to.read_true_observations(number_systems, experiment_number, False)

    target_distribution = jit(lambda params: dist.log_posterior_distribution(
        params, 
        true_observations,
        number_systems
    ))

    kernel = NUTS(potential_fn=target_distribution,
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
    
    # Generate proper number of keys for each chain
    warmup_keys = random.split(key_mcmc_warmup, cs.N_CHAINS)
    posterior_keys = random.split(key_mcmc_posterior, cs.N_CHAINS)

    start_time = time()
    
    # Make sure initial parameters match the number of chains
    mcmc.warmup(warmup_keys, 
                collect_warmup=True, 
                init_params=initial_parameters_tiled)
    warmup_samples = mcmc.get_samples()
    
    mcmc.run(posterior_keys, 
             init_params=initial_parameters_tiled)
    end_time = time()
    
    posterior_samples = mcmc.get_samples()

    # plot_mcmc_chains(posterior_samples, warmup_samples, number_systems, experiment_number)

    # Plot mu parameters (assuming first 4 are mu terms)
    mu_indices = list(range(4))
    plot_mcmc_chains(posterior_samples, warmup_samples, number_systems, experiment_number,
                    param_indices=mu_indices, param_type="mu", filename_suffix="_mu")
    
    # Plot tau parameters (assuming next 4 are tau terms)
    tau_indices = list(range(4, 8))
    plot_mcmc_chains(posterior_samples, warmup_samples, number_systems, experiment_number,
                    param_indices=tau_indices, param_type="tau", filename_suffix="_tau")

    # divergences = mcmc.get_extra_fields()["diverging"]
    # acceptance_prob_est = 1.0 - jnp.mean(divergences)
    # print("Estimated Acceptance Probability:", acceptance_prob_est)

    df_mcmc_time_per_N = pl.DataFrame({
        'number of systems': number_systems, 
        'run time': end_time - start_time},
        schema={'number of systems': pl.UInt16, 
                'run time': pl.Float64})
    
    df_mcmc_results_per_N = parse_mcmc_summary(mcmc, number_systems)
    return df_mcmc_results_per_N, df_mcmc_time_per_N

def run_full_mcmc():
    for experiment_number in range(1, cs.NUMBER_EXPERIMENTS + 1):
        for N in cs.N_VALUES:
            print(f"Running MCMC for experiment = {experiment_number}, N = {N}")
            df_mcmc_results_per_N, df_mcmc_time_per_N = run_mcmc(N, experiment_number)
            df_mcmc_results_per_N.write_parquet(f"{cs.PROBLEM}/MCMC/Results/experiment={experiment_number}_N={N}.parquet")
            df_mcmc_time_per_N.write_parquet(f"{cs.PROBLEM}/MCMC/Runtimes/experiment={experiment_number}_N={N}.parquet")

run_full_mcmc()