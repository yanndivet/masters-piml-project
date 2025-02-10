import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
os.environ['JAX_PLATFORMS'] = 'cpu'


from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.infer import MCMC, NUTS
numpyro.set_platform("cpu")

from time import time
import polars as pl

import io
from contextlib import redirect_stdout

import constants as cs
import distributions as dist
import data_generation as data_gen
import sampled_distributions as sd

import numpy as np

def plot_mcmc_chains(samples, number_systems):
    import matplotlib.pyplot as plt
    
    # Reshape samples to (n_samples, n_params)
    samples_reshaped = samples.squeeze()  # This removes the singleton dimension, making it (2000, 14)
    
    # Create subplots for first few parameters
    n_params_to_plot = min(4, samples_reshaped.shape[1])  # Let's plot first 4 parameters
    fig, axes = plt.subplots(n_params_to_plot, 1, figsize=(12, 3*n_params_to_plot))
    fig.suptitle(f'MCMC Chain Traces for N = {number_systems}')
    
    # Plot each parameter
    for i in range(n_params_to_plot):
        param_samples = samples_reshaped[:, i]
        x = np.arange(len(param_samples))
        
        axes[i].plot(x, param_samples, 'b-', linewidth=0.5)
        axes[i].axhline(y=cs.TARGET_HYPERPARAMETERS[i], color='r', linestyle='--', alpha=0.8, label='True Value')
        param_name = "Hyperparameter" if i < 4 else "Population parameter"
        axes[i].set_title(f'{param_name} {i}')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'simulation_results/new_true_values/mcmc_chains_N={number_systems}.png', dpi=300, bbox_inches='tight')
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

def run_mcmc(number_systems=cs.N_SYSTEMS):
    key = random.key(101)
    key_sample_params, key_mcmc = random.split(key, 2)
    sampled_params = sd.sample_lognormal(key_sample_params, mu=cs.MU_TARGET, tau=cs.TAU_TARGET, m=number_systems)
    sampled_params_concat = jnp.concatenate(sampled_params)
    target_observations_from_sampled_params = data_gen.generate_observations(sampled_params, cs.OBSERVATION_NOISE)

    initial_parameters = jnp.concatenate([cs.INITIAL_HYPERPARAMETERS, sampled_params_concat])
    initial_parameters_tiled = jnp.tile(initial_parameters, (cs.N_CHAINS, 1))
    noisy_initial_parameters_tiled = initial_parameters_tiled # + random.normal(random.key(42), initial_parameters_tiled.shape) * 1e-4

    target_distribution = lambda params: dist.log_posterior_distribution(
        params, 
        target_observations_from_sampled_params, 
        number_systems
    )

    # kernel = NUTS(potential_fn=target_distribution,
    #          step_size=1e-3,  # Start larger, let it adapt
    #          adapt_step_size=True,  # Enable step size adaptation
    #          target_accept_prob=0.8,
    #          adapt_mass_matrix=True,
    #          dense_mass=True)
    kernel = NUTS(potential_fn=target_distribution,
             step_size=1e-4,  # Try smaller step size
             adapt_step_size=True,
             target_accept_prob=0.65,  # Try lower target acceptance
             max_tree_depth=8,  # Add this to control tree depth
             adapt_mass_matrix=True,
             dense_mass=True)
    mcmc = MCMC(kernel, 
                num_warmup=5000,
                num_samples=2000,
                num_chains=cs.N_CHAINS,
                chain_method='parallel')
    rng_keys = random.split(key_mcmc, cs.N_CHAINS)

    start_time = time()
    mcmc.run(rng_keys, init_params=noisy_initial_parameters_tiled)
    end_time = time()

    samples = mcmc.get_samples()
    plot_mcmc_chains(samples, number_systems)

    df_mcmc_time_per_N = pl.DataFrame({
        'number of systems': number_systems, 
        'run time': end_time - start_time}, 
        schema={'number of systems': pl.UInt16, 
                'run time': pl.Float64})
    
    df_mcmc_results_per_N = parse_mcmc_summary(mcmc, number_systems)
    return df_mcmc_results_per_N, df_mcmc_time_per_N

N_values = [1, 5, 10, 50, 100, 500, 1000]
for N in N_values:
    df_mcmc_results_per_N, df_mcmc_time_per_N = run_mcmc(N)
    df_mcmc_results_per_N.write_parquet(f"simulation_results/new_true_values/mcmc_results_N={N}.parquet")
    df_mcmc_time_per_N.write_parquet(f"simulation_results/new_true_values/mcmc_times_N={N}.parquet")