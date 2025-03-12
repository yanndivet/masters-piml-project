import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

from functools import partial 
import jax
import jax.numpy as jnp
from jax import random, jit, debug
import optax
from time import time
import polars as pl

import constants as cs
import data_generation as data_gen
from jx_pot import sliced_wasserstein_distance_CDiag
import sampled_distributions as sd
import true_observations as to

@jit
def kl_lognormal_distributions(estimated_mean, estimated_std, initial_mean, initial_std):
    """
    Source: 
    https://stats.stackexchange.com/questions/289323/kl-divergence-of-multivariate-lognormal-distributions
    """
    return 1 / (2 * initial_std**2) * ((estimated_mean - initial_mean)**2 + estimated_std**2 - initial_std**2) + jnp.log(initial_std / estimated_std)

@jit
def quadratic_regularisation(current_parameters, initial_parameters, c):
    return .5 * jnp.sum(((current_parameters - initial_parameters) / c) ** 2)

@partial(jit, static_argnums=3)
def regularised_divergence(hyperparameters, key, target_observations, number_systems):
    """Optimized regularized divergence computation for GPU"""
    keys = random.split(key, 4)
    mu_hyperparams, tau_hyperparams = hyperparameters[:2], hyperparameters[2:]
    
    sampled_parameters = sd.sample_lognormal(
        keys[1], 
        mu=mu_hyperparams, 
        tau=tau_hyperparams,
        m=number_systems
    )
    
    estimated_observations = data_gen.generate_observations(sampled_parameters, keys[2], cs.OBSERVATION_NOISE)
    
    sw_distance = sliced_wasserstein_distance_CDiag(
        keys[3],
        estimated_observations,
        target_observations,
        C=cs.OBSERVATION_NOISE**2 * jnp.ones(cs.OBSERVATION_LENGTH)
    )

    # return sw_distance
    
    # kl_divergence = jnp.sum(
    #     kl_lognormal_distributions(
    #         mu_hyperparams,
    #         tau_hyperparams,
    #         cs.MU_PHI,
    #         cs.TAU_PHI_MAP
    #     )
    # )

    # return sw_distance + kl_divergence

    regulariser_term = quadratic_regularisation(mu_hyperparams, cs.MU_PHI, 5) + quadratic_regularisation(tau_hyperparams, cs.TAU_PHI_MAP, 2)
    debug.print('Sliced-Wasserstein: {}', sw_distance)
    debug.print('Regulariser: {}', regulariser_term)
    
    return sw_distance + regulariser_term

def run_sw(number_systems=cs.N_SYSTEMS, experiment_number=1):
    global SW_DISTANCES, KL_DIVS
    SW_DISTANCES = []
    KL_DIVS = []  
    unique_seed = experiment_number * 1000 + number_systems
    key_sw = random.key(unique_seed)
    key_init_params, key_regularised_divergence = random.split(key_sw, 2)

    solver = optax.adam(learning_rate=cs.LEARNING_RATE)
    alpha = cs.INITIAL_HYPERPARAMETERS + 1e-2 * random.normal(key_init_params, 4)
    opt_state = solver.init(alpha)
    losses = []

    def loss_fn(parameters):
        return regularised_divergence(parameters, key_regularised_divergence, to.read_true_observations(number_systems, experiment_number), number_systems)
    
    start_time = time()
    for i in range(500):
        grad = jax.grad(loss_fn)(alpha)
        updates, opt_state = solver.update(grad, opt_state, alpha)
        alpha = optax.apply_updates(alpha, updates)
        losses.append(loss_fn(alpha))
    end_time = time()

    df_sw_time_per_N = pl.DataFrame({
        'number of systems': number_systems, 
        'sw run time': end_time - start_time}, 
        schema={'number of systems': pl.UInt16, 
                'sw run time': pl.Float64})
    
    df_sw_result_per_system = pl.DataFrame({
        'number of systems': [number_systems] * len(cs.HYPERPARAMETER_NAMES),
        'hyperparameter': cs.HYPERPARAMETER_NAMES,
        'sw_estimate': alpha.tolist()}, 
        schema=[
            ('number of systems', pl.Int32),
            ('hyperparameter', pl.String),
            ('sw_estimate', pl.Float64)]
    )
    
    return df_sw_result_per_system, df_sw_time_per_N

def run_full_sw(folder_name="sw_results_new_regulariser"):
    for experiment_number in range(1, 11):
        for N in cs.N_VALUES:
            print(f"Running SW method for experiment = {experiment_number}, N = {N}")
            df_mcmc_results_per_N, df_mcmc_time_per_N = run_sw(N, experiment_number)
            df_mcmc_results_per_N.write_parquet(f"simulation_results/{folder_name}/results_experiment={experiment_number}_N={N}.parquet")
            df_mcmc_time_per_N.write_parquet(f"simulation_results/{folder_name}/times_experiment={experiment_number}_N={N}.parquet")

# run_full_sw()

run_sw(1, 1)
run_sw(1000, 1)