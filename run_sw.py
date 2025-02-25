import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

from functools import partial 
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
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

@partial(jit, static_argnums=3)
def regularised_divergence(hyperparameters, key, target_observations, number_systems):
    """Optimized regularized divergence computation for GPU"""
    keys = random.split(key, 4)
    mu_hyperparams, tau_hyperparams = hyperparameters[:2], hyperparameters[2:]
    
    # Batch sample parameters
    sampled_parameters = sd.sample_lognormal_batched(
        keys[1], 
        mu=mu_hyperparams, 
        tau=tau_hyperparams, 
        batch_size=number_systems
    )
    
    # Generate observations in parallel
    estimated_observations = vmap(data_gen.generate_observations)(
        sampled_parameters,
        random.split(keys[2], number_systems),
        jnp.full(number_systems, cs.OBSERVATION_NOISE)
    )
    
    # Compute SW distance
    C = (cs.OBSERVATION_NOISE ** 2) * jnp.ones(cs.OBSERVATION_LENGTH)
    sw_distance = sliced_wasserstein_distance_CDiag(
        keys[3],
        estimated_observations,
        target_observations,
        C
    )
    
    # Vectorized KL divergence computation
    kl_divergence = jnp.sum(
        kl_lognormal_distributions(
            mu_hyperparams,
            tau_hyperparams,
            cs.MU_PHI,
            cs.TAU_PHI_MAP
        )
    )
    
    return sw_distance + kl_divergence

def run_sw(number_systems=cs.N_SYSTEMS):
    solver = optax.adam(learning_rate=cs.LEARNING_RATE)
    alpha = cs.INITIAL_HYPERPARAMETERS
    opt_state = solver.init(alpha)
    losses = []

    key_regularised_divergence = random.key(number_systems)

    def loss_fn(parameters):
        return regularised_divergence(parameters, key_regularised_divergence, to.read_true_observations(number_systems), number_systems)
    
    start_time = time()
    for iteration in range(4001):
        grad = jax.grad(loss_fn)(alpha)
        updates, opt_state = solver.update(grad, opt_state, alpha)
        alpha = optax.apply_updates(alpha, updates)
        losses.append(loss_fn(alpha))

        if len(losses) > 2:
            if abs(losses[-1] - losses[-2]) < cs.EPSILON:
                # print(f'Loss function at step {iteration}: {loss_fn(alpha)}, with real parameters {alpha}')
                break 
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

for N in cs.N_VALUES:
    print("Running for N = ", N)
    df_sw_results_per_N, df_sw_time_per_N = run_sw(N)
    df_sw_results_per_N.write_parquet(f"simulation_results/sw_results_gpu/sw_results_N={N}.parquet")
    df_sw_time_per_N.write_parquet(f"simulation_results/sw_results_gpu/sw_times_N={N}.parquet")