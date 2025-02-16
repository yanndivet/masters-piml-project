import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
os.environ['JAX_PLATFORMS'] = 'cpu'

from functools import partial 
import jax
import jax.numpy as jnp
from jax import random, jit
import optax
from time import time
import polars as pl

import constants as cs
import data_generation as data_gen
from jx_pot import sliced_wasserstein_distance_CDiag
import sampled_distributions as sd

@jit
def kl_multivariate_gaussians(parameter_estimate, initial_parameter=cs.INITIAL_HYPERPARAMETERS):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices
    https://gist.github.com/ChuaCheowHuan/18977a3e77c0655d945e8af60633e4df
    """
    mean_estimate, C_estimate = parameter_estimate[:2], jnp.diag(parameter_estimate[2:])
    mean_init, C_init = initial_parameter[:2], jnp.diag(initial_parameter[2:])
    
    d = mean_estimate - mean_init
    
    c, lower = jax.scipy.linalg.cho_factor(C_init)
    def solve(B):
        return jax.scipy.linalg.cho_solve((c, lower), B)
    
    def logdet(S):
        return jnp.linalg.slogdet(S)[1]

    term1 = jnp.trace(solve(C_estimate))
    term2 = logdet(C_init) - logdet(C_estimate)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2

@partial(jit, static_argnums=3) 
def regularised_divergence(hyperparameters, key, target_observations, number_systems):
    """
    Compute objective using Sliced-Wasserstein distance with transformed parameters and a regularisation KL-divergence term
    """
    key, key_sample_params, key_obs_gen, key_sw = random.split(key, 4)
    mu_hyperparams, tau_hyperparams = hyperparameters[:2], hyperparameters[2:]
    sampled_parameters = sd.sample_lognormal(key_sample_params, mu=mu_hyperparams, tau=tau_hyperparams, m=number_systems)
    
    # Generate trajectories
    estimated_observations = data_gen.generate_observations(sampled_parameters, key_obs_gen, cs.OBSERVATION_NOISE)
    
    # Compute distance
    C = (cs.OBSERVATION_NOISE ** 2) * jnp.ones(cs.N_SAMPLES)
    # C = cs.OBSERVATION_NOISE * jnp.ones(cs.N_SAMPLES)
    sw_distance = sliced_wasserstein_distance_CDiag(key_sw, estimated_observations, target_observations, C)
    
    # Compute regularisation term
    kl_divergence = kl_multivariate_gaussians(hyperparameters)
    
    return sw_distance + kl_divergence

def run_sw(number_systems=cs.N_SYSTEMS):
    solver = optax.adam(learning_rate=cs.LEARNING_RATE)
    alpha = cs.INITIAL_HYPERPARAMETERS
    opt_state = solver.init(alpha)
    losses = []

    key_true_observations = random.key(number_systems)
    key_sample_params_from_true_values, key_true_obs_gen, key_regularised_divergence = random.split(key_true_observations, 3)
    sampled_params = sd.sample_lognormal(key_sample_params_from_true_values, mu=cs.MU_TARGET, tau=cs.TAU_TARGET, m=number_systems)
    target_observations_from_sampled_params = data_gen.generate_observations(sampled_params, key_true_obs_gen, cs.OBSERVATION_NOISE)

    def loss_fn(parameters):
        return regularised_divergence(parameters, key_regularised_divergence, target_observations_from_sampled_params, number_systems)
    
    start_time = time()
    for iteration in range(4001):
        grad = jax.grad(loss_fn)(alpha)
        updates, opt_state = solver.update(grad, opt_state, alpha)
        alpha = optax.apply_updates(alpha, updates)
        losses.append(loss_fn(alpha))

        if len(losses) > 2:
            if abs(losses[-1] - losses[-2]) < cs.EPSILON:
                print(f'Loss function at step {iteration}: {loss_fn(alpha)}, with real parameters {alpha}')
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

    return df_sw_result_per_system, df_sw_time_per_N,

for N in cs.N_VALUES:
    df_sw_results_per_N, df_sw_time_per_N = run_sw(N)
    df_sw_results_per_N.write_parquet(f"simulation_results/sw_results/sw_results_N={N}.parquet")
    df_sw_time_per_N.write_parquet(f"simulation_results/sw_results/sw_times_N={N}.parquet")