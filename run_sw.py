import jax
import jax.numpy as jnp
from jax import random
import optax
from time import time
import polars as pl

import constants as cs
import data_generation as data_gen
from jx_pot import sliced_wasserstein_distance_CDiag

df_sw_time = pl.DataFrame(schema=[
    ('number of systems', pl.UInt16),
    ('sw run time', pl.Float64)
])

df_sw_results = pl.DataFrame(schema=[
    ('number of systems', pl.Int32),
    ('hyperparameter', pl.String),
    ('estimate', pl.Float64), 
])

def kl_multivariate_gaussians(parameter_estimate, initial_parameter=cs.INITIAL_HYPERPARAMETERS, std_error=cs.OBSERVATION_NOISE):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices
    https://gist.github.com/ChuaCheowHuan/18977a3e77c0655d945e8af60633e4df
    """
    mean_estimate, C_estimate = parameter_estimate, jnp.diag(std_error * jnp.ones(3))
    mean_init, C_init = initial_parameter, jnp.diag(std_error * jnp.ones(3))
    
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

def regularised_divergence(alpha, target_observations, number_systems):
    """
    Compute objective using Sliced-Wasserstein distance with transformed parameters and a regularisation KL-divergence term
    """
    try:
        iteration += 1
    except:
        iteration = 0
        
    key = random.key(iteration)
    params = sample_params(key, alpha)
    
    # Generate trajectories
    y_estimate = data_gen.noisy_observations_from_parameters(iteration, params, cs.OBSERVATION_NOISE)
    y_estimates = jnp.tile(y_estimate, (number_systems, 1))
    
    # Compute distance
    C = (cs.OBSERVATION_NOISE ** 2) * jnp.ones(cs.N_SAMPLES)
    sw_distance = sliced_wasserstein_distance_CDiag(random.key(iteration+1), y_estimates, target_observations, C)
    return sw_distance
    
    # Compute regularisation term
    kl_divergence = kl_multivariate_gaussians(params)
    
    return sw_distance + LAMBDA_REG * kl_divergence

def sample_params(sample_key, alpha):
    '''
    p(z^(n) | alpha) with parameter transformation
    '''
    m_alpha, C_alpha = alpha[:2], jnp.exp(alpha[2:])
    return random.multivariate_normal(
        sample_key,
        mean=m_alpha,
        cov=jnp.diag(C_alpha),
    )


def run_sw(number_systems=cs.N_SYSTEMS):
    solver = optax.adam(learning_rate=cs.LEARNING_RATE)
    alpha = cs.INITIAL_HYPERPARAMETERS
    opt_state = solver.init(alpha)
    losses = []

    all_sampled_params = data_gen.sample_parameters(number_systems)
    target_observations_from_sampled_params = data_gen.generate_observations(all_sampled_params, cs.OBSERVATION_NOISE)

    def loss_fn(parameters, target_observations=target_observations_from_sampled_params, N=number_systems):
        return regularised_divergence(parameters, target_observations, N)
    
    for iteration in range(1001):
        grad = jax.grad(loss_fn)(alpha)
        updates, opt_state = solver.update(grad, opt_state, alpha)
        alpha = optax.apply_updates(alpha, updates)
        losses.append(loss_fn(alpha))
        
        if iteration % 100 == 0:
            print(f'Loss function at step {iteration}: {loss_fn(alpha)}, with real parameters {alpha}')

    return alpha.tolist()

N_values = jnp.array([1, 5, 10, 50, 100, 500, 1000])
for N in N_values:
    print("Starting regularised divergence for N = ", N)
    start_time = time()
    final_parameters = run_sw(N)
    end_time = time()
    df_sw_time.extend(pl.DataFrame({
        'number of systems': N, 
        'sw run time': end_time - start_time}, 
        schema={'number of systems': pl.UInt16, 
                'sw run time': pl.Float64}))
    
    df_result_per_system = pl.DataFrame({
        'number of systems': [N] * len(cs.HYPERPARAMETER_NAMES),
        'hyperparameter': cs.HYPERPARAMETER_NAMES,
        'estimate': final_parameters}, 
        schema=[
            ('number of systems', pl.Int32),
            ('hyperparameter', pl.String),
            ('estimate', pl.Float64)]
    )

    df_sw_results.extend(df_result_per_system)
    
df_sw_results.write_parquet("sw_results_full.parquet")
df_sw_time.write_parquet("sw_times_full.parquet")
