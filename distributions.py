from functools import partial
import jax.numpy as jnp
from jax import jit, scipy, vmap, random
from jax.scipy.stats import norm

import data_generation
import constants as cs

# Base distributions
@jit
def log_normal_multi_pdf(x: jnp.array, mean: jnp.array, std) -> float:
    return -jnp.sum(norm.logpdf(x, mean, std))

@jit
def log_invgamma_pdf(x: float, a: float, b: float) -> float:
    log_density = -(a + 1) * jnp.log(x) - b/x - jnp.log(b**(-a) / jnp.exp(scipy.special.gammaln(a)))
    return -jnp.sum(log_density)

# Hyperprior distribution
@jit
def log_hyperprior_distribution(mu, tau, mu_phi=cs.MU_PHI, sigma_phi=cs.SIGMA_PHI, a_phi=cs.A_PHI, b_phi=cs.B_PHI) -> float:
    log_hyperprior_mu = log_normal_multi_pdf(mu, mu_phi, sigma_phi)
    log_hyperprior_tau = log_invgamma_pdf(tau, a_phi, b_phi)
    return log_hyperprior_mu + log_hyperprior_tau

# Population distribution
@jit 
def log_population_distribution_n_systems(z, mu, tau):
    # z shape: (N_SYSTEMS, 2)
    # mu shape: (2,)
    # Vectorize by broadcasting mu
    mu_broadcast = jnp.broadcast_to(mu, z.shape)  # Shape: (N_SYSTEMS, 2)
    return -jnp.sum(norm.logpdf(z, mu_broadcast, tau))

# Likelihood distribution
@jit
def log_likelihood_distribution_n_systems(z, y):
    # Generate keys for all systems
    keys = random.split(random.PRNGKey(0), z.shape[0])
    
    # Vectorize observation generation across all systems
    batch_observations = vmap(data_generation.noisy_observations, in_axes=(0, 0, None))(
        z,
        keys,
        0.0
    )
    
    return -jnp.sum(norm.logpdf(y.reshape(-1), 
                               batch_observations.reshape(-1), 
                               cs.OBSERVATION_NOISE))

# Final distribution
@partial(jit, static_argnums=2) 
def log_posterior_distribution(all_parameters, y, number_systems):
    mu = all_parameters[:2]
    tau = all_parameters[2:4]
    z = all_parameters[4:].reshape(number_systems, 2)
    
    log_hyperprior = log_hyperprior_distribution(mu, tau)
    log_population = log_population_distribution_n_systems(z, mu, tau)
    log_likelihood = log_likelihood_distribution_n_systems(z, y)
    
    return log_hyperprior + log_population + log_likelihood

# @partial(jit, static_argnums=2) 
# def log_posterior_distribution(all_parameters, y, number_systems):
#     mu = all_parameters[:2]
#     tau = all_parameters[2:4]
#     z_raw = all_parameters[4:].reshape(number_systems, 2)
    
#     # Non-centered parameterization for lognormal
#     z = jnp.exp(mu + tau * z_raw)  # z_raw ~ N(0,1) 
    
#     log_hyperprior = log_hyperprior_distribution(mu, tau)
#     log_population = -0.5 * jnp.sum(z_raw**2)  # Standard normal prior for z_raw
#     log_likelihood = log_likelihood_distribution_n_systems(z, y)
    
#     return log_hyperprior + log_population + log_likelihood