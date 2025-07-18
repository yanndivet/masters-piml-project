from functools import partial
import jax.numpy as jnp
from jax import jit, scipy, vmap, random
from jax.scipy.stats import norm

import data_generation
import constants as cs

# Base distributions
@jit
def negative_log_normal_pdf(x: jnp.array, mean: jnp.array, std) -> float:
    return -jnp.sum(norm.logpdf(x, mean, std))

@jit
def negative_log_invgamma_pdf(x: float, a: float, b: float) -> float:
    log_density = -(a + 1) * jnp.log(x) - b/x - jnp.log(b**(-a) / jnp.exp(scipy.special.gammaln(a)))
    return -jnp.sum(log_density)

# Hyperprior distribution
@jit
def log_hyperprior_distribution(mu, tau, mu_phi=cs.MU_PHI, sigma_phi=cs.SIGMA_PHI, a_phi=cs.A_PHI, b_phi=cs.B_PHI) -> float:
    log_hyperprior_mu = negative_log_normal_pdf(mu, mu_phi, sigma_phi)
    log_hyperprior_tau = negative_log_invgamma_pdf(tau, a_phi, b_phi)
    return log_hyperprior_mu + log_hyperprior_tau

# Population distribution
@jit 
def log_population_distribution_n_systems(z, mu, tau) -> float:
    # Vectorize by broadcasting mu
    mu_broadcast = jnp.broadcast_to(mu, z.shape)  # Shape: (N_SYSTEMS, 2)
    return -jnp.sum(norm.logpdf(z, mu_broadcast, tau))

# Likelihood distribution
@jit
def log_likelihood_distribution_n_systems(z, y) -> float:
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
# @partial(jit, static_argnums=2) 
# def log_posterior_distribution(all_parameters, y, number_systems):
#     mu = all_parameters[:len(cs.HYPERPARAMETERS)]
#     tau = all_parameters[len(cs.HYPERPARAMETERS):len(cs.TARGET_HYPERPARAMETERS)]
#     z = all_parameters[len(cs.TARGET_HYPERPARAMETERS):].reshape(number_systems, len(cs.HYPERPARAMETERS))
    
#     log_hyperprior = log_hyperprior_distribution(mu, tau)
#     log_population = log_population_distribution_n_systems(z, mu, tau)
#     log_likelihood = log_likelihood_distribution_n_systems(z, y)
    
#     return log_hyperprior + log_population + log_likelihood

@partial(jit, static_argnums=2)
def log_posterior_distribution(all_parameters, y, number_systems):
    d = len(cs.HYPERPARAMETERS)
    # 1) hyperparameters
    mu  = all_parameters[0    : d]
    tau = all_parameters[d    : 2*d]
    # 2) raw latents
    z_raw = all_parameters[2*d:].reshape(number_systems, d)
    # 3) map back
    z = mu + tau * z_raw

    # 4) hyperprior (negative log)
    log_hyperprior = log_hyperprior_distribution(mu, tau)

    # 5) non-centered population prior (negative log):
    #    -sum log Normal(z_raw;0,1)  +  N * sum log(tau)
    log_population = (
        - jnp.sum(norm.logpdf(z_raw, 0.0, 1.0))
        + number_systems * jnp.sum(jnp.log(tau))
    )

    # 6) likelihood (still negative log)
    log_likelihood = log_likelihood_distribution_n_systems(z, y)

    return log_hyperprior + log_population + log_likelihood