from functools import partial
import jax.numpy as jnp
from jax import jit, random

@partial(jit, static_argnums=3) 
def sample_normal_distribution(key, mean, std, m=1):
   keys = random.split(key, len(mean))
   return jnp.array([
       random.normal(keys[i], shape=(m,)) * std + mean
       for i, (mean, std) in enumerate(zip(mean, std))
   ])

@partial(jit, static_argnums=3) 
def sample_inverse_gamma_distribution(key, a, b, m=1):
   keys = random.split(key, len(a)) 
   return jnp.array([
       b_i / random.gamma(keys[i], a=a_i, shape=(m,))
       for i, (a_i, b_i) in enumerate(zip(a, b))
   ])

@partial(jit, static_argnums=3) 
def sample_lognormal(key, mu, tau, m):
    """
    Sample from a lognormal distribution with desired mean mu and std tau.
    Uses jax.random.lognormal with appropriate sigma parameter.
    """
    # For X ~ LogNormal(0, σ), we have:
    # E[X] = exp(σ²/2)
    # Var(X) = [exp(σ²) - 1]exp(σ²)
    # To get our desired mean and variance, we need to:
    # 1. Find σ that gives us the right relative variance (tau/mu)²
    # 2. Then scale the results to get the right mean
    
    # Calculate sigma for the base distribution
    variance_ratio = (tau/mu)**2
    sigma = jnp.sqrt(jnp.log(1 + variance_ratio))
    
    # Calculate the scaling factor needed to achieve our desired mean
    # If Y ~ LogNormal(0, σ), then E[Y] = exp(σ²/2)
    # So we need to scale by mu/E[Y] = mu/exp(σ²/2)
    scale_factor = mu / jnp.exp(sigma**2/2)
    
    # Split key for each parameter pair
    keys = random.split(key, len(mu))
    
    # Sample and scale
    samples = jnp.array([
        scale_factor[i] * random.lognormal(k, sigma=sigma[i], shape=(m,))
        for i, k in enumerate(keys)
    ])
    
    return samples

@partial(jit, static_argnums=3) 
def sample_lognormal_distribution(key, mu, tau, m):
    """
    Sample from a lognormal distribution with parameters in log space.
    
    Args:
        key: JAX random key
        mu: location parameter (in log space)
        tau: scale parameter (in log space)
        m: number of samples
    """
    # Split key for each parameter
    keys = random.split(key, len(mu))
    
    # JAX's lognormal takes only sigma, so we need to transform normal samples
    samples = jnp.array([
        jnp.exp(mu[i] + tau[i] * random.normal(k, shape=(m,)))
        for i, k in enumerate(keys)
    ])
    
    return samples