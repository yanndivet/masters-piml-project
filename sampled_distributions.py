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
    # Convert mu and tau to parameters of the underlying normal distribution
    sigma_norm_sq = jnp.log(1 + (tau ** 2) / (mu ** 2))
    sigma_norm = jnp.sqrt(sigma_norm_sq)
    mu_norm = jnp.log(mu) - 0.5 * sigma_norm_sq
    
    # Split key for each sample
    keys = random.split(key, len(mu))
    
    # Sample for each parameter set
    samples = jnp.array([
        jnp.exp(mu_n + sigma_n * random.normal(k, (m,)))
        for k, mu_n, sigma_n in zip(keys, mu_norm, sigma_norm)
    ])
    
    return samples

