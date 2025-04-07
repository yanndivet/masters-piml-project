from functools import partial
import jax.numpy as jnp
from jax import jit, random, vmap

@partial(jit, static_argnums=3) 
def sample_normal_distribution(key, mean, std, m=1):
   keys = random.split(key, len(mean))
   sample_single_normal_distribution = lambda k, mu_i, sigma_i: random.normal(k, shape=(m,)) * sigma_i + mu_i
   return jnp.array(vmap(sample_single_normal_distribution)(keys, mean, std))

@partial(jit, static_argnums=3) 
def sample_inverse_gamma_distribution(key, a, b, m=1):
    keys = random.split(key, len(a))
    sample_single_inverse_gamma_distribution = lambda k, a_i, b_i: b_i / random.gamma(k, a=a_i, shape=(m,))
    return jnp.array(vmap(sample_single_inverse_gamma_distribution)(keys, a, b))

# @partial(jit, static_argnums=3) 
# def sample_lognormal(key, mu, tau, m):
#     """
#     Sample from a lognormal distribution with desired mean mu and std tau.
#     Uses jax.random.lognormal with appropriate sigma parameter.
#     """
#     # For X ~ LogNormal(0, σ), we have:
#     # E[X] = exp(σ²/2)
#     # Var(X) = [exp(σ²) - 1]exp(σ²)
#     # To get our desired mean and variance, we need to:
#     # 1. Find σ that gives us the right relative variance (tau/mu)²
#     # 2. Then scale the results to get the right mean
    
#     # Calculate sigma for the base distribution
#     variance_ratio = (tau/mu)**2
#     sigma = jnp.sqrt(jnp.log(1 + variance_ratio))
    
#     # Calculate the scaling factor needed to achieve our desired mean
#     # If Y ~ LogNormal(0, σ), then E[Y] = exp(σ²/2)
#     # So we need to scale by mu/E[Y] = mu/exp(σ²/2)
#     scale_factor = mu / jnp.exp(sigma**2/2)
    
#     # Split key for each parameter pair
#     keys = random.split(key, len(mu))
    
#     # Sample and scale
#     samples = jnp.array([
#         scale_factor[i] * random.lognormal(k, sigma=sigma[i], shape=(m,))
#         for i, k in enumerate(keys)
#     ])
    
#     return samples

@partial(jit, static_argnums=3)
def sample_lognormal(key, mu, tau, m):
    """
    Sample from a lognormal distribution with desired mean mu and std tau in original space.
    Includes safety checks to prevent numerical instability.
    """
    # Add small epsilon to prevent division by zero
    epsilon = 1e-8
    mu_safe = jnp.maximum(mu, epsilon)
    
    # Clip variance ratio to reasonable values
    variance_ratio = jnp.clip((tau/mu_safe)**2, 0.0, 1e6)
    
    # Compute scale parameter (with safety checks)
    scale = jnp.sqrt(jnp.log1p(variance_ratio))  # Using log1p for numerical stability
    scale = jnp.clip(scale, 0.0, 10.0)  # Clip to reasonable range
    
    # Compute location parameter
    loc = jnp.log(mu_safe) - 0.5 * scale**2
    
    # Split key for each parameter
    keys = random.split(key, len(mu))
    
    # Sample using normal + exp approach
    samples = jnp.array([
        jnp.exp(loc[i] + scale[i] * random.normal(k, shape=(m,)))
        for i, k in enumerate(keys)
    ])
    
    # Final safety check for invalid values
    samples = jnp.where(jnp.isfinite(samples), samples, mu_safe.reshape(-1, 1))
    
    return samples

# @partial(jit, static_argnames=["m"])
# def sample_lognormal(key, mu, tau, m):
#     """
#     Sample from a lognormal distribution with desired mean `mu` and std `tau` in original space.
#     Uses numerically stable computations and efficient key splitting.
#     """
#     epsilon = 1e-8
#     mu_safe = jnp.maximum(mu, epsilon)  # Ensure mu is positive
    
#     variance_ratio = jnp.clip((tau / mu_safe) ** 2, 0.0, 1e3)  # Reasonable bound
    
#     scale = jnp.sqrt(jnp.log1p(variance_ratio))  # Log1p for stability
#     scale = jnp.clip(scale, 0.0, 10.0)  # Avoid extreme values
    
#     loc = jnp.log(mu_safe) - 0.5 * scale**2  # Corrected location parameter
    
#     keys = random.split(key, mu.shape[0])  # Efficient key splitting
    
#     samples = jnp.exp(loc[:, None] + scale[:, None] * random.normal(keys, shape=(mu.shape[0], m)))

#     return samples