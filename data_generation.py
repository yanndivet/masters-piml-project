from functools import partial
from jax import random, jit, vmap
import jax.numpy as jnp

from numerical_integrators_2params import leap_frog_harmonic_oscillator
import constants as cs

@partial(jit, static_argnums=2) 
def noisy_observations(z_params, key, sigma_noise):
    true_observations = leap_frog_harmonic_oscillator(params=z_params, number_observations=cs.N_SAMPLES)
    noise = sigma_noise * random.normal(key, cs.N_SAMPLES)
    return true_observations + noise

# @partial(jit, static_argnums=1) 
# def generate_observations(sampled_parameters, sigma_observations):
#     sampled_omegas, sampled_gammas = sampled_parameters
#     return jnp.array([
#         noisy_observations_from_parameters(n, parameters, sigma_observations) 
#         for n, parameters in enumerate(zip(sampled_omegas, sampled_gammas))])

@partial(jit, static_argnums=1)
def generate_observations(params, sigma):
    keys = random.split(random.PRNGKey(0), len(params[0]))
    return vmap(noisy_observations, in_axes=(0, 0, None))(
        jnp.array(list(zip(*params))), 
        keys, 
        sigma
    )