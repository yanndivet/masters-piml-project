from functools import partial
from jax import random, jit, vmap
import jax.numpy as jnp

from numerical_integrators_2params import leap_frog_harmonic_oscillator
import constants as cs

@partial(jit, static_argnums=2) 
def noisy_observations(z_params, key, sigma_noise):
    true_observations = leap_frog_harmonic_oscillator(params=z_params, number_observations=cs.OBSERVATION_LENGTH)
    noise = sigma_noise * random.normal(key, cs.OBSERVATION_LENGTH)
    return true_observations + noise

@partial(jit, static_argnums=2)
def generate_observations(params, key, sigma):
    keys = random.split(key, len(params[0]))
    return vmap(noisy_observations, in_axes=(0, 0, None))(
        jnp.array(list(zip(*params))), 
        keys, 
        sigma
    )