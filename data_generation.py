from functools import partial
from jax import random, jit, vmap

from leapfrog_integrator import damped_harmonic_oscillator, lotka_volterra
import constants as cs

@partial(jit, static_argnums=2) 
def noisy_observations(z_params, key, sigma_noise):
    if cs.PROBLEM == "LV":
        true_observations = lotka_volterra(params=z_params, number_observations=cs.OBSERVATION_LENGTH)
    else:
        true_observations = damped_harmonic_oscillator(params=z_params, number_observations=cs.OBSERVATION_LENGTH)
    noise = sigma_noise * random.normal(key, cs.OBSERVATION_LENGTH)
    return true_observations + noise

@partial(jit, static_argnums=2)
def generate_observations(params, key, sigma):
    keys = random.split(key, params[0].shape)
    return vmap(noisy_observations, in_axes=(0, 0, None))(
        params.T,     
        keys, 
        sigma
    )