from functools import partial
from jax import jit, lax
import constants as cs

@partial(jit, static_argnums=(1,2,3))
def leap_frog_harmonic_oscillator(params, T=cs.T, number_observations=cs.N_SAMPLES, initial_state=(cs.U_0, cs.V_0)):
    """
    Leap-frog integration for a harmonic oscillator using JAX and `jax.scan`.

    Args:
        params: Tuple of (m, c, k) where:
            m: Mass
            c: Damping coefficient
            k: Spring constant
        T: Total simulation time
        N: Number of time steps
        u_0: Initial position
        v_0: Initial velocity
    
    Returns:
        Tuple of (positions, velocities) as JAX arrays.
    """
    # Unpack parameters
    omega_0, gamma = params[0], params[1] # add error if more than 2 params!
    dt = T / number_observations

    # Define the scan update function

    def next_state(state, _):
            u_prev, v_prev = state
            u_next = u_prev + dt * v_prev
            v_next = v_prev - dt * (gamma * v_prev + omega_0**2 * u_next)
            return (u_next, v_next), u_next

    _, displacements = lax.scan(f=next_state, init=initial_state, length=number_observations)
    return displacements