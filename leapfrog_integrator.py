from functools import partial
from jax import jit, lax
import constants as cs

@partial(jit, static_argnums=(1,2,3))
def damped_harmonic_oscillator(params, T=cs.T, number_observations=cs.OBSERVATION_LENGTH, initial_state=(cs.U_0, cs.V_0)):
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

    def next_state(state, _):
        u_prev, v_prev = state
        u_next = u_prev + dt * v_prev
        v_next = v_prev - dt * (gamma * v_prev + omega_0**2 * u_next)
        return (u_next, v_next), u_next

    _, displacements = lax.scan(f=next_state, init=initial_state, length=number_observations)
    return displacements

@partial(jit, static_argnums=(1,2,3))
def lotka_volterra(params, T=cs.T, number_observations=cs.OBSERVATION_LENGTH, initial_state=(cs.X_0, cs.Y_0)):
    """
    Leap-frog integration for the Lotka-Volterra predator-prey model using JAX.

    Args:
        params: Tuple of (alpha, beta, gamma, delta) where:
            alpha: Prey growth rate
            beta: Prey death rate due to predation
            gamma: Predator death rate
            delta: Predator growth rate due to predation
        T: Total simulation time
        number_observations: Number of time steps to record
        initial_state: Tuple of (x_0, y_0) with initial populations
    
    Returns:
        JAX array of populations over time, containing both x and y values.
    """
    
    # Unpack parameters
    alpha, beta, gamma, delta = params
    dt = T / number_observations

    def next_state(state, _):
        x_prev, y_prev = state
        
        # Simple Euler method, matching your original implementation
        x_next = x_prev + dt * (alpha * x_prev - beta * x_prev * y_prev)
        y_next = y_prev + dt * (gamma * x_prev * y_prev - delta * y_prev)
        
        return (x_next, y_next), (x_next, y_next)

    _, populations = lax.scan(f=next_state, init=initial_state, xs=None, length=number_observations)
    x_values, y_values = populations
    
    return x_values, y_values