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

# @partial(jit, static_argnums=(1,2,3))
# def lotka_volterra(params, T=cs.T, number_observations=cs.OBSERVATION_LENGTH, initial_state=(cs.X_0, cs.Y_0)):
#     """
#     Leap-frog integration for the Lotka-Volterra predator-prey model using JAX.

#     Args:
#         params: Tuple of (alpha, beta, gamma, delta) where:
#             alpha: Prey growth rate
#             beta: Prey death rate due to predation
#             gamma: Predator death rate
#             delta: Predator growth rate due to predation
#         T: Total simulation time
#         number_observations: Number of time steps to record
#         initial_state: Tuple of (x_0, y_0) with initial populations
    
#     Returns:
#         JAX array of populations over time, containing both x and y values.
#     """
    
#     # Unpack parameters
#     alpha, beta, gamma, delta = params
#     dt = T / number_observations

#     def next_state(state, _):
#         x_prev, y_prev = state
        
#         # Simple Euler method, matching your original implementation
#         x_next = x_prev + dt * (alpha * x_prev - beta * x_prev * y_prev)
#         y_next = y_prev + dt * (gamma * x_prev * y_prev - delta * y_prev)
        
#         return (x_next, y_next), (x_next, y_next)

#     _, populations = lax.scan(f=next_state, init=initial_state, xs=None, length=number_observations)
#     x_values, y_values = populations
    
#     return x_values, y_values

@partial(jit, static_argnums=(1,2,3))
def lotka_volterra(params, T=cs.T, number_observations=cs.OBSERVATION_LENGTH, initial_state=(cs.X_0, cs.Y_0)):
    """
    4th order Runge-Kutta integration for the Lotka-Volterra predator-prey model using JAX.

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
    
    # Define the Lotka-Volterra derivatives
    def derivatives(x, y):
        dx_dt = alpha * x - beta * x * y
        dy_dt = gamma * x * y - delta * y
        return dx_dt, dy_dt
    
    def next_state(state, _):
        x, y = state
        
        # RK4 step 1: Initial derivatives
        k1_x, k1_y = derivatives(x, y)
        k1_x, k1_y = dt * k1_x, dt * k1_y
        
        # RK4 step 2: Mid-point derivatives using k1
        k2_x, k2_y = derivatives(x + k1_x/2, y + k1_y/2)
        k2_x, k2_y = dt * k2_x, dt * k2_y
        
        # RK4 step 3: Mid-point derivatives using k2
        k3_x, k3_y = derivatives(x + k2_x/2, y + k2_y/2)
        k3_x, k3_y = dt * k3_x, dt * k3_y
        
        # RK4 step 4: End-point derivatives using k3
        k4_x, k4_y = derivatives(x + k3_x, y + k3_y)
        k4_x, k4_y = dt * k4_x, dt * k4_y
        
        # RK4 weighted sum to get next state
        x_next = x + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        y_next = y + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        
        return (x_next, y_next), (x_next, y_next)
    
    # Run the simulation using lax.scan
    _, (x_values, y_values) = lax.scan(f=next_state, init=initial_state, xs=None, length=number_observations)
    return x_values, y_values