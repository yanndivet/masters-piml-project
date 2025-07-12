import jax.numpy as jnp

N_VALUES = [1, 10, 50]
# N_VALUES = [1, 5, 10, 50, 100, 500, 1000] # Number of systems. If N = 1, Classical Bayesian inference. 

OBSERVATION_NOISE = 1e-2
OBSERVATION_LENGTH = 50

NUMBER_EXPERIMENTS = 2

N_CHAINS = 1

SIGMA = 0.02
T = 1
U_0 = 0.0
V_0 = 1.0
X_0 = 1.0
Y_0 = 1.0

LEARNING_RATE = 5e-3
EPSILON = 1e-6

NUMBER_ITERATIONS_SW = 500

INITIAL_HYPERPARAMETERS_NOISE = 1e-2

PROBLEM = "LV"

def calculate_map(a, b):
    return b / (a + 1)

if PROBLEM == "LV": # Lotka-Volterra equations
    # Define hyperprior constants
    # MU_PHI = jnp.array([1.5, 3., 5., 3.])
    # SIGMA_PHI = jnp.array([3., 3.2, 2.8, 1.7])
    # A_PHI = jnp.array([3.0, 3.0, 3.0, 3.0])
    # B_PHI = jnp.array([4.0, 4.0, 4.0, 4.0])

    MU_PHI = jnp.array([2.3, 2., 1.7, 2.1])
    SIGMA_PHI = jnp.array([.5, .7, .4, .3])
    A_PHI = jnp.array([9., 9., 9., 9.])
    B_PHI = jnp.array([7.0, 9.0, 6., 8.])

    # Define target constants
    MU_TARGET = jnp.array([2.3, 2., 1.7, 2.1])
    TAU_TARGET = jnp.array([.7, .9, .6, .8])

    # Define starting value constants
    # MU_INITIAL = jnp.array([1.9, 2.8, 4.7, 2.8])
    # TAU_INITIAL = jnp.array([.9, 1., .5, .9])
    MU_INITIAL = jnp.array([2.3, 2., 1.7, 2.1])
    TAU_INITIAL = jnp.array([.7, .9, .6, .8])

    HYPERPARAMETERS = [r"\alpha", r"\beta", r"\gamma", r"\delta"]

else: # Damped Harmonic Oscillator
    # Define hyperprior constants
    MU_PHI = jnp.array([1.5, 3.])
    SIGMA_PHI = jnp.array([3., 3.2])
    A_PHI = jnp.array([3.0, 3.0])
    B_PHI = jnp.array([4.0, 4.0])

    # Define target constants
    MU_TARGET = jnp.array([2.3, 2.])
    TAU_TARGET = jnp.array([.7, .9])

    # Define starting value constants
    MU_INITIAL = jnp.array([1.9, 2.8])
    TAU_INITIAL = jnp.array([.9, 1.])

    HYPERPARAMETERS = [r"\omega", r"\gamma"]

TAU_PHI_MAP = calculate_map(A_PHI, B_PHI)

TARGET_HYPERPARAMETERS = jnp.concatenate([MU_TARGET, TAU_TARGET])
INITIAL_HYPERPARAMETERS = jnp.concatenate([MU_INITIAL, TAU_INITIAL])

HYPERPARAMETER_NAMES = [f"${prefix}_{suffix}$" for prefix in [r"\mu", r"\tau"] for suffix in HYPERPARAMETERS]

