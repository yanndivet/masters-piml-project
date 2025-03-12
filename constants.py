import jax.numpy as jnp

N_SYSTEMS = 4 # Number of systems. If N = 1, Classical Bayesian inference. 
N_VALUES = [1, 5, 10, 50, 100, 500, 1000]

OBSERVATION_NOISE = 1e-2
# OBSERVATION_LENGTH 

N_CHAINS = 1

SIGMA = 0.02
OBSERVATION_LENGTH = 200
T = 5
U_0 = 0.0
V_0 = 1.0
X_0 = 1.0
Y_0 = 1.0

LEARNING_RATE = 5e-3
EPSILON = 1e-6

CURRENT_ITERATION = 0

def calculate_map(a, b):
    '''
    Calculate Maxmimum A Posteriori values
    '''
    return b / (a + 1)

# Define hyperprior constants
MU_PHI = jnp.array([1.5, 3.])
SIGMA_PHI = jnp.array([3., 3.2])
# A_PHI = jnp.array([9.0, 12.5])
# B_PHI = jnp.array([8.8, 11.4])
A_PHI = jnp.array([3.0, 3.0])
B_PHI = jnp.array([4.0, 4.0])
TAU_PHI_MAP = calculate_map(A_PHI, B_PHI)

# Define target constants
MU_TARGET = jnp.array([2.3, 2.])
TAU_TARGET = jnp.array([.7, .9])

# Define starting value constants
MU_INITIAL = jnp.array([1.9, 2.8])
TAU_INITIAL = jnp.array([.9, 1.])

TARGET_HYPERPARAMETERS = jnp.concatenate([MU_TARGET, TAU_TARGET])
INITIAL_HYPERPARAMETERS = jnp.concatenate([MU_INITIAL, TAU_INITIAL]) 

HYPERPARAMETER_NAMES = ['mu_omega', 'mu_gamma', 'tau_omega', 'tau_gamma']