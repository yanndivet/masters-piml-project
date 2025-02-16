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

LEARNING_RATE = 5e-3
EPSILON = 1e-6

CURRENT_ITERATION = 0

def calculate_map(a, b):
    '''
    Calculate Maxmimum A Posteriori values
    '''
    return b / (a + 1)

# Define hyperprior constants
MU_PHI = jnp.array([2., 3.])
SIGMA_PHI = jnp.array([1.2, 1.8])
A_PHI = jnp.array([4.0, 4.5])
B_PHI = jnp.array([4.8, 4.4])
TAU_PHI_MAP = calculate_map(A_PHI, B_PHI)

# Define target constants
MU_TARGET = jnp.array([2.5, 3.5])
TAU_TARGET = jnp.array([1.4, 1.2])

# Define starting value constants
MU_INITIAL = jnp.array([2.3, 3.2])
# SIGMA_INITIAL = jnp.array([.65, 1.])
# A_INITIAL = jnp.array([4.2, 4.8])
# B_INITIAL = jnp.array([4.5, 4.6])
# TAU_INITIAL_MAP = calculate_map(A_INITIAL, B_INITIAL)
TAU_INITIAL_MAP = jnp.array([1.05, .7])

TARGET_HYPERPARAMETERS = jnp.concatenate([MU_TARGET, TAU_TARGET])
INITIAL_HYPERPARAMETERS = jnp.concatenate([MU_INITIAL, TAU_INITIAL_MAP])
# INITIAL_HYPERPARAMETERS = TARGET_HYPERPARAMETERS

HYPERPARAMETER_NAMES = ['mu_omega', 'mu_gamma', 'tau_omega', 'tau_gamma']

# --------------------------------------------------------------------------
# Define all priors
# MU_M_PRIOR_MEAN = 12.0
# MU_M_PRIOR_STD = 2.0

# MU_C_PRIOR_MEAN = 9.0
# MU_C_PRIOR_STD = 1.5

# MU_K_PRIOR_MEAN = 8.0
# MU_K_PRIOR_STD = 2.0

# MU_PRIOR_MEAN = jnp.array([MU_M_PRIOR_MEAN, MU_C_PRIOR_MEAN, MU_K_PRIOR_MEAN])
# MU_PRIOR_STD = jnp.array([MU_M_PRIOR_STD, MU_C_PRIOR_STD, MU_K_PRIOR_STD])


# TAU_M_PRIOR_MEAN = 1.0
# TAU_M_PRIOR_STD = 0.5

# TAU_C_PRIOR_MEAN = 1.5
# TAU_C_PRIOR_STD = 0.25

# TAU_K_PRIOR_MEAN = 2.0
# TAU_K_PRIOR_STD = 0.75

# PRIOR_MU_PARAMS = jnp.array([
#     [MU_M_PRIOR_MEAN, MU_M_PRIOR_STD], 
#     [MU_C_PRIOR_MEAN, MU_C_PRIOR_STD], 
#     [MU_K_PRIOR_MEAN, MU_K_PRIOR_STD]
# ])

# PRIOR_TAU_PARAMS = jnp.array([
#     [TAU_M_PRIOR_MEAN, TAU_M_PRIOR_STD], 
#     [TAU_C_PRIOR_MEAN, TAU_C_PRIOR_STD], 
#     [TAU_K_PRIOR_MEAN, TAU_K_PRIOR_STD]
# ])

# PRIOR_ALPHA_PARAMS = jnp.array([MU_M_PRIOR_MEAN, MU_C_PRIOR_MEAN, MU_K_PRIOR_MEAN, TAU_M_PRIOR_MEAN, TAU_C_PRIOR_MEAN, TAU_K_PRIOR_MEAN])

# --------------------------------------------------------------------------
# Define all true variables
# MU_M_TRUE_MEAN = 5.0
# MU_M_TRUE_STD = 1.0

# MU_C_TRUE_MEAN = 10.0
# MU_C_TRUE_STD = 0.8

# MU_K_TRUE_MEAN = 7.0
# MU_K_TRUE_STD = 1.5

# MU_TRUE_MEAN = jnp.array([MU_M_TRUE_MEAN, MU_C_TRUE_MEAN, MU_K_TRUE_MEAN])

# TAU_M_TRUE_MEAN = 2.0
# TAU_M_TRUE_STD = 0.5

# TAU_C_TRUE_MEAN = 1.0
# TAU_C_TRUE_STD = 0.25

# TAU_K_TRUE_MEAN = 1.5
# TAU_K_TRUE_STD = 0.75

# TRUE_MU_PARAMS = jnp.array([
#     [MU_M_TRUE_MEAN, MU_M_TRUE_STD], 
#     [MU_C_TRUE_MEAN, MU_C_TRUE_STD], 
#     [MU_K_TRUE_MEAN, MU_K_TRUE_STD]
# ])

# TRUE_TAU_PARAMS = jnp.array([
#     [TAU_M_TRUE_MEAN, TAU_M_TRUE_STD], 
#     [TAU_C_TRUE_MEAN, TAU_C_TRUE_STD], 
#     [TAU_K_TRUE_MEAN, TAU_K_TRUE_STD]
# ])

# TRUE_ALPHA_PARAMS = jnp.array([MU_M_TRUE_MEAN, MU_C_TRUE_MEAN, MU_K_TRUE_MEAN, TAU_M_TRUE_MEAN, TAU_C_TRUE_MEAN, TAU_K_TRUE_MEAN])

# --------------------------------------------------------------------------
# def inverse_diagonal_matrix(diagonal):
#     return jnp.diag(jnp.array(1/diagonal))

# INVERSE_PRIOR_COVARIANCE = inverse_diagonal_matrix(MU_PRIOR_STD)