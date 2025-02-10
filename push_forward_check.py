import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from jax import random
import distributions as dist
import matplotlib.pyplot as plt
import numpy as np

key = random.key(42)
key_mu, key_tau, key_pop = random.split(key, 3)
M = 10000

sampled_mu_thetas = dist.sample_normal_distribution(key_mu, m=M)
sampled_tau_thetas = dist.sample_inverse_gamma_distribution(key_tau, m=M)
samples = dist.sample_lognormal(key_pop, sampled_mu_thetas, sampled_tau_thetas, M)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f'Push Foward Check for M = {M}')

for i in range(2):
    sample_data = np.array(samples[i])
    axes[i].hist(sample_data, bins=50, density=True, alpha=0.6, label='Samples')
    axes[i].legend()
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.savefig(f'simulation_results/push_forward_check_M={M}.png')
plt.close()