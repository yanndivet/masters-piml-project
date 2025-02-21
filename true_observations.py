import os
import polars as pl
from jax import random
import numpy as np

import constants as cs
import data_generation as data_gen
import sampled_distributions as sd

def generate_true_observations(number_systems=cs.N_SYSTEMS):
    key = random.key(number_systems)
    key_sample_params, key_obs_gen = random.split(key, 2)
    sampled_params = sd.sample_lognormal(key_sample_params, mu=cs.MU_TARGET, tau=cs.TAU_TARGET, m=number_systems)
    target_observations_from_sampled_params = data_gen.generate_observations(sampled_params, key_obs_gen, cs.OBSERVATION_NOISE)
    return target_observations_from_sampled_params

def save_true_observations(number_systems=cs.N_SYSTEMS, experiment=1):
    true_observations_np = np.array(generate_true_observations(number_systems))
    df_true_observations = pl.DataFrame(true_observations_np, schema=[f"obs_{i}" for i in range(cs.OBSERVATION_LENGTH)])
    df_true_observations.write_parquet(f"true_observations/experiment_{experiment}/N={number_systems}.parquet")
    return df_true_observations

def read_true_observations(number_systems=cs.N_SYSTEMS, experiment=1):
    true_observations_file_path = f"true_observations/experiment_{experiment}/N={number_systems}.parquet"

    # Check if directory exists, if not create it
    os.makedirs(os.path.dirname(true_observations_file_path), exist_ok=True)

    # Check if file exists
    if os.path.exists(true_observations_file_path):
        df_true_observations = pl.read_parquet(true_observations_file_path)
    else:
        df_true_observations = save_true_observations(number_systems, experiment)
    
    return df_true_observations.to_jax()