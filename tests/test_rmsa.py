import logging
import os
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from optical_rl_gym.envs.phy_rmsa_env import (
    shortest_available_path_first_fit,
    phy_aware_sapff_rmsa,
    phy_aware_bmff_rmsa,
    phy_aware_bmfa_rmsa,
    phy_aware_bmfa_rss_rmsa

)
from optical_rl_gym.utils import evaluate_heuristic, random_policy

load = 1300
# logging.getLogger("rmsaenv").setLevel(logging.INFO)

seed = 20
episodes = 1
episode_length = 100000

monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
topology_name='jpn12'
# topology_name='nsfnet_chen'
with open(
    os.path.join("..", "examples", "topologies", f"{topology_name}_5-paths_6-modulations.h5"), "rb"
) as f:
    topology = pickle.load(f)

mat_file = loadmat('../examples/inputs/Modulation_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')
mat_file1 = loadmat('../examples/inputs/GSNR_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')
mat_file2 = loadmat('../examples/inputs/All_connections_Profile_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')

modulation_jpn12 = mat_file['Modulation_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform']
gsnr_jpn12 = mat_file1['GSNR_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform']
all_connections_jpn12 = mat_file2['All_connections_Profile_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform']


env_args = dict(
    topology=topology,
    seed=10,
    allow_rejection=True,
    load=load,
    mean_service_holding_time=25,
    episode_length=episode_length,
    num_spectrum_resources=64,
    bit_rate_selection="discrete",
    modulation_level=modulation_jpn12,
    connections_detail=all_connections_jpn12,
    gsnr=gsnr_jpn12,
    number_spectrum_channels=80,
    number_spectrum_channels_s_band=108,
)

print("STR".ljust(5), "REW".rjust(7), "STD".rjust(7))


env_phy_df = gym.make("PhyRMSA-v0", **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(
    env_phy_df, phy_aware_sapff_rmsa, n_eval_episodes=episodes
)
print("SAP-FF:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
print(
    "\tBit rate blocking:",
    (env_phy_df.episode_bit_rate_requested - env_phy_df.episode_bit_rate_provisioned)
    / env_phy_df.episode_bit_rate_requested,
)
print(
    "\tRequest blocking:",
    (env_phy_df.episode_services_processed - env_phy_df.episode_services_accepted)
    / env_phy_df.episode_services_processed,
)

env_phy_bmff_df = gym.make("PhyRMSA-v0", **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(
    env_phy_bmff_df, phy_aware_bmff_rmsa, n_eval_episodes=episodes
)
print("BM-FF:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
print(
    "\tBit rate blocking:",
    (env_phy_bmff_df.episode_bit_rate_requested - env_phy_bmff_df.episode_bit_rate_provisioned)
    / env_phy_bmff_df.episode_bit_rate_requested,
)
print(
    "\tRequest blocking:",
    (env_phy_bmff_df.episode_services_processed - env_phy_bmff_df.episode_services_accepted)
    / env_phy_bmff_df.episode_services_processed,
)


env_phy_bmfa_df = gym.make("PhyRMSA-v0", **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(
    env_phy_bmfa_df, phy_aware_bmfa_rmsa, n_eval_episodes=episodes
)
print("BM-FA:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
print(
    "\tBit rate blocking:",
    (env_phy_bmfa_df.episode_bit_rate_requested - env_phy_bmfa_df.episode_bit_rate_provisioned)
    / env_phy_bmfa_df.episode_bit_rate_requested,
)
print(
    "\tRequest blocking:",
    (env_phy_bmfa_df.episode_services_processed - env_phy_bmfa_df.episode_services_accepted)
    / env_phy_bmfa_df.episode_services_processed,
)


env_phy_bmfa_rss_df = gym.make("PhyRMSA-v0", **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(
    env_phy_bmfa_rss_df, phy_aware_bmfa_rss_rmsa, n_eval_episodes=episodes
)
print("BM-FA-Rss:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
print(
    "\tBit rate blocking:",
    (env_phy_bmfa_rss_df.episode_bit_rate_requested - env_phy_bmfa_rss_df.episode_bit_rate_provisioned)
    / env_phy_bmfa_rss_df.episode_bit_rate_requested,
)
print(
    "\tRequest blocking:",
    (env_phy_bmfa_rss_df.episode_services_processed - env_phy_bmfa_rss_df.episode_services_accepted)
    / env_phy_bmfa_rss_df.episode_services_processed,
)